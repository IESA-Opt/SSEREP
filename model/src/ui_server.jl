# =============================================================================
# ui_server.jl -- local browser UI and run API
# =============================================================================

using HTTP
using JSON3
import Distributed

const UI_JOBS = Dict{String,Dict{String,Any}}()
const UI_TASKS = Dict{String,Task}()
const UI_JOBS_LOCK = ReentrantLock()
# Scenario-space campaigns are tracked separately from solve jobs because
# they have a different lifecycle (one campaign supervises many variants
# running on a Distributed worker pool). Each entry is a Dict snapshot
# safe to serialise to JSON; the Task and cancel Ref live alongside it.
const UI_CAMPAIGNS = Dict{String,Dict{String,Any}}()
const UI_CAMPAIGN_TASKS = Dict{String,Task}()
const UI_CAMPAIGN_CANCEL = Dict{String,Ref{Bool}}()
# Per-campaign mutable state needed for pause/resume. Holds the heavy
# objects (loaded ModelData, the pre-sampled per-variant change lists,
# settings) so a resume call can re-launch run_campaign on only the
# variants that have not yet completed without redoing
# read+derive+cluster+sample. Kept out of UI_CAMPAIGNS because the
# values are not JSON-serialisable and would slow down every status
# poll's deepcopy.
const UI_CAMPAIGN_STATE = Dict{String,Dict{Symbol,Any}}()
const UI_CAMPAIGNS_LOCK = ReentrantLock()
const UI_MGA_CAMPAIGNS = Dict{String,Dict{String,Any}}()
const UI_MGA_TASKS = Dict{String,Task}()
const UI_MGA_LOCK = ReentrantLock()
const COMMERCIAL_SOLVER_IDS = ("gurobi", "cplex", "xpress")
const UI_MAX_LOG_LINES = 5000
const UI_DATA_CACHE_LOCK = ReentrantLock()
const UI_CACHE_WARM_LOCK = ReentrantLock()
const UI_CACHE_WARM_TASKS = Dict{String,Task}()
# In-memory ModelData cache so repeated runs from the same workbook do not
# re-deserialize the ~7 MB DuckDB payload (which triggers JIT and dominates the
# read phase for the first run after server start). Keyed by canonical XLSX
# path; invalidated when the workbook mtime or size changes.
const UI_MODEL_DATA_CACHE = Dict{String,NamedTuple{(:mtime, :size, :md),Tuple{Float64,Int64,ModelData}}}()
const UI_MODEL_DATA_CACHE_LOCK = ReentrantLock()
# Reports background warm-up progress to the browser so the user can see when
# Julia and DuckDB are fully ready before clicking Run. Possible state values:
# "idle", "warming", "ready", "failed", "missing".
const UI_WARMUP_STATUS = Dict{String,Any}(
    "state" => "idle",
    "message" => "Julia and DuckDB warm-up has not started yet.",
    "workbook" => "",
    "startedAt" => "",
    "readyAt" => "",
    "elapsedSec" => 0.0,
    "error" => "",
)
const UI_WARMUP_STATUS_LOCK = ReentrantLock()
# Wall-clock anchor (epoch seconds) for the warm-up timer. Set once when
# `serve_ui!()` is entered (which is moments before `HTTP.serve` binds the
# port and the browser becomes reachable). The "Julia ready in N s"
# number reported to the browser is `time() - UI_SERVE_START_T0[]`, so it
# matches what the user counts from the moment the page loads — instead of
# only timing the spawned warm-up task, which started a moment earlier and
# does NOT include the HTTP / JSON / first-request JIT that the user is
# also waiting on. 0.0 means `serve_ui!()` has not been entered yet.
const UI_SERVE_START_T0 = Ref{Float64}(0.0)

"""
    _ui_elapsed_since_serve_start() -> Float64

Seconds elapsed since `serve_ui!()` was entered, or `0.0` if the server is
not running yet (e.g. during tests). Always non-negative.
"""
function _ui_elapsed_since_serve_start()
    t0 = UI_SERVE_START_T0[]
    t0 == 0.0 && return 0.0
    return max(0.0, time() - t0)
end

"""
    serve_ui!(; host="127.0.0.1", port=8123, open_browser=true)

Serve the local IESA-Opt web UI. The server exposes a small JSON API for
listing scenarios, detecting solvers, launching runs, polling progress, and
reading DuckDB result summaries.
"""
function serve_ui!(; host::AbstractString = "127.0.0.1",
                   port::Integer = 8123,
                   open_browser::Bool = true)
    url = "http://$(host):$(port)"
    # Anchor the warm-up timer BEFORE anything else: this is the moment the
    # HTTP server is about to bind the port and the browser becomes
    # reachable, so it is the closest proxy we have on the Julia side to
    # "when the user starts watching". The warm-up task and the cache-hit
    # shortcut both report `time() - UI_SERVE_START_T0[]`, which makes the
    # displayed "Julia ready in N s" match the user's mental stopwatch.
    UI_SERVE_START_T0[] = time()
    if Base.Threads.nthreads() == 1
        @warn "Start Julia with --threads=auto to keep progress polling responsive during long solves"
    end
    # The launcher script (scripts/launcher/start-ui.ps1) sets
    # IESA_OPT_OPEN_BROWSER=0 because it polls the port itself and opens
    # the browser once it is actually listening — that is more reliable
    # than the @async sleep(1.0) below and avoids spawning two browser
    # tabs when both fire at the same time.
    env_open = get(ENV, "IESA_OPT_OPEN_BROWSER", "")
    if env_open == "0" || lowercase(env_open) in ("false", "no", "off")
        open_browser = false
    end
    if open_browser
        @async begin
            sleep(1.0)
            _open_browser(url)
        end
    end
    # Start warming the default workbook in the background so that JIT for the
    # DuckDB-cache deserialize path is paid before the user clicks Run.
    _warm_default_workbook_cache!("Input/1108 SSP.xlsx")
    @info "IESA-Opt UI serving" url julia_threads = Base.Threads.nthreads()
    HTTP.serve(_ui_handler, host, port; verbose = false)
end

function _ui_handler(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    path = isempty(uri.path) ? "/" : String(uri.path)
    query = uri.query === nothing ? nothing : String(uri.query)
    method = String(req.method)
    try
        if startswith(path, "/api/")
            return _api_response(method, path, query, req)
        end
        return _static_response(path)
    catch err
        message = sprint(showerror, err)
        @error "UI request failed" method path error = message exception=(err, catch_backtrace())
        return _json_response(Dict("error" => message); status = 500)
    end
end

function _api_response(method::String, path::String, query::Union{Nothing,String}, req::HTTP.Request)
    if length(path) > 1
        path = rstrip(path, '/')
    end
    if method == "GET" && path == "/api/options"
        return _json_response(_ui_options())
    elseif method == "GET" && path == "/api/status"
        return _json_response(_ui_warmup_status_snapshot())
    elseif method == "GET" && path == "/api/solvers"
        return _json_response(Dict("solvers" => _detect_solvers()))
    elseif method == "GET" && path == "/api/outputs"
        return _json_response(Dict("outputs" => _list_output_runs()))
    elseif method == "POST" && path == "/api/explorer/options"
        return _json_response(_explorer_options(_json_body(req)))
    elseif method == "POST" && path == "/api/explorer/techGraph"
        return _json_response(_explorer_tech_graph(_json_body(req)))
    elseif method == "POST" && path == "/api/explorer/modelBrowser"
        return _json_response(_explorer_model_browser(_json_body(req)))
    elseif method == "POST" && path == "/api/explorer/inputAtlas"
        return _json_response(_explorer_input_atlas(_json_body(req)))
    elseif method == "POST" && path == "/api/run"
        config = _json_body(req)
        job_id = _start_ui_job!(config)
        return _json_response(Dict("jobId" => job_id, "job" => _job_snapshot(job_id)); status = 202)
    elseif method == "POST" && path == "/api/outputs/results"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        return _json_response(_read_ui_results(out_dir))
    elseif method == "POST" && path == "/api/outputs/supplyDemand"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        activity = String(_config_get(body, "activity", ""))
        period = _config_get(body, "period", nothing)
        return _json_response(_supply_demand_payload(out_dir, activity, period))
    elseif method == "POST" && path == "/api/outputs/emissions"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        group_by = String(_config_get(body, "groupBy", "activity"))
        return _json_response(_emissions_payload(out_dir, group_by))
    elseif method == "POST" && path == "/api/outputs/hourlyDispatch"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        node = String(_config_get(body, "node", ""))
        period_raw = _config_get(body, "period", nothing)
        period = period_raw === nothing ? nothing : (period_raw isa Integer ? Int(period_raw) : (period_raw isa AbstractString && !isempty(period_raw) ? parse(Int, period_raw) : nothing))
        return _json_response(_hourly_dispatch_payload(out_dir; node = node, period = period))
    elseif method == "POST" && path == "/api/outputs/flexibility"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        tech = String(_config_get(body, "tech", ""))
        period_raw = _config_get(body, "period", nothing)
        period = period_raw === nothing ? nothing : (period_raw isa Integer ? Int(period_raw) : (period_raw isa AbstractString && !isempty(period_raw) ? parse(Int, period_raw) : nothing))
        return _json_response(_flexibility_payload(out_dir; tech = tech, period = period))
    elseif method == "POST" && path == "/api/outputs/regionalMap"
        body = _json_body(req)
        out_dir = _resolve_output_dir(String(_config_get(body, "outputDir", "")))
        metric = String(_config_get(body, "metric", "stock"))
        commodity = String(_config_get(body, "commodity", "all"))
        period_raw = _config_get(body, "period", nothing)
        period = period_raw === nothing ? nothing : (period_raw isa Integer ? Int(period_raw) : (period_raw isa AbstractString && !isempty(period_raw) ? parse(Int, period_raw) : nothing))
        return _json_response(_regional_map_payload(out_dir; metric = metric, commodity = commodity, period = period))
    elseif method == "POST" && path == "/api/outputs/compare"
        body = _json_body(req)
        return _json_response(_compare_output_runs(_as_string_vector(_config_get(body, "outputDirs", String[]))))
    elseif method == "POST" && path == "/api/outputs/delete"
        body = _json_body(req)
        return _json_response(_delete_output_runs!(body))
    elseif method == "POST" && path == "/api/browseInputFile"
        return _json_response(Dict("path" => _browse_for_input_file()))
    elseif method == "POST" && path == "/api/scenario/validate"
        return _json_response(_scenario_validate(_json_body(req)))
    elseif method == "POST" && path == "/api/scenario/preview"
        return _json_response(_scenario_preview(_json_body(req)))
    elseif method == "POST" && path == "/api/scenario/run"
        return _json_response(_scenario_run(_json_body(req)); status = 202)
    elseif method == "GET" && startswith(path, "/api/scenario/")
        parts = _url_parts(path)
        if length(parts) == 3 && parts[3] == "campaigns"
            return _json_response(_scenario_campaigns())
        elseif length(parts) == 4 && parts[3] == "status"
            return _json_response(_scenario_status(parts[4]))
        elseif length(parts) == 4 && parts[3] == "result"
            return _json_response(_scenario_result(parts[4]))
        end
    elseif method == "POST" && startswith(path, "/api/scenario/")
        parts = _url_parts(path)
        if length(parts) == 4 && parts[3] == "stop"
            return _json_response(_scenario_stop!(parts[4]))
        elseif length(parts) == 4 && parts[3] == "pause"
            return _json_response(_scenario_pause!(parts[4]))
        elseif length(parts) == 4 && parts[3] == "resume"
            return _json_response(_scenario_resume!(parts[4]))
        end
    elseif method == "POST" && path == "/api/mga/preview"
        return _json_response(_mga_preview(_json_body(req)))
    elseif method == "POST" && path == "/api/mga/run"
        return _json_response(_mga_run(_json_body(req)); status = 202)
    elseif method == "GET" && startswith(path, "/api/mga/")
        parts = _url_parts(path)
        if length(parts) == 3 && parts[3] == "campaigns"
            return _json_response(_mga_campaigns())
        elseif length(parts) == 4 && parts[3] == "status"
            return _json_response(_mga_status(parts[4]))
        elseif length(parts) == 4 && parts[3] == "result"
            return _json_response(_mga_result(parts[4]))
        end
    elseif method == "GET" && startswith(path, "/api/jobs/")
        parts = _url_parts(path)
        if length(parts) == 3
            return _json_response(_job_snapshot(parts[3]))
        elseif length(parts) == 4 && parts[4] == "results"
            return _json_response(_job_results(parts[3]))
        end
    elseif method == "POST" && startswith(path, "/api/jobs/")
        parts = _url_parts(path)
        if length(parts) == 4 && parts[4] == "cancel"
            return _json_response(_cancel_ui_job!(parts[3]))
        end
    end
    return _json_response(Dict("error" => "Not found"); status = 404)
end

function _json_body(req::HTTP.Request)
    return isempty(req.body) ? Dict{String,Any}() : JSON3.read(String(req.body))
end

_json_sanitize(value) = value
_json_sanitize(::Nothing) = nothing
_json_sanitize(::Missing) = nothing
_json_sanitize(value::AbstractString) = value
_json_sanitize(value::Symbol) = String(value)
_json_sanitize(value::AbstractFloat) = isfinite(value) ? value : nothing
_json_sanitize(value::Real) = value
_json_sanitize(value::AbstractDict) = Dict{String,Any}(String(key) => _json_sanitize(val) for (key, val) in value)
_json_sanitize(value::AbstractVector) = Any[_json_sanitize(item) for item in value]
_json_sanitize(value::Tuple) = Any[_json_sanitize(item) for item in value]

function _json_response(payload; status::Integer = 200)
    HTTP.Response(status,
        ["Content-Type" => "application/json; charset=utf-8",
         "Cache-Control" => "no-store",
         # Allow the bootstrap loading page (loaded via file://) to read the
         # JSON body cross-origin so it can poll /api/status and only redirect
         # once warm-up is actually complete. The UI server only listens on
         # 127.0.0.1, so opening the API to "*" does not expand the attack
         # surface beyond what file:// already grants.
         "Access-Control-Allow-Origin" => "*"],
        JSON3.write(_json_sanitize(payload)))
end

function _static_response(path::String)
    static_path = path == "/" ? "/index.html" : path
    # The brand logos live under docs/src/assets/ so the Documenter site and
    # the UI share a single source of truth. We accept any iesa-opt-logo*.png
    # name (e.g. iesa-opt-logo.png, iesa-opt-logo-rect.png) and resolve it
    # against that folder before falling through to the regular ui/ static
    # tree. Anything else under /assets/ that does not match the allow-list
    # falls through and is served (or 404'd) from ui/ as usual.
    if startswith(static_path, "/assets/")
        asset_name = lowercase(last(_url_parts(static_path)))
        if startswith(asset_name, "iesa-opt-logo") && endswith(asset_name, ".png")
            docs_path = joinpath(_repo_root(), "docs", "src", "assets", asset_name)
            isfile(docs_path) && return _file_response(docs_path, "image/png")
        end
    end

    parts = _url_parts(static_path)
    full_path = normpath(joinpath(_ui_dir(), parts...))
    root = normpath(_ui_dir())
    startswith(lowercase(full_path), lowercase(root)) || return HTTP.Response(403, "Forbidden")
    isfile(full_path) || return HTTP.Response(404, "Not found")
    return _file_response(full_path, _mime_type(full_path))
end

function _url_parts(path::AbstractString)
    return [String(part) for part in split(strip(path, ['/']), '/') if !isempty(part)]
end

function _file_response(path::AbstractString, mime::AbstractString)
    HTTP.Response(200,
        ["Content-Type" => mime,
         "Cache-Control" => "no-store"],
        read(path))
end

function _mime_type(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext == ".html" && return "text/html; charset=utf-8"
    ext == ".css" && return "text/css; charset=utf-8"
    ext == ".js" && return "text/javascript; charset=utf-8"
    ext == ".json" && return "application/json; charset=utf-8"
    ext == ".geojson" && return "application/geo+json; charset=utf-8"
    ext == ".png" && return "image/png"
    ext == ".svg" && return "image/svg+xml"
    return "application/octet-stream"
end

_repo_root() = normpath(joinpath(@__DIR__, ".."))
_ui_dir() = joinpath(_repo_root(), "ui")

function _open_browser(url::AbstractString)
    try
        if Sys.iswindows()
            run(`cmd /c start "" $url`)
        elseif Sys.isapple()
            run(`open $url`)
        else
            run(`xdg-open $url`)
        end
    catch err
        @warn "Could not open browser automatically" url err
    end
end

function _ui_options()
    options = Dict(
        "scenarios" => _list_workbooks(),
        "cpuThreads" => Sys.CPU_THREADS,
        "periods" => [2022, 2025, 2030, 2035, 2040, 2045, 2050],
        "hoursPerDayOptions" => [24, 12, 8, 6, 4, 3, 2, 1],
        "solveMethods" => [
            Dict("id" => "barrier", "label" => "Barrier"),
            Dict("id" => "barrier_crossover", "label" => "Barrier + crossover"),
            Dict("id" => "concurrent", "label" => "Concurrent"),
            Dict("id" => "dual_simplex", "label" => "Dual simplex"),
            Dict("id" => "primal_simplex", "label" => "Primal simplex"),
        ],
        "clusteringApproaches" => ["kmeans_avg", "kmeans_shape", "kmedoids_shape", "maxdiss", "maxdiss_shape", "hull_convex", "hull_conical"],
        "constraintGroups" => ["Base", "Base + Bunkers", "Base + Scope3", "Base + Bunkers + Scope3", "ADAPT", "TRANSFORM", "ADAPT + bunker aviation & navigation policy", "ADAPT with bunkers in single constraint", "Base + RFNBO targets", "Linking scenario", "Linking scenario + Scope3"],
        "defaults" => Dict(
            "inputWorkbook" => "Input/1108 SSP.xlsx",
            "periods" => [2050],
            "mode" => "timeslice",
            "hoursPerDay" => 24,
            "representativeDays" => 15,
            "solver" => _preferred_default_solver_id(),
            "solveMethod" => "barrier_crossover",
            "threads" => 0,
            "clusteringApproach" => "kmeans_avg",
            "extremePeriods" => true,
            "extremeDays" => 5,
            "boundaryRamping" => true,
            "hourlyReports" => true,
            "showViolations" => false,
            "outputMode" => "automatic",
            "constraintGroup" => "Base + Bunkers + Scope3"
        )
    )
    _warm_default_workbook_cache!(String(options["defaults"]["inputWorkbook"]))
    return options
end

function _list_workbooks()
    root = _repo_root()
    dirs = [joinpath(root, "Input")]
    files = String[]
    for dir in dirs
        isdir(dir) || continue
        for name in sort(readdir(dir))
            path = joinpath(dir, name)
            isfile(path) || continue
            lowercase(splitext(name)[2]) in (".xlsx", ".xls") || continue
            push!(files, replace(relpath(path, root), '\\' => '/'))
        end
    end
    return files
end

function _rewrite_legacy_input_workbook(value::AbstractString)
    text = strip(String(value))
    isempty(text) && return "Input/1108 SSP.xlsx"
    root = _repo_root()
    as_posix = replace(text, '\\' => '/')
    if !isabspath(text) && startswith(lowercase(as_posix), "data/")
        migrated = "Input/" * as_posix[6:end]
        old_path = normpath(joinpath(root, text))
        new_path = normpath(joinpath(root, migrated))
        !isfile(old_path) && isfile(new_path) && return migrated
    elseif isabspath(text)
        old_root = replace(normpath(joinpath(root, "data")), '\\' => '/')
        abs_path = replace(normpath(text), '\\' => '/')
        old_root_lc = lowercase(old_root)
        abs_path_lc = lowercase(abs_path)
        if abs_path_lc == old_root_lc || startswith(abs_path_lc, old_root_lc * "/")
            suffix = relpath(normpath(text), normpath(joinpath(root, "data")))
            migrated = suffix == "." ? normpath(joinpath(root, "Input")) : normpath(joinpath(root, "Input", suffix))
            !isfile(normpath(text)) && isfile(migrated) && return migrated
        end
    end
    return text
end

function _resolve_input_workbook(value::AbstractString; require_exists::Bool = true)
    input_value = _rewrite_legacy_input_workbook(value)
    input_path = isabspath(input_value) ? normpath(input_value) : normpath(joinpath(_repo_root(), input_value))
    require_exists && !isfile(input_path) && error("Input workbook not found: $(input_value)")
    rel = try
        replace(relpath(input_path, _repo_root()), '\\' => '/')
    catch
        input_path
    end
    return input_path, rel
end

# Opens a native OS file-open dialog and returns the absolute path that the user
# picked, or an empty string if the dialog was cancelled or the platform is not
# supported. Windows-only for now (PowerShell + System.Windows.Forms). The dialog
# is launched in a single-threaded apartment (-STA) because OpenFileDialog
# requires it.

# Native Win32 file picker via comdlg32.GetOpenFileNameW. ~50 ms instead of
# ~1 s for a cold PowerShell + WinForms subprocess. The dialog adopts the
# foreground window as its owner so it appears in front of the browser.
# Returns a path string, "" for cancel, or `nothing` if the native picker is
# unavailable (caller falls back to PowerShell). Layout below is x64; on
# anything else we bail out.
const _COMDLG_OFN_SIZE  = 152  # x64 OPENFILENAMEW size in bytes
# OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_NOCHANGEDIR
const _COMDLG_OFN_FLAGS = UInt32(0x00001000 | 0x00000800 | 0x00080000 | 0x00000008)
function _browse_for_input_file_native()
    (Sys.iswindows() && Sys.WORD_SIZE == 64) || return nothing
    initial_dir = joinpath(_repo_root(), "Input")
    isdir(initial_dir) || (initial_dir = _repo_root())
    # OPENFILENAMEW filter format: "<label>\0<patterns>\0…\0\0"
    filter_w  = transcode(UInt16,
        "Excel files (*.xlsx;*.xlsm;*.xls)\0*.xlsx;*.xlsm;*.xls\0" *
        "All files (*.*)\0*.*\0\0")
    title_w   = transcode(UInt16, "Select IESA-Opt input workbook\0")
    initdir_w = transcode(UInt16, initial_dir * "\0")
    nMaxFile  = UInt32(2048)
    file_buf  = zeros(UInt16, nMaxFile)
    ofn       = zeros(UInt8, _COMDLG_OFN_SIZE)
    ok = try
        hwnd_fg = ccall((:GetForegroundWindow, "user32"), Ptr{Cvoid}, ())
        GC.@preserve ofn filter_w title_w initdir_w file_buf begin
            p = pointer(ofn)
            unsafe_store!(Ptr{UInt32}(p + 0),       UInt32(_COMDLG_OFN_SIZE))   # lStructSize
            unsafe_store!(Ptr{Ptr{Cvoid}}(p + 8),   hwnd_fg)                    # hwndOwner
            unsafe_store!(Ptr{Ptr{UInt16}}(p + 24), pointer(filter_w))          # lpstrFilter
            unsafe_store!(Ptr{Ptr{UInt16}}(p + 48), pointer(file_buf))          # lpstrFile
            unsafe_store!(Ptr{UInt32}(p + 56),      nMaxFile)                   # nMaxFile
            unsafe_store!(Ptr{Ptr{UInt16}}(p + 80), pointer(initdir_w))         # lpstrInitialDir
            unsafe_store!(Ptr{Ptr{UInt16}}(p + 88), pointer(title_w))           # lpstrTitle
            unsafe_store!(Ptr{UInt32}(p + 96),      _COMDLG_OFN_FLAGS)          # Flags
            ccall((:GetOpenFileNameW, "comdlg32"), Cint, (Ptr{Cvoid},), p)
        end
    catch err
        @warn "Native file picker unavailable; falling back to PowerShell" error = sprint(showerror, err)
        return nothing
    end
    ok == 0 && return ""  # user cancelled
    n = findfirst(==(UInt16(0)), file_buf)
    n = n === nothing ? length(file_buf) : n - 1
    n == 0 && return ""
    return transcode(String, file_buf[1:n])
end

function _browse_for_input_file()
    Sys.iswindows() || return ""
    native = _browse_for_input_file_native()
    native === nothing || return native
    initial_dir = joinpath(_repo_root(), "Input")
    if !isdir(initial_dir)
        initial_dir = _repo_root()
    end
    ps_quote(s) = "'" * replace(String(s), "'" => "''") * "'"
    # The dialog owner is a 1x1 invisible TopMost form so the picker
    # is brought above the browser window. Without an owner the dialog
    # often opens behind VS Code / Chrome and looks like a hang.
    script = """
\$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null
Add-Type -AssemblyName System.Drawing | Out-Null
\$owner = New-Object System.Windows.Forms.Form
\$owner.TopMost = \$true
\$owner.ShowInTaskbar = \$false
\$owner.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedToolWindow
\$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
\$owner.Opacity = 0
\$owner.Size = New-Object System.Drawing.Size(1, 1)
\$owner.Show()
\$owner.Activate()
\$owner.BringToFront()
try {
    \$dlg = New-Object System.Windows.Forms.OpenFileDialog
    \$dlg.Title = 'Select IESA-Opt input workbook'
    \$dlg.Filter = 'Excel files (*.xlsx;*.xlsm;*.xls)|*.xlsx;*.xlsm;*.xls|All files (*.*)|*.*'
    \$dlg.InitialDirectory = $(ps_quote(initial_dir))
    \$dlg.RestoreDirectory = \$true
    if (\$dlg.ShowDialog(\$owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Out.WriteLine(\$dlg.FileName)
    }
} finally {
    \$owner.Close()
    \$owner.Dispose()
}
"""
    tmpfile = tempname() * ".ps1"
    # Resolve a real PowerShell executable. The bare name `powershell` may
    # not be on PATH (e.g. under VS Code Julia child env) and PS 7 (`pwsh`)
    # also works with WinForms once `-STA` is set.
    ps_exe = Sys.which("powershell")
    if ps_exe === nothing
        ps_exe = Sys.which("pwsh")
    end
    if ps_exe === nothing
        ps_exe = joinpath(get(ENV, "SystemRoot", "C:\\Windows"),
                          "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    end
    try
        write(tmpfile, script)
        out = read(`$ps_exe -STA -NoProfile -ExecutionPolicy Bypass -File $tmpfile`, String)
        return String(strip(out))
    catch err
        @warn "Browse dialog failed" error = sprint(showerror, err)
        return ""
    finally
        try
            isfile(tmpfile) && rm(tmpfile; force = true)
        catch
        end
    end
end

function _warm_default_workbook_cache!(input_workbook::AbstractString)
    # Developer fast-iteration mode: set IESA_OPT_SKIP_WARMUP=1 to bypass the
    # in-server warmup entirely (no DuckDB read, no run-path JIT). The cost
    # is paid lazily on the first Run click instead. Pair with
    # IESA_OPT_SKIP_PRECOMPILE=1 to also skip the @compile_workload at
    # package precompile time, which makes `Pkg.precompile` after source
    # edits ~60-90 s faster.
    if get(ENV, "IESA_OPT_SKIP_WARMUP", "0") == "1"
        _ui_warmup_status_update!(state = "ready", workbook = input_workbook,
                                  message = "Warm-up skipped (IESA_OPT_SKIP_WARMUP=1). First run will JIT on demand.",
                                  readyAt = string(Dates.now()), elapsedSec = 0.0, error = "")
        @info "IESA-Opt.jl UI warm-up skipped" reason = "IESA_OPT_SKIP_WARMUP=1"
        return nothing
    end
    input_path = isabspath(input_workbook) ? normpath(input_workbook) : normpath(joinpath(_repo_root(), input_workbook))
    if !isfile(input_path)
        _ui_warmup_status_update!(state = "missing", workbook = input_workbook,
                                  message = "Workbook not found at $(input_workbook). Place the file under Input/ and reload.",
                                  error = "", elapsedSec = 0.0, readyAt = "")
        return nothing
    end
    if _ui_model_data_cache_hit(input_path)
        # Cache hit usually only happens on a second `serve_ui!()` in the same
        # Julia session (rare in production). Report wall-clock since server
        # start so the number still reflects what the user actually waited
        # (HTTP / first-request JIT) instead of misleadingly saying 0.0.
        elapsed = round(_ui_elapsed_since_serve_start(), digits = 2)
        _ui_warmup_status_update!(state = "ready", workbook = input_workbook,
                                  message = "Workbook already loaded in memory. Ready to run.",
                                  readyAt = string(Dates.now()), elapsedSec = elapsed, error = "")
        return nothing
    end
    key = _canonical_path(input_path)
    lock(UI_CACHE_WARM_LOCK)
    try
        task = get(UI_CACHE_WARM_TASKS, key, nothing)
        task !== nothing && !istaskdone(task) && return nothing
        _ui_warmup_status_update!(state = "warming", workbook = input_workbook,
                                  message = "Loading $(input_workbook) into memory for fast first run.",
                                  startedAt = string(Dates.now()), readyAt = "",
                                  elapsedSec = 0.0, error = "")
        UI_CACHE_WARM_TASKS[key] = Base.Threads.@spawn begin
            try
                @info "IESA-Opt UI warming workbook cache" workbook = input_workbook
                md = _read_ui_data_cached(input_path)
                md === nothing && return nothing
                _ui_warmup_status_update!(message = "Workbook loaded. Compiling run paths for fast first run…")
                _warm_compile_run_paths!(md)
                # Measure from `serve_ui!()` entry, not from the moment this
                # spawned task started running — see comment on
                # UI_SERVE_START_T0. That way the displayed "ready in N s"
                # also accounts for the HTTP/JSON/first-request JIT that
                # competes with this task for CPU while the user is waiting.
                elapsed = round(_ui_elapsed_since_serve_start(), digits = 2)
                _ui_warmup_status_update!(state = "ready",
                                          message = "Workbook loaded and run paths compiled in $(elapsed) seconds. Ready to run.",
                                          readyAt = string(Dates.now()), elapsedSec = elapsed,
                                          error = "")
                @info "IESA-Opt UI workbook cache ready" workbook = input_workbook elapsed_s = elapsed
            catch err
                elapsed = round(_ui_elapsed_since_serve_start(), digits = 2)
                _ui_warmup_status_update!(state = "failed",
                                          message = "Warm-up failed after $(elapsed) seconds.",
                                          elapsedSec = elapsed, error = sprint(showerror, err))
                @warn "IESA-Opt UI workbook cache warm-up failed" workbook = input_workbook err
            end
        end
    finally
        unlock(UI_CACHE_WARM_LOCK)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# MGA exploration — hybrid ORACLE extension API
# ---------------------------------------------------------------------------

function _mga_float(value, default::Float64)
    value === nothing && return default
    value isa Real && return Float64(value)
    try
        return parse(Float64, String(value))
    catch
        return default
    end
end

function _mga_config(body)
    n_directions = clamp(_as_int(_config_get(body, "directions", 6), 6), 1, 240)
    cost_slack = clamp(_mga_float(_config_get(body, "costSlack", 5.0), 5.0), 0.1, 100.0)
    workers = clamp(_as_int(_config_get(body, "workers", 1), 1), 1, max(1, Sys.CPU_THREADS))
    threads = max(0, _as_int(_config_get(body, "threads", 0), 0))
    oracle_iterations = clamp(_as_int(_config_get(body, "oracleIterations", 1), 1), 0, 80)
    oracle_batch = clamp(_as_int(_config_get(body, "oracleBatch", 1), 1), 1, 64)
    tolerance = clamp(_mga_float(_config_get(body, "tolerance", 0.1), 0.1), 0.001, 1.0)
    name = String(_config_get(body, "name", "mga_campaign"))
    mode = String(_config_get(body, "mode", "timeslice"))
    rep_days = clamp(_as_int(_config_get(body, "representativeDays", 15), 15), 1, 365)
    hours_per_day = clamp(_as_int(_config_get(body, "hoursPerDay", 24), 24), 1, 24)
    extreme_days = clamp(_as_int(_config_get(body, "extremeDays", 5), 5), 0, 30)
    extreme_periods = _as_bool(_config_get(body, "extremePeriods", true), true)
    boundary_ramping = _as_bool(_config_get(body, "boundaryRamping", true), true)
    solver = lowercase(String(_config_get(body, "solver", _preferred_default_solver_id())))
    solve_method = lowercase(String(_config_get(body, "solveMethod", "barrier_crossover")))
    input_value = String(_config_get(body, "inputWorkbook", "Input/1108 SSP.xlsx"))
    _, input_rel = _resolve_input_workbook(input_value; require_exists = false)
    return Dict{String,Any}(
        "name" => isempty(strip(name)) ? "mga_campaign" : strip(name),
        "directions" => n_directions,
        "costSlack" => cost_slack,
        "workers" => workers,
        "threads" => threads,
        "oracleIterations" => oracle_iterations,
        "oracleBatch" => oracle_batch,
        "tolerance" => tolerance,
        "inputWorkbook" => input_rel,
        "periods" => _as_int_vector(_config_get(body, "periods", [2050])),
        "mode" => mode,
        "representativeDays" => rep_days,
        "hoursPerDay" => hours_per_day,
        "clusteringApproach" => String(_config_get(body, "clusteringApproach", "kmeans_avg")),
        "extremePeriods" => extreme_periods,
        "extremeDays" => extreme_days,
        "boundaryRamping" => boundary_ramping,
        "constraintGroup" => String(_config_get(body, "constraintGroup", "Base + Bunkers + Scope3")),
        "solver" => solver,
        "solveMethod" => solve_method,
        "method" => String(_config_get(body, "method", "hybrid-oracle-mga")),
    )
end

function _mga_exact_config(cfg::Dict{String,Any})
    mode_text = lowercase(String(get(cfg, "mode", "timeslice")))
    mode = mode_text in ("full_hourly", "full-hourly", "fh") ? :fh : :ts
    periods = _as_int_vector(get(cfg, "periods", [2050]))
    period = isempty(periods) ? 2050 : first(periods)
    return MGAExactConfig(
        directions = Int(cfg["directions"]),
        cost_slack = Float64(cfg["costSlack"]),
        oracle_iterations = Int(cfg["oracleIterations"]),
        oracle_batch = Int(cfg["oracleBatch"]),
        tolerance = Float64(cfg["tolerance"]),
        workers = Int(cfg["workers"]),
        threads = Int(cfg["threads"]),
        solver = String(get(cfg, "solver", _preferred_default_solver_id())),
        solve_method = String(get(cfg, "solveMethod", "barrier_crossover")),
        mode = mode,
        period = period,
        representative_days = Int(get(cfg, "representativeDays", 15)),
        hours_per_day = Int(get(cfg, "hoursPerDay", 24)),
        clustering = Symbol(String(get(cfg, "clusteringApproach", "kmeans_avg"))),
        extreme_periods = Bool(get(cfg, "extremePeriods", true)),
        extreme_days = Int(get(cfg, "extremeDays", 5)),
        boundary_ramping = Bool(get(cfg, "boundaryRamping", true)),
    )
end

function _mga_preview(body)
    cfg = _mga_config(body)
    input_path = _scenario_input_path(Dict("inputWorkbook" => cfg["inputWorkbook"]))
    md = _read_ui_data_cached(input_path)
    design = mga_hybrid_oracle_preview(md, _mga_exact_config(cfg))
    threads = Int(cfg["threads"])
    workers = Int(cfg["workers"])
    # MGA alternatives run sequentially today, so each solve gets the full
    # thread budget. Expose this honestly to the UI summary card.
    return Dict{String,Any}(
        "ok" => true,
        "config" => cfg,
        "method" => design["method"],
        "description" => design["description"],
        "groups" => design["groups"],
        "directions" => design["directions"],
        "oracleTrace" => design["oracleTrace"],
        "certificate" => design["certificate"],
        "parallel" => Dict("workers" => workers, "threadsPerSolve" => threads <= 0 ? "auto" : threads, "parallelSolves" => false),
    )
end

function _mga_run(body)
    preview = _mga_preview(body)
    cfg = preview["config"]
    id = "mga_" * Dates.format(now(), "yyyymmdd_HHMMSS") * "_" * randstring(6)
    direction_states = Dict{String,Any}[]
    for direction in preview["directions"]
        push!(direction_states, Dict{String,Any}(
            "id" => Int(get(direction, "id", length(direction_states) + 1)),
            "label" => String(get(direction, "label", "")),
            "phase" => String(get(direction, "phase", "")),
            "dominantGroup" => String(get(direction, "dominantGroup", "")),
            "status" => "queued",
            "worker" => 0,
            "startedAt" => nothing,
            "durationSeconds" => nothing,
            "errorMessage" => "",
        ))
    end
    threads_per_solve = Int(cfg["threads"]) <= 0 ? "auto" : Int(cfg["threads"])
    workers_info = Dict{String,Any}(
        "configured" => Int(cfg["workers"]),
        "threadsPerSolve" => threads_per_solve,
        "parallelSolves" => false,
        "solver" => String(get(cfg, "solver", "auto")),
        "solveMethod" => String(get(cfg, "solveMethod", "")),
        "current" => nothing,
    )
    snap = Dict{String,Any}(
        "ok" => true,
        "id" => id,
        "campaign" => Dict(
            "name" => cfg["name"],
            "state" => "running",
            "stage" => "Exact MGA solve queued",
            "phase" => "prepare",
            "total" => cfg["directions"],
            "completed" => 0,
            "failed" => 0,
            "workers" => cfg["workers"],
            "started_at" => time(),
        ),
        "config" => cfg,
        "groups" => preview["groups"],
        "directions" => preview["directions"],
        "directionStates" => direction_states,
        "workersInfo" => workers_info,
        "oracleTrace" => preview["oracleTrace"],
        "certificate" => preview["certificate"],
        "results" => Vector{Dict{String,Any}}(),
        "done" => false,
    )
    lock(UI_MGA_LOCK)
    try
        UI_MGA_CAMPAIGNS[id] = snap
    finally
        unlock(UI_MGA_LOCK)
    end
    task = Base.Threads.@spawn _mga_task!(id)
    lock(UI_MGA_LOCK)
    try
        UI_MGA_TASKS[id] = task
    finally
        unlock(UI_MGA_LOCK)
    end
    return Dict("ok" => true, "campaign_id" => id, "snapshot" => _mga_status(id))
end

function _mga_task!(id::String)
    snap = _mga_status(id)
    cfg = get(snap, "config", Dict{String,Any}())
    function progress(payload)
        lock(UI_MGA_LOCK)
        try
            haskey(UI_MGA_CAMPAIGNS, id) || return nothing
            current = UI_MGA_CAMPAIGNS[id]
            campaign = current["campaign"]
            campaign["stage"] = String(get(payload, "message", get(payload, "stage", campaign["stage"])))
            if haskey(payload, "phase")
                campaign["phase"] = String(payload["phase"])
            end
            campaign["completed"] = Int(get(payload, "completed", campaign["completed"]))
            campaign["total"] = Int(get(payload, "total", campaign["total"]))
            haskey(payload, "results") && (current["results"] = get(payload, "results", current["results"]))
            haskey(payload, "oracleTrace") && (current["oracleTrace"] = get(payload, "oracleTrace", current["oracleTrace"]))
            if haskey(payload, "baselineCost")
                campaign["baselineCost"] = Float64(payload["baselineCost"])
            end
            if haskey(payload, "baselineSolveSeconds")
                campaign["baselineSolveSeconds"] = Float64(payload["baselineSolveSeconds"])
            end
            if haskey(payload, "costCap")
                campaign["costCap"] = Float64(payload["costCap"])
            end
            if haskey(payload, "oracleIteration")
                campaign["oracleIteration"] = Int(payload["oracleIteration"])
            end
            states = get(current, "directionStates", Dict{String,Any}[])
            workers = get(current, "workersInfo", Dict{String,Any}())
            if haskey(payload, "directionId")
                direction_id = Int(payload["directionId"])
                ds = String(get(payload, "directionStatus", ""))
                idx = findfirst(s -> Int(get(s, "id", -1)) == direction_id, states)
                if idx !== nothing
                    st = states[idx]
                    if !isempty(ds)
                        st["status"] = ds
                    end
                    if ds == "running"
                        st["worker"] = 1
                        st["startedAt"] = Float64(get(payload, "directionStartedAt", time()))
                        st["durationSeconds"] = nothing
                        workers["current"] = Dict{String,Any}(
                            "directionId" => direction_id,
                            "label" => String(get(payload, "directionLabel", get(payload, "message", ""))),
                            "phase" => String(get(payload, "phase", st["phase"])),
                            "startedAt" => st["startedAt"],
                        )
                    elseif ds in ("solved", "failed")
                        st["durationSeconds"] = Float64(get(payload, "directionDurationSeconds", 0.0))
                        err_msg = String(get(payload, "directionErrorMessage", ""))
                        if !isempty(err_msg)
                            st["errorMessage"] = err_msg
                        end
                        cur = get(workers, "current", nothing)
                        if cur isa AbstractDict && Int(get(cur, "directionId", -1)) == direction_id
                            workers["current"] = nothing
                        end
                    end
                end
            end
            if String(get(campaign, "phase", "")) == "finalize"
                workers["current"] = nothing
            end
        finally
            unlock(UI_MGA_LOCK)
        end
        return nothing
    end
    result_payload = try
        input_path = _scenario_input_path(Dict("inputWorkbook" => cfg["inputWorkbook"]))
        md = _read_ui_data_cached(input_path)
        mga_hybrid_oracle_run(md, _mga_exact_config(cfg); progress = progress)
    catch err
        lock(UI_MGA_LOCK)
        try
            if haskey(UI_MGA_CAMPAIGNS, id)
                failed = UI_MGA_CAMPAIGNS[id]
                failed["done"] = true
                failed["campaign"]["state"] = "failed"
                failed["campaign"]["stage"] = sprint(showerror, err)
                failed["campaign"]["failed"] = Int(get(failed["campaign"], "total", 0))
            end
        finally
            unlock(UI_MGA_LOCK)
        end
        return nothing
    end
    lock(UI_MGA_LOCK)
    try
        haskey(UI_MGA_CAMPAIGNS, id) || return nothing
        snap = UI_MGA_CAMPAIGNS[id]
        snap["groups"] = result_payload["groups"]
        snap["directions"] = result_payload["directions"]
        snap["oracleTrace"] = result_payload["oracleTrace"]
        snap["certificate"] = result_payload["certificate"]
        snap["results"] = result_payload["results"]
        snap["baselineInvestments"] = get(result_payload, "baselineInvestments", Dict{String,Any}[])
        snap["investmentSpread"] = get(result_payload, "investmentSpread", Dict{String,Any}[])
        snap["done"] = true
        snap["campaign"]["state"] = "completed"
        snap["campaign"]["phase"] = "complete"
        snap["campaign"]["stage"] = "Exact Hybrid ORACLE MGA complete"
        snap["campaign"]["completed"] = length(result_payload["results"])
        snap["campaign"]["failed"] = Int(result_payload["certificate"]["failedAlternatives"])
        snap["campaign"]["completed_at"] = time()
        # Reconcile any direction state still tagged "queued"/"running" with the
        # final results so the table never shows a stuck "running" direction.
        states = get(snap, "directionStates", Dict{String,Any}[])
        for row in result_payload["results"]
            direction_id = Int(get(row, "direction", 0))
            idx = findfirst(s -> Int(get(s, "id", -1)) == direction_id, states)
            idx === nothing && continue
            st = states[idx]
            st["status"] = String(get(row, "status", st["status"]))
            st["durationSeconds"] = Float64(get(row, "solveSeconds", get(st, "durationSeconds", 0.0)))
            err_msg = String(get(row, "errorMessage", ""))
            if !isempty(err_msg)
                st["errorMessage"] = err_msg
            end
        end
        workers = get(snap, "workersInfo", Dict{String,Any}())
        workers["current"] = nothing
    finally
        unlock(UI_MGA_LOCK)
    end
    return nothing
end

function _mga_status(id::AbstractString)
    lock(UI_MGA_LOCK)
    try
        snap = get(UI_MGA_CAMPAIGNS, String(id), nothing)
        snap === nothing && return Dict("ok" => false, "error" => "MGA campaign not found: $id")
        return deepcopy(snap)
    finally
        unlock(UI_MGA_LOCK)
    end
end

function _mga_result(id::AbstractString)
    snap = _mga_status(id)
    get(snap, "ok", false) == false && return snap
    return Dict(
        "ok" => true,
        "campaign" => snap["campaign"],
        "config" => snap["config"],
        "groups" => snap["groups"],
        "oracleTrace" => snap["oracleTrace"],
        "certificate" => snap["certificate"],
        "results" => snap["results"],
        "directionStates" => get(snap, "directionStates", Dict{String,Any}[]),
        "workersInfo" => get(snap, "workersInfo", Dict{String,Any}()),
        "baselineInvestments" => get(snap, "baselineInvestments", Dict{String,Any}[]),
        "investmentSpread" => get(snap, "investmentSpread", Dict{String,Any}[]),
    )
end

function _mga_campaigns()
    lock(UI_MGA_LOCK)
    try
        rows = Dict{String,Any}[]
        for (id, snap) in UI_MGA_CAMPAIGNS
            c = get(snap, "campaign", Dict{String,Any}())
            cfg = get(snap, "config", Dict{String,Any}())
            results = get(snap, "results", Any[])
            push!(rows, Dict{String,Any}(
                "id" => id,
                "name" => String(get(c, "name", id)),
                "state" => String(get(c, "state", "")),
                "stage" => String(get(c, "stage", "")),
                "phase" => String(get(c, "phase", "")),
                "total" => Int(get(c, "total", 0)),
                "completed" => Int(get(c, "completed", 0)),
                "failed" => Int(get(c, "failed", 0)),
                "started_at" => Float64(get(c, "started_at", 0.0)),
                "completed_at" => Float64(get(c, "completed_at", 0.0)),
                "done" => Bool(get(snap, "done", false)),
                "result_count" => length(results),
                "solver" => String(get(cfg, "solver", "")),
                "solveMethod" => String(get(cfg, "solveMethod", "")),
                "directions" => Int(get(cfg, "directions", 0)),
                "costSlack" => Float64(get(cfg, "costSlack", 0.0)),
            ))
        end
        sort!(rows; by = r -> Float64(get(r, "started_at", 0.0)), rev = true)
        return Dict{String,Any}("ok" => true, "campaigns" => rows)
    finally
        unlock(UI_MGA_LOCK)
    end
end

function _ui_warmup_status_snapshot()
    lock(UI_WARMUP_STATUS_LOCK)
    snapshot = try
        Dict{String,Any}(UI_WARMUP_STATUS)
    finally
        unlock(UI_WARMUP_STATUS_LOCK)
    end
    # While the warm-up is still in flight, overlay a live wall-clock counter
    # (seconds since `serve_ui!()` entry) so the browser status pill visibly
    # ticks up rather than sitting at 0 for the whole wait. The terminal
    # state (`ready` / `failed`) already has a frozen elapsedSec from the
    # worker task; do not overwrite it here.
    if get(snapshot, "state", "") in ("warming", "idle")
        elapsed = round(_ui_elapsed_since_serve_start(), digits = 2)
        snapshot["elapsedSec"] = elapsed
    end
    return snapshot
end

function _ui_warmup_status_update!(; state = nothing, message = nothing, workbook = nothing,
                                     startedAt = nothing, readyAt = nothing, elapsedSec = nothing,
                                     error = nothing)
    lock(UI_WARMUP_STATUS_LOCK)
    try
        state !== nothing && (UI_WARMUP_STATUS["state"] = state)
        message !== nothing && (UI_WARMUP_STATUS["message"] = message)
        workbook !== nothing && (UI_WARMUP_STATUS["workbook"] = workbook)
        startedAt !== nothing && (UI_WARMUP_STATUS["startedAt"] = startedAt)
        readyAt !== nothing && (UI_WARMUP_STATUS["readyAt"] = readyAt)
        elapsedSec !== nothing && (UI_WARMUP_STATUS["elapsedSec"] = elapsedSec)
        error !== nothing && (UI_WARMUP_STATUS["error"] = error)
    finally
        unlock(UI_WARMUP_STATUS_LOCK)
    end
    return nothing
end

# Compile the heaviest run-time code paths against a tiny 1-period / 1-rep-day
# copy of `md` so the first real user run does not pay JIT cost for derive /
# compute_derived / clustering / model building. Best-effort: any failure is
# logged and silently ignored.
function _warm_compile_run_paths!(md::ModelData)
    try
        md_copy = deepcopy(md)
        periods_available = collect(md_copy.sets.periods)
        isempty(periods_available) && return nothing
        target = 2050 in periods_available ? 2050 : last(periods_available)
        md_copy.sets.periods_solve = [target]
        md_copy.params.hoursPer_day = 24
        md_copy.params.n_repDays = 1
        md_copy.params.hoursPer_day_cluster = 24
        md_copy.params.clustering_approach = :kmeans_avg
        md_copy.params.ts_extremePeriods = false
        md_copy.params.ts_extremeDays_count = 0
        md_copy.params.ts_boundaryRamping = true
        md_copy.params.ts_capacityProfile_autoMode = true
        md_copy.params.ts_capacityProfile_autoFloor = 0.23
        md_copy.params.ts_capacityProfile_autoCap = 1.00
        md_copy.params.ts_capacityProfile_autoFloor_effective = 0.23
        md_copy.params.ts_capacityProfile_envelopeMode = 0
        md_copy.params.dayMix_softness = 0.0
        md_copy.params.dayMix_weightType = :auto

        derive_sets!(md_copy)
        compute_derived_params!(md_copy)
        build_temporal_clusters!(md_copy)
        model = JuMP.Model()
        apply_lp_generation_speedups!(model)
        build_ts_lp!(model, md_copy)
        model = nothing
        GC.gc()
    catch err
        @warn "IESA-Opt UI run-path compile failed (warm-up will still report ready)" err
    end
    return nothing
end

function _cancel_ui_job!(job_id::String)
    model_ref = nothing
    job_status = "unknown"
    lock(UI_JOBS_LOCK)
    try
        job = get(UI_JOBS, job_id, nothing)
        job === nothing && return Dict("error" => "Unknown job id: $(job_id)")
        job_status = String(get(job, "status", "unknown"))
        if job_status in ("completed", "failed", "cancelled")
            return Dict("jobId" => job_id, "cancelled" => false, "status" => job_status,
                        "message" => "Job is already $(job_status); nothing to stop.")
        end
        job["cancelRequested"] = true
        model_ref = get(job, "model", nothing)
    finally
        unlock(UI_JOBS_LOCK)
    end
    _job_update!(job_id; message = "Stop requested by user. Trying to terminate the running solver…")
    terminated = false
    if model_ref !== nothing
        terminated = _terminate_solver!(model_ref)
    end
    msg = terminated ?
        "Solver termination signal sent. The run will stop at the next safe checkpoint." :
        "Stop requested. The run will stop at the next safe checkpoint (no solver running yet, or solver does not support live cancel)."
    return Dict("jobId" => job_id, "cancelled" => true, "status" => job_status,
                "terminatedSolver" => terminated, "message" => msg)
end

function _terminate_solver!(model)
    model isa JuMP.Model || return false
    inner = nothing
    try
        inner = JuMP.unsafe_backend(model)
    catch
        return false
    end
    inner === nothing && return false
    try
        if isdefined(@__MODULE__, :Gurobi) && inner isa Gurobi.Optimizer
            try
                Gurobi.GRBterminate(inner.inner)
                return true
            catch err
                @warn "Gurobi terminate failed" err
            end
        end
        if isdefined(@__MODULE__, :HiGHS) && inner isa HiGHS.Optimizer
            try
                HiGHS.Highs_resetGlobalScheduler(Int32(0))
            catch
            end
            try
                if isdefined(HiGHS, :Highs_interrupt)
                    HiGHS.Highs_interrupt(inner.inner)
                    return true
                end
            catch err
                @warn "HiGHS interrupt failed" err
            end
        end
    catch err
        @warn "Could not terminate solver" err
    end
    return false
end

function _check_cancel(job_id::String)
    lock(UI_JOBS_LOCK)
    try
        job = get(UI_JOBS, job_id, nothing)
        job === nothing && return false
        return get(job, "cancelRequested", false) === true
    finally
        unlock(UI_JOBS_LOCK)
    end
end

function _set_job_model!(job_id::String, model)
    lock(UI_JOBS_LOCK)
    try
        job = get(UI_JOBS, job_id, nothing)
        job === nothing && return nothing
        job["model"] = model
    finally
        unlock(UI_JOBS_LOCK)
    end
    return nothing
end

struct UICancelled <: Exception
    message::String
end
Base.showerror(io::IO, err::UICancelled) = print(io, err.message)

function _read_ui_data_cached(input_path::AbstractString)
    cached = _ui_model_data_cache_lookup(input_path)
    cached !== nothing && return deepcopy(cached)
    lock(UI_DATA_CACHE_LOCK)
    try
        cached = _ui_model_data_cache_lookup(input_path)
        cached !== nothing && return deepcopy(cached)
        md = read_data_cached(input_path)
        _ui_model_data_cache_store!(input_path, md)
        return deepcopy(md)
    finally
        unlock(UI_DATA_CACHE_LOCK)
    end
end

function _ui_model_data_cache_lookup(input_path::AbstractString)
    isfile(input_path) || return nothing
    xstat = stat(input_path)
    key = _canonical_path(input_path)
    lock(UI_MODEL_DATA_CACHE_LOCK)
    try
        entry = get(UI_MODEL_DATA_CACHE, key, nothing)
        entry === nothing && return nothing
        if entry.mtime == xstat.mtime && entry.size == Int64(xstat.size)
            return entry.md
        end
        delete!(UI_MODEL_DATA_CACHE, key)
        return nothing
    finally
        unlock(UI_MODEL_DATA_CACHE_LOCK)
    end
end

function _ui_model_data_cache_store!(input_path::AbstractString, md::ModelData)
    isfile(input_path) || return nothing
    xstat = stat(input_path)
    key = _canonical_path(input_path)
    lock(UI_MODEL_DATA_CACHE_LOCK)
    try
        UI_MODEL_DATA_CACHE[key] = (mtime = xstat.mtime, size = Int64(xstat.size), md = md)
    finally
        unlock(UI_MODEL_DATA_CACHE_LOCK)
    end
    return nothing
end

function _ui_model_data_cache_hit(input_path::AbstractString)
    return _ui_model_data_cache_lookup(input_path) !== nothing
end

function _ui_cache_warm_running(input_path::AbstractString)
    key = _canonical_path(input_path)
    lock(UI_CACHE_WARM_LOCK)
    try
        task = get(UI_CACHE_WARM_TASKS, key, nothing)
        return task !== nothing && !istaskdone(task)
    finally
        unlock(UI_CACHE_WARM_LOCK)
    end
end

function _ui_data_cache_valid(input_path::AbstractString)
    cache_path = _ui_data_cache_path(input_path)
    isfile(input_path) && isfile(cache_path) || return false
    xstat = stat(input_path)
    con = nothing
    try
        con = _duckdb_connect(cache_path; readonly = true)
        metadata = _duckdb_metadata(con)
        return get(metadata, "cache_format", "") == string(_IESA_CACHE_FORMAT_VERSION) &&
               get(metadata, "schema_version", "") == string(_IESA_INPUT_DUCKDB_SCHEMA_VERSION) &&
               get(metadata, "xlsx_mtime", "") == string(xstat.mtime) &&
               get(metadata, "xlsx_size", "") == string(xstat.size)
    catch
        return false
    finally
        con !== nothing && DBInterface.close!(con)
        GC.gc()
    end
end

function _ui_data_cache_path(input_path::AbstractString)
    cache_dir = joinpath(dirname(abspath(input_path)), ".iesa_cache")
    return _duckdb_input_cache_path(input_path, cache_dir)
end

function _resolve_explorer_input(body)
    input_value = String(_config_get(body, "inputWorkbook", "Input/1108 SSP.xlsx"))
    input_path, rel = _resolve_input_workbook(input_value)
    return input_path, rel
end

function _explorer_symbol_label(value; fallback::AbstractString = "Unspecified")
    text = strip(String(value))
    isempty(text) && return String(fallback)
    text == "Symbol(\"\")" && return String(fallback)
    return text
end

function _explorer_technology_universe(md::ModelData)
    techs = Set{Symbol}()
    for collection in (md.sets.technologies, md.sets.tech_balancers, md.sets.tech_infra)
        for t in collection
            t != Symbol("") && push!(techs, t)
        end
    end
    for dict in (md.params.tech_sector, md.params.tech_subsector, md.params.tech_category, md.params.tech_name,
                 md.params.activityPer_tech, md.params.activityPer_techOrig, md.params.tech_activity)
        for t in keys(dict)
            t != Symbol("") && push!(techs, t)
        end
    end
    for ((t, _, _), coef) in _explorer_balance_source(md)
        abs(coef) > 1e-12 && t != Symbol("") && push!(techs, t)
    end
    return sort!(collect(techs); by = string)
end

function _explorer_balance_source(md::ModelData)
    if isempty(md.params.activity_balances) && !isempty(md.params.activity_balancesRef)
        try
            compute_activity_balances!(md)
        catch err
            @warn "Scenario explorer could not derive activity_balances; using activity_balancesRef" err
        end
    end
    return isempty(md.params.activity_balances) ? md.params.activity_balancesRef : md.params.activity_balances
end

function _explorer_periods(md::ModelData)
    periods = Set{Int}(md.sets.periods)
    for ((_, _, period), _) in _explorer_balance_source(md)
        push!(periods, Int(period))
    end
    return sort!(collect(periods))
end

function _explorer_default_period(periods::AbstractVector{Int})
    isempty(periods) && return 2050
    2050 in periods && return 2050
    return last(periods)
end

function _explorer_selected_period(body, periods::AbstractVector{Int})
    requested = _as_int(_config_get(body, "period", _explorer_default_period(periods)), _explorer_default_period(periods))
    requested in periods && return requested
    return _explorer_default_period(periods)
end

function _explorer_count_options(values)
    counts = Dict{String,Int}()
    for value in values
        label = _explorer_symbol_label(value)
        counts[label] = get(counts, label, 0) + 1
    end
    rows = [Dict("id" => key, "label" => key, "count" => counts[key]) for key in sort!(collect(keys(counts)))]
    return rows
end

function _explorer_tech_meta(md::ModelData, t::Symbol)
    p = md.params
    sector = _explorer_symbol_label(get(p.tech_sector, t, Symbol("")))
    subsector = _explorer_symbol_label(get(p.tech_subsector, t, Symbol("")))
    category = _explorer_symbol_label(get(p.tech_category, t, Symbol("")))
    primary = _explorer_symbol_label(get(p.tech_activity, t, get(p.activityPer_tech, t, get(p.activityPer_techOrig, t, Symbol("")))); fallback = "")
    name = strip(String(get(p.tech_name, t, "")))
    return Dict(
        "id" => String(t),
        "name" => isempty(name) ? String(t) : name,
        "sector" => sector,
        "subsector" => subsector,
        "category" => category,
        "primaryActivity" => primary,
    )
end

function _explorer_options(body)
    input_path, workbook = _resolve_explorer_input(body)
    md = _read_ui_data_cached(input_path)
    techs = _explorer_technology_universe(md)
    sectors = [_explorer_tech_meta(md, t)["sector"] for t in techs]
    subsectors = [_explorer_tech_meta(md, t)["subsector"] for t in techs]
    categories = [_explorer_tech_meta(md, t)["category"] for t in techs]
    periods = _explorer_periods(md)
    return Dict(
        "inputWorkbook" => workbook,
        "periods" => periods,
        "defaultPeriod" => _explorer_default_period(periods),
        "technologyCount" => length(techs),
        "sectors" => _explorer_count_options(sectors),
        "subsectors" => _explorer_count_options(subsectors),
        "categories" => _explorer_count_options(categories),
    )
end

function _explorer_filter_set(body, key::String)
    present = false
    if body isa AbstractDict
        present = haskey(body, key) || haskey(body, Symbol(key))
    else
        try
            getproperty(body, Symbol(key))
            present = true
        catch
            present = false
        end
    end
    present || return nothing
    values = _as_string_vector(_config_get(body, key, String[]))
    return Set(strip.(values))
end

function _explorer_tech_matches(meta::AbstractDict, sectors, subsectors, categories)
    (sectors !== nothing && !(String(meta["sector"]) in sectors)) && return false
    (subsectors !== nothing && !(String(meta["subsector"]) in subsectors)) && return false
    (categories !== nothing && !(String(meta["category"]) in categories)) && return false
    return true
end

function _explorer_round(value)
    return round(Float64(value); digits = 6)
end

function _explorer_activity_terms(md::ModelData, tech::Symbol, period::Int, positive::Bool)
    rows = Vector{Dict{String,Any}}()
    for ((t, activity, ps), coef) in _explorer_balance_source(md)
        t == tech && ps == period || continue
        positive ? (coef > 1e-12 || continue) : (coef < -1e-12 || continue)
        push!(rows, Dict("activity" => String(activity), "coefficient" => _explorer_round(coef)))
    end
    sort!(rows; by = r -> -abs(Float64(r["coefficient"])))
    return rows[1:min(length(rows), 6)]
end

function _explorer_activity_group(activity::Symbol)
    text = lowercase(String(activity))
    if occursin("emission", text) || occursin("emitted", text) || occursin("co2", text) || occursin("ghg", text)
        return "Emissions"
    elseif occursin("electric", text)
        return "Electricity"
    elseif occursin("heat", text) || occursin("steam", text)
        return "Heat"
    elseif occursin("hydrogen", text) || occursin("ammonia", text) || occursin("methanol", text)
        return "Molecules"
    elseif occursin("gas", text) || occursin("methane", text) || occursin("lng", text)
        return "Gas"
    elseif occursin("diesel", text) || occursin("kerosene", text) || occursin("gasoline", text) || occursin("naphtha", text) || occursin("lpg", text) || occursin("fuel", text)
        return "Liquid fuels"
    elseif occursin("biomass", text) || occursin("waste", text)
        return "Biogenic"
    else
        return "Other carriers"
    end
end

function _explorer_system_flow_payload(by_activity, selected_activities, meta_by_tech)
    node_map = Dict{String,Dict{String,Any}}()
    links = Vector{Dict{String,Any}}()

    function ensure_activity!(activity::Symbol)
        id = "activity:" * String(activity)
        if !haskey(node_map, id)
            node_map[id] = Dict(
                "id" => id,
                "label" => String(activity),
                "kind" => "activity",
                "group" => _explorer_activity_group(activity),
                "rawId" => String(activity),
            )
        end
        return id
    end

    function ensure_tech!(tech::Symbol, role::String)
        id = "tech:" * role * ":" * String(tech)
        if !haskey(node_map, id)
            meta = meta_by_tech[tech]
            node_map[id] = Dict(
                "id" => id,
                "label" => String(meta["name"]),
                "kind" => "technology",
                "role" => role,
                "rawId" => String(tech),
                "sector" => String(meta["sector"]),
                "subsector" => String(meta["subsector"]),
                "category" => String(meta["category"]),
            )
        end
        return id
    end

    link_count = 0
    for activity in selected_activities
        values = get(by_activity, activity, Tuple{Symbol,Float64}[])
        producers = sort!([x for x in values if x[2] > 1e-12]; by = x -> -abs(x[2]))[1:min(count(x -> x[2] > 1e-12, values), 10)]
        consumers = sort!([x for x in values if x[2] < -1e-12]; by = x -> -abs(x[2]))[1:min(count(x -> x[2] < -1e-12, values), 10)]
        isempty(producers) && continue
        isempty(consumers) && continue
        activity_id = ensure_activity!(activity)
        for (producer, coef) in producers
            link_count >= 700 && break
            source = ensure_tech!(producer, "producer")
            push!(links, Dict(
                "source" => source,
                "target" => activity_id,
                "value" => _explorer_round(abs(coef)),
                "activity" => String(activity),
                "kind" => "output",
                "technology" => String(producer),
                "coefficient" => _explorer_round(coef),
            ))
            link_count += 1
        end
        for (consumer, coef) in consumers
            link_count >= 700 && break
            target = ensure_tech!(consumer, "consumer")
            push!(links, Dict(
                "source" => activity_id,
                "target" => target,
                "value" => _explorer_round(abs(coef)),
                "activity" => String(activity),
                "kind" => "input",
                "technology" => String(consumer),
                "coefficient" => _explorer_round(coef),
            ))
            link_count += 1
        end
        link_count >= 700 && break
    end

    nodes = collect(values(node_map))
    sort!(nodes; by = n -> (String(n["kind"]), get(n, "group", get(n, "sector", "")), String(n["label"])))
    return Dict("nodes" => nodes, "links" => links)
end

function _explorer_detail_flow_payload(by_activity, meta_by_tech)
    node_map = Dict{String,Dict{String,Any}}()
    links = Vector{Dict{String,Any}}()

    function ensure_activity!(activity::Symbol)
        id = "activity:" * String(activity)
        if !haskey(node_map, id)
            node_map[id] = Dict(
                "id" => id,
                "kind" => "activity",
                "label" => String(activity),
                "rawId" => String(activity),
                "group" => _explorer_activity_group(activity),
            )
        end
        return id
    end

    function ensure_tech!(tech::Symbol)
        id = "technology:" * String(tech)
        if !haskey(node_map, id)
            meta = meta_by_tech[tech]
            node_map[id] = Dict(
                "id" => id,
                "kind" => "technology",
                "label" => String(meta["name"]),
                "rawId" => String(tech),
                "sector" => String(meta["sector"]),
                "subsector" => String(meta["subsector"]),
                "category" => String(meta["category"]),
            )
        end
        return id
    end

    for (activity, values) in by_activity
        activity_id = ensure_activity!(activity)
        for (tech, coef) in values
            haskey(meta_by_tech, tech) || continue
            tech_id = ensure_tech!(tech)
            if coef > 1e-12
                push!(links, Dict(
                    "source" => tech_id,
                    "target" => activity_id,
                    "technology" => String(tech),
                    "activity" => String(activity),
                    "group" => _explorer_activity_group(activity),
                    "coefficient" => _explorer_round(coef),
                    "value" => _explorer_round(abs(coef)),
                    "sign" => "positive",
                ))
            elseif coef < -1e-12
                push!(links, Dict(
                    "source" => activity_id,
                    "target" => tech_id,
                    "technology" => String(tech),
                    "activity" => String(activity),
                    "group" => _explorer_activity_group(activity),
                    "coefficient" => _explorer_round(coef),
                    "value" => _explorer_round(abs(coef)),
                    "sign" => "negative",
                ))
            end
        end
    end

    nodes = collect(values(node_map))
    sort!(nodes; by = n -> (String(n["kind"]), get(n, "group", get(n, "sector", "")), String(n["label"])))
    sort!(links; by = l -> (-Float64(l["value"]), String(l["technology"]), String(l["activity"])))
    return Dict("nodes" => nodes, "links" => links)
end

function _explorer_tech_graph(body)
    input_path, workbook = _resolve_explorer_input(body)
    md = _read_ui_data_cached(input_path)
    periods = _explorer_periods(md)
    period = _explorer_selected_period(body, periods)
    max_activities = clamp(_as_int(_config_get(body, "maxActivities", 28), 28), 1, 120)
    sectors = _explorer_filter_set(body, "sectors")
    subsectors = _explorer_filter_set(body, "subsectors")
    categories = _explorer_filter_set(body, "categories")

    meta_by_tech = Dict{Symbol,Dict{String,Any}}()
    for t in _explorer_technology_universe(md)
        meta = _explorer_tech_meta(md, t)
        _explorer_tech_matches(meta, sectors, subsectors, categories) || continue
        meta_by_tech[t] = meta
    end

    by_activity = Dict{Symbol,Vector{Tuple{Symbol,Float64}}}()
    for ((tech, activity, ps), coef) in _explorer_balance_source(md)
        ps == period || continue
        abs(coef) > 1e-12 || continue
        haskey(meta_by_tech, tech) || continue
        push!(get!(by_activity, activity, Vector{Tuple{Symbol,Float64}}()), (tech, Float64(coef)))
    end

    scored = Vector{Tuple{Symbol,Int,Int,Float64,Float64}}()
    for (activity, values) in by_activity
        producers = count(x -> x[2] > 1e-12, values)
        consumers = count(x -> x[2] < -1e-12, values)
        producers > 0 && consumers > 0 || continue
        total_abs = sum(abs(x[2]) for x in values)
        score = producers * consumers + log1p(total_abs)
        push!(scored, (activity, producers, consumers, total_abs, score))
    end
    sort!(scored; by = x -> (-x[5], String(x[1])))
    selected = scored[1:min(length(scored), max_activities)]
    selected_activity_list = [x[1] for x in selected]
    selected_activities = Set(selected_activity_list)

    node_ids = Set{Symbol}()
    edges = Vector{Dict{String,Any}}()
    activity_edge_counts = Dict{Symbol,Int}()
    for activity in selected_activity_list
        values = by_activity[activity]
        producers = sort!([x for x in values if x[2] > 1e-12]; by = x -> -abs(x[2]))
        consumers = sort!([x for x in values if x[2] < -1e-12]; by = x -> -abs(x[2]))
        pairs = Vector{Tuple{Symbol,Float64,Symbol,Float64,Float64}}()
        for (producer, producer_coef) in producers, (consumer, consumer_coef) in consumers
            push!(pairs, (producer, producer_coef, consumer, consumer_coef, abs(producer_coef * consumer_coef)))
        end
        sort!(pairs; by = x -> -x[5])
        per_activity_limit = min(length(pairs), 80)
        for pair in pairs[1:per_activity_limit]
            length(edges) >= 900 && break
            producer, producer_coef, consumer, consumer_coef, _ = pair
            push!(node_ids, producer); push!(node_ids, consumer)
            ratio = abs(consumer_coef) / max(abs(producer_coef), 1e-12)
            push!(edges, Dict(
                "from" => String(producer),
                "to" => String(consumer),
                "activity" => String(activity),
                "outputRatio" => _explorer_round(producer_coef),
                "inputRatio" => _explorer_round(abs(consumer_coef)),
                "inputOutputRatio" => _explorer_round(ratio),
            ))
            activity_edge_counts[activity] = get(activity_edge_counts, activity, 0) + 1
        end
        length(edges) >= 900 && break
    end

    nodes = Vector{Dict{String,Any}}()
    for tech in sort!(collect(node_ids); by = string)
        meta = deepcopy(meta_by_tech[tech])
        meta["inputs"] = _explorer_activity_terms(md, tech, period, false)
        meta["outputs"] = _explorer_activity_terms(md, tech, period, true)
        push!(nodes, meta)
    end

    activities = [Dict(
        "activity" => String(activity),
        "producers" => producers,
        "consumers" => consumers,
        "totalCoefficient" => _explorer_round(total_abs),
        "edges" => get(activity_edge_counts, activity, 0),
    ) for (activity, producers, consumers, total_abs, _) in selected]

    return Dict(
        "inputWorkbook" => workbook,
        "periods" => periods,
        "selectedPeriod" => period,
        "maxActivities" => max_activities,
        "availableActivities" => length(scored),
        "nodes" => nodes,
        "edges" => edges,
        "systemFlows" => _explorer_system_flow_payload(by_activity, selected_activity_list, meta_by_tech),
        "detailFlows" => _explorer_detail_flow_payload(by_activity, meta_by_tech),
        "activities" => activities,
    )
end

function _prepare_explorer_model_data!(md::ModelData; period::Int, mode::Symbol, representative_days::Int, hours_per_day::Int, clustering::Symbol)
    md.sets.periods_solve = [period]
    md.params.hoursPer_day = mode == :ts ? 24 : hours_per_day
    md.params.n_repDays = max(1, representative_days)
    md.params.hoursPer_day_cluster = 24
    md.params.clustering_approach = clustering
    md.params.ts_extremePeriods = false
    md.params.ts_extremeDays_count = 0
    md.params.ts_boundaryRamping = true
    md.params.ts_capacityProfile_autoMode = true
    md.params.ts_capacityProfile_autoFloor = 0.23
    md.params.ts_capacityProfile_autoCap = 1.00
    md.params.ts_capacityProfile_autoFloor_effective = 0.23
    md.params.ts_capacityProfile_envelopeMode = 0
    md.params.dayMix_softness = 0.0
    md.params.dayMix_weightType = :auto
    derive_sets!(md)
    compute_derived_params!(md)
    mode == :ts && build_temporal_clusters!(md)
    return md
end

function _explorer_family_from_name(name::AbstractString)
    text = strip(String(name))
    isempty(text) && return "unnamed"
    idx = findfirst(==('['), text)
    idx === nothing && return text
    return text[begin:prevind(text, idx)]
end

function _explorer_variable_families(model::JuMP.Model)
    groups = Dict{String,Dict{String,Any}}()
    for var in JuMP.all_variables(model)
        name = String(JuMP.name(var))
        family = _explorer_family_from_name(name)
        row = get!(groups, family, Dict("kind" => "Variable", "family" => family, "count" => 0, "type" => "VariableRef", "examples" => String[]))
        row["count"] = Int(row["count"]) + 1
        examples = row["examples"]
        length(examples) < 4 && push!(examples, isempty(name) ? family : name)
    end
    rows = collect(values(groups))
    sort!(rows; by = r -> (String(r["kind"]), String(r["family"])))
    return rows
end

function _explorer_constraint_families(model::JuMP.Model)
    groups = Dict{String,Dict{String,Any}}()
    for (func_type, set_type) in JuMP.list_of_constraint_types(model)
        constraint_type = replace("$(func_type) in $(set_type)", "MathOptInterface." => "MOI.")
        for cref in JuMP.all_constraints(model, func_type, set_type)
            name = String(JuMP.name(cref))
            family = _explorer_family_from_name(name)
            key = family * "\0" * constraint_type
            row = get!(groups, key, Dict("kind" => "Constraint", "family" => family, "count" => 0, "type" => constraint_type, "examples" => String[]))
            row["count"] = Int(row["count"]) + 1
            examples = row["examples"]
            length(examples) < 4 && push!(examples, isempty(name) ? family : name)
        end
    end
    rows = collect(values(groups))
    sort!(rows; by = r -> (String(r["kind"]), String(r["family"])))
    return rows
end

function _explorer_model_browser(body)
    input_path, workbook = _resolve_explorer_input(body)
    md = _read_ui_data_cached(input_path)
    periods = _explorer_periods(md)
    period = _explorer_selected_period(body, periods)
    mode_raw = lowercase(String(_config_get(body, "mode", "timeslice")))
    mode = mode_raw in ("full_hourly", "fh", "full-hourly") ? :fh : (mode_raw == "annual" ? :annual : :ts)
    representative_days = clamp(_as_int(_config_get(body, "representativeDays", 1), 1), 1, 30)
    hours_per_day = clamp(_as_int(_config_get(body, "hoursPerDay", 24), 24), 1, 24)
    clustering = Symbol(String(_config_get(body, "clusteringApproach", "kmeans_avg")))

    _, prepare_seconds = _elapsed() do
        _prepare_explorer_model_data!(md; period = period, mode = mode, representative_days = representative_days, hours_per_day = hours_per_day, clustering = clustering)
    end
    model = JuMP.Model()
    apply_lp_generation_speedups!(model; keep_names = true)
    _, build_seconds = _elapsed() do
        if mode == :annual
            build_annual_lp!(model, md)
        elseif mode == :fh
            build_fh_lp!(model, md)
        else
            build_ts_lp!(model, md)
        end
    end

    variable_rows = _explorer_variable_families(model)
    constraint_rows = _explorer_constraint_families(model)
    n_rows = try
        JuMP.num_constraints(model; count_variable_in_set_constraints = false)
    catch
        sum(Int(r["count"]) for r in constraint_rows)
    end
    n_cols = JuMP.num_variables(model)
    return Dict(
        "inputWorkbook" => workbook,
        "periods" => periods,
        "selectedPeriod" => period,
        "mode" => String(mode),
        "representativeDays" => representative_days,
        "hoursPerDay" => hours_per_day,
        "prepareSeconds" => prepare_seconds,
        "buildSeconds" => build_seconds,
        "rows" => n_rows,
        "columns" => n_cols,
        "variables" => variable_rows,
        "constraints" => constraint_rows,
    )
end

function _explorer_value_or_nothing(value)
    value === nothing && return nothing
    value isa Real || return value
    n = Float64(value)
    isfinite(n) || return nothing
    return _explorer_round(n)
end

function _explorer_sheet_inventory(input_path::AbstractString)
    rows = Vector{Dict{String,Any}}()
    XLSX.openxlsx(input_path, mode = "r") do xf
        for name in XLSX.sheetnames(xf)
            dims = try
                data = xf[name][:]
                (size(data, 1), size(data, 2))
            catch
                (0, 0)
            end
            sheet_group = if name in ("Technologies", "Infrastructure", "EnergyBalance", "Activities", "Feedstocks", "EffLearning", "Retrofitting")
                "Technology system"
            elseif occursin("Profiles", name) || occursin("Hourly", name) || occursin("Extreme", name)
                "Time series"
            elseif name in ("NodeParameters", "Parameters", "Types", "Ranges", "ActGrouping")
                "Configuration"
            else
                "Reference"
            end
            push!(rows, Dict(
                "sheet" => String(name),
                "group" => sheet_group,
                "rows" => Int(dims[1]),
                "columns" => Int(dims[2]),
                "cells" => Int(dims[1] * dims[2]),
            ))
        end
    end
    sort!(rows; by = r -> (-Int(r["cells"]), String(r["sheet"])))
    return rows
end

function _explorer_set_counts(md::ModelData)
    s = md.sets
    rows = [
        ("Technologies", length(s.technologies), "Technology system"),
        ("Balancing technologies", length(s.tech_balancers), "Technology system"),
        ("Infrastructure technologies", length(s.tech_infra), "Technology system"),
        ("Activities", length(s.activities), "Activities"),
        ("Original activities", length(s.activities_original), "Activities"),
        ("Nodes", length(s.nodes), "Geography"),
        ("Periods", length(s.periods), "Time"),
        ("Hourly profile types", length(s.profile_typeRead), "Time series"),
        ("Process types", length(s.process_type), "Taxonomy"),
        ("Sectors", length(s.sectors), "Taxonomy"),
        ("KEV sectors", length(s.sectors_kev), "Taxonomy"),
        ("Energy labels", length(s.energy_labels), "Taxonomy"),
    ]
    return [Dict("name" => name, "count" => count, "group" => group) for (name, count, group) in rows]
end

function _explorer_sector_category_rows(md::ModelData)
    counts = Dict{Tuple{String,String},Int}()
    for t in _explorer_technology_universe(md)
        meta = _explorer_tech_meta(md, t)
        key = (String(meta["sector"]), String(meta["category"]))
        counts[key] = get(counts, key, 0) + 1
    end
    rows = [Dict("sector" => sector, "category" => category, "count" => count) for ((sector, category), count) in counts]
    sort!(rows; by = r -> (-Int(r["count"]), String(r["sector"]), String(r["category"])))
    return rows
end

function _explorer_technology_rows(md::ModelData, period::Int)
    p = md.params
    rows = Vector{Dict{String,Any}}()
    for t in _explorer_technology_universe(md)
        meta = _explorer_tech_meta(md, t)
        push!(rows, merge(meta, Dict(
            "period" => period,
            "investmentCost" => _explorer_value_or_nothing(get(p.inv_cost, (t, period), nothing)),
            "fixedOM" => _explorer_value_or_nothing(get(p.fom_cost, (t, period), nothing)),
            "variableOM" => _explorer_value_or_nothing(get(p.vom_cost, (t, period), nothing)),
            "economicLifetime" => _explorer_value_or_nothing(get(p.economic_lifetime, t, nothing)),
            "technicalLifetime" => _explorer_value_or_nothing(get(p.technical_lifetime, t, nothing)),
            "wacc" => _explorer_value_or_nothing(get(p.WACC, t, nothing)),
            "cap2act" => _explorer_value_or_nothing(get(p.cap2act, t, nothing)),
            "stockExisting" => _explorer_value_or_nothing(get(p.techStock_exist, t, nothing)),
        )))
    end
    return rows
end

function _explorer_activity_balance_atlas(md::ModelData, period::Int)
    sector_activity = Dict{Tuple{String,String},Float64}()
    sector_total = Dict{String,Float64}()
    activity_total = Dict{String,Float64}()
    output_total = Dict{Tuple{String,String},Float64}()
    input_total = Dict{Tuple{String,String},Float64}()
    for ((tech, activity, ps), coef) in _explorer_balance_source(md)
        ps == period || continue
        abs(coef) > 1e-12 || continue
        meta = _explorer_tech_meta(md, tech)
        sector = String(meta["sector"])
        activity_name = String(activity)
        key = (sector, activity_name)
        sector_activity[key] = get(sector_activity, key, 0.0) + Float64(coef)
        sector_total[sector] = get(sector_total, sector, 0.0) + abs(Float64(coef))
        activity_total[activity_name] = get(activity_total, activity_name, 0.0) + abs(Float64(coef))
        if coef > 0
            output_total[key] = get(output_total, key, 0.0) + Float64(coef)
        else
            input_total[key] = get(input_total, key, 0.0) + abs(Float64(coef))
        end
    end
    activities = sort!(collect(keys(activity_total)); by = a -> (-activity_total[a], a))[1:min(length(activity_total), 36)]
    sectors = sort!(collect(keys(sector_total)); by = s -> (-sector_total[s], s))[1:min(length(sector_total), 18)]
    z = [[_explorer_round(get(sector_activity, (sector, activity), 0.0)) for activity in activities] for sector in sectors]
    flows = Vector{Dict{String,Any}}()
    for sector in sectors, activity in activities
        key = (sector, activity)
        out = get(output_total, key, 0.0)
        inn = get(input_total, key, 0.0)
        abs(out) + abs(inn) > 1e-12 || continue
        push!(flows, Dict("sector" => sector, "activity" => activity, "output" => _explorer_round(out), "input" => _explorer_round(inn), "net" => _explorer_round(out - inn)))
    end
    sort!(flows; by = r -> -(abs(Float64(r["output"])) + abs(Float64(r["input"]))))
    return Dict("sectors" => sectors, "activities" => activities, "z" => z, "flows" => flows[1:min(length(flows), 80)])
end

function _explorer_activity_demand_rows(md::ModelData, period::Int)
    rows = Vector{Dict{String,Any}}()
    for ((activity, ps), value) in md.params.activities_netVolumes
        ps == period || continue
        meta_type = _explorer_symbol_label(get(md.params.activityType_act, activity, Symbol("")))
        dispatch = _explorer_symbol_label(get(md.params.dispatchType_act, activity, Symbol("")); fallback = "")
        node = _explorer_symbol_label(get(md.params.nodePer_act, activity, Symbol("")); fallback = "")
        push!(rows, Dict("activity" => String(activity), "type" => meta_type, "dispatch" => dispatch, "node" => node, "value" => _explorer_round(value)))
    end
    sort!(rows; by = r -> -abs(Float64(r["value"])))
    return rows[1:min(length(rows), 80)]
end

function _explorer_monthly_profile_heatmap(md::ModelData)
    profiles = sort!(collect(Set(a for (_, a) in keys(md.params.hourly_profilesReadOrig))); by = string)
    isempty(profiles) && return Dict("profiles" => String[], "months" => Int[], "z" => [])
    totals = Dict{Tuple{Symbol,Int},Float64}()
    counts = Dict{Tuple{Symbol,Int},Int}()
    for ((hour, profile), value) in md.params.hourly_profilesReadOrig
        month = get(md.params.monthPer_hourOrig, hour, 0)
        1 <= month <= 12 || continue
        key = (profile, month)
        totals[key] = get(totals, key, 0.0) + Float64(value)
        counts[key] = get(counts, key, 0) + 1
    end
    variation = Dict{Symbol,Float64}()
    for profile in profiles
        vals = [get(totals, (profile, m), 0.0) / max(get(counts, (profile, m), 0), 1) for m in 1:12]
        variation[profile] = maximum(vals) - minimum(vals)
    end
    selected = sort!(profiles; by = p -> (-variation[p], String(p)))[1:min(length(profiles), 24)]
    z = [[_explorer_round(get(totals, (profile, m), 0.0) / max(get(counts, (profile, m), 0), 1)) for m in 1:12] for profile in selected]
    return Dict("profiles" => String.(selected), "months" => collect(1:12), "z" => z)
end

function _explorer_profile_surfaces(md::ModelData)
    profiles = sort!(collect(Set(profile for (_, profile) in keys(md.params.hourly_profilesReadOrig))); by = string)
    isempty(profiles) && return Dict("profiles" => String[], "days" => Int[], "hours" => Int[], "surfaces" => Dict{String,Any}(), "summary" => [])
    max_hour = maximum(Int(hour) for (hour, _) in keys(md.params.hourly_profilesReadOrig))
    days_count = clamp(cld(max_hour, 24), 1, 366)
    days = collect(1:days_count)
    hours = collect(1:24)
    surfaces = Dict{String,Any}()
    summary = Vector{Dict{String,Any}}()
    for profile in profiles
        z = [zeros(Float64, days_count) for _ in 1:24]
        vals = Float64[]
        for ((hour, prof), value) in md.params.hourly_profilesReadOrig
            prof == profile || continue
            h = Int(hour)
            day = div(h - 1, 24) + 1
            1 <= day <= days_count || continue
            hour_day = mod(h - 1, 24) + 1
            v = Float64(value)
            z[hour_day][day] = _explorer_round(v)
            push!(vals, v)
        end
        label = String(profile)
        surfaces[label] = z
        if isempty(vals)
            push!(summary, Dict("profile" => label, "min" => 0.0, "max" => 0.0, "average" => 0.0, "spread" => 0.0))
        else
            mn = minimum(vals); mx = maximum(vals); avg = sum(vals) / length(vals)
            push!(summary, Dict("profile" => label, "min" => _explorer_round(mn), "max" => _explorer_round(mx), "average" => _explorer_round(avg), "spread" => _explorer_round(mx - mn)))
        end
    end
    sort!(summary; by = r -> (-Float64(r["spread"]), String(r["profile"])))
    return Dict("profiles" => String.(profiles), "days" => days, "hours" => hours, "surfaces" => surfaces, "summary" => summary)
end

function _explorer_policy_component(label::AbstractString, values; limit::Int = 80)
    raw_items = sort!(unique(string.(collect(values))))
    visible = raw_items[1:min(length(raw_items), limit)]
    return Dict(
        "label" => String(label),
        "count" => length(raw_items),
        "items" => visible,
        "truncated" => max(length(raw_items) - length(visible), 0),
    )
end

function _explorer_policy_target_values(target_dict; limit::Int = 80)
    rows = String[]
    for (key, value) in target_dict
        label = key isa Tuple ? join(string.(key), " / ") : string(key)
        push!(rows, "$(label) = $(_explorer_round(value))")
    end
    sort!(rows)
    return rows[1:min(length(rows), limit)]
end

function _explorer_policy_constraint_rows(md::ModelData)
    model_sets = md.sets
    model_params = md.params

    techs_at_node(node::Symbol) = [technology for technology in model_sets.tech_balancers if get(model_params.nodePer_techBal, technology, Symbol("")) == node]
    techs_with_activity(activities; tech_filter::Function = _ -> true) = begin
        activity_set = Set(Symbol.(activities))
        technology_set = Set{Symbol}()
        for ((technology, activity, _period), coefficient) in model_params.activity_balances
            coefficient == 0.0 && continue
            activity in activity_set || continue
            technology in model_sets.tech_balancers || continue
            tech_filter(technology) || continue
            push!(technology_set, technology)
        end
        collect(technology_set)
    end
    techs_with_activity_per(activity::Symbol) = [technology for technology in model_sets.tech_balancers if get(model_params.activityPer_tech, technology, Symbol("")) == activity]
    technologies_named(names::Vector{Symbol}) = [technology for technology in names if technology in model_sets.tech_balancers]
    technologies_matching(filter::Function) = [technology for technology in model_sets.tech_balancers if filter(technology)]
    period_values(target_dict) = _explorer_policy_target_values(target_dict)

    function components(; technologies = Symbol[], activities = Symbol[], targets = String[], notes = String[])
        groups = Vector{Dict{String,Any}}()
        isempty(technologies) || push!(groups, _explorer_policy_component("Technologies", technologies))
        isempty(activities) || push!(groups, _explorer_policy_component("Activities", activities))
        isempty(targets) || push!(groups, _explorer_policy_component("Target values", targets))
        isempty(notes) || push!(groups, _explorer_policy_component("Scenario control", notes; limit = 20))
        return groups
    end

    scenario_note = ["Included by the selected scenario or constraint group; this catalog shows the formulation contents, not a live enabled/disabled state."]
    nl_techs = techs_at_node(:NL)
    eu_techs = techs_at_node(:EU)
    bunker_navigation_techs = technologies_matching(technology -> get(model_params.tech_sector_kev, technology, Symbol("")) == :Bunkerbrandstoffen && get(model_params.tech_activity, technology, Symbol("")) == Symbol("Bunker Navigation"))
    bunker_aviation_techs = technologies_matching(technology -> get(model_params.tech_sector_kev, technology, Symbol("")) == :Bunkerbrandstoffen && get(model_params.tech_activity, technology, Symbol("")) == Symbol("Bunker Aviation"))
    refinery_fossil_techs = technologies_matching(technology -> get(model_params.tech_sector, technology, Symbol("")) == :Refineries && get(model_params.tech_subsector, technology, Symbol("")) == Symbol("Fossil Based"))
    ccus_storage_techs = technologies_matching(technology -> get(model_params.tech_subsector, technology, Symbol("")) == Symbol("CCUS Storage"))
    nuclear_techs = technologies_matching(technology -> occursin("Nuclear", get(model_params.tech_name, technology, "")))
    aviation_consumption_techs = techs_with_activity_per(Symbol("Bunker Aviation"))
    navigation_consumption_techs = techs_with_activity_per(Symbol("Bunker Navigation"))

    rows = [
        Dict("constraint" => "emTargetAir", "category" => "Base emission cap", "description" => "Limits annual air-emission activities by node and period using emissionTargetAir from NodeParameters.", "components" => components(
            technologies = union(nl_techs, technologies_named([:OPE01_03, :OPE02_03, :OPE03_03, :TNB01_05, :TNB01_08, :TAI01_03])),
            activities = union(model_sets.activities_target, model_sets.activities_target_FeedStocks, model_sets.activities_target_Bunkers),
            targets = period_values(model_params.emissionTargetAir), notes = scenario_note)),
        Dict("constraint" => "emTargetBunker", "category" => "Bunker emission cap", "description" => "Limits bunker-navigation and bunker-aviation target emissions where bunker target rows are present.", "components" => components(
            technologies = techs_with_activity(model_sets.activities_target_Bunkers), activities = model_sets.activities_target_Bunkers, targets = period_values(model_params.emissionTargetBunker), notes = scenario_note)),
        Dict("constraint" => "emTargetFS", "category" => "Feedstock emission cap", "description" => "Limits feedstock end-of-life CO2 target activity using the feedstock target columns.", "components" => components(
            technologies = techs_with_activity(model_sets.activities_target_FeedStocks), activities = model_sets.activities_target_FeedStocks, targets = period_values(model_params.emissionTargetFS), notes = scenario_note)),
        Dict("constraint" => "emTargetAll", "category" => "Scope 3 and fuel export cap", "description" => "Limits all counted emissions by node and period using target, feedstock, and bunker target activities.", "components" => components(
            technologies = union(techs_with_activity(model_sets.activities_target), techs_with_activity(model_sets.activities_target_FeedStocks), techs_with_activity(model_sets.activities_target_Bunkers)),
            activities = union(model_sets.activities_target, model_sets.activities_target_FeedStocks, model_sets.activities_target_Bunkers),
            targets = period_values(model_params.emissionTargetAll), notes = scenario_note)),
        Dict("constraint" => "emTargetInclScope3", "category" => "Derived NL scope cap", "description" => "Applies the derived NL inclusive Scope 3 and fuel export cap by period.", "components" => components(
            technologies = union(techs_with_activity(model_sets.activities_target; tech_filter = technology -> technology in nl_techs), techs_with_activity(model_sets.activities_target_FeedStocks; tech_filter = technology -> technology in union(nl_techs, eu_techs))),
            activities = union(model_sets.activities_target, model_sets.activities_target_FeedStocks), targets = period_values(model_params.emissionTarget_inclScope3andFuelex), notes = scenario_note)),
        Dict("constraint" => "emTargetCum", "category" => "Cumulative CO2 cap", "description" => "Constrains cumulative emissions over solved periods by node.", "components" => components(
            technologies = techs_with_activity([activity for activity in model_sets.activities if get(model_params.nodePer_act, activity, Symbol("")) in keys(model_params.emissionTarget_cum)]),
            activities = [activity for activity in model_sets.activities if get(model_params.nodePer_act, activity, Symbol("")) in keys(model_params.emissionTarget_cum)], targets = period_values(model_params.emissionTarget_cum), notes = scenario_note)),
        Dict("constraint" => "co2StorageCum", "category" => "Cumulative CO2 storage cap", "description" => "Limits cumulative stored CO2 by node where storage caps are configured.", "components" => components(
            technologies = ccus_storage_techs, activities = Symbol[], targets = period_values(model_params.cumulative_CO2storage), notes = scenario_note)),
        Dict("constraint" => "AdaptBunkNav50", "category" => "ADAPT bunker navigation target", "description" => "2050 sectoral emission target for bunker navigation after credit subtraction.", "components" => components(
            technologies = union(bunker_navigation_techs, technologies_named([:TNB01_10])), activities = union(model_sets.activities_emission, model_sets.activities_credits), targets = ["2050 bunker navigation RHS = 26.7"], notes = scenario_note)),
        Dict("constraint" => "AdaptBunkAvi50", "category" => "ADAPT bunker aviation target", "description" => "2050 sectoral emission target for bunker aviation after credit subtraction.", "components" => components(
            technologies = union(bunker_aviation_techs, technologies_named([:TAI01_07])), activities = union(model_sets.activities_emission, model_sets.activities_credits), targets = ["2050 bunker aviation RHS = 5.5"], notes = scenario_note)),
        Dict("constraint" => "AdaptRefinProd50", "category" => "ADAPT refinery production cap", "description" => "2050 cap on positive refinery energy output from fossil-based refinery technologies.", "components" => components(
            technologies = refinery_fossil_techs, activities = model_sets.activities_energy, targets = ["2050 refinery production RHS = 1202.0"], notes = scenario_note)),
        Dict("constraint" => "CO2credAvi", "category" => "Aviation CO2 credit balance", "description" => "Balances aviation CO2 credit technology output against e-kerosene and synthetic kerosene credit terms.", "components" => components(
            technologies = union(technologies_named([:TAI01_07]), aviation_consumption_techs), activities = union(model_sets.activities_credits, Symbol[Symbol("E-Kerosene"), Symbol("Syn Kerosene")]), notes = scenario_note)),
        Dict("constraint" => "CO2credNav", "category" => "Navigation CO2 credit balance", "description" => "Balances navigation CO2 credit output against methanol, e-methanol, and synthetic diesel credit terms.", "components" => components(
            technologies = union(technologies_named([:TNB01_10]), navigation_consumption_techs), activities = union(model_sets.activities_credits, Symbol[:Methanol, Symbol("E-Methanol"), Symbol("Syn Diesel")]), notes = scenario_note)),
        Dict("constraint" => "eSAF_Avi", "category" => "ReFuelEU aviation eSAF share", "description" => "Requires e-kerosene consumption in bunker aviation to meet the eSAF share target from 2030 onward.", "components" => components(
            technologies = aviation_consumption_techs, activities = Symbol[Symbol("E-Kerosene"), Symbol("Bunker Aviation")], targets = period_values(model_params.ReFuelEU_Aviation_eSAF_target), notes = scenario_note)),
        Dict("constraint" => "SAF_Avi", "category" => "ReFuelEU aviation SAF share", "description" => "Requires eligible SAF/eSAF/synthetic kerosene consumption in bunker aviation to meet the SAF share target.", "components" => components(
            technologies = aviation_consumption_techs, activities = Symbol[Symbol("E-Kerosene"), Symbol("Bio Kerosene"), Symbol("Syn Kerosene"), Symbol("Bunker Aviation")], targets = period_values(model_params.ReFuelEU_Aviation_SAF_target), notes = scenario_note)),
        Dict("constraint" => "H2credAvi", "category" => "Aviation hydrogen credit balance", "description" => "Links aviation hydrogen-credit technology output to e-kerosene use with the model credit factor.", "components" => components(
            technologies = union(technologies_named([:TAI01_06]), aviation_consumption_techs), activities = union(model_sets.activities_credits, Symbol[Symbol("E-Kerosene")]), targets = ["Aviation hydrogen credit factor = 2.1"], notes = scenario_note)),
        Dict("constraint" => "H2credNav", "category" => "Navigation hydrogen credit balance", "description" => "Links navigation hydrogen-credit technology output to ammonia and methanol bunker navigation fuels.", "components" => components(
            technologies = union(technologies_named([:TNB01_09]), navigation_consumption_techs), activities = union(model_sets.activities_credits, Symbol[:Ammonia, :Methanol]), targets = ["Ammonia credit factor = 1.15", "Methanol credit factor = 1.20"], notes = scenario_note)),
        Dict("constraint" => "SectorTgtBunkNav", "category" => "FuelEU maritime target", "description" => "Limits bunker-navigation target emissions net of CO2 credits using FuelEU Maritime target values.", "components" => components(
            technologies = union(bunker_navigation_techs, technologies_named([:TNB01_10])), activities = union(model_sets.activities_target_Bunkers, model_sets.activities_credits), targets = period_values(model_params.FuelEU_Maritime_target), notes = scenario_note)),
        Dict("constraint" => "MinLoad", "category" => "Nuclear minimum load", "description" => "Optional full-hourly nuclear minimum-load constraint. It mirrors an IESA-Opt 1.0 constraint that is not in standard groups.", "components" => components(
            technologies = nuclear_techs, activities = Symbol[:Flat], targets = ["Minimum hourly output = 30% of stock x cap2act x Flat profile"], notes = scenario_note)),
    ]
    sort!(rows; by = row -> String(row["constraint"]))
    return rows
end

function _explorer_target_rows(md::ModelData)
    p = md.params
    rows = Vector{Dict{String,Any}}()
    function add_period_dict!(label, dict)
        for ((node, period), value) in dict
            push!(rows, Dict("target" => label, "node" => String(node), "period" => Int(period), "value" => _explorer_round(value)))
        end
    end
    add_period_dict!("Air", p.emissionTargetAir)
    add_period_dict!("All", p.emissionTargetAll)
    add_period_dict!("Bunker", p.emissionTargetBunker)
    add_period_dict!("Feedstock", p.emissionTargetFS)
    for (node, value) in p.CO2_cumulative_budget
        push!(rows, Dict("target" => "Cumulative CO2 budget", "node" => String(node), "period" => 0, "value" => _explorer_round(value)))
    end
    for (period, value) in p.ReFuelEU_Aviation_eSAF_target
        push!(rows, Dict("target" => "ReFuelEU eSAF", "node" => "Policy", "period" => Int(period), "value" => _explorer_round(value)))
    end
    for (period, value) in p.ReFuelEU_Aviation_SAF_target
        push!(rows, Dict("target" => "ReFuelEU SAF", "node" => "Policy", "period" => Int(period), "value" => _explorer_round(value)))
    end
    for (period, value) in p.FuelEU_Maritime_target
        push!(rows, Dict("target" => "FuelEU Maritime", "node" => "Policy", "period" => Int(period), "value" => _explorer_round(value)))
    end
    sort!(rows; by = r -> (String(r["target"]), String(r["node"]), Int(r["period"])))
    return rows
end

function _explorer_input_atlas(body)
    input_path, workbook = _resolve_explorer_input(body)
    md = _read_ui_data_cached(input_path)
    periods = _explorer_periods(md)
    period = _explorer_selected_period(body, periods)
    tech_rows = _explorer_technology_rows(md, period)
    sheet_rows = _explorer_sheet_inventory(input_path)
    return Dict(
        "inputWorkbook" => workbook,
        "periods" => periods,
        "selectedPeriod" => period,
        "sheets" => sheet_rows,
        "setCounts" => _explorer_set_counts(md),
        "sectorCategories" => _explorer_sector_category_rows(md),
        "technologies" => tech_rows,
        "balance" => _explorer_activity_balance_atlas(md, period),
        "demands" => _explorer_activity_demand_rows(md, period),
        "profiles" => _explorer_monthly_profile_heatmap(md),
        "profileSurfaces" => _explorer_profile_surfaces(md),
        "targets" => _explorer_target_rows(md),
        "policyConstraints" => _explorer_policy_constraint_rows(md),
        "metrics" => Dict(
            "sheets" => length(sheet_rows),
            "technologies" => length(tech_rows),
            "activities" => length(md.sets.activities),
            "nodes" => length(md.sets.nodes),
            "periods" => length(periods),
            "profileTypes" => length(md.sets.profile_typeRead),
        ),
    )
end

function _detect_solvers()
    solvers = Vector{Dict{String,Any}}()
    highs_version = _solver_display_version(:HiGHS)
    highs_wrapper_version = _solver_wrapper_version(:HiGHS)
    highs_available = isdefined(@__MODULE__, :HiGHS)
    push!(solvers, Dict(
        "id" => "highs",
        "label" => "HiGHS",
        "available" => highs_available,
        "default" => false,
        "commercial" => false,
        "version" => highs_version,
        "wrapperVersion" => highs_wrapper_version,
        "message" => ""
    ))
    gurobi_version = _solver_display_version(:Gurobi)
    gurobi_wrapper_version = _solver_wrapper_version(:Gurobi)
    gurobi_available = isdefined(@__MODULE__, :Gurobi)
    push!(solvers, Dict(
        "id" => "gurobi",
        "label" => "Gurobi",
        "available" => gurobi_available,
        "default" => false,
        "commercial" => true,
        "version" => gurobi_version,
        "wrapperVersion" => gurobi_wrapper_version,
        "message" => gurobi_available ? _commercial_solver_message(:Gurobi, "Gurobi") : ""
    ))
    cplex_ok, cplex_msg, cplex_version = _try_import_solver_module(:CPLEX, "CPLEX")
    push!(solvers, Dict("id" => "cplex", "label" => "CPLEX", "available" => cplex_ok, "default" => false, "commercial" => true, "version" => cplex_version, "message" => cplex_msg))
    xpress_ok, xpress_msg, xpress_version = _try_import_solver_module(:Xpress, "XPRESS")
    push!(solvers, Dict("id" => "xpress", "label" => "XPRESS", "available" => xpress_ok, "default" => false, "commercial" => true, "version" => xpress_version, "message" => xpress_msg))
    push!(solvers, Dict("id" => "auto", "label" => "Auto", "available" => true, "default" => false, "commercial" => false, "version" => "", "message" => ""))
    default_id = _preferred_solver_id(solvers)
    for solver in solvers
        solver["default"] = solver["id"] == default_id
    end
    return solvers
end

function _try_import_solver_module(name::Symbol, label::AbstractString = String(name))
    if isdefined(@__MODULE__, name)
        version = _solver_display_version(name)
        return true, _commercial_solver_message(name, label), version
    end
    try
        Core.eval(@__MODULE__, Meta.parse("import $(String(name))"))
        version = _solver_display_version(name)
        return true, _commercial_solver_message(name, label), version
    catch err
        return false, "", ""
    end
end

function _solver_display_version(name::Symbol)
    native_version = _solver_native_version(name)
    return isempty(native_version) ? _solver_wrapper_version(name) : native_version
end

function _solver_wrapper_version(name::Symbol)
    isdefined(@__MODULE__, name) || return ""
    solver_module = getfield(@__MODULE__, name)
    try
        version = Base.pkgversion(solver_module)
        version === nothing && return ""
        return "$(String(name)).jl $(version)"
    catch
        return ""
    end
end

function _solver_native_version(name::Symbol)
    isdefined(@__MODULE__, name) || return ""
    name == :Gurobi && return _gurobi_native_version()
    return ""
end

function _gurobi_native_version()
    isdefined(@__MODULE__, :Gurobi) || return ""
    try
        major = getfield(Gurobi, :GRB_VERSION_MAJOR)
        minor = getfield(Gurobi, :GRB_VERSION_MINOR)
        technical = getfield(Gurobi, :GRB_VERSION_TECHNICAL)
        return "$(major).$(minor).$(technical)"
    catch
        return ""
    end
end

function _commercial_solver_message(name::Symbol, label::AbstractString)
    name == :Gurobi && return _gurobi_license_message()
    return ""
end

function _gurobi_license_message()
    info = _gurobi_license_file_info()
    isempty(info) && return ""
    license_type = _license_type_label(get(info, "TYPE", get(info, "LICENSE_TYPE", "")))
    if isempty(license_type) && (haskey(info, "TOKENSERVER") || haskey(info, "PORT"))
        license_type = "Network license"
    end
    expires = strip(get(info, "EXPIRATION", get(info, "EXPIRES", get(info, "EXPIRATION_DATE", ""))))
    parts = String[]
    isempty(license_type) || push!(parts, license_type)
    isempty(expires) || push!(parts, "expires $(expires)")
    return join(parts, ", ")
end

function _license_type_label(value)
    text = strip(String(value))
    isempty(text) && return ""
    normalized = uppercase(replace(text, '_' => ' ', '-' => ' '))
    occursin("ACADEMIC", normalized) && return "Academic license"
    occursin("COMMERCIAL", normalized) && return "Commercial license"
    occursin("WLS", normalized) && return "WLS license"
    occursin("TOKEN", normalized) && return "Network license"
    return titlecase(lowercase(normalized)) * " license"
end

function _gurobi_license_file_info()
    for path in _gurobi_license_file_candidates()
        isfile(path) || continue
        info = Dict{String,String}()
        try
            for line in eachline(path)
                stripped = strip(line)
                isempty(stripped) && continue
                startswith(stripped, "#") && continue
                occursin("=", stripped) || continue
                key, value = split(stripped, "="; limit = 2)
                info[uppercase(strip(key))] = strip(value)
            end
        catch
            empty!(info)
        end
        isempty(info) || return info
    end
    return Dict{String,String}()
end

function _gurobi_license_file_candidates()
    candidates = String[]
    if haskey(ENV, "GRB_LICENSE_FILE")
        raw_value = strip(ENV["GRB_LICENSE_FILE"])
        if !isempty(raw_value)
            values = Sys.iswindows() ? split(raw_value, ';') : split(raw_value, ':')
            append!(candidates, strip.(String.(values)))
            push!(candidates, raw_value)
        end
    end
    push!(candidates, joinpath(homedir(), "gurobi.lic"))
    if Sys.iswindows()
        push!(candidates, raw"C:\gurobi\gurobi.lic")
    else
        push!(candidates, "/opt/gurobi/gurobi.lic")
    end
    haskey(ENV, "GUROBI_HOME") && push!(candidates, joinpath(ENV["GUROBI_HOME"], "gurobi.lic"))
    return unique(filter(path -> !isempty(path) && !occursin('@', path), candidates))
end

function _preferred_solver_id(solvers::Vector{Dict{String,Any}})
    for solver_id in COMMERCIAL_SOLVER_IDS
        any(solver -> solver["id"] == solver_id && solver["available"] == true, solvers) && return solver_id
    end
    any(solver -> solver["id"] == "highs" && solver["available"] == true, solvers) && return "highs"
    return "auto"
end

function _preferred_default_solver_id()
    isdefined(@__MODULE__, :Gurobi) && return "gurobi"
    cplex_ok, _, _ = _try_import_solver_module(:CPLEX, "CPLEX")
    cplex_ok && return "cplex"
    xpress_ok, _, _ = _try_import_solver_module(:Xpress, "XPRESS")
    xpress_ok && return "xpress"
    return isdefined(@__MODULE__, :HiGHS) ? "highs" : "auto"
end

_short_error(err) = first(split(sprint(showerror, err), '\n'))

# Report constraints whose elastic slack is non-zero after a "Show
# violations" run.  Emits a single multi-line job log message so users
# can see the worst offenders in the Progress tab.
function _report_elastic_slacks!(job_id::AbstractString, penalty_map; max_show::Int = 25)
    rows = try
        report_nonzero_slacks(penalty_map)
    catch err
        _job_update!(job_id; stage = "solve",
                     message = "Could not extract slack values: $(_short_error(err))")
        return
    end
    if isempty(rows)
        _job_update!(job_id; stage = "solve",
                     message = "Show violations: model is naturally feasible — no constraint needed slack.")
        return
    end
    top = rows[1:min(max_show, length(rows))]
    fams = unique([r.family for r in rows])
    lines = String[]
    push!(lines, "Show violations: $(length(rows)) constraints needed slack to make the model feasible.")
    push!(lines, "Families involved: $(join(fams, ", "))")
    push!(lines, "Top $(length(top)) by slack magnitude:")
    for r in top
        push!(lines, "  • $(r.name)  slack = $(round(r.slack; sigdigits = 4))")
    end
    _job_update!(job_id; stage = "solve", message = join(lines, "\n"),
                 extra = Dict("violations" => Dict(
                    "count" => length(rows),
                    "families" => fams,
                    "top" => [Dict("name" => r.name, "slack" => r.slack, "family" => r.family) for r in top])))
end

# IIS analysis when an unrelaxed model returns INFEASIBLE.  Commercial
# solvers only (Gurobi / CPLEX / Xpress); for others, emit a hint.
function _report_iis!(job_id::AbstractString, model::JuMP.Model, solver_label::AbstractString)
    _job_update!(job_id; stage = "solve",
                 message = "Model is infeasible — running IIS analysis with $(solver_label)…")
    rep = try
        compute_iis_report(model)
    catch err
        _job_update!(job_id; stage = "solve",
                     message = "IIS analysis failed: $(_short_error(err)). Enable Show violations for an elastic re-solve that works on every solver.")
        return
    end
    if !rep.supported
        msg = "$(solver_label) does not implement IIS / conflict refinement"
        isempty(rep.error) || (msg *= " ($(rep.error))")
        msg *= ". Switch to Gurobi / CPLEX / Xpress, or enable Show violations to find the offending constraints with an elastic re-solve."
        _job_update!(job_id; stage = "solve", message = msg)
        return
    end
    if isempty(rep.names)
        msg = "IIS analysis ran but the solver reported no conflict."
        isempty(rep.error) || (msg *= " $(rep.error)")
        _job_update!(job_id; stage = "solve", message = msg)
        return
    end
    top = rep.names[1:min(25, length(rep.names))]
    lines = String[]
    push!(lines, "Irreducible infeasible subsystem: $(length(rep.names)) constraints.")
    push!(lines, "Families involved: $(join(rep.families, ", "))")
    push!(lines, "First $(length(top)):")
    for n in top
        push!(lines, "  • $(n)")
    end
    hint = iis_suggestion(rep.families)
    isempty(hint) || push!(lines, "Suggestion: $(hint)")
    _job_update!(job_id; stage = "solve", message = join(lines, "\n"),
                 extra = Dict("iis" => Dict(
                    "count" => length(rep.names),
                    "families" => rep.families,
                    "names" => top,
                    "suggestion" => hint)))
end

function _start_ui_job!(raw_config)
    config = _normalize_run_config(raw_config)
    job_id = Dates.format(now(), "yyyymmdd_HHMMSS") * "_" * randstring(6)
    created_at = string(now())
    created_epoch = time()
    job = Dict{String,Any}(
        "id" => job_id,
        "status" => "queued",
        "stage" => "queued",
        "createdAt" => created_at,
        "createdAtEpoch" => created_epoch,
        "updatedAt" => created_at,
        "config" => config,
        "logs" => Vector{Dict{String,String}}(),
        "outputDir" => "",
        "effectiveSolver" => "",
        "resultReady" => false,
        "cancelRequested" => false,
        "model" => nothing
    )
    lock(UI_JOBS_LOCK)
    try
        UI_JOBS[job_id] = job
    finally
        unlock(UI_JOBS_LOCK)
    end
    task = Base.Threads.@spawn _run_ui_job!(job_id, config, created_epoch)
    lock(UI_JOBS_LOCK)
    try
        UI_TASKS[job_id] = task
    finally
        unlock(UI_JOBS_LOCK)
    end
    return job_id
end

function _normalize_run_config(raw_config)
    input_value = String(_config_get(raw_config, "inputWorkbook", "Input/1108 SSP.xlsx"))
    input_path, input_rel = _resolve_input_workbook(input_value)

    mode_raw = lowercase(String(_config_get(raw_config, "mode", "timeslice")))
    mode = mode_raw in ("full_hourly", "fh", "full-hourly") ? "full_hourly" : "timeslice"
    periods = _as_int_vector(_config_get(raw_config, "periods", [2050]))
    isempty(periods) && error("Select at least one solve period")

    output_mode = lowercase(String(_config_get(raw_config, "outputMode", "automatic")))
    output_name = String(_config_get(raw_config, "outputName", ""))
    scenario_name = splitext(basename(input_path))[1]
    if output_mode == "custom" && !isempty(strip(output_name))
        out_name = _sanitize_run_name(output_name)
    else
        stamp = Dates.format(now(), "yymmdd_HHMMSS")
        rep_days = _as_int(_config_get(raw_config, "representativeDays", 15), 15)
        hours_per_day = _as_int(_config_get(raw_config, "hoursPerDay", 24), 24)
        mode_tag = mode == "timeslice" ? "$(rep_days)rd" : "$(hours_per_day)h"
        periods_tag = isempty(periods) ? "" : join(string.(periods), "-")
        out_name = isempty(periods_tag) ? "$(stamp)_$(mode_tag)" : "$(stamp)_$(mode_tag)_$(periods_tag)"
    end

    return Dict{String,Any}(
        "inputWorkbook" => input_rel,
        "inputPath" => input_path,
        "scenario" => scenario_name,
        "periods" => periods,
        "mode" => mode,
        "hoursPerDay" => _as_int(_config_get(raw_config, "hoursPerDay", 24), 24),
        "representativeDays" => _as_int(_config_get(raw_config, "representativeDays", 15), 15),
        "solver" => lowercase(String(_config_get(raw_config, "solver", "highs"))),
        "solveMethod" => lowercase(String(_config_get(raw_config, "solveMethod", "barrier_crossover"))),
        "threads" => _as_int(_config_get(raw_config, "threads", 0), 0),
        "clusteringApproach" => lowercase(String(_config_get(raw_config, "clusteringApproach", "kmeans_avg"))),
        "extremePeriods" => _as_bool(_config_get(raw_config, "extremePeriods", true), true),
        "extremeDays" => _as_int(_config_get(raw_config, "extremeDays", 5), 5),
        "boundaryRamping" => _as_bool(_config_get(raw_config, "boundaryRamping", true), true),
        "hourlyReports" => _as_bool(_config_get(raw_config, "hourlyReports", true), true),
        "showViolations" => _as_bool(_config_get(raw_config, "showViolations", false), false),
        "multiRegion" => _as_bool(_config_get(raw_config, "multiRegion", false), false),
        "constraintGroup" => String(_config_get(raw_config, "constraintGroup", "Base + Bunkers + Scope3")),
        "outputMode" => output_mode,
        "outputName" => out_name,
        "outputDir" => normpath(joinpath(_repo_root(), "Output", out_name))
    )
end

function _sanitize_run_name(name::AbstractString)
    cleaned = replace(strip(name), r"[^A-Za-z0-9_.-]+" => "_")
    isempty(cleaned) && error("Custom output name is empty after sanitizing")
    return cleaned
end

function _config_get(config, key::String, default)
    if config isa AbstractDict
        haskey(config, key) && return config[key]
        sym = Symbol(key)
        haskey(config, sym) && return config[sym]
    end
    try
        value = getproperty(config, Symbol(key))
        value === nothing || return value
    catch
    end
    try
        return config[key]
    catch
    end
    return default
end

function _as_int(value, default::Int)
    value === nothing && return default
    value isa Integer && return Int(value)
    value isa Real && return Int(round(value))
    try
        return parse(Int, String(value))
    catch
        return default
    end
end

function _as_bool(value, default::Bool)
    value === nothing && return default
    value isa Bool && return value
    value isa Number && return value != 0
    text = lowercase(String(value))
    return text in ("1", "true", "yes", "on")
end

function _as_int_vector(value)
    if value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return Int[]
        return [parse(Int, strip(part)) for part in split(stripped, ',')]
    end
    return [Int(v) for v in collect(value)]
end

function _as_string_vector(value)
    value === nothing && return String[]
    value isa AbstractString && return isempty(strip(value)) ? String[] : [String(value)]
    return [String(v) for v in collect(value)]
end

function _job_update!(job_id::String; status = nothing, stage = nothing, message = nothing, extra = Dict{String,Any}())
    lock(UI_JOBS_LOCK)
    try
        job = get(UI_JOBS, job_id, nothing)
        job === nothing && return nothing
        status !== nothing && (job["status"] = status)
        stage !== nothing && (job["stage"] = stage)
        job["updatedAt"] = string(now())
        for (key, value) in extra
            job[key] = value
        end
        if message !== nothing
            push!(job["logs"], Dict(
                "time" => Dates.format(now(), "HH:MM:SS"),
                "stage" => String(job["stage"]),
                "message" => String(message)
            ))
            if length(job["logs"]) > UI_MAX_LOG_LINES
                deleteat!(job["logs"], 1:(length(job["logs"]) - UI_MAX_LOG_LINES))
            end
        end
    finally
        unlock(UI_JOBS_LOCK)
    end
    yield()
    return nothing
end
function _job_snapshot(job_id::String)
    lock(UI_JOBS_LOCK)
    try
        haskey(UI_JOBS, job_id) || error("Unknown job id: $job_id")
        # Drop the live JuMP model from the snapshot — it cannot be JSON encoded.
        snap = copy(UI_JOBS[job_id])
        snap["model"] = nothing
        return snap
    finally
        unlock(UI_JOBS_LOCK)
    end
end

function _elapsed(f)
    start = time()
    value = f()
    return value, round(time() - start, digits = 3)
end

function _run_ui_job!(job_id::String, config::Dict{String,Any}, queued_start::Float64 = time())
    stage_times = Dict{String,Float64}()
    run_start = time()
    stage_times["queue_sec"] = round(max(0.0, run_start - queued_start), digits = 3)
    total_start = queued_start
    try
        _check_cancel(job_id) && throw(UICancelled("Run was stopped before reading the workbook."))
        memory_hit = _ui_model_data_cache_hit(config["inputPath"])
        duckdb_ready = !memory_hit && _ui_data_cache_valid(config["inputPath"])
        cache_warm_running = !memory_hit && !duckdb_ready && _ui_cache_warm_running(config["inputPath"])
        read_message = memory_hit ?
            "Loading workbook $(config["inputWorkbook"]) from in-memory cache" :
            duckdb_ready ?
                "Loading workbook $(config["inputWorkbook"]) from DuckDB input cache" :
                cache_warm_running ?
                    "Waiting for workbook cache warm-up, then loading $(config["inputWorkbook"])" :
                    "Reading workbook $(config["inputWorkbook"]) from XLSX and building DuckDB input cache for faster next runs"
        _job_update!(job_id; status = "running", stage = "reading", message = read_message)
        md, read_seconds = _elapsed() do
            _read_ui_data_cached(config["inputPath"])
        end
        stage_times["data_read_sec"] = read_seconds
        done_message = memory_hit ?
            "Workbook loaded from in-memory cache in $(read_seconds) seconds" :
            duckdb_ready ?
                "Workbook loaded from DuckDB input cache in $(read_seconds) seconds" :
                "Workbook read and DuckDB input cache updated in $(read_seconds) seconds"
        _job_update!(job_id; stage = "reading", message = done_message)

        selected_periods = [p for p in config["periods"] if p in md.sets.periods]
        isempty(selected_periods) && error("Selected periods are not present in the workbook: $(config["periods"])")
        md.sets.periods_solve = selected_periods

        mode = config["mode"] == "full_hourly" ? :fh : :ts
        md.params.hoursPer_day = mode == :ts ? 24 : config["hoursPerDay"]
        md.params.n_repDays = max(1, config["representativeDays"])
        md.params.hoursPer_day_cluster = 24
        md.params.clustering_approach = Symbol(config["clusteringApproach"])
        md.params.ts_extremePeriods = config["extremePeriods"]
        md.params.ts_extremeDays_count = max(0, config["extremeDays"])
        md.params.ts_boundaryRamping = config["boundaryRamping"]
        md.params.ts_capacityProfile_autoMode = true
        md.params.ts_capacityProfile_autoFloor = 0.23
        md.params.ts_capacityProfile_autoCap = 1.00
        md.params.ts_capacityProfile_autoFloor_effective = 0.23
        md.params.ts_capacityProfile_envelopeMode = 0
        md.params.dayMix_softness = 0.0
        md.params.dayMix_weightType = :auto
        # ---- Project-specific extensions (opt-in) -------------------------
        # Each toggle in the UI maps to a Symbol in md.params.extensions.
        # An empty set leaves the core model unchanged.
        md.params.extensions = Set{Symbol}()
        if get(config, "multiRegion", false) === true
            push!(md.params.extensions, :multi_region)
            _job_update!(job_id; stage = "reading", message = "MultiRegion extension is ENABLED for this run")
        end
        _job_update!(job_id; stage = "reading", message = "Selected solve periods: $(join(string.(selected_periods), ", "))")

        _job_update!(job_id; stage = "preparing", message = "Deriving sets and parameters for periods $(selected_periods)")
        _, derive_seconds = _elapsed() do
            derive_sets!(md)
            compute_derived_params!(md)
        end
        stage_times["derive_sec"] = derive_seconds
        _job_update!(job_id; stage = "preparing", message = "Derived sets and parameters in $(derive_seconds) seconds")

        if mode == :ts
            _check_cancel(job_id) && throw(UICancelled("Run was stopped before clustering."))
            _job_update!(job_id; stage = "clustering", message = "Preparing $(md.params.n_repDays) representative days x $(md.params.hoursPer_day_cluster) hours")
            _, cluster_seconds = _elapsed() do
                build_temporal_clusters!(md)
            end
            stage_times["cluster_sec"] = cluster_seconds
            _job_update!(job_id; stage = "clustering", message = "Clustering complete in $(cluster_seconds) seconds")
        else
            stage_times["cluster_sec"] = 0.0
            _job_update!(job_id; stage = "clustering", message = "Full-hourly mode selected; representative-day clustering is skipped")
        end

        _check_cancel(job_id) && throw(UICancelled("Run was stopped before model generation."))
        solver_log_path = joinpath(tempdir(), "iesa_opt_$(job_id)_solver.log")
        rm(solver_log_path; force = true)
        # Optimizer/JuMP wiring time (license check + attribute apply + Model
        # construction). Surfaced as its own bar in the timing chart so the
        # residual "Other" segment shrinks.
        local optimizer, attrs, effective_solver, model
        _, optimizer_init_seconds = _elapsed() do
            optimizer, attrs, effective_solver = _optimizer_for_ui_run(config["solver"], config["solveMethod"], config["threads"]; solver_log_path = solver_log_path, rep_days = config["representativeDays"])
            model_label = mode == :ts ? "time-slice" : "full-hourly"
            _job_update!(job_id; stage = "generation", message = "Generating $(model_label) model with $(effective_solver)", extra = Dict("effectiveSolver" => effective_solver))
            model = Model(optimizer)
            # Result reporting extracts commodity shadow prices by constraint
            # name, and diagnostics also need readable constraint labels.
            apply_lp_generation_speedups!(model; keep_names = true)
            _set_job_model!(job_id, model)
        end
        stage_times["optimizer_init_sec"] = optimizer_init_seconds
        vars, generation_seconds = _elapsed() do
            mode == :ts ? build_ts_lp!(model, md) : build_fh_lp!(model, md)
        end
        stage_times["generation_sec"] = generation_seconds
        # Optional elastic relaxation (Show violations checkbox).  Wraps every
        # non-bound constraint with slack + penalty so the model is always
        # feasible; we report the constraints with non-zero slack after solve.
        elastic_penalties = nothing
        if config["showViolations"] === true
            _job_update!(job_id; stage = "generation",
                         message = "Adding elastic slack variables (Show violations enabled)…")
            try
                elastic_penalties = apply_elastic_relaxation!(model)
                _job_update!(job_id; stage = "generation",
                             message = "Elastic slacks added on $(length(elastic_penalties)) constraints (penalty = $(_ELASTIC_DEFAULT_PENALTY) per unit). Objective will be inflated by the cost of any slack used.")
            catch err
                elastic_penalties = nothing
                _job_update!(job_id; stage = "generation",
                             message = "Failed to add elastic slacks: $(_short_error(err)). Continuing without violation diagnostics.")
            end
        end
        n_rows = try
            num_constraints(model; count_variable_in_set_constraints = false)
        catch
            0
        end
        n_cols = num_variables(model)
        _job_update!(job_id; stage = "generation", message = "Model generated in $(generation_seconds) seconds ($(n_rows) rows, $(n_cols) columns)", extra = Dict("nRows" => n_rows, "nCols" => n_cols))

        _check_cancel(job_id) && throw(UICancelled("Run was stopped before solving."))
        _job_update!(job_id; stage = "solve", message = "Solving with $(effective_solver) using $(config["solveMethod"]) on $(n_rows) rows and $(n_cols) columns")
        _, solve_seconds = _elapsed() do
            _optimize_with_solver_progress!(model, job_id, effective_solver, solver_log_path)
        end
        stage_times["solve_sec"] = solve_seconds
        term = string(termination_status(model))
        primal = string(primal_status(model))
        if _check_cancel(job_id) || term in ("INTERRUPTED", "USER_LIMIT")
            _job_update!(job_id; status = "cancelled", stage = "cancelled",
                         message = "Run stopped by user (solver status: $(term)). No results were written.",
                         extra = Dict("cancelled" => true, "resultReady" => false))
            return nothing
        end
        obj = try
            objective_value(model)
        catch
            NaN
        end
        _job_update!(job_id; stage = "solve", message = "Solve complete in $(solve_seconds) seconds, status=$(term), objective=$(round(obj, digits = 4))")

        # Diagnostics: report constraints with nonzero slack (elastic mode)
        # or run an IIS analysis when the model came back infeasible.
        if elastic_penalties !== nothing
            _report_elastic_slacks!(job_id, elastic_penalties)
        elseif term == "INFEASIBLE"
            _report_iis!(job_id, model, effective_solver)
        end

        out_dir = config["outputDir"]
        mkpath(out_dir)
        rr = RunResult(
            out_dir, now(), mode,
            term, primal, _categorize_status(termination_status(model), primal_status(model)),
            Float64(obj), solve_seconds, round(time() - total_start, digits = 3),
            n_rows, n_cols, 0, 0, 0,
            Dict{String,Any}(attrs), config["scenario"],
            md.params.n_repDays, md.params.hoursPer_day,
            md.params.clustering_approach,
        )

        _job_update!(job_id; stage = "writing", message = "Writing DuckDB outputs to $(replace(relpath(out_dir, _repo_root()), '\\' => '/'))")
        written = Dict{Symbol,String}()
        writer_progress = _job_writer_progress(job_id)
        db_path = joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
        _remove_duckdb_database!(db_path)
        # Pulling JuMP duals back out of the solved model is pure Julia work
        # that runs after the solve.  On full-hourly runs with many balance
        # constraints this is the biggest residual segment, so we time it
        # explicitly and surface it in the timing chart.
        local co2_prices, emission_prices, activity_prices, activity_prices_hourly, activity_prices_daily
        _, dual_extract_seconds = _elapsed() do
            co2_prices = _extract_co2_prices(model, md)
            emission_prices = _extract_emission_prices(model, md)
            activity_prices = _extract_activity_prices(model, md)
            activity_prices_hourly = _extract_activity_prices_hourly(model, md, mode)
            activity_prices_daily  = _extract_activity_prices_daily(model, md, mode)
        end
        stage_times["dual_extract_sec"] = dual_extract_seconds
        _with_duckdb_write_connection(db_path; persist = true) do
            _, write_seconds = _elapsed() do
                merge!(written, write_duckdb_results(rr, vars, md, out_dir; mode = mode, reset = false,
                    co2_prices = co2_prices,
                    activity_prices = activity_prices,
                    emission_prices = emission_prices,
                    activity_prices_hourly = activity_prices_hourly,
                    activity_prices_daily = activity_prices_daily,
                    progress = writer_progress))
            end
            stage_times["results_write_sec"] = write_seconds
            stage_times["total_sec"] = round(time() - total_start, digits = 3)
            _write_ui_run_metadata(out_dir, config, stage_times, rr, effective_solver, attrs; progress = writer_progress)
        end
        rel_out = replace(relpath(out_dir, _repo_root()), '\\' => '/')
        _job_update!(job_id; status = "completed", stage = "done", message = "Run complete. Results are in $(rel_out)", extra = Dict("outputDir" => out_dir, "resultReady" => true, "written" => [String(k) for k in keys(written)]))
    catch err
        if err isa UICancelled
            _job_update!(job_id; status = "cancelled", stage = "cancelled",
                         message = err.message,
                         extra = Dict("cancelled" => true, "resultReady" => false))
        else
            message = sprint(showerror, err)
            _job_update!(job_id; status = "failed", stage = "failed", message = message, extra = Dict("error" => message))
        end
    end
    return nothing
end

function _optimizer_for_ui_run(requested_solver::AbstractString, method::AbstractString, threads::Integer; solver_log_path::AbstractString = "", rep_days::Union{Nothing,Integer} = nothing)
    solver = lowercase(String(requested_solver))
    solver = solver == "auto" ? _choose_auto_solver() : solver
    if solver == "highs"
        attrs = default_highs_attributes(; threads = Int(threads))
        _apply_highs_method!(attrs, method)
        return highs_optimizer(; attrs), attrs, "HiGHS"
    elseif solver == "gurobi"
        attrs = default_gurobi_attributes(; threads = Int(threads), rep_days = rep_days)
        _apply_gurobi_method!(attrs, method)
        isempty(solver_log_path) || (attrs["LogFile"] = solver_log_path)
        return gurobi_optimizer(; attrs), attrs, "Gurobi"
    elseif solver == "cplex"
        attrs = _cplex_attributes(method, Int(threads))
        return _optional_optimizer(:CPLEX, attrs), attrs, "CPLEX"
    elseif solver == "xpress"
        attrs = _xpress_attributes(method, Int(threads))
        return _optional_optimizer(:Xpress, attrs), attrs, "XPRESS"
    end
    error("Unknown solver: $requested_solver")
end

function _optimize_with_solver_progress!(model::JuMP.Model, job_id::String, solver_label::AbstractString, solver_log_path::AbstractString)
    if lowercase(String(solver_label)) == "gurobi" && !isempty(solver_log_path)
        stop_tail = Base.Threads.Atomic{Bool}(false)
        tail_task = Base.Threads.@spawn _tail_solver_log!(job_id, solver_log_path, stop_tail)
        try
            optimize!(model)
        finally
            stop_tail[] = true
            try
                wait(tail_task)
            catch err
                _job_update!(job_id; stage = "solve", message = "Solver log tail stopped: $(_short_error(err))")
            end
        end
    else
        optimize!(model)
    end
    return nothing
end

function _tail_solver_log!(job_id::String, solver_log_path::AbstractString, stop_tail::Base.Threads.Atomic{Bool})
    offset = 0
    while !stop_tail[]
        offset = _flush_solver_log!(job_id, solver_log_path, offset)
        sleep(0.4)
    end
    for _ in 1:3
        offset = _flush_solver_log!(job_id, solver_log_path, offset)
        sleep(0.05)
    end
    return nothing
end

function _flush_solver_log!(job_id::String, solver_log_path::AbstractString, offset::Integer)
    isfile(solver_log_path) || return offset
    current_size = filesize(solver_log_path)
    current_size < offset && (offset = 0)
    current_size == offset && return offset
    text = open(solver_log_path, "r") do io
        seek(io, offset)
        read(io, String)
    end
    new_offset = offset + ncodeunits(text)
    for raw_line in split(replace(replace(text, "\r\n" => "\n"), '\r' => '\n'), '\n')
        line = strip(raw_line)
        isempty(line) && continue
        _job_update!(job_id; stage = "solve", message = line)
    end
    return new_offset
end

function _choose_auto_solver()
    isdefined(@__MODULE__, :Gurobi) && return "gurobi"
    cplex_ok, _, _ = _try_import_solver_module(:CPLEX, "CPLEX")
    cplex_ok && return "cplex"
    xpress_ok, _, _ = _try_import_solver_module(:Xpress, "XPRESS")
    xpress_ok && return "xpress"
    return "highs"
end

function _apply_gurobi_method!(attrs::Dict{String,Any}, method::AbstractString)
    m = lowercase(String(method))
    if m == "barrier"
        attrs["Method"] = 2
        attrs["Crossover"] = 0
        attrs["BarHomogeneous"] = 1
    elseif m == "barrier_crossover"
        attrs["Method"] = 2
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    elseif m == "concurrent"
        attrs["Method"] = 3
        attrs["Crossover"] = -1
        attrs["BarHomogeneous"] = 0
    elseif m == "primal_simplex"
        attrs["Method"] = 0
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    elseif m == "dual_simplex"
        attrs["Method"] = 1
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    end
    return attrs
end

function _apply_highs_method!(attrs::Dict{String,Any}, method::AbstractString)
    m = lowercase(String(method))
    if m == "barrier"
        attrs["solver"] = "ipm"
        attrs["run_crossover"] = "off"
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "barrier_crossover"
        attrs["solver"] = "ipm"
        attrs["run_crossover"] = "on"
        attrs["primal_feasibility_tolerance"] = 1e-6
        attrs["dual_feasibility_tolerance"] = 1e-6
        attrs["ipm_optimality_tolerance"] = 1e-4
        attrs["start_crossover_tolerance"] = 1e-4
        attrs["max_dual_simplex_cleanup_level"] = 0
        attrs["max_dual_simplex_phase1_cleanup_level"] = 0
        attrs["simplex_iteration_limit"] = 0
    elseif m == "concurrent"
        attrs["solver"] = "choose"
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "primal_simplex"
        attrs["solver"] = "simplex"
        attrs["simplex_strategy"] = 4
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "dual_simplex"
        attrs["solver"] = "simplex"
        attrs["simplex_strategy"] = 1
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    end
    return attrs
end

function _cplex_attributes(method::AbstractString, threads::Int)
    attrs = Dict{String,Any}()
    threads > 0 && (attrs["CPX_PARAM_THREADS"] = threads)
    m = lowercase(String(method))
    if m == "barrier"
        attrs["CPX_PARAM_LPMETHOD"] = 4
        attrs["CPX_PARAM_BARCROSSALG"] = 0
    elseif m == "barrier_crossover"
        attrs["CPX_PARAM_LPMETHOD"] = 4
        attrs["CPX_PARAM_BARCROSSALG"] = -1
    elseif m == "concurrent"
        attrs["CPX_PARAM_LPMETHOD"] = 6
        attrs["CPX_PARAM_BARCROSSALG"] = -1
    elseif m == "primal_simplex"
        attrs["CPX_PARAM_LPMETHOD"] = 1
        attrs["CPX_PARAM_BARCROSSALG"] = -1
    elseif m == "dual_simplex"
        attrs["CPX_PARAM_LPMETHOD"] = 2
        attrs["CPX_PARAM_BARCROSSALG"] = -1
    end
    return attrs
end

function _xpress_attributes(method::AbstractString, threads::Int)
    attrs = Dict{String,Any}()
    threads > 0 && (attrs["THREADS"] = threads)
    m = lowercase(String(method))
    if m == "barrier"
        attrs["DEFAULTALG"] = 3
        attrs["CROSSOVER"] = 0
    elseif m == "barrier_crossover"
        attrs["DEFAULTALG"] = 3
        attrs["CROSSOVER"] = 1
    elseif m == "primal_simplex"
        attrs["DEFAULTALG"] = 1
    elseif m == "dual_simplex"
        attrs["DEFAULTALG"] = 2
    end
    return attrs
end

function _optional_optimizer(module_name::Symbol, attrs::AbstractDict)
    ok, msg, _ = _try_import_solver_module(module_name)
    ok || error(msg)
    optimizer = getfield(getfield(@__MODULE__, module_name), :Optimizer)
    pairs_vec = [string(k) => v for (k, v) in attrs]
    return optimizer_with_attributes(optimizer, pairs_vec...)
end

function _job_writer_progress(job_id::String)
    return function (name::Symbol, path::AbstractString, event::Symbol, seconds::Real)
        target = basename(String(path))
        table = string(name)
        if event == :start
            _job_update!(job_id; stage = "writing", message = "Saving $(table) to $(target)")
        elseif event == :finish
            size_text = isfile(path) ? " ($(_format_bytes(filesize(path))))" : ""
            _job_update!(job_id; stage = "writing", message = "Saved $(table) in $(seconds) seconds$(size_text)")
        elseif event == :failed
            _job_update!(job_id; stage = "writing", message = "Failed to save $(table) to $(target) after $(seconds) seconds")
        end
    end
end

function _write_table_with_progress!(df, path::AbstractString, name::Symbol, progress::Function)
    progress(name, _storage_display_path(path), :start, 0.0)
    started = time()
    try
        written_path = _write_table(df, path)
        elapsed = round(time() - started, digits = 3)
        progress(name, _storage_display_path(written_path), :finish, elapsed)
        return written_path
    catch err
        elapsed = round(time() - started, digits = 3)
        progress(name, _storage_display_path(path), :failed, elapsed)
        rethrow()
    end
end

function _format_bytes(bytes::Integer)
    value = Float64(bytes)
    for unit in ("B", "KB", "MB")
        value < 1024 && return unit == "B" ? "$(Int(round(value))) B" : "$(round(value, digits = 1)) $(unit)"
        value /= 1024
    end
    return "$(round(value, digits = 1)) GB"
end

# Read the dual of the per-period emission-cap constraints to compute an
# implied CO2 price (EUR / tCO2eq). Mirrors IESA-Opt 1.0's CO2_price report,
# which is the shadow price of the EU/NL emission target constraint.
# Returns an empty dict if duals are unavailable for the active solver/method.
function _extract_co2_prices(model, md::ModelData)
    prices = Dict{Int,Float64}()
    # Probe a generous set of constraint names per period. The first one that
    # actually exists with a finite shadow price wins (per IESA-Opt 1.0
    # convention this is `emTargetAir[NL,p]` when air-only mode is on).
    candidate_names = ps -> String[
        "emTargetAir[NL,$(ps)]",
        "emTargetInclScope3FuelEx[$(ps)]",
        "emTargetInclScope3[$(ps)]",
        "emTargetAll[NL,$(ps)]",
        "emTargetBunker[NL,$(ps)]",
        "emTargetFS[NL,$(ps)]",
    ]
    for ps in md.sets.periods_solve
        for name in candidate_names(ps)
            con = try
                constraint_by_name(model, name)
            catch
                nothing
            end
            con === nothing && continue
            price = try
                # `<= cap` minimization: shadow_price ≤ 0; the implied CO2 price
                # (cost of one more tCO2 of headroom) is its absolute magnitude.
                # IESA-Opt 1.0 reports EUR/tCO2; with objective in MEUR and
                # caps in MtonCO2 the ratio is already in EUR/tCO2.
                abs(shadow_price(con))
            catch
                NaN
            end
            isfinite(price) || continue
            prices[ps] = price
            break
        end
    end
    return prices
end

# Sweep every per-period emission-cap constraint registered in the model and
# extract its absolute shadow price (EUR / tCO2eq). Returns a Vector of Dicts
# suitable for `write_emission_prices_parquet`. Robust to constraints that are
# not in this build (returns an empty list when no duals are available).
function _extract_emission_prices(model, md::ModelData)
    out = Vector{Dict{String,Any}}()
    nodes = unique(vcat([:NL, :EU], md.sets.nodes))
    # (constraint base name, node-scoped?) — for each period we generate every
    # plausible registered name and capture the shadow price if it exists.
    cap_kinds = [
        ("emTargetAir",                true),
        ("emTargetBunker",             true),
        ("emTargetFS",                 true),
        ("emTargetAll",                true),
        ("emTargetInclScope3",         false),
        ("emTargetInclScope3FuelEx",   false),
        ("co2StorageCum",              true),  # cumulative, period column = 0
    ]
    for ps in md.sets.periods_solve
        for (base, per_node) in cap_kinds
            iter = per_node ? nodes : [Symbol("")]
            for n in iter
                name = per_node ? "$(base)[$(n),$(ps)]" : "$(base)[$(ps)]"
                con = try
                    constraint_by_name(model, name)
                catch
                    nothing
                end
                con === nothing && continue
                price = try
                    abs(shadow_price(con))
                catch
                    NaN
                end
                isfinite(price) || continue
                push!(out, Dict{String,Any}(
                    "name"   => base,
                    "node"   => per_node ? string(n) : "",
                    "period" => Int(ps),
                    "price"  => Float64(price),
                ))
            end
        end
    end
    return out
end

# Extract the annual activity prices = shadow prices of `balance[<a>,<p>]`,
# `balanceFix[<a>,<p>]`, and `balanceMatconv[<a>,<p>]` constraints. Returned as
# a Dict keyed by (activity::Symbol, period::Int, constraint_kind::Symbol) →
# price::Float64. The constraint_kind tag distinguishes the three balance
# families (one activity can appear in multiple). The dict can be passed
# straight to `write_activity_prices_parquet`.
function _extract_activity_prices(model, md::ModelData)
    prices = Dict{Tuple{Symbol,Int,Symbol},Float64}()
    # Activities that ever appear in `activity_balances`; capture all three
    # constraint families (most activities only show up in one of them).
    activities_seen = Set{Symbol}()
    for ((_, a, _), _) in md.params.activity_balances
        push!(activities_seen, a)
    end
    isempty(activities_seen) && (activities_seen = Set(md.sets.activities))
    constraint_families = (:balance, :balanceFix, :balanceMatconv)
    for ps in md.sets.periods_solve, a in activities_seen, kind in constraint_families
        name = "$(kind)[$(a),$(ps)]"
        con = try
            constraint_by_name(model, name)
        catch
            nothing
        end
        con === nothing && continue
        v = try
            shadow_price(con)
        catch
            NaN
        end
        isfinite(v) || continue
        prices[(a, Int(ps), kind)] = Float64(v)
    end
    return prices
end

# Hourly shadow prices for `balH_TS[<a>,<hc>,<ps>]` (TS) or `balH[<a>,<h>,<ps>]`
# (FH). Iterates over `md.sets.activities_hour` × hours × periods. Only returns
# entries whose absolute shadow price exceeds `threshold`. The mode tag is
# stored alongside each entry so downstream tools can interpret `time_index`.
function _extract_activity_prices_hourly(model, md::ModelData, mode_sym::Symbol; threshold::Float64 = 1e-6)
    out = Vector{Dict{String,Any}}()
    activities = md.sets.activities_hour
    isempty(activities) && return out
    periods = md.sets.periods_solve
    is_ts = mode_sym == :ts
    hours = is_ts ? md.sets.hours_cluster : md.sets.hours
    isempty(hours) && return out
    base = is_ts ? "balH_TS" : "balH"
    mode_str = is_ts ? "ts" : "fh"
    for ps in periods, a in activities, h in hours
        name = "$(base)[$(a),$(h),$(ps)]"
        con = try
            constraint_by_name(model, name)
        catch
            nothing
        end
        con === nothing && continue
        v = try
            shadow_price(con)
        catch
            NaN
        end
        (isfinite(v) && abs(v) > threshold) || continue
        push!(out, Dict{String,Any}(
            "activity"   => String(a),
            "period"     => Int(ps),
            "mode"       => mode_str,
            "time_index" => Int(h),
            "price"      => Float64(v),
        ))
    end
    return out
end

# Daily shadow prices for `balD_TS[<a>,<rd>,<ps>]` (TS) or `balD[<a>,<d>,<ps>]`
# (FH). Same convention as the hourly extractor.
function _extract_activity_prices_daily(model, md::ModelData, mode_sym::Symbol; threshold::Float64 = 1e-6)
    out = Vector{Dict{String,Any}}()
    activities = md.sets.activities_day
    isempty(activities) && return out
    periods = md.sets.periods_solve
    is_ts = mode_sym == :ts
    days = is_ts ? md.sets.repDays : md.sets.days
    isempty(days) && return out
    base = is_ts ? "balD_TS" : "balD"
    mode_str = is_ts ? "ts" : "fh"
    for ps in periods, a in activities, d in days
        name = "$(base)[$(a),$(d),$(ps)]"
        con = try
            constraint_by_name(model, name)
        catch
            nothing
        end
        con === nothing && continue
        v = try
            shadow_price(con)
        catch
            NaN
        end
        (isfinite(v) && abs(v) > threshold) || continue
        push!(out, Dict{String,Any}(
            "activity"   => String(a),
            "period"     => Int(ps),
            "mode"       => mode_str,
            "time_index" => Int(d),
            "price"      => Float64(v),
        ))
    end
    return out
end

function _write_ui_run_metadata(out_dir::AbstractString, config::Dict{String,Any}, stage_times::Dict{String,Float64}, rr::RunResult, effective_solver::String, attrs::AbstractDict; progress::Function = _no_writer_progress)
    solver_version = _solver_version_for_label(effective_solver)
    timing = DataFrames.DataFrame(
        engine = ["Julia"],
        scenario = [config["scenario"]],
        inputWorkbook = [config["inputWorkbook"]],
        mode = [String(rr.mode)],
        periods = [join(string.(config["periods"]), ",")],
        solver = [effective_solver],
        solverVersion = [solver_version],
        solveMethod = [config["solveMethod"]],
        n_repDays = [rr.n_repDays],
        hoursPer_day = [rr.hoursPer_day],
        queue_sec = [get(stage_times, "queue_sec", 0.0)],
        dataRead_sec = [get(stage_times, "data_read_sec", 0.0)],
        derive_sec = [get(stage_times, "derive_sec", 0.0)],
        cluster_sec = [get(stage_times, "cluster_sec", 0.0)],
        optimizerInit_sec = [get(stage_times, "optimizer_init_sec", 0.0)],
        generation_sec = [get(stage_times, "generation_sec", 0.0)],
        solve_sec = [get(stage_times, "solve_sec", 0.0)],
        dualExtract_sec = [get(stage_times, "dual_extract_sec", 0.0)],
        resultsWrite_sec = [get(stage_times, "results_write_sec", 0.0)],
        total_sec = [get(stage_times, "total_sec", rr.total_seconds)],
        n_rows = [rr.n_rows],
        n_cols = [rr.n_cols],
        objective = [rr.objective_value],
        termination_status = [rr.termination_status],
    )
    # The UI no longer renders the wide timing_summary table; instead the same
    # information is folded into the long-form `solver_settings` (attribute,
    # value) table next to the stacked-bar chart. We still persist the wide
    # table for tooling that already consumes it.
    db_path = joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
    _with_duckdb_write_connection(db_path) do
        _write_table_with_progress!(timing, _duckdb_table_uri(db_path, "timing_summary"), :timing_summary, progress)

        settings = DataFrames.DataFrame(attribute = String[], value = String[])
        # Run identity / configuration
        push!(settings, ("engine", "Julia"))
        push!(settings, ("scenario", String(config["scenario"])))
        push!(settings, ("inputWorkbook", String(config["inputWorkbook"])))
        push!(settings, ("mode", String(rr.mode)))
        push!(settings, ("periods", join(string.(config["periods"]), ",")))
        push!(settings, ("solver", effective_solver))
        push!(settings, ("solver_version", solver_version))
        push!(settings, ("requested_solver", String(config["solver"])))
        push!(settings, ("solve_method", String(config["solveMethod"])))
        push!(settings, ("n_repDays", string(rr.n_repDays)))
        push!(settings, ("hoursPer_day", string(rr.hoursPer_day)))
        # Sizing
        push!(settings, ("n_rows", string(rr.n_rows)))
        push!(settings, ("n_cols", string(rr.n_cols)))
        # Outcome
        push!(settings, ("objective", string(rr.objective_value)))
        push!(settings, ("termination_status", string(rr.termination_status)))
        # Stage timings (seconds). Same numbers shown in the stacked bar.
        for (label, key) in [
                ("queue_sec",         "queue_sec"),
                ("dataRead_sec",      "data_read_sec"),
                ("derive_sec",        "derive_sec"),
                ("cluster_sec",       "cluster_sec"),
                ("optimizerInit_sec", "optimizer_init_sec"),
                ("generation_sec",    "generation_sec"),
                ("solve_sec",         "solve_sec"),
                ("dualExtract_sec",   "dual_extract_sec"),
                ("resultsWrite_sec",  "results_write_sec"),
                ("total_sec",         "total_sec"),
            ]
            v = get(stage_times, key, key == "total_sec" ? rr.total_seconds : 0.0)
            push!(settings, (label, string(round(Float64(v); digits = 3))))
        end
        # Solver attributes (sorted), prefixed so they don't collide with the
        # canonical UI keys above.
        for (key, value) in sort(collect(attrs); by = first)
            push!(settings, ("attr:" * string(key), string(value)))
        end
        _write_table_with_progress!(settings, _duckdb_table_uri(db_path, "solver_settings"), :solver_settings, progress)
    end
    return nothing
end

function _solver_version_for_label(label::AbstractString)
    lower_label = lowercase(String(label))
    lower_label == "highs" && return _solver_display_version(:HiGHS)
    lower_label == "gurobi" && return _solver_display_version(:Gurobi)
    lower_label == "cplex" && return _solver_display_version(:CPLEX)
    lower_label == "xpress" && return _solver_display_version(:Xpress)
    return ""
end

function _job_results(job_id::String)
    job = _job_snapshot(job_id)
    out_dir = String(get(job, "outputDir", ""))
    isempty(out_dir) && error("No output directory is available for job $job_id")
    isdir(out_dir) || error("Output directory does not exist: $out_dir")
    results = _read_ui_results(out_dir)
    results["job"] = job
    return results
end

function _output_roots()
    root = _repo_root()
    return [joinpath(root, "Output")]
end

function _is_child_path(path::AbstractString, root::AbstractString)
    rel = relpath(normpath(path), normpath(root))
    parts = splitpath(rel)
    return rel != "." && !isempty(parts) && first(parts) != ".." && !isabspath(rel)
end

function _resolve_output_dir(output_id::AbstractString)
    cleaned = replace(strip(String(output_id)), '\\' => '/')
    isempty(cleaned) && error("No output folder was selected")
    parts = [part for part in split(cleaned, '/') if !isempty(part)]
    candidate = isabspath(cleaned) ? normpath(cleaned) : normpath(joinpath(_repo_root(), parts...))
    any(root -> _is_child_path(candidate, root), _output_roots()) || error("Output folder is outside Output/: $output_id")
    isdir(candidate) || error("Output folder does not exist: $output_id")
    return candidate
end

function _has_direct_result_files(dir::AbstractString)
    isdir(dir) || return false
    isfile(joinpath(dir, IESA_RESULTS_DUCKDB_FILE)) && return true
    return any(name -> isfile(joinpath(dir, name)) && lowercase(splitext(name)[2]) == ".parquet", readdir(dir))
end

function _list_output_runs()
    runs = Vector{Dict{String,Any}}()
    for root in _output_roots()
        isdir(root) || continue
        local names
        try
            names = sort(readdir(root))
        catch
            continue
        end
        for name in names
            dir = joinpath(root, name)
            try
                isdir(dir) || continue
                _has_direct_result_files(dir) || continue
                push!(runs, _output_run_summary(dir))
            catch
                # Skip folders that can't currently be summarized (e.g., temporarily
                # locked files on Windows after a recent close). They will reappear on
                # the next listing once the OS releases the handles.
                continue
            end
        end
    end
    sort!(runs; by = run -> get(run, "modifiedAt", ""), rev = true)
    return runs
end

function _output_run_summary(out_dir::AbstractString)
    rel = replace(relpath(out_dir, _repo_root()), '\\' => '/')
    files = _result_file_names(out_dir)
    timing = _first_result_row(out_dir, "timing_summary")
    stats = _first_result_row(out_dir, "run_statistics")
    total = _first_result_row(out_dir, "totalCosts")
    modified = Dates.unix2datetime(stat(out_dir).mtime)
    return Dict{String,Any}(
        "id" => rel,
        "name" => basename(out_dir),
        "path" => rel,
        "storage" => isfile(_result_db_path(out_dir)) ? "DuckDB" : "Parquet",
        "modifiedAt" => string(modified),
        "fileCount" => length(files),
        "files" => files,
        "scenario" => string(get(timing, "scenario", basename(out_dir))),
        "mode" => string(get(timing, "mode", "")),
        "periods" => string(get(timing, "periods", "")),
        "solver" => string(get(timing, "solver", "")),
        "solverVersion" => string(get(timing, "solverVersion", "")),
        "status" => string(get(stats, "termination_status", get(timing, "termination_status", ""))),
        "objective" => get(total, "value", get(stats, "objective", get(timing, "objective", nothing))),
        "totalSeconds" => get(timing, "total_sec", get(stats, "total_seconds", nothing)),
        "solveSeconds" => get(timing, "solve_sec", get(stats, "solve_seconds", nothing)),
    )
end

function _first_row(path::AbstractString)
    rows = _read_table_rows(path, 1)
    return isempty(rows) ? Dict{String,Any}() : first(rows)
end

function _first_result_row(out_dir::AbstractString, table_name::AbstractString)
    rows = _read_result_table_rows(out_dir, table_name, 1)
    return isempty(rows) ? Dict{String,Any}() : first(rows)
end

function _delete_output_runs!(body)
    ids = _as_string_vector(_config_get(body, "outputDirs", String[]))
    if isempty(ids)
        ids = _as_string_vector(_config_get(body, "outputDir", ""))
    end
    isempty(ids) && error("Select at least one output folder to delete")
    deleted = String[]
    failed = Vector{Dict{String,String}}()
    for id in ids
        out_dir = _resolve_output_dir(id)
        if _active_output_dir(out_dir)
            push!(failed, Dict("id" => id, "error" => "Cannot delete $(basename(out_dir)) because a run is still using that output folder"))
            continue
        end
        try
            _remove_output_dir!(out_dir)
            push!(deleted, replace(relpath(out_dir, _repo_root()), '\\' => '/'))
        catch err
            push!(failed, Dict("id" => id, "error" => _short_error(err)))
        end
    end
    return Dict("deleted" => deleted, "failed" => failed, "outputs" => _list_output_runs())
end

function _active_output_dir(out_dir::AbstractString)
    target = _canonical_path(out_dir)
    lock(UI_JOBS_LOCK)
    try
        for job in values(UI_JOBS)
            status = String(get(job, "status", ""))
            status in ("queued", "running") || continue
            output_dir = String(get(job, "outputDir", ""))
            !isempty(output_dir) && _canonical_path(output_dir) == target && return true
            config = get(job, "config", Dict{String,Any}())
            config isa AbstractDict || continue
            configured_dir = String(get(config, "outputDir", ""))
            !isempty(configured_dir) && _canonical_path(configured_dir) == target && return true
        end
    finally
        unlock(UI_JOBS_LOCK)
    end
    return false
end

function _canonical_path(path::AbstractString)
    canonical = normpath(abspath(path))
    return Sys.iswindows() ? lowercase(canonical) : canonical
end

function _remove_output_dir!(out_dir::AbstractString)
    isdir(out_dir) || return nothing
    db_path = joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
    try
        _close_duckdb_write_connection!(db_path)
    catch
    end
    # Force any lingering DuckDB finalizers to run so Windows releases file handles
    # that were opened by read-only views earlier in this session.
    for _ in 1:3
        GC.gc(true)
    end
    Sys.iswindows() && sleep(0.1)

    last_error = nothing
    for attempt in 1:10
        try
            _make_tree_writable!(out_dir)
        catch err
            last_error = err
        end
        # Delete each file individually so we surface (and retry) per-file lock errors
        # instead of letting recursive rm silently swallow them.
        try
            _delete_files_in_tree!(out_dir, last_error)
        catch err
            last_error = err
        end
        try
            rm(out_dir; recursive = true, force = true)
        catch err
            last_error = err
        end
        isdir(out_dir) || return nothing
        if Sys.iswindows() && attempt >= 2
            try
                _windows_force_remove_dir!(out_dir)
            catch err
                last_error = err
            end
            isdir(out_dir) || return nothing
        end
        for _ in 1:2
            GC.gc(true)
        end
        attempt < 10 && sleep(0.3 * attempt)
    end
    error("Could not delete $(replace(relpath(out_dir, _repo_root()), '\\' => '/')). Close any program that may be viewing files in this folder and try again. Last error: $(_short_error(last_error))")
end

function _delete_files_in_tree!(out_dir::AbstractString, last_error)
    isdir(out_dir) || return last_error
    for (root, _dirs, files) in walkdir(out_dir; topdown = false)
        for file in files
            path = joinpath(root, file)
            try
                rm(path; force = true)
            catch err
                last_error = err
            end
        end
    end
    return last_error
end

function _windows_force_remove_dir!(out_dir::AbstractString)
    target = abspath(out_dir)
    isdir(target) || return nothing
    # cmd /c rd /s /q "<path>" -- recursive, quiet, no confirmation
    cmd = Cmd(`cmd /c rd /s /q $target`; windows_verbatim = true)
    try
        run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = true)
    catch
    end
    return nothing
end

function _make_tree_writable!(dir::AbstractString)
    isdir(dir) || return nothing
    for (root, dirs, files) in walkdir(dir; topdown = false)
        for file in files
            path = joinpath(root, file)
            try
                chmod(path, 0o666)
            catch
            end
        end
        for child in dirs
            path = joinpath(root, child)
            try
                chmod(path, 0o777)
            catch
            end
        end
    end
    try
        chmod(dir, 0o777)
    catch
    end
    return nothing
end

function _compare_output_runs(output_ids::Vector{String})
    isempty(output_ids) && error("Select at least one output folder to compare")
    runs = Vector{Dict{String,Any}}()
    total_costs = Vector{Dict{String,Any}}()
    cost_components = Vector{Dict{String,Any}}()
    timing_rows = Vector{Dict{String,Any}}()
    for id in output_ids
        out_dir = _resolve_output_dir(id)
        summary = _output_run_summary(out_dir)
        push!(runs, summary)
        output_name = String(summary["name"])
        output_id = String(summary["id"])

        for row in _read_result_table_rows(out_dir, "totalCosts", 100)
            item = copy(row)
            item["output"] = output_name
            item["outputId"] = output_id
            push!(total_costs, item)
        end
        for row in _cost_by_component(out_dir)
            item = copy(row)
            item["output"] = output_name
            item["outputId"] = output_id
            push!(cost_components, item)
        end
        timing = _first_result_row(out_dir, "timing_summary")
        if !isempty(timing)
            timing["output"] = output_name
            timing["outputId"] = output_id
            push!(timing_rows, timing)
        else
            push!(timing_rows, Dict("output" => output_name, "outputId" => output_id))
        end
    end
    return Dict(
        "runs" => runs,
        "totalCosts" => total_costs,
        "costByComponent" => cost_components,
        "timing" => timing_rows,
    )
end

function _read_ui_results(out_dir::AbstractString)
    files = _result_file_names(out_dir)
    nodes_df = _read_result_df(out_dir, "nodes_meta")
    nodes_list = isempty(nodes_df) ? String[] : String.(nodes_df.node)
    result = Dict{String,Any}(
        "outputDir" => replace(relpath(out_dir, _repo_root()), '\\' => '/'),
        "storage" => isfile(_result_db_path(out_dir)) ? "DuckDB" : "Parquet",
        "files" => files,
        "runStatistics" => _read_result_table_rows(out_dir, "run_statistics", 20),
        "timingSummary" => _read_result_table_rows(out_dir, "timing_summary", 20),
        "solverSettings" => _read_result_table_rows(out_dir, "solver_settings", 200),
        "totalCosts" => _read_result_table_rows(out_dir, "totalCosts", 50),
        "costByComponent" => _cost_by_component(out_dir),
        "costByTechnology" => _cost_by_technology(out_dir),
        "co2Price" => _read_result_table_rows(out_dir, "CO2_price", 50),
        "emissionPrices" => _read_result_table_rows(out_dir, "emission_prices", 500),
        "activityPrices" => _read_result_table_rows(out_dir, "activity_prices", 5000),
        "activityPricesHourly" => _read_result_table_rows(out_dir, "activity_prices_hourly", 200_000),
        "activityPricesDaily"  => _read_result_table_rows(out_dir, "activity_prices_daily", 50_000),
        "nodes" => nodes_list,
        "powerCapacities" => _power_capacities(out_dir),
        "hourlyDispatch" => _hourly_dispatch_payload(out_dir),
        "flexibility" => _flexibility_payload(out_dir),
        "hourlyProfiles" => _hourly_profile_preview(out_dir),
        "balanceActivities" => _balance_activity_options(out_dir),
        "emissionGroupings" => _emission_grouping_options(out_dir),
    )
    return result
end

function _result_db_path(out_dir::AbstractString)
    return joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
end

function _result_file_names(out_dir::AbstractString)
    if isfile(_result_db_path(out_dir))
        return [IESA_RESULTS_DUCKDB_FILE]
    end
    return sort([name for name in readdir(out_dir) if isfile(joinpath(out_dir, name)) && lowercase(splitext(name)[2]) == ".parquet"])
end

function _read_result_df(out_dir::AbstractString, table_name::AbstractString)
    db_path = _result_db_path(out_dir)
    if isfile(db_path)
        df = _read_duckdb_table_df(db_path, table_name)
        isempty(df) || return df
    end
    return _read_parquet_df(joinpath(out_dir, table_name * ".parquet"))
end

function _read_result_table_rows(out_dir::AbstractString, table_name::AbstractString, limit::Int)
    return _df_rows(_read_result_df(out_dir, table_name), limit)
end

function _read_duckdb_table_df(db_path::AbstractString, table_name::AbstractString)
    query = "SELECT * FROM $(_duckdb_quote_identifier(table_name))"
    active_con = _active_duckdb_write_connection(db_path)
    if active_con !== nothing
        try
            return _duckdb_query_df(active_con, query)
        catch
            return DataFrames.DataFrame()
        end
    end

    con = nothing
    try
        con = _duckdb_connect(db_path; readonly = true)
        return _duckdb_query_df(con, query)
    catch
        return DataFrames.DataFrame()
    finally
        con !== nothing && DBInterface.close!(con)
        GC.gc()
    end
end

function _read_parquet_df(path::AbstractString)
    isfile(path) || return DataFrames.DataFrame()
    return DataFrames.DataFrame(Parquet2.Dataset(path))
end

function _read_table_rows(path::AbstractString, limit::Int)
    df = _read_parquet_df(path)
    return _df_rows(df, limit)
end

function _df_rows(df::DataFrames.DataFrame, limit::Int = 100)
    isempty(df) && return Vector{Dict{String,Any}}()
    rows = Vector{Dict{String,Any}}()
    max_rows = min(limit, nrow(df))
    for row in eachrow(first(df, max_rows))
        item = Dict{String,Any}()
        for name in names(df)
            value = row[name]
            item[name] = _json_value(value)
        end
        push!(rows, item)
    end
    return rows
end

function _json_value(value)
    value === missing && return nothing
    value isa Symbol && return String(value)
    value isa DateTime && return string(value)
    value isa AbstractFloat && !isfinite(value) && return string(value)
    return value
end

function _cost_by_component(out_dir::AbstractString)
    df = _read_result_df(out_dir, "cost_breakdown")
    isempty(df) && return Vector{Dict{String,Any}}()
    all(col in names(df) for col in ["component", "cost_MEUR"]) || return _df_rows(df, 40)
    grouped = DataFrames.combine(DataFrames.groupby(df, :component), :cost_MEUR => sum => :cost_MEUR)
    grouped.abs_cost = abs.(grouped.cost_MEUR)
    sort!(grouped, :abs_cost; rev = true)
    select!(grouped, Not(:abs_cost))
    return _df_rows(grouped, 40)
end

function _cost_by_technology(out_dir::AbstractString)
    df = _read_result_df(out_dir, "cost_breakdown")
    isempty(df) && return Vector{Dict{String,Any}}()
    all(col in names(df) for col in ["tech", "cost_MEUR"]) || return Vector{Dict{String,Any}}()
    grouped = DataFrames.combine(DataFrames.groupby(df, :tech), :cost_MEUR => sum => :cost_MEUR)
    grouped.abs_cost = abs.(grouped.cost_MEUR)
    sort!(grouped, :abs_cost; rev = true)
    select!(grouped, Not(:abs_cost))
    return _df_rows(grouped, 25)
end

function _hourly_profile_preview(out_dir::AbstractString)
    df = _read_result_df(out_dir, "tech_use_TS")
    isempty(df) && (df = _read_result_df(out_dir, "tech_use_h"))
    isempty(df) && return Dict("timeColumn" => "", "rows" => Vector{Dict{String,Any}}(), "topTechnologies" => String[])
    time_col = "hc" in names(df) ? :hc : :hour
    all(col in names(df) for col in [String(time_col), "tech", "value"]) || return Dict("timeColumn" => String(time_col), "rows" => _df_rows(df, 300), "topTechnologies" => String[])

    tech_totals = DataFrames.combine(DataFrames.groupby(df, :tech), :value => (x -> sum(abs, x)) => :activity)
    sort!(tech_totals, :activity; rev = true)
    top_tech = String.(tech_totals.tech[1:min(6, nrow(tech_totals))])
    grouped = DataFrames.combine(DataFrames.groupby(df, [time_col, :tech]), :value => sum => :value)
    filtered = grouped[in.(String.(grouped.tech), Ref(top_tech)), :]
    sort!(filtered, [time_col, :tech])
    return Dict("timeColumn" => String(time_col), "rows" => _df_rows(filtered, 900), "topTechnologies" => top_tech)
end

# -------------------------------------------------------------------
# Power capacities, hourly dispatch payload, supply/demand, emissions
# -------------------------------------------------------------------

_is_power_tech_row(row) = begin
    fields = (lowercase(string(get(row, "sector", ""))),
              lowercase(string(get(row, "subsector", ""))),
              lowercase(string(get(row, "category", ""))),
              lowercase(string(get(row, "label", ""))))
    any(f -> occursin("power", f) || occursin("electricity", f), fields)
end

# Best-effort coercion to Int. Tolerates Int/Float/String columns coming back
# from Parquet or DuckDB; returns nothing for missing/empty/unparseable.
function _to_int_safe(x)::Union{Nothing,Int}
    x === missing && return nothing
    x === nothing && return nothing
    x isa Integer && return Int(x)
    if x isa AbstractFloat
        isfinite(x) || return nothing
        return Int(round(x))
    end
    if x isa AbstractString
        s = strip(String(x))
        isempty(s) && return nothing
        v = tryparse(Int, s)
        v !== nothing && return v
        f = tryparse(Float64, s)
        return f === nothing ? nothing : Int(round(f))
    end
    return nothing
end

function _to_float_safe(x)::Union{Nothing,Float64}
    x === missing && return nothing
    x === nothing && return nothing
    x isa Real && return isfinite(Float64(x)) ? Float64(x) : nothing
    if x isa AbstractString
        s = strip(String(x))
        isempty(s) && return nothing
        v = tryparse(Float64, s)
        return v === nothing || !isfinite(v) ? nothing : v
    end
    return nothing
end

function _xc_cluster_id(text)::Union{Nothing,Int}
    value = strip(String(text))
    m = match(r"\bCL\s*(\d+)\b"i, value)
    m === nothing && (m = match(r"^(\d+)$", value))
    m === nothing && return nothing
    return tryparse(Int, m.captures[1])
end

function _xc_endpoint_id(text)::Union{Nothing,String}
    value = String(text)
    m = match(r"\bCL\s*(\d+)\b"i, value)
    m !== nothing && return "CL$(m.captures[1])"
    m = match(r"\bNS\s*(\d+)\b"i, value)
    m !== nothing && return "NS$(m.captures[1])"
    (occursin(r"\b(Denmark|Danish)\b"i, value) || occursin(r"\bDK\b", value)) && return "DK"
    (occursin(r"\b(United Kingdom|Great Britain|Britain)\b"i, value) || occursin(r"\bUK\b", value) || occursin(r"\bGB\b", value)) && return "UK"
    (occursin(r"\b(Norway|Norwey|Norwegian)\b"i, value) || occursin(r"\bNO\b", value)) && return "NO"
    (occursin(r"\b(Germany|German)\b"i, value) || occursin(r"\bDE\b", value)) && return "DE"
    (occursin(r"\b(Belgium|Belgian)\b"i, value) || occursin(r"\bBE\b", value)) && return "BE"
    occursin(r"\bEU\b"i, value) && return "DE"
    return nothing
end

function _xc_endpoint_label(id::AbstractString)
    s = String(id)
    s == "CL1" && return "Noordzeekanaalgebied"
    s == "CL2" && return "Noord-Nederland"
    s == "CL3" && return "Chemelot"
    s == "CL4" && return "Zeeland/West Brabant"
    s == "CL5" && return "Rotterdam-Moerdijk"
    startswith(s, "CL") && return s
    startswith(s, "NS") && return s
    s in ("DK", "UK", "NO", "DE", "BE") && return s
    return s
end

function _xc_endpoint_cluster(id::AbstractString)::Union{Nothing,Int}
    m = match(r"^CL(\d+)$", String(id))
    m === nothing && return nothing
    return tryparse(Int, m.captures[1])
end

function _xc_trade_commodity(row::AbstractDict)
    text = lowercase(join((
        string(get(row, "tech", "")),
        string(get(row, "name", "")),
        string(get(row, "sector", "")),
        string(get(row, "subsector", "")),
        string(get(row, "activity", "")),
        string(get(row, "label", "")),
    ), " "))
    if occursin("ccus", text) || occursin("_ccs_", text) || occursin(" ccs ", text) || occursin("- ccs", text)
        return "ccus"
    elseif occursin("hydrogen", text) || occursin("_hyd", text) || occursin(" hyd", text)
        return "hydrogen"
    elseif occursin("natural gas", text) || occursin(" gas ", text) || occursin("gas pool", text)
        return "natural_gas"
    elseif occursin("electricity", text) || occursin("power", text) || occursin("peu", text) || occursin("pnl", text)
        return "electricity"
    end
    return "other"
end

function _regional_map_asset_paths(node_level::Int)
    level = max(1, node_level)
    assets = Dict{String,Any}(
        "nodeLevel" => level,
        "clusters" => "/assets/maps/clusters_$(level).geojson",
        "centroids" => "/assets/maps/centroids_$(level).geojson",
        "northSeaHubs" => "/assets/maps/north_sea_hubs.geojson",
    )
    assets["available"] = all(path -> isfile(joinpath(_ui_dir(), split(strip(path, ['/']), '/')...)), String[assets["clusters"], assets["centroids"]])
    return assets
end

function _regional_map_inferred_node_level(out_dir::AbstractString, link_by_tech::Dict{String,Dict{String,Any}})
    cluster_ids = Set{Int}()
    path_hint = match(r"(\d+)\s*node"i, replace(String(out_dir), ['_', '-'] => " "))
    path_hint === nothing || push!(cluster_ids, parse(Int, path_hint.captures[1]))
    nodes_df = _read_result_df(out_dir, "nodes_meta")
    if !isempty(nodes_df) && "node" in names(nodes_df)
        for value in nodes_df.node
            cluster = _xc_cluster_id(string(value))
            cluster === nothing || push!(cluster_ids, cluster)
        end
    end
    for link in values(link_by_tech)
        src_cluster = _xc_endpoint_cluster(string(link["source"]))
        dst_cluster = _xc_endpoint_cluster(string(link["target"]))
        src_cluster === nothing || push!(cluster_ids, src_cluster)
        dst_cluster === nothing || push!(cluster_ids, dst_cluster)
    end
    return isempty(cluster_ids) ? 15 : maximum(cluster_ids)
end

function _xc_trade_links(out_dir::AbstractString)
    meta_df = _read_result_df(out_dir, "tech_meta")
    isempty(meta_df) && return Dict{String,Dict{String,Any}}()
    rows = _df_rows(meta_df, nrow(meta_df))
    links = Dict{String,Dict{String,Any}}()
    for row in rows
        string(get(row, "category", "")) == "XC Trade" || continue
        src = _xc_endpoint_id(get(row, "subsector", ""))
        dst = _xc_endpoint_id(get(row, "sector", ""))
        src === nothing && continue
        dst === nothing && continue
        src == dst && continue
        tech = string(get(row, "tech", ""))
        isempty(tech) && continue
        commodity = _xc_trade_commodity(row)
        links[tech] = Dict{String,Any}(
            "tech" => tech,
            "name" => string(get(row, "name", "")),
            "source" => src,
            "target" => dst,
            "commodity" => commodity,
            "sector" => string(get(row, "sector", "")),
            "subsector" => string(get(row, "subsector", "")),
        )
    end
    return links
end

function _commodity_label(id::AbstractString)
    id == "electricity" && return "Electricity"
    id == "natural_gas" && return "Natural gas"
    id == "hydrogen" && return "Hydrogen"
    id == "ccus" && return "CCUS"
    id == "other" && return "Other"
    id == "all" && return "All"
    return String(id)
end

function _regional_map_periods(out_dir::AbstractString)
    periods = Set{Int}()
    for table in ("techStock", "tech_use_TS", "tech_use_h", "tech_use")
        df = _read_result_df(out_dir, table)
        isempty(df) && continue
        "period" in names(df) || continue
        for value in df.period
            p = _to_int_safe(value)
            p === nothing || push!(periods, p)
        end
    end
    return sort!(collect(periods))
end

function _regional_map_input_workbook(out_dir::AbstractString)
    settings = _read_result_df(out_dir, "solver_settings")
    if !isempty(settings) && all(name in names(settings) for name in ("attribute", "value"))
        for row in _df_rows(settings, nrow(settings))
            string(get(row, "attribute", "")) == "inputWorkbook" || continue
            workbook = strip(string(get(row, "value", "")))
            isempty(workbook) || return workbook
        end
    end
    return "Input/1108 SSP.xlsx"
end

function _regional_map_unit_info(out_dir::AbstractString, link_by_tech::Dict{String,Dict{String,Any}}, metric_id::AbstractString)
    kind = metric_id == "use" ? "UoA" : "UoC"
    fallback = Dict{String,Any}("kind" => kind, "unit" => "", "label" => "")
    isempty(link_by_tech) && return fallback
    try
        input_path, _ = _resolve_input_workbook(_regional_map_input_workbook(out_dir); require_exists = true)
        md = _read_ui_data_cached(input_path)
        counts = Dict{String,Float64}()
        for tech in keys(link_by_tech)
            t = Symbol(tech)
            unit_sym = if metric_id == "use"
                act = get(md.params.activityPer_tech, t, get(md.params.activityPer_techOrig, t, Symbol("")))
                get(md.params.act_units, act, Symbol(kind))
            else
                get(md.params.tech_units, t, Symbol(kind))
            end
            unit = strip(string(unit_sym))
            (isempty(unit) || unit == kind) && continue
            counts[unit] = get(counts, unit, 0.0) + 1.0
        end
        isempty(counts) && return fallback
        unit = first(sort!(collect(keys(counts)); by = u -> (-counts[u], u)))
        return Dict{String,Any}("kind" => kind, "unit" => unit, "label" => unit)
    catch err
        @debug "Regional map unit lookup failed" err
        return fallback
    end
end

function _regional_map_payload(out_dir::AbstractString; metric::AbstractString = "stock", commodity::AbstractString = "all", period::Union{Nothing,Integer} = nothing)
    link_by_tech = _xc_trade_links(out_dir)
    periods = _regional_map_periods(out_dir)
    sel_period = period === nothing ? (isempty(periods) ? 0 : periods[end]) : Int(period)
    sel_period in periods || (sel_period = isempty(periods) ? 0 : periods[end])
    metric_id = lowercase(strip(String(metric))) in ("use", "techuse", "flow", "flows") ? "use" : "stock"
    commodity_id = lowercase(strip(String(commodity)))
    commodity_id = commodity_id in ("all", "electricity", "natural_gas", "hydrogen", "ccus", "other") ? commodity_id : "all"

    level = _regional_map_inferred_node_level(out_dir, link_by_tech)

    all_commodities = sort!(collect(Set{String}(string(link["commodity"]) for link in values(link_by_tech))))
    commodity_options = [Dict("id" => "all", "label" => "All")]
    for id in ("electricity", "natural_gas", "hydrogen", "ccus", "other")
        id in all_commodities && push!(commodity_options, Dict("id" => id, "label" => _commodity_label(id)))
    end

    unit_info = _regional_map_unit_info(out_dir, link_by_tech, metric_id)
    links = metric_id == "use" ?
        _regional_map_use_links(out_dir, link_by_tech, sel_period, commodity_id, unit_info) :
        _regional_map_stock_links(out_dir, link_by_tech, sel_period, commodity_id, unit_info)

    return Dict{String,Any}(
        "available" => !isempty(link_by_tech),
        "metric" => metric_id,
        "commodity" => commodity_id,
        "periods" => periods,
        "selectedPeriod" => sel_period,
        "nodeLevel" => level,
        "assets" => _regional_map_asset_paths(level),
        "commodityOptions" => commodity_options,
        "unit" => unit_info,
        "links" => links["links"],
        "frames" => get(links, "frames", Vector{Dict{String,Any}}()),
        "stats" => Dict(
            "xcTechs" => length(link_by_tech),
            "shownLinks" => length(links["links"]),
            "commodities" => all_commodities,
        ),
    )
end

function _regional_map_stock_links(out_dir::AbstractString, link_by_tech::Dict{String,Dict{String,Any}}, period::Int, commodity::String, unit_info::Dict{String,Any})
    stock_df = _read_result_df(out_dir, "techStock")
    isempty(stock_df) && return Dict("links" => Vector{Dict{String,Any}}(), "unit" => unit_info)
    directed = Dict{Tuple{String,String,String},Float64}()
    for row in _df_rows(stock_df, nrow(stock_df))
        p = _to_int_safe(get(row, "period", nothing))
        p === period || continue
        tech = string(get(row, "tech", ""))
        link = get(link_by_tech, tech, nothing)
        link === nothing && continue
        comm = string(link["commodity"])
        commodity == "all" || comm == commodity || continue
        v = _to_float_safe(get(row, "value", nothing))
        v === nothing && continue
        abs(v) < 1e-9 && continue
        key = (string(link["source"]), string(link["target"]), comm)
        directed[key] = get(directed, key, 0.0) + v
    end

    pair_values = Dict{Tuple{String,String,String},Tuple{Float64,Float64}}()
    for ((src, dst, comm), v) in directed
        a, b = src <= dst ? (src, dst) : (dst, src)
        prev = get(pair_values, (a, b, comm), (0.0, 0.0))
        pair_values[(a, b, comm)] = src == a ? (max(prev[1], abs(v)), prev[2]) : (prev[1], max(prev[2], abs(v)))
    end
    rows = Vector{Dict{String,Any}}()
    for ((a, b, comm), (ab, ba)) in pair_values
        value = max(ab, ba)
        value > 1e-9 || continue
        push!(rows, Dict{String,Any}(
            "source" => a,
            "target" => b,
            "sourceLabel" => _xc_endpoint_label(a),
            "targetLabel" => _xc_endpoint_label(b),
            "commodity" => comm,
            "commodityLabel" => _commodity_label(comm),
            "value" => value,
            "unit" => unit_info,
            "reverseValue" => ba,
            "metric" => "stock",
            "directional" => false,
        ))
    end
    sort!(rows; by = row -> -Float64(row["value"]))
    return Dict("links" => rows, "unit" => unit_info)
end

function _regional_map_use_links(out_dir::AbstractString, link_by_tech::Dict{String,Dict{String,Any}}, period::Int, commodity::String, unit_info::Dict{String,Any})
    directed = Dict{Tuple{String,String,String},Float64}()
    by_time = Dict{Tuple{Int,String,String,String},Float64}()
    for (table, time_col) in (("tech_use_TS", "hc"), ("tech_use_h", "hour"), ("tech_use", ""))
        use_df = _read_result_df(out_dir, table)
        isempty(use_df) && continue
        has_time = !isempty(time_col) && time_col in names(use_df)
        for row in _df_rows(use_df, nrow(use_df))
            p = _to_int_safe(get(row, "period", nothing))
            p === period || continue
            tech = string(get(row, "tech", ""))
            link = get(link_by_tech, tech, nothing)
            link === nothing && continue
            comm = string(link["commodity"])
            commodity == "all" || comm == commodity || continue
            v = _to_float_safe(get(row, "value", nothing))
            v === nothing && continue
            abs(v) < 1e-12 && continue
            src, dst = string(link["source"]), string(link["target"])
            key = (src, dst, comm)
            directed[key] = get(directed, key, 0.0) + v
            if has_time
                t = _to_int_safe(get(row, time_col, nothing))
                t === nothing || (by_time[(t, src, dst, comm)] = get(by_time, (t, src, dst, comm), 0.0) + v)
            end
        end
    end
    links = _regional_map_net_links(directed, "use", unit_info)
    frames = _regional_map_net_frames(by_time, unit_info)
    return Dict("links" => links, "frames" => frames, "unit" => unit_info)
end

function _regional_map_net_links(directed::Dict{Tuple{String,String,String},Float64}, metric::AbstractString, unit_info::Dict{String,Any})
    pair_keys = Set{Tuple{String,String,String}}()
    for (src, dst, comm) in keys(directed)
        a, b = src <= dst ? (src, dst) : (dst, src)
        push!(pair_keys, (a, b, comm))
    end
    rows = Vector{Dict{String,Any}}()
    for (a, b, comm) in pair_keys
        ab = get(directed, (a, b, comm), 0.0)
        ba = get(directed, (b, a, comm), 0.0)
        net = ab - ba
        abs(net) > 1e-9 || continue
        src, dst = net >= 0 ? (a, b) : (b, a)
        push!(rows, Dict{String,Any}(
            "source" => src,
            "target" => dst,
            "sourceLabel" => _xc_endpoint_label(src),
            "targetLabel" => _xc_endpoint_label(dst),
            "commodity" => comm,
            "commodityLabel" => _commodity_label(comm),
            "value" => abs(net),
            "unit" => unit_info,
            "forwardValue" => ab,
            "reverseValue" => ba,
            "signedNet" => net,
            "metric" => String(metric),
            "directional" => true,
        ))
    end
    sort!(rows; by = row -> -Float64(row["value"]))
    return rows
end

function _regional_map_net_frames(by_time::Dict{Tuple{Int,String,String,String},Float64}, unit_info::Dict{String,Any})
    isempty(by_time) && return Vector{Dict{String,Any}}()
    times = sort!(collect(Set{Int}(t for (t, _, _, _) in keys(by_time))))
    frames = Vector{Dict{String,Any}}()
    max_frames = 80
    step = max(1, cld(length(times), max_frames))
    for t in times[1:step:end]
        directed = Dict{Tuple{String,String,String},Float64}()
        for ((tt, src, dst, comm), v) in by_time
            tt == t || continue
            directed[(src, dst, comm)] = get(directed, (src, dst, comm), 0.0) + v
        end
        links = _regional_map_net_links(directed, "use", unit_info)
        shown = isempty(links) ? links : links[1:min(length(links), 60)]
        push!(frames, Dict{String,Any}("time" => t, "links" => shown))
    end
    return frames
end

function _power_capacities(out_dir::AbstractString)
    stock = _read_result_df(out_dir, "techStock")
    meta = _read_result_df(out_dir, "tech_meta")
    isempty(stock) && return Dict("rows" => Vector{Dict{String,Any}}(), "periods" => Int[])
    rows = _df_rows(stock, 50_000)

    meta_by_tech = Dict{String,Dict{String,Any}}()
    if !isempty(meta)
        for r in _df_rows(meta, 5_000)
            meta_by_tech[string(get(r, "tech", ""))] = r
        end
    end

    out = Vector{Dict{String,Any}}()
    period_set = Set{Int}()
    for r in rows
        tech = string(get(r, "tech", ""))
        m = get(meta_by_tech, tech, nothing)
        keep = m === nothing ? false : _is_power_tech_row(m)
        keep || continue
        v = Float64(get(r, "value", 0))
        abs(v) < 1e-6 && continue
        item = Dict{String,Any}(
            "tech" => tech,
            "period" => Int(get(r, "period", 0)),
            "value" => v,
            "sector" => m === nothing ? "" : string(get(m, "sector", "")),
            "subsector" => m === nothing ? "" : string(get(m, "subsector", "")),
            "category" => m === nothing ? "" : string(get(m, "category", "")),
        )
        push!(out, item)
        push!(period_set, Int(get(r, "period", 0)))
    end
    sort!(out; by = x -> (x["period"], -Float64(x["value"])))
    return Dict("rows" => out, "periods" => sort!(collect(period_set)))
end

"""
    _default_ui_node(nodes_list) -> String

Pick a sensible default node for UI selectors. Prefers exact "NL", then any
case-insensitive "NL" match, then any node starting with `NL_` / `NL-`, and
finally falls back to the first node in the supplied list. Returns the empty
string when `nodes_list` is empty.
"""
function _default_ui_node(nodes_list::AbstractVector{<:AbstractString})
    isempty(nodes_list) && return ""
    "NL" in nodes_list && return "NL"
    for n in nodes_list
        uppercase(strip(String(n))) == "NL" && return String(n)
    end
    for n in nodes_list
        s = uppercase(strip(String(n)))
        (startswith(s, "NL_") || startswith(s, "NL-")) && return String(n)
    end
    return String(nodes_list[1])
end

function _hourly_dispatch_payload(out_dir::AbstractString; node::AbstractString = "", period::Union{Nothing,Integer} = nothing)
    df = _read_result_df(out_dir, "tech_use_TS")
    mode = :ts
    time_col = :hc
    if isempty(df)
        df = _read_result_df(out_dir, "tech_use_h")
        mode = :fh
        time_col = :hour
    end
    empty_payload = Dict(
        "timeColumn" => "hour",
        "periods" => Int[],
        "techs" => String[],
        "nodes" => String[],
        "selectedNode" => "",
        "selectedPeriod" => 0,
        "hours" => Int[],
        "techMeta" => Vector{Dict{String,Any}}(),
        "series" => Vector{Dict{String,Any}}(),
        "rows" => Vector{Dict{String,Any}}(),
        "mode" => String(mode),
    )
    isempty(df) && return empty_payload
    all(col in names(df) for col in [String(time_col), "tech", "value", "period"]) || return empty_payload

    # Load tech metadata so we can filter to power / XC-interconnection techs at
    # a chosen node. Older runs without `nodes_meta` still work; node selectors
    # just won't filter anything in that case.
    meta_df = _read_result_df(out_dir, "tech_meta")
    nodes_df = _read_result_df(out_dir, "nodes_meta")
    nodes_list = isempty(nodes_df) ? String[] : String.(nodes_df.node)
    if isempty(nodes_list) && !isempty(meta_df) && "node" in names(meta_df)
        nodes_list = sort!(unique(filter(!isempty, String.(meta_df.node))))
    end
    sort!(nodes_list)

    meta_by_tech = Dict{String,Dict{String,Any}}()
    if !isempty(meta_df)
        for r in _df_rows(meta_df, 5_000)
            meta_by_tech[string(get(r, "tech", ""))] = r
        end
    end

    # Power-tech + cross-border-interconnection eligibility helpers
    is_xc_tech = function (m)
        m === nothing && return false
        ptype = lowercase(string(get(m, "process_type", "")))
        cat   = lowercase(string(get(m, "category", "")))
        sub   = lowercase(string(get(m, "subsector", "")))
        return occursin("interconnect", ptype) || occursin("interconnect", cat) ||
               occursin("interconnect", sub) ||
               occursin("hourly interconnected", ptype)
    end
    keep_for_node = function (tech, sel_node)
        m = get(meta_by_tech, tech, nothing)
        m === nothing && return isempty(sel_node)
        is_power = _is_power_tech_row(m)
        is_xc = is_xc_tech(m)
        (is_power || is_xc) || return false
        if !isempty(sel_node)
            tech_node = string(get(m, "node", ""))
            tech_node == sel_node || return false
        end
        return true
    end

    sel_node = String(node)
    if !isempty(sel_node) && !isempty(nodes_list) && !(sel_node in nodes_list)
        sel_node = ""
    end
    if isempty(sel_node) && !isempty(nodes_list)
        # Default: prefer the Netherlands node ("NL", or any NL_* / NL-*
        # variant) when present, otherwise the first node alphabetically.
        sel_node = _default_ui_node(nodes_list)
    end

    mask = [keep_for_node(String(t), sel_node) for t in df.tech]
    filtered = df[mask, :]
    isempty(filtered) && return merge(empty_payload, Dict(
        "nodes" => nodes_list, "selectedNode" => sel_node,
    ))

    periods_avail = sort!(unique(Int[v for v in (_to_int_safe(x) for x in filtered.period) if v !== nothing]))
    sel_period = period === nothing ? (isempty(periods_avail) ? 0 : periods_avail[1]) : Int(period)
    sel_period in periods_avail || (sel_period = isempty(periods_avail) ? 0 : periods_avail[1])
    sel_period == 0 && return merge(empty_payload, Dict("nodes" => nodes_list, "selectedNode" => sel_node))

    period_mask = [(_to_int_safe(p) === sel_period) for p in filtered.period]
    period_df = filtered[period_mask, [time_col, :tech, :value]]
    isempty(period_df) && return merge(empty_payload, Dict(
        "nodes" => nodes_list, "selectedNode" => sel_node,
        "periods" => periods_avail, "selectedPeriod" => sel_period,
    ))

    # In TS mode, expand the (rep-hour-of-rep-day) axis to a full calendar year
    # using `cluster_map` (calendar_day → rep_day) and the rep-day length
    # derived from the data. In FH mode the axis is already 1..8760.
    DAYS_PER_YEAR = 365
    hours_full = collect(1:(DAYS_PER_YEAR * 24))

    # tech → vector of length 8760
    techs_present = unique(String.(period_df.tech))
    series_per_tech = Dict{String,Vector{Float64}}()
    for t in techs_present
        series_per_tech[t] = zeros(Float64, DAYS_PER_YEAR * 24)
    end

    if mode == :fh
        for r in eachrow(period_df)
            t = String(r.tech)
            h = Int(r.hour)
            (1 <= h <= length(hours_full)) || continue
            series_per_tech[t][h] += Float64(r.value)
        end
    else
        cluster_df = _read_result_df(out_dir, "cluster_map")
        # Determine hours-per-rep-day from the data: max(hc) / n_repDays.
        max_hc = isempty(period_df) ? 0 : maximum(_to_int_safe(h) === nothing ? 0 : _to_int_safe(h)::Int for h in period_df.hc)
        rep_day_ints = isempty(cluster_df) ? Int[] : Int[v for v in (_to_int_safe(x) for x in cluster_df.rep_day) if v !== nothing]
        n_repDays = isempty(rep_day_ints) ? 0 : length(unique(rep_day_ints))
        if n_repDays <= 0
            # Fall back: assume hpd=24 and infer rep_days from max_hc
            n_repDays = max(1, div(max_hc, 24))
        end
        hpd = max(1, n_repDays == 0 ? 24 : div(max_hc, n_repDays))

        # Build (rep_day, slot-in-day) → value per tech
        rep_lookup = Dict{Tuple{String,Int,Int},Float64}()
        for r in eachrow(period_df)
            hc_v = _to_int_safe(r.hc)
            hc_v === nothing && continue
            hc = hc_v::Int
            rd = ((hc - 1) ÷ hpd) + 1
            slot = ((hc - 1) % hpd) + 1
            rep_lookup[(String(r.tech), rd, slot)] = Float64(r.value)
        end
        # calendar_day → rep_day. cluster_map writes `calendar_day::String` and
        # `rep_day::Float64`, so coerce defensively.
        cal_to_rep = Dict{Int,Int}()
        if !isempty(cluster_df) && "calendar_day" in names(cluster_df) && "rep_day" in names(cluster_df)
            for r in eachrow(cluster_df)
                cd = _to_int_safe(r.calendar_day)
                rd_v = _to_int_safe(r.rep_day)
                (cd === nothing || rd_v === nothing) && continue
                cal_to_rep[cd] = rd_v
            end
        end
        # If cluster_map is missing, map every calendar day to rep_day 1 .. n_repDays cyclically
        if isempty(cal_to_rep)
            for d in 1:DAYS_PER_YEAR
                cal_to_rep[d] = ((d - 1) % n_repDays) + 1
            end
        end

        # Expand to 8760 by distributing each rep-day slot evenly across the
        # calendar day's `hours_per_calendar_day = 24` (slots themselves cover
        # `24/hpd` real hours each).
        hours_per_slot = 24 ÷ max(1, hpd)
        hours_per_slot == 0 && (hours_per_slot = 1)
        for cal_day in 1:DAYS_PER_YEAR
            rd = get(cal_to_rep, cal_day, 0)
            rd == 0 && continue
            for t in techs_present
                for slot in 1:hpd
                    val = get(rep_lookup, (t, rd, slot), 0.0)
                    abs(val) < 1e-9 && continue
                    for k in 0:(hours_per_slot - 1)
                        h_in_day = (slot - 1) * hours_per_slot + k + 1
                        h_in_day > 24 && break
                        h = (cal_day - 1) * 24 + h_in_day
                        series_per_tech[t][h] = val
                    end
                end
            end
        end
    end

    # Order techs so the BIGGEST (by total absolute hourly contribution) sits
    # at the BOTTOM of a stacked area. We return the order; the frontend stacks
    # in that order from bottom to top.
    techs_by_total = sort(collect(keys(series_per_tech)); by = t -> -sum(abs, series_per_tech[t]))

    tech_meta_out = Vector{Dict{String,Any}}()
    for t in techs_by_total
        m = get(meta_by_tech, t, nothing)
        push!(tech_meta_out, Dict{String,Any}(
            "tech" => t,
            "name" => m === nothing ? "" : string(get(m, "name", "")),
            "sector" => m === nothing ? "" : string(get(m, "sector", "")),
            "subsector" => m === nothing ? "" : string(get(m, "subsector", "")),
            "category" => m === nothing ? "" : string(get(m, "category", "")),
            "node" => m === nothing ? "" : string(get(m, "node", "")),
            "process_type" => m === nothing ? "" : string(get(m, "process_type", "")),
            "isInterconnection" => m === nothing ? false : is_xc_tech(m),
            "total" => sum(abs, series_per_tech[t]),
        ))
    end

    series_out = Vector{Dict{String,Any}}()
    for t in techs_by_total
        push!(series_out, Dict{String,Any}(
            "tech" => t,
            "values" => series_per_tech[t],
        ))
    end

    return Dict(
        "timeColumn" => "hour",
        "periods" => periods_avail,
        "techs" => techs_by_total,
        "nodes" => nodes_list,
        "selectedNode" => sel_node,
        "selectedPeriod" => sel_period,
        "hours" => hours_full,
        "techMeta" => tech_meta_out,
        "series" => series_out,
        "rows" => Vector{Dict{String,Any}}(),  # legacy field; series is the new API
        "mode" => String(mode),
    )
end

# ----------------------------------------------------------------------------
# Flexibility (reference vs flex demand profile, per flex tech)
# ----------------------------------------------------------------------------
# Reads the AIMMS-compatible `flexibility_profile_price_h` parquet/table
# produced by `write_flexibility_profile_parquet`.  Returns the union of:
#   - tech selector list (with human names from tech_meta, ranked by
#     |shiftNet_h| so the most "active" flex tech is preselected)
#   - period selector list
#   - per-hour series (reference / flex / shiftNet / price) for the
#     requested (tech, period) — or the auto-selected default
#   - per-tech/period indicator rows (n_hours, ref/flex volume, shift UP/DW,
#     cost savings, average prices weighted by activity)
function _flexibility_payload(out_dir::AbstractString; tech::AbstractString = "", period::Union{Nothing,Integer} = nothing)
    df = _read_result_df(out_dir, "flexibility_profile_price_h")
    empty_payload = Dict(
        "techs"          => Vector{Dict{String,Any}}(),
        "periods"        => Int[],
        "selectedTech"   => "",
        "selectedPeriod" => 0,
        "hours"          => Int[],
        "reference"      => Float64[],
        "flex"           => Float64[],
        "shiftNet"       => Float64[],
        "price"          => Float64[],
        "indicators"     => Vector{Dict{String,Any}}(),
        "available"      => false,
    )
    isempty(df) && return empty_payload
    needed = ["hour", "technology", "period", "referenceProfile_h", "shiftNet_h", "flexProfile_h", "electricityPrice_h"]
    all(c in names(df) for c in needed) || return empty_payload

    # Look up human-readable tech names for the selector.
    meta = _read_result_df(out_dir, "tech_meta")
    name_by_tech = Dict{String,String}()
    if !isempty(meta) && "tech" in names(meta) && "name" in names(meta)
        for r in _df_rows(meta, 5_000)
            name_by_tech[String(get(r, "tech", ""))] = String(get(r, "name", ""))
        end
    end

    # Aggregate per (tech, period) — used both for the selector ranking and
    # for the indicator table (mirrors compute_indicators in flex_report.py).
    indicators = Vector{Dict{String,Any}}()
    tech_shift = Dict{String,Float64}()
    periods_seen = Set{Int}()
    techs_seen = Set{String}()
    by_pair = Dict{Tuple{String,Int},Tuple{Int,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}}()
    # tuple = (n, ref_sum, flex_sum, shift_pos, shift_neg, shift_abs,
    #          w_ref_price_num, w_ref_w, w_flex_price_num, w_flex_w, cost_ref - cost_flex)
    for r in _df_rows(df, 2_000_000)
        t = String(get(r, "technology", ""))
        ps = Int(get(r, "period", 0))
        push!(techs_seen, t)
        push!(periods_seen, ps)
        ref = Float64(get(r, "referenceProfile_h", 0.0))
        flx = Float64(get(r, "flexProfile_h", 0.0))
        sh  = Float64(get(r, "shiftNet_h", 0.0))
        pr  = Float64(get(r, "electricityPrice_h", 0.0))
        prev = get(by_pair, (t, ps), (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        n          = prev[1] + 1
        ref_sum    = prev[2] + ref
        flex_sum   = prev[3] + flx
        shift_pos  = prev[4] + max(0.0, sh)
        shift_neg  = prev[5] + min(0.0, sh)
        shift_abs  = prev[6] + abs(sh)
        wrp        = prev[7] + pr * abs(ref)
        wrw        = prev[8] + abs(ref)
        wfp        = prev[9] + pr * abs(flx)
        wfw        = prev[10] + abs(flx)
        savings    = prev[11] + pr * (ref - flx)   # cost_ref - cost_flex
        by_pair[(t, ps)] = (n, ref_sum, flex_sum, shift_pos, shift_neg, shift_abs,
                            wrp, wrw, wfp, wfw, savings)
        tech_shift[t] = get(tech_shift, t, 0.0) + abs(sh)
    end

    for ((t, ps), v) in by_pair
        n = v[1]
        ref_sum, flex_sum, shift_pos, shift_neg, shift_abs = v[2], v[3], v[4], v[5], v[6]
        wrp, wrw, wfp, wfw, savings = v[7], v[8], v[9], v[10], v[11]
        ref_price  = wrw > 1e-12 ? wrp / wrw : 0.0
        flex_price = wfw > 1e-12 ? wfp / wfw : 0.0
        push!(indicators, Dict{String,Any}(
            "tech"                   => t,
            "tech_name"              => get(name_by_tech, t, ""),
            "period"                 => ps,
            "n_hours"                => n,
            "ref_volume"             => ref_sum,
            "flex_volume"            => flex_sum,
            "shift_UP"               => shift_pos,
            "shift_DW"               => -shift_neg,
            "shift_abs"              => shift_abs,
            "ref_avg_price"          => ref_price,
            "flex_avg_price"         => flex_price,
            "cost_savings"           => savings,
        ))
    end
    sort!(indicators; by = x -> (-Float64(get(x, "shift_abs", 0.0)), String(get(x, "tech", ""))))

    periods_avail = sort!(collect(periods_seen))
    # Tech selector ordered by total |shiftNet|, descending.
    techs_sorted = sort!(collect(techs_seen); by = t -> -get(tech_shift, t, 0.0))
    tech_list = Vector{Dict{String,Any}}()
    for t in techs_sorted
        push!(tech_list, Dict{String,Any}(
            "tech"  => t,
            "name"  => get(name_by_tech, t, ""),
            "shift_abs" => get(tech_shift, t, 0.0),
        ))
    end

    sel_tech = String(tech)
    if isempty(sel_tech) || !(sel_tech in techs_sorted)
        sel_tech = isempty(techs_sorted) ? "" : techs_sorted[1]
    end
    sel_period = period === nothing ? (isempty(periods_avail) ? 0 : periods_avail[end]) : Int(period)
    sel_period in periods_avail || (sel_period = isempty(periods_avail) ? 0 : periods_avail[end])

    # Filter rows for the requested (tech, period) and emit aligned series.
    hours_out = Int[]
    ref_out   = Float64[]
    flex_out  = Float64[]
    shift_out = Float64[]
    price_out = Float64[]
    if !isempty(sel_tech) && sel_period > 0
        mask = (String.(df.technology) .== sel_tech) .& (Int.(df.period) .== sel_period)
        sub = df[mask, :]
        if !isempty(sub)
            # Sort by hour to make the line chart monotonic.
            ord = sortperm(Int.(sub.hour))
            for i in ord
                push!(hours_out, Int(sub.hour[i]))
                push!(ref_out,   Float64(sub.referenceProfile_h[i]))
                push!(flex_out,  Float64(sub.flexProfile_h[i]))
                push!(shift_out, Float64(sub.shiftNet_h[i]))
                push!(price_out, Float64(sub.electricityPrice_h[i]))
            end
        end
    end

    return Dict(
        "techs"          => tech_list,
        "periods"        => periods_avail,
        "selectedTech"   => sel_tech,
        "selectedPeriod" => sel_period,
        "hours"          => hours_out,
        "reference"      => ref_out,
        "flex"           => flex_out,
        "shiftNet"       => shift_out,
        "price"          => price_out,
        "indicators"     => indicators,
        "available"      => true,
    )
end

function _balance_activity_options(out_dir::AbstractString)
    bal = _read_result_df(out_dir, "activity_balances")
    isempty(bal) && return Dict("activities" => Vector{Dict{String,Any}}(), "periods" => Int[])
    acts_seen = unique(String.(bal.activity))
    periods = sort!(unique(Int.(bal.period)))
    meta_df = _read_result_df(out_dir, "activities_meta")
    label_by = Dict{String,String}()
    type_by = Dict{String,String}()
    if !isempty(meta_df)
        for r in _df_rows(meta_df, 10_000)
            a = string(get(r, "activity", ""))
            label_by[a] = string(get(r, "label", ""))
            type_by[a] = string(get(r, "type", ""))
        end
    end
    out = Vector{Dict{String,Any}}()
    for a in acts_seen
        push!(out, Dict{String,Any}(
            "activity" => a,
            "label" => get(label_by, a, ""),
            "type" => get(type_by, a, ""),
        ))
    end
    sort!(out; by = x -> lowercase(String(get(x, "activity", ""))))
    return Dict("activities" => out, "periods" => periods)
end

function _emission_grouping_options(out_dir::AbstractString)
    bal = _read_result_df(out_dir, "activity_balances")
    isempty(bal) && return Dict("periods" => Int[])
    periods = sort!(unique(Int.(bal.period)))
    return Dict("periods" => periods)
end

function _supply_demand_payload(out_dir::AbstractString, activity::AbstractString, period)
    isempty(activity) && return Dict("rows" => Vector{Dict{String,Any}}(), "supplyTotal" => 0.0, "demandTotal" => 0.0)
    bal = _read_result_df(out_dir, "activity_balances")
    use = _read_result_df(out_dir, "tech_use")
    (isempty(bal) || isempty(use)) && return Dict("rows" => Vector{Dict{String,Any}}(), "supplyTotal" => 0.0, "demandTotal" => 0.0)

    # Human-readable tech names (only present in tech_meta runs since the
    # `name` column was added). Empty string when not available.
    meta = _read_result_df(out_dir, "tech_meta")
    tech_name_by = Dict{String,String}()
    if !isempty(meta) && "name" in names(meta) && "tech" in names(meta)
        for r in _df_rows(meta, 5_000)
            tech_name_by[String(get(r, "tech", ""))] = String(get(r, "name", ""))
        end
    end

    periods = period === nothing ? sort!(unique(Int.(bal.period))) : [Int(period)]
    rows = Vector{Dict{String,Any}}()
    supply_total = 0.0
    demand_total = 0.0
    use_lookup = Dict{Tuple{String,Int},Float64}()
    for r in _df_rows(use, 200_000)
        use_lookup[(String(get(r, "tech", "")), Int(get(r, "period", 0)))] = Float64(get(r, "value", 0))
    end
    activity_str = String(activity)
    for r in _df_rows(bal, 200_000)
        String(get(r, "activity", "")) == activity_str || continue
        ps = Int(get(r, "period", 0))
        ps in periods || continue
        coef = Float64(get(r, "coef", 0))
        tech = String(get(r, "tech", ""))
        u = get(use_lookup, (tech, ps), 0.0)
        contribution = coef * u
        abs(contribution) < 1e-6 && continue
        push!(rows, Dict{String,Any}(
            "tech" => tech,
            "tech_name" => get(tech_name_by, tech, ""),
            "period" => ps,
            "coef" => coef,
            "use" => u,
            "value" => contribution,
        ))
        if contribution > 0
            supply_total += contribution
        else
            demand_total += contribution
        end
    end
    sort!(rows; by = x -> (Int(x["period"]), -Float64(x["value"])))
    return Dict(
        "activity" => activity_str,
        "periods" => periods,
        "rows" => rows,
        "supplyTotal" => supply_total,
        "demandTotal" => demand_total,
    )
end

function _emissions_payload(out_dir::AbstractString, group_by::AbstractString)
    bal = _read_result_df(out_dir, "activity_balances")
    use = _read_result_df(out_dir, "tech_use")
    acts = _read_result_df(out_dir, "activities_meta")
    meta = _read_result_df(out_dir, "tech_meta")
    (isempty(bal) || isempty(use) || isempty(acts)) && return Dict("rows" => Vector{Dict{String,Any}}(), "groupBy" => String(group_by))

    emission_acts = Set{String}()
    for r in _df_rows(acts, 10_000)
        t = lowercase(string(get(r, "type", "")))
        if t == "emission" || t == "emissionreport"
            push!(emission_acts, String(get(r, "activity", "")))
        end
    end
    isempty(emission_acts) && return Dict("rows" => Vector{Dict{String,Any}}(), "groupBy" => String(group_by))

    use_lookup = Dict{Tuple{String,Int},Float64}()
    for r in _df_rows(use, 200_000)
        use_lookup[(String(get(r, "tech", "")), Int(get(r, "period", 0)))] = Float64(get(r, "value", 0))
    end

    meta_by_tech = Dict{String,Dict{String,Any}}()
    if !isempty(meta)
        for r in _df_rows(meta, 5_000)
            meta_by_tech[String(get(r, "tech", ""))] = r
        end
    end

    group_key = lowercase(strip(String(group_by)))
    function _group_for(tech::String, activity::String)
        if group_key == "activity"
            return activity
        end
        m = get(meta_by_tech, tech, nothing)
        m === nothing && return ""
        if group_key == "sector"
            return string(get(m, "sector", ""))
        elseif group_key == "subsector"
            return string(get(m, "subsector", ""))
        elseif group_key == "kev" || group_key == "sectorkev" || group_key == "sector_kev"
            return string(get(m, "sector_kev", ""))
        elseif group_key == "category"
            return string(get(m, "category", ""))
        end
        return string(get(m, "sector", ""))
    end

    totals = Dict{Tuple{String,Int},Float64}()
    # Per-tech breakdown so the UI can surface "what's inside this group?"
    # in the optional detail table. Keyed by (group, tech, activity, period).
    tech_totals = Dict{Tuple{String,String,String,Int},Float64}()
    period_set = Set{Int}()
    for r in _df_rows(bal, 400_000)
        activity = String(get(r, "activity", ""))
        activity in emission_acts || continue
        ps = Int(get(r, "period", 0))
        push!(period_set, ps)
        coef = Float64(get(r, "coef", 0))
        tech = String(get(r, "tech", ""))
        u = get(use_lookup, (tech, ps), 0.0)
        contribution = coef * u
        abs(contribution) < 1e-9 && continue
        g = _group_for(tech, activity)
        isempty(g) && (g = "(unmapped)")
        totals[(g, ps)] = get(totals, (g, ps), 0.0) + contribution
        tk = (g, tech, activity, ps)
        tech_totals[tk] = get(tech_totals, tk, 0.0) + contribution
    end

    rows = Vector{Dict{String,Any}}()
    for ((g, ps), v) in totals
        push!(rows, Dict{String,Any}("group" => g, "period" => ps, "value" => v))
    end
    sort!(rows; by = x -> (Int(x["period"]), -abs(Float64(x["value"]))))

    # Detail rows are sorted by period, then group (alphabetical), then by
    # |value| within each group so the dominant techs surface first.
    tech_name_by = Dict{String,String}()
    if !isempty(meta) && "name" in names(meta) && "tech" in names(meta)
        for r in _df_rows(meta, 5_000)
            tech_name_by[String(get(r, "tech", ""))] = String(get(r, "name", ""))
        end
    end
    detail_rows = Vector{Dict{String,Any}}()
    for ((g, tech, activity, ps), v) in tech_totals
        push!(detail_rows, Dict{String,Any}(
            "group" => g,
            "tech" => tech,
            "tech_name" => get(tech_name_by, tech, ""),
            "activity" => activity,
            "period" => ps,
            "value" => v,
        ))
    end
    sort!(detail_rows; by = x -> (Int(x["period"]), String(x["group"]), -abs(Float64(x["value"]))))
    return Dict(
        "groupBy" => String(group_by),
        "periods" => sort!(collect(period_set)),
        "rows" => rows,
        "detailRows" => detail_rows,
    )
end

# =============================================================================
# Scenario-space exploration (Phase 1: validate + preview only)
#
# These two endpoints let the UI's ScenarioSpace tab round-trip a campaign
# spec to the server before the user clicks Run:
#   * POST /api/scenario/validate -> errors/warnings + implied sample size
#   * POST /api/scenario/preview  -> first N rows of the sampled matrix
# Neither endpoint touches the model. Phase 2+ will add /start, /:id, etc.
# =============================================================================

"""
    _scenario_validate(body) -> Dict

Build a `CampaignSpec` from the JSON body, run [`validate_spec`](@ref), and
report the implied sample size. Always returns a 200 with `valid=false`
when validation fails — the UI is responsible for displaying messages.
"""
function _scenario_validate(body)
    try
        spec = _scenario_spec_with_gsa_method(spec_from_dict(body), body)
        v = _validate_scenario_direct_run_spec(spec, _scenario_input_path(body))
        return Dict{String,Any}(
            "valid" => v.valid,
            "errors" => v.errors,
            "warnings" => v.warnings,
            "impliedSampleSize" => v.valid ? implied_sample_size(spec) : -1,
            "uniqueParameters" => unique_parameters(spec),
        )
    catch err
        return Dict{String,Any}(
            "valid" => false,
            "errors" => [sprint(showerror, err)],
            "warnings" => String[],
            "impliedSampleSize" => -1,
            "uniqueParameters" => String[],
        )
    end
end

"""
    _scenario_preview(body) -> Dict

Sample the spec and return the first `previewRows` (default 20) rows of the
matrix so the user can sanity-check before launching a campaign. Includes
the implied sample size and seed-determined first-row values to make
client-side reproducibility checks cheap.
"""
function _scenario_preview(body)
    preview_rows = Int(_config_get(body, "previewRows", 20))
    preview_rows = clamp(preview_rows, 1, 200)
    try
        spec = _scenario_spec_with_gsa_method(spec_from_dict(body), body)
        v = _validate_scenario_direct_run_spec(spec, _scenario_input_path(body))
        v.valid || return Dict{String,Any}(
            "ok" => false,
            "errors" => v.errors,
            "warnings" => v.warnings,
        )
        sample = sample_campaign(spec)
        n_show = min(preview_rows, sample.n_variants)
        rows = Vector{Dict{String,Any}}(undef, n_show)
        for i in 1:n_show
            row = Dict{String,Any}("variant" => i)
            for (j, p) in pairs(sample.parameters)
                row[p] = sample.values[i, j]
            end
            rows[i] = row
        end
        return Dict{String,Any}(
            "ok" => true,
            "errors" => String[],
            "warnings" => v.warnings,
            "impliedSampleSize" => sample.n_variants,
            "parameters" => sample.parameters,
            "rows" => rows,
            "shown" => n_show,
        )
    catch err
        return Dict{String,Any}(
            "ok" => false,
            "errors" => [sprint(showerror, err)],
            "warnings" => String[],
        )
    end
end

function _scenario_gsa_method(body)
    raw = lowercase(strip(String(_config_get(body, "gsaMethod", _config_get(body, "gsa_method", "rank")))))
    raw in ("moment_delta", "moment-delta", "moment independent", "moment-independent", "borgonovo", "delta", "borgonovo_delta", "borgonovo-delta") && return "moment_delta"
    raw in ("morris", "elementary", "elementary-effects", "elementary_effects") && return "morris"
    raw in ("sobol", "variance", "variance-based", "variance_based") && return "sobol"
    return "rank"
end

function _scenario_sampler_for_gsa(method::AbstractString, fallback::Symbol)
    method == "morris" && return :morris
    method == "sobol" && return :sobol
    method in ("rank", "moment_delta") && return :lhs
    return fallback
end

function _scenario_spec_with_gsa_method(spec::CampaignSpec, body)
    gsa_method = _scenario_gsa_method(body)
    sampler = _scenario_sampler_for_gsa(gsa_method, spec.method)
    sampler == spec.method && return spec
    return CampaignSpec(
        name = spec.name,
        method = sampler,
        n_variants = spec.n_variants,
        seed = spec.seed,
        rows = spec.rows,
    )
end

# =============================================================================
# Scenario-space campaigns — Phase 3.5: end-to-end run from the UI
#
# The /api/scenario/run endpoint launches a campaign on a background task
# that updates a JSON-friendly snapshot under UI_CAMPAIGNS_LOCK. The
# browser polls /api/scenario/status/<id> every second to refresh the
# progress dashboard. /api/scenario/stop/<id> flips a cancel Ref so the
# orchestrator stops dispatching new variants. /api/scenario/result/<id>
# returns the final summary once the task has completed.
#
# Bridge from CampaignSpec to ScenarioSpec:
#   The UI stores workbook Sheet/Cell coordinates in `row.sheet`/`row.cell`
#   (matching SSDashboard's Parameter Space schema). Before workers launch,
#   those coordinates are resolved to ModelParams leaf fields + indices using
#   the same workbook layout rules as data_reading.jl. Direct ModelParams
#   field/indices rows remain accepted for backward compatibility.
# =============================================================================

function _scenario_supported_direct_fields_text()
    return join(string.(registered_mutation_fields()), ", ")
end

const UI_CAMPAIGN_PHASE_DEFS = (
    (id = "workers",  label = "Making workers"),
    (id = "assign",   label = "Assigning tasks"),
    (id = "generate", label = "Generating"),
    (id = "solve",    label = "Solve"),
    (id = "write",    label = "Export"),
)

function _campaign_phase_skeleton()
    return [Dict{String,Any}(
        "id" => d.id,
        "label" => d.label,
        "status" => "pending",
        "detail" => "",
        "seconds" => nothing,
    ) for d in UI_CAMPAIGN_PHASE_DEFS]
end

function _campaign_phase_index(id::AbstractString)
    for (i, d) in pairs(UI_CAMPAIGN_PHASE_DEFS)
        d.id == id && return i
    end
    return 0
end

function _campaign_set_phase!(snap::Dict{String,Any}, id::AbstractString, status::AbstractString;
                              detail = nothing, seconds = nothing,
                              advance::Bool = true)
    phases = get!(snap, "phases", _campaign_phase_skeleton())
    idx = _campaign_phase_index(id)
    for p in phases
        pidx = _campaign_phase_index(String(get(p, "id", "")))
        if status == "active" && advance && idx > 0 && pidx > 0 && pidx < idx &&
           get(p, "status", "pending") in ("pending", "active")
            p["status"] = "done"
        elseif status == "active" && get(p, "status", "") == "active" &&
               get(p, "id", "") != id
            p["status"] = "done"
        end
    end
    phase = nothing
    for p in phases
        if get(p, "id", "") == id
            phase = p
            break
        end
    end
    if phase === nothing
        phase = Dict{String,Any}("id" => String(id), "label" => String(id),
                                 "status" => "pending", "detail" => "",
                                 "seconds" => nothing)
        push!(phases, phase)
    end
    phase["status"] = String(status)
    detail !== nothing && (phase["detail"] = String(detail))
    seconds !== nothing && (phase["seconds"] = round(Float64(seconds), digits = 3))
    phase["updated_at"] = time()
    return phase
end

function _campaign_reset_after_phase!(snap::Dict{String,Any}, id::AbstractString)
    phases = get!(snap, "phases", _campaign_phase_skeleton())
    idx = _campaign_phase_index(id)
    idx <= 0 && return nothing
    for p in phases
        pidx = _campaign_phase_index(String(get(p, "id", "")))
        if pidx > idx
            p["status"] = "pending"
            p["detail"] = ""
            p["seconds"] = nothing
            p["updated_at"] = time()
        end
    end
    return nothing
end

function _campaign_int_or_nothing(x)
    x === nothing && return nothing
    try
        return Int(x)
    catch
        return nothing
    end
end

function _campaign_worker_slot(workers::AbstractVector, worker_pid, fallback::Integer)
    isempty(workers) && return 0
    pid = _campaign_int_or_nothing(worker_pid)
    if pid !== nothing && pid > 0
        for (i, w) in pairs(workers)
            _campaign_int_or_nothing(get(w, "pid", nothing)) == pid && return i
        end
    end
    return clamp(Int(fallback), 1, length(workers))
end

function _campaign_set_worker_pid!(worker::Dict{String,Any}, worker_pid)
    pid = _campaign_int_or_nothing(worker_pid)
    pid !== nothing && pid > 0 && (worker["pid"] = pid)
    return nothing
end

function _campaign_planned_worker_counts(total::Integer, n_workers::Integer)
    nw = max(1, Int(n_workers))
    counts = zeros(Int, nw)
    for vid in 1:max(0, Int(total))
        counts[((vid - 1) % nw) + 1] += 1
    end
    return counts
end

function _campaign_set_worker_task_totals!(snap::Dict{String,Any}, total::Integer,
                                           n_workers::Integer)
    counts = _campaign_planned_worker_counts(total, n_workers)
    workers = get(snap, "workers", Any[])
    for (i, w) in pairs(workers)
        planned = i <= length(counts) ? counts[i] : 0
        finished = Int(get(w, "completed", 0)) + Int(get(w, "failed", 0))
        running = get(w, "status", "") == "running" ? 1 : 0
        w["assigned"] = max(planned, finished + running)
        haskey(w, "started") || (w["started"] = finished + running)
    end
    return nothing
end

function _campaign_fail_active_phase!(snap::Dict{String,Any}, msg::AbstractString)
    phases = get!(snap, "phases", _campaign_phase_skeleton())
    failed_any = false
    for p in phases
        if get(p, "status", "") == "active"
            p["status"] = "failed"
            p["detail"] = String(msg)
            p["updated_at"] = time()
            failed_any = true
        end
    end
    failed_any || _campaign_set_phase!(snap, "solve", "failed"; detail = msg)
    return nothing
end

"""
    _parse_indices_cell(s::AbstractString) -> Tuple

Parse a UI "Cell" string into a Julia indices tuple.

Accepts: `"NL"`, `"2050"`, `":NL"`, `"(NL, 2050)"`, `"(:NL, 2050)"`,
`"NL,2050"`. Bare identifiers are converted to `Symbol`s, integer-looking
tokens to `Int`, anything else to a `String`. Throws `ArgumentError` if
the string is empty or unparseable.
"""
function _parse_indices_cell(s::AbstractString)
    txt = strip(s)
    isempty(txt) && throw(ArgumentError("Cell is empty — expected an index like NL or (NL, 2050)."))
    # Strip outer parentheses if present
    if startswith(txt, "(") && endswith(txt, ")")
        txt = strip(txt[2:end-1])
    end
    parts = [strip(p) for p in split(txt, ",") if !isempty(strip(p))]
    isempty(parts) && throw(ArgumentError("Cell has no usable tokens after parsing '$s'."))
    out = Any[]
    for p in parts
        tok = strip(p)
        startswith(tok, ":") && (tok = tok[2:end])
        if tryparse(Int, tok) !== nothing
            push!(out, parse(Int, tok))
        elseif occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", tok)
            push!(out, Symbol(tok))
        else
            push!(out, String(tok))
        end
    end
    return Tuple(out)
end

    function _scenario_input_path(body)::String
        input_value = String(_config_get(body, "inputWorkbook", "Input/1108 SSP.xlsx"))
        input_path, _ = _resolve_input_workbook(input_value; require_exists = false)
        return input_path
    end

    function _parse_excel_cell_ref(cell::AbstractString)
        txt = strip(cell)
        m = match(r"^\$?([A-Za-z]{1,3})\$?([0-9]+)$", txt)
        m === nothing && throw(ArgumentError("Cell `$(cell)` is not an Excel A1 coordinate like AE5."))
        col = uppercase(String(m.captures[1]))
        row = parse(Int, m.captures[2])
        row > 0 || throw(ArgumentError("Cell `$(cell)` has an invalid row number."))
        return (; col, row, col_index = _col_index(col), a1 = string(col, row))
    end

    function _row_sheet_cell(row::ParameterRow)
        sheet = String(strip(row.sheet))
        cell = String(strip(row.cell))
        m = match(r"^(?:'([^']+)'|([^!]+))!(.+)$", cell)
        if m !== nothing
            sheet_from_cell = m.captures[1] === nothing ? String(m.captures[2]) : String(m.captures[1])
            sheet = isempty(sheet) ? String(strip(sheet_from_cell)) : sheet
            cell = String(strip(String(m.captures[3])))
        end
        isempty(sheet) && throw(ArgumentError("Sheet is empty."))
        isempty(cell) && throw(ArgumentError("Cell is empty."))
        return (; sheet, cell)
    end

    _coord_candidate(field::Symbol, indices::Tuple, source::String, value) =
        (field = field, indices = indices, source = source, value = value)

    function _sheet_value(sh, row::Int, col::Int)
        try
            return sh[row, col]
        catch
            return nothing
        end
    end

    function _sheet_symbol(sh, row::Int, col::Int, label::AbstractString)
        raw = _str(_sheet_value(sh, row, col))
        isempty(strip(raw)) && throw(ArgumentError("No $(label) key found at $(_col_letter(col))$(row)."))
        return Symbol(strip(raw))
    end

    function _sheet_period(sh, row::Int, col::Int)
        raw = _sheet_value(sh, row, col)
        raw === nothing || ismissing(raw) && throw(ArgumentError("No period header found at $(_col_letter(col))$(row)."))
        if raw isa Number
            return Int(round(Float64(raw)))
        end
        txt = strip(_str(raw))
        n = tryparse(Int, txt)
        n === nothing && throw(ArgumentError("Header $(_col_letter(col))$(row) is `$(txt)`, not a period year."))
        return n
    end

    function _field_for_shared_node_target(row::ParameterRow, candidates)
        length(candidates) == 1 && return first(candidates)
        hint = lowercase(string(row.parameter, " ", row.subparameter, " ", row.notes))
        if occursin("feedstock", hint) || occursin("feed stock", hint) || occursin(" fs", hint)
            for c in candidates
                c.field == :emissionTargetFS && return c
            end
        elseif occursin("bunker", hint)
            for c in candidates
                c.field == :emissionTargetBunker && return c
            end
        end
        fields = join(string.(getfield.(candidates, :field)), ", ")
        throw(ArgumentError("Coordinate maps to multiple model parameters ($(fields)). " *
                            "Use Parameter/Sub-parameter/Notes text containing `Bunker` or `Feedstock` to choose one."))
    end

    function _coordinate_candidates(sheet::AbstractString, sh, ref, row::ParameterRow)
        sheet_key = lowercase(strip(sheet))
        r, c = ref.row, ref.col_index
        value = _sheet_value(sh, r, c)
        src = string(sheet, "!", ref.a1)

        if sheet_key == "nodeparameters"
            r >= 5 || throw(ArgumentError("$(src) is in the NodeParameters header area; choose a data row (5 or later)."))
            node = _sheet_symbol(sh, r, _col_index("A"), "node")
            if _col_index("B") <= c <= _col_index("H")
                ps = _sheet_period(sh, 3, c)
                return [_coord_candidate(:emissionTargetAir, (node, ps), src, value)]
            elseif c == _col_index("I")
                return [_coord_candidate(:CO2_cumulative_budget, (node,), src, value)]
            elseif c == _col_index("J")
                return [_coord_candidate(:cumulative_CO2storage, (node,), src, value)]
            elseif _col_index("R") <= c <= _col_index("X")
                ps = _sheet_period(sh, 3, c)
                return [_coord_candidate(:emissionTargetAll, (node, ps), src, value)]
            elseif _col_index("Y") <= c <= _col_index("AE")
                ps = _sheet_period(sh, 3, c)
                return [_coord_candidate(:emissionTargetBunker, (node, ps), src, value)]
            elseif _col_index("AF") <= c <= _col_index("AL")
                ps = _sheet_period(sh, 3, c)
                return [_coord_candidate(:emissionTargetFS, (node, ps), src, value)]
            end
        elseif sheet_key == "technologies"
            r >= 7 || throw(ArgumentError("$(src) is in the Technologies header area; choose a technology row (7 or later)."))
            tech = _sheet_symbol(sh, r, _col_index("A"), "technology")
            ranges = (
                (:inv_cost,       "I",  "O",  4),
                (:fom_cost,       "Q",  "W",  4),
                (:vom_cost,       "X",  "AD", 4),
                (:decom_planned,  "BO", "BT", 5),
                (:techStock_min,  "BU", "CA", 5),
                (:techStock_max,  "CB", "CH", 5),
                (:techUse_min,    "CI", "CO", 5),
                (:techUse_max,    "CP", "CV", 5),
                (:no_new_invest,  "CW", "DC", 5),
                (:no_eco_decom,   "DD", "DJ", 5),
            )
            for (field, first_col, last_col, header_row) in ranges
                if _col_index(first_col) <= c <= _col_index(last_col)
                    ps = _sheet_period(sh, header_row, c)
                    return [_coord_candidate(field, (tech, ps), src, value)]
                end
            end
            singles = Dict(
                _col_index("P")  => :Salvage_value,
                _col_index("AE") => :WACC,
                _col_index("AF") => :construction_time,
                _col_index("AG") => :economic_lifetime,
                _col_index("AH") => :technical_lifetime,
                _col_index("AI") => :cap2act,
                _col_index("AL") => :ramping,
                _col_index("AO") => :CHP_eta,
                _col_index("AQ") => :CHP_dev_use,
                _col_index("AR") => :CHP_dev_PtoH,
                _col_index("AS") => :shed_capacity_percentage,
                _col_index("AT") => :shed_volume,
                _col_index("AV") => :phs_capacity,
                _col_index("AW") => :reservoir_capacity,
                _col_index("AX") => :phs_Losses,
                _col_index("BA") => :flex_capacity_pct,
                _col_index("BB") => :flex_storage,
                _col_index("BD") => :flex_losses_legacy,
                _col_index("BE") => :flex_nnLoad,
                _col_index("BF") => :avg_journey,
                _col_index("BG") => :avg_speed,
                _col_index("BI") => :bufferUP_capacity,
                _col_index("BJ") => :bufferDW_capacity,
                _col_index("BL") => :buffer_storage,
                _col_index("BM") => :techChange_max,
                _col_index("BN") => :techStock_exist,
            )
            if haskey(singles, c)
                return [_coord_candidate(singles[c], (tech,), src, value)]
            end
        elseif sheet_key == "infrastructure"
            r >= 6 || throw(ArgumentError("$(src) is in the Infrastructure header area; choose an infrastructure row (6 or later)."))
            tech = _sheet_symbol(sh, r, _col_index("A"), "infrastructure technology")
            ranges = (
                (:inv_cost,      "H",  "N",  4),
                (:fom_cost,      "P",  "V",  4),
                (:decom_planned, "AF", "AK", 3),
                (:techStock_min, "AL", "AR", 3),
                (:techStock_max, "AS", "AY", 3),
            )
            for (field, first_col, last_col, header_row) in ranges
                if _col_index(first_col) <= c <= _col_index(last_col)
                    ps = _sheet_period(sh, header_row, c)
                    return [_coord_candidate(field, (tech, ps), src, value)]
                end
            end
            singles = Dict(
                _col_index("O")  => :Salvage_value,
                _col_index("W")  => :WACC,
                _col_index("X")  => :economic_lifetime,
                _col_index("Y")  => :technical_lifetime,
                _col_index("Z")  => :cap2act,
                _col_index("AD") => :techChange_max,
                _col_index("AE") => :techStock_exist,
            )
            if haskey(singles, c)
                return [_coord_candidate(singles[c], (tech,), src, value)]
            end
        elseif sheet_key == "parameters"
            scalar_cells = Dict(
                "B5" => :XC_TransmissionLoss_global,
                "B6" => :baseload_treshold,
                "B7" => :shedding_inLoad,
                "B12" => :social_discount_rate,
                "B13" => :base_year,
                "B40" => :electricity_trade_ratio,
                "B41" => :electricity_trade_volume,
                "B46" => :ActiveConstraintSet,
            )
            if haskey(scalar_cells, ref.a1)
                return [_coord_candidate(scalar_cells[ref.a1], Tuple{}, src, value)]
            end
        end

        throw(ArgumentError("No model-parameter mapping is registered for $(src)."))
    end

    function _resolve_excel_coordinate(row::ParameterRow, input_path::AbstractString)
        isfile(input_path) || throw(ArgumentError("Input workbook not found: $(input_path)"))
        sc = _row_sheet_cell(row)
        ref = _parse_excel_cell_ref(sc.cell)
        candidates = XLSX.openxlsx(input_path, mode = "r") do xf
            sh = try
                xf[sc.sheet]
            catch
                throw(ArgumentError("Workbook has no sheet named `$(sc.sheet)`."))
            end
            _coordinate_candidates(sc.sheet, sh, ref, row)
        end
        return _field_for_shared_node_target(row, candidates)
    end

    function _resolve_parameter_row(row::ParameterRow, input_path::AbstractString)
        sc = _row_sheet_cell(row)
        field = Symbol(sc.sheet)
        if hasproperty(ModelParams(), field)
            indices = _parse_indices_cell(sc.cell)
            return (field = field, indices = indices,
                    source = string("ModelParams.", field, "[", sc.cell, "]"), value = nothing)
        end
        return _resolve_excel_coordinate(row, input_path)
    end

    function _format_indices(indices::Tuple)
        isempty(indices) && return "()"
        return string("(", join(string.(indices), ", "), length(indices) == 1 ? "," : "", ")")
    end

    function _mutation_effect_text(field::Symbol, indices::Tuple, value::Float64)
        muts = build_mutations(ModelData(), field, indices, value)
        isempty(muts) && return "no live model edits"
        targets = String[]
        for m in muts
            if m.kind === :rhs
                push!(targets, string("RHS ", m.constraint_name))
            elseif m.kind === :coef
                push!(targets, string("coef ", m.constraint_name, " / ", m.var_name))
            elseif m.kind === :obj
                push!(targets, string("objective ", m.var_name))
            end
        end
        return join(targets, "; ")
    end

function _scenario_leaf_validation(spec::CampaignSpec, input_path::AbstractString)
    errors = String[]
    warnings = String[]
    supported = Set(registered_mutation_fields())
    supported_text = _scenario_supported_direct_fields_text()
    empty_params = ModelParams()
    for (i, row) in pairs(spec.rows)
        prefix = "Row $i ('$(row.parameter)' / '$(row.subparameter)'):"
        if row.min === nothing || row.max === nothing
            push!(errors, "$prefix Live-run mode requires Min and Max on every row. " *
                          "Shared SSDashboard-style rows with inherited bounds are not supported by the live runner yet.")
        end
        resolved = try
            _resolve_parameter_row(row, input_path)
        catch err
            push!(errors, "$prefix $(sprint(showerror, err))")
            continue
        end
        field = resolved.field
        if !hasproperty(empty_params, field)
            push!(errors, "$prefix $(resolved.source) resolved to `$(field)`, which is not a ModelParams field.")
            continue
        end
        if !(field in supported)
            push!(errors, "$prefix $(resolved.source) resolved to ModelParams.$(field)$(_format_indices(resolved.indices)), " *
                          "but no live scenario mutation builder is registered for `$(field)`. " *
                          "Currently runnable fields: $(supported_text).")
            continue
        end
        try
            effect = _mutation_effect_text(field, resolved.indices, row.min === nothing ? 0.0 : row.min)
            push!(warnings, "$prefix $(resolved.source) -> ModelParams.$(field)$(_format_indices(resolved.indices)); affects $(effect).")
        catch err
            msg = sprint(showerror, err)
            if occursin("expects", msg) || occursin("No mutation builder", msg)
                push!(errors, "$prefix $msg")
            else
                rethrow()
            end
        end
    end
    return (; errors, warnings)
end

function _validate_scenario_direct_run_spec(spec::CampaignSpec, input_path::AbstractString = normpath(joinpath(_repo_root(), "Input", "1108 SSP.xlsx")))
    v = validate_spec(spec)
    errors = String.(v.errors)
    warnings = String.(v.warnings)
    if isempty(errors)
        leaf = _scenario_leaf_validation(spec, input_path)
        append!(errors, leaf.errors)
        append!(warnings, leaf.warnings)
    end
    return (; valid = isempty(errors), errors, warnings)
end

"""
    _row_to_leaf_target(row::ParameterRow) -> LeafTarget

Convert a UI parameter row to a `LeafTarget`, resolving workbook Sheet/Cell
coordinates to the internal ModelParams field/index first. Direct ModelParams
field/indices rows are also accepted for backward compatibility. The
validation path must reject rows without per-row bounds before this function
is called.
"""
function _row_to_leaf_target(row, input_path::AbstractString = normpath(joinpath(_repo_root(), "Input", "1108 SSP.xlsx")))
    if row.min === nothing || row.max === nothing
        throw(ArgumentError("Live-run rows require Min and Max on every row."))
    end
    resolved = _resolve_parameter_row(row, input_path)
    label = isempty(row.subparameter) || row.subparameter == row.parameter ?
        row.parameter : "$(row.parameter)/$(row.subparameter)"
    return LeafTarget(resolved.field, resolved.indices;
                      type = row.type,
                      min = row.min,
                      max = row.max,
                      step = row.step,
                      label = label)
end

"""
    _campaign_session_skeleton(id, spec, n_workers, threads_per_worker, solver, mode) -> Dict

Build the JSON-serialisable snapshot dict that the browser polls.
`workers` is a fixed-length vector with one entry per worker slot;
`campaign` carries the meta-fields the dashboard renders.

The `state` field on `campaign` is the canonical lifecycle indicator the
frontend uses to drive button visibility. Possible values:
  "queued" | "preparing" | "running" | "pausing" | "paused" |
  "resuming" | "cancelling" | "cancelled" | "completed" | "failed"
"""
function _campaign_session_skeleton(id::String, spec, n_workers::Int,
                                    threads_per_worker::Int, solver::Symbol,
                                    mode::Symbol; gsa_method::AbstractString = "rank")
    workers = [Dict{String,Any}(
        "id" => i,
        "status" => "idle",
        "variant_id" => nothing,
        "assigned" => 0,
        "started" => 0,
        "completed" => 0,
        "failed" => 0,
        "progress" => 0.0,
        "started_at" => nothing,
        "pid" => nothing,
        "rss_bytes" => 0,
        "last_error" => nothing,
        "last_term" => nothing,
        "last_failed_variant" => nothing,
    ) for i in 1:max(1, n_workers)]
    return Dict{String,Any}(
        "campaign" => Dict{String,Any}(
            "id" => id,
            "name" => spec.name,
            "method" => String(spec.method),
            "gsa_method" => String(gsa_method),
            "total" => spec.n_variants,
            "n_workers" => max(1, n_workers),
            "threads_per_worker" => threads_per_worker,
            "solver" => String(solver),
            "mode" => String(mode),
            "started_at" => time(),
            "completed" => 0,
            "failed" => 0,
            "status" => "queued",
            "state" => "queued",
            "stage" => "Queued",
            "avg_task_seconds" => nothing,
            "task_seconds_count" => 0,
            "collect_save_seconds" => nothing,
        ),
        "workers" => workers,
        "phases" => _campaign_phase_skeleton(),
        "parameter_labels" => [t.label for t in spec.targets],
        "result_points" => Vector{Dict{String,Any}}(),
        "failures" => Vector{Dict{String,Any}}(),  # most-recent first, capped
        "done" => false,
        "error" => nothing,
    )
end

"""
    _scenario_run(body) -> Dict

Launch a scenario-space campaign on a background task. Returns the
campaign id and the initial snapshot. The browser then polls
`/api/scenario/status/<id>` to drive the dashboard. Always returns 200
with `ok=false` + `errors` when the spec fails validation, so the UI can
surface error messages without distinguishing HTTP status codes.
"""
function _scenario_run(body)
    try
        gsa_method = _scenario_gsa_method(body)
        spec = _scenario_spec_with_gsa_method(spec_from_dict(body), body)
        input_path = _scenario_input_path(body)
        v = _validate_scenario_direct_run_spec(spec, input_path)
        v.valid || return Dict{String,Any}(
            "ok" => false,
            "errors" => v.errors,
            "warnings" => v.warnings,
        )

        # Bridge CampaignSpec rows -> ScenarioSpec LeafTargets.
        targets = LeafTarget[]
        seen = Set{Tuple{Symbol,Tuple}}()
        for r in spec.rows
            lt = _row_to_leaf_target(r, input_path)
            key = (lt.field, lt.indices)
            key in seen && continue
            push!(seen, key)
            push!(targets, lt)
        end
        isempty(targets) && return Dict{String,Any}(
            "ok" => false,
            "errors" => ["No usable parameter targets after bridging. " *
                         "Each row's Sheet/Cell must resolve to a registered live ModelParams mutation target " *
                         "(for example NodeParameters!AE5 -> emissionTargetBunker[NL,2050])."],
            "warnings" => v.warnings,
        )

        scenario_spec = ScenarioSpec(
            name = spec.name,
            method = spec.method,
            n_variants = spec.n_variants,
            seed = spec.seed,
            targets = targets,
        )

        # Workbook + solver settings
        input_value = String(_config_get(body, "inputWorkbook", "Input/1108 SSP.xlsx"))
        isfile(input_path) || return Dict{String,Any}(
            "ok" => false,
            "errors" => ["Input workbook not found: $input_value"],
            "warnings" => String[],
        )
        n_workers = max(1, _as_int(_config_get(body, "n_workers", _config_get(body, "workers", 1)), 1))
        threads_per_worker = max(1, _as_int(_config_get(body, "threads_per_worker", 1), 1))
        solver_sym = Symbol(lowercase(String(_config_get(body, "solver", "highs"))))
        mode_raw = lowercase(String(_config_get(body, "mode", "ts")))
        mode_sym = mode_raw in ("fh", "full_hourly", "full-hourly") ? :fh :
                   mode_raw in ("annual",) ? :annual : :ts
        periods = _as_int_vector(_config_get(body, "periods", [2050]))

        # Allocate id and snapshot
        id = "camp_" * Dates.format(now(), "yyyymmdd_HHMMSS") * "_" * randstring(6)
        snapshot = _campaign_session_skeleton(id, scenario_spec, n_workers,
                                              threads_per_worker, solver_sym, mode_sym;
                                              gsa_method = gsa_method)
        cancel_ref = Ref(false)
        state = Dict{Symbol,Any}(
            :input_path         => input_path,
            :periods            => periods,
            :scenario_spec      => scenario_spec,
            :n_workers          => n_workers,
            :threads_per_worker => threads_per_worker,
            :solver             => solver_sym,
            :mode               => mode_sym,
            # Filled in by _prepare_campaign_state! on the first run; reused
            # by every subsequent /resume so we never re-read the workbook.
            :base_md            => nothing,
            :all_changes        => nothing,
            :completed          => Set{Int}(),
            :prepared           => false,
        )
        lock(UI_CAMPAIGNS_LOCK)
        try
            UI_CAMPAIGNS[id] = snapshot
            UI_CAMPAIGN_CANCEL[id] = cancel_ref
            UI_CAMPAIGN_STATE[id] = state
        finally
            unlock(UI_CAMPAIGNS_LOCK)
        end

        # Launch background task. All updates go through helpers that take
        # the lock; renderProgress on the browser side reads via /status.
        task = Base.Threads.@spawn _run_campaign_task!(id, cancel_ref)
        lock(UI_CAMPAIGNS_LOCK)
        try
            UI_CAMPAIGN_TASKS[id] = task
        finally
            unlock(UI_CAMPAIGNS_LOCK)
        end

        return Dict{String,Any}(
            "ok" => true,
            "campaign_id" => id,
            "snapshot" => _scenario_status(id),
            "warnings" => v.warnings,
        )
    catch err
        return Dict{String,Any}(
            "ok" => false,
            "errors" => [sprint(showerror, err)],
            "warnings" => String[],
        )
    end
end

"""
    _run_campaign_task!(id, cancel)

Background task body. On the first call it primes ModelData + samples the
spec (heavy work, done once and cached in `UI_CAMPAIGN_STATE[id]`); then
it executes the variants that have not yet completed. On every later call
(triggered by `/api/scenario/resume/<id>`) it skips the prep and just
executes the still-pending variants.

`cancel[]` is honoured between variants by the orchestrator. Setting it
mid-run causes `run_campaign` to drain its in-flight variants and return;
this task then either parks in the "paused" state (waiting for /resume)
or terminates in "cancelled" (terminal stop). Which one is decided by
the value of `state[:terminal_stop]` at return time.
"""
function _run_campaign_task!(id::String, cancel::Ref{Bool})
    try
        state = _campaign_state(id)
        state === nothing && return nothing

        if !state[:prepared]
            _prepare_campaign_state!(id)
            state[:prepared] = true
        end

        _execute_pending_variants!(id, cancel)
    catch err
        msg = sprint(showerror, err)
        @error "Campaign task failed" id error=msg exception=(err, catch_backtrace())
        _campaign_update!(id) do snap
            snap["campaign"]["status"] = "failed"
            snap["campaign"]["state"] = "failed"
            snap["campaign"]["stage"] = "Failed: $msg"
            snap["done"] = true
            snap["error"] = msg
            _campaign_fail_active_phase!(snap, msg)
        end
    end
    return nothing
end

"""
    _campaign_state(id) -> Dict{Symbol,Any} or nothing

Fetch the mutable per-campaign state dict (held outside UI_CAMPAIGNS
because its values aren't JSON-serialisable). Returns `nothing` if the
campaign has been forgotten.
"""
function _campaign_state(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        return get(UI_CAMPAIGN_STATE, String(id), nothing)
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
end

"""
    _prepare_campaign_state!(id)

One-time setup: read the workbook (cached), derive sets/params, cluster
representative days if needed, then sample the scenario space and
generate the per-variant change lists. Results are stored in
`UI_CAMPAIGN_STATE[id]` so /resume can re-use them without re-reading.
"""
function _prepare_campaign_state!(id::String)
    state = _campaign_state(id)
    state === nothing && return nothing

    _campaign_update!(id) do snap
        snap["campaign"]["status"] = "reading"
        snap["campaign"]["state"] = "preparing"
        snap["campaign"]["stage"] = "Reading workbook"
        _campaign_set_phase!(snap, "generate", "active";
                             detail = "Reading workbook", advance = false)
    end
    md = _read_ui_data_cached(state[:input_path])
    periods = state[:periods]
    if !isempty(periods)
        selected = [p for p in periods if p in md.sets.periods]
        if !isempty(selected)
            md.sets.periods_solve = selected
        end
    end

    _campaign_update!(id) do snap
        snap["campaign"]["status"] = "preparing"
        snap["campaign"]["state"] = "preparing"
        snap["campaign"]["stage"] = "Deriving sets and parameters"
        _campaign_set_phase!(snap, "generate", "active";
                             detail = "Deriving sets and parameters", advance = false)
    end
    derive_sets!(md)
    compute_derived_params!(md)
    if state[:mode] == :ts
        _campaign_update!(id) do snap
            snap["campaign"]["stage"] = "Clustering representative days"
            _campaign_set_phase!(snap, "generate", "active";
                                 detail = "Clustering representative days", advance = false)
        end
        build_temporal_clusters!(md)
    end

    _campaign_update!(id) do snap
        snap["campaign"]["stage"] = "Sampling scenario space"
        _campaign_set_phase!(snap, "generate", "active";
                             detail = "Sampling scenario space", advance = false)
    end
    spec = state[:scenario_spec]
    samples = sample_scenario_space(spec)
    all_changes = samples_to_changes(spec, samples)

    state[:base_md]     = md
    state[:samples]     = samples
    state[:all_changes] = all_changes
    _campaign_update!(id) do snap
        _campaign_set_phase!(snap, "generate", "done";
                             detail = string(length(all_changes), " variants generated"))
    end
    return nothing
end

"""
    _execute_pending_variants!(id, cancel)

Compute the set of variants that have not yet completed and dispatch
them via `run_campaign`. On normal return (all done) the snapshot is
flipped to "completed". On a cooperative cancel that was triggered by
/api/scenario/pause/<id>, the snapshot lands in "paused" and the cached
state is left intact so /resume can pick up where we left off. On a
cancel triggered by /api/scenario/stop/<id> the snapshot lands in
"cancelled" (terminal).
"""
function _execute_pending_variants!(id::String, cancel::Ref{Bool})
    state = _campaign_state(id)
    state === nothing && return nothing

    all_changes = state[:all_changes]::AbstractVector
    total = length(all_changes)
    completed = state[:completed]::Set{Int}
    pending = sort!(collect(setdiff(1:total, completed)))

    if isempty(pending)
        _campaign_update!(id) do snap
            snap["campaign"]["status"] = "completed"
            snap["campaign"]["state"] = "completed"
            snap["campaign"]["stage"] = "All variants completed"
            _campaign_set_phase!(snap, "solve", "done"; detail = "No variants pending")
            _campaign_set_phase!(snap, "write", "skipped"; detail = "Export not enabled")
            snap["done"] = true
        end
        return nothing
    end

    nw = max(1, state[:n_workers]::Int)
    # Round-robin worker assignment uses the original variant_id so the
    # dashboard's worker grid stays stable across pause/resume cycles.
    assign_worker = vid -> ((vid - 1) % nw) + 1

    _campaign_update!(id) do snap
        snap["campaign"]["status"] = "running"
        snap["campaign"]["state"] = "running"
        _campaign_set_worker_task_totals!(snap, total, nw)
        if state[:n_workers] <= 1
            snap["campaign"]["stage"] = string("Solving variants (",
                                               length(pending), " pending of ",
                                               total, ")")
            _campaign_set_phase!(snap, "workers", "done"; detail = "Using UI process")
            _campaign_set_phase!(snap, "assign", "done"; detail = string(length(pending), " variants queued"))
            _campaign_set_phase!(snap, "solve", "active";
                                 detail = string("0/", length(pending), " variants solved"))
        else
            snap["campaign"]["stage"] = string("Making workers (",
                                               state[:n_workers], " requested)")
            _campaign_set_phase!(snap, "workers", "active";
                                 detail = string("Starting ", state[:n_workers], " workers"),
                                 advance = false)
            _campaign_reset_after_phase!(snap, "workers")
        end
        for w in snap["workers"]
            if w["status"] == "idle"
                # leave counters intact; just mark as queued so the UI
                # shows it as picked up by the running campaign
                w["progress"] = 0.0
            end
        end
    end

    # The runner numbers variants 1..length(subset_changes); we translate
    # back to the original 1..total index for the snapshot.
    subset_changes = all_changes[pending]
    on_progress = function (info)
        cancel[] && return
        real_vid = pending[info.variant_id]
        fallback_wid = assign_worker(real_vid)
        worker_pid = get(info, :worker_pid, nothing)
        if info.stage == "done"
            res = get(info, :result, nothing)
            res !== nothing && (worker_pid = res.worker_pid)
        end
        _campaign_update!(id) do snap
            wid = _campaign_worker_slot(snap["workers"], worker_pid, fallback_wid)
            wid == 0 && return
            w = snap["workers"][wid]
            _campaign_set_worker_pid!(w, worker_pid)
            if info.stage == "start"
                w["status"] = "running"
                w["variant_id"] = real_vid
                w["started"] = Int(get(w, "started", 0)) + 1
                w["assigned"] = max(Int(get(w, "assigned", 0)),
                                    Int(get(w, "completed", 0)) + Int(get(w, "failed", 0)) + 1)
                w["started_at"] = time()
                w["progress"] = 0.0
            elseif info.stage == "done"
                res = get(info, :result, nothing)
                term = res === nothing ? "UNKNOWN" : String(res.term_status)
                failed = res === nothing ? false :
                         !(uppercase(term) in ("OPTIMAL", "LOCALLY_SOLVED"))
                if res !== nothing
                    obj = isfinite(res.objective) ? res.objective : nothing
                    co2p = isfinite(res.co2_price) ? res.co2_price : nothing
                    task_seconds = res.build_seconds + res.apply_seconds + res.solve_seconds
                    if !(isfinite(task_seconds) && task_seconds > 0)
                        started_at = get(w, "started_at", nothing)
                        if started_at isa Number
                            task_seconds = max(0.0, time() - Float64(started_at))
                        end
                    end
                    if isfinite(task_seconds) && task_seconds > 0
                        prev_sum = Float64(get(snap["campaign"], "task_seconds_sum", 0.0))
                        prev_count = Int(get(snap["campaign"], "task_seconds_count", 0))
                        next_sum = prev_sum + task_seconds
                        next_count = prev_count + 1
                        snap["campaign"]["task_seconds_sum"] = next_sum
                        snap["campaign"]["task_seconds_count"] = next_count
                        snap["campaign"]["avg_task_seconds"] = next_sum / next_count
                        snap["campaign"]["collect_save_seconds"] = max(20.0,
                            2.0 * Float64(max(1, state[:n_workers]::Int)) +
                            0.05 * Float64(total))
                    end
                    push!(snap["result_points"], Dict{String,Any}(
                        "variant_id" => real_vid,
                        "worker_id" => wid,
                        "system_cost" => obj,
                        "co2_price" => co2p,
                        "term_status" => term,
                        "parameters" => Dict(String(label) => val for (label, val) in zip(get(snap, "parameter_labels", String[]), res.leaf_values)),
                    ))
                end
                w["status"] = failed ? "failed" : "done"
                w["variant_id"] = real_vid
                if failed
                    w["failed"] += 1
                    snap["campaign"]["failed"] += 1
                    # Record diagnostic details on the worker card AND in a
                    # campaign-wide failures list so the UI can show them.
                    err_text = (res === nothing || res.error === nothing) ?
                               "No error message (term=$term)" :
                               String(res.error)
                    w["last_error"] = err_text
                    w["last_term"] = term
                    w["last_failed_variant"] = real_vid
                    failures = snap["failures"]::Vector{Dict{String,Any}}
                    pushfirst!(failures, Dict{String,Any}(
                        "variant_id" => real_vid,
                        "worker_id" => wid,
                        "worker_pid" => (res === nothing ? nothing : res.worker_pid),
                        "term_status" => term,
                        "error" => err_text,
                        "at" => time(),
                    ))
                    # Cap to 50 most recent failures to keep snapshot small.
                    length(failures) > 50 && resize!(failures, 50)
                else
                    w["completed"] += 1
                    snap["campaign"]["completed"] += 1
                end
                solved = snap["campaign"]["completed"] + snap["campaign"]["failed"]
                _campaign_set_phase!(snap, "solve", "active";
                                     detail = string(solved, "/", total, " variants solved"))
                w["progress"] = 1.0
            end
        end
        if info.stage == "done"
            # Variant is durably done — record it so /resume skips it.
            push!(completed, real_vid)
        end
    end

    on_result = function (_r)
        # Reserved for streaming per-variant detail (results table /
        # DuckDB write). on_progress already drives the dashboard.
        return nothing
    end

    on_phase = function (info)
        phase = Symbol(get(info, :phase, :unknown))
        seconds = get(info, :seconds, nothing)
        pids = get(info, :pids, Int[])
        n_pids = length(pids)
        _campaign_update!(id) do snap
            if phase == :addprocs
                snap["campaign"]["stage"] = string("Making workers (", n_pids, " started)")
                for (slot, pid) in enumerate(pids)
                    if slot <= length(snap["workers"])
                        snap["workers"][slot]["pid"] = Int(pid)
                    end
                end
                _campaign_set_phase!(snap, "workers", "active";
                                     detail = string("Started ", n_pids, " workers"),
                                     seconds = seconds, advance = false)
                _campaign_reset_after_phase!(snap, "workers")
            elseif phase == :workers_loaded
                snap["campaign"]["stage"] = "Assigning tasks"
                _campaign_set_phase!(snap, "workers", "done";
                                     detail = string(n_pids, " workers loaded IESAOpt"),
                                     seconds = seconds)
                _campaign_set_phase!(snap, "assign", "active";
                                     detail = string(length(pending), " variants waiting"))
                _campaign_reset_after_phase!(snap, "assign")
            elseif phase == :ship_base_data
                snap["campaign"]["stage"] = string("Solving variants (",
                                                   length(pending), " pending of ",
                                                   total, ")")
                _campaign_set_phase!(snap, "assign", "done";
                                     detail = string("Base data shipped to ", n_pids, " workers"),
                                     seconds = seconds)
                _campaign_set_phase!(snap, "generate", "done";
                                     detail = string(length(pending), " variant inputs generated"))
                _campaign_set_phase!(snap, "solve", "active";
                                     detail = string("0/", length(pending), " variants solved"))
            elseif phase == :rmprocs_failed
                err = get(info, :error, "Worker cleanup failed")
                _campaign_set_phase!(snap, "solve", "failed"; detail = String(err))
            end
        end
        return nothing
    end

    md = state[:base_md]
    t0 = time()
    # Start a background RAM sampler that polls peak RSS on each Distributed
    # worker (or the master, for the in-process serial path) every 2 s and
    # writes the result into the snapshot. The sampler stops itself when
    # `sampler_done[]` is set; we set it right after run_campaign returns.
    sampler_done = Ref(false)
    sampler = Base.Threads.@spawn _campaign_rss_sampler!(id, sampler_done)
    try
        run_campaign(md, subset_changes;
                     n_workers = state[:n_workers],
                     threads_per_worker = state[:threads_per_worker],
                     solver = state[:solver],
                     mode = state[:mode],
                     cancel = cancel,
                     on_progress = on_progress,
                     on_result = on_result,
                     on_phase = on_phase)
    finally
        sampler_done[] = true
        try; wait(sampler); catch; end
    end
    leg_seconds = round(time() - t0, digits = 3)

    # Decide why we returned. If cancel was raised, /pause vs /stop is
    # signalled by the snapshot's current state field which the handler
    # set before flipping the Ref.
    snap_state_after = _campaign_snapshot_field(id, "state")
    cancelled = cancel[]
    if cancelled && snap_state_after == "pausing"
        _campaign_update!(id) do snap
            snap["campaign"]["status"] = "paused"
            snap["campaign"]["state"] = "paused"
            snap["campaign"]["stage"] = string("Paused after ",
                                               snap["campaign"]["completed"], "/",
                                               snap["campaign"]["total"],
                                               " variants — press Resume to continue")
            _campaign_set_phase!(snap, "solve", "active";
                                 detail = string("Paused at ", snap["campaign"]["completed"] + snap["campaign"]["failed"], "/",
                                                 snap["campaign"]["total"], " variants"))
            prev = get(snap["campaign"], "runtime_sec", 0.0)
            snap["campaign"]["runtime_sec"] = round(prev + leg_seconds, digits = 3)
            for w in snap["workers"]
                if w["status"] == "running"
                    w["status"] = "idle"
                end
            end
        end
    elseif cancelled
        _campaign_update!(id) do snap
            snap["campaign"]["status"] = "cancelled"
            snap["campaign"]["state"] = "cancelled"
            snap["campaign"]["stage"] = string("Stopped after ",
                                               snap["campaign"]["completed"], "/",
                                               snap["campaign"]["total"],
                                               " variants")
            _campaign_set_phase!(snap, "solve", "skipped";
                                 detail = string("Stopped at ", snap["campaign"]["completed"] + snap["campaign"]["failed"], "/",
                                                 snap["campaign"]["total"], " variants"))
            prev = get(snap["campaign"], "runtime_sec", 0.0)
            snap["campaign"]["runtime_sec"] = round(prev + leg_seconds, digits = 3)
            snap["done"] = true
            for w in snap["workers"]
                if w["status"] == "running"
                    w["status"] = "idle"
                end
            end
        end
        _forget_campaign_state!(id)
    else
        _campaign_update!(id) do snap
            snap["campaign"]["status"] = "completed"
            snap["campaign"]["state"] = "completed"
            snap["campaign"]["stage"] = "All variants completed"
            _campaign_set_phase!(snap, "solve", "done";
                                 detail = string(snap["campaign"]["completed"] + snap["campaign"]["failed"], "/",
                                                 snap["campaign"]["total"], " variants solved"))
            _campaign_set_phase!(snap, "write", "skipped"; detail = "Export not enabled")
            prev = get(snap["campaign"], "runtime_sec", 0.0)
            snap["campaign"]["runtime_sec"] = round(prev + leg_seconds, digits = 3)
            snap["done"] = true
            for w in snap["workers"]
                if w["status"] == "running"
                    w["status"] = "idle"
                end
            end
        end
        _forget_campaign_state!(id)
    end
    return nothing
end

"""
    _campaign_snapshot_field(id, key) -> String or nothing

Read a single string field from `campaign` in the live snapshot under
the lock. Returns `nothing` if the campaign or the key is missing.
"""
function _campaign_snapshot_field(id::AbstractString, key::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        snap = get(UI_CAMPAIGNS, String(id), nothing)
        snap === nothing && return nothing
        val = get(snap["campaign"], key, nothing)
        return val === nothing ? nothing : String(val)
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
end

"""
    _forget_campaign_state!(id)

Drop the heavy mutable state for a terminally-finished campaign. The
JSON snapshot in UI_CAMPAIGNS is kept so the browser can still poll
/status and /result.
"""
function _forget_campaign_state!(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        delete!(UI_CAMPAIGN_STATE, String(id))
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    return nothing
end

"""
    _campaign_rss_sampler!(id, done_ref)

Background loop that polls peak RSS on each Distributed worker (or the
master, for the in-process serial path) every ~2 s and writes the
result into the snapshot. Reported value is `Sys.maxrss()` — peak
resident set size in bytes, which never decreases for the lifetime of
the process but is exactly what you want for monitoring solver memory
pressure. Exits when `done_ref[]` becomes true.

Worker pids are bound to UI slots in the order returned by
`Distributed.workers()`; that order is stable for the lifetime of a
single `run_campaign` call, so each UI card shows RAM for the same
process throughout the leg.
"""
function _campaign_rss_sampler!(id::AbstractString, done_ref::Ref{Bool})
    sid = String(id)
    sample_period = 2.0
    while !done_ref[]
        try
            pids = try
                Int[Int(p) for p in Distributed.workers()]
            catch
                Int[]
            end
            samples = Tuple{Int,Int}[]  # (ui_slot, rss_bytes)
            if isempty(pids) || (length(pids) == 1 && pids[1] == 1)
                # Serial / in-process path: just sample the master.
                rss = try Int(Sys.maxrss()) catch; 0 end
                push!(samples, (1, rss))
            else
                for (i, pid) in enumerate(pids)
                    rss = 0
                    try
                        rss = Int(Distributed.remotecall_fetch(Sys.maxrss, pid))
                    catch
                        # Worker may have been removed between workers() and
                        # the remotecall; skip silently.
                    end
                    push!(samples, (i, rss))
                end
            end
            if !isempty(samples)
                _campaign_update!(sid) do snap
                    wlist = snap["workers"]
                    nw = length(wlist)
                    for (slot, rss) in samples
                        if 1 <= slot <= nw
                            w = wlist[slot]
                            w["rss_bytes"] = rss
                            if slot <= length(pids)
                                w["pid"] = pids[slot]
                            elseif length(pids) == 0
                                w["pid"] = Int(getpid())
                            end
                        end
                    end
                end
            end
        catch err
            @debug "rss sampler error" id=sid err
        end
        # Sleep in short slices so we honour done_ref quickly when the
        # campaign returns or is cancelled.
        slept = 0.0
        while slept < sample_period && !done_ref[]
            sleep(0.25)
            slept += 0.25
        end
    end
    return nothing
end

"""
    _campaign_update!(f, id::String)

Apply `f(snap)` to the campaign snapshot under the lock. `f` is called
with the live `Dict{String,Any}` and may mutate it in place.
"""
function _campaign_update!(f, id::String)
    lock(UI_CAMPAIGNS_LOCK)
    try
        snap = get(UI_CAMPAIGNS, id, nothing)
        snap === nothing && return nothing
        f(snap)
        snap["campaign"]["updated_at"] = time()
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    yield()
    return nothing
end

"""
    _scenario_status(id) -> Dict

Return a deep-ish copy of the campaign snapshot keyed by `id`, or an
error dict if the id is unknown.
"""
function _scenario_status(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        snap = get(UI_CAMPAIGNS, String(id), nothing)
        snap === nothing && return Dict{String,Any}(
            "ok" => false,
            "error" => "Unknown campaign id: $id",
        )
        return deepcopy(snap)
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
end

"""
    _scenario_stop!(id) -> Dict

Terminal cancel for a campaign. Flips the cancel Ref so the runner
drains its in-flight variants and returns; the snapshot lands in
"cancelled" and the cached state is discarded so /resume is not
allowed afterwards. Returns `{ok: true}` on success, or an error dict
if the id is unknown.
"""
function _scenario_stop!(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        cref = get(UI_CAMPAIGN_CANCEL, String(id), nothing)
        cref === nothing && return Dict{String,Any}(
            "ok" => false,
            "error" => "Unknown campaign id: $id",
        )
        snap = get(UI_CAMPAIGNS, String(id), nothing)
        if snap !== nothing
            cur = String(get(snap["campaign"], "state", "queued"))
            if cur in ("completed", "cancelled", "failed")
                return Dict{String,Any}(
                    "ok" => false,
                    "error" => "Campaign $id already finished ($cur).",
                )
            end
            snap["campaign"]["status"] = "cancelling"
            snap["campaign"]["state"] = "cancelling"
            snap["campaign"]["stage"] = "Stop requested — finishing in-flight variants"
        end
        cref[] = true
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    return Dict{String,Any}("ok" => true, "campaign_id" => id)
end

"""
    _scenario_pause!(id) -> Dict

Cooperative pause. Sets the snapshot state to "pausing" then flips the
cancel Ref; the runner finishes its in-flight variants and returns. The
task body sees state=="pausing" and parks in "paused" instead of
"cancelled", keeping the cached `UI_CAMPAIGN_STATE[id]` alive so /resume
can pick up where we left off.
"""
function _scenario_pause!(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    try
        cref = get(UI_CAMPAIGN_CANCEL, String(id), nothing)
        cref === nothing && return Dict{String,Any}(
            "ok" => false,
            "error" => "Unknown campaign id: $id",
        )
        snap = get(UI_CAMPAIGNS, String(id), nothing)
        if snap !== nothing
            cur = String(get(snap["campaign"], "state", "queued"))
            if cur != "running"
                return Dict{String,Any}(
                    "ok" => false,
                    "error" => "Cannot pause campaign in state '$cur' (only 'running' is pauseable).",
                )
            end
            # IMPORTANT: set state to "pausing" BEFORE flipping the Ref
            # so the task body sees the right state when it returns.
            snap["campaign"]["status"] = "pausing"
            snap["campaign"]["state"] = "pausing"
            snap["campaign"]["stage"] = "Pause requested — finishing in-flight variants"
        end
        cref[] = true
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    return Dict{String,Any}("ok" => true, "campaign_id" => id)
end

"""
    _scenario_resume!(id) -> Dict

Spin up a fresh background task that calls `_run_campaign_task!` again
with a new cancel Ref. The task body's gate `state[:prepared]` ensures
we skip the prep work and go straight to executing whatever variants
have not yet completed. Returns an error dict if the campaign is not in
"paused" state.
"""
function _scenario_resume!(id::AbstractString)
    sid = String(id)
    new_cancel = Ref(false)
    lock(UI_CAMPAIGNS_LOCK)
    try
        snap = get(UI_CAMPAIGNS, sid, nothing)
        snap === nothing && return Dict{String,Any}(
            "ok" => false,
            "error" => "Unknown campaign id: $id",
        )
        cur = String(get(snap["campaign"], "state", "queued"))
        if cur != "paused"
            return Dict{String,Any}(
                "ok" => false,
                "error" => "Cannot resume campaign in state '$cur' (only 'paused' is resumable).",
            )
        end
        state = get(UI_CAMPAIGN_STATE, sid, nothing)
        if state === nothing
            return Dict{String,Any}(
                "ok" => false,
                "error" => "Campaign state for $id has been forgotten and cannot be resumed.",
            )
        end
        snap["campaign"]["status"] = "resuming"
        snap["campaign"]["state"] = "resuming"
        snap["campaign"]["stage"] = "Resuming campaign"
        # Replace the cancel Ref so the new task gets a fresh one.
        UI_CAMPAIGN_CANCEL[sid] = new_cancel
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end

    task = Base.Threads.@spawn _run_campaign_task!(sid, new_cancel)
    lock(UI_CAMPAIGNS_LOCK)
    try
        UI_CAMPAIGN_TASKS[sid] = task
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    return Dict{String,Any}("ok" => true, "campaign_id" => sid)
end

"""
    _scenario_result(id) -> Dict

Final summary for a completed campaign. Returns `done=false` if the
task is still running so the UI can keep polling.
"""
function _scenario_delta_gsa(points, labels; nboot::Int = 100)
    isempty(points) && return Dict{String,Any}[]
    targets = LeafTarget[]
    for (index, label) in enumerate(labels)
        values = Float64[Float64(point["parameters"][String(label)]) for point in points
                         if haskey(point["parameters"], String(label))]
        isempty(values) && continue
        push!(targets, LeafTarget(Symbol("ui_parameter_$index"), (index,);
                                  min = minimum(values), max = maximum(values),
                                  label = String(label)))
    end
    isempty(targets) && return Dict{String,Any}[]
    samples = Matrix{Float64}(undef, length(points), length(targets))
    for (row, point) in enumerate(points), (column, target) in enumerate(targets)
        samples[row, column] = Float64(point["parameters"][target.label])
    end
    variants = [VariantResult(
        variant_id = Int(point["variant_id"]),
        objective = point["system_cost"] === nothing ? NaN : Float64(point["system_cost"]),
        co2_price = point["co2_price"] === nothing ? NaN : Float64(point["co2_price"]),
        term_status = uppercase(String(point["term_status"])) in ("OPTIMAL", "LOCALLY_SOLVED") ?
                      "OPTIMAL" : String(point["term_status"]),
    ) for point in points]
    spec = ScenarioSpec(name = "ui_delta", method = :lhs,
                        n_variants = length(points), seed = 0, targets = targets)
    result = ScenarioResult(spec, samples, variants, 0.0)
    cost = delta_sensitivity(result; output = :objective, nboot = nboot,
                             ygrid_length = 512, min_optimal = 20)
    co2 = try
        delta_sensitivity(result; output = :co2_price, nboot = nboot,
                          ygrid_length = 512, min_optimal = 20)
    catch
        DataFrame(target = String[], adjusted_delta = Float64[],
                  conf_low = Float64[], conf_high = Float64[])
    end
    co2_by_target = Dict(String(row.target) => row for row in eachrow(co2))
    rows = Dict{String,Any}[]
    for row in eachrow(cost)
        co2_row = get(co2_by_target, String(row.target), nothing)
        cost_delta = Float64(row.adjusted_delta)
        co2_delta = co2_row === nothing ? nothing : Float64(co2_row.adjusted_delta)
        push!(rows, Dict{String,Any}(
            "label" => String(row.target),
            "method" => "moment_delta",
            "costDelta" => cost_delta,
            "costLow" => Float64(row.conf_low),
            "costHigh" => Float64(row.conf_high),
            "co2Delta" => co2_delta,
            "co2Low" => co2_row === nothing ? nothing : Float64(co2_row.conf_low),
            "co2High" => co2_row === nothing ? nothing : Float64(co2_row.conf_high),
            "influence" => max(cost_delta, co2_delta === nothing ? 0.0 : co2_delta),
        ))
    end
    sort!(rows; by = row -> Float64(row["influence"]), rev = true)
    return rows
end

function _scenario_result(id::AbstractString)
    lock(UI_CAMPAIGNS_LOCK)
    snap = nothing
    try
        snap = get(UI_CAMPAIGNS, String(id), nothing)
        snap === nothing && return Dict{String,Any}(
            "ok" => false,
            "error" => "Unknown campaign id: $id",
        )
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
    if snap["done"] && get(snap["campaign"], "gsa_method", "rank") == "moment_delta" &&
       !haskey(snap, "gsa_rows")
        try
            rows = _scenario_delta_gsa(deepcopy(get(snap, "result_points", Any[])),
                                       copy(get(snap, "parameter_labels", String[])))
            lock(UI_CAMPAIGNS_LOCK)
            try
                snap["gsa_rows"] = rows
            finally
                unlock(UI_CAMPAIGNS_LOCK)
            end
        catch err
            snap["gsa_error"] = sprint(showerror, err)
        end
    end
    return Dict{String,Any}(
        "ok" => true,
        "done" => snap["done"],
        "campaign" => snap["campaign"],
        "workers" => snap["workers"],
        "parameter_labels" => get(snap, "parameter_labels", String[]),
        "scatter" => get(snap, "result_points", Vector{Dict{String,Any}}()),
        "gsa_rows" => get(snap, "gsa_rows", Any[]),
        "gsa_error" => get(snap, "gsa_error", nothing),
        "error" => snap["error"],
    )
end

function _scenario_campaigns()
    lock(UI_CAMPAIGNS_LOCK)
    try
        rows = Dict{String,Any}[]
        for (id, snap) in UI_CAMPAIGNS
            c = get(snap, "campaign", Dict{String,Any}())
            push!(rows, Dict{String,Any}(
                "id" => id,
                "name" => String(get(c, "name", id)),
                "state" => String(get(c, "state", get(c, "status", ""))),
                "stage" => String(get(c, "stage", "")),
                "total" => Int(get(c, "total", 0)),
                "completed" => Int(get(c, "completed", 0)),
                "failed" => Int(get(c, "failed", 0)),
                "started_at" => Float64(get(c, "started_at", 0.0)),
                "done" => Bool(get(snap, "done", false)),
                "result_count" => length(get(snap, "result_points", Any[])),
            ))
        end
        sort!(rows; by = r -> Float64(get(r, "started_at", 0.0)), rev = true)
        return Dict{String,Any}("ok" => true, "campaigns" => rows)
    finally
        unlock(UI_CAMPAIGNS_LOCK)
    end
end