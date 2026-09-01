# =============================================================================
# writers.jl — result table writers
#
# Result-table writers used by single runs and campaign variants.
#
# Output schema convention (long format, one row per indexed value):
#   - Columns: index names + "value" (Float64)
#   - Table/file names follow IESA-Opt 1.0 result naming where possible
#
# Top-level entry points:
#   - `write_duckdb_results(...)` stores all result tables in `results.duckdb`
#   - `write_parquet_results(...)` remains available for legacy/export use
# =============================================================================

using Printf
using Parquet2
using DuckDB
import DBInterface

_no_writer_progress(args...) = nothing
const IESA_RESULTS_DUCKDB_FILE = "results.duckdb"
const _DUCKDB_TABLE_URI_PREFIX = "duckdb://"
const _DUCKDB_WRITE_CONNECTIONS = Dict{String,Any}()
const _DUCKDB_WRITE_CONNECTIONS_LOCK = ReentrantLock()

# ============================================================================
# Generic dispatcher
# ============================================================================

"""
    write_parquet_results(rr, vars, md, out_dir; only=nothing, mode=:fh)

Write all IESA-Opt 1.0-compatible result tables for a single solve to `out_dir`.

Arguments:
  - `rr`     : `RunResult` (provides solve metadata + objective)
  - `vars`   : `AnnualVars` (provides JuMP variable handles)
  - `md`     : `ModelData`
  - `out_dir`: target directory (created if missing)
  - `only`   : optional `Vector{Symbol}` of writer names to run (subset)
  - `mode`   : `:fh` (write `*` files) or `:ts` (write `*_TS` files)

Returns a `Dict{Symbol,String}` mapping writer name → full output path.
"""
function write_parquet_results(rr::RunResult, vars::AnnualVars, md::ModelData,
                                out_dir::AbstractString;
                                only::Union{Nothing,Vector{Symbol}} = nothing,
                                mode::Symbol = :fh,
                                co2_prices::Union{Nothing,AbstractDict} = nothing,
                                activity_prices::Union{Nothing,AbstractDict} = nothing,
                                emission_prices::Union{Nothing,AbstractVector} = nothing,
                                activity_prices_hourly::Union{Nothing,AbstractVector} = nothing,
                                activity_prices_daily::Union{Nothing,AbstractVector} = nothing,
                                progress::Function = _no_writer_progress)
    mkpath(out_dir)
    written = Dict{Symbol,String}()

    # Annual writers (always)
    annual_writers = Dict{Symbol,Function}(
        :tech_use            => write_tech_use_parquet,
        :variable_values     => write_variable_values_parquet,
        :techStock           => write_techStock_parquet,
        :totalCosts          => write_totalCosts_parquet,
        :CO2_price           => write_CO2_price_parquet,
        :cost_breakdown      => write_cost_breakdown_parquet,
        :cluster_map         => write_cluster_map_parquet,
        :tech_meta           => write_tech_meta_parquet,
        :nodes_meta          => write_nodes_meta_parquet,
        :activity_balances   => write_activity_balances_parquet,
        :activities_meta     => write_activities_meta_parquet,
        :activity_prices     => write_activity_prices_parquet,
        :emission_prices     => write_emission_prices_parquet,
        :activity_prices_hourly => write_activity_prices_hourly_parquet,
        :activity_prices_daily  => write_activity_prices_daily_parquet,
    )
    for (name, fn) in annual_writers
        only === nothing || name in only || continue
        path = joinpath(out_dir, string(name) * ".parquet")
        try
            _write_result_with_progress!(written, name, path, progress) do
                if name == :totalCosts
                    return fn(vars, md, path, rr.objective_value)
                elseif name == :CO2_price
                    return fn(vars, md, path; co2_prices = co2_prices)
                elseif name == :activity_prices
                    return fn(vars, md, path; activity_prices = activity_prices)
                elseif name == :emission_prices
                    return fn(vars, md, path; emission_prices = emission_prices)
                elseif name == :activity_prices_hourly
                    return fn(vars, md, path; activity_prices_hourly = activity_prices_hourly)
                elseif name == :activity_prices_daily
                    return fn(vars, md, path; activity_prices_daily = activity_prices_daily)
                else
                    return fn(vars, md, path)
                end
            end
        catch err
            @warn "Writer $name failed" err = err
        end
    end

    # Hourly / TS writers (only if vars have hourly fields populated)
    if mode == :fh && vars.tech_useHourly !== nothing
        path = joinpath(out_dir, "tech_use_h.parquet")
        _write_result_with_progress!(written, :tech_use_h, path, progress) do
            write_tech_useHourly_parquet(vars, md, path; mode = :fh)
        end
        path = joinpath(out_dir, "flexibility_profile_price_h.parquet")
        try
            _write_result_with_progress!(written, :flexibility_profile_price_h, path, progress) do
                write_flexibility_profile_parquet(vars, md, path; mode = :fh,
                    activity_prices_hourly = activity_prices_hourly)
            end
        catch err
            @warn "Writer flexibility_profile_price_h failed" err = err
        end
    elseif mode == :ts && vars.tech_useHourly_TS !== nothing
        path = joinpath(out_dir, "tech_use_TS.parquet")
        _write_result_with_progress!(written, :tech_use_TS, path, progress) do
            write_tech_useHourly_parquet(vars, md, path; mode = :ts)
        end
        path = joinpath(out_dir, "flexibility_profile_price_h.parquet")
        try
            _write_result_with_progress!(written, :flexibility_profile_price_h, path, progress) do
                write_flexibility_profile_parquet(vars, md, path; mode = :ts,
                    activity_prices_hourly = activity_prices_hourly)
            end
        catch err
            @warn "Writer flexibility_profile_price_h failed" err = err
        end
    end

    # Run statistics (always)
    rs_path = joinpath(out_dir, "run_statistics.parquet")
    try
        _write_result_with_progress!(written, :run_statistics, rs_path, progress) do
            write_run_statistics_parquet(rr, rs_path)
        end
    catch err
        @warn "Writer run_statistics failed" err = err
    end

    return written
end

function write_duckdb_results(rr::RunResult, vars::AnnualVars, md::ModelData,
                              out_dir::AbstractString;
                              only::Union{Nothing,Vector{Symbol}} = nothing,
                              mode::Symbol = :fh,
                              reset::Bool = true,
                              co2_prices::Union{Nothing,AbstractDict} = nothing,
                              activity_prices::Union{Nothing,AbstractDict} = nothing,
                              emission_prices::Union{Nothing,AbstractVector} = nothing,
                              activity_prices_hourly::Union{Nothing,AbstractVector} = nothing,
                              activity_prices_daily::Union{Nothing,AbstractVector} = nothing,
                              progress::Function = _no_writer_progress)
    mkpath(out_dir)
    db_path = joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
    reset && _remove_duckdb_database!(db_path)
    written = Dict{Symbol,String}()

    _with_duckdb_write_connection(db_path) do
        annual_writers = Dict{Symbol,Function}(
            :tech_use            => write_tech_use_parquet,
            :variable_values     => write_variable_values_parquet,
            :techStock           => write_techStock_parquet,
            :totalCosts          => write_totalCosts_parquet,
            :CO2_price           => write_CO2_price_parquet,
            :cost_breakdown      => write_cost_breakdown_parquet,
            :cluster_map         => write_cluster_map_parquet,
            :tech_meta           => write_tech_meta_parquet,
            :nodes_meta          => write_nodes_meta_parquet,
            :activity_balances   => write_activity_balances_parquet,
            :activities_meta     => write_activities_meta_parquet,
            :activity_prices     => write_activity_prices_parquet,
            :emission_prices     => write_emission_prices_parquet,
            :activity_prices_hourly => write_activity_prices_hourly_parquet,
            :activity_prices_daily  => write_activity_prices_daily_parquet,
        )
        for (name, fn) in annual_writers
            only === nothing || name in only || continue
            path = _duckdb_table_uri(db_path, string(name))
            try
                _write_result_with_progress!(written, name, path, progress) do
                    if name == :totalCosts
                        return fn(vars, md, path, rr.objective_value)
                    elseif name == :CO2_price
                        return fn(vars, md, path; co2_prices = co2_prices)
                    elseif name == :activity_prices
                        return fn(vars, md, path; activity_prices = activity_prices)
                    elseif name == :emission_prices
                        return fn(vars, md, path; emission_prices = emission_prices)
                    elseif name == :activity_prices_hourly
                        return fn(vars, md, path; activity_prices_hourly = activity_prices_hourly)
                    elseif name == :activity_prices_daily
                        return fn(vars, md, path; activity_prices_daily = activity_prices_daily)
                    else
                        return fn(vars, md, path)
                    end
                end
            catch err
                @warn "Writer $name failed" err = err
            end
        end

        if mode == :fh && vars.tech_useHourly !== nothing
            path = _duckdb_table_uri(db_path, "tech_use_h")
            _write_result_with_progress!(written, :tech_use_h, path, progress) do
                write_tech_useHourly_parquet(vars, md, path; mode = :fh)
            end
            path = _duckdb_table_uri(db_path, "flexibility_profile_price_h")
            try
                _write_result_with_progress!(written, :flexibility_profile_price_h, path, progress) do
                    write_flexibility_profile_parquet(vars, md, path; mode = :fh,
                        activity_prices_hourly = activity_prices_hourly)
                end
            catch err
                @warn "Writer flexibility_profile_price_h failed" err = err
            end
        elseif mode == :ts && vars.tech_useHourly_TS !== nothing
            path = _duckdb_table_uri(db_path, "tech_use_TS")
            _write_result_with_progress!(written, :tech_use_TS, path, progress) do
                write_tech_useHourly_parquet(vars, md, path; mode = :ts)
            end
            path = _duckdb_table_uri(db_path, "flexibility_profile_price_h")
            try
                _write_result_with_progress!(written, :flexibility_profile_price_h, path, progress) do
                    write_flexibility_profile_parquet(vars, md, path; mode = :ts,
                        activity_prices_hourly = activity_prices_hourly)
                end
            catch err
                @warn "Writer flexibility_profile_price_h failed" err = err
            end
        end

        rs_path = _duckdb_table_uri(db_path, "run_statistics")
        try
            _write_result_with_progress!(written, :run_statistics, rs_path, progress) do
                write_run_statistics_parquet(rr, rs_path)
            end
        catch err
            @warn "Writer run_statistics failed" err = err
        end
    end

    return written
end

function _write_result_with_progress!(writer::Function, written::Dict{Symbol,String}, name::Symbol, path::AbstractString, progress::Function)
    progress(name, _storage_display_path(path), :start, 0.0)
    started = time()
    try
        written_path = writer()
        elapsed = round(time() - started, digits = 3)
        written[name] = written_path
        progress(name, _storage_display_path(written_path), :finish, elapsed)
        return written_path
    catch err
        elapsed = round(time() - started, digits = 3)
        progress(name, _storage_display_path(path), :failed, elapsed)
        rethrow()
    end
end

# ============================================================================
# Low-level table writer
# ============================================================================

"""
    _write_table(df::DataFrame, path::AbstractString)

Write `df` to `path`. Only `.parquet` paths are accepted. Symbol columns are
written as strings so the parquet schema is directly comparable with IESA-Opt
1.0 output.
"""
function _write_table(df::DataFrames.DataFrame, path::AbstractString)
    if _is_duckdb_table_uri(path)
        db_path, table_name = _parse_duckdb_table_uri(path)
        _write_duckdb_table(df, db_path, table_name)
        return path
    end

    ext = lowercase(splitext(path)[2])
    if ext == ".parquet"
        parquet_df = _parquet_compatible_table(df)
        Parquet2.writefile(path, parquet_df)
        return path
    end
    error("IESA-Opt result writers only support .parquet output paths: $path")
end

function _write_duckdb_table(df::DataFrames.DataFrame, db_path::AbstractString, table_name::AbstractString)
    mkpath(dirname(db_path))
    active_con = _active_duckdb_write_connection(db_path)
    if active_con !== nothing
        _duckdb_replace_table!(active_con, _parquet_compatible_table(df), table_name)
        return db_path
    end

    con = _duckdb_connect(db_path)
    try
        _duckdb_replace_table!(con, _parquet_compatible_table(df), table_name)
    finally
        DBInterface.close!(con)
        GC.gc()
    end
    return db_path
end

function _duckdb_write_connection_key(db_path::AbstractString)
    return normpath(abspath(String(db_path)))
end

function _active_duckdb_write_connection(db_path::AbstractString)
    key = _duckdb_write_connection_key(db_path)
    lock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    try
        return get(_DUCKDB_WRITE_CONNECTIONS, key, nothing)
    finally
        unlock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    end
end

function _with_duckdb_write_connection(f::Function, db_path::AbstractString; persist::Bool = false)
    key = _duckdb_write_connection_key(db_path)
    if _active_duckdb_write_connection(db_path) !== nothing
        return f()
    end

    con = _duckdb_connect(db_path)
    lock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    try
        _DUCKDB_WRITE_CONNECTIONS[key] = con
    finally
        unlock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    end
    completed = false
    try
        value = f()
        completed = true
        return value
    finally
        if !persist || !completed
            _close_duckdb_write_connection!(db_path)
        end
    end
end

function _duckdb_table_uri(db_path::AbstractString, table_name::AbstractString)
    return _DUCKDB_TABLE_URI_PREFIX * String(db_path) * "#" * String(table_name)
end

function _is_duckdb_table_uri(path::AbstractString)
    return startswith(String(path), _DUCKDB_TABLE_URI_PREFIX)
end

function _parse_duckdb_table_uri(path::AbstractString)
    body = String(path)[lastindex(_DUCKDB_TABLE_URI_PREFIX)+1:end]
    marker = findlast(==('#'), body)
    marker === nothing && error("Invalid DuckDB table URI: $path")
    return body[begin:prevind(body, marker)], body[nextind(body, marker):end]
end

function _storage_display_path(path::AbstractString)
    _is_duckdb_table_uri(path) || return String(path)
    db_path, _ = _parse_duckdb_table_uri(path)
    return db_path
end

function _remove_duckdb_database!(db_path::AbstractString)
    _close_duckdb_write_connection!(db_path)
    for path in (String(db_path), String(db_path) * ".wal")
        last_error = nothing
        for attempt in 1:7
            try
                rm(path; force = true)
                last_error = nothing
                break
            catch err
                last_error = err
                GC.gc()
                attempt < 7 && sleep(0.2 * attempt)
            end
        end
        last_error === nothing || throw(last_error)
    end
    return nothing
end

function _close_duckdb_write_connection!(db_path::AbstractString)
    key = _duckdb_write_connection_key(db_path)
    con = nothing
    lock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    try
        con = pop!(_DUCKDB_WRITE_CONNECTIONS, key, nothing)
    finally
        unlock(_DUCKDB_WRITE_CONNECTIONS_LOCK)
    end
    if con !== nothing
        try
            DBInterface.close!(con)
        catch
        end
        try
            DuckDB.close_database(con)
        catch
        end
        try
            Base.finalize(con)
        catch
        end
        con = nothing
        for _ in 1:3
            GC.gc(true)
        end
    end
    return nothing
end

function _parquet_compatible_table(df::DataFrames.DataFrame)
    symbol_columns = [colname for colname in names(df) if Base.nonmissingtype(eltype(df[!, colname])) <: Symbol]
    isempty(symbol_columns) && return df

    out = copy(df)
    for colname in symbol_columns
        col = out[!, colname]
        converted = Vector{Union{Missing,String}}(undef, length(col))
        for i in eachindex(col)
            value_i = col[i]
            converted[i] = ismissing(value_i) ? missing : String(value_i)
        end
        out[!, colname] = converted
    end
    return out
end

# ============================================================================
# Annual writers
# ============================================================================

# Iterate values of a DenseAxisArray defensively (handles missing fields)
function _collect_axis_array(arr, axes_names::Tuple, period_filter = nothing)
    rows = Tuple[]
    if arr === nothing
        return rows
    end
    for idx in Iterators.product(JuMP.axes(arr)...)
        if period_filter !== nothing
            ps_idx = findfirst(==(:period), axes_names)
            ps_idx !== nothing && !(idx[ps_idx] in period_filter) && continue
        end
        try
            v = value(arr[idx...])
            push!(rows, (idx..., v))
        catch
            # variable might not exist for this combination
        end
    end
    return rows
end

"""
Sum an hourly deviation variable over the year for one technology.

FH sums the hour slices directly; TS weights cluster hours by `clusterHourWeight`.
Returns 0.0 when the technology has no such variable.
"""
function _annual_hourly_delta(fh_container, ts_container, md::ModelData,
                              t::Symbol, ps::Int, members)
    t in members || return 0.0
    if ts_container !== nothing
        total = 0.0
        for hc in md.sets.hours_cluster
            w = get(md.params.clusterHourWeight, hc, 0.0)
            w == 0.0 && continue
            total += w * Float64(value(ts_container[hc, t, ps]))
        end
        return total
    elseif fh_container !== nothing
        total = 0.0
        for h in md.sets.hours
            total += Float64(value(fh_container[h, t, ps]))
        end
        return total
    end
    return 0.0
end

function write_tech_use_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    techs = md.sets.tech_balancers
    periods = md.sets.periods_solve
    chp_members = Set(md.sets.tech_hourlyCHPflex)
    shed_members = Set(md.sets.tech_shedding)
    n = length(techs) * length(periods)
    tech_col = Vector{String}(undef, n)
    period_col = Vector{Int}(undef, n)
    value_col = Vector{Float64}(undef, n)
    tech_names = string.(techs)
    i = 0
    # Net use: tech_use + Σ_h deltaU_CHP + Σ_h deltaS_shed, matching the
    # min/max_techUse and emission-target constraints.
    @inbounds for (tech_i, t) in pairs(techs), ps in periods
        i += 1
        tech_col[i] = tech_names[tech_i]
        period_col[i] = ps
        value_col[i] = Float64(value(vars.tech_use[t, ps])) +
            _annual_hourly_delta(vars.deltaU_CHP, vars.deltaU_CHP_TS, md, t, ps, chp_members) +
            _annual_hourly_delta(vars.deltaS_shed, vars.deltaS_shed_TS, md, t, ps, shed_members)
    end
    df = DataFrames.DataFrame(tech = tech_col, period = period_col, value = value_col; copycols = false)
    return _write_table(df, path)
end

function write_techStock_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    techs = md.sets.technologies
    periods = md.sets.periods_solve
    n = length(techs) * length(periods)
    tech_col = Vector{String}(undef, n)
    period_col = Vector{Int}(undef, n)
    value_col = Vector{Float64}(undef, n)
    tech_names = string.(techs)
    i = 0
    @inbounds for (tech_i, t) in pairs(techs), ps in periods
        i += 1
        tech_col[i] = tech_names[tech_i]
        period_col[i] = ps
        value_col[i] = Float64(value(vars.techStock[t, ps]))
    end
    df = DataFrames.DataFrame(tech = tech_col, period = period_col, value = value_col; copycols = false)
    return _write_table(df, path)
end

function write_totalCosts_parquet(vars::AnnualVars, md::ModelData, path::AbstractString, objective_value::Real)
    ps = length(md.sets.periods_solve) == 1 ? only(md.sets.periods_solve) : 0
    df = DataFrames.DataFrame(period = [ps], value = [Float64(objective_value)]; copycols = false)
    return _write_table(df, path)
end

function write_CO2_price_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                  co2_prices::Union{Nothing,AbstractDict} = nothing)
    if co2_prices === nothing || isempty(co2_prices)
        df = DataFrames.DataFrame(period = Int[], value = Float64[])
        return _write_table(df, path)
    end
    keys_sorted = sort(collect(keys(co2_prices)))
    period_col = Int[Int(k) for k in keys_sorted]
    value_col = Float64[Float64(co2_prices[k]) for k in keys_sorted]
    df = DataFrames.DataFrame(period = period_col, value = value_col; copycols = false)
    return _write_table(df, path)
end

function write_tech_meta_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    p = md.params
    techs = md.sets.technologies
    n = length(techs)
    tech_col      = Vector{String}(undef, n)
    name_col      = Vector{String}(undef, n)
    sector_col    = Vector{String}(undef, n)
    subsector_col = Vector{String}(undef, n)
    category_col  = Vector{String}(undef, n)
    sector_kev_col = Vector{String}(undef, n)
    activity_col  = Vector{String}(undef, n)
    label_col     = Vector{String}(undef, n)
    node_col      = Vector{String}(undef, n)
    process_col   = Vector{String}(undef, n)
    for (i, t) in pairs(techs)
        tech_col[i] = string(t)
        name_col[i] = string(get(p.tech_name, t, ""))
        sector_col[i] = string(get(p.tech_sector, t, Symbol("")))
        subsector_col[i] = string(get(p.tech_subsector, t, Symbol("")))
        category_col[i] = string(get(p.tech_category, t, Symbol("")))
        sector_kev_col[i] = string(get(p.tech_sector_kev, t, Symbol("")))
        a = get(p.activityPer_tech, t, Symbol(""))
        activity_col[i] = string(a)
        label_col[i] = string(get(p.labelPer_act, a, Symbol("")))
        # nodePer_techBal is populated for tech_balancers; nodePer_tech is
        # populated for the full technology set when available. Fall back to
        # nodePer_tech, then to the empty string.
        node_sym = get(p.nodePer_techBal, t, Symbol(""))
        node_sym == Symbol("") && (node_sym = get(p.nodePer_tech, t, Symbol("")))
        node_col[i] = string(node_sym)
        process_col[i] = string(get(p.processType_tech, t, Symbol("")))
    end
    df = DataFrames.DataFrame(
        tech         = tech_col,
        name         = name_col,
        sector       = sector_col,
        subsector    = subsector_col,
        category     = category_col,
        sector_kev   = sector_kev_col,
        activity     = activity_col,
        label        = label_col,
        node         = node_col,
        process_type = process_col;
        copycols = false,
    )
    return _write_table(df, path)
end

function write_nodes_meta_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    s = md.sets
    nodes = s.nodes
    iem = Set(s.nodes_IEM)
    node_col = Vector{String}(undef, length(nodes))
    iem_col  = Vector{Bool}(undef, length(nodes))
    for (i, n) in pairs(nodes)
        node_col[i] = string(n)
        iem_col[i]  = n in iem
    end
    df = DataFrames.DataFrame(node = node_col, is_IEM = iem_col; copycols = false)
    return _write_table(df, path)
end

# activity_prices: annual shadow prices of `balance[<activity>,<period>]` (and
# `balanceFix[…]`, `balanceMatconv[…]`) constraints, captured post-solve and
# passed in through the `activity_prices` dict (keyed by (activity::Symbol,
# period::Int) → price::Float64 in EUR / unit-of-activity). When the dict is
# nothing or empty the table is still created (with zero rows) so downstream
# loaders never have to special-case its absence.
function write_activity_prices_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                        activity_prices::Union{Nothing,AbstractDict} = nothing)
    if activity_prices === nothing || isempty(activity_prices)
        df = DataFrames.DataFrame(activity = String[], period = Int[], price = Float64[], constraint = String[])
        return _write_table(df, path)
    end
    n = length(activity_prices)
    activity_col   = Vector{String}(undef, n)
    period_col     = Vector{Int}(undef, n)
    price_col      = Vector{Float64}(undef, n)
    constraint_col = Vector{String}(undef, n)
    i = 0
    for ((activity, period, ckind), price) in activity_prices
        i += 1
        activity_col[i]   = String(activity)
        period_col[i]     = Int(period)
        price_col[i]      = Float64(price)
        constraint_col[i] = String(ckind)
    end
    df = DataFrames.DataFrame(activity = activity_col, period = period_col, price = price_col, constraint = constraint_col; copycols = false)
    sort!(df, [:period, :activity])
    return _write_table(df, path)
end

# emission_prices: shadow prices for every per-period emission-cap constraint
# (`emTargetAir`, `emTargetBunker`, `emTargetFS`, `emTargetInclScope3*`,
# `emTargetAll`, `emTargetCum`, `co2StorageCum`). Each entry is a NamedTuple-
# like Dict with keys "name", "node", "period", "price" (the absolute shadow
# price in EUR / tCO2eq).
function write_emission_prices_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                        emission_prices::Union{Nothing,AbstractVector} = nothing)
    if emission_prices === nothing || isempty(emission_prices)
        df = DataFrames.DataFrame(name = String[], node = String[], period = Int[], price = Float64[])
        return _write_table(df, path)
    end
    n = length(emission_prices)
    name_col   = Vector{String}(undef, n)
    node_col   = Vector{String}(undef, n)
    period_col = Vector{Int}(undef, n)
    price_col  = Vector{Float64}(undef, n)
    for (i, row) in pairs(emission_prices)
        name_col[i]   = String(get(row, :name, get(row, "name", "")))
        node_col[i]   = String(get(row, :node, get(row, "node", "")))
        period_col[i] = Int(get(row, :period, get(row, "period", 0)))
        price_col[i]  = Float64(get(row, :price, get(row, "price", 0.0)))
    end
    df = DataFrames.DataFrame(name = name_col, node = node_col, period = period_col, price = price_col; copycols = false)
    sort!(df, [:period, :name, :node])
    return _write_table(df, path)
end

# activity_prices_hourly: shadow prices of `balH_TS[<a>,<hc>,<ps>]` (TS mode)
# or `balH[<a>,<h>,<ps>]` (FH mode). The extractor only emits non-trivial
# values, so an empty vector still creates a typed empty table.
# Each entry is a Dict-like object with keys "activity", "period",
# "time_index" (hc for TS, hour for FH), "mode" ("ts" or "fh"), "price".
function write_activity_prices_hourly_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                                activity_prices_hourly::Union{Nothing,AbstractVector} = nothing)
    if activity_prices_hourly === nothing || isempty(activity_prices_hourly)
        df = DataFrames.DataFrame(activity = String[], period = Int[], mode = String[],
                                   time_index = Int[], price = Float64[])
        return _write_table(df, path)
    end
    n = length(activity_prices_hourly)
    act_col   = Vector{String}(undef, n)
    per_col   = Vector{Int}(undef, n)
    mode_col  = Vector{String}(undef, n)
    idx_col   = Vector{Int}(undef, n)
    price_col = Vector{Float64}(undef, n)
    for (i, row) in pairs(activity_prices_hourly)
        act_col[i]   = String(get(row, :activity, get(row, "activity", "")))
        per_col[i]   = Int(get(row, :period, get(row, "period", 0)))
        mode_col[i]  = String(get(row, :mode, get(row, "mode", "")))
        idx_col[i]   = Int(get(row, :time_index, get(row, "time_index", 0)))
        price_col[i] = Float64(get(row, :price, get(row, "price", 0.0)))
    end
    df = DataFrames.DataFrame(activity = act_col, period = per_col, mode = mode_col,
                               time_index = idx_col, price = price_col; copycols = false)
    sort!(df, [:period, :activity, :time_index])
    return _write_table(df, path)
end

# activity_prices_daily: shadow prices of `balD_TS[<a>,<rd>,<ps>]` (TS mode)
# or `balD[<a>,<d>,<ps>]` (FH mode). Schema mirrors the hourly table; the
# `time_index` column is the rep-day in TS mode and the calendar-day in FH.
function write_activity_prices_daily_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                                activity_prices_daily::Union{Nothing,AbstractVector} = nothing)
    if activity_prices_daily === nothing || isempty(activity_prices_daily)
        df = DataFrames.DataFrame(activity = String[], period = Int[], mode = String[],
                                   time_index = Int[], price = Float64[])
        return _write_table(df, path)
    end
    n = length(activity_prices_daily)
    act_col   = Vector{String}(undef, n)
    per_col   = Vector{Int}(undef, n)
    mode_col  = Vector{String}(undef, n)
    idx_col   = Vector{Int}(undef, n)
    price_col = Vector{Float64}(undef, n)
    for (i, row) in pairs(activity_prices_daily)
        act_col[i]   = String(get(row, :activity, get(row, "activity", "")))
        per_col[i]   = Int(get(row, :period, get(row, "period", 0)))
        mode_col[i]  = String(get(row, :mode, get(row, "mode", "")))
        idx_col[i]   = Int(get(row, :time_index, get(row, "time_index", 0)))
        price_col[i] = Float64(get(row, :price, get(row, "price", 0.0)))
    end
    df = DataFrames.DataFrame(activity = act_col, period = per_col, mode = mode_col,
                               time_index = idx_col, price = price_col; copycols = false)
    sort!(df, [:period, :activity, :time_index])
    return _write_table(df, path)
end

function write_activities_meta_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    p = md.params
    acts = md.sets.activities
    n = length(acts)
    act_col  = Vector{String}(undef, n)
    type_col = Vector{String}(undef, n)
    label_col = Vector{String}(undef, n)
    for (i, a) in pairs(acts)
        act_col[i] = string(a)
        type_col[i] = string(get(p.activityType_act, a, Symbol("")))
        label_col[i] = string(get(p.labelPer_act, a, Symbol("")))
    end
    df = DataFrames.DataFrame(activity = act_col, type = type_col, label = label_col; copycols = false)
    return _write_table(df, path)
end

function write_activity_balances_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    p = md.params
    pss = Set(md.sets.periods_solve)
    n_est = length(p.activity_balances)
    tech_col   = Vector{String}(undef, n_est)
    act_col    = Vector{String}(undef, n_est)
    period_col = Vector{Int}(undef, n_est)
    coef_col   = Vector{Float64}(undef, n_est)
    i = 0
    for ((t, a, per), coef) in p.activity_balances
        per in pss || continue
        coef == 0.0 && continue
        i += 1
        tech_col[i]   = string(t)
        act_col[i]    = string(a)
        period_col[i] = per
        coef_col[i]   = Float64(coef)
    end
    resize!(tech_col, i)
    resize!(act_col, i)
    resize!(period_col, i)
    resize!(coef_col, i)
    df = DataFrames.DataFrame(tech = tech_col, activity = act_col, period = period_col, coef = coef_col; copycols = false)
    return _write_table(df, path)
end

function write_cluster_map_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    days = sort(collect(keys(md.params.mapDay_repDay)))
    calendar_day = Vector{String}(undef, length(days))
    rep_day = Vector{Float64}(undef, length(days))
    @inbounds for (i, d) in pairs(days)
        calendar_day[i] = string(d)
        rep_day[i] = Float64(md.params.mapDay_repDay[d])
    end
    df = DataFrames.DataFrame(calendar_day = calendar_day, rep_day = rep_day; copycols = false)
    return _write_table(df, path)
end

function write_variable_values_parquet(vars::AnnualVars, md::ModelData, path::AbstractString; threshold::Float64 = 0.0)
    variable_col = String[]
    index_key_col = String[]
    index_cols = [String[] for _ in 1:5]
    value_col = Float64[]

    for name in fieldnames(AnnualVars)
        container = getfield(vars, name)
        _append_variable_values!(variable_col, index_key_col, index_cols, value_col, String(name), container, threshold)
    end

    df = DataFrames.DataFrame(variable = variable_col, index_key = index_key_col, value = value_col; copycols = false)
    for i in reverse(eachindex(index_cols))
        insertcols!(df, 2, Symbol("index_$(i)") => index_cols[i]; copycols = false)
    end
    return _write_table(df, path)
end

function _append_variable_values!(variable_col::Vector{String}, index_key_col::Vector{String}, index_cols::Vector{Vector{String}}, value_col::Vector{Float64}, name::String, container, threshold::Float64)
    container === nothing && return nothing
    if container isa AbstractDict
        for (key, var) in container
            labels = key isa Tuple ? key : (key,)
            _push_variable_value!(variable_col, index_key_col, index_cols, value_col, name, labels, _solvalue(var), threshold)
        end
        return nothing
    end
    if hasproperty(container, :axes)
        axes = getproperty(container, :axes)
        for ci in CartesianIndices(container)
            labels = Tuple(axes[d][ci[d]] for d in 1:length(ci.I))
            _push_variable_value!(variable_col, index_key_col, index_cols, value_col, name, labels, _solvalue(container[labels...]), threshold)
        end
    end
    return nothing
end

function _push_variable_value!(variable_col::Vector{String}, index_key_col::Vector{String}, index_cols::Vector{Vector{String}}, value_col::Vector{Float64}, name::String, labels::Tuple, val::Float64, threshold::Float64)
    threshold > 0.0 && abs(val) <= threshold && return nothing
    push!(variable_col, name)
    label_strings = string.(labels)
    for i in eachindex(index_cols)
        push!(index_cols[i], i <= length(label_strings) ? label_strings[i] : "")
    end
    push!(index_key_col, join(label_strings, "|"))
    push!(value_col, val)
    return nothing
end

function _solvalue(var)
    try
        return Float64(value(var))
    catch
        return 0.0
    end
end

function _push_cost!(tech_col::Vector{String}, period_col::Vector{Int}, component_col::Vector{String}, cost_col::Vector{Float64}, tech::Symbol, period::Int, component::String, cost::Float64)
    abs(cost) <= 1e-9 && return nothing
    push!(tech_col, string(tech))
    push!(period_col, period)
    push!(component_col, component)
    push!(cost_col, cost)
    return nothing
end

function write_cost_breakdown_parquet(vars::AnnualVars, md::ModelData, path::AbstractString)
    s = md.sets
    p = md.params
    tech_col = String[]
    period_col = Int[]
    component_col = String[]
    cost_col = Float64[]
    lifetime_weight = Dict{Tuple{Symbol,Int},Float64}()
    for ((t_life, _jp, ps_life), w) in p.InvMat_lifeTime
        lifetime_weight[(t_life, ps_life)] = get(lifetime_weight, (t_life, ps_life), 0.0) + w
    end

    for ps in s.periods_solve
        sdf = get(p.social_discount_factor, ps, 1.0)
        sdf == 0.0 && continue
        prev_ps = _prev_period(s.periods_solve, ps)

        for t in s.technologies
            crf = get(p.CRF, t, 0.0)

            capex = 0.0
            for jp in s.periods_solve
                w = get(p.InvMat_lifeTime, (t, jp, ps), 0.0)
                w == 0.0 && continue
                cost = get(p.inv_cost, (t, jp), 0.0)
                (cost == 0.0 || crf == 0.0) && continue
                capex += sdf * w * _solvalue(vars.cap_investments[t, jp]) * cost * crf
            end
            _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "capex", capex)

            retrofit = 0.0
            w_lifetime = get(lifetime_weight, (t, ps), 0.0)
            if w_lifetime != 0.0 && crf != 0.0
                # `vars.retrofitting` is now a sparse Dict keyed by (it,jt,ps)
                # over actual retrofit pairs only. Iterate inbound retrofits to t.
                for it in get(p.retrofit_in_by_tech, t, Symbol[])
                    haskey(vars.retrofitting, (it, t, ps)) || continue
                    rv = _solvalue(vars.retrofitting[(it, t, ps)])
                    rv == 0.0 && continue
                    retrofit += sdf * w_lifetime * rv * crf * get(p.retrofit_cost, (it, t, ps), 0.0)
                end
            end
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "retrofit", retrofit)

            salvage_value = get(p.Salvage_value, t, 0.0)
            inv_cost = get(p.inv_cost, (t, ps), 0.0)
            if salvage_value != 0.0 && inv_cost != 0.0 && crf != 0.0
                ed_delta = _solvalue(vars.eco_decommisioning[t, ps])
                if prev_ps !== nothing
                    ed_delta -= _solvalue(vars.eco_decommisioning[t, prev_ps])
                end
                salvage = -sdf * ed_delta * salvage_value * inv_cost * crf
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "salvage", salvage)
            end

            fom = sdf * _solvalue(vars.techStock[t, ps]) * get(p.fom_cost, (t, ps), 0.0)
            _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "fom", fom)
        end

        for t in s.tech_balancers
            use = _solvalue(vars.tech_use[t, ps])
            vom = sdf * use * get(p.vom_cost, (t, ps), 0.0)
            _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "vom", vom)
        end

        ainEU = :var"Electricity EU"
        if vars.tech_useHourly_TS !== nothing
            tuh = vars.tech_useHourly_TS
            for thh in s.tech_hourlyDispatch
                get(p.tech_category, thh, Symbol("")) == :var"XC Trade" || continue
                is_import = get(p.tech_subsector, thh, Symbol("")) == :var"Power EU"
                is_export = get(p.tech_sector, thh, Symbol("")) == :var"Power EU"
                (is_import || is_export) || continue
                cost = 0.0
                for hc in s.hours_cluster
                    w = get(p.clusterHourWeight, hc, 1.0)
                    price = get(p.interconnectedHourly_prices_cluster, (hc, ainEU, ps), 0.0)
                    cost += sdf * w * _solvalue(tuh[hc, thh, ps]) * price
                end
                is_import && _push_cost!(tech_col, period_col, component_col, cost_col, thh, ps, "Electricity import", cost)
                is_export && _push_cost!(tech_col, period_col, component_col, cost_col, thh, ps, "Electricity export", -cost)
            end
        elseif vars.tech_useHourly !== nothing
            tuh = vars.tech_useHourly
            for thh in s.tech_hourlyDispatch
                get(p.tech_category, thh, Symbol("")) == :var"XC Trade" || continue
                is_import = get(p.tech_subsector, thh, Symbol("")) == :var"Power EU"
                is_export = get(p.tech_sector, thh, Symbol("")) == :var"Power EU"
                (is_import || is_export) || continue
                cost = 0.0
                for h in s.hours
                    price = get(p.interconnectedHourly_prices, (h, ainEU, ps), 0.0)
                    cost += sdf * _solvalue(tuh[h, thh, ps]) * price
                end
                is_import && _push_cost!(tech_col, period_col, component_col, cost_col, thh, ps, "Electricity import", cost)
                is_export && _push_cost!(tech_col, period_col, component_col, cost_col, thh, ps, "Electricity export", -cost)
            end
        end

        if vars.deltaU_CHP_TS !== nothing
            du = vars.deltaU_CHP_TS
            for t in s.tech_hourlyCHPflex
                vc = get(p.vom_cost, (t, ps), 0.0)
                vc == 0.0 && continue
                cost = 0.0
                for hc in s.hours_cluster
                    cost += sdf * get(p.clusterHourWeight, hc, 1.0) * _solvalue(du[hc, t, ps]) * vc
                end
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "flex_vom", cost)
            end
        elseif vars.deltaU_CHP !== nothing
            du = vars.deltaU_CHP
            for t in s.tech_hourlyCHPflex
                vc = get(p.vom_cost, (t, ps), 0.0)
                vc == 0.0 && continue
                cost = sum(sdf * _solvalue(du[h, t, ps]) * vc for h in s.hours)
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "flex_vom", cost)
            end
        end

        if vars.deltaS_shed_TS !== nothing
            ds = vars.deltaS_shed_TS
            for t in s.tech_shedding
                vc = get(p.vom_cost, (t, ps), 0.0)
                pen = get(p.shed_penalty, t, 0.0)
                flex_vom = 0.0
                shed_penalty = 0.0
                for hc in s.hours_cluster
                    w = get(p.clusterHourWeight, hc, 1.0)
                    dv = _solvalue(ds[hc, t, ps])
                    flex_vom += sdf * w * dv * vc
                    shed_penalty += sdf * w * (-dv) * pen
                end
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "flex_vom", flex_vom)
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "shed_penalty", shed_penalty)
            end
        elseif vars.deltaS_shed !== nothing
            ds = vars.deltaS_shed
            for t in s.tech_shedding
                vc = get(p.vom_cost, (t, ps), 0.0)
                pen = get(p.shed_penalty, t, 0.0)
                flex_vom = sum(sdf * _solvalue(ds[h, t, ps]) * vc for h in s.hours)
                shed_penalty = sum(sdf * (-_solvalue(ds[h, t, ps])) * pen for h in s.hours)
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "flex_vom", flex_vom)
                _push_cost!(tech_col, period_col, component_col, cost_col, t, ps, "shed_penalty", shed_penalty)
            end
        end
    end

    df = DataFrames.DataFrame(tech = tech_col, period = period_col, component = component_col, cost_MEUR = cost_col; copycols = false)
    return _write_table(df, path)
end

# ============================================================================
# Hourly / TS writer (combined; dispatch on mode)
# ============================================================================

function write_tech_useHourly_parquet(vars::AnnualVars, md::ModelData, path::AbstractString;
                                       mode::Symbol = :fh)
    if mode == :fh
        tuh = vars.tech_useHourly
        tuh === nothing && error("tech_useHourly is nothing — call build_fh_lp! and solve first")
        hours = md.sets.hours
        techs = md.sets.tech_hourlyDispatch
        periods = md.sets.periods_solve
        n = length(hours) * length(techs) * length(periods)
        hour_col = Vector{Int}(undef, n)
        tech_col = Vector{String}(undef, n)
        period_col = Vector{Int}(undef, n)
        value_col = Vector{Float64}(undef, n)
        tech_names = string.(techs)
        i = 0
        @inbounds for h in hours, (tech_i, th) in pairs(techs), ps in periods
            i += 1
            hour_col[i] = h
            tech_col[i] = tech_names[tech_i]
            period_col[i] = ps
            value_col[i] = Float64(value(tuh[h, th, ps]))
        end
        df = DataFrames.DataFrame(hour = hour_col, tech = tech_col, period = period_col, value = value_col; copycols = false)
        return _write_table(df, path)
    elseif mode == :ts
        tuh = vars.tech_useHourly_TS
        tuh === nothing && error("tech_useHourly_TS is nothing — call build_ts_lp! and solve first")
        hours_cluster = md.sets.hours_cluster
        techs = md.sets.tech_hourlyDispatch
        periods = md.sets.periods_solve
        n = length(hours_cluster) * length(techs) * length(periods)
        hc_col = Vector{Int}(undef, n)
        tech_col = Vector{String}(undef, n)
        period_col = Vector{Int}(undef, n)
        value_col = Vector{Float64}(undef, n)
        tech_names = string.(techs)
        i = 0
        @inbounds for hc in hours_cluster, (tech_i, th) in pairs(techs), ps in periods
            i += 1
            hc_col[i] = hc
            tech_col[i] = tech_names[tech_i]
            period_col[i] = ps
            value_col[i] = Float64(value(tuh[hc, th, ps]))
        end
        df = DataFrames.DataFrame(hc = hc_col, tech = tech_col, period = period_col, value = value_col; copycols = false)
        return _write_table(df, path)
    else
        error("Unknown mode: $mode")
    end
end

# ============================================================================
# Run statistics
# ============================================================================

function write_run_statistics_parquet(rr::RunResult, path::AbstractString)
    df = DataFrames.DataFrame(
        timestamp          = [string(rr.timestamp)],
        mode               = [string(rr.mode)],
        termination_status = [rr.termination_status],
        primal_status      = [rr.primal_status],
        program_status     = [rr.program_status],
        objective          = [rr.objective_value],
        solve_seconds      = [rr.solve_seconds],
        total_seconds      = [rr.total_seconds],
        n_rows             = [rr.n_rows],
        n_cols             = [rr.n_cols],
        n_repDays          = [rr.n_repDays],
        hoursPer_day       = [rr.hoursPer_day],
        clustering         = [string(rr.clustering_approach)],
    )
    return _write_table(df, path)
end

# ============================================================================
# Flexibility profile (reference vs flex demand per tech)
# ============================================================================
# Reproduces the AIMMS `flexibility_profile_price_h` parquet so the same
# downstream flex-report visualisations work on IESA-Opt.jl outputs.
#
# For every tech in `tech_flexible`, the model splits its electricity demand
# into a passive reference profile (tech_use × hourly profile × balance to
# the electricity activity) and a shift `(deltaQ_UP + deltaQ_DW) × balance`
# from the flex variables.  Following the AIMMS sign convention, the column
# semantics are:
#   - referenceProfile_h  : passive demand if no flex shifting (positive ⇒ load)
#   - shiftNet_h          : flex - reference  (positive ⇒ demand INCREASED)
#   - flexProfile_h       : actual demand with flex shifting
#   - electricityPrice_h  : shadow price of the electricity balance constraint
function _empty_flex_profile_df()
    return DataFrames.DataFrame(
        hour               = Int[],
        technology         = String[],
        period             = Int[],
        referenceProfile_h = Float64[],
        shiftNet_h         = Float64[],
        flexProfile_h      = Float64[],
        electricityPrice_h = Float64[],
    )
end

function write_flexibility_profile_parquet(vars::AnnualVars, md::ModelData,
                                            path::AbstractString;
                                            mode::Symbol = :fh,
                                            activity_prices_hourly::Union{Nothing,AbstractVector} = nothing)
    s = md.sets
    p = md.parameters

    flex_techs = s.tech_flexible
    isempty(flex_techs) && return _write_table(_empty_flex_profile_df(), path)

    is_ts = mode === :ts
    dqUP  = is_ts ? vars.deltaQ_UP_TS : vars.deltaQ_UP
    dqDW  = is_ts ? vars.deltaQ_DW_TS : vars.deltaQ_DW
    (dqUP === nothing || dqDW === nothing) && return _write_table(_empty_flex_profile_df(), path)

    hours_axis = is_ts ? s.hours_cluster : s.hours
    profiles   = is_ts ? p.hourly_profiles_cluster : p.hourly_profiles
    isempty(hours_axis) && return _write_table(_empty_flex_profile_df(), path)

    # Map each flex tech to its electricity activity via dQ_hourly indicator.
    # If multiple activities match, prefer the one with the largest |coef|.
    elec_activity = Dict{Symbol,Symbol}()
    elec_coef     = Dict{Symbol,Float64}()
    flex_set = Set(flex_techs)
    for ((tb, ah), v) in p.dQ_hourly
        abs(v) < 1e-9 && continue
        tb in flex_set || continue
        if abs(v) > abs(get(elec_coef, tb, 0.0))
            elec_activity[tb] = ah
            elec_coef[tb]     = v
        end
    end
    isempty(elec_activity) && return _write_table(_empty_flex_profile_df(), path)

    # Lookup table for the dual on balH[ah, h, ps] (or balH_TS[ah, hc, ps]).
    price_lookup = Dict{Tuple{Symbol,Int,Int},Float64}()
    if activity_prices_hourly !== nothing
        for r in activity_prices_hourly
            a  = Symbol(get(r, "activity", ""))
            ps = Int(get(r, "period", 0))
            h  = Int(get(r, "time_index", 0))
            price_lookup[(a, ps, h)] = Float64(get(r, "price", 0.0))
        end
    end

    periods = s.periods_solve
    eps = 1e-9
    n_max = length(flex_techs) * length(periods) * length(hours_axis)
    hour_col  = Vector{Int}();     sizehint!(hour_col, n_max)
    tech_col  = Vector{String}();  sizehint!(tech_col, n_max)
    per_col   = Vector{Int}();     sizehint!(per_col,  n_max)
    ref_col   = Vector{Float64}(); sizehint!(ref_col,  n_max)
    shift_col = Vector{Float64}(); sizehint!(shift_col,n_max)
    flex_col  = Vector{Float64}(); sizehint!(flex_col, n_max)
    price_col = Vector{Float64}(); sizehint!(price_col,n_max)

    for tf in flex_techs
        ah_e = get(elec_activity, tf, Symbol(""))
        ah_e === Symbol("") && continue
        pt    = get(p.profileType_tech, tf, :Flat)
        tname = String(tf)
        for ps in periods
            bal = get(p.activity_balances, (tf, ah_e, ps), 0.0)
            abs(bal) < eps && continue
            tu_val = Float64(value(vars.tech_use[tf, ps]))
            for h in hours_axis
                prof = get(profiles, (h, pt), 0.0)
                # AIMMS sign convention: -tu × prof × bal so load reads
                # positive when bal<0 (input/consumption activities).
                ref = -tu_val * prof * bal
                d_up = Float64(value(dqUP[h, tf, ps]))
                d_dw = Float64(value(dqDW[h, tf, ps]))
                shift = -(d_up + d_dw) * bal
                flex_v = ref + shift
                if abs(ref) < eps && abs(shift) < eps && abs(flex_v) < eps
                    continue
                end
                price = get(price_lookup, (ah_e, ps, Int(h)), 0.0)
                push!(hour_col,  Int(h))
                push!(tech_col,  tname)
                push!(per_col,   Int(ps))
                push!(ref_col,   ref)
                push!(shift_col, shift)
                push!(flex_col,  flex_v)
                push!(price_col, price)
            end
        end
    end

    df = DataFrames.DataFrame(
        hour               = hour_col,
        technology         = tech_col,
        period             = per_col,
        referenceProfile_h = ref_col,
        shiftNet_h         = shift_col,
        flexProfile_h      = flex_col,
        electricityPrice_h = price_col;
        copycols = false,
    )
    return _write_table(df, path)
end
