using Distributed

Base.@kwdef struct ProductionCampaignSummary
    database_path::String
    total::Int
    previously_completed::Int
    attempted::Int
    optimal::Int
    failed::Int
    runtime_seconds::Float64
end

mutable struct _ProductionCampaignStore
    path::String
    database::DuckDB.DB
end

mutable struct _SystemCpuSampler
    idle::UInt64
    kernel::UInt64
    user::UInt64
end

function _system_cpu_times()
    Sys.iswindows() || return nothing
    idle = Ref{UInt64}(0)
    kernel = Ref{UInt64}(0)
    user = Ref{UInt64}(0)
    success = ccall((:GetSystemTimes, "kernel32"), stdcall, Int32,
                    (Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}), idle, kernel, user)
    success == 0 && return nothing
    return (idle[], kernel[], user[])
end

function _system_cpu_sampler()
    times = _system_cpu_times()
    times === nothing && return nothing
    return _SystemCpuSampler(times...)
end

function _sample_system_cpu!(sampler::_SystemCpuSampler)
    times = _system_cpu_times()
    times === nothing && return NaN
    idle, kernel, user = times
    idle_delta = idle - sampler.idle
    total_delta = (kernel - sampler.kernel) + (user - sampler.user)
    sampler.idle, sampler.kernel, sampler.user = times
    total_delta == 0 && return NaN
    return clamp(100 * (1 - idle_delta / total_delta), 0.0, 100.0)
end

function _production_solver_phases(active::AbstractDict, output_dir::AbstractString)
    barrier = 0
    crossover = 0
    for (_, (variant_id, _)) in active
        path = joinpath(output_dir, "solver_logs",
            "variant_$(lpad(string(variant_id), 6, '0')).log")
        if isfile(path)
            text = open(path, "r") do io
                size = filesize(path)
                seek(io, max(0, size - 131_072))
                read(io, String)
            end
            if occursin("Barrier solved model", text)
                crossover += 1
                continue
            end
        end
        barrier += 1
    end
    return (barrier = barrier, crossover = crossover)
end

function _production_worker_capacity(barrier::Int, crossover::Int,
                                     threads_per_worker::Int;
                                     core_capacity::Int = Sys.CPU_THREADS)
    effective_cores = barrier * threads_per_worker + crossover
    available_cores = max(0, core_capacity - effective_cores)
    return (effective_cores = effective_cores,
            additional_workers = fld(available_cores, threads_per_worker))
end

function _print_campaign_progress(; state::AbstractString, completed::Int, total::Int,
                                  active::AbstractDict = Dict(), started_at::Real = time(),
                                  counts::AbstractDict = Dict{Symbol,Int}(),
                                  current_workers::Int = 0, max_workers::Int = 0,
                                  cpu_load::Real = NaN,
                                  barrier_workers::Int = 0, crossover_workers::Int = 0,
                                  message::AbstractString = "")
    longest_active = isempty(active) ? 0.0 :
        maximum(time() - item[2] for item in values(active))
    percent = total == 0 ? 100.0 : 100completed / total
    print(stdout, "[campaign] state=", state,
          " completed=", completed, "/", total,
          " (", round(percent; digits = 1), "%)",
          " active=", length(active),
          " optimal=", get(counts, :optimal, 0),
          " infeasible=", get(counts, :infeasible, 0),
          " timeout=", get(counts, :timeout, 0),
          " errors=", get(counts, :error, 0),
          " longest=", round(longest_active; digits = 1), "s",
          " elapsed=", round(time() - started_at; digits = 1), "s")
    max_workers > 0 && print(stdout, " workers=", current_workers, "/", max_workers)
    isfinite(cpu_load) && print(stdout, " cpu=", round(cpu_load; digits = 1), "%")
    (barrier_workers > 0 || crossover_workers > 0) &&
        print(stdout, " barrier=", barrier_workers, " crossover=", crossover_workers)
    isempty(message) || print(stdout, " message=", message)
    println(stdout)
    flush(stdout)
    return nothing
end

function _write_campaign_status(output_dir::AbstractString; state::AbstractString,
                                completed::Int, total::Int,
                                active::AbstractDict = Dict(),
                                counts::AbstractDict = Dict{Symbol,Int}(),
                                current_workers::Int = 0, max_workers::Int = 0,
                                cpu_load::Real = NaN,
                                barrier_workers::Int = 0, crossover_workers::Int = 0,
                                run_started_at::Union{Nothing,DateTime} = nothing,
                                message::AbstractString = "")
    path = joinpath(output_dir, "campaign_status.json")
    temporary = path * ".tmp"
    payload = Dict(
        "state" => String(state),
        "completed" => completed,
        "total" => total,
        "remaining" => total - completed,
        "optimal" => get(counts, :optimal, 0),
        "infeasible" => get(counts, :infeasible, 0),
        "timeout" => get(counts, :timeout, 0),
        "errors" => get(counts, :error, 0),
        "current_workers" => current_workers,
        "max_workers" => max_workers,
        "cpu_load_percent" => isfinite(cpu_load) ? round(cpu_load; digits = 2) : nothing,
        "barrier_workers" => barrier_workers,
        "crossover_workers" => crossover_workers,
        "active" => Dict(string(worker) => Dict(
            "variant_id" => item[1],
            "elapsed_seconds" => round(time() - item[2]; digits = 2),
        ) for (worker, item) in active),
        "message" => String(message),
        "coordinator_pid" => getpid(),
        "run_started_at" => run_started_at === nothing ? nothing : string(run_started_at),
        "updated_at" => string(now()),
    )
    write(temporary, JSON3.write(payload))
    mv(temporary, path; force = true)
    return path
end

function _campaign_result_category(status::AbstractString)
    status == "OPTIMAL" && return :optimal
    occursin("INFEASIBLE", status) && return :infeasible
    occursin("TIME", status) && return :timeout
    return :error
end

function _production_result_counts(store::_ProductionCampaignStore)
    counts = Dict(:optimal => 0, :infeasible => 0, :timeout => 0, :error => 0)
    table = _production_query(store.database,
        "SELECT term_status, count(*) AS n FROM results GROUP BY term_status")
    for row in eachrow(table)
        counts[_campaign_result_category(String(row.term_status))] += Int(row.n)
    end
    return counts
end

function _start_production_feeder(task_channel, pending, samples)
    return @async begin
        for variant_id in pending
            put!(task_channel,
                 (variant_id, collect(samples.values[variant_id, :])))
        end
    end
end

function _production_query(database, sql::AbstractString, parameters = nothing)
    result = parameters === nothing ? DBInterface.execute(database, sql) :
             DBInterface.execute(database, sql, parameters)
    try
        return DataFrame(result)
    finally
        try
            DBInterface.close!(result)
        catch
        end
    end
end

function _production_execute(database, sql::AbstractString, parameters = nothing)
    result = parameters === nothing ? DBInterface.execute(database, sql) :
             DBInterface.execute(database, sql, parameters)
    try
        return nothing
    finally
        try
            DBInterface.close!(result)
        catch
        end
    end
end

function _production_metadata(bundle::CampaignBundle, spec::CampaignSpec,
                              samples::SampleMatrix, solver::Symbol, mode::Symbol,
                              representative_days::Union{Nothing,Int},
                              solver_attrs::AbstractDict,
                              variant_timeout_seconds::Real,
                              worker_timeout_seconds::Real,
                              hours_per_day::Int,
                              store_full_outputs::Bool)
    crossover = get(solver_attrs, solver === :gurobi ? "Crossover" : "run_crossover",
                    solver === :gurobi ? 0 : "off")
    solve_method = crossover in (-1, "on") ? "barrier_crossover" : "barrier"
    return Dict(
        "schema_version" => "2",
        "campaign_name" => spec.name,
        "method" => String(spec.method),
        "seed" => string(spec.seed),
        "n_evaluations" => string(size(samples.values, 1)),
        "n_parameters" => string(size(samples.values, 2)),
        "input_workbook" => bundle.input_workbook,
        "input_sha256" => bundle.input_sha256,
        "config_sha256" => bytes2hex(SHA.sha256(read(bundle.config_path))),
        "solver" => String(solver),
        "mode" => String(mode),
        "representative_days" => representative_days === nothing ? "n/a" : string(representative_days),
        "hours_per_day" => string(hours_per_day),
        "solve_method" => solve_method,
        "solver_settings" => join(("$key=$(solver_attrs[key])" for key in sort!(collect(keys(solver_attrs)))), ";"),
        "variant_timeout_seconds" => string(Float64(variant_timeout_seconds)),
        "worker_timeout_seconds" => string(Float64(worker_timeout_seconds)),
        "store_full_outputs" => string(store_full_outputs),
        "julia_version" => string(VERSION),
    )
end

function _solver_settings_without_timeout(settings::AbstractString)
    parts = filter(part -> !startswith(part, "TimeLimit="), split(settings, ';'))
    return join(parts, ';')
end

function _open_production_store(output_dir::AbstractString, bundle::CampaignBundle,
                                spec::CampaignSpec, samples::SampleMatrix;
                                solver::Symbol, mode::Symbol,
                                representative_days::Union{Nothing,Int} = nothing,
                                solver_attrs::AbstractDict = Dict{String,Any}(),
                                variant_timeout_seconds::Real = Inf,
                                worker_timeout_seconds::Real = variant_timeout_seconds,
                                hours_per_day::Int = 24,
                                store_full_outputs::Bool = false,
                                overwrite::Bool)
    mkpath(output_dir)
    path = abspath(joinpath(output_dir, "campaign.duckdb"))
    if overwrite
        isfile(path) && rm(path; force = true)
        isfile(path * ".wal") && rm(path * ".wal"; force = true)
    end
    database = DuckDB.DB(path)
    store = _ProductionCampaignStore(path, database)
    try
        _production_execute(database, "CREATE TABLE IF NOT EXISTS metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
        _production_execute(database, "CREATE TABLE IF NOT EXISTS parameters (parameter_id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, minimum DOUBLE, maximum DOUBLE)")
        _production_execute(database, "ALTER TABLE parameters ADD COLUMN IF NOT EXISTS minimum DOUBLE")
        _production_execute(database, "ALTER TABLE parameters ADD COLUMN IF NOT EXISTS maximum DOUBLE")
        _production_execute(database, "CREATE TABLE IF NOT EXISTS samples (variant_id INTEGER, parameter_id INTEGER, value DOUBLE, PRIMARY KEY (variant_id, parameter_id))")
        _production_execute(database, """
            CREATE TABLE IF NOT EXISTS results (
                variant_id INTEGER PRIMARY KEY, objective DOUBLE, co2_price DOUBLE,
                term_status VARCHAR, primal_status VARCHAR, worker_pid INTEGER,
                build_seconds DOUBLE, apply_seconds DOUBLE, solve_seconds DOUBLE,
                output_seconds DOUBLE, output_path VARCHAR,
                attempts INTEGER, error VARCHAR, completed_at TIMESTAMP)
        """)
            _production_execute(database, "ALTER TABLE results ADD COLUMN IF NOT EXISTS output_seconds DOUBLE")
            _production_execute(database, "ALTER TABLE results ADD COLUMN IF NOT EXISTS output_path VARCHAR")
        expected = _production_metadata(bundle, spec, samples, solver, mode,
                representative_days, solver_attrs, variant_timeout_seconds,
                worker_timeout_seconds, hours_per_day, store_full_outputs)
        existing = _production_query(database, "SELECT key, value FROM metadata")
        if nrow(existing) > 0
            actual = Dict(String(row.key) => String(row.value) for row in eachrow(existing))
            for (key, value) in expected
                stored = get(actual, key, nothing)
                stored == value && continue
                timeout_increase = key in ("variant_timeout_seconds", "worker_timeout_seconds") &&
                    stored !== nothing && parse(Float64, value) >= parse(Float64, stored)
                solver_timeout_only = key == "solver_settings" && stored !== nothing &&
                    _solver_settings_without_timeout(stored) ==
                    _solver_settings_without_timeout(value)
                if timeout_increase || solver_timeout_only
                    _production_execute(database,
                        "UPDATE metadata SET value = ? WHERE key = ?", [value, key])
                    continue
                end
                throw(ArgumentError(
                    "Cannot resume campaign: metadata '$key' differs (stored=$stored, requested=$value)."))
            end
        else
            _production_execute(database, "BEGIN TRANSACTION")
            try
                for (key, value) in expected
                    _production_execute(database, "INSERT INTO metadata VALUES (?, ?)", [key, value])
                end
                bounds = parameter_bounds(spec)
                for (parameter_id, name) in pairs(samples.parameters)
                    minimum, maximum = bounds[parameter_id]
                    _production_execute(database, "INSERT INTO parameters VALUES (?, ?, ?, ?)",
                                        [parameter_id, name, minimum, maximum])
                end
                n_evaluations, n_parameters = size(samples.values)
                sample_table = DataFrame(
                    variant_id = repeat(collect(1:n_evaluations); inner = n_parameters),
                    parameter_id = repeat(collect(1:n_parameters); outer = n_evaluations),
                    value = vec(permutedims(samples.values)))
                DuckDB.appendDataFrame(sample_table, database, "samples")
                _production_execute(database, "COMMIT")
            catch
                _production_execute(database, "ROLLBACK")
                rethrow()
            end
        end
        stored_parameters = _production_query(database,
            "SELECT parameter_id, minimum, maximum FROM parameters ORDER BY parameter_id")
        if any(ismissing, stored_parameters.minimum) || any(ismissing, stored_parameters.maximum)
            bounds = parameter_bounds(spec)
            for (parameter_id, (minimum, maximum)) in pairs(bounds)
                _production_execute(database,
                    "UPDATE parameters SET minimum = ?, maximum = ? WHERE parameter_id = ?",
                    [minimum, maximum, parameter_id])
            end
        end
        return store
    catch
        DBInterface.close!(database)
        rethrow()
    end
end

function _close_production_store!(store::_ProductionCampaignStore)
    try
        _production_execute(store.database, "CHECKPOINT")
    finally
        DBInterface.close!(store.database)
        finalize(store.database)
        GC.gc(true)
    end
    return nothing
end

function _production_completed_ids(store::_ProductionCampaignStore)
    table = _production_query(store.database,
        "SELECT variant_id FROM results WHERE term_status = 'OPTIMAL'")
    return Set(Int(row.variant_id) for row in eachrow(table))
end

function _write_production_result!(store::_ProductionCampaignStore,
                                   result::VariantResult, attempts::Int)
    _production_execute(store.database, "BEGIN TRANSACTION")
    try
        _production_execute(store.database, """
            INSERT OR REPLACE INTO results
                (variant_id, objective, co2_price, term_status, primal_status,
                 worker_pid, build_seconds, apply_seconds, solve_seconds,
                 output_seconds, output_path, attempts, error, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
        """, [result.variant_id, result.objective, result.co2_price,
               result.term_status, result.primal_status, result.worker_pid,
               result.build_seconds, result.apply_seconds, result.solve_seconds,
               result.output_seconds,
               result.output_path === nothing ? missing : result.output_path,
               attempts, result.error === nothing ? missing : result.error])
        _production_execute(store.database, "COMMIT")
    catch
        _production_execute(store.database, "ROLLBACK")
        rethrow()
    end
    return nothing
end

function _weighted_quantile(values::Vector{Float64}, weights::Vector{Float64}, probability::Float64)
    order = sortperm(values)
    cumulative = cumsum(weights[order])
    return values[order[searchsortedfirst(cumulative, probability * cumulative[end])]]
end

function _production_price_summary(model::JuMP.Model, md::ModelData, mode::Symbol)
    rows = NamedTuple[]
    specifications = mode === :ts ? (
        (resolution = "hourly", activities = md.sets.activities_hour,
         indices = md.sets.hours_cluster, base = "balH_TS"),
        (resolution = "daily", activities = md.sets.activities_day,
         indices = md.sets.repDays, base = "balD_TS"),
    ) : (
        (resolution = "hourly", activities = md.sets.activities_hour,
         indices = md.sets.hours, base = "balH"),
        (resolution = "daily", activities = md.sets.activities_day,
         indices = md.sets.days, base = "balD"),
    )
    for specification in specifications, period in md.sets.periods_solve,
        activity in specification.activities
        shadow_values = Float64[]
        weights = Float64[]
        for index in specification.indices
            constraint = constraint_by_name(
                model, "$(specification.base)[$activity,$index,$period]")
            constraint === nothing && continue
            shadow = try
                Float64(shadow_price(constraint))
            catch
                NaN
            end
            isfinite(shadow) || continue
            weight = specification.resolution == "hourly" ?
                (mode === :ts ? get(md.params.clusterHourWeight, index, 0.0) :
                 get(md.params.slice_width_hours, index, 1.0)) : 1.0
            weight > 0 || continue
            push!(shadow_values, shadow)
            push!(weights, Float64(weight))
        end
        isempty(shadow_values) && continue
        economic_values = .-shadow_values
        total_weight = sum(weights)
        economic_mean = sum(economic_values .* weights) / total_weight
        economic_variance = sum(weights .* (economic_values .- economic_mean) .^ 2) / total_weight
        shadow_mean = sum(shadow_values .* weights) / total_weight
        push!(rows, (
            activity = string(activity), period = Int(period),
            resolution = specification.resolution, observations = length(shadow_values),
            weight = total_weight, shadow_mean = shadow_mean,
            shadow_min = minimum(shadow_values), shadow_max = maximum(shadow_values),
            economic_mean = economic_mean, economic_std = sqrt(economic_variance),
            economic_variance = economic_variance,
            economic_min = minimum(economic_values), economic_max = maximum(economic_values),
            economic_p05 = _weighted_quantile(economic_values, weights, 0.05),
            economic_median = _weighted_quantile(economic_values, weights, 0.50),
            economic_p95 = _weighted_quantile(economic_values, weights, 0.95),
            negative_weight = sum(weights[economic_values .< 0]),
            over_100_weight = sum(weights[economic_values .> 100]),
        ))
    end
    return DataFrame(rows)
end

function _production_system_kpis(vars::AnnualVars, md::ModelData)
    rows = NamedTuple[]
    flexible = Set(md.sets.tech_flexible)
    groups = Dict(
        "hydrogen" => r"hydrogen|h2",
        "solar" => r"solar|photovoltaic|\bpv\b",
        "wind" => r"wind",
        "nuclear" => r"nuclear|\bsmr\b",
        "battery" => r"battery|li-ion",
        "battery grid" => r"battery|li-ion",
        "battery vehicles" => r"battery|li-ion",
    )
    # For these groups the Primary-category rows are energy-accounting entries
    # that duplicate the conversion technology's output (e.g. `Wind Energy`
    # alongside `Offshore Wind`), or a fuel import cap (`Imported Uranium`).
    exclude_primary = Set(["solar", "wind", "nuclear"])
    sector_filter = Dict("battery grid" => Symbol("Power NL"),
                         "battery vehicles" => :Transport)
    for period in md.sets.periods_solve
        for (group, pattern) in groups
            technologies = Symbol[]
            required_sector = get(sector_filter, group, nothing)
            for technology in md.sets.technologies
                activity = get(md.params.activityPer_tech, technology, Symbol(""))
                description = lowercase(join((string(technology),
                    string(get(md.params.tech_name, technology, "")), string(activity),
                    string(get(md.params.labelPer_act, activity, ""))), " "))
                occursin(pattern, description) || continue
                group in exclude_primary &&
                    get(md.params.tech_category, technology, Symbol("")) == :Primary && continue
                required_sector === nothing ||
                    get(md.params.tech_sector, technology, Symbol("")) == required_sector || continue
                push!(technologies, technology)
            end
            capacity = sum(value(vars.techStock[technology, period]) for technology in technologies;
                           init = 0.0)
            use_technologies = intersect(Set(technologies), Set(md.sets.tech_balancers))
            annual_use = sum(value(vars.tech_use[technology, period]) for technology in use_technologies;
                             init = 0.0)
            production = sum(
                value(vars.tech_use[technology, period]) * coefficient
                for ((technology, _, balance_period), coefficient) in md.params.activity_balances
                if balance_period == period && technology in technologies && coefficient > 0 &&
                   technology in md.sets.tech_balancers; init = 0.0)
            investment = sum(value(vars.cap_investments[technology, period]) for technology in technologies;
                             init = 0.0)
            push!(rows, (category = group, metric = "capacity", period = Int(period),
                         value = Float64(capacity), unit = "model_capacity", technologies = length(technologies)))
            push!(rows, (category = group, metric = "annual_use", period = Int(period),
                         value = Float64(annual_use), unit = "model_activity", technologies = length(use_technologies)))
            push!(rows, (category = group, metric = "production", period = Int(period),
                         value = Float64(production), unit = "model_activity", technologies = length(use_technologies)))
            push!(rows, (category = group, metric = "investment", period = Int(period),
                         value = Float64(investment), unit = "model_capacity", technologies = length(technologies)))
        end
        hydrogen_supply = sum(
            value(vars.tech_use[technology, period]) * coefficient
            for ((technology, activity, balance_period), coefficient) in md.params.activity_balances
            if balance_period == period && coefficient > 0 &&
               technology in md.sets.tech_balancers &&
               occursin(r"hydrogen|\bh2\b", lowercase(string(activity)));
            init = 0.0)
        push!(rows, (category = "hydrogen", metric = "supply", period = Int(period),
                     value = Float64(hydrogen_supply), unit = "model_activity", technologies = 0))
        flex_types = unique(get(md.params.flexibilityType_tech, technology, Symbol("unspecified"))
                            for technology in flexible)
        for flex_type in flex_types
            technologies = [technology for technology in flexible
                            if get(md.params.flexibilityType_tech, technology, Symbol("unspecified")) == flex_type]
            capacity = sum(value(vars.techStock[technology, period]) for technology in technologies;
                           init = 0.0)
            annual_use = sum(value(vars.tech_use[technology, period])
                             for technology in intersect(Set(technologies), Set(md.sets.tech_balancers));
                             init = 0.0)
            flex_capacity = sum(get(md.params.flex_capacity, (technology, period), 0.0)
                                for technology in technologies; init = 0.0)
            category = "flexibility:" * string(flex_type)
            push!(rows, (category = category, metric = "capacity", period = Int(period),
                         value = Float64(capacity), unit = "model_capacity", technologies = length(technologies)))
            push!(rows, (category = category, metric = "annual_use", period = Int(period),
                         value = Float64(annual_use), unit = "model_activity", technologies = length(technologies)))
            push!(rows, (category = category, metric = "flex_capacity", period = Int(period),
                         value = Float64(flex_capacity), unit = "model_capacity", technologies = length(technologies)))
        end
    end
    return DataFrame(rows)
end

function _write_production_variant_outputs(model::JuMP.Model, md::ModelData,
                                           variant_id::Int, output_root::AbstractString,
                                           mode::Symbol, build_seconds::Real,
                                           apply_seconds::Real, solve_seconds::Real,
                                           solver_attrs::AbstractDict)
    vars = model.ext[:iesa_vars]
    variant_name = lpad(string(variant_id), 6, '0')
    variants_dir = joinpath(output_root, "variants")
    mkpath(variants_dir)
    final_dir = joinpath(variants_dir, variant_name)
    completion_marker = joinpath(final_dir, "COMPLETED")
    isfile(completion_marker) && return relpath(final_dir, output_root)
    mkpath(final_dir)
    term = string(termination_status(model))
    rr = RunResult(
        final_dir, now(), mode, term, string(primal_status(model)), term,
        objective_value(model), solve_seconds,
        apply_seconds + build_seconds + solve_seconds,
        num_constraints(model; count_variable_in_set_constraints = false),
        num_variables(model), 0, 0, 0, Dict{String,Any}(solver_attrs),
        "1108 SSP", mode === :ts ? md.params.n_repDays : 365,
        md.params.hoursPer_day, md.params.clustering_approach)
    activity_prices = _extract_activity_prices(model, md)
    emission_prices = _extract_emission_prices(model, md)
    co2_prices = _extract_co2_prices(model, md)
    write_parquet_results(rr, vars, md, final_dir; mode = :annual,
        only = [:totalCosts, :tech_use, :techStock, :cost_breakdown, :tech_meta,
                :activity_prices, :emission_prices, :CO2_price],
        activity_prices = activity_prices, emission_prices = emission_prices,
        co2_prices = co2_prices)
    investments = DataFrame(
        tech = string.(md.sets.technologies),
        period = fill(only(md.sets.periods_solve), length(md.sets.technologies)),
        value = [Float64(value(vars.cap_investments[technology, only(md.sets.periods_solve)]))
                 for technology in md.sets.technologies])
    price_summary = _production_price_summary(model, md, mode)
    system_kpis = _production_system_kpis(vars, md)
    _write_table(investments, joinpath(final_dir, "cap_investments.parquet"))
    _write_table(price_summary, joinpath(final_dir, "activity_price_summary.parquet"))
    _write_table(system_kpis, joinpath(final_dir, "system_kpis.parquet"))
    marker_temporary = completion_marker * ".tmp"
    write(marker_temporary, "completed_at=$(now())\n")
    mv(marker_temporary, completion_marker; force = true)
    return relpath(final_dir, output_root)
end

function _solve_1108_production_variant(context::Legacy1108Context,
                                        sampled_values::Vector{Float64}, variant_id::Int;
                                        solver::Symbol, threads::Int, mode::Symbol,
                                        representative_days::Union{Nothing,Int},
                                        periods::Vector{Int}, solver_attrs::AbstractDict,
                                        variant_timeout_seconds::Real,
                                        max_attempts::Int,
                                        fallback_solver_attrs::Union{Nothing,AbstractDict} = nothing,
                                        hours_per_day::Int = 24,
                                        output_dir::Union{Nothing,AbstractString} = nothing,
                                        store_full_outputs::Bool = false)
    last_result = VariantResult(variant_id = variant_id, term_status = "ERROR",
                                error = "Variant was not attempted.")
    for attempt in 1:max_attempts
        # Retries fall back to crossover-enabled settings; barrier-only can stop
        # short of tolerance or hit the time limit on a minority of variants.
        attempt_attrs = (attempt == 1 || fallback_solver_attrs === nothing) ? solver_attrs :
            merge(Dict{String,Any}(solver_attrs), Dict{String,Any}(fallback_solver_attrs))
        try
            md = nothing
            model = nothing
            apply_seconds = @elapsed begin
                clustered_template = nothing
                if mode === :ts
                    representative_days === nothing && error(
                        "TS production campaigns require representative_days for cluster-cache selection.")
                    weather_parameter_id = findfirst(==("Weather Conditions"),
                                                     context.parameter_names)
                    weather_parameter_id === nothing && error(
                        "Weather Conditions is missing from the campaign parameter order.")
                    weather_state = clamp(floor(Int, sampled_values[weather_parameter_id]), 1, 7)
                    clustered_template = load_1108_cluster_cache(
                        context, representative_days, weather_state)
                end
                md = prepare_1108_variant(context, sampled_values;
                                          clustered_template = clustered_template)
                md.sets.periods_solve = copy(periods)
                representative_days === nothing ||
                    (md.params.n_repDays = representative_days)
                mode === :fh && (md.params.hoursPer_day = hours_per_day)
                derive_sets!(md)
                compute_derived_params!(md)
                mode === :ts && compute_flex_TS_helpers!(md)
            end
            build_seconds = @elapsed model = _build_campaign_model(md;
                solver = solver, threads = threads, mode = mode,
                attrs_override = if solver === :gurobi && output_dir !== nothing
                    log_dir = joinpath(output_dir, "solver_logs")
                    mkpath(log_dir)
                    log_path = joinpath(log_dir,
                        "variant_$(lpad(string(variant_id), 6, '0')).log")
                    isfile(log_path) && rm(log_path; force = true)
                    merge(Dict{String,Any}(attempt_attrs), Dict{String,Any}(
                        "OutputFlag" => 1,
                        "LogToConsole" => 0,
                        "LogFile" => log_path,
                    ))
                else
                    attempt_attrs
                end)
            remaining_seconds = variant_timeout_seconds - apply_seconds - build_seconds
            remaining_seconds > 0 || error(
                "Variant runtime budget exhausted before solve ($(round(apply_seconds + build_seconds; digits = 3)) s).")
            if isfinite(remaining_seconds)
                timeout_attribute = solver === :gurobi ? "TimeLimit" : "time_limit"
                set_optimizer_attribute(model, timeout_attribute, remaining_seconds)
            end
            solve_seconds = @elapsed optimize!(model)
            term = string(termination_status(model))
            primal = string(primal_status(model))
            objective = term == "OPTIMAL" ? objective_value(model) : NaN
            co2_price = term == "OPTIMAL" ? _variant_co2_price(model, md) : NaN
            output_seconds = 0.0
            output_path = nothing
            if term == "OPTIMAL" && store_full_outputs
                output_dir === nothing && error("Full scenario outputs require an output directory.")
                output_seconds = @elapsed output_path = _write_production_variant_outputs(
                    model, md, variant_id, output_dir, mode, build_seconds,
                    apply_seconds, solve_seconds, attempt_attrs)
            end
            result = VariantResult(
                variant_id = variant_id, leaf_values = sampled_values,
                objective = objective, co2_price = co2_price,
                term_status = term, primal_status = primal,
                build_seconds = build_seconds, apply_seconds = apply_seconds,
                solve_seconds = solve_seconds, output_seconds = output_seconds,
                output_path = output_path, worker_pid = Distributed.myid())
            if term != "OPTIMAL" && attempt < max_attempts &&
                    _campaign_result_category(term) !== :infeasible
                last_result = result
                continue
            end
            return result, attempt
        catch err
            last_result = VariantResult(
                variant_id = variant_id, leaf_values = sampled_values,
                term_status = "ERROR", worker_pid = Distributed.myid(),
                error = sprint(showerror, err, catch_backtrace()))
        end
    end
    return last_result, max_attempts
end

function _solve_1108_production_variant_from_config(
        config_path::AbstractString, sampled_values::Vector{Float64}, variant_id::Int,
        solver_attrs::AbstractDict; representative_days::Int = 5,
        variant_timeout_seconds::Real = 20.0)
    bundle = load_campaign_bundle(config_path; verify_input = false)
    context = load_1108_worker_context(bundle)
    return _solve_1108_production_variant(
        context, sampled_values, variant_id;
        solver = :gurobi, threads = 2, mode = :ts,
        representative_days = representative_days, periods = [2050],
        solver_attrs = solver_attrs,
        variant_timeout_seconds = variant_timeout_seconds, max_attempts = 1)
end

function _production_worker_loop(task_channel::RemoteChannel, event_channel::RemoteChannel,
                                 config_path::AbstractString, solver::Symbol, threads::Int,
                                 mode::Symbol, representative_days::Union{Nothing,Int},
                                 periods::Vector{Int}, solver_attrs::AbstractDict,
                                 variant_timeout_seconds::Real,
                                 max_attempts::Int, hours_per_day::Int,
                                 output_dir::AbstractString,
                                 store_full_outputs::Bool,
                                 fallback_solver_attrs::Union{Nothing,AbstractDict} = nothing)
    bundle = load_campaign_bundle(config_path; verify_input = false)
    context = mode === :ts ? load_1108_worker_context(bundle) : load_1108_context(bundle)
    while true
        task = take!(task_channel)
        task === nothing && break
        variant_id, sampled_values = task[1], task[2]
        # Coordinator re-queues carry a flag requesting the crossover fallback.
        use_fallback = length(task) >= 3 && task[3] === true
        attempt_attrs = (use_fallback && fallback_solver_attrs !== nothing) ?
            merge(Dict{String,Any}(solver_attrs), Dict{String,Any}(fallback_solver_attrs)) :
            solver_attrs
        worker_id = Distributed.myid()
        put!(event_channel, (:started, worker_id, variant_id, time()))
        result, attempts = _solve_1108_production_variant(
            context, sampled_values, variant_id; solver = solver, threads = threads,
            mode = mode, representative_days = representative_days,
            periods = periods, solver_attrs = attempt_attrs,
            variant_timeout_seconds = variant_timeout_seconds,
            max_attempts = max_attempts, hours_per_day = hours_per_day,
            fallback_solver_attrs = fallback_solver_attrs,
            output_dir = output_dir, store_full_outputs = store_full_outputs)
        put!(event_channel, (:finished, worker_id, result, attempts))
    end
    return nothing
end

function _add_production_workers(count::Int, worker_flags; label::AbstractString)
    count <= 0 && return Int[]
    added = addprocs(count; exeflags = worker_flags)
    println(stdout, "[campaign] ", label, " workers spawned=", length(added),
            " loading IESAOpt...")
    flush(stdout)
    boot_events = Channel{Any}(length(added))
    for worker in added
        @async try
            remotecall_wait(Main.eval, worker, quote
                using IESAOpt
                using Logging
                global_logger(ConsoleLogger(stderr, Logging.Warn))
            end)
            put!(boot_events, (:ready, worker, nothing))
        catch err
            put!(boot_events, (:error, worker, err))
        end
    end
    for ready_count in eachindex(added)
        boot_wait_started = time()
        while timedwait(() -> isready(boot_events), 2.0) !== :ok
            elapsed = time() - boot_wait_started
            println(stdout, "[campaign] ", label, " workers ready=", ready_count - 1,
                    "/", length(added), " bootstrap_wait=", round(elapsed; digits = 1), "s")
            flush(stdout)
            elapsed <= 300 || error("No worker initialized for 300 seconds.")
        end
        state, worker, err = take!(boot_events)
        state === :ready || throw(err)
        println(stdout, "[campaign] ", label, " workers ready=", ready_count,
                "/", length(added), " latest=", worker)
        flush(stdout)
    end
    return added
end

"""
    run_1108_production_campaign(context, spec; output_dir, kwargs...)

Run or resume a deterministic 1108 SSP campaign. Every evaluation starts
from the immutable in-memory baseline and rebuilds JuMP. Samples and results
are incrementally persisted to `campaign.duckdb`.
"""
function run_1108_production_campaign(context::Legacy1108Context, spec::CampaignSpec;
                                      output_dir::AbstractString,
                                      n_workers::Int = 1,
                                      initial_workers::Int = n_workers,
                                      adaptive_workers::Bool = false,
                                      target_cpu_percent::Real = 95.0,
                                      cpu_sample_seconds::Real = 10.0,
                                      cpu_sustain_samples::Int = 3,
                                      worker_scale_step::Int = 2,
                                      threads_per_worker::Int = 1,
                                      solver::Symbol = :gurobi,
                                      solver_attrs::AbstractDict = Dict{String,Any}(),
                                      fallback_solver_attrs::Union{Nothing,AbstractDict} = nothing,
                                      mode::Symbol = :ts,
                                      representative_days::Union{Nothing,Int} = nothing,
                                      periods::Vector{Int} = [2050],
                                      variant_timeout_seconds::Real = Inf,
                                      worker_timeout_seconds::Real = variant_timeout_seconds,
                                      progress_timeout_seconds::Real = 30.0,
                                      max_attempts::Int = 2,
                                      hours_per_day::Int = 24,
                                      store_full_outputs::Bool = false,
                                      overwrite::Bool = false)
    max_attempts > 0 || throw(ArgumentError("max_attempts must be positive."))
    0 < initial_workers <= n_workers || throw(ArgumentError(
        "initial_workers must be between 1 and n_workers."))
    0 < target_cpu_percent <= 100 || throw(ArgumentError(
        "target_cpu_percent must be in (0, 100]."))
    cpu_sample_seconds > 0 || throw(ArgumentError("cpu_sample_seconds must be positive."))
    cpu_sustain_samples > 0 || throw(ArgumentError("cpu_sustain_samples must be positive."))
    worker_scale_step > 0 || throw(ArgumentError("worker_scale_step must be positive."))
    variant_timeout_seconds > 0 || throw(ArgumentError(
        "variant_timeout_seconds must be positive."))
    worker_timeout_seconds >= variant_timeout_seconds || throw(ArgumentError(
        "worker_timeout_seconds must be at least variant_timeout_seconds."))
    progress_timeout_seconds > worker_timeout_seconds || throw(ArgumentError(
        "progress_timeout_seconds must be greater than worker_timeout_seconds."))
    hours_per_day in (4, 6, 8, 12, 24) || throw(ArgumentError(
        "hours_per_day must be one of 4, 6, 8, 12, or 24."))
    representative_days === nothing || representative_days > 0 ||
        throw(ArgumentError("representative_days must be positive."))
    effective_representative_days = mode === :ts ?
        something(representative_days, context.base_md.params.n_repDays) : nothing
    mode === :ts && verify_1108_cluster_caches(
        context, effective_representative_days)
    samples = sample_campaign(spec)
    store = _open_production_store(output_dir, context.bundle, spec, samples;
                                   solver = solver, mode = mode,
                                   representative_days = effective_representative_days,
                                   solver_attrs = solver_attrs,
                                   variant_timeout_seconds = variant_timeout_seconds,
                                   worker_timeout_seconds = worker_timeout_seconds,
                                   hours_per_day = hours_per_day,
                                   store_full_outputs = store_full_outputs,
                                   overwrite = overwrite)
    started = time()
    run_started_at = now()
    effective_worker_timeout_seconds = worker_timeout_seconds
    try
        completed = _production_completed_ids(store)
        counts = _production_result_counts(store)
        pending = [variant_id for variant_id in axes(samples.values, 1)
                   if variant_id ∉ completed]
        _write_campaign_status(output_dir; state = "starting",
            completed = length(completed), total = size(samples.values, 1), counts = counts,
            run_started_at = run_started_at)
        _print_campaign_progress(state = "starting", completed = length(completed),
            total = size(samples.values, 1), started_at = started, counts = counts)
        optimal = 0
        failed = 0
        if n_workers <= 1
            for (position, variant_id) in pairs(pending)
                result, attempts = _solve_1108_production_variant(
                    context, collect(samples.values[variant_id, :]), variant_id;
                    solver = solver, threads = threads_per_worker, mode = mode,
                    representative_days = effective_representative_days,
                    periods = periods, solver_attrs = solver_attrs,
                    variant_timeout_seconds = variant_timeout_seconds,
                    max_attempts = max_attempts, hours_per_day = hours_per_day,
                    output_dir = output_dir, store_full_outputs = store_full_outputs)
                _write_production_result!(store, result, attempts)
                counts[_campaign_result_category(result.term_status)] += 1
                result.term_status == "OPTIMAL" ? (optimal += 1) : (failed += 1)
                _write_campaign_status(output_dir; state = "running",
                    completed = length(completed) + position,
                    total = size(samples.values, 1), counts = counts)
                _print_campaign_progress(state = "running",
                    completed = length(completed) + position,
                    total = size(samples.values, 1), started_at = started, counts = counts)
                @info "Campaign result persisted" variant_id position total=length(pending) status=result.term_status
            end
        elseif !isempty(pending)
            project_dir = dirname(Base.active_project())
            worker_flags = ["--project=$project_dir", "--threads=$(max(1, threads_per_worker))"]
            starting_workers = adaptive_workers ? initial_workers : n_workers
            println(stdout, "[campaign] spawning workers=", starting_workers,
                    " max_workers=", n_workers,
                    " threads_per_worker=", threads_per_worker,
                    " adaptive=", adaptive_workers,
                    " target_cpu=", target_cpu_percent, "%")
            flush(stdout)
            workers_added = _add_production_workers(
                starting_workers, worker_flags; label = "initial")
            all_workers_added = copy(workers_added)
            try
                task_channel = RemoteChannel(() -> Channel{Any}(max(16, 4n_workers)))
                event_channel = RemoteChannel(() -> Channel{Any}(max(16, 4n_workers)))
                futures = Dict(worker => remotecall(
                    _production_worker_loop, worker, task_channel, event_channel,
                    context.bundle.config_path, solver, threads_per_worker, mode,
                    effective_representative_days, periods, solver_attrs,
                    variant_timeout_seconds, 1, hours_per_day,
                    output_dir, store_full_outputs, fallback_solver_attrs) for worker in workers_added)
                feeder = _start_production_feeder(task_channel, pending, samples)
                active_workers = Set(workers_added)
                in_flight = Dict{Int,Tuple{Int,Float64}}()
                completed_now = Set{Int}()
                attempt_counts = Dict{Int,Int}()
                first_pass_remaining = Set(pending)
                deferred_retries = Set{Int}()
                last_event_at = time()
                last_heartbeat_at = 0.0
                cpu_sampler = adaptive_workers ? _system_cpu_sampler() : nothing
                last_cpu_sample_at = time()
                cpu_load = NaN
                below_target_samples = 0
                phase_counts = (barrier = 0, crossover = 0)
                core_capacity = Sys.CPU_THREADS
                while length(completed_now) < length(pending)
                    if timedwait(() -> isready(event_channel), 0.25) === :ok
                        event = take!(event_channel)
                        last_event_at = time()
                        if event[1] === :started
                            _, worker, variant_id, started_at = event
                            variant_id in completed_now ||
                                (in_flight[worker] = (variant_id, started_at))
                        else
                            _, worker, result, attempts = event
                            pop!(in_flight, worker, nothing)
                            delete!(first_pass_remaining, result.variant_id)
                            if result.variant_id ∉ completed_now
                                attempt_count = get(attempt_counts, result.variant_id, 0) + attempts
                                attempt_counts[result.variant_id] = attempt_count
                                # Infeasibility is deterministic - crossover cannot
                                # make an infeasible LP feasible, so accept it at once.
                                attempt_budget = _campaign_result_category(result.term_status) === :infeasible ?
                                    1 : max_attempts
                                if result.term_status != "OPTIMAL" && attempt_count < attempt_budget
                                    _write_production_result!(store, result, attempt_count)
                                    if _campaign_result_category(result.term_status) === :timeout &&
                                            !isempty(first_pass_remaining)
                                        push!(deferred_retries, result.variant_id)
                                        @warn "Campaign timeout deferred until end of first pass" variant_id=result.variant_id attempt=attempt_count remaining_first_pass=length(first_pass_remaining)
                                    else
                                        put!(task_channel, (result.variant_id,
                                            collect(samples.values[result.variant_id, :]), true))
                                        @warn "Campaign result queued for retry" variant_id=result.variant_id status=result.term_status attempt=attempt_count
                                    end
                                else
                                    _write_production_result!(store, result, attempt_count)
                                    push!(completed_now, result.variant_id)
                                    counts[_campaign_result_category(result.term_status)] += 1
                                    result.term_status == "OPTIMAL" ? (optimal += 1) : (failed += 1)
                                    @info "Campaign result persisted" variant_id=result.variant_id position=length(completed_now) total=length(pending) status=result.term_status attempts=attempt_count
                                end
                            end
                        end
                    end
                    if isempty(first_pass_remaining) && !isempty(deferred_retries)
                        println(stdout, "[campaign] queueing deferred timeout retries=",
                            length(deferred_retries))
                        flush(stdout)
                        for variant_id in sort!(collect(deferred_retries))
                            put!(task_channel,
                                (variant_id, collect(samples.values[variant_id, :]), true))
                        end
                        empty!(deferred_retries)
                    end

                    stalled = [worker for (worker, (_, started_at)) in in_flight
                               if time() - started_at > effective_worker_timeout_seconds]
                    replacement_count = 0
                    for worker in stalled
                        variant_id, started_at = pop!(in_flight, worker)
                        variant_id in completed_now && continue
                        delete!(first_pass_remaining, variant_id)
                        count = get(attempt_counts, variant_id, 0) + 1
                        attempt_counts[variant_id] = count
                        @error "Campaign worker stalled" worker variant_id elapsed_seconds=time()-started_at recovery_attempt=count
                        delete!(active_workers, worker)
                        if count < max_attempts
                            if isempty(first_pass_remaining)
                                put!(task_channel,
                                    (variant_id, collect(samples.values[variant_id, :]), true))
                            else
                                push!(deferred_retries, variant_id)
                            end
                        else
                            result = VariantResult(
                                variant_id = variant_id,
                                leaf_values = collect(samples.values[variant_id, :]),
                                term_status = "WORKER_TIMEOUT",
                                worker_pid = worker,
                                error = "Worker exceeded $(effective_worker_timeout_seconds) seconds on $count attempts.")
                            _write_production_result!(store, result, count)
                            push!(completed_now, variant_id)
                            counts[:timeout] += 1
                            failed += 1
                        end
                        replacement_count += 1
                    end
                    if !isempty(stalled)
                        try
                            rmprocs(stalled; waitfor = 0)
                        catch err
                            @warn "Failed to remove stalled worker batch" stalled err
                        end
                    end
                    if replacement_count > 0 && length(completed_now) < length(pending)
                        # Stalled workers are removed with waitfor=0, so their OS
                        # processes can linger; cap replacements at the worker
                        # budget or the pool drifts above n_workers and oversubscribes.
                        replacement_count = min(replacement_count,
                                                max(0, n_workers - length(active_workers)))
                    end
                    if replacement_count > 0 && length(completed_now) < length(pending)
                        println(stdout, "[campaign] replacing stalled workers=", replacement_count,
                            " active=", length(active_workers), "/", n_workers)
                        flush(stdout)
                        replacements = _add_production_workers(
                            replacement_count, worker_flags; label = "replacement")
                        append!(all_workers_added, replacements)
                        union!(active_workers, replacements)
                        for replacement in replacements
                            futures[replacement] = remotecall(
                                _production_worker_loop, replacement, task_channel,
                                event_channel, context.bundle.config_path, solver, threads_per_worker,
                                mode, effective_representative_days, periods,
                                solver_attrs, variant_timeout_seconds, 1,
                                hours_per_day, output_dir, store_full_outputs,
                                fallback_solver_attrs)
                        end
                    end
                        if adaptive_workers && cpu_sampler !== nothing &&
                                time() - last_cpu_sample_at >= cpu_sample_seconds
                            cpu_load = _sample_system_cpu!(cpu_sampler)
                            last_cpu_sample_at = time()
                            phase_counts = _production_solver_phases(in_flight, output_dir)
                            all_busy = !isempty(active_workers) &&
                                length(in_flight) >= length(active_workers)
                            below_target_samples = isfinite(cpu_load) &&
                                cpu_load < target_cpu_percent && all_busy &&
                                length(active_workers) < n_workers ? below_target_samples + 1 : 0
                            capacity = _production_worker_capacity(
                                phase_counts.barrier, phase_counts.crossover,
                                threads_per_worker; core_capacity = core_capacity)
                            effective_core_demand = capacity.effective_cores
                            phase_additions = all_busy && phase_counts.crossover > 0 ?
                                capacity.additional_workers : 0
                            requested_additions = min(
                                phase_additions, capacity.additional_workers)
                            if requested_additions > 0
                                available = n_workers - length(active_workers)
                                add_count = min(requested_additions, available,
                                    length(pending) - length(completed_now) - length(in_flight))
                                if add_count > 0
                                    additions = _add_production_workers(
                                        add_count, worker_flags; label = "adaptive")
                                    append!(all_workers_added, additions)
                                    union!(active_workers, additions)
                                    for worker in additions
                                        futures[worker] = remotecall(
                                            _production_worker_loop, worker, task_channel,
                                            event_channel, context.bundle.config_path, solver,
                                            threads_per_worker, mode,
                                            effective_representative_days, periods,
                                            solver_attrs, variant_timeout_seconds, 1,
                                            hours_per_day, output_dir, store_full_outputs,
                                            fallback_solver_attrs)
                                    end
                                    println(stdout, "[campaign] adaptive scale-up added=", add_count,
                                        " workers=", length(active_workers), "/", n_workers,
                                        " sampled_cpu=", round(cpu_load; digits = 1), "%",
                                        " barrier=", phase_counts.barrier,
                                        " crossover=", phase_counts.crossover,
                                        " effective_cores=", effective_core_demand,
                                        " core_capacity=", core_capacity)
                                    flush(stdout)
                                end
                                below_target_samples = 0
                            end
                        end
                    if isempty(in_flight) && time() - last_event_at > progress_timeout_seconds
                        message = "No worker event for $(round(time() - last_event_at; digits = 1)) seconds."
                        _write_campaign_status(output_dir; state = "stalled",
                            completed = length(completed) + length(completed_now),
                            total = size(samples.values, 1), active = in_flight,
                            counts = counts, message = message)
                        _print_campaign_progress(state = "stalled",
                            completed = length(completed) + length(completed_now),
                            total = size(samples.values, 1), active = in_flight,
                            started_at = started, counts = counts, message = message)
                        error("$message Stopping for safe resume.")
                    end
                    if time() - last_heartbeat_at >= 2
                        _write_campaign_status(output_dir; state = "running",
                            completed = length(completed) + length(completed_now),
                            total = size(samples.values, 1), active = in_flight,
                            counts = counts, current_workers = length(active_workers),
                            max_workers = n_workers, cpu_load = cpu_load,
                            barrier_workers = phase_counts.barrier,
                            crossover_workers = phase_counts.crossover,
                            run_started_at = run_started_at)
                        _print_campaign_progress(state = "running",
                            completed = length(completed) + length(completed_now),
                            total = size(samples.values, 1), active = in_flight,
                            started_at = started, counts = counts,
                            current_workers = length(active_workers),
                            max_workers = n_workers, cpu_load = cpu_load,
                            barrier_workers = phase_counts.barrier,
                            crossover_workers = phase_counts.crossover)
                        last_heartbeat_at = time()
                    end
                end
                wait(feeder)
                foreach(_ -> put!(task_channel, nothing), active_workers)
                for worker in active_workers
                    try
                        fetch(futures[worker])
                    catch err
                        @warn "Production worker exited during shutdown" worker err
                    end
                end
            finally
                live_workers = filter(worker -> worker in workers(), all_workers_added)
                isempty(live_workers) || rmprocs(live_workers)
            end
        end
        _write_campaign_status(output_dir; state = "completed",
            completed = size(samples.values, 1), total = size(samples.values, 1), counts = counts,
            run_started_at = run_started_at)
        _print_campaign_progress(state = "completed",
            completed = size(samples.values, 1), total = size(samples.values, 1),
            started_at = started, counts = counts)
        return ProductionCampaignSummary(
            database_path = store.path, total = size(samples.values, 1),
            previously_completed = length(completed), attempted = length(pending),
            optimal = optimal, failed = failed, runtime_seconds = time() - started)
    finally
        _close_production_store!(store)
    end
end