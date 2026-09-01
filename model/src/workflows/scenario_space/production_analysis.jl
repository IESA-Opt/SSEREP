function _production_analysis_data(database)
    metadata_table = _production_query(database, "SELECT key, value FROM metadata")
    metadata = Dict(String(row.key) => String(row.value) for row in eachrow(metadata_table))
    parameters = _production_query(database,
        "SELECT parameter_id, name, minimum, maximum FROM parameters ORDER BY parameter_id")
    samples_table = _production_query(database,
        "SELECT variant_id, parameter_id, value FROM samples ORDER BY variant_id, parameter_id")
    results = _production_query(database, """
        SELECT variant_id, objective, co2_price, term_status
        FROM results ORDER BY variant_id
    """)
    n_evaluations = parse(Int, metadata["n_evaluations"])
    n_parameters = nrow(parameters)
    samples = fill(NaN, n_evaluations, n_parameters)
    for row in eachrow(samples_table)
        samples[Int(row.variant_id), Int(row.parameter_id)] = Float64(row.value)
    end
    bounds = [(Float64(row.minimum), Float64(row.maximum)) for row in eachrow(parameters)]
    return metadata, String.(parameters.name), bounds, samples, results
end

function _output_summary_table(rows)
    isempty(rows) && return DataFrame(
        checkpoint = Int[], output = String[], n_used = Int[], mean = Float64[],
        std = Float64[], minimum = Float64[], q05 = Float64[], median = Float64[],
        q95 = Float64[], maximum = Float64[])
    return DataFrame(rows)
end

function _delta_convergence_table(rows)
    isempty(rows) && return DataFrame(
        checkpoint = Int[], output = String[], parameter_id = Int[],
        parameter = String[], rank = Int[], delta = Float64[],
        adjusted_delta = Float64[], conf_low = Float64[], conf_high = Float64[],
        n_used = Int[])
    return DataFrame(rows)
end

function _morris_convergence_table(rows)
    isempty(rows) && return DataFrame(
        trajectories = Int[], output = String[], parameter_id = Int[],
        parameter = String[], rank = Int[], mu = Float64[], mu_star = Float64[],
        sigma = Float64[], n_evaluations = Int[])
    return DataFrame(rows)
end

function _output_summary_rows(results::DataFrame, checkpoints::Vector{Int})
    rows = NamedTuple[]
    for checkpoint in checkpoints, output in (:objective, :co2_price)
        values = Float64[]
        for row in eachrow(results)
            row.variant_id <= checkpoint || continue
            row.term_status == "OPTIMAL" || continue
            value = Float64(getproperty(row, output))
            isfinite(value) && push!(values, value)
        end
        isempty(values) && continue
        push!(rows, (checkpoint = checkpoint, output = String(output),
                     n_used = length(values), mean = mean(values), std = std(values),
                     minimum = minimum(values), q05 = quantile(values, 0.05),
                     median = quantile(values, 0.5), q95 = quantile(values, 0.95),
                     maximum = maximum(values)))
    end
    return _output_summary_table(rows)
end

function _delta_convergence_rows(samples::Matrix{Float64}, results::DataFrame,
                                 parameter_names::Vector{String}, checkpoints::Vector{Int};
                                 nboot::Int, seed::Int)
    rows = NamedTuple[]
    for checkpoint in checkpoints, output in (:objective, :co2_price)
        result_by_id = Dict(Int(row.variant_id) => row for row in eachrow(results))
        ids = [variant_id for variant_id in 1:min(checkpoint, size(samples, 1))
               if haskey(result_by_id, variant_id) &&
                  result_by_id[variant_id].term_status == "OPTIMAL" &&
                  isfinite(Float64(getproperty(result_by_id[variant_id], output)))]
        length(ids) >= 20 || continue
        output_values = Float64[getproperty(result_by_id[id], output) for id in ids]
        method = DeltaMoment(nboot = nboot, conf_level = 0.95, Ygrid_length = 2048)
        delta = gsa(permutedims(samples[ids, :]), output_values, method;
                    rng = MersenneTwister(seed + checkpoint))
        order = sortperm(vec(delta.adjusted_deltas); rev = true)
        for (rank, parameter_id) in pairs(order)
            push!(rows, (checkpoint = checkpoint, output = String(output),
                         parameter_id = parameter_id,
                         parameter = parameter_names[parameter_id], rank = rank,
                         delta = vec(delta.deltas)[parameter_id],
                         adjusted_delta = vec(delta.adjusted_deltas)[parameter_id],
                         conf_low = vec(delta.adjusted_deltas_low)[parameter_id],
                         conf_high = vec(delta.adjusted_deltas_hi)[parameter_id],
                         n_used = length(ids)))
        end
    end
    return _delta_convergence_table(rows)
end

function _morris_effects(samples::AbstractMatrix{<:Real}, outputs::AbstractVector{<:Real},
                         parameter_names::Vector{String}; trajectories::Int,
                         bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing)
    parameter_count = length(parameter_names)
    trajectory_size = parameter_count + 1
    size(samples, 1) == trajectories * trajectory_size || throw(DimensionMismatch(
        "Morris data must contain $trajectories complete $trajectory_size-point trajectories."))
    effects = [Float64[] for _ in 1:parameter_count]
    for trajectory in 1:trajectories
        first_row = (trajectory - 1) * trajectory_size + 1
        for offset in 1:parameter_count
            row_a = first_row + offset - 1
            row_b = row_a + 1
            changed = findall(j -> !isapprox(samples[row_a, j], samples[row_b, j]),
                              axes(samples, 2))
            length(changed) == 1 || throw(ArgumentError(
                "Morris trajectory $trajectory step $offset changes $(length(changed)) parameters."))
            parameter_id = only(changed)
            physical_step = Float64(samples[row_b, parameter_id]) -
                        Float64(samples[row_a, parameter_id])
            coded_step = bounds === nothing ? physical_step :
                physical_step / (bounds[parameter_id][2] - bounds[parameter_id][1])
            push!(effects[parameter_id],
                  (Float64(outputs[row_b]) - Float64(outputs[row_a])) /
                coded_step)
        end
    end
    return DataFrame(
        parameter_id = collect(1:parameter_count), parameter = parameter_names,
        mu = [mean(values) for values in effects],
        mu_star = [mean(abs, values) for values in effects],
        sigma = [length(values) > 1 ? std(values) : 0.0 for values in effects],
        trajectories = fill(trajectories, parameter_count),
    )
end

function _morris_convergence_rows(samples::Matrix{Float64}, results::DataFrame,
                                  parameter_names::Vector{String}, checkpoints::Vector{Int};
                                  bounds::Union{Nothing,Vector{Tuple{Float64,Float64}}} = nothing)
    result_by_id = Dict(Int(row.variant_id) => row for row in eachrow(results))
    rows = NamedTuple[]
    trajectory_size = length(parameter_names) + 1
    for trajectories in checkpoints, output in (:objective, :co2_price)
        n_rows = trajectories * trajectory_size
        n_rows <= size(samples, 1) || continue
        ids = 1:n_rows
        all(id -> haskey(result_by_id, id) && result_by_id[id].term_status == "OPTIMAL" &&
                  isfinite(Float64(getproperty(result_by_id[id], output))), ids) || continue
        values = Float64[getproperty(result_by_id[id], output) for id in ids]
        table = _morris_effects(samples[ids, :], values, parameter_names;
                                trajectories = trajectories, bounds = bounds)
        order = sortperm(table.mu_star; rev = true)
        for (rank, row_id) in pairs(order)
            row = table[row_id, :]
            push!(rows, (trajectories = trajectories, output = String(output),
                         parameter_id = row.parameter_id, parameter = row.parameter,
                         rank = rank, mu = row.mu, mu_star = row.mu_star,
                         sigma = row.sigma, n_evaluations = n_rows))
        end
    end
    return _morris_convergence_table(rows)
end

function _global_morris_rows(samples::Matrix{Float64}, results::DataFrame,
                             parameter_names::Vector{String},
                             bounds::Vector{Tuple{Float64,Float64}};
                             trajectories::Int, seed::Int, levels::Int = 4)
    result_by_id = Dict(Int(row.variant_id) => row for row in eachrow(results))
    expected = trajectories * (length(parameter_names) + 1)
    expected == size(samples, 1) || throw(DimensionMismatch(
        "Morris analysis expected $expected samples, found $(size(samples, 1))."))
    all(id -> haskey(result_by_id, id) && result_by_id[id].term_status == "OPTIMAL",
        1:expected) || throw(ArgumentError(
        "GlobalSensitivity Morris analysis requires every trajectory evaluation to be optimal."))

    unit_bounds = [[0.0, 1.0] for _ in parameter_names]
    rows = NamedTuple[]
    for output in (:objective, :co2_price)
        lookup = Dict{Tuple,Float64}(
            Tuple(samples[id, :]) => Float64(getproperty(result_by_id[id], output))
            for id in 1:expected)
        model = function (unit_values)
            physical = Tuple(bounds[j][1] + Float64(unit_values[j]) *
                             (bounds[j][2] - bounds[j][1])
                             for j in eachindex(bounds))
            haskey(lookup, physical) || error("Morris design does not match persisted samples.")
            return lookup[physical]
        end
        method = Morris(
            p_steps = fill(levels, length(parameter_names)), relative_scale = false,
            num_trajectory = trajectories, total_num_trajectory = 5trajectories,
            len_design_mat = length(parameter_names) + 1)
        sensitivity = gsa(model, method, unit_bounds;
                          rng = MersenneTwister(seed))
        means = vec(sensitivity.means)
        means_star = vec(sensitivity.means_star)
        sigma = sqrt.(max.(vec(sensitivity.variances), 0.0))
        length(means) == length(parameter_names) || error(
            "GlobalSensitivity Morris returned $(length(means)) effects for $(length(parameter_names)) parameters.")
        order = sortperm(means_star; rev = true)
        for (rank, parameter_id) in pairs(order)
            push!(rows, (
                trajectories = trajectories, output = String(output),
                parameter_id = parameter_id, parameter = parameter_names[parameter_id],
                rank = rank, mu = means[parameter_id],
                mu_star = means_star[parameter_id], sigma = sigma[parameter_id],
                n_evaluations = expected))
        end
    end
    return _morris_convergence_table(rows)
end
"""Analyze a completed or partially completed production campaign in place."""
function analyze_production_campaign(database_path::AbstractString;
                                     checkpoints::AbstractVector{<:Integer} = Int[],
                                     nboot::Integer = 200)
    isfile(database_path) || throw(ArgumentError("Campaign database not found: $database_path"))
    database = DuckDB.DB(database_path)
    try
        metadata, parameter_names, bounds, samples, results = _production_analysis_data(database)
        method = Symbol(metadata["method"])
        total = size(samples, 1)
        if method === :lhs
            selected = isempty(checkpoints) ?
                sort!(unique(filter(<=(total), [100, 250, 500, 1000, 2500, 5000, 10000, total]))) :
                sort!(unique(Int.(checkpoints)))
            summary = _output_summary_rows(results, selected)
            delta = _delta_convergence_rows(samples, results, parameter_names, selected;
                                            nboot = Int(nboot),
                                            seed = parse(Int, metadata["seed"]))
            _duckdb_replace_table!(database, summary, "convergence_summary")
            _duckdb_replace_table!(database, delta, "delta_convergence")
            return (summary = summary, sensitivity = delta)
        elseif method === :morris
            total_trajectories = div(total, length(parameter_names) + 1)
            selected = isempty(checkpoints) ?
                sort!(unique(filter(<=(total_trajectories),
                                    [10, 20, 30, 40, 50, 75, 100, total_trajectories]))) :
                sort!(unique(Int.(checkpoints)))
            convergence = _morris_convergence_rows(
                samples, results, parameter_names, selected; bounds = bounds)
            morris = _global_morris_rows(
                samples, results, parameter_names, bounds;
                trajectories = total_trajectories,
                seed = parse(Int, metadata["seed"]))
            _duckdb_replace_table!(database, convergence, "morris_convergence")
            _duckdb_replace_table!(database, morris, "morris_sensitivity")
            return (summary = DataFrame(), sensitivity = morris,
                    convergence = convergence)
        end
        throw(ArgumentError("Unsupported production analysis method: $method"))
    finally
        DBInterface.close!(database)
        GC.gc(true)
    end
end