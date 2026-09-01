using DataFrames
using GlobalSensitivity
using Random

"""
    delta_sensitivity(result; output=:objective, nboot=500, conf_level=0.95,
                      ygrid_length=2048, seed=result.spec.seed, min_optimal=20)

Estimate Borgonovo delta indices from completed scenario-space runs using
`GlobalSensitivity.DeltaMoment`. Only optimal runs with finite output values
are retained. `output` may be any numeric field of [`VariantResult`](@ref),
including `:objective` and `:co2_price`.
"""
function delta_sensitivity(result::ScenarioResult;
                           output::Symbol = :objective,
                           nboot::Integer = 500,
                           conf_level::Real = 0.95,
                           ygrid_length::Integer = 2048,
                           seed::Integer = result.spec.seed,
                           min_optimal::Integer = 20)
    hasfield(VariantResult, output) ||
        throw(ArgumentError("VariantResult has no output field $(repr(output))"))
    keep = [variant.term_status == "OPTIMAL" &&
            isfinite(Float64(getfield(variant, output))) for variant in result.variants]
    n_used = count(keep)
    n_used >= min_optimal || throw(ArgumentError(
        "delta_sensitivity needs at least $min_optimal finite optimal variants, have $n_used"))

    samples = permutedims(result.samples[keep, :])
    outputs = Float64[getfield(variant, output) for variant in result.variants[keep]]
    method = DeltaMoment(nboot = Int(nboot), conf_level = Float64(conf_level),
                         Ygrid_length = Int(ygrid_length))
    result_delta = gsa(samples, outputs, method; rng = MersenneTwister(seed))

    table = DataFrame(
        target = [target.label for target in result.spec.targets],
        delta = vec(result_delta.deltas),
        adjusted_delta = vec(result_delta.adjusted_deltas),
        conf_low = vec(result_delta.adjusted_deltas_low),
        conf_high = vec(result_delta.adjusted_deltas_hi),
        n_used = fill(n_used, length(result.spec.targets)),
    )
    sort!(table, :adjusted_delta; rev = true)
    table.rank = collect(1:nrow(table))
    return table
end

"""
    morris_sensitivity(f, targets; num_trajectories=20,
                       candidate_trajectories=5num_trajectories,
                       levels=4, relative_scale=true, seed=0, batch=false)

Run an optimized Morris experiment generated and analyzed by
`GlobalSensitivity.jl`. `f` receives one parameter vector, or a batch matrix
when `batch=true`, and must return a scalar output. The model is evaluated
exactly `num_trajectories * (length(targets) + 1)` times.
"""
function morris_sensitivity(f, targets::AbstractVector{LeafTarget};
                            num_trajectories::Integer = 20,
                            candidate_trajectories::Integer = 5 * num_trajectories,
                            levels::Integer = 4,
                            relative_scale::Bool = true,
                            seed::Integer = 0,
                            batch::Bool = false)
    isempty(targets) && throw(ArgumentError("Morris analysis requires at least one target"))
    num_trajectories > 0 || throw(ArgumentError("num_trajectories must be positive"))
    candidate_trajectories >= num_trajectories || throw(ArgumentError(
        "candidate_trajectories must be at least num_trajectories"))
    levels >= 2 || throw(ArgumentError("levels must be at least 2"))

    parameter_count = length(targets)
    method = Morris(
        p_steps = fill(Int(levels), parameter_count),
        relative_scale = relative_scale,
        num_trajectory = Int(num_trajectories),
        total_num_trajectory = Int(candidate_trajectories),
        len_design_mat = parameter_count + 1,
    )
    bounds = [[target.min, target.max] for target in targets]
    result_morris = gsa(f, method, bounds;
                        batch = batch, rng = MersenneTwister(seed))
    means = vec(result_morris.means)
    means_star = vec(result_morris.means_star)
    variances = vec(result_morris.variances)
    table = DataFrame(
        target = [target.label for target in targets],
        mu = means,
        mu_star = means_star,
        sigma = sqrt.(max.(variances, 0.0)),
        n_evaluations = fill(Int(num_trajectories) * (parameter_count + 1),
                             parameter_count),
    )
    sort!(table, :mu_star; rev = true)
    table.rank = collect(1:nrow(table))
    return table
end