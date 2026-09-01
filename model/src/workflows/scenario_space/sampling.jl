# =============================================================================
# workflows/scenario_space/sampling.jl -- parameter-space samplers (LHS, Morris, Sobol, Factorial)
#
# All samplers consume a `CampaignSpec` and return a `SampleMatrix`:
#   * `parameters` :: Vector{String}        (unique parameter names, columns of `values`)
#   * `values`     :: Matrix{Float64}       (n_variants × n_parameters)
#   * `n_variants` :: Int
#
# The variant 0 is reserved for the *reference* run (base workbook unchanged);
# `SampleMatrix` only holds variants 1..N. The campaign runner is responsible
# for solving Ref before the sampled variants.
#
# References:
#   * LHS: McKay, Beckman & Conover (1979) — implemented via random shuffles of
#     stratified per-parameter intervals.
#   * Morris trajectories: Morris (1991) — radial design of length k+1 per
#     trajectory in the [0,1]^k unit cube, Δ = p / (2(p-1)) per Saltelli (2008).
#   * Sobol: Joe & Kuo (2008) low-discrepancy sequence via the `Sobol.jl`
#     package; we skip the all-zeros origin point as is conventional.
#   * Factorial: Cartesian product of discretised parameter ranges.
# =============================================================================

using Random
using Sobol
using GlobalSensitivity

"""
    SampleMatrix

Result of sampling a [`CampaignSpec`](@ref). `values[i, j]` is the value of
parameter `parameters[j]` for variant `i` (1-indexed). The reference run
(variant 0) is *not* included — the runner adds it automatically.
"""
struct SampleMatrix
    parameters::Vector{String}
    values::Matrix{Float64}
    n_variants::Int
end

function SampleMatrix(parameters::AbstractVector{<:AbstractString}, values::AbstractMatrix{<:Real})
    n = size(values, 1)
    size(values, 2) == length(parameters) || throw(DimensionMismatch(
        "values has $(size(values,2)) columns but parameters has $(length(parameters)) entries"))
    return SampleMatrix(String.(parameters), Matrix{Float64}(values), n)
end

# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

"""
    sample_campaign(spec) -> SampleMatrix

Generate the variant matrix for `spec`. Dispatches on `spec.method`. Each
backend handles its own bounds/step constraints and seeding.
"""
function sample_campaign(spec::CampaignSpec)
    params = unique_parameters(spec)
    bounds = parameter_bounds(spec)
    if spec.method == :lhs
        return _sample_lhs(spec, params, bounds)
    elseif spec.method == :sobol
        return _sample_sobol(spec, params, bounds)
    elseif spec.method == :morris
        return _sample_morris(spec, params, bounds)
    elseif spec.method == :factorial
        return _sample_factorial(spec, params, bounds, parameter_steps(spec))
    else
        throw(ArgumentError("Unsupported sampling method: $(spec.method)"))
    end
end

# -----------------------------------------------------------------------------
# Latin hypercube sampling
# -----------------------------------------------------------------------------

"""
    _sample_lhs(spec, params, bounds) -> SampleMatrix

McKay's Latin hypercube: stratify each parameter into `n_variants` equal
intervals on `[min, max]`, take one random point per interval, then permute
each column independently. Result is a `n × k` matrix of real values.
"""
function _sample_lhs(spec::CampaignSpec, params::Vector{String}, bounds::Vector{Tuple{Float64,Float64}})
    rng = MersenneTwister(spec.seed)
    n = spec.n_variants
    k = length(params)
    values = Matrix{Float64}(undef, n, k)
    for j in 1:k
        lo, hi = bounds[j]
        # One uniform draw per stratum, then shuffle the strata
        u = rand(rng, n)
        perm = randperm(rng, n)
        for i in 1:n
            stratum = perm[i] - 1
            t = (stratum + u[i]) / n
            values[i, j] = lo + t * (hi - lo)
        end
    end
    return SampleMatrix(params, values, n)
end

# -----------------------------------------------------------------------------
# Sobol low-discrepancy sequence
# -----------------------------------------------------------------------------

function _sample_sobol(spec::CampaignSpec, params::Vector{String}, bounds::Vector{Tuple{Float64,Float64}})
    n = spec.n_variants
    k = length(params)
    seq = SobolSeq(k)
    Sobol.skip(seq, 1)
    values = Matrix{Float64}(undef, n, k)
    for i in 1:n
        u = next!(seq)
        for j in 1:k
            lo, hi = bounds[j]
            values[i, j] = lo + u[j] * (hi - lo)
        end
    end
    return SampleMatrix(params, values, n)
end

# -----------------------------------------------------------------------------
# Morris trajectories
# -----------------------------------------------------------------------------

"""
    _sample_morris(spec, params, bounds; p = 4) -> SampleMatrix

Morris elementary-effects design. Builds `r = spec.n_variants` radial
trajectories of length `k+1` in a `p`-level grid on `[0,1]^k`, with step
size `Δ = p / (2(p-1))` (Saltelli 2008, Eq. 3.1). Each trajectory contributes
`k+1` rows to the sample matrix, so the total number of model evaluations is
`r * (k+1)`. The user-supplied `n_variants` is interpreted as the number of
trajectories `r`.

Storing the implied count back into the result lets the UI display
"`r × (k+1)` model evaluations" so the user understands the total cost.
"""
function _sample_morris(spec::CampaignSpec, params::Vector{String}, bounds::Vector{Tuple{Float64,Float64}}; p::Int = 4)
    rng = MersenneTwister(spec.seed)
    r = spec.n_variants
    k = length(params)
    unit_points = Vector{Vector{Float64}}()
    method = Morris(
        p_steps = fill(p, k), relative_scale = false,
        num_trajectory = r, total_num_trajectory = 5r,
        len_design_mat = k + 1)
    unit_bounds = [[0.0, 1.0] for _ in 1:k]
    gsa(x -> (push!(unit_points, Float64.(x)); 0.0), method, unit_bounds; rng = rng)
    total_rows = r * (k + 1)
    length(unit_points) == total_rows || error(
        "GlobalSensitivity Morris generated $(length(unit_points)) points; expected $total_rows.")
    values = Matrix{Float64}(undef, total_rows, k)
    for row in 1:total_rows, parameter_id in 1:k
        lo, hi = bounds[parameter_id]
        values[row, parameter_id] = lo + unit_points[row][parameter_id] * (hi - lo)
    end
    return SampleMatrix(params, values, total_rows)
end

# -----------------------------------------------------------------------------
# Factorial design
# -----------------------------------------------------------------------------

"""
    _sample_factorial(spec, params, bounds, steps) -> SampleMatrix

Cartesian product of discretised parameter ranges. Each parameter must have a
positive `step` (validated upstream). Result size is `prod(length.(grids))`,
which can explode quickly — caller is expected to display the implied count
to the user before launching.
"""
function _sample_factorial(spec::CampaignSpec, params::Vector{String}, bounds::Vector{Tuple{Float64,Float64}}, steps::Vector{Union{Float64,Nothing}})
    grids = Vector{Vector{Float64}}(undef, length(params))
    for j in eachindex(params)
        steps[j] === nothing && throw(ArgumentError("Factorial sampling requires Step on parameter '$(params[j])'."))
        lo, hi = bounds[j]
        s = steps[j]::Float64
        s > 0 || throw(ArgumentError("Step on parameter '$(params[j])' must be positive."))
        grids[j] = collect(lo:s:hi)
        # Make sure the upper bound is included even if (hi - lo) is not a multiple of s
        if last(grids[j]) < hi - 1e-12
            push!(grids[j], hi)
        end
    end
    n_total = prod(length.(grids))
    n_total > 0 || throw(ArgumentError("Factorial design produced 0 combinations (check Min/Max/Step)."))
    values = Matrix{Float64}(undef, n_total, length(params))
    iter = Iterators.product(grids...)
    for (i, combo) in enumerate(iter)
        for j in eachindex(combo)
            values[i, j] = combo[j]
        end
    end
    return SampleMatrix(params, values, n_total)
end

# -----------------------------------------------------------------------------
# Helpers exposed to the UI for the implied-sample-size readout
# -----------------------------------------------------------------------------

"""
    implied_sample_size(spec) -> Int

Return the actual number of variants the campaign will produce. For LHS and
Sobol this equals `spec.n_variants`; for Morris it is `r * (k+1)`; for
Factorial it is `prod(length.(grids))`.
"""
function implied_sample_size(spec::CampaignSpec)
    k = length(unique_parameters(spec))
    if spec.method == :lhs || spec.method == :sobol
        return spec.n_variants
    elseif spec.method == :morris
        return spec.n_variants * (k + 1)
    elseif spec.method == :factorial
        steps = parameter_steps(spec)
        bounds = parameter_bounds(spec)
        any(s -> s === nothing, steps) && return -1
        n = 1
        for j in 1:k
            lo, hi = bounds[j]
            s = steps[j]::Float64
            len = length(collect(lo:s:hi))
            if len == 0 || (length(collect(lo:s:hi)) > 0 && last(collect(lo:s:hi)) < hi - 1e-12)
                len += 1
            end
            n *= len
        end
        return n
    end
    return spec.n_variants
end
