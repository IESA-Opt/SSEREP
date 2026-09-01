# =============================================================================
# workflows/scenario_space/orchestrator.jl -- Phase 4: high-level run_scenario_space() entry
#
# Phases 1-3 give us:
#   * Phase 1: spec parsing + sampling (CampaignSpec/SSDashboard xlsx format,
#     uses (sheet, cell) coordinates that need a workbook to resolve)
#   * Phase 2: in-place leaf mutation (LeafChange → mutate md → re-apply to
#     the JuMP model via the mutation manifest)
#   * Phase 3: campaign runner (Vector{Vector{LeafChange}} → Vector{VariantResult},
#     serial or distributed)
#
# Phase 4 closes the loop with a PROGRAMMATIC spec — `ScenarioSpec` — that
# references leaf parameters directly (no workbook coordinates needed) and
# wraps the sample-generate → leaf-changes → run_campaign pipeline into one
# call. The result is bundled into `ScenarioResult` for downstream analysis.
#
#   ┌─────────────┐  sample_scenario_space  ┌─────────────┐  samples_to_changes  ┌────────────────┐
#   │ ScenarioSpec│ ───────────────────────▶│ Matrix{F64} │ ────────────────────▶│ Vec{Vec{Leaf…}}│
#   └─────────────┘                         └─────────────┘                       └────────┬───────┘
#                                                                                          │ run_campaign
#                                                                                          ▼
#                                                                                ┌──────────────────┐
#                                                                                │ Vector{Variant…} │
#                                                                                └────────┬─────────┘
#                                                                                         │
#                                                                                         ▼
#                                                                                ┌──────────────────┐
#                                                                                │  ScenarioResult  │
#                                                                                └──────────────────┘
#
# The SSDashboard `CampaignSpec` API (Phase 1) is unchanged and still useful
# for spec exchange; bridging xlsx-CampaignSpec → ScenarioSpec is a separate
# concern (`cell_to_leaf_target` lookup table) that lives in a future phase.
# =============================================================================

using Random
using Sobol

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

"""
    LeafTarget(field, indices; type=:set, min, max, step=nothing, label="")

One leaf-parameter coordinate that a scenario campaign perturbs. The
`(field, indices)` pair maps directly to `ModelData.params.<field>[indices]`
(same convention as [`LeafChange`](@ref)).

* `type`  — `:set` overrides the existing value with the sampled value;
            `:multiply` multiplies the existing value by the sampled value.
* `min`   — sampling lower bound.
* `max`   — sampling upper bound (must be ≥ min).
* `step`  — discretisation step (only used by the `:factorial` sampler).
* `label` — human-readable name used in result tables and plots. Defaults to
            `"\$field\$indices"` when empty.

```julia
LeafTarget(:emissionTargetBunker, (:NL, 2050); type=:multiply, min=0.5, max=1.5)
```
"""
struct LeafTarget
    field::Symbol
    indices::Tuple
    type::Symbol
    min::Float64
    max::Float64
    step::Union{Float64,Nothing}
    label::String
end

function LeafTarget(field::Symbol, indices;
                    type::Symbol = :set,
                    min::Real,
                    max::Real,
                    step::Union{Real,Nothing} = nothing,
                    label::AbstractString = "")
    type in (:set, :multiply) ||
        throw(ArgumentError("LeafTarget.type must be :set or :multiply, got $(repr(type))"))
    min_f = Float64(min); max_f = Float64(max)
    max_f >= min_f ||
        throw(ArgumentError("LeafTarget bounds: max ($max_f) must be ≥ min ($min_f)"))
    idx = indices isa Tuple ? indices : (indices,)
    step_f = step === nothing ? nothing : Float64(step)
    if step_f !== nothing
        step_f > 0 || throw(ArgumentError("LeafTarget.step must be positive, got $step_f"))
    end
    lbl = isempty(label) ? string(field, idx) : String(label)
    return LeafTarget(field, idx, type, min_f, max_f, step_f, lbl)
end

"""
    ScenarioSpec(; name, method, n_variants, seed, targets)

A programmatic scenario-space exploration spec. Independent of any workbook
or (sheet, cell) coordinate; each [`LeafTarget`](@ref) references a leaf
parameter on `ModelData.params` directly.

* `name`        — campaign label (used by persistence as the output folder name).
* `method`      — `:lhs | :sobol | :morris | :factorial`.
* `n_variants`  — number of variants to generate (factorial ignores this and
                  uses the Cartesian product over discretised target ranges).
* `seed`        — RNG seed for reproducibility.
* `targets`     — non-empty vector of [`LeafTarget`](@ref)s.
"""
struct ScenarioSpec
    name::String
    method::Symbol
    n_variants::Int
    seed::Int
    targets::Vector{LeafTarget}
end

function ScenarioSpec(; name::AbstractString,
                       method,
                       n_variants::Integer,
                       seed::Integer,
                       targets::AbstractVector{LeafTarget})
    isempty(targets) && throw(ArgumentError("ScenarioSpec.targets must be non-empty"))
    n_variants > 0 || throw(ArgumentError("ScenarioSpec.n_variants must be positive, got $n_variants"))
    m = parse_sampling_method(method)
    return ScenarioSpec(String(name), m, Int(n_variants), Int(seed), collect(targets))
end

"""
    ScenarioResult(spec, samples, variants, runtime_seconds)

Bundles a complete `run_scenario_space` execution: the spec that was run, the
sample matrix that was generated, the per-variant solve results, and the
wall-clock runtime of the campaign (excludes sampling cost).

* `spec`            — the [`ScenarioSpec`](@ref) that was run.
* `samples`         — `n_variants × n_targets` matrix of sampled values.
                      `samples[i, j]` is the value of `spec.targets[j]` for
                      variant `i`.
* `variants`        — `Vector{VariantResult}` (length == `n_variants`).
* `runtime_seconds` — total wall-clock cost of the `run_campaign` call,
                      including distributed bootstrap/teardown when relevant.
"""
struct ScenarioResult
    spec::ScenarioSpec
    samples::Matrix{Float64}
    variants::Vector{VariantResult}
    runtime_seconds::Float64
end

# -----------------------------------------------------------------------------
# Sampling — operates directly on bounds (lighter than `sample_campaign`,
# which needs a CampaignSpec). Mirrors the logic in `workflows/scenario_space/sampling.jl`.
# -----------------------------------------------------------------------------

"""
    sample_scenario_space(spec::ScenarioSpec) -> Matrix{Float64}

Generate the sample matrix for `spec`. The matrix has `spec.n_variants` rows
(or the cartesian-product size for factorial) and `length(spec.targets)`
columns. Dispatches on `spec.method`.
"""
function sample_scenario_space(spec::ScenarioSpec)
    if spec.method == :lhs
        return _sample_lhs_targets(spec)
    elseif spec.method == :sobol
        return _sample_sobol_targets(spec)
    elseif spec.method == :morris
        return _sample_morris_targets(spec)
    elseif spec.method == :factorial
        return _sample_factorial_targets(spec)
    else
        throw(ArgumentError("Unsupported sampling method: $(spec.method)"))
    end
end

function _sample_lhs_targets(spec::ScenarioSpec)
    rng = MersenneTwister(spec.seed)
    n = spec.n_variants; k = length(spec.targets)
    values = Matrix{Float64}(undef, n, k)
    for j in 1:k
        t = spec.targets[j]
        u = rand(rng, n)
        perm = randperm(rng, n)
        for i in 1:n
            stratum = perm[i] - 1
            ti = (stratum + u[i]) / n
            values[i, j] = t.min + ti * (t.max - t.min)
        end
    end
    return values
end

function _sample_sobol_targets(spec::ScenarioSpec)
    n = spec.n_variants; k = length(spec.targets)
    s = SobolSeq(k)
    next!(s)  # skip the all-zeros origin
    values = Matrix{Float64}(undef, n, k)
    for i in 1:n
        u = next!(s)
        for j in 1:k
            t = spec.targets[j]
            values[i, j] = t.min + u[j] * (t.max - t.min)
        end
    end
    return values
end

function _sample_morris_targets(spec::ScenarioSpec)
    # Radial Morris design (Saltelli 2008). One trajectory = k+1 points.
    rng = MersenneTwister(spec.seed)
    n = spec.n_variants; k = length(spec.targets)
    p = 4                                # levels — matches sampling.jl default
    delta = p / (2 * (p - 1))
    grid = collect(range(0.0, stop = 1.0 - delta, length = p ÷ 2))
    pts_per = k + 1
    n % pts_per == 0 || throw(ArgumentError(
        "Morris sampling requires complete trajectories: n_variants ($n) " *
        "must be divisible by k + 1 ($pts_per)"))
    r = div(n, pts_per)
    rows = Vector{Vector{Float64}}()
    for _ in 1:r
        base = [rand(rng, grid) for _ in 1:k]
        push!(rows, copy(base))
        order = randperm(rng, k)
        for j in order
            base[j] = base[j] + delta <= 1.0 ? base[j] + delta : base[j] - delta
            push!(rows, copy(base))
        end
    end
    values = Matrix{Float64}(undef, n, k)
    for i in 1:n, j in 1:k
        t = spec.targets[j]
        values[i, j] = t.min + rows[i][j] * (t.max - t.min)
    end
    return values
end

function _sample_factorial_targets(spec::ScenarioSpec)
    grids = Vector{Vector{Float64}}(undef, length(spec.targets))
    for (j, t) in enumerate(spec.targets)
        t.step === nothing &&
            throw(ArgumentError("Factorial sampling requires a step for every target — target '$(t.label)' has none."))
        # collect(min:step:max) — guards against floating-point overshoot
        g = Float64[]; v = t.min
        while v <= t.max + 1e-12
            push!(g, v); v += t.step
        end
        # Make sure the upper bound is included even if step doesn't land on it.
        if !isempty(g) && abs(g[end] - t.max) > 1e-9 && g[end] < t.max
            push!(g, t.max)
        end
        grids[j] = g
    end
    n = prod(length, grids)
    values = Matrix{Float64}(undef, n, length(spec.targets))
    # Cartesian-product iteration: column 1 varies fastest.
    counters = ones(Int, length(spec.targets))
    for i in 1:n
        for (j, g) in enumerate(grids)
            values[i, j] = g[counters[j]]
        end
        # increment counters (column 1 fastest)
        c = 1
        while c <= length(counters)
            counters[c] += 1
            if counters[c] > length(grids[c])
                counters[c] = 1; c += 1
            else
                break
            end
        end
    end
    return values
end

# -----------------------------------------------------------------------------
# Bridge to LeafChange
# -----------------------------------------------------------------------------

"""
    samples_to_changes(spec, samples) -> Vector{Vector{LeafChange}}

Convert a sample matrix from [`sample_scenario_space`](@ref) into the
per-variant [`LeafChange`](@ref) list that [`run_campaign`](@ref) consumes.
"""
function samples_to_changes(spec::ScenarioSpec, samples::AbstractMatrix{<:Real})
    n, k = size(samples)
    k == length(spec.targets) || throw(DimensionMismatch(
        "samples has $k columns but spec has $(length(spec.targets)) targets"))
    out = Vector{Vector{LeafChange}}(undef, n)
    for i in 1:n
        out[i] = [LeafChange(t.field, t.indices, samples[i, j], t.type)
                  for (j, t) in enumerate(spec.targets)]
    end
    return out
end

# -----------------------------------------------------------------------------
# Orchestrator entry point
# -----------------------------------------------------------------------------

"""
    run_scenario_space(base_md, spec; kwargs...) -> ScenarioResult

End-to-end campaign execution. Generates samples, expands them into
[`LeafChange`](@ref) lists, and runs [`run_campaign`](@ref) on top of
`base_md`.

Keyword arguments forward to `run_campaign`:

* `n_workers::Int = 0`           — `0` or `1` selects the serial (single-model
                                   warm-reuse) path; `≥2` selects the
                                   distributed worker pool.
* `threads_per_worker::Int = 1`  — solver thread count per worker.
* `solver::Symbol = :highs`      — `:highs` or `:gurobi`.
* `solver_attrs::AbstractDict`   — extra solver attributes (merged on top of
                                   campaign defaults). For example
                                   `Dict("Method"=>2, "Crossover"=>-1)` for
                                   Gurobi BarrierCrossover.
* `mode::Symbol = :ts`           — `:ts | :fh | :annual`.
* `cancel::Ref{Bool} = Ref(false)` — set `cancel[] = true` to halt.
* `on_progress` / `on_result`    — optional callbacks (see `run_campaign`).

Returns a [`ScenarioResult`](@ref) bundling the spec, sample matrix, variant
results, and total runtime.
"""
function run_scenario_space(base_md, spec::ScenarioSpec;
                            n_workers::Int = 0,
                            threads_per_worker::Int = 1,
                            solver::Symbol = :highs,
                            solver_attrs::AbstractDict = Dict{String,Any}(),
                            mode::Symbol = :ts,
                            cancel::Ref{Bool} = Ref(false),
                            on_progress = _noop_progress,
                            on_result   = _noop_result)
    samples = sample_scenario_space(spec)
    n_variants = size(samples, 1)
    @info "run_scenario_space: samples generated" name=spec.name method=spec.method n_variants=n_variants n_targets=length(spec.targets)
    changes_per_variant = samples_to_changes(spec, samples)
    t0 = time()
    variants = run_campaign(base_md, changes_per_variant;
                            n_workers = n_workers,
                            threads_per_worker = threads_per_worker,
                            solver = solver,
                            solver_attrs = solver_attrs,
                            mode = mode,
                            cancel = cancel,
                            on_progress = on_progress,
                            on_result   = on_result)
    runtime = time() - t0
    return ScenarioResult(spec, samples, variants, runtime)
end
