# =============================================================================
# workflows/scenario_space/analysis.jl -- Phase 5: lightweight result-analysis helpers
#
# Phase 4 hands us a `ScenarioResult` (spec + sample matrix + per-variant
# solves). Phase 5 provides three small DataFrame-returning helpers that cover
# the most common questions analysts ask:
#
#   1. objective_table(result)        — wide table: sample columns + objective +
#                                       solve metadata, one row per variant.
#   2. sensitivity_scan(result)       — Pearson correlation of each target vs
#                                       objective, ranked by |correlation|.
#   3. pareto_front(result, x, y)     — non-dominated subset over two columns
#                                       from objective_table.
#
# These are intentionally narrow: they don't pull in Plots/Makie or any
# statistics package beyond Statistics (stdlib). For richer SA — Morris EE,
# Sobol indices, etc. — use the GlobalSensitivity.jl ecosystem on the sample
# matrix + objective vector returned from objective_table.
# =============================================================================

using DataFrames
using Statistics: cor, std

# -----------------------------------------------------------------------------
# 1. objective_table
# -----------------------------------------------------------------------------

"""
    objective_table(result::ScenarioResult) -> DataFrame

Wide tidy view of a `ScenarioResult`: one row per variant, one column per
target (named by `LeafTarget.label`), plus solve metadata columns
(`objective`, `term_status`, `primal_status`, `worker_pid`, `build_seconds`,
`apply_seconds`, `solve_seconds`, `error`).

Variants with non-`OPTIMAL` termination keep their (possibly garbage)
objective; downstream helpers like [`sensitivity_scan`](@ref) and
[`pareto_front`](@ref) automatically filter on `term_status == "OPTIMAL"`.

```julia
df = objective_table(result)
filter(:term_status => ==("OPTIMAL"), df)
```
"""
function objective_table(result::ScenarioResult)
    spec = result.spec
    n = size(result.samples, 1)
    length(result.variants) == n || throw(ArgumentError(
        "ScenarioResult is inconsistent: $n samples but $(length(result.variants)) variants"))
    df = DataFrame()
    df.variant_id = [v.variant_id for v in result.variants]
    # Sample columns first — one per target, named by its label.
    for (j, t) in enumerate(spec.targets)
        col = Symbol(t.label)
        df[!, col] = result.samples[:, j]
    end
    # Solve metadata.
    df.objective     = [v.objective     for v in result.variants]
    df.term_status   = [v.term_status   for v in result.variants]
    df.primal_status = [v.primal_status for v in result.variants]
    df.worker_pid    = [v.worker_pid    for v in result.variants]
    df.build_seconds = [v.build_seconds for v in result.variants]
    df.apply_seconds = [v.apply_seconds for v in result.variants]
    df.solve_seconds = [v.solve_seconds for v in result.variants]
    df.error         = [v.error === nothing ? "" : String(v.error)
                        for v in result.variants]
    return df
end

# -----------------------------------------------------------------------------
# 2. sensitivity_scan
# -----------------------------------------------------------------------------

"""
    sensitivity_scan(result::ScenarioResult; min_optimal=3) -> DataFrame

Linear (Pearson) sensitivity scan of objective vs each target. Returns a
DataFrame with columns:

* `target`      — `LeafTarget.label`
* `correlation` — Pearson correlation coefficient on the optimal subset
* `abs_corr`    — `abs(correlation)` (also the sort key)
* `rank`        — 1 = most influential target by `abs_corr`
* `n_used`      — number of optimal variants the correlation was computed on

Filters to `term_status == "OPTIMAL"` first; throws if fewer than
`min_optimal` optimal variants remain. Pearson correlation is undefined when
a target has zero variance; in that case the row reports `NaN` and is ranked
last.
"""
function sensitivity_scan(result::ScenarioResult; min_optimal::Integer = 3)
    df = objective_table(result)
    opt = filter(:term_status => ==("OPTIMAL"), df)
    n = nrow(opt)
    n >= min_optimal || throw(ArgumentError(
        "sensitivity_scan needs at least $min_optimal optimal variants, " *
        "have $n (out of $(nrow(df))). " *
        "Increase n_variants or check solver settings."))
    y = Float64.(opt.objective)
    targets = result.spec.targets
    rows = NamedTuple{(:target, :correlation, :abs_corr, :n_used),
                      Tuple{String,Float64,Float64,Int}}[]
    for t in targets
        x = Float64.(opt[!, Symbol(t.label)])
        c = (std(x) > 0 && std(y) > 0) ? cor(x, y) : NaN
        push!(rows, (target = t.label,
                     correlation = c,
                     abs_corr = isnan(c) ? -1.0 : abs(c),
                     n_used = n))
    end
    out = DataFrame(rows)
    sort!(out, :abs_corr; rev = true)
    out.abs_corr = [isnan(r.correlation) ? NaN : r.abs_corr for r in eachrow(out)]
    out.rank = collect(1:nrow(out))
    return select(out, :target, :correlation, :abs_corr, :rank, :n_used)
end

# -----------------------------------------------------------------------------
# 3. pareto_front
# -----------------------------------------------------------------------------

"""
    pareto_front(result::ScenarioResult, x::Symbol, y::Symbol;
                 minimize = (true, true)) -> DataFrame

Non-dominated subset of `objective_table(result)` over the two columns
`(x, y)`. A row dominates another when it is no worse on both objectives and
strictly better on at least one. `minimize` is a 2-tuple of booleans — `true`
means "smaller is better" for that coordinate, `false` means "larger is
better". Both `x` and `y` must be present in the result's objective table
(typically `:objective` plus one of the target labels or solve-timing
columns).

Returns the rows of `objective_table` that lie on the Pareto frontier,
sorted by `x` ascending. Filters out non-`OPTIMAL` variants before computing
the front so infeasible/garbage points cannot dominate optimal ones.

```julia
# Trade off objective vs CO₂ price target
pareto_front(result, :objective, :co2_NL; minimize = (true, false))
```
"""
function pareto_front(result::ScenarioResult, x::Symbol, y::Symbol;
                      minimize::Tuple{Bool,Bool} = (true, true))
    df = objective_table(result)
    opt = filter(:term_status => ==("OPTIMAL"), df)
    hasproperty(opt, x) || throw(ArgumentError("Column $(repr(x)) not in objective_table; have $(propertynames(opt))"))
    hasproperty(opt, y) || throw(ArgumentError("Column $(repr(y)) not in objective_table; have $(propertynames(opt))"))
    n = nrow(opt)
    n == 0 && return opt  # nothing optimal -> empty front, same schema
    xs = minimize[1] ? Float64.(opt[!, x]) : -Float64.(opt[!, x])
    ys = minimize[2] ? Float64.(opt[!, y]) : -Float64.(opt[!, y])
    keep = trues(n)
    @inbounds for i in 1:n
        for j in 1:n
            i == j && continue
            # j dominates i if j <= i on both AND strictly < on at least one
            if xs[j] <= xs[i] && ys[j] <= ys[i] &&
               (xs[j] < xs[i] || ys[j] < ys[i])
                keep[i] = false
                break
            end
        end
    end
    front = opt[keep, :]
    sort!(front, x)
    return front
end
