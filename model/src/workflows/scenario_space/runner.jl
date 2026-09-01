# =============================================================================
# workflows/scenario_space/runner.jl — Per-variant runner (serial + Distributed.jl)
#
# Phase 3 of scenario-space exploration. Takes a base `ModelData` that has
# already been derived + clustered, a list of `LeafChange` lists (one per
# variant), and returns a `Vector{VariantResult}` with objective, status, and
# timing for each variant.
#
# Two execution modes (chosen via `n_workers`):
#   * `n_workers <= 1` → serial path. Builds the LP once in this process,
#     applies each variant via `apply_variant!`, solves, captures result.
#     This is the path used by the unit tests and by small / debug campaigns.
#   * `n_workers >= 2` → distributed path. `addprocs(n_workers)`, `@everywhere
#     using IESAOpt`, each worker holds its own model + ModelData copy and
#     runs a long-lived loop pulling variants from a `RemoteChannel`. Master
#     collects results on a bounded result channel for backpressure. After
#     the last variant, master sends N "poison pill" sentinels then
#     `rmprocs(workers)` to fully release Gurobi.Env + JuMP memory.
#
# Solver lifecycle (single-machine, unlimited Gurobi seats assumed):
#   * One JuMP model per worker (or in master for serial), built ONCE with
#     `apply_lp_generation_speedups!(m; keep_names=true)` so `constraint_by_name`
#     can find the emission-cap base names.
#   * `apply_variant!` mutates RHS/coef in place; warm-start basis is retained.
#   * No `JuMP.empty!` between variants.
# =============================================================================

using Distributed
using Dates

"""
    VariantResult(; variant_id, leaf_values, objective, term_status,
                    primal_status, build_seconds, apply_seconds,
                    solve_seconds, error)

Outcome of one variant solve. `leaf_values` is the vector of *effective* leaf
parameter values used for this variant (post-`:multiply`), aligned with the
`LeafChange` list the caller passed in. `error === nothing` for successful
solves; on failure it carries a short stringified message.
"""
Base.@kwdef struct VariantResult
    variant_id::Int
    leaf_values::Vector{Float64}     = Float64[]
    objective::Float64               = NaN
    co2_price::Float64               = NaN
    term_status::String              = ""
    primal_status::String            = ""
    build_seconds::Float64           = 0.0
    apply_seconds::Float64           = 0.0
    solve_seconds::Float64           = 0.0
    output_seconds::Float64          = 0.0
    output_path::Union{String,Nothing} = nothing
    worker_pid::Int                  = 1
    error::Union{String,Nothing}     = nothing
end

function _variant_co2_price(model::JuMP.Model, md::ModelData)
    candidate_names = ps -> String[
        "emTargetAir[NL,$(ps)]",
        "emTargetInclScope3FuelEx[$(ps)]",
        "emTargetInclScope3[$(ps)]",
        "emTargetAll[NL,$(ps)]",
        "emTargetBunker[NL,$(ps)]",
        "emTargetFS[NL,$(ps)]",
    ]
    prices = Float64[]
    for ps in md.sets.periods_solve
        for name in candidate_names(ps)
            con = try
                JuMP.constraint_by_name(model, name)
            catch
                nothing
            end
            con === nothing && continue
            price = try
                abs(JuMP.shadow_price(con))
            catch
                NaN
            end
            isfinite(price) || continue
            push!(prices, price)
            break
        end
    end
    isempty(prices) && return NaN
    return sum(prices) / length(prices)
end

# Default no-op callbacks used by `run_campaign`. Defined before
# `run_campaign` so default-arg expressions resolve at definition time.
_noop_progress(_x) = nothing
_noop_result(_x) = nothing
_noop_phase(_x) = nothing

const _DEFAULT_HIGHS_ATTRS_CAMPAIGN = Dict{String,Any}(
    "presolve"          => "on",
    "solver"            => "simplex",
    "parallel"          => "off",
    "output_flag"       => false,
)

"""
    _campaign_optimizer(solver::Symbol, threads::Int; attrs_override, rep_days) -> JuMP-optimizer-factory

Build a per-worker optimizer factory. `solver` is `:highs` (default,
license-free) or `:gurobi` (requires Gurobi.jl + GUROBI_HOME). `threads` is
the per-instance thread cap; for parallel campaigns set this low (e.g. 1)
so workers do not oversubscribe the CPU.  `attrs_override::AbstractDict`
is merged on top of the campaign defaults so callers can override one or
two solver tunings (e.g. `"Crossover" => -1` for Gurobi barrier+crossover).
For Gurobi time-slice campaigns, `rep_days` applies the same representative-day
tuned defaults used by single runs and the UI.
"""
function _campaign_optimizer(solver::Symbol, threads::Int;
                             attrs_override::AbstractDict = Dict{String,Any}(),
                             rep_days::Union{Nothing,Integer} = nothing)
    if solver === :highs
        attrs = copy(_DEFAULT_HIGHS_ATTRS_CAMPAIGN)
        # HiGHS uses "threads" if positive; ignored otherwise.
        threads > 0 && (attrs["threads"] = threads)
        for (k, v) in attrs_override
            attrs[k] = v
        end
        return highs_optimizer(; attrs = attrs)
    elseif solver === :gurobi
        attrs = _campaign_gurobi_attributes(threads, rep_days, attrs_override)
        return gurobi_optimizer(; attrs = attrs)
    else
        throw(ArgumentError("Unknown solver `$solver`; expected :highs or :gurobi."))
    end
end

function _campaign_gurobi_attributes(threads::Int,
                                     rep_days::Union{Nothing,Integer},
                                     attrs_override::AbstractDict = Dict{String,Any}())
    attrs = default_gurobi_attributes(; threads = max(0, threads), rep_days = rep_days)
    attrs["OutputFlag"] = 0
    for (k, v) in attrs_override
        attrs[k] = v
    end
    return attrs
end

"""
    _build_campaign_model(md; solver, threads, mode) -> JuMP.Model

Build the LP for `md` with constraint names preserved so the manifest can
look them up. Caller must have already run `derive_sets!`, `compute_derived_params!`,
and (for `mode === :ts`) `build_temporal_clusters!`.

!!! note "Per-variant clustering (Phase 3.5)"
    `build_temporal_clusters!` is run ONCE on `base_md` before each
    *clustering group* (see [`_partition_variants_by_cluster_key`](@ref)),
    not once per variant. Variants whose `LeafChange` lists only touch
    scalar leaves all share the same group and pay one cluster build for
    the whole campaign — bit-for-bit equivalent to the Phase 3 fast path.

    Variants that mutate a leaf tagged via
    [`register_clustering_affecting!`](@ref) are partitioned into separate
    groups (one per unique `(field, indices, value, type)` sub-state) and
    each group pays its own cluster + LP rebuild. N variants drawing from
    only K << N unique profile sets therefore cost K cluster builds, not N.
"""
function _build_campaign_model(md::ModelData; solver::Symbol, threads::Int,
                               mode::Symbol,
                               attrs_override::AbstractDict = Dict{String,Any}())
    rep_days = mode === :ts ? md.params.n_repDays : nothing
    m = Model(_campaign_optimizer(solver, threads;
        attrs_override = attrs_override,
        rep_days = rep_days))
    # Scenario-space MUST keep names so constraint_by_name works.
    apply_lp_generation_speedups!(m; keep_names = true)
    vars = if mode === :ts
        build_ts_lp!(m, md)
    elseif mode === :fh
        build_fh_lp!(m, md)
    elseif mode === :annual
        build_annual_lp!(m, md)
    else
        throw(ArgumentError("Unknown mode `$mode`; expected :ts, :fh, or :annual."))
    end
    m.ext[:iesa_vars] = vars
    return m
end

"""
    _run_one_variant!(model, md, changes, variant_id) -> VariantResult

Apply one variant's `LeafChange` list to a pre-built model + ModelData and
solve. Caller is responsible for `deepcopy(md)` if multiple workers share the
base.
"""
function _run_one_variant!(model::JuMP.Model, md::ModelData,
                           changes::AbstractVector{LeafChange},
                           variant_id::Int)
    try
        out = nothing
        t_apply = @elapsed begin
            out = apply_variant!(model, md, changes; rederive = true)
        end
        t_solve = @elapsed optimize!(model)
        term = string(termination_status(model))
        prim = string(primal_status(model))
        obj  = (term == "OPTIMAL") ? objective_value(model) : NaN
        co2p = (term == "OPTIMAL") ? _variant_co2_price(model, md) : NaN
        return VariantResult(
            variant_id     = variant_id,
            leaf_values    = collect(out.values),
            objective      = obj,
            co2_price      = co2p,
            term_status    = term,
            primal_status  = prim,
            apply_seconds  = t_apply,
            solve_seconds  = t_solve,
        )
    catch err
        return VariantResult(
            variant_id  = variant_id,
            error       = sprint(showerror, err),
            term_status = "ERROR",
        )
    end
end

# -----------------------------------------------------------------------------
# Phase 3.5 — per-variant clustering with a shared template per cluster group
# -----------------------------------------------------------------------------
#
# When NO leaves are tagged via `register_clustering_affecting!` (the
# default), every variant's cluster key is the empty tuple `()`, every
# variant lands in the same group, and the runner falls back to a single
# template + single LP build — bit-for-bit equivalent to the Phase 3 fast
# path.
#
# When some leaves ARE tagged (e.g. `:hourly_profilesReadOrig`), variants
# are partitioned by the tuple of `(field, indices, value)` for their
# clustering-affecting subset. Each group pays exactly ONE cluster build +
# ONE LP build; variants within a group warm-apply the scalar remainder of
# their LeafChange list on the shared model (the existing apply_variant!
# fast path).

"""
    _cluster_cache_key(changes::AbstractVector{LeafChange}) -> Tuple

Stable, hashable key identifying the clustering-affecting sub-state of a
variant. Empty tuple `()` when no leaves are tagged (or when none of the
variant's changes touch a tagged leaf). Sorting the tuple makes the key
invariant under permutation of `changes`, so two variants with the same
profile mutations always share a group regardless of input order.
"""
function _cluster_cache_key(changes::AbstractVector{LeafChange})
    # Fast path: when nothing is tagged, every variant hashes to `()`.
    isempty(CLUSTERING_AFFECTING_FIELDS) && return ()
    rel = Tuple{Symbol,Tuple,Float64,Symbol}[]
    for c in changes
        is_clustering_affecting(c.field) || continue
        push!(rel, (c.field, c.indices, c.value, c.type))
    end
    isempty(rel) && return ()
    sort!(rel; by = x -> (x[1], string(x[2]), x[3], x[4]))
    return Tuple(rel)
end

"""
    _partition_variants_by_cluster_key(changes_per_variant) -> (ordered_keys, groups)

Partition variants into groups that share the same clustering-affecting
sub-state. `ordered_keys::Vector` preserves first-seen order so log output
is deterministic; `groups::Dict{Any,Vector{Int}}` maps key → original
variant ids. With no tagged leaves the result is `([()], Dict(() => 1:n))`
— one group, no overhead.
"""
function _partition_variants_by_cluster_key(changes_per_variant::AbstractVector)
    ordered_keys = Any[]
    groups = Dict{Any,Vector{Int}}()
    for (i, ch) in pairs(changes_per_variant)
        k = _cluster_cache_key(ch)
        if !haskey(groups, k)
            push!(ordered_keys, k)
            groups[k] = Int[]
        end
        push!(groups[k], i)
    end
    return ordered_keys, groups
end

"""
    _split_cluster_changes(changes) -> (cluster_changes, scalar_changes)

Partition a variant's `LeafChange` list into the subset that affects
temporal clustering (already applied to the group's template `md`) and the
scalar subset that still needs to flow through `apply_variant!` on every
solve. Preserves the original order within each subset.
"""
function _split_cluster_changes(changes::AbstractVector{LeafChange})
    isempty(CLUSTERING_AFFECTING_FIELDS) && return (LeafChange[], collect(changes))
    cluster_changes = LeafChange[]
    scalar_changes  = LeafChange[]
    for c in changes
        if is_clustering_affecting(c.field)
            push!(cluster_changes, c)
        else
            push!(scalar_changes, c)
        end
    end
    return (cluster_changes, scalar_changes)
end

"""
    _build_cluster_template(base_md, cluster_changes; solver, threads, mode, attrs_override)
        -> (md_template, model)

Build one (`md_template`, `model`) pair for a cluster group. When
`cluster_changes` is empty this returns `deepcopy(base_md)` + an LP built
from it — identical to Phase 3's single-template behaviour. Otherwise it
deep-copies the base, applies just the clustering-affecting leaves,
re-runs `compute_derived_params!`, `build_temporal_clusters!`, and finally
[`_build_campaign_model`](@ref).
"""
function _build_cluster_template(base_md::ModelData,
                                 cluster_changes::AbstractVector{LeafChange};
                                 solver::Symbol, threads::Int, mode::Symbol,
                                 attrs_override::AbstractDict)
    md_template = deepcopy(base_md)
    if !isempty(cluster_changes)
        apply_leaf_changes!(md_template, cluster_changes)
        compute_derived_params!(md_template)
        if mode === :ts
            build_temporal_clusters!(md_template)
        end
    end
    model = _build_campaign_model(md_template; solver = solver,
                                  threads = threads, mode = mode,
                                  attrs_override = attrs_override)
    return md_template, model
end

"""
    _run_one_variant_filtered!(model, md_template, full_changes, scalar_changes,
                                cluster_changes, variant_id) -> VariantResult

Like [`_run_one_variant!`](@ref) but only the `scalar_changes` flow through
`apply_variant!`; `cluster_changes` were baked into `md_template` by the
caller. The `leaf_values` field of the result is assembled in the order of
`full_changes`, with values read straight off `md_template` for the cluster
leaves (since they are already mutated there) and from the apply_variant!
output for the scalar leaves.
"""
function _run_one_variant_filtered!(model::JuMP.Model, md_template::ModelData,
                                    full_changes::AbstractVector{LeafChange},
                                    scalar_changes::AbstractVector{LeafChange},
                                    variant_id::Int)
    try
        md = deepcopy(md_template)
        t_apply = @elapsed begin
            out = apply_variant!(model, md, scalar_changes; rederive = true)
        end
        # Reassemble leaf_values in `full_changes` order. Cluster leaves were
        # already mutated into md_template by _build_cluster_template, so we
        # read them straight back. Scalar leaves come from apply_variant!.
        leaf_vals = Vector{Float64}(undef, length(full_changes))
        scalar_iter = 1
        @inbounds for (i, ch) in pairs(full_changes)
            if is_clustering_affecting(ch.field)
                d = _get_param_dict(md, ch.field)
                k = _scalar_key(ch.indices)
                leaf_vals[i] = get(d, k, NaN)
            else
                leaf_vals[i] = out.values[scalar_iter]
                scalar_iter += 1
            end
        end
        t_solve = @elapsed optimize!(model)
        term = string(termination_status(model))
        prim = string(primal_status(model))
        obj  = (term == "OPTIMAL") ? objective_value(model) : NaN
        co2p = (term == "OPTIMAL") ? _variant_co2_price(model, md) : NaN
        return VariantResult(
            variant_id     = variant_id,
            leaf_values    = leaf_vals,
            objective      = obj,
            co2_price      = co2p,
            term_status    = term,
            primal_status  = prim,
            apply_seconds  = t_apply,
            solve_seconds  = t_solve,
        )
    catch err
        return VariantResult(
            variant_id  = variant_id,
            error       = sprint(showerror, err),
            term_status = "ERROR",
        )
    end
end

# -----------------------------------------------------------------------------
# Serial path
# -----------------------------------------------------------------------------
"""
    _run_campaign_serial(base_md, changes_per_variant; solver, threads, mode,
                         on_progress, on_result, cancel) -> Vector{VariantResult}

Single-process loop. Variants are partitioned into clustering groups via
[`_partition_variants_by_cluster_key`](@ref); each group pays one
deepcopy + one cluster build (if needed) + one LP build, then warm-applies
its scalar variants on the shared model. With no clustering-affecting
leaves registered (default state), there is exactly one group containing
every variant and the loop collapses to Phase 3's behaviour.
"""
function _run_campaign_serial(base_md::ModelData,
                              changes_per_variant::AbstractVector;
                              solver::Symbol, threads::Int, mode::Symbol,
                              attrs_override::AbstractDict,
                              on_progress::Function, on_result::Function,
                              cancel::Ref{Bool})
    n = length(changes_per_variant)
    results = Vector{VariantResult}(undef, n)
    ordered_keys, groups = _partition_variants_by_cluster_key(changes_per_variant)
    n_groups = length(ordered_keys)
    n_groups > 1 && @info "_run_campaign_serial: per-variant clustering active" n_groups n_variants=n

    pid = Distributed.myid()
    @inbounds for (gi, key) in enumerate(ordered_keys)
        vids = groups[key]
        # Representative variant — all in the group share its cluster_changes.
        rep_changes = changes_per_variant[vids[1]]
        cluster_changes, _ = _split_cluster_changes(rep_changes)
        md_template, model = nothing, nothing
        t_build = @elapsed begin
            md_template, model = _build_cluster_template(base_md, cluster_changes;
                solver = solver, threads = threads, mode = mode,
                attrs_override = attrs_override)
        end
        n_groups > 1 && @info "  group $gi/$n_groups" key=key n_variants=length(vids) t_build=t_build

        for (k, vid) in pairs(vids)
            if cancel[]
                results[vid] = VariantResult(
                    variant_id  = vid,
                    term_status = "CANCELLED",
                    error       = "Campaign cancelled before variant $vid.",
                    worker_pid  = pid)
                continue
            end
            on_progress((variant_id = vid, total = n, stage = "start", worker_pid = pid))
            full_changes = changes_per_variant[vid]
            _, scalar_changes = _split_cluster_changes(full_changes)
            r = _run_one_variant_filtered!(model, md_template, full_changes,
                                           scalar_changes, vid)
            # Stamp worker_pid; record build_seconds on the FIRST variant
            # of the FIRST group (matches Phase 3 semantics when there's
            # only one group; otherwise each group's first variant pays).
            r = VariantResult(
                variant_id = r.variant_id, leaf_values = r.leaf_values,
                objective = r.objective, term_status = r.term_status,
                primal_status = r.primal_status,
                co2_price = r.co2_price,
                build_seconds = (k == 1 ? t_build : 0.0),
                apply_seconds = r.apply_seconds, solve_seconds = r.solve_seconds,
                worker_pid = pid,
                error = r.error)
            results[vid] = r
            on_result(r)
            on_progress((variant_id = vid, total = n, stage = "done", result = r))
        end
    end
    return results
end

# -----------------------------------------------------------------------------
# Distributed path
# -----------------------------------------------------------------------------
"""
    _worker_loop(task_ch, result_ch, base_md, solver, threads, mode)

Long-running per-worker loop. Pulls `(variant_id, changes)` tuples from
`task_ch`; pushes `VariantResult` to `result_ch`. Terminates when it
receives `nothing` (poison pill).

Maintains a per-worker cluster cache `Dict{cluster_key, (md, model)}` so
each unique clustering-affecting sub-state on this worker costs ONE
deepcopy + ONE cluster build + ONE LP build, no matter how many variants
share that key. With no clustering-affecting leaves registered (the
default), every variant hashes to `()` and the cache has exactly one entry
— same behaviour as Phase 3.
"""
function _worker_loop(task_ch::RemoteChannel, result_ch::RemoteChannel,
                      base_md::ModelData,
                      solver::Symbol, threads::Int, mode::Symbol,
                      attrs_override::AbstractDict)
    # cluster_key -> (md_template, model)
    cache = Dict{Any,Tuple{ModelData,JuMP.Model}}()
    pid = Distributed.myid()
    n_done = 0
    try
        while true
            item = take!(task_ch)
            item === nothing && break  # poison pill — clean shutdown
            variant_id, changes = item
            put!(result_ch, (variant_id = variant_id, worker_pid = pid, stage = "start"))
            key = _cluster_cache_key(changes)
            t_build_this = 0.0
            entry = get(cache, key, nothing)
            if entry === nothing
                cluster_changes, _ = _split_cluster_changes(changes)
                t_build_this = @elapsed begin
                    md_t, model_t = _build_cluster_template(base_md, cluster_changes;
                        solver = solver, threads = threads, mode = mode,
                        attrs_override = attrs_override)
                end
                cache[key] = (md_t, model_t)
                entry = (md_t, model_t)
            end
            md, model = entry
            _, scalar_changes = _split_cluster_changes(changes)
            r = _run_one_variant_filtered!(model, md, changes, scalar_changes, variant_id)
            # Stamp worker_pid on every result; record build_seconds on the
            # variant that paid for the cluster-group build (cache miss).
            r = VariantResult(
                variant_id = r.variant_id, leaf_values = r.leaf_values,
                objective = r.objective, term_status = r.term_status,
                primal_status = r.primal_status,
                co2_price = r.co2_price,
                build_seconds = t_build_this,
                apply_seconds = r.apply_seconds, solve_seconds = r.solve_seconds,
                worker_pid = pid,
                error = r.error)
            n_done += 1
            put!(result_ch, r)
        end
    catch err
        # Push a synthetic failure so the master knows this worker died.
        put!(result_ch, VariantResult(
            variant_id  = -1,
            term_status = "WORKER_ERROR",
            worker_pid  = pid,
            error       = sprint(showerror, err),
        ))
    end
    return n_done
end

"""
    _run_campaign_distributed(base_md, changes_per_variant; n_workers,
                              threads_per_worker, solver, mode,
                              on_progress, on_result, cancel) -> Vector{VariantResult}

Spawn `n_workers` Julia worker processes, ship `base_md` once, then dispatch
variants over a bounded `RemoteChannel`. Cleans up workers (`rmprocs`) before
returning, releasing all model memory + Gurobi.Env handles.
"""
function _run_campaign_distributed(base_md::ModelData,
                                   changes_per_variant::AbstractVector;
                                   n_workers::Int, threads_per_worker::Int,
                                   solver::Symbol, mode::Symbol,
                                   attrs_override::AbstractDict,
                                   on_progress::Function, on_result::Function,
                                   on_phase::Function,
                                   cancel::Ref{Bool})
    n = length(changes_per_variant)
    # Reuse the env Julia process is already in (same Project.toml).
    project_path = Base.active_project()
    project_path === nothing && error("run_campaign: no active project; addprocs would not inherit IESAOpt.")
    project_dir = dirname(project_path)
    exeflags = ["--project=$project_dir", "--threads=$(max(1, threads_per_worker))"]
    @info "run_campaign: spawning $n_workers worker(s)" exeflags solver mode
    t_addprocs = @elapsed pids = addprocs(n_workers; exeflags = exeflags)
    @info "run_campaign: addprocs done" t_addprocs pids
    on_phase((phase = :addprocs, seconds = t_addprocs, pids = pids))
    try
        # Bootstrap workers with IESAOpt.  We cannot use `@everywhere using ...`
        # inside a function body (the macro expands to a top-level expression).
        # We also cannot send a closure that *references* IESAOpt before the
        # worker has loaded it (the closure deserializer needs the parent
        # module). Workaround: send a quoted Expr to `Main.eval` — `Main`
        # exists on every fresh worker.  We fan out via @async so the per-worker
        # `using IESAOpt` cost runs in parallel rather than sequentially.
        t_using = @elapsed begin
            @sync for p in pids
                @async remotecall_wait(Main.eval, p, :(using IESAOpt))
            end
        end
        @info "run_campaign: workers loaded IESAOpt" t_using
        on_phase((phase = :workers_loaded, seconds = t_using, pids = pids))

        # Bounded result channel — caps memory pressure on master under
        # a slow downstream consumer.
        task_cap = max(n_workers * 4, 16)
        res_cap  = max(n_workers * 4, 16)
        task_ch  = RemoteChannel(() -> Channel{Any}(task_cap))
        result_ch = RemoteChannel(() -> Channel{Any}(res_cap))

        # Start the worker loops.  This is where `base_md` gets serialized
        # and shipped to each worker — one ship per worker.  We fan out
        # via @async so the (sequential-by-default) shipping happens in
        # parallel rather than blocking the master per-worker.
        worker_futures = Future[]
        t_ship = @elapsed begin
            @sync for p in pids
                @async push!(worker_futures,
                      remotecall(IESAOpt._worker_loop, p, task_ch, result_ch,
                                 base_md, solver, threads_per_worker, mode,
                                 attrs_override))
            end
        end
        @info "run_campaign: base ModelData shipped to all workers" t_ship
        on_phase((phase = :ship_base_data, seconds = t_ship, pids = pids))

        # Producer: push variants then `nothing` x n_workers (poison pills).
        producer = @async begin
            try
                for i in 1:n
                    cancel[] && break
                    put!(task_ch, (i, changes_per_variant[i]))
                end
            finally
                for _ in 1:length(pids)
                    put!(task_ch, nothing)
                end
            end
        end

        # Collector: pull n completed results, plus lightweight start events
        # emitted by workers as soon as they take a task from the queue.
        results = Vector{VariantResult}(undef, n)
        n_collected = 0
        while n_collected < n
            item = take!(result_ch)
            if !(item isa VariantResult)
                if get(item, :stage, "") == "start"
                    on_progress((variant_id = get(item, :variant_id, 0),
                                 total = n,
                                 stage = "start",
                                 worker_pid = get(item, :worker_pid, 0)))
                end
                continue
            end
            r = item::VariantResult
            if r.variant_id <= 0 || r.variant_id > n
                # Worker error sentinel — log and break to avoid hanging.
                @warn "Campaign worker error" error = r.error
                # Replace any unfilled slot with a synthetic error result.
                for j in 1:n
                    isassigned(results, j) ||
                        (results[j] = VariantResult(variant_id = j,
                                                    term_status = "WORKER_ERROR",
                                                    error = r.error))
                end
                break
            end
            results[r.variant_id] = r
            on_result(r)
            on_progress((variant_id = r.variant_id, total = n, stage = "done", result = r))
            n_collected += 1
        end
        wait(producer)
        # Wait for worker loops to drain so rmprocs is clean.
        for f in worker_futures
            try; fetch(f); catch; end
        end
        # Fill any still-unassigned slot with a cancellation result.
        for j in 1:n
            isassigned(results, j) ||
                (results[j] = VariantResult(variant_id = j,
                                            term_status = "CANCELLED",
                                            error = "Variant $j not completed (cancelled or worker died)."))
        end
        return results
    finally
        try
            t_rmprocs = @elapsed rmprocs(pids; waitfor = 60)
            @info "run_campaign: rmprocs done" t_rmprocs
            on_phase((phase = :rmprocs, seconds = t_rmprocs, pids = pids))
        catch err
            @warn "rmprocs failed; workers may linger" err
            on_phase((phase = :rmprocs_failed, error = sprint(showerror, err), pids = pids))
        end
    end
end

# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------
"""
    run_campaign(base_md, changes_per_variant; kwargs...) -> Vector{VariantResult}

Top-level entry for Phase 3. `base_md::ModelData` must already have
`derive_sets!`, `compute_derived_params!`, and (for `mode === :ts`)
`build_temporal_clusters!` applied — `read_data_cached` returns a freshly
loaded ModelData; the campaign caller is responsible for the derivation pass.

`changes_per_variant::AbstractVector{<:AbstractVector{LeafChange}}` is one
list of `LeafChange` per variant. Index `i` of the input maps to
`VariantResult.variant_id == i` in the output (variant_id is 1-based).

Keyword arguments:
  * `n_workers::Int = 0` — `<= 1` runs in-process serial; `>= 2` spawns
    Julia worker processes via `Distributed.addprocs`. The serial path is
    much faster for tiny campaigns (no addprocs / `@everywhere using IESAOpt`
    bootstrap cost ~10-15 s on cold cache).
  * `threads_per_worker::Int = 1` — solver thread cap per process. With
    `n_workers * threads_per_worker > num_physical_cores` you will likely
    see oversubscription slowdowns. Default 1 keeps it conservative.
  * `solver::Symbol = :highs` — `:highs` (license-free, default) or `:gurobi`.
  * `solver_attrs::AbstractDict = Dict()` — extra solver attributes merged on
    top of the campaign defaults. For Gurobi barrier+crossover use
    `Dict("Method" => 2, "Crossover" => -1)`. For Gurobi barrier alone use
    `Dict("Method" => 2, "Crossover" => 0)`. Keys must match the solver-native
    option names (HiGHS uses lowercase strings, Gurobi uses MixedCase).
  * `mode::Symbol = :ts` — `:ts`, `:fh`, or `:annual`. Must match what
    `base_md` was prepared for (TS requires the cluster pass).
  * `cancel::Ref{Bool} = Ref(false)` — set to `true` from another task to
    request an orderly stop (master stops dispatching new variants).
    * `on_progress::Function = _noop_progress` — called as
        `on_progress((variant_id, total, stage, [worker_pid], [result]))` when a
        worker starts a variant and again after each variant finishes.
  * `on_result::Function = _noop_result` — called as `on_result(r::VariantResult)`
    for every completed variant; useful for streaming to disk / DuckDB.
"""
function run_campaign(base_md::ModelData,
                      changes_per_variant::AbstractVector;
                      n_workers::Int                = 0,
                      threads_per_worker::Int       = 1,
                      solver::Symbol                = :highs,
                      solver_attrs::AbstractDict    = Dict{String,Any}(),
                      mode::Symbol                  = :ts,
                      cancel::Ref{Bool}             = Ref(false),
                      on_progress::Function         = _noop_progress,
                      on_result::Function           = _noop_result,
                      on_phase::Function            = _noop_phase)
    n = length(changes_per_variant)
    n == 0 && return VariantResult[]
    @info "run_campaign: starting" n_variants=n n_workers=n_workers solver=solver mode=mode
    if n_workers <= 1
        return _run_campaign_serial(base_md, changes_per_variant;
                                    solver = solver, threads = threads_per_worker,
                                    mode = mode, attrs_override = solver_attrs,
                                    on_progress = on_progress,
                                    on_result = on_result, cancel = cancel)
    else
        return _run_campaign_distributed(base_md, changes_per_variant;
                                         n_workers = n_workers,
                                         threads_per_worker = threads_per_worker,
                                         solver = solver, mode = mode,
                                         attrs_override = solver_attrs,
                                         on_progress = on_progress,
                                         on_result = on_result, on_phase = on_phase,
                                         cancel = cancel)
    end
end

# (default no-op callbacks are defined near the top of this file)
