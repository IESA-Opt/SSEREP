# =============================================================================
# workflows/scenario_space/manifest.jl — Constraint-reference manifest for in-place LP mutation
#
# Phase 2 of scenario-space exploration. Lets a campaign change a leaf
# parameter (e.g. emissionTargetAir[(:NL, 2050)]) and push the new value into
# the *already-built* JuMP model — no rebuild — by calling
# `set_normalized_rhs` / `set_normalized_coefficient` / `set_objective_coefficient`
# on the right constraint/variable references.
#
# Design:
#   * `Mutation` is a tiny POD describing one atomic edit (RHS / coef / obj).
#   * `MUTATION_REGISTRY` maps a leaf-parameter symbol (e.g. `:emissionTargetAir`)
#     to a builder function `(md, indices, new_value) -> Vector{Mutation}` that
#     knows which constraint base_name(s) consume that parameter.
#   * Constraints are looked up at apply time via JuMP's `constraint_by_name`,
#     keyed by the `base_name = "..."` string already used in the builders
#     (e.g. "emTargetAir[NL,2050]"). No builder refactor needed — this layer
#     reuses what `_extract_co2_prices` / `_extract_emission_prices` already do
#     in `ui_server.jl`.
#   * Default mutations are registered in `__init__` (see IESAOpt.jl) for the
#     leaf parameters that map 1:1 to a single constraint RHS:
#       - emissionTargetAir, emissionTargetBunker, emissionTargetFS,
#         emissionTargetAll, emissionTarget_inclScope3andFuelex,
#         CO2_cumulative_budget, cumulative_CO2storage
#
# Adding more parameters later:
#   `register_mutation!(:my_field, (md, idx, v) -> Mutation[...])`
#   In the builder you list every (constraint_name, var_name?, kind) tuple
#   touched by `(field, idx)`. For parameters consumed by
#   `compute_derived_params!`, the variant runner re-derives first and the
#   builder reads the post-derivation value from `md`.
# =============================================================================

"""
    Mutation(kind, constraint_name, var_name, new_value)

A single atomic edit to push into a built JuMP model.

`kind` is one of:
- `:rhs` — `set_normalized_rhs(constraint_by_name(model, constraint_name), new_value)`
- `:coef` — `set_normalized_coefficient(constraint_by_name(model, constraint_name),
            variable_by_name(model, var_name), new_value)`
- `:obj` — `set_objective_coefficient(model, variable_by_name(model, var_name), new_value)`

`constraint_name` is the exact string passed as `base_name` in the
`@constraint(model, expr, base_name = "...")` macro inside the builder.
For `:obj` mutations it should be `""`. For `:rhs` mutations `var_name`
should be `""`.
"""
struct Mutation
    kind::Symbol
    constraint_name::String
    var_name::String
    new_value::Float64
end

Mutation(; kind::Symbol, constraint_name::AbstractString = "",
         var_name::AbstractString = "", new_value::Real) =
    Mutation(kind, String(constraint_name), String(var_name), Float64(new_value))

const _ALLOWED_KINDS = (:rhs, :coef, :obj)

"""
    MutationBuilder

A function `(md::ModelData, indices::Tuple, new_value::Float64) -> Vector{Mutation}`
that turns one leaf-parameter override into the list of concrete constraint /
variable edits that must be pushed to the JuMP model.
"""
const MutationBuilder = Function

# Global registry: leaf-parameter symbol -> builder fn
const MUTATION_REGISTRY = Dict{Symbol,MutationBuilder}()

"""
    register_mutation!(field::Symbol, builder::Function)

Associate `builder(md, indices, new_value) -> Vector{Mutation}` with a
ModelParams leaf field. Replaces any previous registration. Returns nothing.
"""
function register_mutation!(field::Symbol, builder::Function)
    MUTATION_REGISTRY[field] = builder
    return nothing
end

"""
    is_mutation_registered(field::Symbol) -> Bool
"""
is_mutation_registered(field::Symbol) = haskey(MUTATION_REGISTRY, field)

"""
    registered_mutation_fields() -> Vector{Symbol}

Sorted list of every leaf-parameter symbol that currently has a registered
mutation builder.
"""
registered_mutation_fields() = sort!(collect(keys(MUTATION_REGISTRY)))

# -----------------------------------------------------------------------------
# Clustering-affecting registry (Phase 3.5)
#
# A leaf is "clustering-affecting" when changing its value invalidates the
# representative-day clustering computed by `build_temporal_clusters!`. The
# canonical example is `hourly_profilesReadOrig` (the 8760-hour read profiles
# clustering reads from). When a campaign variant touches such a leaf, the
# runner must re-cluster + rebuild the LP for that variant instead of
# warm-applying onto the pre-clustered base model.
#
# Default state: EMPTY. No leaves are tagged out of the box, which means the
# Phase 3 fast path is preserved bit-for-bit for any campaign that only
# perturbs scalar leaves (prices, caps, emission targets — every leaf
# registered by `_register_default_mutations!`). Tagging happens opt-in via
# `register_clustering_affecting!`, typically alongside a custom mutation
# builder for a profile leaf.
# -----------------------------------------------------------------------------

const CLUSTERING_AFFECTING_FIELDS = Set{Symbol}()

"""
    register_clustering_affecting!(field::Symbol)

Mark `field` as a leaf whose mutation invalidates the temporal clustering.
[`run_campaign`](@ref) will group variants by their clustering-affecting
sub-state and rebuild the cluster + LP once per group (Phase 3.5).
Idempotent. Returns nothing.

```julia
register_clustering_affecting!(:hourly_profilesReadOrig)
```
"""
function register_clustering_affecting!(field::Symbol)
    push!(CLUSTERING_AFFECTING_FIELDS, field)
    return nothing
end

"""
    unregister_clustering_affecting!(field::Symbol) -> Bool

Remove `field` from the clustering-affecting registry. Returns `true` if it
was present (and is now gone), `false` if it was not registered.
"""
function unregister_clustering_affecting!(field::Symbol)
    was_present = field in CLUSTERING_AFFECTING_FIELDS
    delete!(CLUSTERING_AFFECTING_FIELDS, field)
    return was_present
end

"""
    is_clustering_affecting(field::Symbol) -> Bool

`true` iff `field` is in the clustering-affecting registry.
"""
is_clustering_affecting(field::Symbol) = field in CLUSTERING_AFFECTING_FIELDS

"""
    clustering_affecting_fields() -> Vector{Symbol}

Sorted list of every leaf tagged as clustering-affecting.
"""
clustering_affecting_fields() = sort!(collect(CLUSTERING_AFFECTING_FIELDS))

"""
    variant_affects_clustering(changes) -> Bool

`true` iff any [`LeafChange`](@ref) in `changes` targets a clustering-
affecting field.
"""
variant_affects_clustering(changes) =
    any(c -> is_clustering_affecting(c.field), changes)

"""
    build_mutations(md, field::Symbol, indices, new_value) -> Vector{Mutation}

Look up the registered builder for `field` and call it. Throws an
`ArgumentError` if no builder is registered (use `register_mutation!`).
"""
function build_mutations(md, field::Symbol, indices, new_value::Real)
    haskey(MUTATION_REGISTRY, field) ||
        throw(ArgumentError("No mutation builder registered for parameter `$(field)`. " *
                            "Call `register_mutation!($(repr(field)), builder)` to add one."))
    idx_tuple = indices isa Tuple ? indices : (indices,)
    return MUTATION_REGISTRY[field](md, idx_tuple, Float64(new_value))
end

"""
    apply_mutation!(model::JuMP.Model, m::Mutation) -> Mutation

Push a single `Mutation` into a built JuMP model. The model must already have
been built; the constraint / variable referenced by `m` must exist by name
(see `constraint_by_name` / `variable_by_name`). Returns the mutation for
chaining. Throws `ErrorException` if the constraint / variable is not found.

**Requires constraint names to be preserved at build time.** Call
`apply_lp_generation_speedups!(model; keep_names = true)` *before*
`build_ts_lp!` / `build_fh_lp!` (or set `IESA_OPT_KEEP_NAMES=1`); otherwise
JuMP strips the `base_name=` strings on creation and `constraint_by_name`
returns `nothing` for every lookup.
"""
function apply_mutation!(model::JuMP.Model, m::Mutation)
    if m.kind === :rhs
        con = constraint_by_name(model, m.constraint_name)
        con === nothing &&
            error("apply_mutation!(:rhs): constraint `$(m.constraint_name)` not found in model. " *
                  "Either the model has not been built yet, or the registered builder produced a wrong name.")
        set_normalized_rhs(con, m.new_value)
    elseif m.kind === :coef
        con = constraint_by_name(model, m.constraint_name)
        con === nothing &&
            error("apply_mutation!(:coef): constraint `$(m.constraint_name)` not found in model.")
        var = variable_by_name(model, m.var_name)
        var === nothing &&
            error("apply_mutation!(:coef): variable `$(m.var_name)` not found in model.")
        set_normalized_coefficient(con, var, m.new_value)
    elseif m.kind === :obj
        var = variable_by_name(model, m.var_name)
        var === nothing &&
            error("apply_mutation!(:obj): variable `$(m.var_name)` not found in model.")
        set_objective_coefficient(model, var, m.new_value)
    else
        error("apply_mutation!: unknown kind `$(m.kind)`; expected one of $_ALLOWED_KINDS.")
    end
    return m
end

"""
    apply_mutations!(model::JuMP.Model, ms) -> Int

Apply every `Mutation` in `ms` to `model`. Returns the number applied.
"""
function apply_mutations!(model::JuMP.Model, ms)
    n = 0
    for m in ms
        apply_mutation!(model, m)
        n += 1
    end
    return n
end

# -----------------------------------------------------------------------------
# Default registrations
#
# Called once by `IESAOpt.__init__` so the registry is populated whether the
# package is loaded from a precompiled image or fresh. Every leaf parameter
# here maps to a single constraint with `kind = :rhs`, mirroring the
# `base_name` patterns in `src/model/balance.jl`.
# -----------------------------------------------------------------------------

const _NODE_PERIOD_RHS_PARAMS = (
    (:emissionTargetAir,    "emTargetAir"),
    (:emissionTargetBunker, "emTargetBunker"),
    (:emissionTargetFS,     "emTargetFS"),
    (:emissionTargetAll,    "emTargetAll"),
)

const _PERIOD_ONLY_RHS_PARAMS = (
    (:emissionTarget_inclScope3andFuelex, "emTargetInclScope3"),
)

const _NODE_ONLY_RHS_PARAMS = (
    (:CO2_cumulative_budget, "emTargetCum"),
    (:cumulative_CO2storage, "co2StorageCum"),
)

function _electricity_trade_limit_mutations(md, idx, _value::Float64)
    isempty(idx) || throw(ArgumentError(
        "Electricity trade controls are scalars and expect empty indices; got $(idx)"))
    p = md.params
    return Mutation[
        Mutation(:rhs, "maxUse[PEU01_03,2050]", "",
                 p.techUse_max[(:PEU01_03, 2050)]),
        Mutation(:rhs, "maxUse[PNL04_01,2050]", "",
                 p.techUse_max[(:PNL04_01, 2050)]),
    ]
end

function _register_default_mutations!()
    empty!(MUTATION_REGISTRY)
    for (field, base) in _NODE_PERIOD_RHS_PARAMS
        local b = String(base)
        register_mutation!(field, function (_md, idx, v::Float64)
            length(idx) == 2 || throw(ArgumentError(
                "Mutation builder for `$(field)` expects (node, period); got $(idx)"))
            n, ps = idx
            return Mutation[Mutation(:rhs, "$(b)[$(n),$(ps)]", "", v)]
        end)
    end
    for (field, base) in _PERIOD_ONLY_RHS_PARAMS
        local b = String(base)
        register_mutation!(field, function (_md, idx, v::Float64)
            length(idx) == 1 || throw(ArgumentError(
                "Mutation builder for `$(field)` expects (period,); got $(idx)"))
            ps, = idx
            return Mutation[Mutation(:rhs, "$(b)[$(ps)]", "", v)]
        end)
    end
    for (field, base) in _NODE_ONLY_RHS_PARAMS
        local b = String(base)
        register_mutation!(field, function (_md, idx, v::Float64)
            length(idx) == 1 || throw(ArgumentError(
                "Mutation builder for `$(field)` expects (node,); got $(idx)"))
            n, = idx
            return Mutation[Mutation(:rhs, "$(b)[$(n)]", "", v)]
        end)
    end
    register_mutation!(:electricity_trade_ratio, _electricity_trade_limit_mutations)
    register_mutation!(:electricity_trade_volume, _electricity_trade_limit_mutations)
    return nothing
end
