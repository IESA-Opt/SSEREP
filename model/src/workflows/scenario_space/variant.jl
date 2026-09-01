# =============================================================================
# workflows/scenario_space/variant.jl — Apply a CampaignSpec row to ModelData + a built model
#
# Phase 2: bridges spec rows (Parameter / Type / value) to leaf-parameter
# mutations + already-built JuMP model. Uses `MUTATION_REGISTRY` from
# `workflows/scenario_space/manifest.jl` to know which constraints to touch.
#
# Lifecycle (per variant):
#   1. Caller deep-copies the base `ModelData` (or clones the leaf params it
#      will touch) so the base remains a clean reference.
#   2. For each `LeafChange`, mutate the leaf parameter in `md.params`.
#   3. Re-run `compute_derived_params!(md)` so any derived params are fresh.
#   4. For each `LeafChange`, ask the registry to build a `Vector{Mutation}`
#      using the *post-derivation* value, then apply to the model.
#   5. `optimize!(model)`.
#
# Notes:
#   - `:set`      → `new_value` overrides the existing leaf entry.
#   - `:multiply` → new_value is a multiplier on the existing entry. If the
#                   entry does not exist, treat the implicit base as 0.0 and
#                   warn (a multiplier on zero has no effect).
#   - The registry-built mutations are kept separate from leaf mutation so that
#     a parameter consumed via a *derived* path can re-derive and read the new
#     effective value (Option C in the master plan).
# =============================================================================

"""
    LeafChange(field, indices, value; type=:set)

A request to change one leaf parameter in `ModelData.params`. `field` is the
ModelParams field name (e.g. `:emissionTargetAir`). `indices` is the lookup
key inside that field's `Dict` — a single value or a `Tuple` (e.g.
`(:NL, 2050)` for `emissionTargetAir`). `value` is the override (when
`type === :set`) or the multiplier (when `type === :multiply`).
"""
struct LeafChange
    field::Symbol
    indices::Tuple
    value::Float64
    type::Symbol
    function LeafChange(field::Symbol, indices, value::Real, type::Symbol = :set)
        type in (:set, :multiply) ||
            throw(ArgumentError("LeafChange.type must be :set or :multiply, got $(repr(type))"))
        idx = indices isa Tuple ? indices : (indices,)
        return new(field, idx, Float64(value), type)
    end
end

LeafChange(; field::Symbol, indices, value::Real, type::Symbol = :set) =
    LeafChange(field, indices, value, type)

"""
    _scalar_key(indices) -> key

If a parameter's Dict is keyed by a bare scalar (e.g. `Dict{Symbol,Float64}`),
unwrap a one-tuple to the scalar so `getindex` works. Otherwise return the
tuple as-is for `Dict{Tuple{...}, Float64}` lookups.
"""
@inline _scalar_key(indices::Tuple{T}) where {T} = indices[1]
@inline _scalar_key(indices::Tuple) = indices

"""
    _get_param_dict(md, field::Symbol) -> AbstractDict

Look up `md.params.<field>` and assert it is a Dict-like container that
supports `get`/`setindex!`.
"""
function _get_param_dict(md, field::Symbol)
    hasproperty(md.params, field) ||
        throw(ArgumentError("ModelParams has no field `$(field)`. " *
                            "Check the spec — typo or unsupported parameter."))
    d = getproperty(md.params, field)
    d isa AbstractDict ||
        throw(ArgumentError("ModelParams.$(field) is a $(typeof(d)); expected a Dict-like parameter."))
    return d
end

"""
    apply_leaf_change!(md, ch::LeafChange) -> Float64

Mutate one entry of one leaf parameter on `md` according to `ch`. Returns the
new effective value (so the caller can feed it to the mutation registry).

For `:set`, the dict entry is overwritten. For `:multiply`, the dict entry is
multiplied by `ch.value`; if the entry is missing, a warning is emitted and
the entry is created at `0 * value = 0`.
"""
function apply_leaf_change!(md, ch::LeafChange)
    isempty(ch.indices) && return _apply_scalar_change!(md, ch)
    d = _get_param_dict(md, ch.field)
    key = _scalar_key(ch.indices)
    if ch.type === :set
        new_val = ch.value
        d[key] = new_val
        return new_val
    else  # :multiply
        old_val = get(d, key, 0.0)
        if old_val == 0.0
            @warn "apply_leaf_change!: multiply on missing/zero entry has no effect" field=ch.field indices=ch.indices multiplier=ch.value
        end
        new_val = old_val * ch.value
        d[key] = new_val
        return new_val
    end
end

function _apply_scalar_change!(md, ch::LeafChange)
    hasproperty(md.params, ch.field) ||
        throw(ArgumentError("ModelParams has no field `$(ch.field)`. " *
                            "Check the spec — typo or unsupported parameter."))
    old_value = getproperty(md.params, ch.field)
    old_value isa Real || throw(ArgumentError(
        "ModelParams.$(ch.field) is a $(typeof(old_value)); expected a numeric scalar."))
    new_value = ch.type === :set ? ch.value : Float64(old_value) * ch.value
    setproperty!(md.params, ch.field, convert(typeof(old_value), new_value))
    return Float64(new_value)
end

"""
    apply_leaf_changes!(md, changes) -> Vector{Float64}

Apply every `LeafChange` in `changes` to `md.params`. Returns the vector of
new effective values in the same order. Does *not* re-run
`compute_derived_params!` — call it explicitly after if downstream derived
params depend on the changed leaves.
"""
function apply_leaf_changes!(md, changes::AbstractVector{LeafChange})
    out = Vector{Float64}(undef, length(changes))
    @inbounds for (i, ch) in pairs(changes)
        out[i] = apply_leaf_change!(md, ch)
    end
    return out
end

"""
    apply_variant!(model, md, changes; rederive=true) -> NamedTuple

Top-level per-variant flow:
  1. Apply every `LeafChange` to `md.params`.
  2. If `rederive`, call `compute_derived_params!(md)`.
  3. For each change, build mutations via the registered builder and apply
     them to the model.

Returns `(values=Vector{Float64}, n_mutations=Int)` for logging.

`model` MUST already be built (`build_ts_lp!` / `build_fh_lp!` already ran)
and `md` MUST be the same `ModelData` used to build it (otherwise the
constraint base_names will not match).
"""
function apply_variant!(model::JuMP.Model, md, changes::AbstractVector{LeafChange};
                        rederive::Bool = true)
    values = apply_leaf_changes!(md, changes)
    rederive && compute_derived_params!(md)
    total = 0
    @inbounds for (i, ch) in pairs(changes)
        muts = build_mutations(md, ch.field, ch.indices, values[i])
        total += apply_mutations!(model, muts)
    end
    return (values = values, n_mutations = total)
end
