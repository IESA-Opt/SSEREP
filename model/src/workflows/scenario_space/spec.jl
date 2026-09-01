# =============================================================================
# workflows/scenario_space/spec.jl -- types and parsing for a scenario-space campaign spec
#
# A campaign is a single scenario-space exploration exercise. It consists of:
#   * meta-settings (sampling method, number of variants, random seed)
#   * a list of parameter rows that describe what to vary, where to find each
#     value in the input workbook, and the parameter's range.
#
# The schema mirrors SSDashboard's `parameter_space.xlsx` (Settings sheet +
# Parameter Space sheet) so that existing spec files are byte-compatible.
# =============================================================================

"""
    SamplingMethod

Enumerates the supported sampling methods for a `CampaignSpec`. Stored on the
spec as a `Symbol` so JSON serialization stays straightforward.

* `:lhs`        — Latin hypercube sampling (continuous parameter space)
* `:morris`     — Morris elementary-effects trajectories
* `:sobol`      — Sobol low-discrepancy sequence
* `:factorial`  — full factorial over discretised ranges (requires `step`)
"""
const SAMPLING_METHODS = (:lhs, :morris, :sobol, :factorial)

"""
    parse_sampling_method(s) -> Symbol

Accept any of the labels used in the SSDashboard xlsx ("Latin hypercube",
"Morris", "Sobol", "Factorial") or the canonical IESA-Opt symbols. Returns one
of [`SAMPLING_METHODS`](@ref) or throws `ArgumentError`.
"""
function parse_sampling_method(s)
    s isa Symbol && return parse_sampling_method(string(s))
    raw = strip(lowercase(String(s)))
    raw in ("lhs", "latin hypercube", "latin-hypercube", "latin_hypercube") && return :lhs
    raw in ("morris",) && return :morris
    raw in ("sobol",) && return :sobol
    raw in ("factorial", "full factorial", "full-factorial") && return :factorial
    throw(ArgumentError("Unknown sampling method '$s'. Expected one of: lhs, morris, sobol, factorial."))
end

"""
    parse_param_type(s) -> Symbol

Parse the SSDashboard `Type` column. Returns `:set` (set the cell to the
sampled value) or `:multiply` (multiply the cell's existing value by the
sampled multiplier).
"""
function parse_param_type(s)
    s isa Symbol && return parse_param_type(string(s))
    raw = strip(lowercase(String(s)))
    raw in ("set",) && return :set
    raw in ("multiply", "mult", "*") && return :multiply
    throw(ArgumentError("Unknown parameter type '$s'. Expected 'set' or 'multiply'."))
end

"""
    ParameterRow

A single row of the parameter-space table. Mirrors the SSDashboard
`Parameter Space` sheet header:

| Parameter | Sub-parameter | Sheet | Cell | Type | Min | Max | Step | Notes |

Multiple `ParameterRow`s may share the same `parameter` name; in that case the
sampler treats them as one parameter with multiple sub-parameters that all
receive the same sampled value (e.g. one `RES capex multiplier` parameter
applied to Wind/Solar/Biomass cells). Min/Max/Step then only need to be
populated on the first row of each parameter group.
"""
struct ParameterRow
    parameter::String
    subparameter::String
    sheet::String
    cell::String
    type::Symbol            # :set | :multiply
    min::Union{Float64,Nothing}
    max::Union{Float64,Nothing}
    step::Union{Float64,Nothing}
    notes::String
end

function ParameterRow(; parameter::AbstractString,
                       subparameter::AbstractString = "",
                       sheet::AbstractString,
                       cell::AbstractString,
                       type,
                       min = nothing,
                       max = nothing,
                       step = nothing,
                       notes::AbstractString = "")
    return ParameterRow(String(parameter),
                        isempty(subparameter) ? String(parameter) : String(subparameter),
                        String(sheet),
                        String(cell),
                        parse_param_type(type),
                        _maybe_float(min),
                        _maybe_float(max),
                        _maybe_float(step),
                        String(notes))
end

_maybe_float(::Nothing) = nothing
_maybe_float(x::AbstractFloat) = isnan(x) ? nothing : Float64(x)
_maybe_float(x::Integer) = Float64(x)
function _maybe_float(x::AbstractString)
    s = strip(x)
    isempty(s) && return nothing
    return parse(Float64, s)
end
_maybe_float(x) = throw(ArgumentError("Cannot convert $(typeof(x)) value '$x' to Float64."))

"""
    CampaignSpec

A single scenario-space campaign: meta-settings + a list of
[`ParameterRow`](@ref)s. Name is human-readable and used as the campaign
folder name; sampling settings are stored as a `Symbol`/`Int` triple.
"""
struct CampaignSpec
    name::String
    method::Symbol
    n_variants::Int
    seed::Int
    rows::Vector{ParameterRow}
end

function CampaignSpec(; name::AbstractString,
                       method,
                       n_variants::Integer,
                       seed::Integer,
                       rows::AbstractVector{ParameterRow})
    n_variants > 0 || throw(ArgumentError("n_variants must be positive, got $n_variants"))
    return CampaignSpec(String(name),
                        parse_sampling_method(method),
                        Int(n_variants),
                        Int(seed),
                        collect(rows))
end

"""
    unique_parameters(spec) -> Vector{String}

Return the deduplicated list of parameter names in the order they first appear
in the spec. This is the column order used by the sampling matrix.
"""
function unique_parameters(spec::CampaignSpec)
    seen = Set{String}()
    out = String[]
    for r in spec.rows
        if !(r.parameter in seen)
            push!(seen, r.parameter)
            push!(out, r.parameter)
        end
    end
    return out
end

"""
    parameter_bounds(spec) -> Vector{Tuple{Float64,Float64}}

Return `(min, max)` for each unique parameter, in the same order as
[`unique_parameters`](@ref). The bounds are taken from the *first* row of
each parameter group (matching SSDashboard's convention that only the first
row needs to populate Min/Max for shared-range parameters). Throws
`ArgumentError` if a parameter has no row with both Min and Max populated.
"""
function parameter_bounds(spec::CampaignSpec)
    bounds = Tuple{Float64,Float64}[]
    seen = Dict{String,Tuple{Float64,Float64}}()
    for r in spec.rows
        haskey(seen, r.parameter) && continue
        if r.min === nothing || r.max === nothing
            # Try later rows for the same parameter
            found = false
            for r2 in spec.rows
                if r2.parameter == r.parameter && r2.min !== nothing && r2.max !== nothing
                    seen[r.parameter] = (r2.min::Float64, r2.max::Float64)
                    push!(bounds, seen[r.parameter])
                    found = true
                    break
                end
            end
            found || throw(ArgumentError("Parameter '$(r.parameter)' has no row with both Min and Max set."))
        else
            seen[r.parameter] = (r.min::Float64, r.max::Float64)
            push!(bounds, seen[r.parameter])
        end
    end
    return bounds
end

"""
    parameter_steps(spec) -> Vector{Union{Float64,Nothing}}

Return the `step` value of each unique parameter (taken from the first row
of each parameter group). Used by the factorial sampler; LHS/Sobol/Morris
ignore steps.
"""
function parameter_steps(spec::CampaignSpec)
    steps = Union{Float64,Nothing}[]
    seen = Dict{String,Union{Float64,Nothing}}()
    for r in spec.rows
        haskey(seen, r.parameter) && continue
        # Use first row's step (consistent with SSDashboard sampling.discretize_parameter_space)
        seen[r.parameter] = r.step
        push!(steps, r.step)
    end
    return steps
end

"""
    validate_spec(spec; require_n_variants = true) -> NamedTuple

Cheap server-side validation that flags row-level issues before sampling.
Returns `(valid::Bool, errors::Vector{String}, warnings::Vector{String})`.

Checks:
  * spec name is non-empty
  * sampling method is one of the supported symbols
  * n_variants > 0 (when `require_n_variants` is true; for Morris/Factorial
    the implied count overrides the user input later)
  * at least one row
  * each row has `parameter`, `sheet`, `cell` non-empty
  * each row's `type` is `:set` or `:multiply`
  * each parameter has at least one row with both Min and Max populated
  * if any row has Step, Min, and Max all populated, the discretised range
    yields at least 1 value
"""
function validate_spec(spec::CampaignSpec; require_n_variants::Bool = true)
    errors = String[]
    warnings = String[]

    isempty(strip(spec.name)) && push!(errors, "Campaign name is empty.")
    spec.method in SAMPLING_METHODS || push!(errors, "Unknown sampling method '$(spec.method)'.")
    if require_n_variants && spec.method in (:lhs, :sobol)
        spec.n_variants > 0 || push!(errors, "Number of variants must be positive (got $(spec.n_variants)).")
    end
    isempty(spec.rows) && push!(errors, "Parameter space is empty (add at least one parameter row).")

    seen_param_with_bounds = Set{String}()
    for (i, r) in pairs(spec.rows)
        prefix = "Row $i ('$(r.parameter)' / '$(r.subparameter)'):"
        isempty(strip(r.parameter)) && push!(errors, "$prefix Parameter is empty.")
        isempty(strip(r.sheet))     && push!(errors, "$prefix Sheet is empty.")
        isempty(strip(r.cell))      && push!(errors, "$prefix Cell is empty.")
        r.type in (:set, :multiply) || push!(errors, "$prefix Type must be 'set' or 'multiply'.")
        if r.min !== nothing && r.max !== nothing
            r.min > r.max && push!(errors, "$prefix Min ($(r.min)) exceeds Max ($(r.max)).")
            push!(seen_param_with_bounds, r.parameter)
            if r.step !== nothing
                r.step <= 0 && push!(errors, "$prefix Step must be positive (got $(r.step)).")
                if r.step > 0 && r.max - r.min < r.step
                    push!(warnings, "$prefix Step ($(r.step)) is larger than the Min..Max span; only one value will be sampled.")
                end
            end
        end
    end

    # Every parameter must have at least one row with both bounds populated
    for p in unique_parameters(spec)
        p in seen_param_with_bounds || push!(errors, "Parameter '$p' has no row with both Min and Max populated.")
    end

    if spec.method == :factorial
        # Factorial requires Step on every parameter
        for r in spec.rows
            if r.parameter ∉ Set([rr.parameter for rr in spec.rows if rr.step !== nothing])
                push!(errors, "Factorial sampling requires a Step on every parameter; '$(r.parameter)' has none.")
                break
            end
        end
    end

    return (; valid = isempty(errors), errors, warnings)
end

# -----------------------------------------------------------------------------
# JSON-friendly conversion (Dict <-> CampaignSpec)
# -----------------------------------------------------------------------------

"""
    spec_from_dict(d) -> CampaignSpec

Construct a `CampaignSpec` from a JSON-like Dict (as produced by JSON3 when
parsing the request body of `POST /api/scenario/...`).
"""
function spec_from_dict(d)
    rows = ParameterRow[]
    rows_in = get(d, "rows", get(d, :rows, ()))
    for r in rows_in
        get_field = (k1, k2) -> begin
            haskey(r, k1) && return r[k1]
            haskey(r, k2) && return r[k2]
            return nothing
        end
        push!(rows, ParameterRow(
            parameter    = String(get_field("parameter", :parameter)),
            subparameter = _opt_string(get_field("subparameter", :subparameter)),
            sheet        = String(get_field("sheet", :sheet)),
            cell         = String(get_field("cell", :cell)),
            type         = String(get_field("type", :type)),
            min          = _opt_float(get_field("min", :min)),
            max          = _opt_float(get_field("max", :max)),
            step         = _opt_float(get_field("step", :step)),
            notes        = _opt_string(get_field("notes", :notes)),
        ))
    end
    return CampaignSpec(
        name       = String(get(d, "name", get(d, :name, ""))),
        method     = String(get(d, "method", get(d, :method, "lhs"))),
        n_variants = Int(get(d, "n_variants", get(d, :n_variants, get(d, "nVariants", get(d, :nVariants, 1))))),
        seed       = Int(get(d, "seed", get(d, :seed, 12345))),
        rows       = rows,
    )
end

_opt_string(::Nothing) = ""
_opt_string(x) = String(x)

_opt_float(::Nothing) = nothing
_opt_float(x::Number) = Float64(x)
_opt_float(x::AbstractString) = isempty(strip(x)) ? nothing : parse(Float64, x)

"""
    spec_to_dict(spec) -> Dict{String,Any}

JSON-friendly serialization for the UI's import/export round-trip.
"""
function spec_to_dict(spec::CampaignSpec)
    return Dict{String,Any}(
        "name" => spec.name,
        "method" => String(spec.method),
        "n_variants" => spec.n_variants,
        "seed" => spec.seed,
        "rows" => [Dict{String,Any}(
            "parameter" => r.parameter,
            "subparameter" => r.subparameter,
            "sheet" => r.sheet,
            "cell" => r.cell,
            "type" => String(r.type),
            "min" => r.min === nothing ? "" : r.min,
            "max" => r.max === nothing ? "" : r.max,
            "step" => r.step === nothing ? "" : r.step,
            "notes" => r.notes,
        ) for r in spec.rows],
    )
end
