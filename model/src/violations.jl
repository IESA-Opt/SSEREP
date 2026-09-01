# =============================================================================
# violations.jl — diagnostics for infeasible / near-infeasible runs
#
# Two complementary tools, both opt-in from the UI:
#
#   1. Elastic re-solve (Show violations checkbox = on).  Wraps the model
#      with `JuMP.relax_with_penalty!` so every (non-bound) constraint
#      becomes elastic with a default penalty per unit of violation.  The
#      problem is always feasible after this transformation; after solve,
#      `report_nonzero_slacks` returns the constraints that needed slack.
#      Works with every solver, including HiGHS.
#
#   2. IIS conflict (always tried when termination = INFEASIBLE and
#      showViolations was off).  Calls `JuMP.compute_conflict!` and lists
#      the constraints in the irreducible infeasible subsystem.  Only
#      commercial solvers (Gurobi, CPLEX, Xpress) implement this — for
#      HiGHS / SCIP / etc. the helper reports that and suggests the user
#      enable Show violations for an elastic re-solve instead.
# =============================================================================

const _ELASTIC_DEFAULT_PENALTY = 1.0e7

"""
    apply_elastic_relaxation!(m::JuMP.Model; default_penalty=1e7) -> Dict

Wrap `m` with JuMP's `relax_with_penalty!` so every non-bound constraint
becomes elastic.  Returns a `Dict{ConstraintRef,AffExpr}` mapping each
constraint to the JuMP expression whose post-solve `value` is the slack
used by that constraint.  Call after the model is built but before
`optimize!`.
"""
function apply_elastic_relaxation!(m::JuMP.Model;
                                   default_penalty::Real = _ELASTIC_DEFAULT_PENALTY)
    return JuMP.relax_with_penalty!(m; default = Float64(default_penalty))
end

"""
    report_nonzero_slacks(penalty_map; threshold=1e-6) -> Vector{NamedTuple}

Inspect the dict returned by [`apply_elastic_relaxation!`](@ref) after
the model has been optimised and return the constraints with slack
greater than `threshold`.  Each entry has `name`, `slack`, and `family`
(the prefix of the JuMP constraint name, useful for grouping).
"""
function report_nonzero_slacks(penalty_map; threshold::Real = 1e-6)
    rows = NamedTuple{(:name, :slack, :family),Tuple{String,Float64,String}}[]
    for (cref, expr) in penalty_map
        v = try
            JuMP.value(expr)
        catch
            0.0
        end
        v isa Number || (v = 0.0)
        v <= Float64(threshold) && continue
        nm = try
            String(JuMP.name(cref))
        catch
            ""
        end
        isempty(nm) && (nm = string(cref))
        family = String(first(split(nm, '['; limit = 2)))
        push!(rows, (name = nm, slack = float(v), family = family))
    end
    sort!(rows; by = r -> -r.slack)
    return rows
end

"""
    compute_iis_report(m::JuMP.Model) -> NamedTuple

Run `JuMP.compute_conflict!` on the (already-optimised, infeasible)
`m` and report the constraints that ended up in the irreducible
infeasible subsystem.  Fields of the returned NamedTuple:

* `supported` — `true` if the solver returned a conflict, `false` if
  the solver does not implement conflict refinement.
* `families`  — unique constraint-name prefixes inside the IIS.
* `names`     — full names of all conflicting constraints.
* `error`     — solver error message (empty when `supported`).

Only commercial solvers (Gurobi, CPLEX, Xpress) implement this method.
"""
function compute_iis_report(m::JuMP.Model)
    families = String[]
    names    = String[]
    errmsg   = ""
    try
        JuMP.compute_conflict!(m)
    catch err
        errmsg = first(split(sprint(showerror, err), '\n'))
        return (supported = false, families = families, names = names, error = errmsg)
    end
    status = try
        MOI.get(m, MOI.ConflictStatus())
    catch
        MOI.NO_CONFLICT_EXISTS
    end
    if status != MOI.CONFLICT_FOUND
        return (supported = true, families = families, names = names,
                error = "Solver did not produce a conflict (status = $(status))")
    end
    for cref in JuMP.all_constraints(m; include_variable_in_set_constraints = false)
        s = try
            MOI.get(m, MOI.ConstraintConflictStatus(), cref)
        catch
            MOI.NOT_IN_CONFLICT
        end
        s == MOI.IN_CONFLICT || continue
        nm = try
            String(JuMP.name(cref))
        catch
            ""
        end
        isempty(nm) && (nm = string(cref))
        push!(names, nm)
        push!(families, String(first(split(nm, '['; limit = 2))))
    end
    return (supported = true, families = unique(families),
            names = names, error = "")
end

"""
    iis_suggestion(families) -> String

Heuristic English suggestion based on the constraint-name prefixes
appearing in the IIS.  Returns an empty string when nothing matches.
"""
function iis_suggestion(families::AbstractVector{<:AbstractString})
    fams = lowercase.(String.(families))
    s = String[]
    any(startswith(f, "balance") for f in fams) && push!(s,
        "Energy balance is infeasible — check supply/demand inputs and shedding penalty for the affected commodity.")
    any(occursin("emission", f) for f in fams) && push!(s,
        "Emission constraint is binding — increase the emission cap or allow more credits.")
    any(occursin("stock", f) || occursin("capacity", f) for f in fams) && push!(s,
        "Stock or capacity ceiling — increase techStock_max or relax forced decommissioning.")
    any(occursin("policy", f) for f in fams) && push!(s,
        "Policy / regulatory constraint — review RES targets, capacity caps, or exogenous build profiles.")
    any(occursin("infrastructure", f) || occursin("transport", f) for f in fams) && push!(s,
        "Infrastructure / transport cap — ensure pipeline / interconnect capacities are large enough.")
    any(occursin("storage", f) || occursin("reservoir", f) for f in fams) && push!(s,
        "Storage / reservoir balance — check seasonal/cyclic-closure and initial-state inputs.")
    isempty(s) && return ""
    return join(s, " ")
end
