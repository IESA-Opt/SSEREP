# =============================================================================
# cyclic_closures.jl — FH-only stateCycleAnnual_* constraints
#
# IESA-Opt 1.0 source: lines 4420-4480 (`stateCycleAnnual_dQ_S`, `stateCycleAnnual_dW_S`,
# `stateCycleAnnual_dQ_backlog_DR`, `stateCycleAnnual_dQ_backlog_BE`,
# `stateCycleAnnual_dB_S`).
#
# These constraints state that the year-start storage/reservoir/backlog level
# equals the year-end level (after appropriate decay/flux).  They are
# **LOGICALLY REDUNDANT** with the existing `_add_storage_state!`,
# `_add_reservoir!`, `_add_backlog!`, `_add_gasbuffer!` helpers in
# `model/hourly.jl`, all of which already implement the cyclic recurrence via
# `h_prev = i == 1 ? hours[end] : hours[i - 1]` (or the daily analog for
# `deltaB_S`).
#
# The IESA-Opt 1.0 source itself explicitly notes this redundancy
# (see `IESA-Opt.ams` line 4415-4419):
#
#   ! These constraints are logically REDUNDANT with the existing variable
#   ! Definitions and stateClosure_* constraints (which already enforce cyclic /
#   ! zero closure), but they are requested for symmetry with the TS calendar-day
#   ! cyclic closure ... Presolve will eliminate redundant rows; no impact on
#   ! LP optimum is expected.
#
# Therefore, these rows should not move the optimum, but AIMMS still sends them
# to Gurobi.  We emit them explicitly so row counts and solver-facing matrices
# match the AIMMS listing before presolve.
# =============================================================================

"""
    add_cyclic_closures!(m, vars, md)

Emit AIMMS-compatible redundant FH annual cyclic closure constraints.

Returns `m` for chaining.
"""
function add_cyclic_closures!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    pss = s.periods_solve
    added = 0

    if vars.deltaQ_S !== nothing && vars.deltaQ_UP !== nothing && vars.deltaQ_DW !== nothing && !isempty(s.hours)
        h_first = first(s.hours)
        h_last = last(s.hours)
        for tfwb in s.tech_fWithBattery, ps in pss
            sw = get(p.slice_width_hours, h_first, 1.0)
            decay = (1.0 - get(p.flex_standing_loss_effective, tfwb, 0.0))^sw
            chg = 1.0 - get(p.flex_loss_charge, tfwb, 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, vars.deltaQ_S[h_first, tfwb, ps])
            add_to_expression!(expr, -decay, vars.deltaQ_S[h_last, tfwb, ps])
            add_to_expression!(expr, -chg, vars.deltaQ_UP[h_first, tfwb, ps])
            add_to_expression!(expr, -1.0, vars.deltaQ_DW[h_first, tfwb, ps])
            @constraint(m, expr == 0.0, base_name = "stateCycleAnnual_dQ_S[$tfwb,$ps]")
            added += 1
        end
    end

    if vars.deltaW_S !== nothing && vars.deltaW_UP !== nothing && vars.tech_useHourly !== nothing && !isempty(s.hours)
        h_first = first(s.hours)
        h_last = last(s.hours)
        for tw in s.tech_reservoir, ps in pss
            prof_t = get(p.profileType_tech, tw, :Flat)
            prof = get(p.hourly_profiles, (h_first, prof_t), 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, vars.deltaW_S[h_first, tw, ps])
            add_to_expression!(expr, -1.0, vars.deltaW_S[h_last, tw, ps])
            add_to_expression!(expr, -(1.0 - get(p.phs_Losses, tw, 0.0)), vars.deltaW_UP[h_first, tw, ps])
            add_to_expression!(expr, -get(p.cap2act, tw, 0.0) * prof, vars.techStock[tw, ps])
            add_to_expression!(expr, 1.0, vars.tech_useHourly[h_first, tw, ps])
            @constraint(m, expr == 0.0, base_name = "stateCycleAnnual_dW_S[$tw,$ps]")
            added += 1
        end
    end

    if vars.deltaQ_backlog_DR !== nothing && vars.deltaQ_UP !== nothing && vars.deltaQ_DW !== nothing && !isempty(s.hours)
        h_first = first(s.hours)
        h_last = last(s.hours)
        for tfs in s.tech_fDRshifting, ps in pss
            chg = 1.0 - get(p.flex_loss_charge, tfs, 0.0)
            disc = 1.0 - get(p.flex_loss_discharge_eff, tfs, 0.0)
            disc <= 0.0 && (disc = 1.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, vars.deltaQ_backlog_DR[h_first, tfs, ps])
            add_to_expression!(expr, -1.0, vars.deltaQ_backlog_DR[h_last, tfs, ps])
            add_to_expression!(expr, -chg, vars.deltaQ_UP[h_first, tfs, ps])
            add_to_expression!(expr, -(1.0 / disc), vars.deltaQ_DW[h_first, tfs, ps])
            @constraint(m, expr == 0.0, base_name = "stateCycleAnnual_dQ_backlog_DR[$tfs,$ps]")
            added += 1
        end
    end

    if vars.deltaQ_backlog_BE !== nothing && vars.deltaQ_UP !== nothing && vars.deltaQ_DW !== nothing && !isempty(s.hours)
        h_first = first(s.hours)
        h_last = last(s.hours)
        for tfe in s.tech_fBEshifting, ps in pss
            chg = 1.0 - get(p.flex_loss_charge, tfe, 0.0)
            disc = 1.0 - get(p.flex_loss_discharge_eff, tfe, 0.0)
            disc <= 0.0 && (disc = 1.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, vars.deltaQ_backlog_BE[h_first, tfe, ps])
            add_to_expression!(expr, -1.0, vars.deltaQ_backlog_BE[h_last, tfe, ps])
            add_to_expression!(expr, -chg, vars.deltaQ_UP[h_first, tfe, ps])
            add_to_expression!(expr, -(1.0 / disc), vars.deltaQ_DW[h_first, tfe, ps])
            @constraint(m, expr == 0.0, base_name = "stateCycleAnnual_dQ_backlog_BE[$tfe,$ps]")
            added += 1
        end
    end

    if vars.deltaB_S !== nothing && vars.deltaB_UP !== nothing && vars.deltaB_DW !== nothing && !isempty(s.days)
        d_first = first(s.days)
        d_last = last(s.days)
        for tg in s.tech_gasBuffer, ps in pss
            get(p.techStock_max, (tg, ps), 0.0) > 0.0 || continue
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, vars.deltaB_S[d_first, tg, ps])
            add_to_expression!(expr, -1.0, vars.deltaB_S[d_last, tg, ps])
            add_to_expression!(expr, -1.0, vars.deltaB_UP[d_first, tg, ps])
            add_to_expression!(expr, -1.0, vars.deltaB_DW[d_first, tg, ps])
            @constraint(m, expr == 0.0, base_name = "stateCycleAnnual_dB_S[$tg,$ps]")
            added += 1
        end
    end

    @info "add_cyclic_closures! - added explicit redundant AIMMS closure rows" rows=added
    flush(stderr)
    return m
end
