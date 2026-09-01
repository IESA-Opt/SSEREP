# =============================================================================
# variables.jl — JuMP variable declarations for the IESA-Opt port
#
# Phase 2: annual LP variables only.
#   - tech_use(tb, ps)           ≥ 0
#   - techStock(t, ps)           ≥ 0   (recursive — defined via constraint)
#   - cap_investments(t, ps)     ≥ 0
#   - eco_decommisioning(t, ps)  ≥ 0
#   - decomStock(t, ps)          ≥ 0   (recursive — defined via constraint)
#   - retrofitting(it, jt, ps)   ≥ 0
#
# Phase 3+ (hourly dispatch, flex, CHP, storage, shedding, reservoir, gasbuffer)
# are introduced in their own files (`balance.jl`, `storage.jl`, …).
#
# IESA-Opt 1.0 source: lines ~2422-2829 of `MainProject/IESA-Opt.ams`. Ranges + index
# domains match exactly:
#   Variable tech_use            { IndexDomain: (tb,ps);    Range: nonnegative; }
#   Variable techStock           { IndexDomain: (t,ps);     Range: nonnegative; }
#   Variable cap_investments     { IndexDomain: (t,ps);     Range: nonnegative; }
#   Variable retrofitting        { IndexDomain: (it,jt,ps); Range: nonnegative; }
#   Variable eco_decommisioning  { IndexDomain: (t,ps);     Range: nonnegative; }
#   Variable decomStock          { IndexDomain: (t,ps);     Range: nonnegative; }
# =============================================================================

"""
    AnnualVars

Holds JuMP variable references produced by `add_annual_variables!` and
`add_hourly_variables!` (the latter only when building the FH/TS LP).

The hourly fields default to `nothing` so Phase-2 (annual-only) callers
continue to work without modification.

Per IESA-Opt 1.0 Variable declarations (Sections HourlyDispatch, Daily, GasBuffer,
Reservoir, Flex, CHP, Shedding; ~lines 3055-4180):

| Field             | Domain            | Range      | IESA-Opt 1.0 line |
|-------------------|-------------------|------------|------------|
| tech_useHourly    | (h, thh, ps)      | ≥ 0        | 3055       |
| tech_useDaily     | (d, td, ps)       | ≥ 0        | 3153       |
| deltaB_UP         | (d, tg, ps)       | ≤ 0        | 3157       |
| deltaB_DW         | (d, tg, ps)       | ≥ 0        | 3161       |
| deltaB_S          | (d, tg, ps)       | ≤ 0 (def)  | 3165       |
| deltaU_CHP        | (h, tk, ps)       | free       | 3469       |
| deltaP_CHP        | (h, tk, ps)       | free       | 3473       |
| deltaS_shed       | (h, ts, ps)       | ≤ 0        | 4080       |
| deltaQ_UP         | (h, tf, ps)       | ≤ 0        | 4084       |
| deltaQ_DW         | (h, tf, ps)       | ≥ 0        | 4091       |
| deltaQ_S          | (h, tfwb, ps)     | ≤ 0 (def)  | 4095       |
| deltaQ_backlog_DR | (h, tfs, ps)      | ≥ 0        | 4110       |
| deltaQ_backlog_BE | (h, tfe, ps)      | ≥ 0        | 4114       |
| deltaW_UP         | (h, tw, ps)       | ≥ 0        | 4128       |
| deltaW_S          | (h, tw, ps)       | free (def) | 4132       |
"""
mutable struct AnnualVars
    tech_use::JuMP.Containers.DenseAxisArray
    techStock::JuMP.Containers.DenseAxisArray
    cap_investments::JuMP.Containers.DenseAxisArray
    eco_decommisioning::JuMP.Containers.DenseAxisArray
    decomStock::JuMP.Containers.DenseAxisArray
    retrofitting::Any

    # Phase 3 hourly + daily variables — populated by add_hourly_variables!
    # All are Union{Nothing, ...} so AnnualVars stays back-compat with Phase 2
    tech_useHourly::Any           # DenseAxisArray (h, thh, ps) or nothing
    tech_useDaily::Any            # DenseAxisArray (d, td, ps)
    deltaB_UP::Any                # DenseAxisArray (d, tg, ps)
    deltaB_DW::Any
    deltaB_S::Any
    deltaU_CHP::Any               # DenseAxisArray (h, tk, ps)
    deltaP_CHP::Any
    deltaS_shed::Any              # DenseAxisArray (h, ts, ps)
    deltaQ_UP::Any                # DenseAxisArray (h, tf, ps)
    deltaQ_DW::Any
    deltaQ_S::Any                 # DenseAxisArray (h, tfwb, ps)
    deltaQ_backlog_DR::Any
    deltaQ_backlog_BE::Any
    deltaW_UP::Any                # DenseAxisArray (h, tw, ps)
    deltaW_S::Any
    # IESA-Opt 1.0 day-aggregated long-term flex deltas (FH; Variable-with-Definition,
    # IESA-Opt.ams lines 4129/4134). Domain (d, tfl, ps) where tfl = tech_flexLT.
    deltaQd_UP::Any               # ≤ 0  defining
    deltaQd_DW::Any               # ≥ 0  defining

    # Phase 5 TS (rep-day clustered) variables — populated by add_ts_variables!
    # All indexed by `hc` (cluster hour) instead of `h`.
    tech_useHourly_TS::Any
    tech_useDaily_TS::Any
    deltaB_UP_TS::Any
    deltaB_DW_TS::Any
    deltaB_S_TS::Any
    deltaU_CHP_TS::Any
    deltaP_CHP_TS::Any
    deltaS_shed_TS::Any
    deltaQ_UP_TS::Any
    deltaQ_DW_TS::Any
    deltaQ_S_TS::Any
    deltaQ_backlog_DR_TS::Any
    deltaQ_backlog_BE_TS::Any
    deltaW_UP_TS::Any
    deltaW_S_TS::Any
    # Calendar-day anchor variables (Phase 4 + Phase 5: cross-period linking)
    deltaQ_dayStart_TS::Any       # (rd, tfwb, ps)
    deltaQ_dayEnd_TS::Any         # (rd, tfwb, ps)
    deltaQ_calDayLevel_TS::Any    # (d, tfwb, ps)  cal-day cumulative state
    deltaW_dayStart_TS::Any       # (rd, tw, ps)
    deltaW_dayEnd_TS::Any
    deltaW_calDayLevel_TS::Any
    deltaQ_backlog_BE_dayStart_TS::Any
    deltaQ_backlog_BE_dayEnd_TS::Any
    deltaQ_backlog_BE_calDayLevel_TS::Any
    # IESA-Opt 1.0-canonical TS rep-day-aggregated flex deltas (Variable-with-Definition)
    deltaB_dayStart_TS::Any        # (rd, tg, ps)  IESA-Opt 1.0 gas-buffer rep-day start — nonpositive
    deltaQd_UP_TS::Any             # (rd, tfl, ps) IESA-Opt 1.0 rep-day aggregated flex UP   — nonpositive (defining)
    deltaQd_DW_TS::Any             # (rd, tfl, ps) IESA-Opt 1.0 rep-day aggregated flex DW   — nonnegative (defining)
end

# Convenience constructor: annual-only vars (hourly fields = nothing)
function AnnualVars(tu, ts, ci, ed, ds, rt)
    AnnualVars(tu, ts, ci, ed, ds, rt,
        nothing, nothing,                       # tech_useHourly, tech_useDaily
        nothing, nothing, nothing,              # deltaB_UP/DW/S
        nothing, nothing,                       # deltaU_CHP, deltaP_CHP
        nothing,                                # deltaS_shed
        nothing, nothing, nothing,              # deltaQ_UP/DW/S
        nothing, nothing,                       # deltaQ_backlog_DR/BE
        nothing, nothing,                       # deltaW_UP/S
        nothing, nothing,                       # deltaQd_UP, deltaQd_DW (FH)
        # Phase 5 TS (24 fields)
        nothing, nothing,                       # tech_useHourly_TS, tech_useDaily_TS
        nothing, nothing, nothing,              # deltaB_UP/DW/S_TS
        nothing, nothing,                       # deltaU_CHP/P_CHP_TS
        nothing,                                # deltaS_shed_TS
        nothing, nothing, nothing,              # deltaQ_UP/DW/S_TS
        nothing, nothing,                       # deltaQ_backlog_DR/BE_TS
        nothing, nothing,                       # deltaW_UP/S_TS
        nothing, nothing, nothing,              # deltaQ_dayStart/End/calDay_TS
        nothing, nothing, nothing,              # deltaW_dayStart/End/calDay_TS
        nothing, nothing, nothing,              # deltaQ_backlog_BE_*_TS
        nothing,                                # deltaB_dayStart_TS
        nothing, nothing)                       # deltaQd_UP_TS, deltaQd_DW_TS
end

"""
    add_annual_variables!(model::JuMP.Model, md::ModelData) -> AnnualVars

Declare all annual LP variables on `model`.

Index sets:
- `tech_use`           over `tb ∈ tech_balancers, ps ∈ periods_solve`
- `techStock`,
  `cap_investments`,
  `eco_decommisioning`,
  `decomStock`         over `t ∈ technologies,    ps ∈ periods_solve`
- `retrofitting`       over `(it, jt) ∈ technologies x technologies, ps ∈ periods_solve`
"""
function add_annual_variables!(model::JuMP.Model, md::ModelData)
    s   = md.sets
    p   = md.params
    pss = s.periods_solve
    isempty(pss) && error("md.sets.periods_solve is empty — cannot declare variables")
    isempty(s.tech_balancers) && error("md.sets.tech_balancers is empty — call derive_sets! first")
    isempty(s.technologies)   && error("md.sets.technologies is empty — call derive_sets! first")

    # tech_use(tb, ps) ≥ 0
    @variable(model, tech_use[tb = s.tech_balancers, ps = pss] >= 0)

    # techStock(t, ps) ≥ 0  — recursion supplied by `techStock_def_constraint`
    @variable(model, techStock[t = s.technologies, ps = pss] >= 0)

    # cap_investments(t, ps) ≥ 0
    @variable(model, cap_investments[t = s.technologies, ps = pss] >= 0)

    # eco_decommisioning(t, ps) ≥ 0
    @variable(model, eco_decommisioning[t = s.technologies, ps = pss] >= 0)

    # decomStock(t, ps) ≥ 0  — recursion supplied by `decomStock_def_constraint`
    @variable(model, decomStock[t = s.technologies, ps = pss] >= 0)

    # IESA-Opt 1.0 declares retrofitting(it,jt,ps) over the full (technologies
    # × technologies × periods) cross-product, but Gurobi presolve discards
    # ~99.99 % of those variables because the constraint
    #   retrofit_relations(it,jt) * (...)  -  retrofitting(it,jt,ps)  >= 0
    # forces retrofitting=0 whenever `retrofit_relations(it,jt) == false`.
    # AIMMS strips them via NetVarMatrix before sending the LP to Gurobi;
    # we build the variable sparse so the matrix matrix Gurobi receives is
    # comparable to AIMMS in dimension. `p.retrofit_pairs` is populated by
    # `derive_sets!::_derive_retrofit_pairs!`.
    retrofitting = Dict{Tuple{Symbol,Symbol,Int}, JuMP.VariableRef}()
    sizehint!(retrofitting, length(p.retrofit_pairs) * length(pss))
    keep_names = get(ENV, "IESA_OPT_KEEP_NAMES", "0") == "1"
    for (it_, jt_) in p.retrofit_pairs, ps in pss
        v = @variable(model, lower_bound = 0.0)
        if keep_names
            JuMP.set_name(v, "retrofitting[$(it_),$(jt_),$(ps)]")
        end
        retrofitting[(it_, jt_, ps)] = v
    end

    return AnnualVars(
        tech_use,
        techStock,
        cap_investments,
        eco_decommisioning,
        decomStock,
        retrofitting,
    )
end

"""
    _prev_period(pss::AbstractVector{Int}, ps::Int) -> Union{Int,Nothing}

Return the previous period in `pss`, or `nothing` if `ps` is the first.
Mirrors IESA-Opt 1.0 `ps-1` / `element(periods_selection, 1)` logic.
"""
_prev_period(pss::AbstractVector{Int}, ps::Int) = begin
    i = findfirst(==(ps), pss)
    (i === nothing || i == 1) ? nothing : pss[i - 1]
end

"""
    _is_first_period(pss::AbstractVector{Int}, ps::Int) -> Bool

True iff `ps` is the first element of `periods_solve`.
"""
_is_first_period(pss::AbstractVector{Int}, ps::Int) = !isempty(pss) && ps == first(pss)


# =============================================================================
# Phase 3 — Hourly + Daily variable declarations (FH mode only; TS = Phase 5)
# =============================================================================

"""
    add_hourly_variables!(model::JuMP.Model, vars::AnnualVars, md::ModelData)
        -> AnnualVars

Mutates `vars` in place, populating all hourly/daily fields. Returns `vars`.

Set memberships used (see `derive_sets!`):
- `tech_hourlyDispatch` → `tech_useHourly`, `capacity_techHourly`, `rampingUP/DW`
- `tech_dailyDispatch`  → `tech_useDaily`, `capacity_techDaily`
- `tech_gasBuffer`      → `deltaB_UP/DW/S`
- `tech_fEV ∪ tech_fStorage ∪ tech_fOther` (= `tech_flexible`) → `deltaQ_UP/DW`
- `tech_fWithBattery` (storage+EV+battery) → `deltaQ_S` (state)
- `tech_flexH` (short-term DR) → `deltaQ_backlog_DR`
- `tech_flexE` (energy-backed, e.g. battery) → `deltaQ_backlog_BE`
- `tech_reservoir` → `deltaW_UP/S`
- `tech_hourlyCHPflex` → `deltaU_CHP`, `deltaP_CHP`
- `tech_shedding` → `deltaS_shed`
"""
function add_hourly_variables!(model::JuMP.Model, vars::AnnualVars, md::ModelData)
    s   = md.sets
    pss = s.periods_solve
    isempty(pss) && error("md.sets.periods_solve is empty")
    isempty(s.hours) && error("md.sets.hours is empty — call derive_sets! after setting hoursPer_day")
    isempty(s.days)  && error("md.sets.days is empty")

    # ------ tech_useHourly(h, thh, ps) >= 0 — only if there are hourly-dispatch techs
    if !isempty(s.tech_hourlyDispatch)
        @variable(model, tech_useHourly[h = s.hours, thh = s.tech_hourlyDispatch, ps = pss] >= 0)
        vars.tech_useHourly = tech_useHourly
    end

    # ------ tech_useDaily(d, td, ps) >= 0
    if !isempty(s.tech_dailyDispatch)
        @variable(model, tech_useDaily[d = s.days, td = s.tech_dailyDispatch, ps = pss] >= 0)
        vars.tech_useDaily = tech_useDaily
    end

    # ------ Gas buffer (d, tg, ps): deltaB_UP <= 0, deltaB_DW >= 0, deltaB_S <= 0
    if !isempty(s.tech_gasBuffer)
        @variable(model, deltaB_UP[d = s.days, tg = s.tech_gasBuffer, ps = pss] <= 0)
        @variable(model, deltaB_DW[d = s.days, tg = s.tech_gasBuffer, ps = pss] >= 0)
        @variable(model, deltaB_S[d = s.days, tg = s.tech_gasBuffer, ps = pss] <= 0)
        vars.deltaB_UP = deltaB_UP
        vars.deltaB_DW = deltaB_DW
        vars.deltaB_S  = deltaB_S
    end

    # ------ CHP (h, tk, ps): deltaU_CHP free, deltaP_CHP free
    if !isempty(s.tech_hourlyCHPflex)
        @variable(model, deltaU_CHP[h = s.hours, tk = s.tech_hourlyCHPflex, ps = pss])
        @variable(model, deltaP_CHP[h = s.hours, tk = s.tech_hourlyCHPflex, ps = pss])
        vars.deltaU_CHP = deltaU_CHP
        vars.deltaP_CHP = deltaP_CHP
    end

    # ------ Shedding (h, ts, ps): deltaS_shed <= 0
    if !isempty(s.tech_shedding)
        @variable(model, deltaS_shed[h = s.hours, ts = s.tech_shedding, ps = pss] <= 0)
        vars.deltaS_shed = deltaS_shed
    end

    # ------ Flex (h, tf, ps): deltaQ_UP <= 0, deltaQ_DW >= 0
    if !isempty(s.tech_flexible)
        @variable(model, deltaQ_UP[h = s.hours, tf = s.tech_flexible, ps = pss] <= 0)
        @variable(model, deltaQ_DW[h = s.hours, tf = s.tech_flexible, ps = pss] >= 0)
        vars.deltaQ_UP = deltaQ_UP
        vars.deltaQ_DW = deltaQ_DW
    end

    # ------ Day-aggregated long-term flex deltas (d, tfl, ps).
    # IESA-Opt 1.0 lines 4129/4134:
    #   Variable deltaQd_UP { IndexDomain: (d,tfl,ps); Range: nonpositive;
    #     Definition: sum[ih | dayPer_hour(ih)=d, deltaQ_UP(ih,tfl,ps)] }
    #   Variable deltaQd_DW { IndexDomain: (d,tfl,ps); Range: nonnegative;
    #     Definition: sum[ih | dayPer_hour(ih)=d, deltaQ_DW(ih,tfl,ps)] }
    # Used by balanceD/R/W/M/S/B/Y_deltaQd (IESA-Opt.ams lines 4203-4232).
    if !isempty(s.tech_flexLT)
        @variable(model, deltaQd_UP[d = s.days, tfl = s.tech_flexLT, ps = pss] <= 0)
        @variable(model, deltaQd_DW[d = s.days, tfl = s.tech_flexLT, ps = pss] >= 0)
        vars.deltaQd_UP = deltaQd_UP
        vars.deltaQd_DW = deltaQd_DW
    end

    # ------ Storage state (h, tfwb, ps): deltaQ_S <= 0
    if !isempty(s.tech_fWithBattery)
        @variable(model, deltaQ_S[h = s.hours, tfwb = s.tech_fWithBattery, ps = pss] <= 0)
        vars.deltaQ_S = deltaQ_S
    end

    # IESA-Opt 1.0 deltaQ_backlog_DR(h, tfs, ps) — index tfs = tech_fDRshifting
    if !isempty(s.tech_fDRshifting)
        @variable(model, deltaQ_backlog_DR[h = s.hours, tfs = s.tech_fDRshifting, ps = pss] >= 0)
        vars.deltaQ_backlog_DR = deltaQ_backlog_DR
    end
    # IESA-Opt 1.0 deltaQ_backlog_BE(h, tfe, ps) — index tfe = tech_fBEshifting
    if !isempty(s.tech_fBEshifting)
        @variable(model, deltaQ_backlog_BE[h = s.hours, tfe = s.tech_fBEshifting, ps = pss] >= 0)
        vars.deltaQ_backlog_BE = deltaQ_backlog_BE
    end

    # ------ Reservoir (h, tw, ps): deltaW_UP >= 0, deltaW_S free
    if !isempty(s.tech_reservoir)
        @variable(model, deltaW_UP[h = s.hours, tw = s.tech_reservoir, ps = pss] >= 0)
        @variable(model, deltaW_S[h = s.hours, tw = s.tech_reservoir, ps = pss])
        vars.deltaW_UP = deltaW_UP
        vars.deltaW_S  = deltaW_S
    end

    return vars
end


# =============================================================================
# Hour-index helpers (cyclic year-end wrap)
# =============================================================================

"""
    _prev_hour(hours::AbstractVector{Int}, h::Int) -> Int

Return the cyclic predecessor in `hours`: previous hour, wrapping `first → last`.
"""
function _prev_hour(hours::AbstractVector{Int}, h::Int)
    h == first(hours) ? last(hours) : hours[findfirst(==(h), hours) - 1]
end

"""
    _prev_day(days::AbstractVector{Int}, d::Int) -> Int

Cyclic previous day; wraps `first → last`.
"""
function _prev_day(days::AbstractVector{Int}, d::Int)
    d == first(days) ? last(days) : days[findfirst(==(d), days) - 1]
end


# =============================================================================
# Phase 5 — TS (clustered rep-day) variables
# =============================================================================

"""
    add_ts_variables!(model::JuMP.Model, vars::AnnualVars, md::ModelData) -> AnnualVars

Declare all `_TS` variables indexed by `hc ∈ s.hours_cluster` for the
time-slice / representative-day LP. Mirrors `add_hourly_variables!` but with
`hc` substituted for `h`. Also declares calendar-day anchor variables for
cross-period storage state propagation.

Must be called AFTER `build_temporal_clusters!(md)` has populated
`md.sets.hours_cluster`, `md.sets.repDays`, and the cluster parameter Dicts.
"""
function add_ts_variables!(model::JuMP.Model, vars::AnnualVars, md::ModelData)
    s = md.sets
    pss = s.periods_solve
    isempty(pss) && error("md.sets.periods_solve is empty")
    isempty(s.hours_cluster) && error("md.sets.hours_cluster is empty — call build_temporal_clusters! first")
    isempty(s.repDays) && error("md.sets.repDays is empty — call build_temporal_clusters! first")

    if !isempty(s.tech_hourlyDispatch)
        @variable(model, tech_useHourly_TS[hc = s.hours_cluster, thh = s.tech_hourlyDispatch, ps = pss] >= 0)
        vars.tech_useHourly_TS = tech_useHourly_TS
    end
    if !isempty(s.tech_dailyDispatch)
        @variable(model, tech_useDaily_TS[rd = s.repDays, td = s.tech_dailyDispatch, ps = pss] >= 0)
        vars.tech_useDaily_TS = tech_useDaily_TS
    end
    if !isempty(s.tech_gasBuffer)
        @variable(model, deltaB_UP_TS[rd = s.repDays, tg = s.tech_gasBuffer, ps = pss] <= 0)
        @variable(model, deltaB_DW_TS[rd = s.repDays, tg = s.tech_gasBuffer, ps = pss] >= 0)
        # IESA-Opt 1.0 deltaB_S_TS — Range: nonpositive (line 3296)
        @variable(model, deltaB_S_TS[rd = s.repDays, tg = s.tech_gasBuffer, ps = pss] <= 0)
        # IESA-Opt 1.0 deltaB_dayStart_TS — Range: nonpositive (line 3293)
        @variable(model, deltaB_dayStart_TS[rd = s.repDays, tg = s.tech_gasBuffer, ps = pss] <= 0)
        vars.deltaB_UP_TS       = deltaB_UP_TS
        vars.deltaB_DW_TS       = deltaB_DW_TS
        vars.deltaB_S_TS        = deltaB_S_TS
        vars.deltaB_dayStart_TS = deltaB_dayStart_TS
    end
    if !isempty(s.tech_hourlyCHPflex)
        @variable(model, deltaU_CHP_TS[hc = s.hours_cluster, tk = s.tech_hourlyCHPflex, ps = pss])
        @variable(model, deltaP_CHP_TS[hc = s.hours_cluster, tk = s.tech_hourlyCHPflex, ps = pss])
        vars.deltaU_CHP_TS = deltaU_CHP_TS
        vars.deltaP_CHP_TS = deltaP_CHP_TS
    end
    if !isempty(s.tech_shedding)
        @variable(model, deltaS_shed_TS[hc = s.hours_cluster, ts = s.tech_shedding, ps = pss] <= 0)
        vars.deltaS_shed_TS = deltaS_shed_TS
    end
    if !isempty(s.tech_flexible)
        @variable(model, deltaQ_UP_TS[hc = s.hours_cluster, tf = s.tech_flexible, ps = pss] <= 0)
        @variable(model, deltaQ_DW_TS[hc = s.hours_cluster, tf = s.tech_flexible, ps = pss] >= 0)
        vars.deltaQ_UP_TS = deltaQ_UP_TS
        vars.deltaQ_DW_TS = deltaQ_DW_TS
    end
    # IESA-Opt 1.0 deltaQd_UP_TS / deltaQd_DW_TS — rep-day aggregated flex deltas over tech_flexLT
    if !isempty(s.tech_flexLT)
        @variable(model, deltaQd_UP_TS[rd = s.repDays, tfl = s.tech_flexLT, ps = pss] <= 0)
        @variable(model, deltaQd_DW_TS[rd = s.repDays, tfl = s.tech_flexLT, ps = pss] >= 0)
        vars.deltaQd_UP_TS = deltaQd_UP_TS
        vars.deltaQd_DW_TS = deltaQd_DW_TS
    end
    if !isempty(s.tech_fWithBattery)
        @variable(model, deltaQ_S_TS[hc = s.hours_cluster, tfwb = s.tech_fWithBattery, ps = pss] <= 0)
        vars.deltaQ_S_TS = deltaQ_S_TS
        # IESA-Opt 1.0 deltaQ_calDayLevel_TS — Range: nonpositive (line 4646)
        @variable(model, deltaQ_calDayLevel_TS[d = s.days, tfwb = s.tech_fWithBattery, ps = pss] <= 0)
        vars.deltaQ_calDayLevel_TS = deltaQ_calDayLevel_TS
    end
    # IESA-Opt 1.0 deltaQ_backlog_DR_TS(hc, tfs, ps) — index tfs = tech_fDRshifting
    if !isempty(s.tech_fDRshifting)
        @variable(model, deltaQ_backlog_DR_TS[hc = s.hours_cluster, tfs = s.tech_fDRshifting, ps = pss] >= 0)
        vars.deltaQ_backlog_DR_TS = deltaQ_backlog_DR_TS
    end
    # IESA-Opt 1.0 deltaQ_backlog_BE_TS(hc, tfe, ps) — index tfe = tech_fBEshifting
    if !isempty(s.tech_fBEshifting)
        @variable(model, deltaQ_backlog_BE_TS[hc = s.hours_cluster, tfe = s.tech_fBEshifting, ps = pss] >= 0)
        vars.deltaQ_backlog_BE_TS = deltaQ_backlog_BE_TS
        # IESA-Opt 1.0 dayStart/dayEnd backlog anchors + calendar-day chain
        @variable(model, deltaQ_backlog_BE_dayStart_TS[rd = s.repDays, tfe = s.tech_fBEshifting, ps = pss] >= 0)
        @variable(model, deltaQ_backlog_BE_dayEnd_TS[rd = s.repDays, tfe = s.tech_fBEshifting, ps = pss] >= 0)
        @variable(model, deltaQ_backlog_BE_calDayLevel_TS[d = s.days, tfe = s.tech_fBEshifting, ps = pss] >= 0)
        vars.deltaQ_backlog_BE_dayStart_TS    = deltaQ_backlog_BE_dayStart_TS
        vars.deltaQ_backlog_BE_dayEnd_TS      = deltaQ_backlog_BE_dayEnd_TS
        vars.deltaQ_backlog_BE_calDayLevel_TS = deltaQ_backlog_BE_calDayLevel_TS
    end
    if !isempty(s.tech_reservoir)
        @variable(model, deltaW_UP_TS[hc = s.hours_cluster, tw = s.tech_reservoir, ps = pss] >= 0)
        # IESA-Opt 1.0 deltaW_S_TS — Range: nonnegative (line 4617)
        @variable(model, deltaW_S_TS[hc = s.hours_cluster, tw = s.tech_reservoir, ps = pss] >= 0)
        vars.deltaW_UP_TS = deltaW_UP_TS
        vars.deltaW_S_TS  = deltaW_S_TS
        # IESA-Opt 1.0 deltaW_dayStart_TS — Range: nonnegative (line 4636)
        @variable(model, deltaW_dayStart_TS[rd = s.repDays, tw = s.tech_reservoir, ps = pss] >= 0)
        # IESA-Opt 1.0 deltaW_dayEnd_TS — Range: nonnegative (line 4640)
        @variable(model, deltaW_dayEnd_TS[rd = s.repDays, tw = s.tech_reservoir, ps = pss] >= 0)
        # IESA-Opt 1.0 deltaW_calDayLevel_TS — Range: nonnegative (line 4654)
        @variable(model, deltaW_calDayLevel_TS[d = s.days, tw = s.tech_reservoir, ps = pss] >= 0)
        vars.deltaW_dayStart_TS    = deltaW_dayStart_TS
        vars.deltaW_dayEnd_TS      = deltaW_dayEnd_TS
        vars.deltaW_calDayLevel_TS = deltaW_calDayLevel_TS
    end
    return vars
end

"""
    _prev_clusterHour(hours_cluster, hc, hoursPer_day_cluster) -> Int

Cyclic predecessor of `hc` WITHIN the same rep-day. If `hc` is the first slot
of its rep-day, returns the last slot of the same rep-day (intra-rep-day
cyclic). Use this for `_TS` ramping and state-recursion constraints.
"""
function _prev_clusterHour(hc::Int, hoursPer_day_cluster::Int)
    slot = mod(hc - 1, hoursPer_day_cluster) + 1
    if slot == 1
        return hc + hoursPer_day_cluster - 1   # wrap to last slot of same rep-day
    else
        return hc - 1
    end
end
