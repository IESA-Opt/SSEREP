# =============================================================================
# ts.jl — Phase 5/6: time-slice (rep-day) constraint families
#
# Mirrors `model/hourly.jl` for the TS LP:
#   - h        → hc (cluster hour)
#   - s.hours  → s.hours_cluster
#   - prev_hour cyclic-year-end  → _prev_clusterHour cyclic-WITHIN-rep-day
#   - annual sums use clusterHourWeight(hc) instead of plain Σ_h
#
# Plus calendar-day anchor constraints (Phase 4 cross-period storage linking):
#   - seasonalDaily_dQ_TS:   calLevel(d) = (1-loss)^24 * calLevel(d-1)
#                                          + Σ_rd dayMix_weight(d,rd) * (dayEnd(rd) - dayStart(rd))
#   - anchor_dQStart_TS:     dayStart(rd)*dayWeight(rd) = Σ_d dayMix_weight(d,rd) * calLevel(d-1)
#   - cycle:                 calLevel(1)  = (1-loss)^24 * calLevel(end) + ...
#
# Entry point: `add_ts_constraints!(m, vars, md)`
#
# IESA-Opt 1.0 source: lines 3215-3692 (TS hourly/daily/CHP), 4551-4644 (storage),
#               4804-5095 (DR/BE/storage/reservoir/anchor).
# =============================================================================

# Same FP-noise threshold as hourly.jl
const _IJ_COEF_EPS_TS = 1e-12

"""
    add_ts_constraints!(m, vars, md) -> Nothing

Add the full TS constraint set to `m`. Requires `vars` to have been populated
by `add_annual_variables!` AND `add_ts_variables!`, and `md` to have been
processed by `build_temporal_clusters!`.
"""
function add_ts_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    @info "  add_ts_constraints! - hourly/daily balance + capacity"
    flush(stderr)
    if get(ENV, "IESA_OPT_TS_SKIP_BALANCE_H", "0") != "1"
        _add_balance_hourly_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED balance hourly (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_BALANCE_D", "0") != "1"
        _add_balance_daily_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED balance daily (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_CAPACITY_H", "0") != "1"
        _add_capacity_hourly_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED capacity hourly (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_RAMPING_H", "0") != "1"
        _add_ramping_hourly_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED ramping hourly (env override)"
    end
    if get(ENV, "IESA_OPT_ENABLE_LINKED_XC", "0") == "1" && get(ENV, "IESA_OPT_TS_SKIP_LINKED_XC", "0") != "1"
        _add_linked_hourly_XC_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED linked hourly XC (env override)"
    end

    @info "  add_ts_constraints! - storage state + flex bounds"
    flush(stderr)
    if get(ENV, "IESA_OPT_TS_SKIP_STORAGE_STATE", "0") != "1"
        _add_storage_state_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED storage state (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_FLEX_BOUNDS", "0") != "1"
        _add_flex_bounds_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED flex bounds (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_FLEX_CLOSED", "0") != "1"
        _add_flex_closed_loop_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED flex closed loop (env override)"
    end

    @info "  add_ts_constraints! - reservoir + gas buffer"
    flush(stderr)
    if get(ENV, "IESA_OPT_TS_SKIP_RESERVOIR", "0") != "1"
        _add_reservoir_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED reservoir (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_GASBUFFER", "0") != "1"
        _add_gasbuffer_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED gas buffer (env override)"
    end

    @info "  add_ts_constraints! - shedding + backlog + CHP"
    flush(stderr)
    if get(ENV, "IESA_OPT_TS_SKIP_SHEDDING", "0") != "1"
        _add_shedding_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED shedding (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_BACKLOG", "0") != "1"
        _add_backlog_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED backlog (env override)"
    end
    if get(ENV, "IESA_OPT_TS_SKIP_CHP", "0") != "1"
        _add_chp_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED CHP (env override)"
    end

    if get(ENV, "IESA_OPT_TS_SKIP_ANCHOR", "0") != "1"
        @info "  add_ts_constraints! - calendar-day storage anchor (Phase 4/5)"
        flush(stderr)
        _add_anchor_constraints_TS!(m, vars, md)
    else
        @info "  add_ts_constraints! - SKIPPED anchor constraints (env override)"
        flush(stderr)
    end
    return nothing
end

# ============================================================================
# Helpers
# ============================================================================

# Pre-aggregate activity_balances by (activity, period) → [(tech_balancer, coef)]
# Same shape as in hourly.jl; duplicated to avoid coupling between modules.
function _ts_build_balance_by_act(p::ModelParams, acts::Vector{Symbol},
                                   pss::AbstractVector{Int}, tb_set::Set{Symbol})
    idx = Dict{Tuple{Symbol,Int}, Vector{Tuple{Symbol,Float64}}}()
    acts_set = Set(acts)
    for ((tb, a, per), coef) in p.activity_balances
        abs(coef) < _IJ_COEF_EPS_TS && continue
        a in acts_set || continue
        per in pss   || continue
        tb in tb_set || continue
        push!(get!(() -> Tuple{Symbol,Float64}[], idx, (a, per)), (tb, coef))
    end
    return idx
end

# Group cluster hours by rep-day for daily aggregations
function _ts_hc_per_rd(s::ModelSets, p::ModelParams)
    out = Dict{Int,Vector{Int}}()
    for hc in s.hours_cluster
        rd = get(p.repDay_of_clusterHour, hc, 0)
        rd == 0 && continue
        push!(get!(() -> Int[], out, rd), hc)
    end
    return out
end

# ============================================================================
# Section 1 — Hourly balance + capacity + ramping (TS)
# ============================================================================

function _add_balance_hourly_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    pss = s.periods_solve
    tu  = vars.tech_use
    isempty(s.hours_cluster) && return
    isempty(s.activities_hour) && return
    tb_set = Set(s.tech_balancers)

    bal_idx = _ts_build_balance_by_act(p, s.activities_hour, pss, tb_set)

    hdisp_set = Set(s.tech_hourlyDispatch)
    ddisp_set = Set(s.tech_dailyDispatch)
    res_set   = Set(s.tech_reservoir)
    flex_set  = Set(s.tech_flexible)
    chp_set   = Set(s.tech_hourlyCHPflex)
    shed_set  = Set(s.tech_shedding)
    # IESA-Opt 1.0 line 2953 — tech_Operation = {t | processType_tech(t)='Operation'} ∪
    #   {t | t∈activities_group-extension via XC/HV-to-MV/LV transformer-loss balances}.
    # In datasets where ActGrouping is empty, the union reduces to processType=='Operation'.
    # NOTE: This set is NOT mutually exclusive with tech_flexible — in IESA-Opt 1.0, both
    # `sum[tp, …]` (passive base) and `sum[tf, (dQ_UP+dQ_DW)*dQ_hourly]` are summed
    # over `tech_balancers`, so a tech with processType='Operation' AND
    # flexibility='DR shifting' contributes to BOTH terms.
    op_set    = _build_tech_Operation(s, p)

    tuh   = vars.tech_useHourly_TS
    tud   = vars.tech_useDaily_TS
    dwUP  = vars.deltaW_UP_TS
    dqUP  = vars.deltaQ_UP_TS
    dqDW  = vars.deltaQ_DW_TS
    duCHP = vars.deltaU_CHP_TS
    dpCHP = vars.deltaP_CHP_TS
    dShed = vars.deltaS_shed_TS

    hpd_c = p.hoursPer_day_cluster

    # Build `dQ_hourly` and `dW_hourly` indicator indices: ah => Vector{(tb, coef)}.
    # IESA-Opt 1.0 line 3219 sums the flex-delta term `(deltaQ_UP_TS+deltaQ_DW_TS)*dQ_hourly(tf,ah)`
    # over ALL tf (not filtered by activity_balances), and the reservoir-charge term
    # `-deltaW_UP_TS*dW_hourly(tw,ah)` over ALL tw. A flex/reservoir tech can therefore
    # contribute to balH_TS[ah,…] purely via dQ_hourly/dW_hourly even when its
    # `activity_balances(tb,ah,ps)` is zero (e.g. LTN01_06 storage on Heat LT Network:
    # dQ_hourly=1, activity_balances=0; the round-trip-loss demand only enters via
    # dQ_hourly). These overlays are added separately below, OUTSIDE the per-(tb,coef)
    # loop driven by activity_balances.
    flex_overlay_idx = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
    for ((tb, a), v) in p.dQ_hourly
        abs(v) < _IJ_COEF_EPS_TS && continue
        tb in flex_set || continue
        push!(get!(() -> Tuple{Symbol,Float64}[], flex_overlay_idx, a), (tb, v))
    end
    res_overlay_idx = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
    for ((tb, a), v) in p.dW_hourly
        abs(v) < _IJ_COEF_EPS_TS && continue
        tb in res_set || continue
        push!(get!(() -> Tuple{Symbol,Float64}[], res_overlay_idx, a), (tb, v))
    end

    for ah in s.activities_hour, ps in pss
        terms       = get(bal_idx, (ah, ps), Tuple{Symbol,Float64}[])
        flex_extras = get(flex_overlay_idx, ah, Tuple{Symbol,Float64}[])
        res_extras  = get(res_overlay_idx,  ah, Tuple{Symbol,Float64}[])
        isempty(terms) && isempty(flex_extras) && isempty(res_extras) && continue
        for hc in s.hours_cluster
            rd = get(p.repDay_of_clusterHour, hc, 0)
            expr = AffExpr(0.0)
            for (tb, coef) in terms
                abs(coef) < _IJ_COEF_EPS_TS && continue

                # Process-type-based contribution (mutually exclusive across
                # th/td/tw/tk/ts/tp). tg has no hourly-balance term in IESA-Opt 1.0.
                if tb in hdisp_set && tuh !== nothing
                    add_to_expression!(expr, coef, tuh[hc, tb, ps])
                elseif tb in ddisp_set && tud !== nothing && rd > 0
                    add_to_expression!(expr, coef / Float64(hpd_c), tud[rd, tb, ps])
                elseif tb in res_set && tuh !== nothing
                    add_to_expression!(expr, coef, tuh[hc, tb, ps])
                elseif tb in chp_set
                    prof = get(p.hourly_profiles_cluster, (hc, get(p.profileType_tech, tb, :Flat)), 0.0)
                    if prof != 0.0
                        add_to_expression!(expr, coef * prof, tu[tb, ps])
                    end
                    duCHP !== nothing && add_to_expression!(expr, coef, duCHP[hc, tb, ps])
                    if dpCHP !== nothing
                        dPe = get(p.dP_electricity, (tb, ah), 0.0)
                        abs(dPe) > _IJ_COEF_EPS_TS && add_to_expression!(expr, dPe, dpCHP[hc, tb, ps])
                    end
                elseif tb in shed_set
                    prof = get(p.hourly_profiles_cluster, (hc, get(p.profileType_tech, tb, :Flat)), 0.0)
                    if prof != 0.0
                        add_to_expression!(expr, coef * prof, tu[tb, ps])
                    end
                    dShed !== nothing && add_to_expression!(expr, coef, dShed[hc, tb, ps])
                elseif tb in op_set
                    # Passive base term: tech_use × profile × activity_balances
                    prof = get(p.hourly_profiles_cluster, (hc, get(p.profileType_tech, tb, :Flat)), 0.0)
                    if prof != 0.0
                        add_to_expression!(expr, coef * prof, tu[tb, ps])
                    end
                end
            end

            # Flex-delta overlay (IESA-Opt 1.0 sum[tf, (deltaQ_UP+deltaQ_DW)*dQ_hourly]).
            # Independent of activity_balances. Iterates over ALL flex techs with
            # non-zero dQ_hourly[(tb,ah)] — including techs with no activity_balances
            # entry for this activity (e.g. LTN01_06 flex=Storage carrying only
            # round-trip charge losses into the Heat LT Network balance).
            if dqUP !== nothing && dqDW !== nothing
                for (tb, dQh) in flex_extras
                    add_to_expression!(expr, dQh, dqUP[hc, tb, ps])
                    add_to_expression!(expr, dQh, dqDW[hc, tb, ps])
                end
            end

            # Reservoir charge term (IESA-Opt 1.0 sum[tw, -deltaW_UP_TS*dW_hourly]).
            # Independent of activity_balances; mirrors flex overlay above.
            if dwUP !== nothing
                for (tb, dWh) in res_extras
                    add_to_expression!(expr, -dWh, dwUP[hc, tb, ps])
                end
            end

            @constraint(m, expr == 0.0, base_name = "balH_TS[$ah,$hc,$ps]")
        end
    end
end

# IESA-Opt 1.0 line 3219 — balance_yearlyhourly_TS:
#   tech_use(th, ps) = Σ_hc clusterHourWeight(hc) * tech_useHourly_TS(hc, th, ps)
function _add_balance_yearlyhourly_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.tech_useHourly_TS === nothing && return
    pss = s.periods_solve
    tu  = vars.tech_use
    tuh = vars.tech_useHourly_TS
    for th in s.tech_hourlyDispatch, ps in pss
        th in s.tech_balancers || continue
        @constraint(m,
            tu[th, ps] == sum(get(p.clusterHourWeight, hc, 1.0) * tuh[hc, th, ps]
                              for hc in s.hours_cluster),
            base_name = "balYH_TS[$th,$ps]")
    end
end

# IESA-Opt 1.0 line 3232 — capacity_techHourly_TS
# Uses MEB CapBound = (1-α)·medoid + α·envelope when ts_capacityProfile_autoMode is on;
# otherwise falls back to cluster-mean (hourly_profiles_cluster).
function _add_capacity_hourly_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.tech_useHourly_TS === nothing && return
    pss = s.periods_solve
    ts  = vars.techStock
    tuh = vars.tech_useHourly_TS
    _add_balance_yearlyhourly_TS!(m, vars, md)
    use_capbound = p.ts_capacityProfile_autoMode && !isempty(p.hourly_profiles_clusterCapBound)
    for th in s.tech_hourlyDispatch, ps in pss
        c2a = get(p.cap2act, th, 0.0)
        prof_t = get(p.profileType_tech, th, :Flat)
        for hc in s.hours_cluster
            prof = if use_capbound
                v = get(p.hourly_profiles_clusterCapBound, (hc, prof_t), nothing)
                v === nothing ? get(p.hourly_profiles_cluster, (hc, prof_t), 0.0) : v
            else
                get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            end
            # IESA-Opt 1.0 capacity_techHourly_TS (line 3232) — NO filter; always generated.
            # When prof*c2a = 0 the constraint reduces to tuh <= 0, which forces
            # tuh = 0 (since tuh >= 0). Skipping leaves tuh unbounded above.
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, tuh[hc, th, ps])
            coef = prof * c2a
            coef == 0.0 || add_to_expression!(expr, -coef, ts[th, ps])
            @constraint(m, expr <= 0.0, base_name = "capH_TS[$th,$hc,$ps]")
        end
    end
end

# IESA-Opt 1.0 lines 3245-3262 — ramping_TS with optional boundary ramping.
# Uses MEB CapBound for the profile envelope, matching IESA-Opt 1.0.
function _add_ramping_hourly_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.tech_useHourly_TS === nothing && return
    pss = s.periods_solve
    ts  = vars.techStock
    tuh = vars.tech_useHourly_TS
    first_slot = isempty(s.hours_inDay_cluster) ? 1 : first(s.hours_inDay_cluster)
    first_hc = first(s.hours_cluster)
    last_hc = last(s.hours_cluster)
    boundary = p.ts_boundaryRamping
    use_capbound = p.ts_capacityProfile_autoMode && !isempty(p.hourly_profiles_clusterCapBound)
    for thr in s.tech_hourlyDispatch, ps in pss
        ramp = get(p.ramping, thr, 0.0)
        ramp == 0.0 && continue
        c2a = get(p.cap2act, thr, 0.0)
        c2a == 0.0 && continue
        prof_t = get(p.profileType_tech, thr, :Flat)
        for hc in s.hours_cluster
            slot = get(p.intradaySlot_of_clusterHour, hc, first_slot)
            (!boundary && slot == first_slot) && continue
            prof = if use_capbound
                v = get(p.hourly_profiles_clusterCapBound, (hc, prof_t), nothing)
                v === nothing ? get(p.hourly_profiles_cluster, (hc, prof_t), 0.0) : v
            else
                get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            end
            prof == 0.0 && continue
            hc_prev = hc == first_hc ? last_hc : hc - 1
            rhs = ramp * c2a * prof * ts[thr, ps]
            @constraint(m, tuh[hc, thr, ps] - tuh[hc_prev, thr, ps] <=  rhs, base_name = "rampUH_TS[$thr,$hc,$ps]")
            @constraint(m, tuh[hc, thr, ps] - tuh[hc_prev, thr, ps] >= -rhs, base_name = "rampDH_TS[$thr,$hc,$ps]")
        end
    end
end

# XC bilateral trade cap (TS version)
function _add_linked_hourly_XC_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.tech_useHourly_TS === nothing && return
    pss = s.periods_solve
    tuh = vars.tech_useHourly_TS
    ts  = vars.techStock
    xc_pairs = Tuple{Symbol,Symbol}[]
    for thh in s.tech_hourlyDispatch
        get(p.tech_category, thh, Symbol("")) == :var"XC Trade" || continue
        sec_thh = get(p.tech_sector, thh, Symbol(""))
        sub_thh = get(p.tech_subsector, thh, Symbol(""))
        for ithh in s.tech_hourlyDispatch
            ithh == thh && continue
            get(p.tech_sector, ithh, Symbol("")) == sub_thh || continue
            get(p.tech_subsector, ithh, Symbol("")) == sec_thh || continue
            push!(xc_pairs, (thh, ithh))
        end
    end
    use_capbound = p.ts_capacityProfile_autoMode && !isempty(p.hourly_profiles_clusterCapBound)
    for (thh, ithh) in xc_pairs, ps in pss
        c2a = get(p.cap2act, thh, 0.0)
        c2a == 0.0 && continue
        prof_t = get(p.profileType_tech, thh, :Flat)
        for hc in s.hours_cluster
            prof = if use_capbound
                v = get(p.hourly_profiles_clusterCapBound, (hc, prof_t), nothing)
                v === nothing ? get(p.hourly_profiles_cluster, (hc, prof_t), 0.0) : v
            else
                get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            end
            prof == 0.0 && continue
            @constraint(m, tuh[hc, thh, ps] + tuh[hc, ithh, ps] <= prof * c2a * ts[thh, ps],
                        base_name = "linkXC_TS[$thh,$ithh,$hc,$ps]")
        end
    end
end

# ============================================================================
# Section 2 — Daily balance + capacity (TS)
# ============================================================================

function _add_balance_daily_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    pss = s.periods_solve
    tu  = vars.tech_use
    tuh = vars.tech_useHourly_TS
    tud = vars.tech_useDaily_TS
    duCHP = vars.deltaU_CHP_TS
    dShed = vars.deltaS_shed_TS
    dB_UP = vars.deltaB_UP_TS
    dB_DW = vars.deltaB_DW_TS

    isempty(s.activities_day) && return
    isempty(s.repDays) && return
    tb_set = Set(s.tech_balancers)
    hdisp_set = Set(s.tech_hourlyDispatch)
    ddisp_set = Set(s.tech_dailyDispatch)
    chp_set   = Set(s.tech_hourlyCHPflex)
    shed_set  = Set(s.tech_shedding)
    gb_set    = Set(s.tech_gasBuffer)
    # IESA-Opt 1.0 line 3303 (balance_activitiesDaily_TS): th, td, tg, tp, tk, ts terms only —
    # NO tw and NO tf in the daily balance. So `op_set` here is the IESA-Opt 1.0-faithful
    # tech_Operation, *not* the complement (the complement would wrongly include
    # tech_reservoir and tech_flexible in the passive base).
    op_set    = _build_tech_Operation(s, p)

    bal_idx = _ts_build_balance_by_act(p, s.activities_day, pss, tb_set)
    hc_per_rd = _ts_hc_per_rd(s, p)

    # Build `dB_daily` indicator index: ad => Vector{(tb, coef)}.
    # IESA-Opt 1.0 line 3303 sums `(deltaB_UP_TS+deltaB_DW_TS)*dB_daily(tg,ad)` over ALL tg,
    # independent of activity_balances. Mirror the hourly TS fix for tf/tw overlays.
    gb_overlay_idx = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
    for ((tb, a), v) in p.dB_daily
        abs(v) < _IJ_COEF_EPS_TS && continue
        tb in gb_set || continue
        push!(get!(() -> Tuple{Symbol,Float64}[], gb_overlay_idx, a), (tb, v))
    end

    for ad in s.activities_day, ps in pss
        terms     = get(bal_idx, (ad, ps), Tuple{Symbol,Float64}[])
        gb_extras = get(gb_overlay_idx, ad, Tuple{Symbol,Float64}[])
        isempty(terms) && isempty(gb_extras) && continue
        for rd in s.repDays
            hcs = get(hc_per_rd, rd, Int[])
            isempty(hcs) && continue
            expr = AffExpr(0.0)
            for (tb, coef) in terms
                abs(coef) < _IJ_COEF_EPS_TS && continue
                if tb in hdisp_set && tuh !== nothing
                    for ih in hcs
                        add_to_expression!(expr, coef, tuh[ih, tb, ps])
                    end
                elseif tb in ddisp_set && tud !== nothing
                    add_to_expression!(expr, coef, tud[rd, tb, ps])
                elseif tb in gb_set
                    # Buffer activity_balances contribution (rare; usually 0 for gas
                    # buffers because they only enter the daily balance via the dB_daily
                    # overlay below). Kept for completeness.
                    # NOTE: gas buffers do not have a direct dispatch variable in the
                    # daily balance — IESA-Opt 1.0 line 3303 has no `tech_use*activity_balances`
                    # term for tg. So treat this branch as no-op.
                    nothing
                elseif tb in chp_set
                    prof_t = get(p.profileType_tech, tb, :Flat)
                    for ih in hcs
                        prof = get(p.hourly_profiles_cluster, (ih, prof_t), 0.0)
                        prof != 0.0 && add_to_expression!(expr, coef * prof, tu[tb, ps])
                        duCHP !== nothing && add_to_expression!(expr, coef, duCHP[ih, tb, ps])
                    end
                elseif tb in shed_set
                    prof_t = get(p.profileType_tech, tb, :Flat)
                    for ih in hcs
                        prof = get(p.hourly_profiles_cluster, (ih, prof_t), 0.0)
                        prof != 0.0 && add_to_expression!(expr, coef * prof, tu[tb, ps])
                        dShed !== nothing && add_to_expression!(expr, coef, dShed[ih, tb, ps])
                    end
                elseif tb in op_set
                    prof_t = get(p.profileType_tech, tb, :Flat)
                    for ih in hcs
                        prof = get(p.hourly_profiles_cluster, (ih, prof_t), 0.0)
                        prof != 0.0 && add_to_expression!(expr, coef * prof, tu[tb, ps])
                    end
                end
            end

            # Gas buffer overlay (IESA-Opt 1.0 sum[tg, (deltaB_UP+deltaB_DW)*dB_daily]).
            # Independent of activity_balances; iterates over ALL gas buffers with
            # non-zero dB_daily[(tb,ad)] regardless of their activity_balances entry.
            if dB_UP !== nothing && dB_DW !== nothing
                for (tb, dBd) in gb_extras
                    add_to_expression!(expr, dBd, dB_UP[rd, tb, ps])
                    add_to_expression!(expr, dBd, dB_DW[rd, tb, ps])
                end
            end

            @constraint(m, expr == 0.0, base_name = "balD_TS[$ad,$rd,$ps]")
        end
    end

    # balance_yearlyDaily_TS: tech_use(td) = Σ_rd dayWeight(rd) * tech_useDaily_TS(rd, td)
    if tud !== nothing
        for td in s.tech_dailyDispatch, ps in pss
            td in s.tech_balancers || continue
            @constraint(m,
                tu[td, ps] == sum(get(p.dayWeight, rd, 1.0) * tud[rd, td, ps] for rd in s.repDays),
                base_name = "balYD_TS[$td,$ps]")
        end

        # capacity_techDaily_TS
        ts = vars.techStock
        use_capbound = p.ts_capacityProfile_autoMode && !isempty(p.hourly_profiles_clusterCapBound)
        for td in s.tech_dailyDispatch, ps in pss
            c2a = get(p.cap2act, td, 0.0)
            prof_t = get(p.profileType_tech, td, :Flat)
            for rd in s.repDays
                hcs = get(hc_per_rd, rd, Int[])
                isempty(hcs) && continue
                prof_sum = sum((if use_capbound
                    v = get(p.hourly_profiles_clusterCapBound, (ih, prof_t), nothing)
                    v === nothing ? get(p.hourly_profiles_cluster, (ih, prof_t), 0.0) : v
                else
                    get(p.hourly_profiles_cluster, (ih, prof_t), 0.0)
                end) for ih in hcs)
                # IESA-Opt 1.0 capacity_techDaily_TS — NO filter; tud <= 0 when c2a*prof_sum = 0
                if c2a == 0.0 || prof_sum == 0.0
                    set_upper_bound(tud[rd, td, ps], 0.0)
                else
                    @constraint(m, tud[rd, td, ps] <= c2a * ts[td, ps] * prof_sum,
                                base_name = "capD_TS[$td,$rd,$ps]")
                end
            end
        end
    end
end

# ============================================================================
# Section 3 — Gas buffer state recursion (TS)
# ============================================================================

function _add_gasbuffer_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaB_S_TS === nothing && return
    pss = s.periods_solve
    ts  = vars.techStock
    dB_UP    = vars.deltaB_UP_TS
    dB_DW    = vars.deltaB_DW_TS
    dB_S     = vars.deltaB_S_TS
    dB_start = vars.deltaB_dayStart_TS
    rds = s.repDays

    for tg in s.tech_gasBuffer, ps in pss
        bUP = get(p.bufferUP_capacity, tg, 0.0)
        bDW = get(p.bufferDW_capacity, tg, 0.0)
        bSt = get(p.buffer_storage, tg, 0.0)
        # IESA-Opt 1.0 deltaB_S_TS Variable-Definition (line 3296):
        #     deltaB_S_TS(rd,tg,ps) = deltaB_dayStart_TS(rd,tg,ps)
        #                           + dayWeight(rd)*(deltaB_UP_TS+deltaB_DW_TS)
        for rd in rds
            dwt = get(p.dayWeight, rd, 0.0)
            @constraint(m,
                dB_S[rd, tg, ps] - dB_start[rd, tg, ps]
                    - dwt * (dB_UP[rd, tg, ps] + dB_DW[rd, tg, ps]) == 0.0,
                base_name = "deltaB_S_TS_definition[$rd,$tg,$ps]")
        end
        # IESA-Opt 1.0 seasonalLink_dB_TS (line 3332): deltaB_dayStart_TS(rd,tg,ps)
        # = deltaB_S_TS(rd-1,tg,ps), cyclic.
        for (i, rd) in enumerate(rds)
            rd_prev = i == 1 ? rds[end] : rds[i - 1]
            @constraint(m,
                dB_start[rd, tg, ps] - dB_S[rd_prev, tg, ps] == 0.0,
                base_name = "seasonalLink_dB_TS[$rd,$tg,$ps]")
        end
        # IESA-Opt 1.0 balanceY_dB_TS (line 3324): Σ_rd dayWeight*(deltaB_UP_TS+deltaB_DW_TS) = 0
        @constraint(m,
            sum(get(p.dayWeight, rd, 1.0) * (dB_UP[rd, tg, ps] + dB_DW[rd, tg, ps]) for rd in rds) == 0.0,
            base_name = "balanceY_dB_TS[$tg,$ps]")
        for rd in rds
            bUP > 0.0 && @constraint(m, dB_UP[rd, tg, ps] >= -bUP * ts[tg, ps],
                                     base_name = "capacityUP_dB_TS[$rd,$tg,$ps]")
            bDW > 0.0 && @constraint(m, dB_DW[rd, tg, ps] <=  bDW * ts[tg, ps],
                                     base_name = "capacityDW_dB_TS[$rd,$tg,$ps]")
            # IESA-Opt 1.0 cummulativeS_dB_TS (note: IESA-Opt 1.0 keeps the typo 'cummulativeS')
            if bSt > 0.0 && bDW > 0.0
                @constraint(m, dB_S[rd, tg, ps] >= -ts[tg, ps] * bSt * bDW,
                            base_name = "cummulativeS_dB_TS[$rd,$tg,$ps]")
            end
        end
    end
end

# ============================================================================
# Section 4 — Storage state recursion (TS, cyclic intra-rep-day)
# ============================================================================

function _add_storage_state_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaQ_S_TS === nothing && return
    pss = s.periods_solve
    ts   = vars.techStock
    tu   = vars.tech_use
    dq_S = vars.deltaQ_S_TS
    dqUP = vars.deltaQ_UP_TS
    dqDW = vars.deltaQ_DW_TS
    hpd_c = p.hoursPer_day_cluster
    isempty(s.hours_cluster) && return

    # IESA-Opt 1.0 deltaQ_S_TS Variable-Definition (line 4559):
    #   if intradaySlot(hc) = first(hours_inDay_cluster):
    #     deltaQ_S_TS(hc) = (1-lch) * deltaQ_UP_TS(hc) + deltaQ_DW_TS(hc)
    #   else:
    #     deltaQ_S_TS(hc) = (1-sl) * deltaQ_S_TS(hc-1)
    #                     + (1-lch) * deltaQ_UP_TS(hc) + deltaQ_DW_TS(hc)
    for tfwb in s.tech_fWithBattery, ps in pss
        sl  = get(p.flex_standing_loss_effective, tfwb, 0.0)
        lch = get(p.flex_loss_charge, tfwb, 0.0)
        chg_factor = 1.0 - lch
        for hc in s.hours_cluster
            sw    = get(p.slice_width_hours, hc, 24.0 / hpd_c)
            decay = (1.0 - sl)^sw
            slot  = get(p.intradaySlot_of_clusterHour, hc, 1)
            rd    = get(p.repDay_of_clusterHour, hc, 1)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dq_S[hc, tfwb, ps])
            if slot == 1
                # IESA-Opt 1.0 prunes inactive deltaQ_dayStart_TS from the generated LP.
            else
                add_to_expression!(expr, -decay, dq_S[hc - 1, tfwb, ps])
            end
            dqUP !== nothing && add_to_expression!(expr, -chg_factor, dqUP[hc, tfwb, ps])
            dqDW !== nothing && add_to_expression!(expr, -1.0, dqDW[hc, tfwb, ps])
            @constraint(m, expr == 0.0, base_name = "deltaQ_S_TS_definition[$hc,$tfwb,$ps]")
        end
    end

    # IESA-Opt 1.0 cumulativeS_dQtfb_TS / cumulativeS_dQtfb_dayStart_TS (lines 4893/4897)
    # for tech_fStorage (tfb).
    for tfb in s.tech_fStorage, ps in pss
        fS = get(p.flex_storage, tfb, 0.0)
        fC = get(p.flex_capacity, (tfb, ps), 0.0)
        (fS > 0.0 && fC > 0.0) || continue
        rhs = -1.0 * fS * fC  # multiplied by techStock at LHS
        for hc in s.hours_cluster
            @constraint(m, dq_S[hc, tfb, ps] + fS * fC * ts[tfb, ps] >= 0.0,
                        base_name = "cumulativeS_dQtfb_TS[$hc,$tfb,$ps]")
        end
    end

    # IESA-Opt 1.0 cumulativeS_dQtfv_TS / cumulativeS_dQtfv_dayStart_TS (lines 4901/4912)
    # IESA-Opt 1.0 minSoC_dQtfv_TS / minSoC_dQtfv_dayStart_TS (lines 4923/4933)
    # for tech_fEV (tfv).
    ev_min = p.ev_min_soc_fraction_default
    for tfv in s.tech_fEV, ps in pss
        fS = get(p.flex_storage, tfv, 0.0)
        fC = get(p.flex_capacity, (tfv, ps), 0.0)
        (fS > 0.0 && fC > 0.0) || continue
        fa = get(p.flex_activity, tfv, Symbol(""))
        ab = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfv, fa, ps), 0.0)
        for hc in s.hours_cluster
            h1 = get(p.cumulativeS_dQtfv_helper1_TS, (tfv, hc), 0.0)
            h2 = get(p.cumulativeS_dQtfv_helper2_TS, (tfv, hc), 0.0)
            # cumulativeS_dQtfv_TS:
            #   deltaQ_S_TS(hc,tfv) >= - ( fS*fC*(ts(tfv) - tu(tfv)*helper1)
            #                            + tu(tfv)*ab*helper2 )
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dq_S[hc, tfv, ps])
            add_to_expression!(expr, fS * fC, ts[tfv, ps])
            add_to_expression!(expr, -fS * fC * h1, tu[tfv, ps])
            add_to_expression!(expr, ab * h2, tu[tfv, ps])
            @constraint(m, expr >= 0.0, base_name = "cumulativeS_dQtfv_TS[$hc,$tfv,$ps]")
            # minSoC_dQtfv_TS:
            #   deltaQ_S_TS(hc,tfv) >= -(1-ev_min)*ts*fS*fC
            @constraint(m, dq_S[hc, tfv, ps] + (1.0 - ev_min) * fS * fC * ts[tfv, ps] >= 0.0,
                        base_name = "minSoC_dQtfv_TS[$hc,$tfv,$ps]")
        end
        for rd in s.repDays
            @constraint(m, (1.0 - ev_min) * fS * fC * ts[tfv, ps] >= 0.0,
                        base_name = "minSoC_dQtfv_dayStart_TS[$rd,$tfv,$ps]")
        end
    end
end

# ============================================================================
# Section 5 — Flex bounds + closed-loop balances (TS)
# ============================================================================

function _add_flex_bounds_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaQ_UP_TS === nothing && return
    pss = s.periods_solve
    tu   = vars.tech_use
    ts   = vars.techStock
    dqUP = vars.deltaQ_UP_TS
    dqDW = vars.deltaQ_DW_TS
    tb_set = Set(s.tech_balancers)
    ev_v2g = p.ev_v2g_power_fraction_default

    # Fine-grained env toggles for diagnostics
    skip_up_BE = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_UP_BE", "0") == "1"
    skip_up_DR = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_UP_DR", "0") == "1"
    skip_up_ST = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_UP_ST", "0") == "1"
    skip_up_EV = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_UP_EV", "0") == "1"
    skip_dw_BE = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_DW_BE", "0") == "1"
    skip_dw_DR = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_DW_DR", "0") == "1"
    skip_dw_ST = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_DW_ST", "0") == "1"
    skip_dw_EVc = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_DW_EVC", "0") == "1"
    skip_dw_EVg = get(ENV, "IESA_OPT_TS_SKIP_FLEXBND_DW_EVG", "0") == "1"

    # IESA-Opt 1.0 capacityUP_dQtfe_TS (line 4732) for tech_fBEshifting
    if !skip_up_BE
    for tfe in s.tech_fBEshifting, ps in pss
        tfe in tb_set || continue
        nnLoad = get(p.flex_nnLoad, tfe, 0.0)
        fC     = get(p.flex_capacity, (tfe, ps), 0.0)
        prof_t = get(p.profileType_tech, tfe, :Flat)
        fa     = get(p.flex_activity, tfe, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfe, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqUP[hc, tfe, ps])
            add_to_expression!(expr, nnLoad * fC, ts[tfe, ps])
            add_to_expression!(expr, nnLoad * prof * ab_fa, tu[tfe, ps])
            @constraint(m, expr >= 0.0, base_name = "capacityUP_dQtfe_TS[$hc,$tfe,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityUP_dQtfs_TS (line 4740) for tech_fDRshifting (no nnLoad)
    if !skip_up_DR
    for tfs in s.tech_fDRshifting, ps in pss
        tfs in tb_set || continue
        fC     = get(p.flex_capacity, (tfs, ps), 0.0)
        prof_t = get(p.profileType_tech, tfs, :Flat)
        fa     = get(p.flex_activity, tfs, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfs, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqUP[hc, tfs, ps])
            add_to_expression!(expr, fC, ts[tfs, ps])
            add_to_expression!(expr, prof * ab_fa, tu[tfs, ps])
            @constraint(m, expr >= 0.0, base_name = "capacityUP_dQtfs_TS[$hc,$tfs,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityUP_dQtfb_TS (line 4748) for tech_fStorage (tfb)
    if !skip_up_ST
    for tfb in s.tech_fStorage, ps in pss
        fC = get(p.flex_capacity, (tfb, ps), 0.0)
        for hc in s.hours_cluster
            @constraint(m, dqUP[hc, tfb, ps] + fC * ts[tfb, ps] >= 0.0,
                        base_name = "capacityUP_dQtfb_TS[$hc,$tfb,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityUP_dQtfv_TS (line 4752) for tech_fEV (tfv)
    if !skip_up_EV
    for tfv in s.tech_fEV, ps in pss
        tfv in tb_set || continue
        fC     = get(p.flex_capacity, (tfv, ps), 0.0)
        prof_t = get(p.profileType_tech, tfv, :Flat)
        fa     = get(p.flex_activity, tfv, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfv, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqUP[hc, tfv, ps])
            add_to_expression!(expr, fC, ts[tfv, ps])
            add_to_expression!(expr, prof * ab_fa, tu[tfv, ps])
            @constraint(m, expr >= 0.0, base_name = "capacityUP_dQtfv_TS[$hc,$tfv,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityDW_dQtfe_TS (line 4766) for tech_fBEshifting
    if !skip_dw_BE
    for tfe in s.tech_fBEshifting, ps in pss
        tfe in tb_set || continue
        nnLoad = get(p.flex_nnLoad, tfe, 0.0)
        prof_t = get(p.profileType_tech, tfe, :Flat)
        fa     = get(p.flex_activity, tfe, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfe, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            @constraint(m, dqDW[hc, tfe, ps] + prof * ab_fa * (1.0 - nnLoad) * tu[tfe, ps] <= 0.0,
                        base_name = "capacityDW_dQtfe_TS[$hc,$tfe,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityDW_dQtfs_TS (line 4770) for tech_fDRshifting
    if !skip_dw_DR
    for tfs in s.tech_fDRshifting, ps in pss
        tfs in tb_set || continue
        nnLoad = get(p.flex_nnLoad, tfs, 0.0)
        prof_t = get(p.profileType_tech, tfs, :Flat)
        fa     = get(p.flex_activity, tfs, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfs, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            @constraint(m, dqDW[hc, tfs, ps] + prof * ab_fa * (1.0 - nnLoad) * tu[tfs, ps] <= 0.0,
                        base_name = "capacityDW_dQtfs_TS[$hc,$tfs,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityDW_dQtfb_TS (line 4774) for tech_fStorage
    if !skip_dw_ST
    for tfb in s.tech_fStorage, ps in pss
        fC = get(p.flex_capacity, (tfb, ps), 0.0)
        for hc in s.hours_cluster
            @constraint(m, dqDW[hc, tfb, ps] - fC * ts[tfb, ps] <= 0.0,
                        base_name = "capacityDW_dQtfb_TS[$hc,$tfb,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityDW_dQtfvc_TS (line 4778) for tech_fEVcharging
    if !skip_dw_EVc
    for tfvc in s.tech_fEVcharging, ps in pss
        tfvc in tb_set || continue
        nnLoad = get(p.flex_nnLoad, tfvc, 0.0)
        prof_t = get(p.profileType_tech, tfvc, :Flat)
        fa     = get(p.flex_activity, tfvc, Symbol(""))
        ab_fa  = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfvc, fa, ps), 0.0)
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            @constraint(m, dqDW[hc, tfvc, ps] + prof * ab_fa * (1.0 - nnLoad) * tu[tfvc, ps] <= 0.0,
                        base_name = "capacityDW_dQtfvc_TS[$hc,$tfvc,$ps]")
        end
    end
    end

    # IESA-Opt 1.0 capacityDW_dQtfvg_TS (line 4782) for tech_fEVgrid
    #   deltaQ_DW_TS(hc,tfvg) <= ev_v2g * fC * (ts(tfvg) - tu(tfvg)*profile_EVuse/avg_speed)
    if !skip_dw_EVg
    for tfvg in s.tech_fEVgrid, ps in pss
        tfvg in tb_set || continue
        fC = get(p.flex_capacity, (tfvg, ps), 0.0)
        prof_ev = get(p.profileType_EVuse, tfvg, Symbol(""))
        spd = get(p.avg_speed, tfvg, 0.0)
        for hc in s.hours_cluster
            ratio = (prof_ev != Symbol("") && spd > 0.0) ?
                    get(p.hourly_profiles_cluster, (hc, prof_ev), 0.0) / spd : 0.0
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqDW[hc, tfvg, ps])
            add_to_expression!(expr, -ev_v2g * fC, ts[tfvg, ps])
            add_to_expression!(expr,  ev_v2g * fC * ratio, tu[tfvg, ps])
            @constraint(m, expr <= 0.0, base_name = "capacityDW_dQtfvg_TS[$hc,$tfvg,$ps]")
        end
    end
    end
end

# IESA-Opt 1.0 deltaQd_UP_TS / deltaQd_DW_TS Variable-Definitions (lines 4587/4593) +
# the seven balance_deltaQd_TS / balanceQ_deltaQtfe_TS rep-day constraints.
function _add_flex_closed_loop_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaQ_UP_TS === nothing && return
    pss = s.periods_solve
    dqUP = vars.deltaQ_UP_TS
    dqDW = vars.deltaQ_DW_TS
    dqdUP = vars.deltaQd_UP_TS
    dqdDW = vars.deltaQd_DW_TS

    hc_per_rd = _ts_hc_per_rd(s, p)

    # IESA-Opt 1.0 deltaQd_UP_TS Variable-Definition: deltaQd_UP_TS(rd,tfl) = sum_hc_in_rd dqUP
    if dqdUP !== nothing
        for tfl in s.tech_flexLT, ps in pss, rd in s.repDays
            hcs = get(hc_per_rd, rd, Int[])
            isempty(hcs) && continue
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqdUP[rd, tfl, ps])
            for hc in hcs
                add_to_expression!(expr, -1.0, dqUP[hc, tfl, ps])
            end
            @constraint(m, expr == 0.0, base_name = "deltaQd_UP_TS_definition[$rd,$tfl,$ps]")
        end
    end
    if dqdDW !== nothing
        for tfl in s.tech_flexLT, ps in pss, rd in s.repDays
            hcs = get(hc_per_rd, rd, Int[])
            isempty(hcs) && continue
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dqdDW[rd, tfl, ps])
            for hc in hcs
                add_to_expression!(expr, -1.0, dqDW[hc, tfl, ps])
            end
            @constraint(m, expr == 0.0, base_name = "deltaQd_DW_TS_definition[$rd,$tfl,$ps]")
        end
    end

    # Helper: per-tech disc factor (1 - flex_loss_discharge_eff), guarded > 0.
    _disc(t) = (d = 1.0 - get(p.flex_loss_discharge_eff, t, 0.0); d <= 0.0 ? 1.0 : d)
    _chg(t)  = 1.0 - get(p.flex_loss_charge, t, 0.0)

    # IESA-Opt 1.0 balanceD_deltaQd_TS (line 4704) for tech_flexD
    if dqdUP !== nothing && dqdDW !== nothing
        for tf_d in s.tech_flexD, ps in pss, rd in s.repDays
            chg = _chg(tf_d); disc = _disc(tf_d)
            @constraint(m,
                chg * dqdUP[rd, tf_d, ps] + (1.0 / disc) * dqdDW[rd, tf_d, ps] == 0.0,
                base_name = "balanceD_deltaQd_TS[$rd,$tf_d,$ps]")
        end

        # IESA-Opt 1.0 balanceR_deltaQd_TS (line 4708) for tech_flexR over r_dayWindow
        for tf_r in s.tech_flexR, ps in pss, r in s.r_dayWindow
            chg = _chg(tf_r); disc = _disc(tf_r)
            terms = AffExpr(0.0)
            for rd in s.repDays
                w = get(p.repDayWeight_range, (rd, r), 0.0)
                w == 0.0 && continue
                add_to_expression!(terms, w * chg, dqdUP[rd, tf_r, ps])
                add_to_expression!(terms, w / disc, dqdDW[rd, tf_r, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceR_deltaQd_TS[$r,$tf_r,$ps]")
        end

        # IESA-Opt 1.0 balanceW_deltaQd_TS (line 4712) for tech_flexW over weeks
        for tf_w in s.tech_flexW, ps in pss, w in s.weeks
            chg = _chg(tf_w); disc = _disc(tf_w)
            terms = AffExpr(0.0)
            for rd in s.repDays
                wt = get(p.repDayWeight_week, (rd, w), 0.0)
                wt == 0.0 && continue
                add_to_expression!(terms, wt * chg, dqdUP[rd, tf_w, ps])
                add_to_expression!(terms, wt / disc, dqdDW[rd, tf_w, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceW_deltaQd_TS[$w,$tf_w,$ps]")
        end

        # IESA-Opt 1.0 balanceM_deltaQd_TS (line 4716) for tech_flexM over months
        for tf_m in s.tech_flexM, ps in pss, mo in s.months
            chg = _chg(tf_m); disc = _disc(tf_m)
            terms = AffExpr(0.0)
            for rd in s.repDays
                wt = get(p.repDayWeight_month, (rd, mo), 0.0)
                wt == 0.0 && continue
                add_to_expression!(terms, wt * chg, dqdUP[rd, tf_m, ps])
                add_to_expression!(terms, wt / disc, dqdDW[rd, tf_m, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceM_deltaQd_TS[$mo,$tf_m,$ps]")
        end

        # IESA-Opt 1.0 balanceS_deltaQd_TS (line 4720) for tech_flexS over seasons
        for tf_s in s.tech_flexS, ps in pss, sn in s.seasons
            chg = _chg(tf_s); disc = _disc(tf_s)
            terms = AffExpr(0.0)
            for rd in s.repDays
                wt = get(p.repDayWeight_season, (rd, sn), 0.0)
                wt == 0.0 && continue
                add_to_expression!(terms, wt * chg, dqdUP[rd, tf_s, ps])
                add_to_expression!(terms, wt / disc, dqdDW[rd, tf_s, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceS_deltaQd_TS[$sn,$tf_s,$ps]")
        end

        # IESA-Opt 1.0 balanceB_deltaQd_TS (line 4724) for tech_flexB over semesters
        for tf_b in s.tech_flexB, ps in pss, b in s.semesters
            chg = _chg(tf_b); disc = _disc(tf_b)
            terms = AffExpr(0.0)
            for rd in s.repDays
                wt = get(p.repDayWeight_semester, (rd, b), 0.0)
                wt == 0.0 && continue
                add_to_expression!(terms, wt * chg, dqdUP[rd, tf_b, ps])
                add_to_expression!(terms, wt / disc, dqdDW[rd, tf_b, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceB_deltaQd_TS[$b,$tf_b,$ps]")
        end

        # IESA-Opt 1.0 balanceY_deltaQd_TS (line 4728) for tech_flexY (sum over rd × dayWeight)
        for tf_y in s.tech_flexY, ps in pss
            chg = _chg(tf_y); disc = _disc(tf_y)
            terms = AffExpr(0.0)
            for rd in s.repDays
                dwt = get(p.dayWeight, rd, 0.0)
                dwt == 0.0 && continue
                add_to_expression!(terms, dwt * chg, dqdUP[rd, tf_y, ps])
                add_to_expression!(terms, dwt / disc, dqdDW[rd, tf_y, ps])
            end
            @constraint(m, terms == 0.0, base_name = "balanceY_deltaQd_TS[$tf_y,$ps]")
        end
    end

    # IESA-Opt 1.0 balanceQ_deltaQtfe_TS (line 4695) for tech_fBEshifting over qc
    for tfe in s.tech_fBEshifting, ps in pss, qc in s.q_hourWindow_cluster
        chg = _chg(tfe); disc = _disc(tfe)
        terms = AffExpr(0.0)
        for hc in s.hours_cluster
            qchc = get(p.quarterPer_clusterHour, hc, 0)
            qchc == qc || continue
            w = get(p.clusterHourWeight, hc, 1.0)
            add_to_expression!(terms, w * chg, dqUP[hc, tfe, ps])
            add_to_expression!(terms, w / disc, dqDW[hc, tfe, ps])
        end
        @constraint(m, terms == 0.0, base_name = "balanceQ_deltaQtfe_TS[$qc,$tfe,$ps]")
    end
end

# ============================================================================
# Section 6 — Reservoir (TS)
# ============================================================================

function _add_reservoir_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaW_S_TS === nothing && return
    pss = s.periods_solve
    dwS  = vars.deltaW_S_TS
    dwUP = vars.deltaW_UP_TS
    dwStart = vars.deltaW_dayStart_TS
    dwEnd   = vars.deltaW_dayEnd_TS
    ts   = vars.techStock
    tuh  = vars.tech_useHourly_TS
    hpd_c = p.hoursPer_day_cluster
    GWtoPJ_y = p.GWtoPJ_y
    GWhtoPJ  = p.GWhtoPJ

    for tw in s.tech_reservoir, ps in pss
        loss = get(p.phs_Losses, tw, 0.0)
        c2a  = get(p.cap2act, tw, 0.0)
        prof_t = get(p.profileType_tech, tw, :Flat)
        phs_cap = get(p.phs_capacity, tw, 0.0)
        phs_str = get(p.phs_storage, tw, 0.0)

        # IESA-Opt 1.0 deltaW_S_TS Variable-Definition (line 4616): emit defining equality.
        for hc in s.hours_cluster
            slot = get(p.intradaySlot_of_clusterHour, hc, 1)
            rd   = get(p.repDay_of_clusterHour, hc, 1)
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            expr = AffExpr(0.0)
            add_to_expression!(expr, 1.0, dwS[hc, tw, ps])
            if slot == 1
                add_to_expression!(expr, -1.0, dwStart[rd, tw, ps])
            else
                add_to_expression!(expr, -1.0, dwS[hc - 1, tw, ps])
            end
            add_to_expression!(expr, -(1.0 - loss), dwUP[hc, tw, ps])
            add_to_expression!(expr, -c2a * prof, ts[tw, ps])
            tuh !== nothing && add_to_expression!(expr, 1.0, tuh[hc, tw, ps])
            @constraint(m, expr == 0.0, base_name = "deltaW_S_TS[$hc,$tw,$ps]")

            # IESA-Opt 1.0 capacityUP_dW_TS (line 4762): dwUP <= ts*phs_capacity*Flat*GWtoPJ_y
            flat = get(p.hourly_profiles_cluster, (hc, :Flat), 1.0)
            if phs_cap > 0.0
                @constraint(m, dwUP[hc, tw, ps] - phs_cap * flat * GWtoPJ_y * ts[tw, ps] <= 0.0,
                            base_name = "capacityUP_dW_TS[$hc,$tw,$ps]")
            end
            # IESA-Opt 1.0 capacityDW_dW_TS (line 4786): tech_useHourly_TS <= ts*Flat*GWtoPJ_y
            if tuh !== nothing
                @constraint(m, tuh[hc, tw, ps] - flat * GWtoPJ_y * ts[tw, ps] <= 0.0,
                            base_name = "capacityDW_dW_TS[$hc,$tw,$ps]")
            end
            # IESA-Opt 1.0 cumulativeS_dW_TS (line 4947): dwS <= ts*phs_storage*GWhtoPJ
            if phs_str > 0.0
                @constraint(m, dwS[hc, tw, ps] - phs_str * GWhtoPJ * ts[tw, ps] <= 0.0,
                            base_name = "cumulativeS_dW_TS[$hc,$tw,$ps]")
            end
        end

        # IESA-Opt 1.0 deltaW_dayEnd_TS Variable-Definition (line 4639):
        #   dwEnd(rd) = dwS(last hc of rd)
        for rd in s.repDays
            hc_last = rd * hpd_c
            @constraint(m, dwEnd[rd, tw, ps] - dwS[hc_last, tw, ps] == 0.0,
                        base_name = "deltaW_dayEnd_TS[$rd,$tw,$ps]")
        end
        # IESA-Opt 1.0 cumulativeS_dW_dayStart_TS (line 4951)
        if phs_str > 0.0
            for rd in s.repDays
                @constraint(m, dwStart[rd, tw, ps] - phs_str * GWhtoPJ * ts[tw, ps] <= 0.0,
                            base_name = "cumulativeS_dW_dayStart_TS[$rd,$tw,$ps]")
            end
        end
    end
end

# ============================================================================
# Section 7 — Shedding (TS)
# ============================================================================

function _add_shedding_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaS_shed_TS === nothing && return
    pss = s.periods_solve
    tu  = vars.tech_use
    ts  = vars.techStock
    dS  = vars.deltaS_shed_TS

    # IESA-Opt 1.0 capacity_deltaS_TS (line 4688): deltaS_shed_TS >= -techStock*shed_capacity
    # IESA-Opt 1.0 sufficiency_deltaS_TS (line 4692): deltaS_shed_TS >= -tech_use*hourly_profile
    # IESA-Opt 1.0 balanceH_deltaS_TS (line 4660): deltaS_shed_TS >= -tech_use*shed_volume*profile (over ts_h)
    # IESA-Opt 1.0 nonPos_Shed_TS (line 5095): deltaS_shed_TS <= 0
    for tsh in s.tech_shedding, ps in pss
        prof_t = get(p.profileType_tech, tsh, :Flat)
        shedCap = get(p.shed_capacity, (tsh, ps), 0.0)
        shedVol = get(p.shed_volume, tsh, 0.0)
        is_shedH = tsh in s.tech_shedH
        is_balancer = tsh in s.tech_balancers
        for hc in s.hours_cluster
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
            # capacity_deltaS_TS
            if shedCap > 0.0
                @constraint(m, dS[hc, tsh, ps] + shedCap * ts[tsh, ps] >= 0.0,
                            base_name = "capacity_deltaS_TS[$hc,$tsh,$ps]")
            end
            # sufficiency_deltaS_TS
            if is_balancer
                @constraint(m, dS[hc, tsh, ps] + prof * tu[tsh, ps] >= 0.0,
                            base_name = "sufficiency_deltaS_TS[$hc,$tsh,$ps]")
            end
            # balanceH_deltaS_TS  (only for tech_shedH)
            if is_shedH && is_balancer
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, dS[hc, tsh, ps])
                coef = shedVol * prof
                coef == 0.0 || add_to_expression!(expr, coef, tu[tsh, ps])
                @constraint(m, expr >= 0.0, base_name = "balanceH_deltaS_TS[$hc,$tsh,$ps]")
            end
            # nonPos_Shed_TS
            @constraint(m, dS[hc, tsh, ps] <= 0.0,
                        base_name = "nonPos_Shed_TS[$hc,$tsh,$ps]")
        end
    end

    # IESA-Opt 1.0 balanceW_deltaS_TS (line 4664): rolling-window day-budget for tech_shedW.
    # Maps cluster hours per calendar-day window to deltaS_shed_TS via mapHour_clusterHour
    # and dayPer_hour. n_days = card(days), horizon = shed_budget_horizon_days(ts_w).
    if !isempty(s.tech_shedW) && !isempty(s.days)
        n_days = length(s.days)
        for tsw in s.tech_shedW, ps in pss
            tsw in s.tech_balancers || continue
            horizon = get(p.shed_budget_horizon_days, tsw, 0)
            horizon > 0 || continue
            shedVol = get(p.shed_volume, tsw, 0.0)
            shedVol > 0.0 || continue
            prof_t = get(p.profileType_tech, tsw, :Flat)
            for d in s.days
                # Determine the rolling window of calendar-days [d - h + 1, d] cyclic.
                lhs = AffExpr(0.0)
                rhs_const = 0.0
                for ih in s.hours
                    dh = get(p.dayPer_hour, ih, 0)
                    in_window = if d > horizon - 1
                        (dh <= d) && (dh > d - horizon)
                    else
                        (dh > n_days + d - horizon) || (dh <= d)
                    end
                    in_window || continue
                    hc = get(p.mapHour_clusterHour, ih, 0)
                    hc == 0 && continue
                    add_to_expression!(lhs, 1.0, dS[hc, tsw, ps])
                    rhs_const += get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
                end
                @constraint(m,
                    lhs + shedVol * rhs_const * tu[tsw, ps] >= 0.0,
                    base_name = "balanceW_deltaS_TS[$d,$tsw,$ps]")
            end
        end
    end
end

# ============================================================================
# Section 8 — Backlog DR + BE (TS) intra-rep-day cyclic
# ============================================================================

function _add_backlog_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    pss = s.periods_solve
    ts   = vars.techStock
    tu   = vars.tech_use
    dqUP = vars.deltaQ_UP_TS
    dqDW = vars.deltaQ_DW_TS
    hpd_c = p.hoursPer_day_cluster
    hcv = collect(s.hours_cluster)
    isempty(hcv) && return

    # Helpers
    _disc(t) = (d = 1.0 - get(p.flex_loss_discharge_eff, t, 0.0); d <= 0.0 ? 1.0 : d)
    _chg(t)  = 1.0 - get(p.flex_loss_charge, t, 0.0)

    # ===== IESA-Opt 1.0 DR shifting (tech_fDRshifting / index tfs) =====
    # IESA-Opt 1.0 stateBalance_dQtfs_TS (line 4790): linear chain over hours_cluster, NOT
    # per-rep-day. Resets at hc=1 only. Closure constraints reset at horizon-day
    # boundaries within the cluster (mod val(rd), backlog_horizon == 0) and at last(repDays).
    if vars.deltaQ_backlog_DR_TS !== nothing && !isempty(s.tech_fDRshifting)
        b = vars.deltaQ_backlog_DR_TS
        for tfs in s.tech_fDRshifting, ps in pss
            chg  = _chg(tfs)
            disc = _disc(tfs)
            fS = get(p.flex_storage, tfs, 0.0)
            fC = get(p.flex_capacity, (tfs, ps), 0.0)
            for (i, hc) in enumerate(hcv)
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, b[hc, tfs, ps])
                if i != 1
                    add_to_expression!(expr, -1.0, b[hcv[i - 1], tfs, ps])
                end
                if dqDW !== nothing
                    add_to_expression!(expr, -(1.0 / disc), dqDW[hc, tfs, ps])
                end
                if dqUP !== nothing
                    add_to_expression!(expr, -chg, dqUP[hc, tfs, ps])
                end
                @constraint(m, expr == 0.0, base_name = "stateBalance_dQtfs_TS[$hc,$tfs,$ps]")
                # IESA-Opt 1.0 stateCap_dQtfs_TS (line 4822): only when flex_storage > 0
                if fS > 0.0 && fC > 0.0
                    @constraint(m, b[hc, tfs, ps] - ts[tfs, ps] * fS * fC <= 0.0,
                                base_name = "stateCap_dQtfs_TS[$hc,$tfs,$ps]")
                end
            end
            # IESA-Opt 1.0 stateClosure_dQtfs_TS (line 4828): backlog at last hc of every horizon-rd = 0
            horiz = get(p.flex_backlog_horizon_days, tfs, 0)
            for (idx, rd) in enumerate(s.repDays)
                fire = (rd == s.repDays[end])
                if !fire && horiz > 0
                    fire = (mod(rd, horiz) == 0)
                end
                fire || continue
                hc_last = rd * hpd_c
                @constraint(m, b[hc_last, tfs, ps] == 0.0,
                            base_name = "stateClosure_dQtfs_TS[$rd,$tfs,$ps]")
            end
        end

        # IESA-Opt 1.0 cumulativeUP_dQtfs_TS (line 4806) for tech_fDRshifting
        for tfs in s.tech_fDRshifting, ps in pss, rd in s.repDays
            tfs in s.tech_balancers || continue
            prof_t = get(p.profileType_tech, tfs, :Flat)
            fa = get(p.flex_activity, tfs, Symbol(""))
            ab = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfs, fa, ps), 0.0)
            lhs = AffExpr(0.0)
            rhs = AffExpr(0.0)
            for hc in s.hours_cluster
                rdhc = get(p.repDay_of_clusterHour, hc, 0)
                rdhc == rd || continue
                add_to_expression!(lhs, 1.0, dqUP[hc, tfs, ps])
                prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
                add_to_expression!(rhs, prof * ab, tu[tfs, ps])
            end
            @constraint(m, lhs - rhs >= 0.0, base_name = "cumulativeUP_dQtfs_TS[$rd,$tfs,$ps]")
        end

        # IESA-Opt 1.0 cumulativeDW_dQtfs_TS (line 4832) for tech_fDRshifting
        for tfs in s.tech_fDRshifting, ps in pss, rd in s.repDays
            tfs in s.tech_balancers || continue
            prof_t = get(p.profileType_tech, tfs, :Flat)
            fa = get(p.flex_activity, tfs, Symbol(""))
            ab = fa == Symbol("") ? 0.0 : get(p.activity_balances, (tfs, fa, ps), 0.0)
            lhs = AffExpr(0.0)
            rhs = AffExpr(0.0)
            for hc in s.hours_cluster
                rdhc = get(p.repDay_of_clusterHour, hc, 0)
                rdhc == rd || continue
                add_to_expression!(lhs, 1.0, dqDW[hc, tfs, ps])
                prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
                add_to_expression!(rhs, -prof * ab, tu[tfs, ps])
            end
            @constraint(m, lhs - rhs <= 0.0, base_name = "cumulativeDW_dQtfs_TS[$rd,$tfs,$ps]")
        end
    end

    # ===== IESA-Opt 1.0 BE shifting (tech_fBEshifting / index tfe) =====
    # IESA-Opt 1.0 stateBalance_dQtfe_TS (line 4844): per-rep-day chain that resets at
    # the first slot of each rep-day to dQ_backlog_BE_dayStart_TS(rd).
    if vars.deltaQ_backlog_BE_TS !== nothing && !isempty(s.tech_fBEshifting)
        b      = vars.deltaQ_backlog_BE_TS
        bStart = vars.deltaQ_backlog_BE_dayStart_TS
        bEnd   = vars.deltaQ_backlog_BE_dayEnd_TS
        for tfe in s.tech_fBEshifting, ps in pss
            chg  = _chg(tfe)
            disc = _disc(tfe)
            fS = get(p.flex_storage, tfe, 0.0)
            fC = get(p.flex_capacity, (tfe, ps), 0.0)
            for hc in s.hours_cluster
                slot = get(p.intradaySlot_of_clusterHour, hc, 1)
                rd   = get(p.repDay_of_clusterHour, hc, 1)
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, b[hc, tfe, ps])
                if slot == 1
                    add_to_expression!(expr, -1.0, bStart[rd, tfe, ps])
                else
                    add_to_expression!(expr, -1.0, b[hc - 1, tfe, ps])
                end
                if dqDW !== nothing
                    add_to_expression!(expr, -(1.0 / disc), dqDW[hc, tfe, ps])
                end
                if dqUP !== nothing
                    add_to_expression!(expr, -chg, dqUP[hc, tfe, ps])
                end
                @constraint(m, expr == 0.0, base_name = "stateBalance_dQtfe_TS[$hc,$tfe,$ps]")
                # IESA-Opt 1.0 stateCap_dQtfe_TS (line 4862): cap on intra-day backlog
                if fS > 0.0 && fC > 0.0
                    @constraint(m, b[hc, tfe, ps] - ts[tfe, ps] * fS * fC <= 0.0,
                                base_name = "stateCap_dQtfe_TS[$hc,$tfe,$ps]")
                end
            end
            # IESA-Opt 1.0 stateCap_dQtfe_dayStart_TS (line 4870)
            if fS > 0.0 && fC > 0.0
                for rd in s.repDays
                    @constraint(m, bStart[rd, tfe, ps] - ts[tfe, ps] * fS * fC <= 0.0,
                                base_name = "stateCap_dQtfe_dayStart_TS[$rd,$tfe,$ps]")
                end
            end
            # IESA-Opt 1.0 stateClosureLast_dQtfe_TS (line 4877):
            #   deltaQ_backlog_BE_dayEnd_TS(last(repDays), tfe) = 0
            @constraint(m, bEnd[s.repDays[end], tfe, ps] == 0.0,
                        base_name = "stateClosureLast_dQtfe_TS[$tfe,$ps]")
            # IESA-Opt 1.0 stateClosureBoundary_dQtfe_TS (line 4881): rd != last AND
            #   (flex_range = '4 hours [q]' OR mod(val(rd), backlog_horizon) = 0)
            horiz = get(p.flex_backlog_horizon_days, tfe, 0)
            frange = get(p.flex_range, tfe, Symbol(""))
            for rd in s.repDays
                rd == s.repDays[end] && continue
                fire = (frange == Symbol("4 hours [q]"))
                if !fire && horiz > 0
                    fire = (mod(rd, horiz) == 0)
                end
                fire || continue
                @constraint(m, bEnd[rd, tfe, ps] == 0.0,
                            base_name = "stateClosureBoundary_dQtfe_TS[$rd,$tfe,$ps]")
            end
        end

        # IESA-Opt 1.0 cumulativeUP_dQtfe_TS (line 4791) for tech_fBEshifting over qc
        for tfe in s.tech_fBEshifting, ps in pss, qc in s.q_hourWindow_cluster
            tfe in s.tech_balancers || continue
            rhs_const = get(p.cumulative_dQtfe_rhs_TS, (qc, tfe, ps), 0.0)
            lhs = AffExpr(0.0)
            for hc in s.hours_cluster
                qchc = get(p.quarterPer_clusterHour, hc, 0)
                qchc == qc || continue
                w = get(p.clusterHourWeight, hc, 1.0)
                add_to_expression!(lhs, w, dqUP[hc, tfe, ps])
            end
            # IESA-Opt 1.0 form: lhs >= rhs_const * tu(tfe)
            @constraint(m, lhs - rhs_const * tu[tfe, ps] >= 0.0,
                        base_name = "cumulativeUP_dQtfe_TS[$qc,$tfe,$ps]")
        end

        # IESA-Opt 1.0 cumulativeDW_dQtfe_TS (line 4838) for tech_fBEshifting over qc
        for tfe in s.tech_fBEshifting, ps in pss, qc in s.q_hourWindow_cluster
            tfe in s.tech_balancers || continue
            rhs_const = get(p.cumulative_dQtfe_rhs_TS, (qc, tfe, ps), 0.0)
            lhs = AffExpr(0.0)
            for hc in s.hours_cluster
                qchc = get(p.quarterPer_clusterHour, hc, 0)
                qchc == qc || continue
                w = get(p.clusterHourWeight, hc, 1.0)
                add_to_expression!(lhs, w, dqDW[hc, tfe, ps])
            end
            @constraint(m, lhs + rhs_const * tu[tfe, ps] <= 0.0,
                        base_name = "cumulativeDW_dQtfe_TS[$qc,$tfe,$ps]")
        end
    end
end

# ============================================================================
# Section 9 — CHP (TS)
# ============================================================================

function _add_chp_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    vars.deltaU_CHP_TS === nothing && return
    pss = s.periods_solve
    tu    = vars.tech_use
    duCHP = vars.deltaU_CHP_TS
    dpCHP = vars.deltaP_CHP_TS
    hcv   = s.hours_cluster
    first_slot = isempty(s.hours_inDay_cluster) ? 1 : first(s.hours_inDay_cluster)
    boundary  = p.ts_boundaryRamping

    # IESA-Opt 1.0 balanceH_deltaHchp_TS (line 3586) only fires for tech_hourlyCHPflexH (tk_h).
    # All other CHP equations are over the parent set tech_hourlyCHPflex (tk).
    tk_h_set = Set(s.tech_hourlyCHPflexH)

    for tk in s.tech_hourlyCHPflex, ps in pss
        prof_t = get(p.profileType_tech, tk, :Flat)
        eta    = get(p.CHP_eta, tk, 0.0)
        eps_   = max(get(p.CHP_eps, (tk, ps), 0.0), 0.01)
        dev_u  = get(p.CHP_dev_use, tk, 0.0)
        dev_pH = get(p.CHP_dev_PtoH, tk, 0.0)
        ramp_t = get(p.ramping, tk, 0.0)
        ramp_eff = ramp_t > 0.0 ? ramp_t : 1.0
        prod_a = get(p.CHP_prod, tk, Symbol(""))
        ab_prod = prod_a == Symbol("") ? 0.0 : get(p.activity_balances, (tk, prod_a, ps), 0.0)

        ab_elec = 0.0
        for ((tk2, iah), v) in p.dP_electricity
            tk2 == tk || continue
            v == 1.0 || continue
            ab_elec += get(p.activity_balances, (tk, iah, ps), 0.0)
        end

        is_h = tk in tk_h_set

        for (i, hc) in enumerate(hcv)
            prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)

            # IESA-Opt 1.0 balanceH_deltaHchp_TS (line 3586) — only for tk_h
            if is_h && abs(ab_prod) > _IJ_COEF_EPS_TS && eta > 0.0
                @constraint(m, ab_prod * duCHP[hc, tk, ps] - (eta / eps_) * dpCHP[hc, tk, ps] == 0.0,
                            base_name = "balanceH_deltaHchp_TS[$hc,$tk,$ps]")
            end

            # IESA-Opt 1.0 capacityUP_deltaUchp_TS (line 3616): dU <= tu*prof*CHPdev_use
            # IESA-Opt 1.0 capacityDW_deltaUchp_TS (line 3620): dU >= -tu*prof*CHPdev_use
            coef_u = prof * dev_u
            expr_up_u = AffExpr(0.0)
            add_to_expression!(expr_up_u, 1.0, duCHP[hc, tk, ps])
            abs(coef_u) <= _IJ_COEF_EPS_TS || add_to_expression!(expr_up_u, -coef_u, tu[tk, ps])
            @constraint(m, expr_up_u <= 0.0, base_name = "capacityUP_deltaUchp_TS[$hc,$tk,$ps]")

            expr_dw_u = AffExpr(0.0)
            add_to_expression!(expr_dw_u, 1.0, duCHP[hc, tk, ps])
            abs(coef_u) <= _IJ_COEF_EPS_TS || add_to_expression!(expr_dw_u, coef_u, tu[tk, ps])
            @constraint(m, expr_dw_u >= 0.0, base_name = "capacityDW_deltaUchp_TS[$hc,$tk,$ps]")

            # IESA-Opt 1.0 capacityUP_deltaPchp_TS (line 3624) /
            #       capacityDW_deltaPchp_TS (line 3632) — also unconditional.
            if abs(ab_elec * dev_pH) > _IJ_COEF_EPS_TS
                @constraint(m, dpCHP[hc, tk, ps]
                              - ab_elec * dev_pH * prof * tu[tk, ps]
                              - ab_elec * dev_pH * duCHP[hc, tk, ps] <= 0.0,
                            base_name = "capacityUP_deltaPchp_TS[$hc,$tk,$ps]")
                @constraint(m, dpCHP[hc, tk, ps]
                              + ab_elec * dev_pH * prof * tu[tk, ps]
                              + ab_elec * dev_pH * duCHP[hc, tk, ps] >= 0.0,
                            base_name = "capacityDW_deltaPchp_TS[$hc,$tk,$ps]")
            else
                set_upper_bound(dpCHP[hc, tk, ps], 0.0)
                set_lower_bound(dpCHP[hc, tk, ps], 0.0)
            end

            # IESA-Opt 1.0 rampingUP/DW_delta{U,P}chp_TS (lines 3640-3715)
            slot = get(p.intradaySlot_of_clusterHour, hc, 1)
            chp_ramp_enabled = get(ENV, "IESA_DISABLE_CHP_RAMP", "0") != "1"
            apply_ramp = chp_ramp_enabled && (boundary || slot != first_slot) && prof > 0.0 && tk in s.tech_balancers
            if apply_ramp
                hc_prev = i == 1 ? hcv[end] : hcv[i - 1]
                rhs_u = prof * dev_u * ramp_eff
                if abs(rhs_u) > _IJ_COEF_EPS_TS
                    @constraint(m, duCHP[hc, tk, ps] - duCHP[hc_prev, tk, ps]
                                   - rhs_u * tu[tk, ps] <= 0.0,
                                base_name = "rampingUP_deltaUchp_TS[$hc,$tk,$ps]")
                    @constraint(m, duCHP[hc, tk, ps] - duCHP[hc_prev, tk, ps]
                                   + rhs_u * tu[tk, ps] >= 0.0,
                                base_name = "rampingDW_deltaUchp_TS[$hc,$tk,$ps]")
                end
                rhs_p = prof * ab_elec * dev_pH * ramp_eff
                if abs(rhs_p) > _IJ_COEF_EPS_TS
                    @constraint(m, dpCHP[hc, tk, ps] - dpCHP[hc_prev, tk, ps]
                                   - rhs_p * tu[tk, ps] <= 0.0,
                                base_name = "rampingUP_deltaPchp_TS[$hc,$tk,$ps]")
                    @constraint(m, dpCHP[hc, tk, ps] - dpCHP[hc_prev, tk, ps]
                                   + rhs_p * tu[tk, ps] >= 0.0,
                                base_name = "rampingDW_deltaPchp_TS[$hc,$tk,$ps]")
                end
            end
        end
    end

    # IESA-Opt 1.0 balanceD_deltaHchp_TS (line 3594) for tech_hourlyCHPflexD (tk_d).
    # Σ_{ihc | rd_of_ihc=rd} [ (tu*prof + dU)*ab_prod − eta*dP/eps ] = Σ_{ihc} (tu*prof)*ab_prod
    # Cancellation gives:  Σ dU * ab_prod  −  Σ (eta/eps) * dP  = 0  (per rd, tk_d, ps).
    if !isempty(s.tech_hourlyCHPflexD)
        for tk_d in s.tech_hourlyCHPflexD, ps in pss, rd in s.repDays
            prod_a = get(p.CHP_prod, tk_d, Symbol(""))
            ab_prod = prod_a == Symbol("") ? 0.0 : get(p.activity_balances, (tk_d, prod_a, ps), 0.0)
            eta = get(p.CHP_eta, tk_d, 0.0)
            eps_ = max(get(p.CHP_eps, (tk_d, ps), 0.0), 0.01)
            (abs(ab_prod) < _IJ_COEF_EPS_TS && eta == 0.0) && continue
            expr = AffExpr(0.0)
            for hc in s.hours_cluster
                rdhc = get(p.repDay_of_clusterHour, hc, 0)
                rdhc == rd || continue
                add_to_expression!(expr, ab_prod, duCHP[hc, tk_d, ps])
                add_to_expression!(expr, -(eta / eps_), dpCHP[hc, tk_d, ps])
            end
            @constraint(m, expr == 0.0, base_name = "balanceD_deltaHchp_TS[$rd,$tk_d,$ps]")
        end
    end

    # IESA-Opt 1.0 balanceW_deltaHchp_TS (line 3605) for tech_hourlyCHPflexW (tk_w).
    # Same cancellation as above, weighted by repDayWeight_week(rd, w).
    if !isempty(s.tech_hourlyCHPflexW) && !isempty(s.weeks)
        for tk_w in s.tech_hourlyCHPflexW, ps in pss, w in s.weeks
            prod_a = get(p.CHP_prod, tk_w, Symbol(""))
            ab_prod = prod_a == Symbol("") ? 0.0 : get(p.activity_balances, (tk_w, prod_a, ps), 0.0)
            eta = get(p.CHP_eta, tk_w, 0.0)
            eps_ = max(get(p.CHP_eps, (tk_w, ps), 0.0), 0.01)
            (abs(ab_prod) < _IJ_COEF_EPS_TS && eta == 0.0) && continue
            expr = AffExpr(0.0)
            for rd in s.repDays
                wt = get(p.repDayWeight_week, (rd, w), 0.0)
                wt == 0.0 && continue
                for hc in s.hours_cluster
                    rdhc = get(p.repDay_of_clusterHour, hc, 0)
                    rdhc == rd || continue
                    add_to_expression!(expr, wt * ab_prod, duCHP[hc, tk_w, ps])
                    add_to_expression!(expr, -wt * (eta / eps_), dpCHP[hc, tk_w, ps])
                end
            end
            @constraint(m, expr == 0.0, base_name = "balanceW_deltaHchp_TS[$w,$tk_w,$ps]")
        end
    end
end

# ============================================================================
# Section 10 — Calendar-day storage anchor (Phase 4: cross-period linking)
#
# This is the key TS mechanism that propagates storage state across days
# beyond the within-rep-day cyclic loop.  Each calendar day d gets an
# "anchor" level `calLevel(d)` that is itself cyclic over 365 days.
#
# IESA-Opt 1.0 source (lines 5011-5089):
#   seasonalDaily_dQ_TS:      calLevel(d) − (1−sl)^24·calLevel(d−1)
#                             − Σ_rd dayMix_weight(d,rd) · (dayEnd(rd) − dayStart(rd)) = 0
#   anchor_dQStart_TS:        dayStart(rd) · dayWeight(rd)
#                             = Σ_d dayMix_weight(d,rd) · calLevel(d−1)
#   Cycle: calLevel(1) refers to calLevel(365) via cyclic recursion.
# ============================================================================

function _add_anchor_constraints_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    pss = s.periods_solve
    days = s.days
    rds  = s.repDays
    ts   = vars.techStock

    # =====================================================================
    # IESA-Opt 1.0 Constraints (calendar-day chain over storage & reservoir & BE):
    #   seasonalDaily_dQ_TS         (line 5011) — tfwb, decayed
    #   anchor_dQStart_TS           (line 5038) — tfwb
    #   capLevelLB_dQ_TS            (line 5055) — tfwb (deltaQ is nonpositive)
    #   seasonalDaily_dW_TS         (line 5060) — tw, no decay
    #   anchor_dWStart_TS           (line 5076) — tw
    #   seasonalDaily_dQbacklogBE_TS(line 4965) — tfe, no decay
    #   anchor_dQbacklogBEStart_TS  (line 4990) — tfe
    #   capLevelUB_dQbacklogBE_TS   (line 5006) — tfe
    # =====================================================================

    # ---------- Storage (tech_fWithBattery) ----------
    if vars.deltaQ_calDayLevel_TS !== nothing
        calL   = vars.deltaQ_calDayLevel_TS

        for tfwb in s.tech_fWithBattery, ps in pss
            sl = get(p.flex_standing_loss_effective, tfwb, 0.0)
            decay = (1.0 - sl)^24.0
            fS = get(p.flex_storage, tfwb, 0.0)
            fC = get(p.flex_capacity, (tfwb, ps), 0.0)

            # IESA-Opt 1.0 seasonalDaily_dQ_TS
            for (i, d) in enumerate(days)
                d_prev = i == 1 ? days[end] : days[i - 1]
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, calL[d, tfwb, ps])
                add_to_expression!(expr, -decay, calL[d_prev, tfwb, ps])
                @constraint(m, expr == 0.0, base_name = "seasonalDaily_dQ_TS[$d,$tfwb,$ps]")

                # IESA-Opt 1.0 capLevelLB_dQ_TS (line 5055): deltaQ_calDayLevel_TS >= -techStock*flex_storage*flex_capacity
                if fS > 0.0 && fC > 0.0
                    @constraint(m, calL[d, tfwb, ps] + fS * fC * ts[tfwb, ps] >= 0.0,
                                base_name = "capLevelLB_dQ_TS[$d,$tfwb,$ps]")
                end
            end

            # IESA-Opt 1.0 anchor_dQStart_TS
            for rd in rds
                dw = get(p.dayWeight, rd, 0.0)
                dw == 0.0 && continue
                expr = AffExpr(0.0)
                for (i, d) in enumerate(days)
                    w = get(p.dayMix_weight, (d, rd), 0.0)
                    w == 0.0 && continue
                    d_prev = i == 1 ? days[end] : days[i - 1]
                    add_to_expression!(expr, -w, calL[d_prev, tfwb, ps])
                end
                @constraint(m, expr == 0.0, base_name = "anchor_dQStart_TS[$rd,$tfwb,$ps]")
            end
        end
    end

    # ---------- Reservoir (tech_reservoir) ----------
    if vars.deltaW_calDayLevel_TS !== nothing
        calL   = vars.deltaW_calDayLevel_TS
        dStart = vars.deltaW_dayStart_TS
        dEnd   = vars.deltaW_dayEnd_TS
        ts     = vars.techStock

        for tw in s.tech_reservoir, ps in pss
            # IESA-Opt 1.0 seasonalDaily_dW_TS (line 5060) — no decay
            for (i, d) in enumerate(days)
                d_prev = i == 1 ? days[end] : days[i - 1]
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, calL[d, tw, ps])
                add_to_expression!(expr, -1.0, calL[d_prev, tw, ps])
                for rd in rds
                    w = get(p.dayMix_weight, (d, rd), 0.0)
                    w == 0.0 && continue
                    add_to_expression!(expr, -w, dEnd[rd, tw, ps])
                    add_to_expression!(expr,  w, dStart[rd, tw, ps])
                end
                @constraint(m, expr == 0.0, base_name = "seasonalDaily_dW_TS[$d,$tw,$ps]")
            end
            # IESA-Opt 1.0 anchor_dWStart_TS (line 5076)
            for rd in rds
                dwt = get(p.dayWeight, rd, 0.0)
                dwt == 0.0 && continue
                expr = AffExpr(0.0)
                add_to_expression!(expr, dwt, dStart[rd, tw, ps])
                for (i, d) in enumerate(days)
                    w = get(p.dayMix_weight, (d, rd), 0.0)
                    w == 0.0 && continue
                    d_prev = i == 1 ? days[end] : days[i - 1]
                    add_to_expression!(expr, -w, calL[d_prev, tw, ps])
                end
                @constraint(m, expr == 0.0, base_name = "anchor_dWStart_TS[$rd,$tw,$ps]")
            end
            # IESA-Opt 1.0 capLevelUB_dW_TS (line 5089):
            #   deltaW_calDayLevel_TS(d,tw,ps) <= techStock(tw,ps)*phs_storage(tw)*GWhtoPJ
            phs = get(p.phs_storage, tw, 0.0)
            if phs > 0.0
                for d in days
                    @constraint(m, calL[d, tw, ps] - phs * p.GWhtoPJ * ts[tw, ps] <= 0.0,
                                base_name = "capLevelUB_dW_TS[$d,$tw,$ps]")
                end
            end
        end
    end

    # ---------- BE backlog (tech_fBEshifting) ----------
    if vars.deltaQ_backlog_BE_calDayLevel_TS !== nothing
        calL   = vars.deltaQ_backlog_BE_calDayLevel_TS
        dStart = vars.deltaQ_backlog_BE_dayStart_TS
        dEnd   = vars.deltaQ_backlog_BE_dayEnd_TS

        for tfe in s.tech_fBEshifting, ps in pss
            fS = get(p.flex_storage, tfe, 0.0)
            fC = get(p.flex_capacity, (tfe, ps), 0.0)

            # IESA-Opt 1.0 seasonalDaily_dQbacklogBE_TS (line 4965) — no decay
            for (i, d) in enumerate(days)
                d_prev = i == 1 ? days[end] : days[i - 1]
                expr = AffExpr(0.0)
                add_to_expression!(expr, 1.0, calL[d, tfe, ps])
                add_to_expression!(expr, -1.0, calL[d_prev, tfe, ps])
                for rd in rds
                    w = get(p.dayMix_weight, (d, rd), 0.0)
                    w == 0.0 && continue
                    add_to_expression!(expr, -w, dEnd[rd, tfe, ps])
                    add_to_expression!(expr,  w, dStart[rd, tfe, ps])
                end
                @constraint(m, expr == 0.0, base_name = "seasonalDaily_dQbacklogBE_TS[$d,$tfe,$ps]")

                # IESA-Opt 1.0 capLevelUB_dQbacklogBE_TS (line 5006): only when flex_storage > 0
                if fS > 0.0 && fC > 0.0
                    @constraint(m, calL[d, tfe, ps] - fS * fC * ts[tfe, ps] <= 0.0,
                                base_name = "capLevelUB_dQbacklogBE_TS[$d,$tfe,$ps]")
                end
            end
            # IESA-Opt 1.0 anchor_dQbacklogBEStart_TS (line 4990)
            for rd in rds
                dwt = get(p.dayWeight, rd, 0.0)
                dwt == 0.0 && continue
                expr = AffExpr(0.0)
                add_to_expression!(expr, dwt, dStart[rd, tfe, ps])
                for (i, d) in enumerate(days)
                    w = get(p.dayMix_weight, (d, rd), 0.0)
                    w == 0.0 && continue
                    d_prev = i == 1 ? days[end] : days[i - 1]
                    add_to_expression!(expr, -w, calL[d_prev, tfe, ps])
                end
                @constraint(m, expr == 0.0, base_name = "anchor_dQbacklogBEStart_TS[$rd,$tfe,$ps]")
            end
        end
    end
end
