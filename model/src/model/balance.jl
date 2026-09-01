# =============================================================================
# balance.jl — Activity balance + capacity + emission-target constraints
#
# Phase 2 (annual LP). Mirrors IESA-Opt 1.0:
#   - balance_activities             (ab, ps) >= rhs
#   - balance_activitiesFix          (af, ps) == rhs
#   - balance_activities_matconv     (amc, ps) == rhs
#   - balance_activities_emissionsFix (acf, ps) == 0
#   - balance_activities_EmissionTargetAir         (n, ps | n='NL') <= cap
#   - balance_activities_EmissionTarget_inclScope3andFuelEx (ps)   <= cap
#   - balance_activities_EmissionTargetBunker      (n, ps)          <= cap
#   - balance_activities_EmissionTargetFS          (n, ps)          <= cap
#   - balance_activities_EmissionTargetAll         (n, ps)          <= cap
#   - capacity_technologies          (t, ps)
#   - emission_targetCum             (ni)
#   - storageCO2_cumulative          (ni)
#   - min_techUse_constraint, max_techUse_constraint (t, ps | <>na)
#
# Annual-only builds see every `delta*` helper as zero. FH/TS builds add the
# aggregate deviation terms declared before this file is called.
# =============================================================================

const _IJ_COEF_EPS_BAL = 1e-12

"""
    add_balance_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData) -> Nothing
"""
function add_balance_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s   = md.sets
    p   = md.params
    pss = s.periods_solve
    tu  = vars.tech_use
    ts  = vars.techStock

    # Precompute reverse map: activity → list of (tb, coef) for fast LP build
    bal_by_act = _build_balance_index(s, p, pss)

    # -------------------------------------------------------------------------
    # balance_activities (IESA-Opt 1.0 line ~2426): >=
    #   sum[tb, tech_use(tb,ps)*activity_balances(tb,ab,ps)] >= activities_netVolumes(ab, ps)
    # -------------------------------------------------------------------------
    for ab in s.activities_balance, ps in pss
        terms = get(bal_by_act, (ab, ps), Tuple{Symbol,Float64}[])
        rhs = get(p.activities_netVolumes, (ab, ps), 0.0)
        expr = AffExpr(0.0)
        has_terms = false
        for (tb, coef) in terms
            add_to_expression!(expr, coef, tu[tb, ps])
            has_terms = true
        end
        has_terms |= _add_activity_deviation_terms!(expr, vars, s, p, ab, ps)
        if !has_terms
            rhs <= 0.0 && continue   # 0 >= rhs<=0 is trivially true
            @constraint(m, 0.0 >= rhs, base_name = "balance[$(ab),$(ps)]")
            continue
        end
        @constraint(m, expr >= rhs, base_name = "balance[$(ab),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # balance_activitiesFix (IESA-Opt 1.0 line ~2455): ==
    # -------------------------------------------------------------------------
    for af in s.activities_fixEnergy, ps in pss
        terms = get(bal_by_act, (af, ps), Tuple{Symbol,Float64}[])
        rhs = get(p.activities_netVolumes, (af, ps), 0.0)
        expr = AffExpr(0.0)
        has_terms = false
        for (tb, coef) in terms
            add_to_expression!(expr, coef, tu[tb, ps])
            has_terms = true
        end
        has_terms |= _add_activity_deviation_terms!(expr, vars, s, p, af, ps)
        if !has_terms
            rhs == 0.0 && continue
            @constraint(m, 0.0 == rhs, base_name = "balanceFix[$(af),$(ps)]")
            continue
        end
        @constraint(m, expr == rhs, base_name = "balanceFix[$(af),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # balance_activities_matconv (IESA-Opt 1.0 line ~2471): ==
    # -------------------------------------------------------------------------
    for amc in s.activities_materialConversion, ps in pss
        terms = get(bal_by_act, (amc, ps), Tuple{Symbol,Float64}[])
        rhs = get(p.activities_netVolumes, (amc, ps), 0.0)
        expr = AffExpr(0.0)
        has_terms = false
        for (tb, coef) in terms
            add_to_expression!(expr, coef, tu[tb, ps])
            has_terms = true
        end
        has_terms |= _add_activity_deviation_terms!(expr, vars, s, p, amc, ps)
        if !has_terms
            rhs == 0.0 && continue
            @constraint(m, 0.0 == rhs, base_name = "balanceMatconv[$(amc),$(ps)]")
            continue
        end
        @constraint(m, expr == rhs, base_name = "balanceMatconv[$(amc),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # balance_activities_emissionsFix (IESA-Opt 1.0 line ~2485): == 0
    #   For each emission-fix activity acf, sum over all techs of
    #   tech_use(t, ps) * activity_balances(t, acf, ps) == 0
    # (Phase 2 drops the deltaU_CHP and deltaS_shed terms — they sum to 0 in annual mode anyway.)
    # -------------------------------------------------------------------------
    for acf in _activities_emissionFix(s, p), ps in pss
        terms = _bal_terms_all_techs(s, p, acf, ps)
        isempty(terms) && continue
        @constraint(m,
            sum(coef * tu[tb, ps] for (tb, coef) in terms) == 0,
            base_name = "balanceEmFix[$(acf),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # capacity_technologies (IESA-Opt 1.0 line ~2566):
    #   tech_use(t, ps) <= cap2act(t)*techStock(t, ps)
    #                    + sum(deltaW_UP[_TS])*(1 - phs_Losses(t))
    # The reservoir adder is present only in FH/TS builds and is zero for
    # annual-only builds where deltaW_UP variables are absent.
    # -------------------------------------------------------------------------
    balancer_set = Set(s.tech_balancers)
    for t in s.technologies, ps in pss
        c2a = get(p.cap2act, t, 0.0)
        res_adder, has_res_adder = _reservoir_capacity_adder(vars, s, p, t, ps)
        expr = AffExpr(0.0)
        t in balancer_set && add_to_expression!(expr, 1.0, tu[t, ps])
        c2a == 0.0 || add_to_expression!(expr, -c2a, ts[t, ps])
        if has_res_adder
            add_to_expression!(expr, -1.0, res_adder)
        end
        @constraint(m, expr <= 0.0, base_name = "cap[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # min_techUse / max_techUse (IESA-Opt 1.0 lines ~2588-2611): only where <>na
    # (Phase 2 drops deltaU_CHP + deltaS_shed sums.)
    # -------------------------------------------------------------------------
    for ((t, ps), v) in p.techUse_min
        ps in pss || continue
        t in s.tech_balancers || continue
        @constraint(m, tu[t, ps] >= v, base_name = "minUse[$(t),$(ps)]")
    end
    for ((t, ps), v) in p.techUse_max
        ps in pss || continue
        t in s.tech_balancers || continue
        @constraint(m, tu[t, ps] <= v, base_name = "maxUse[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # Emission-target constraints (Phase 2 — annual portion only, deltas dropped):
    # -------------------------------------------------------------------------
    _add_emission_targets!(m, vars, md, bal_by_act)

    return nothing
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Pre-aggregate activity_balances by (activity, period) → list of (tech, coef)
# Skips zero coefficients. Only includes tb ∈ tech_balancers (where tu is defined).
function _build_balance_index(s::ModelSets, p::ModelParams, pss::AbstractVector{Int})
    index = Dict{Tuple{Symbol,Int}, Vector{Tuple{Symbol,Float64}}}()
    tb_set = Set(s.tech_balancers)
    for ((tb, a, per), coef) in p.activity_balances
        coef == 0.0 && continue
        tb in tb_set || continue
        per in pss   || continue
        push!(get!(() -> Tuple{Symbol,Float64}[], index, (a, per)), (tb, coef))
    end
    return index
end

function _add_activity_deviation_terms!(expr::AffExpr, vars::AnnualVars,
                                        s::ModelSets, p::ModelParams,
                                        activity::Symbol, ps::Int)
    has_terms = false

    dqUP_TS = vars.deltaQ_UP_TS
    dqDW_TS = vars.deltaQ_DW_TS
    if dqUP_TS !== nothing && dqDW_TS !== nothing
        for tf in s.tech_flexible
            dQh = get(p.dQ_hourly, (tf, activity), 0.0)
            abs(dQh) < _IJ_COEF_EPS_BAL && continue
            for hc in s.hours_cluster
                w = get(p.clusterHourWeight, hc, 0.0)
                abs(w) < _IJ_COEF_EPS_BAL && continue
                coef = w * dQh
                add_to_expression!(expr, coef, dqUP_TS[hc, tf, ps])
                add_to_expression!(expr, coef, dqDW_TS[hc, tf, ps])
                has_terms = true
            end
        end
    elseif vars.deltaQ_UP !== nothing && vars.deltaQ_DW !== nothing
        for tf in s.tech_flexible
            dQh = get(p.dQ_hourly, (tf, activity), 0.0)
            abs(dQh) < _IJ_COEF_EPS_BAL && continue
            for h in s.hours
                add_to_expression!(expr, dQh, vars.deltaQ_UP[h, tf, ps])
                add_to_expression!(expr, dQh, vars.deltaQ_DW[h, tf, ps])
                has_terms = true
            end
        end
    end

    duCHP_TS = vars.deltaU_CHP_TS
    dpCHP_TS = vars.deltaP_CHP_TS
    if duCHP_TS !== nothing || dpCHP_TS !== nothing
        for tk in s.tech_hourlyCHPflex
            ab = get(p.activity_balances, (tk, activity, ps), 0.0)
            dPe = get(p.dP_electricity, (tk, activity), 0.0)
            dPh = get(p.dP_heat, (tk, activity), 0.0)
            eta = get(p.CHP_eta, tk, 0.0)
            eps = max(get(p.CHP_eps, (tk, ps), 0.0), 0.01)
            dp_coef = dPe - (eta / eps) * dPh
            for hc in s.hours_cluster
                w = get(p.clusterHourWeight, hc, 0.0)
                abs(w) < _IJ_COEF_EPS_BAL && continue
                if duCHP_TS !== nothing && abs(ab) >= _IJ_COEF_EPS_BAL
                    add_to_expression!(expr, w * ab, duCHP_TS[hc, tk, ps])
                    has_terms = true
                end
                if dpCHP_TS !== nothing && abs(dp_coef) >= _IJ_COEF_EPS_BAL
                    add_to_expression!(expr, w * dp_coef, dpCHP_TS[hc, tk, ps])
                    has_terms = true
                end
            end
        end
    elseif vars.deltaU_CHP !== nothing || vars.deltaP_CHP !== nothing
        for tk in s.tech_hourlyCHPflex
            ab = get(p.activity_balances, (tk, activity, ps), 0.0)
            dPe = get(p.dP_electricity, (tk, activity), 0.0)
            dPh = get(p.dP_heat, (tk, activity), 0.0)
            eta = get(p.CHP_eta, tk, 0.0)
            eps = max(get(p.CHP_eps, (tk, ps), 0.0), 0.01)
            dp_coef = dPe - (eta / eps) * dPh
            for h in s.hours
                if vars.deltaU_CHP !== nothing && abs(ab) >= _IJ_COEF_EPS_BAL
                    add_to_expression!(expr, ab, vars.deltaU_CHP[h, tk, ps])
                    has_terms = true
                end
                if vars.deltaP_CHP !== nothing && abs(dp_coef) >= _IJ_COEF_EPS_BAL
                    add_to_expression!(expr, dp_coef, vars.deltaP_CHP[h, tk, ps])
                    has_terms = true
                end
            end
        end
    end

    dS_TS = vars.deltaS_shed_TS
    if dS_TS !== nothing
        for tsh in s.tech_shedding
            ab = get(p.activity_balances, (tsh, activity, ps), 0.0)
            abs(ab) < _IJ_COEF_EPS_BAL && continue
            for hc in s.hours_cluster
                w = get(p.clusterHourWeight, hc, 0.0)
                abs(w) < _IJ_COEF_EPS_BAL && continue
                add_to_expression!(expr, w * ab, dS_TS[hc, tsh, ps])
                has_terms = true
            end
        end
    elseif vars.deltaS_shed !== nothing
        for tsh in s.tech_shedding
            ab = get(p.activity_balances, (tsh, activity, ps), 0.0)
            abs(ab) < _IJ_COEF_EPS_BAL && continue
            for h in s.hours
                add_to_expression!(expr, ab, vars.deltaS_shed[h, tsh, ps])
                has_terms = true
            end
        end
    end

    dW_TS = vars.deltaW_UP_TS
    if dW_TS !== nothing
        for tw in s.tech_reservoir
            dWh = get(p.dW_hourly, (tw, activity), 0.0)
            abs(dWh) < _IJ_COEF_EPS_BAL && continue
            for hc in s.hours_cluster
                w = get(p.clusterHourWeight, hc, 0.0)
                abs(w) < _IJ_COEF_EPS_BAL && continue
                add_to_expression!(expr, -w * dWh, dW_TS[hc, tw, ps])
                has_terms = true
            end
        end
    elseif vars.deltaW_UP !== nothing
        for tw in s.tech_reservoir
            dWh = get(p.dW_hourly, (tw, activity), 0.0)
            abs(dWh) < _IJ_COEF_EPS_BAL && continue
            for h in s.hours
                add_to_expression!(expr, -dWh, vars.deltaW_UP[h, tw, ps])
                has_terms = true
            end
        end
    end

    return has_terms
end

function _reservoir_capacity_adder(vars::AnnualVars, s::ModelSets, p::ModelParams,
                                   t::Symbol, ps::Int)
    t in Set(s.tech_reservoir) || return AffExpr(0.0), false
    loss = get(p.phs_Losses, t, 0.0)
    factor = 1.0 - loss
    expr = AffExpr(0.0)
    has_term = false
    if vars.deltaW_UP_TS !== nothing
        for hc in s.hours_cluster
            w = get(p.clusterHourWeight, hc, 0.0)
            w == 0.0 && continue
            add_to_expression!(expr, factor * w, vars.deltaW_UP_TS[hc, t, ps])
            has_term = true
        end
    elseif vars.deltaW_UP !== nothing
        for h in s.hours
            add_to_expression!(expr, factor, vars.deltaW_UP[h, t, ps])
            has_term = true
        end
    end
    return expr, has_term
end

# All-techs version (used by emissionsFix where t ranges over ALL techs)
function _bal_terms_all_techs(s::ModelSets, p::ModelParams, a::Symbol, per::Int)
    out = Tuple{Symbol,Float64}[]
    tb_set = Set(s.tech_balancers)
    for tb in s.tech_balancers
        coef = get(p.activity_balances, (tb, a, per), 0.0)
        coef == 0.0 || push!(out, (tb, coef))
    end
    return out
end

# activities_emissionFix = {a | activityType_act(a) == 'Emission'}
function _activities_emissionFix(s::ModelSets, p::ModelParams)
    ACTIVITY_EMISSION = :Emission
    return [a for a in s.activities if get(p.activityType_act, a, Symbol("")) == ACTIVITY_EMISSION]
end

# -----------------------------------------------------------------------------
# Emission-target constraints
# -----------------------------------------------------------------------------
function _add_emission_targets!(m::JuMP.Model, vars::AnnualVars, md::ModelData,
                                bal_by_act::Dict{Tuple{Symbol,Int},Vector{Tuple{Symbol,Float64}}})
    s   = md.sets
    p   = md.params
    pss = s.periods_solve
    tu  = vars.tech_use
    tb_set = Set(s.tech_balancers)
    nl_node = :NL
    eu_node = :EU
    active_constraint_set = something(tryparse(Int, p.ActiveConstraintSet), 0)
    include_bunker = active_constraint_set == 2 || active_constraint_set >= 3
    include_feedstock = active_constraint_set == 1 || active_constraint_set >= 3

    # ------------------------------------------------------------------------
    # 2026-06-15: IESA-Opt 1.0 BaseET_BFS group includes only:
    #   balance_activities_EmissionTargetAir / Bunker / FS.
    # The constraints EmissionTargetAll, EmissionTarget_inclScope3andFuelEx,
    # emission_targetCum and storageCO2_cumulative are NOT in BaseET_BFS
    # (and explicitly commented out in the IESA-Opt 1.0 BaseConstraints set).
    # Make them opt-in via env vars to mirror the IESA-Opt 1.0 reference run.
    # ------------------------------------------------------------------------
    enable_em(name) = get(ENV, "IESA_EMISSION_ENABLE_$(name)", "0") == "1"

    # Precompute node → list of tech_balancers
    techs_at_node = Dict{Symbol,Vector{Symbol}}()
    for tb in s.tech_balancers
        n = get(p.nodePer_techBal, tb, Symbol(""))
        n == Symbol("") && continue
        push!(get!(() -> Symbol[], techs_at_node, n), tb)
    end

    # ─── balance_activities_EmissionTargetAir (NL only, IESA-Opt 1.0 line ~2498) ───
    # sum[(tb,act), (nodePer_techBal(tb)=NL) * tu(tb,ps) * activity_balances(tb,act,ps)]
    #   where act ∈ activities_target (NOT activities_emission!)
    # + CHP delta terms (act, atf) and shed delta terms
    # + 3 EU-side synfuel-export tech adjustments (atf ∈ activities_target_FeedStocks)
    # + 3 NL-side bunker tech adjustments         (atb ∈ activities_target_Bunkers)
    # <= emissionTargetAir(NL, ps)
    duCHP_TS = vars.deltaU_CHP_TS
    dShed_TS = vars.deltaS_shed_TS
    duCHP_FH = vars.deltaU_CHP        # full-hourly fallback (rare)
    dShed_FH = vars.deltaS_shed

    function _add_delta_terms!(expr, act_set, n_filter, ps)
        # Add CHP delta terms: sum[(tk,a), (nodePer_techBal(tk)=n_filter) *
        #   sum[hc, clusterHourWeight(hc) * deltaU_CHP_TS(hc,tk,ps)] * activity_balances(tk,a,ps)]
        if duCHP_TS !== nothing
            for tk in s.tech_hourlyCHPflex
                get(p.nodePer_techBal, tk, Symbol("")) == n_filter || continue
                for a in act_set
                    coef = get(p.activity_balances, (tk, a, ps), 0.0)
                    coef == 0.0 && continue
                    for hc in s.hours_cluster
                        w = get(p.clusterHourWeight, hc, 0.0)
                        w == 0.0 && continue
                        add_to_expression!(expr, coef * w, duCHP_TS[hc, tk, ps])
                    end
                end
            end
        elseif duCHP_FH !== nothing
            for tk in s.tech_hourlyCHPflex
                get(p.nodePer_techBal, tk, Symbol("")) == n_filter || continue
                for a in act_set
                    coef = get(p.activity_balances, (tk, a, ps), 0.0)
                    coef == 0.0 && continue
                    for h in s.hours
                        add_to_expression!(expr, coef, duCHP_FH[h, tk, ps])
                    end
                end
            end
        end
        # Shed delta terms
        if dShed_TS !== nothing
            for ts_ in s.tech_shedding
                get(p.nodePer_techBal, ts_, Symbol("")) == n_filter || continue
                for a in act_set
                    coef = get(p.activity_balances, (ts_, a, ps), 0.0)
                    coef == 0.0 && continue
                    for hc in s.hours_cluster
                        w = get(p.clusterHourWeight, hc, 0.0)
                        w == 0.0 && continue
                        add_to_expression!(expr, coef * w, dShed_TS[hc, ts_, ps])
                    end
                end
            end
        elseif dShed_FH !== nothing
            for ts_ in s.tech_shedding
                get(p.nodePer_techBal, ts_, Symbol("")) == n_filter || continue
                for a in act_set
                    coef = get(p.activity_balances, (ts_, a, ps), 0.0)
                    coef == 0.0 && continue
                    for h in s.hours
                        add_to_expression!(expr, coef, dShed_FH[h, ts_, ps])
                    end
                end
            end
        end
        return nothing
    end

    for ((n, ps), cap) in p.emissionTargetAir
        n == nl_node || continue
        ps in pss || continue
        techs_nl = get(techs_at_node, nl_node, Symbol[])
        techs_eu = get(techs_at_node, eu_node, Symbol[])
        isempty(techs_nl) && isempty(techs_eu) && continue

        expr = AffExpr(0.0)
        # Main NL-node body: techs at NL × activities_target (CO2 Air ETS / n-ETS)
        for tb in techs_nl
            for ac in s.activities_target
                coef = get(p.activity_balances, (tb, ac, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end
        # CHP/shed deltas for NL-node × activities_target
        _add_delta_terms!(expr, s.activities_target, nl_node, ps)
        # EU-side synfuel-export adjustments (atf ∈ activities_target_FeedStocks)
        for eu_tech_name in (:OPE01_03, :OPE02_03, :OPE03_03)
            eu_tech_name in tb_set || continue
            get(p.nodePer_techBal, eu_tech_name, Symbol("")) == eu_node || continue
            for atf in s.activities_target_FeedStocks
                coef = get(p.activity_balances, (eu_tech_name, atf, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[eu_tech_name, ps])
            end
        end
        # NL-side bunker-tech adjustments (atb ∈ activities_target_Bunkers)
        for nl_tech_name in (:TNB01_05, :TNB01_08, :TAI01_03)
            nl_tech_name in tb_set || continue
            get(p.nodePer_techBal, nl_tech_name, Symbol("")) == nl_node || continue
            for atb in s.activities_target_Bunkers
                coef = get(p.activity_balances, (nl_tech_name, atb, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[nl_tech_name, ps])
            end
        end

        @constraint(m, expr <= cap, base_name = "emTargetAir[$(n),$(ps)]")
    end

    # ─── balance_activities_EmissionTargetBunker (IESA-Opt 1.0 line ~2534) ───
    # sum[(tb,atb), (nodePer_techBal(tb)=n) * tu(tb,ps) * activity_balances(tb,atb,ps)]
    #   where atb ∈ activities_target_Bunkers
    # <= emissionTarget_bunkers(n,ps)
    if include_bunker
    for ps in pss
        nodes = Set{Symbol}(n for ((n, p_cap), _) in p.emissionTargetBunker if p_cap == ps)
        for (n, techs_n) in techs_at_node
            any(tb -> any(atb -> get(p.activity_balances, (tb, atb, ps), 0.0) != 0.0,
                          s.activities_target_Bunkers), techs_n) && push!(nodes, n)
        end

        for n in sort!(collect(nodes))
            techs_n = get(techs_at_node, n, Symbol[])
            expr = AffExpr(0.0)
            for tb in techs_n
                for atb in s.activities_target_Bunkers
                    coef = get(p.activity_balances, (tb, atb, ps), 0.0)
                    coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
                end
            end
            cap = get(p.emissionTargetBunker, (n, ps), 0.0)
            @constraint(m, expr <= cap, base_name = "emTargetBunker[$(n),$(ps)]")
        end
    end
    end

    # ─── balance_activities_EmissionTargetFS (feedstock, IESA-Opt 1.0 line ~2542) ───
    # sum[(tb,atf), tu * activity_balances(tb,atf,ps)] + CHP/shed deltas
    #   where atf ∈ activities_target_FeedStocks
    if include_feedstock
    for ps in pss
        nodes = Set{Symbol}(n for ((n, p_cap), _) in p.emissionTargetFS if p_cap == ps)
        for (n, techs_n) in techs_at_node
            any(tb -> any(atf -> get(p.activity_balances, (tb, atf, ps), 0.0) != 0.0,
                          s.activities_target_FeedStocks), techs_n) && push!(nodes, n)
        end

        for n in sort!(collect(nodes))
            techs_n = get(techs_at_node, n, Symbol[])
            expr = AffExpr(0.0)
            for tb in techs_n
                for atf in s.activities_target_FeedStocks
                    coef = get(p.activity_balances, (tb, atf, ps), 0.0)
                    coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
                end
            end
            _add_delta_terms!(expr, s.activities_target_FeedStocks, n, ps)
            cap = get(p.emissionTargetFS, (n, ps), 0.0)
            @constraint(m, expr <= cap, base_name = "emTargetFS[$(n),$(ps)]")
        end
    end
    end

    # ─── balance_activities_EmissionTargetAll (per-node total, IESA-Opt 1.0 line ~2552) ───
    # sum[(tb,act∈activities_target)] + CHP/shed deltas
    # + sum[(tb,atf∈activities_target_FeedStocks)] + CHP/shed deltas
    # + sum[(tb,atb∈activities_target_Bunkers)]   (no delta terms for bunkers)
    # NOTE 2026-06-15: NOT in BaseET_BFS — opt-in via IESA_EMISSION_ENABLE_ALL=1.
    if enable_em("ALL")
    for ((n, ps), cap) in p.emissionTargetAll
        ps in pss || continue
        techs_n = get(techs_at_node, n, Symbol[])
        isempty(techs_n) && continue

        expr = AffExpr(0.0)
        # 1) activities_target
        for tb in techs_n
            for ac in s.activities_target
                coef = get(p.activity_balances, (tb, ac, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end
        _add_delta_terms!(expr, s.activities_target, n, ps)
        # 2) activities_target_FeedStocks
        for tb in techs_n
            for atf in s.activities_target_FeedStocks
                coef = get(p.activity_balances, (tb, atf, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end
        _add_delta_terms!(expr, s.activities_target_FeedStocks, n, ps)
        # 3) activities_target_Bunkers
        for tb in techs_n
            for atb in s.activities_target_Bunkers
                coef = get(p.activity_balances, (tb, atb, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end

        @constraint(m, expr <= cap, base_name = "emTargetAll[$(n),$(ps)]")
    end
    end  # end if enable_em("ALL")

    # ─── balance_activities_EmissionTarget_inclScope3andFuelEx (IESA-Opt 1.0 line ~2518) ───
    # NL-side: sum[(tb,act∈activities_target)] + CHP/shed +
    #          sum[(tb,atf∈activities_target_FeedStocks)] + CHP/shed
    # EU-side: sum[(tb,atf∈activities_target_FeedStocks)]   (fuel exports only)
    # NOTE 2026-06-15: commented out in BaseConstraints / BaseET_BFS — opt-in.
    if enable_em("INCLSCOPE3")
    for (ps, cap) in p.emissionTarget_inclScope3andFuelex
        ps in pss || continue
        techs_nl = get(techs_at_node, nl_node, Symbol[])
        techs_eu = get(techs_at_node, eu_node, Symbol[])

        expr = AffExpr(0.0)
        # NL-side: activities_target
        for tb in techs_nl
            for ac in s.activities_target
                coef = get(p.activity_balances, (tb, ac, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end
        _add_delta_terms!(expr, s.activities_target, nl_node, ps)
        # NL-side: activities_target_FeedStocks
        for tb in techs_nl
            for atf in s.activities_target_FeedStocks
                coef = get(p.activity_balances, (tb, atf, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end
        _add_delta_terms!(expr, s.activities_target_FeedStocks, nl_node, ps)
        # EU-side: activities_target_FeedStocks (fuel exports)
        for tb in techs_eu
            for atf in s.activities_target_FeedStocks
                coef = get(p.activity_balances, (tb, atf, ps), 0.0)
                coef == 0.0 || add_to_expression!(expr, coef, tu[tb, ps])
            end
        end

        @constraint(m, expr <= cap, base_name = "emTargetInclScope3[$(ps)]")
    end
    end  # end if enable_em("INCLSCOPE3")

    # ─── emission_targetCum (IESA-Opt 1.0 line ~2573): cumulative CO2 cap per node ───
    # sum[(tb,act,ps), tech_use(tb,ps) * (nodePer_act(act)=ni) * activity_balances(tb,act,ps)
    #                    * period_weight(ps) * transition_interval] <= emissionTarget_cum(ni)
    # NOTE 2026-06-15: commented out in BaseConstraints / BaseET_BFS — opt-in.
    if enable_em("CUM")
    ti_scalar = Float64(get(p.transition_interval, 0, 0))
    for (ni, cap) in p.emissionTarget_cum
        ti_scalar == 0.0 && break
        # Build per-(tb,act,ps) coefficient
        expr = AffExpr(0.0)
        for ((tb, a, per), coef) in p.activity_balances
            tb in tb_set || continue
            per in pss   || continue
            coef == 0.0  && continue
            get(p.nodePer_act, a, Symbol("")) == ni || continue
            w = get(p.period_weight, per, 0.0)
            w == 0.0 && continue
            add_to_expression!(expr, coef * w * ti_scalar, tu[tb, per])
        end
        @constraint(m, expr <= cap, base_name = "emTargetCum[$(ni)]")
    end
    end  # end if enable_em("CUM")

    # ─── storageCO2_cumulative (IESA-Opt 1.0 line ~2581) ───
    # sum[(tb,ps), tu(tb,ps) * (tech_subsector(tb)='CCUS Storage') * period_weight(ps) * transition_interval]
    #   <= cumulative_CO2storage(ni)
    # NOTE 2026-06-15: commented out in BaseConstraints / BaseET_BFS — opt-in.
    if enable_em("CO2STORAGECUM")
    ti_scalar2 = Float64(get(p.transition_interval, 0, 0))
    ccus_subsector = :var"CCUS Storage"
    for (ni, cap) in p.cumulative_CO2storage
        ti_scalar2 == 0.0 && break
        ccus_techs = [tb for tb in s.tech_balancers
                      if get(p.tech_subsector, tb, Symbol("")) == ccus_subsector]
        isempty(ccus_techs) && continue
        expr = AffExpr(0.0)
        for tb in ccus_techs, ps in pss
            w = get(p.period_weight, ps, 0.0)
            w == 0.0 && continue
            add_to_expression!(expr, w * ti_scalar2, tu[tb, ps])
        end
        @constraint(m, expr <= cap, base_name = "co2StorageCum[$(ni)]")
    end
    end  # end if enable_em("CO2STORAGECUM")

    return nothing
end
