# =============================================================================
# stock.jl — Tech-stock evolution + investment + decommissioning constraints
#
# Phase 2 (annual LP). Mirrors IESA-Opt 1.0:
#   - Variable techStock      Definition: cumulative recursion (lines ~2814)
#   - Variable decomStock     Definition: cumulative recursion (lines ~2801)
#   - Constraint actStock_constraints       (av, ps)
#   - Constraint techStockCap_contraints_mat (t_m, ps)
#   - Constraint minStock_constraints, maxStock_constraints (t, ps | <>na)
#   - Constraint actInv_constraints         (ar, ps)
#   - Constraint techInv_constraints        (tr, ps)
#   - Constraint retrofit_constraint        (it, jt, ps)
#   - Constraint eco_decom_constraint       (t, ps)
#   - Constraint eco_decom_max              (t, ps)
#   - Constraint limit_new_invest           (t, ps)
#   - Constraint linked_investments_XC      (t, it, ps | XC Trade matching)
#   - Constraint linked_Stock_XC            (t, it, ps | XC Trade matching)
#
# Phase 2 drops the `deltaS_shed*` (sum[hc, …] or sum[h, …]) terms — these are
# added by Phase 4 (`shedding.jl`).
# =============================================================================

"""
    _shed_sum_for_tech(vars, s, p, t, ps) -> AffExpr

Build the IESA-Opt 1.0 slack term
  - TS mode (vars.deltaS_shed_TS set):   sum_{hc} clusterHourWeight(hc) * deltaS_shed_TS[hc,t,ps]
  - FH mode (vars.deltaS_shed set):      sum_{h}                            deltaS_shed[h,t,ps]
  - Annual-only mode:                    0
Caller must ensure `t in s.tech_shedding` (shed vars are only defined for those).
"""
function _shed_sum_for_tech(vars::AnnualVars, s::ModelSets, p::ModelParams,
                             t::Symbol, ps::Int)
    e = AffExpr(0.0)
    if vars.deltaS_shed_TS !== nothing
        @inbounds for hc in s.hours_cluster
            w = get(p.clusterHourWeight, hc, 0.0)
            w == 0.0 && continue
            add_to_expression!(e, w, vars.deltaS_shed_TS[hc, t, ps])
        end
    elseif vars.deltaS_shed !== nothing
        @inbounds for h in s.hours
            add_to_expression!(e, 1.0, vars.deltaS_shed[h, t, ps])
        end
    end
    return e
end

"""
    add_stock_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData) -> Nothing

Add all stock + investment + decommissioning constraints to `m`. Idempotent:
caller is expected to pass a fresh `m` each time.
"""
function add_stock_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s   = md.sets
    p   = md.params
    pss = s.periods_solve
    tech = s.technologies

    ts  = vars.techStock
    ci  = vars.cap_investments
    ed  = vars.eco_decommisioning
    ds  = vars.decomStock
    rt  = vars.retrofitting
    tu  = vars.tech_use

    # -------------------------------------------------------------------------
    # decomStock recursion (IESA-Opt 1.0 Variable.Definition, lines ~2801-2812):
    #   decomStock(t, ps) = decomStock(t, ps-1)                 [0 if first]
    #                       + decom_plannedSel(t, ps)
    #                       + sum[pa, decomMat_NewInv(t, pa, ps) *
    #                                 (cap_investments(t, pa) + sum[it, retrofitting(it, t, pa)])]
    #                       + eco_decommisioning(t, ps)
    # `rt` is a sparse Dict{(it,jt,ps),VariableRef}; only iterate inbound retrofits.
    # -------------------------------------------------------------------------
    for t in tech, ps in pss
        prev_ps = _prev_period(pss, ps)
        prev_ds = prev_ps === nothing ? 0.0 : ds[t, prev_ps]
        planned = get(p.decom_plannedSel, (t, ps), 0.0)

        new_decom_expr = AffExpr(0.0)
        retro_in_t = get(p.retrofit_in_by_tech, t, Symbol[])
        for pa in pss
            coef = get(p.decomMat_NewInv, (t, pa, ps), 0.0)
            coef == 0.0 && continue
            add_to_expression!(new_decom_expr, coef, ci[t, pa])
            # retrofits in onto `t` at period pa (sparse over actual retrofit pairs)
            for it_ in retro_in_t
                add_to_expression!(new_decom_expr, coef, rt[(it_, t, pa)])
            end
        end

        @constraint(m, ds[t, ps] == prev_ds + planned + new_decom_expr + ed[t, ps],
                    base_name = "decomStock_def[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # techStock recursion (IESA-Opt 1.0 Variable.Definition, lines ~2814-2832):
    #   if ps == first:
    #       techStock(t, ps) = techStock_exist(t)
    #                          + cap_investments(t, ps)
    #                          + sum[it, retrofitting(it, t, ps)]
    #                          - sum[it, retrofitting(t, it, ps)]
    #                          - (decomStock(t, ps) - decomStock(t, ps-1)=0)
    #   else:
    #       techStock(t, ps) = techStock(t, ps-1)
    #                          + cap_investments(t, ps)
    #                          + sum[it, retrofitting(it, t, ps)]
    #                          - sum[it, retrofitting(t, it, ps)]
    #                          - (decomStock(t, ps) - decomStock(t, ps-1))
    # -------------------------------------------------------------------------
    for t in tech, ps in pss
        prev_ps = _prev_period(pss, ps)
        prev_stock = prev_ps === nothing ? get(p.techStock_exist, t, 0.0) : ts[t, prev_ps]
        prev_ds_val = prev_ps === nothing ? 0.0 : ds[t, prev_ps]

        retro_in  = AffExpr(0.0)
        retro_out = AffExpr(0.0)
        for it_ in get(p.retrofit_in_by_tech, t, Symbol[])
            add_to_expression!(retro_in,  1.0, rt[(it_, t, ps)])
        end
        for jt_ in get(p.retrofit_out_by_tech, t, Symbol[])
            add_to_expression!(retro_out, 1.0, rt[(t, jt_, ps)])
        end

        @constraint(m,
            ts[t, ps] == prev_stock + ci[t, ps] + retro_in - retro_out - (ds[t, ps] - prev_ds_val),
            base_name = "techStock_def[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # actStock_constraints (IESA-Opt 1.0 line ~2836): for driver activities
    #   sum[t | activityPer_tech(t) = av,
    #       techStock(t, ps) * cap2act(t)
    #     + (if use_clustering then sum[hc, clusterHourWeight(hc)*deltaS_shed_TS(hc,t,ps)]
    #        else sum[h, deltaS_shed(h,t,ps)] endif)
    #   ] = activities_netVolumes(av, ps)
    # Shedding acts as a slack — without it, IESA-Opt 1.0-derived data with
    # techStock_max < activities_netVolumes/cap2act is infeasible.
    # -------------------------------------------------------------------------
    shed_set = Set(s.tech_shedding)
    for av in s.activities_driver, ps in pss
        techs_for_av = [t for t in tech if get(p.activityPer_tech, t, Symbol("")) == av]
        isempty(techs_for_av) && continue
        rhs = get(p.activities_netVolumes, (av, ps), 0.0)
        @constraint(m,
            sum(get(p.cap2act, t, 0.0) * ts[t, ps] for t in techs_for_av) +
            sum(_shed_sum_for_tech(vars, s, p, t, ps) for t in techs_for_av if t in shed_set; init = AffExpr(0.0))
                == rhs,
            base_name = "actStock[$(av),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # techStockCap_contraints_mat (IESA-Opt 1.0 line ~2841): for material-conversion techs
    #   techStock(t_m, ps) * cap2act(t_m)
    #     + (if use_clustering then sum[hc, clusterHourWeight(hc)*deltaS_shed_TS(hc,t_m,ps)]
    #        else sum[h, deltaS_shed(h,t_m,ps)] endif)
    #   >= tech_use(t_m, ps)
    # -------------------------------------------------------------------------
    for t_m in s.tech_materialConversion, ps in pss
        c2a = get(p.cap2act, t_m, 0.0)
        c2a == 0.0 && continue
        t_m in s.tech_balancers || continue   # tu only defined on tech_balancers
        shed_term = t_m in shed_set ? _shed_sum_for_tech(vars, s, p, t_m, ps) : AffExpr(0.0)
        @constraint(m,
            c2a * ts[t_m, ps] + shed_term >= tu[t_m, ps],
            base_name = "techStockCap_mat[$(t_m),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # minStock / maxStock (IESA-Opt 1.0 lines ~2845-2851): only where <>na
    # -------------------------------------------------------------------------
    for ((t, ps), v) in p.techStock_min
        ps in pss || continue
        t in tech || continue
        @constraint(m, ts[t, ps] >= v, base_name = "minStock[$(t),$(ps)]")
    end
    for ((t, ps), v) in p.techStock_max
        ps in pss || continue
        t in tech || continue
        @constraint(m, ts[t, ps] <= v, base_name = "maxStock[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # actInv_constraints (IESA-Opt 1.0 line ~2853): activity-level investment caps
    #   sum[t | activityPer_tech(t) = ar,
    #       cap_investments(t, ps) * cap2act(t)
    #     + (if use_clustering then sum[hc, clusterHourWeight(hc)*deltaS_shed_TS(hc,t,ps)]
    #        else sum[h, deltaS_shed(h,t,ps)] endif)
    #   ] <= activities_netVolumes(ar, ps) * actChange_max(ar) * period_span(ps) / 100
    # -------------------------------------------------------------------------
    for ar in s.activities_driver, ps in pss
        amax = get(p.actChange_max, ar, 0.0)
        amax > 0 || continue
        techs_for_ar = [t for t in tech if get(p.activityPer_tech, t, Symbol("")) == ar]
        isempty(techs_for_ar) && continue
        rhs = get(p.activities_netVolumes, (ar, ps), 0.0) * amax * get(p.period_span, ps, 1) / 100
        @constraint(m,
            sum(get(p.cap2act, t, 0.0) * ci[t, ps] for t in techs_for_ar) +
            sum(_shed_sum_for_tech(vars, s, p, t, ps) for t in techs_for_ar if t in shed_set; init = AffExpr(0.0))
                <= rhs,
            base_name = "actInv[$(ar),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # techInv_constraints (IESA-Opt 1.0 line ~2857): tech-level investment caps
    #   cap_investments(tr, ps) <= techChange_max(tr) * period_span(ps)
    # `tr` is `techInv_ramp = {t | techChange_max(t) > 0}`.
    # -------------------------------------------------------------------------
    for (tr, tmax) in p.techChange_max
        tmax > 0 || continue
        tr in tech || continue
        for ps in pss
            @constraint(m,
                ci[tr, ps] <= tmax * get(p.period_span, ps, 1),
                base_name = "techInv[$(tr),$(ps)]")
        end
    end

    # -------------------------------------------------------------------------
    # retrofit_constraint (IESA-Opt 1.0 line ~2861):
    #   if first(ps): retrofit_relations(it, jt) * techStock_exist(it) - retrofitting(it, jt, ps) >= 0
    #   else:         retrofit_relations(it, jt) * techStock(it, ps-1) - retrofitting(it, jt, ps) >= 0
    #
    # Default (sparse) mode: `p.retrofit_pairs` already filters to active pairs
    # (rel=1.0 for every entry), so we drop the `rel` multiplier.
    # Dense (AIMMS-comparable) mode: `p.retrofit_pairs` enumerates the full
    # technologies×technologies cross-product. We look up the relation factor;
    # for inactive pairs (rel=0) the constraint reduces to `-rt >= 0` which
    # pins the retrofitting var to 0 (matching AIMMS).
    # -------------------------------------------------------------------------
    for (it_, jt_) in p.retrofit_pairs, ps in pss
        # rel = retrofit_relations(it,jt). Missing key => implicitly inactive
        # (the workbook only stores entries that exist in the Retrofitting sheet,
        # whether true or false). In SPARSE mode every pair in retrofit_pairs has
        # an explicit `true` entry so rel is always 1.0; in DENSE mode the
        # cross-product enumerates pairs that are NOT in the dict, and those
        # MUST be treated as 0.0 (otherwise the model gets free retrofitting).
        rel = get(p.retrofit_relations, (it_, jt_), false) ? 1.0 : 0.0
        if rel == 0.0
            @constraint(m, -rt[(it_, jt_, ps)] >= 0,
                        base_name = "retrofit[$(it_),$(jt_),$(ps)]")
        else
            prev_ps = _prev_period(pss, ps)
            ub = if prev_ps === nothing
                get(p.techStock_exist, it_, 0.0)
            else
                ts[it_, prev_ps]
            end
            @constraint(m, ub - rt[(it_, jt_, ps)] >= 0,
                        base_name = "retrofit[$(it_),$(jt_),$(ps)]")
        end
    end

    # -------------------------------------------------------------------------
    # eco_decom_constraint (IESA-Opt 1.0 line ~2872):
    #   no_eco_decom(t, ps) * eco_decommisioning(t, ps) = 0
    # i.e. when no_eco_decom binary flag is 1, force eco = 0
    # -------------------------------------------------------------------------
    for ((t, ps), v) in p.no_eco_decom
        v && (ps in pss) && (t in tech) || continue
        @constraint(m, ed[t, ps] == 0, base_name = "noEcoDecom[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # eco_decom_max (IESA-Opt 1.0 line ~2877):
    #   eco_decommisioning(t, ps) <= max(0, techStock_exist(t) - decom_plannedSel(t, ps))
    #   with subnormal snap to 0 when |diff| < 1e-9.
    # -------------------------------------------------------------------------
    for t in tech, ps in pss
        exist   = get(p.techStock_exist, t, 0.0)
        planned = get(p.decom_plannedSel, (t, ps), 0.0)
        diff    = exist - planned
        rhs     = (abs(diff) < 1e-9) ? 0.0 : diff
        @constraint(m, ed[t, ps] <= max(0.0, rhs), base_name = "ecoDecomMax[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # limit_new_invest (IESA-Opt 1.0 line ~2894): when no_new_invest binary is 1, force ci=0
    # -------------------------------------------------------------------------
    for ((t, ps), v) in p.no_new_invest
        v && (ps in pss) && (t in tech) || continue
        @constraint(m, ci[t, ps] == 0, base_name = "noNewInv[$(t),$(ps)]")
    end

    # -------------------------------------------------------------------------
    # linked_investments_XC + linked_Stock_XC (IESA-Opt 1.0 lines ~2901-2918):
    #   For (t, it) where tech_category(t)='XC Trade' AND
    #                     tech_sector(t)=tech_subsector(it) AND
    #                     tech_sector(it)=tech_subsector(t):
    #     cap_investments(t, ps) = cap_investments(it, ps)
    #     techStock(t, ps)       = techStock(it, ps)
    # -------------------------------------------------------------------------
    xc_category = :var"XC Trade"
    xc_pairs = Tuple{Symbol,Symbol}[]
    for t in tech
        get(p.tech_category, t, Symbol("")) == xc_category || continue
        sec_t   = get(p.tech_sector, t, Symbol(""))
        subs_t  = get(p.tech_subsector, t, Symbol(""))
        for it_ in tech
            it_ == t && continue
            sec_it  = get(p.tech_sector, it_, Symbol(""))
            subs_it = get(p.tech_subsector, it_, Symbol(""))
            # IESA-Opt 1.0 multiplicative: sec_t==subs_it AND sec_it==subs_t
            if sec_t != Symbol("") && sec_t == subs_it &&
               sec_it != Symbol("") && sec_it == subs_t
                push!(xc_pairs, (t, it_))
            end
        end
    end
    for (t, it_) in xc_pairs
        for ps in pss
            @constraint(m, ci[t, ps] == ci[it_, ps],
                        base_name = "linkedInvXC[$(t),$(it_),$(ps)]")
            @constraint(m, ts[t, ps] == ts[it_, ps],
                        base_name = "linkedStockXC[$(t),$(it_),$(ps)]")
        end
    end

    return nothing
end
