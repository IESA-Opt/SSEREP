# =============================================================================
# objective.jl — totalCosts objective (Phase 2: annual portion only)
#
# Mirrors IESA-Opt 1.0 `Variable totalCosts` Definition (lines ~5543-5594) as direct
# solver objective coefficients. IESA-Opt 1.0 exports `totalCosts` as output, but its
# Gurobi model does not include a scalar totalCosts column plus definition row.
#
# Phase 2 components:
#   sum[ps, social_discount_factor(ps) * (
#       + sum[(t,jp), InvMat_lifeTime(t,jp,ps) * cap_investments(t,jp) * inv_cost(t,jp) * CRF(t)]
#       + sum[(t,jp), InvMat_lifeTime(t,jp,ps) * sum[it, retrofitting(it,t,ps) * CRF(t) *
#                                                  (retrofit_cost(it,t,ps) + p_epsilon)]]
#       - sum[t, (eco_decommisioning(t,ps) - eco_decommisioning(t,ps-1)) *
#               Salvage_value(t) * inv_cost(t,ps) * CRF(t)]
#       + sum[t, techStock(t,ps) * fom_cost(t,ps)]
#       + sum[t, tech_use(t,ps) * (vom_cost(t,ps) + p_epsilon)]
#   )]
#
# DROPPED in Phase 2 (variables not yet declared):
#   - tech_useHourly[_TS]   × interconnectedHourly_prices
#   - deltaU_CHP[_TS]       × vom_cost
#   - deltaS_shed[_TS]      × shed_penalty + vom_cost
#   - p_epsilon × (deltaQ_DW − deltaQ_UP)
#   - p_epsilon × deltaS_shed
# =============================================================================

"""
    add_objective!(m::JuMP.Model, vars::AnnualVars, md::ModelData) -> Nothing

Sets `@objective(m, Min, totalCosts_annual)`. Returns nothing.
"""
function add_objective!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s   = md.sets
    p   = md.params
    pss = s.periods_solve

    tech = s.technologies
    tb   = s.tech_balancers
    ts   = vars.techStock
    ci   = vars.cap_investments
    ed   = vars.eco_decommisioning
    rt   = vars.retrofitting
    tu   = vars.tech_use

    obj = AffExpr(0.0)
    p_eps = p.p_epsilon

    lifetime_weight = Dict{Tuple{Symbol,Int},Float64}()
    for ((t_life, _jp, ps_life), w) in p.InvMat_lifeTime
        lifetime_weight[(t_life, ps_life)] = get(lifetime_weight, (t_life, ps_life), 0.0) + w
    end

    for ps in pss
        sdf = get(p.social_discount_factor, ps, 1.0)
        sdf == 0.0 && continue

        # ── CAPEX: sum[(t,jp), InvMat_lifeTime(t,jp,ps) * cap_investments(t,jp) *
        #                       inv_cost(t,jp) * CRF(t)]
        # Here jp indexes the cap_investments variable itself, so only active
        # variable periods are added as columns. This is distinct from pure
        # lifetime-weight sums over jp, where IESA-Opt 1.0 uses the full period set.
        for ((t_inv, jp, ps_in), w) in p.InvMat_lifeTime
            ps_in == ps || continue
            t_inv in tech || continue
            jp in pss || continue
            cost = get(p.inv_cost, (t_inv, jp), 0.0)
            cost == 0.0 && continue
            crf = get(p.CRF, t_inv, 0.0)
            crf == 0.0 && continue
            coef = sdf * w * cost * crf
            coef == 0.0 || add_to_expression!(obj, coef, ci[t_inv, jp])
        end

        # ── Retrofit CAPEX: sum[(t,jp), InvMat_lifeTime(t,jp,ps) *
        #     sum[it, retrofitting(it,t,ps) * CRF(t) * (retrofit_cost(it,t,ps) + p_eps)]]
        # Note: IESA-Opt 1.0 uses retrofitting(it,t,ps) (current period ps), not jp.
        # `rt` is now sparse over `p.retrofit_pairs` (only entries where
        # `retrofit_relations(it,t)==true`); skip pairs with no entry.
        for (it_, t_) in p.retrofit_pairs
            crf = get(p.CRF, t_, 0.0)
            crf == 0.0 && continue
            rc = get(p.retrofit_cost, (it_, t_, ps), 0.0)
            (rc + p_eps) == 0.0 && continue
            # IESA-Opt 1.0 sums InvMat_lifeTime(t, jp, ps) over the full period set;
            # jp is not the retrofit variable period in this term.
            w_lifetime = get(lifetime_weight, (t_, ps), 0.0)
            w_lifetime == 0.0 && continue
            coef = sdf * w_lifetime * crf * (rc + p_eps)
            coef == 0.0 || add_to_expression!(obj, coef, rt[(it_, t_, ps)])
        end

        # ── Salvage value (negative cost):
        # − sum[t, (ed(t,ps) − ed(t,ps-1)) * Salvage_value(t) * inv_cost(t,ps) * CRF(t)]
        prev_ps = _prev_period(pss, ps)
        for t in tech
            sv = get(p.Salvage_value, t, 0.0)
            sv == 0.0 && continue
            ic = get(p.inv_cost, (t, ps), 0.0)
            ic == 0.0 && continue
            crf = get(p.CRF, t, 0.0)
            crf == 0.0 && continue
            coef = -sdf * sv * ic * crf
            add_to_expression!(obj, coef, ed[t, ps])
            if prev_ps !== nothing
                add_to_expression!(obj, -coef, ed[t, prev_ps])
            end
        end

        # ── FOM: sum[t, techStock(t,ps) * fom_cost(t,ps)]
        for t in tech
            fc = get(p.fom_cost, (t, ps), 0.0)
            fc == 0.0 && continue
            add_to_expression!(obj, sdf * fc, ts[t, ps])
        end

        # ── VOM: sum[t, tech_use(t,ps) * (vom_cost(t,ps) + p_eps)]
        # IESA-Opt 1.0 uses `t` (full technologies set), but tech_use is only defined for tb.
        for t in tb
            vc = get(p.vom_cost, (t, ps), 0.0)
            (vc + p_eps) == 0.0 && continue
            add_to_expression!(obj, sdf * (vc + p_eps), tu[t, ps])
        end

        # ============================================================
        # Phase 3 — hourly objective terms (only if hourly vars exist).
        # IESA-Opt 1.0 source lines 5566-5577 (FH branch of totalCosts).
        # ============================================================
        if vars.tech_useHourly !== nothing
            tuh = vars.tech_useHourly
            ainEU = :var"Electricity EU"
            for thh in s.tech_hourlyDispatch
                get(p.tech_category, thh, Symbol("")) == :var"XC Trade" || continue
                is_imp = get(p.tech_subsector, thh, Symbol("")) == :var"Power EU"
                is_exp = get(p.tech_sector,    thh, Symbol("")) == :var"Power EU"
                (is_imp || is_exp) || continue
                sign = is_imp ? +1.0 : -1.0
                for h in s.hours
                    price = get(p.interconnectedHourly_prices, (h, ainEU, ps), 0.0)
                    price == 0.0 && continue
                    add_to_expression!(obj, sdf * sign * price, tuh[h, thh, ps])
                end
            end
        end

        # CHP variable cost on |deltaU_CHP|: IESA-Opt 1.0 uses vom_cost on deltaU_CHP directly.
        if vars.deltaU_CHP !== nothing
            duCHP = vars.deltaU_CHP
            for t in s.tech_hourlyCHPflex
                vc = get(p.vom_cost, (t, ps), 0.0)
                vc == 0.0 && continue
                for h in s.hours
                    add_to_expression!(obj, sdf * vc, duCHP[h, t, ps])
                end
            end
        end

        # AIMMS totalCosts includes only deltaS_shed * vom_cost.
        if vars.deltaS_shed !== nothing
            dS = vars.deltaS_shed
            for tsh in s.tech_shedding
            coef = get(p.vom_cost, (tsh, ps), 0.0)
                coef == 0.0 && continue
                for h in s.hours
                    add_to_expression!(obj, sdf * coef, dS[h, tsh, ps])
                end
            end
            # Tiny p_eps tie-breaker to avoid degenerate alternative optima.
            for tsh in s.tech_shedding, h in s.hours
                add_to_expression!(obj, -sdf * p_eps, dS[h, tsh, ps])
            end
        end

        # Flex tie-breaker: +p_eps × (deltaQ_DW − deltaQ_UP)
        if vars.deltaQ_UP !== nothing && vars.deltaQ_DW !== nothing
            dqUP = vars.deltaQ_UP
            dqDW = vars.deltaQ_DW
            for t in s.tech_flexible, h in s.hours
                add_to_expression!(obj,  sdf * p_eps, dqDW[h, t, ps])
                add_to_expression!(obj, -sdf * p_eps, dqUP[h, t, ps])
            end
        end

        # ============================================================
        # Phase 5/6 — TS-mode hourly objective terms.
        # Each hourly term is weighted by clusterHourWeight(hc) to
        # represent the full year (IESA-Opt 1.0 lines 5566-5577 TS branch).
        # ============================================================
        if vars.tech_useHourly_TS !== nothing
            tuh = vars.tech_useHourly_TS
            ainEU = :var"Electricity EU"
            for thh in s.tech_hourlyDispatch
                get(p.tech_category, thh, Symbol("")) == :var"XC Trade" || continue
                is_imp = get(p.tech_subsector, thh, Symbol("")) == :var"Power EU"
                is_exp = get(p.tech_sector,    thh, Symbol("")) == :var"Power EU"
                (is_imp || is_exp) || continue
                sign = is_imp ? +1.0 : -1.0
                for hc in s.hours_cluster
                    w = get(p.clusterHourWeight, hc, 1.0)
                    price = get(p.interconnectedHourly_prices_cluster, (hc, ainEU, ps), 0.0)
                    price == 0.0 && continue
                    add_to_expression!(obj, sdf * sign * w * price, tuh[hc, thh, ps])
                end
            end
        end
        if vars.deltaU_CHP_TS !== nothing
            duCHP = vars.deltaU_CHP_TS
            for t in s.tech_hourlyCHPflex
                vc = get(p.vom_cost, (t, ps), 0.0)
                vc == 0.0 && continue
                for hc in s.hours_cluster
                    w = get(p.clusterHourWeight, hc, 1.0)
                    add_to_expression!(obj, sdf * w * vc, duCHP[hc, t, ps])
                end
            end
        end
        if vars.deltaS_shed_TS !== nothing
            dS = vars.deltaS_shed_TS
            for tsh in s.tech_shedding
                pen = get(p.shed_penalty, tsh, 0.0)
                vc  = get(p.vom_cost, (tsh, ps), 0.0)
                coef = vc - pen
                for hc in s.hours_cluster
                    w = get(p.clusterHourWeight, hc, 1.0)
                    coef != 0.0 && add_to_expression!(obj, sdf * w * coef, dS[hc, tsh, ps])
                    add_to_expression!(obj, -sdf * w * p_eps, dS[hc, tsh, ps])
                end
            end
        end
        if vars.deltaQ_UP_TS !== nothing && vars.deltaQ_DW_TS !== nothing
            dqUP = vars.deltaQ_UP_TS
            dqDW = vars.deltaQ_DW_TS
            for t in s.tech_flexible, hc in s.hours_cluster
                w = get(p.clusterHourWeight, hc, 1.0)
                add_to_expression!(obj,  sdf * w * p_eps, dqDW[hc, t, ps])
                add_to_expression!(obj, -sdf * w * p_eps, dqUP[hc, t, ps])
            end
        end
    end

    @objective(m, Min, obj)
    return nothing
end
