# =============================================================================
# policy.jl — Policy / regulatory / sectoral-target constraints (ps-active only)
#
# Implements only constraints that are active for `ps = 2050` in the IESA-Opt 1.0 source.
# Constraints active only at earlier periods (ps = 2030/2035/2040/2045) are NOT
# implemented here yet — extend `_period_filter` if you start solving them.
#
# IESA-Opt 1.0 source ranges:
#   - Adapt_* (Bunker Nav / Avi / Refineries):     lines 20620-20842
#   - CO2_credits_Avi / CO2_credits_Nav:           lines 20844-20920
#   - ReFuelEU_Aviation (eSAF, SAF, H2 credits):   lines 20910-21015
#   - FuelEU_Maritime (SectorTarget_emi_BunkerNav, H2_credits_Nav): lines 21015-21100
#   - MinLoad (Nuclear):                           line 20510
#
# All constraints use only `tech_use(t,ps)` and `deltaU_CHP(h,t,ps)` /
# `deltaS_shed(h,t,ps)` — the latter two are summed over hours so they reduce
# to scalar contributions per (t,ps).  In TS mode we substitute the TS variables
# and weight by repDayWeight_d (sum over rd of repDayWeight_d × Σ_ihc∈rd ...).
#
# Adapt_SectoTarget_emi*_BunkerNav uses `tech_activity` which we derive in
# `compute_tech_activity!` (parameters.jl).
# =============================================================================

const _IJ_POLICY_EPS = 1e-12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sum over hours of an hourly variable for tech t and period ps.
# Returns AffExpr.  In FH: Σ_h var[h,t,ps]. In TS: Σ_rd dayWeight(rd) × Σ_ihc∈rd var_TS[ihc,t,ps].
function _sum_hours_var(vars::AnnualVars, var_name::Symbol, t::Symbol, ps::Int,
                        md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    expr = AffExpr(0.0)
    if mode === :fh
        v_fh = getproperty(vars, var_name)
        v_fh === nothing && return expr
        # Check the second axis (techs) — DenseAxisArray supports `in`.
        tech_ax = axes(v_fh, 2)
        t in tech_ax || return expr
        for h in s.hours
            add_to_expression!(expr, 1.0, v_fh[h, t, ps])
        end
        return expr
    else
        var_TS = getproperty(vars, Symbol(string(var_name), "_TS"))
        var_TS === nothing && return expr
        tech_ax = axes(var_TS, 2)
        t in tech_ax || return expr
        for hc in s.hours_cluster
            rd = get(p.repDay_of_clusterHour, hc, 0)
            w = rd > 0 ? get(p.dayWeight, rd, 1.0) : 1.0
            add_to_expression!(expr, w, var_TS[hc, t, ps])
        end
        return expr
    end
end

# Build "sector flow expression": Σ_{(a,t)|filter} (tu(t,ps) + Σh ΔU_CHP + Σh ΔS_shed) × ab(t,a,ps)
function _sector_flow_expr(vars::AnnualVars, md::ModelData, ps::Int, mode::Symbol;
                            tech_filter::Function,
                            act_filter::Function = a -> true,
                            ab_filter::Function = ab -> true)
    s, p = md.sets, md.params
    tu = vars.tech_use
    expr = AffExpr(0.0)
    for t in s.tech_balancers
        tech_filter(t) || continue
        # Precompute the hourly sums for CHP / shedding only once per tech
        chp_sum = (t in s.tech_hourlyCHPflex) ?
                   _sum_hours_var(vars, :deltaU_CHP, t, ps, md, mode) : AffExpr(0.0)
        shed_sum = (t in s.tech_shedding) ?
                    _sum_hours_var(vars, :deltaS_shed, t, ps, md, mode) : AffExpr(0.0)
        for ((tt, a, pp), ab) in p.activity_balances
            tt == t || continue
            pp == ps || continue
            act_filter(a) || continue
            ab_filter(ab) || continue
            add_to_expression!(expr, ab, tu[t, ps])
            add_to_expression!(expr, ab, chp_sum)
            add_to_expression!(expr, ab, shed_sum)
        end
    end
    return expr
end

# Simpler tu×ab expression: Σ_{(a,t)|filter} sign × tu(t,ps) × ab(t,a,ps).
function _credit_expr(vars::AnnualVars, md::ModelData, ps::Int;
                      tech_filter::Function,
                      act_filter::Function = a -> true,
                      sign::Float64 = -1.0)
    s, p = md.sets, md.params
    tu = vars.tech_use
    expr = AffExpr(0.0)
    for t in s.tech_balancers
        tech_filter(t) || continue
        for ((tt, a, pp), ab) in p.activity_balances
            tt == t || continue
            pp == ps || continue
            act_filter(a) || continue
            add_to_expression!(expr, sign * ab, tu[t, ps])
        end
    end
    return expr
end

# ---------------------------------------------------------------------------
# Sectoral emission targets (Bunker Navigation/Aviation, ps=2050)
# IESA-Opt 1.0 lines 20664, 20759
# ---------------------------------------------------------------------------
function _add_Adapt_SectoTarget_BunkerNav_2050!(m::JuMP.Model, vars::AnnualVars,
                                                md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    2050 in s.periods_solve || return
    target_sector_kev = :Bunkerbrandstoffen
    target_activity   = :var"Bunker Navigation"
    rhs = 26.7

    # IESA-Opt 1.0: sum[(ac,t), ...] with ac in activities_emission
    lhs = _sector_flow_expr(vars, md, 2050, mode;
        tech_filter = t -> (get(p.tech_sector_kev, t, Symbol("")) == target_sector_kev) &&
                           (get(p.tech_activity, t, Symbol("")) == target_activity),
        act_filter  = a -> a in s.activities_emission)
    # IESA-Opt 1.0: sum[(tb,acr) | tb='TNB01_10', (-1) * tu * ab] with acr in activities_credits
    credit_tb = :TNB01_10
    cr = _credit_expr(vars, md, 2050; tech_filter = t -> t == credit_tb,
                      act_filter = a -> a in s.activities_credits, sign = -1.0)
    @constraint(m, lhs - cr <= rhs, base_name = "AdaptBunkNav50")
    return nothing
end

function _add_Adapt_SectoTarget_BunkerAvi_2050!(m::JuMP.Model, vars::AnnualVars,
                                                md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    2050 in s.periods_solve || return
    target_sector_kev = :Bunkerbrandstoffen
    target_activity   = :var"Bunker Aviation"
    rhs = 5.5

    lhs = _sector_flow_expr(vars, md, 2050, mode;
        tech_filter = t -> (get(p.tech_sector_kev, t, Symbol("")) == target_sector_kev) &&
                           (get(p.tech_activity, t, Symbol("")) == target_activity),
        act_filter  = a -> a in s.activities_emission)
    credit_tb = :TAI01_07
    cr = _credit_expr(vars, md, 2050; tech_filter = t -> t == credit_tb,
                      act_filter = a -> a in s.activities_credits, sign = -1.0)
    @constraint(m, lhs - cr <= rhs, base_name = "AdaptBunkAvi50")
    return nothing
end

# ---------------------------------------------------------------------------
# Refinery production cap (ps=2050) — IESA-Opt 1.0 line 20800
# ---------------------------------------------------------------------------
function _add_Adapt_RefinProd_2050!(m::JuMP.Model, vars::AnnualVars,
                                    md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    2050 in s.periods_solve || return
    rhs = 1202.0
    target_sector = :Refineries
    target_subsec = :var"Fossil Based"

    lhs = _sector_flow_expr(vars, md, 2050, mode;
        tech_filter = t -> (get(p.tech_sector, t, Symbol("")) == target_sector) &&
                           (get(p.tech_subsector, t, Symbol("")) == target_subsec),
        act_filter  = a -> a in s.activities_energy,
        ab_filter   = ab -> ab > _IJ_POLICY_EPS)
    @constraint(m, lhs <= rhs, base_name = "AdaptRefinProd50")
    return nothing
end

# ---------------------------------------------------------------------------
# CO2 credits — Aviation & Navigation (IESA-Opt 1.0 lines 20841, 20871)
# ---------------------------------------------------------------------------
function _add_CO2_credits_Avi!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        # LHS = (-1) × Σ_{tb='TAI01_07', acr in activities_credits} tu × ab
        lhs = _credit_expr(vars, md, ps; tech_filter = t -> t == :TAI01_07,
                           act_filter = a -> a in s.activities_credits, sign = -1.0)
        # Subtract (E-Kerosene + Syn Kerosene over Bunker Aviation techs) × 0.072
        ekero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"E-Kerosene", sign = -1.0)
        synkero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"Syn Kerosene", sign = -1.0)
        @constraint(m, lhs - 0.072 * (ekero + synkero) == 0.0,
                    base_name = "CO2credAvi[$ps]")
    end
    return nothing
end

function _add_CO2_credits_Nav!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        lhs = _credit_expr(vars, md, ps; tech_filter = t -> t == :TNB01_10,
                           act_filter = a -> a in s.activities_credits, sign = -1.0)
        meth = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Navigation",
            act_filter  = a -> a == :Methanol, sign = -1.0)
        emeth = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Navigation",
            act_filter  = a -> a == :var"E-Methanol", sign = -1.0)
        syndies = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Navigation",
            act_filter  = a -> a == :var"Syn Diesel", sign = -1.0)
        @constraint(m, lhs - (0.074 * meth + 0.074 * emeth + 0.073 * syndies) == 0.0,
                    base_name = "CO2credNav[$ps]")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# ReFuelEU Aviation — eSAF & SAF (IESA-Opt 1.0 lines 20925, 20948)
# ---------------------------------------------------------------------------
function _add_eSAF_Aviation!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        ps >= 2030 || continue
        target = get(p.ReFuelEU_Aviation_eSAF_target, ps, 0.0)
        target == 0.0 && continue
        # E-kerosene consumption in Bunker Aviation
        ekero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"E-Kerosene", sign = -1.0)
        # Total Bunker Aviation consumption (all activities)
        total = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            sign = -1.0)
        @constraint(m, ekero - target * total >= 0.0, base_name = "eSAF_Avi[$ps]")
    end
    return nothing
end

function _add_SAF_Aviation!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        ps >= 2030 || continue
        target = get(p.ReFuelEU_Aviation_SAF_target, ps, 0.0)
        target == 0.0 && continue
        ekero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"E-Kerosene", sign = -1.0)
        biokero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"Bio Kerosene", sign = -1.0)
        synkero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"Syn Kerosene", sign = -1.0)
        total = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            sign = -1.0)
        @constraint(m, ekero + biokero + synkero - target * total >= 0.0,
                    base_name = "SAF_Avi[$ps]")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# H2 credits — Aviation & Navigation (IESA-Opt 1.0 lines 20984, 21041)
# ---------------------------------------------------------------------------
function _add_H2_credits_Avi!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        # (-1) × Σ_{tb='TAI01_06', acr in activities_credits} tu × ab − 2.1 × (-1) × Σ E-Kero in Avi = 0
        lhs = _credit_expr(vars, md, ps; tech_filter = t -> t == :TAI01_06,
                           act_filter = a -> a in s.activities_credits, sign = -1.0)
        ekero = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Aviation",
            act_filter  = a -> a == :var"E-Kerosene", sign = -1.0)
        @constraint(m, lhs - 2.1 * ekero == 0.0, base_name = "H2credAvi[$ps]")
    end
    return nothing
end

function _add_H2_credits_Nav!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        lhs = _credit_expr(vars, md, ps; tech_filter = t -> t == :TNB01_09,
                           act_filter = a -> a in s.activities_credits, sign = -1.0)
        amm = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Navigation",
            act_filter  = a -> a == :Ammonia, sign = -1.0)
        meth = _credit_expr(vars, md, ps;
            tech_filter = t -> get(p.activityPer_tech, t, Symbol("")) == :var"Bunker Navigation",
            act_filter  = a -> a == :Methanol, sign = -1.0)
        @constraint(m, lhs - (1.15 * amm + 1.20 * meth) == 0.0,
                    base_name = "H2credNav[$ps]")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FuelEU Maritime — SectorTarget_emi_BunkerNav (p>2030)
# IESA-Opt 1.0 line 21020
# ---------------------------------------------------------------------------
function _add_SectorTarget_emi_BunkerNav!(m::JuMP.Model, vars::AnnualVars,
                                          md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    for ps in s.periods_solve
        ps > 2030 || continue
        target = get(p.FuelEU_Maritime_target, ps, 0.0)
        target == 0.0 && continue
        # IESA-Opt 1.0: sum[(atb,t), ...] with atb in activities_target_Bunkers
        lhs = _sector_flow_expr(vars, md, ps, mode;
            tech_filter = t -> (get(p.tech_sector_kev, t, Symbol("")) == :Bunkerbrandstoffen) &&
                               (get(p.tech_activity, t, Symbol("")) == :var"Bunker Navigation"),
            act_filter  = a -> a in s.activities_target_Bunkers)
        cr = _credit_expr(vars, md, ps; tech_filter = t -> t == :TNB01_10,
                          act_filter = a -> a in s.activities_credits, sign = -1.0)
        @constraint(m, lhs - cr <= target, base_name = "SectorTgtBunkNav[$ps]")
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Nuclear MinLoad — IESA-Opt 1.0 line 20510
# IESA-Opt 1.0:
#   Constraint MinLoad {
#     IndexDomain: (h,thh,ps) | (StringOccurrences(tech_name(thh),'Nuclear')>=1);
#     Definition:  tech_useHourly(h,thh,ps) >= 0.3*techStock(thh,ps)*cap2act(thh)*hourly_profiles(h,'Flat')
#   }
# IMPORTANT:
#   - IESA-Opt 1.0 defines ONLY the FH form (tech_useHourly) — no _TS variant.
#   - `MinLoad` is in NO constraint group (BaseConstraints, LinkingConstraints, …)
#     so it is never activated under any standard scenario in IESA-Opt 1.0.
#   - Provided here as opt-in only (set IESA_POLICY_ENABLE_MINLOAD=1 to add).
#     In TS mode, IESA-Opt 1.0 solves without this constraint; we follow suit.
# ---------------------------------------------------------------------------
function _add_MinLoad!(m::JuMP.Model, vars::AnnualVars, md::ModelData, mode::Symbol)
    s, p = md.sets, md.params
    pss = s.periods_solve
    ts  = vars.techStock
    is_nuclear(t) = occursin("Nuclear", get(p.tech_name, t, ""))
    if mode === :fh
        tuH = vars.tech_useHourly
        tuH === nothing && return
        for thh in s.tech_hourlyDispatch
            is_nuclear(thh) || continue
            c2a = get(p.cap2act, thh, 0.0)
            c2a > 0.0 || continue
            for ps in pss, h in s.hours
                flat = get(p.hourly_profiles, (h, :Flat), 1.0)
                @constraint(m, tuH[h, thh, ps] - 0.3 * c2a * flat * ts[thh, ps] >= 0.0,
                            base_name = "MinLoad[$thh,$h,$ps]")
            end
        end
    else
        # IESA-Opt 1.0 has no TS form of MinLoad; nothing to add for parity.
        @info "MinLoad: skipped in TS mode (IESA-Opt 1.0 has no _TS variant; constraint is in no group)"
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
"""
    add_policy_constraints!(m, vars, md; mode::Symbol = :fh)

Add policy / regulatory constraints (Adapt_*, CO2/H2 credits, ReFuelEU eSAF/SAF,
FuelEU Maritime SectorTarget, Nuclear MinLoad).  Currently only the constraints
active for `ps=2050` are wired up.  For earlier periods, see the IESA-Opt 1.0 source
(IESA-Opt.ams lines 20620-21100) and extend this module.

Mode `:fh` uses hourly variables; `:ts` uses TS cluster-hour variables with
rep-day weighting (`repDayWeight_day`).
"""
function add_policy_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData;
                                 mode::Symbol = :fh)
    mode in (:fh, :ts) || throw(ArgumentError("mode must be :fh or :ts"))
    if get(ENV, "IESA_DISABLE_POLICY", "0") == "1"
        @info "add_policy_constraints! ($mode) — DISABLED by IESA_DISABLE_POLICY=1"
        return m
    end
    @info "add_policy_constraints! ($mode)"
    flush(stderr)

    skip(name) = get(ENV, "IESA_POLICY_SKIP_$(name)", "0") == "1"
    enable(name) = get(ENV, "IESA_POLICY_ENABLE_$(name)", "0") == "1"

    # ------------------------------------------------------------------------
    # 2026-06-15: Default constraint-group parity with IESA-Opt 1.0 BaseET_BFS.
    #
    # The IESA-Opt 1.0 default ActiveConstraintGroup is 'Base + Bunkers + Scope3'
    # (≈ BaseET_BFS), which is the constraint set used to produce the
    # Sweep_TS_PostFix reference runs (obj=55,884.12 MEUR).  BaseET_BFS
    # contains ONLY base balance / capacity / stock / investment / retrofit
    # constraints plus the bunker & feedstock emission-target balances.
    #
    # All "policy" constraints below (Adapt_*, CO2 credits, RFNBO eSAF/SAF,
    # H2 credits, FuelEU Maritime SectorTarget, MinLoad) belong to
    # ADAPTConstraints / ADAPTbunkerConstraints / RFNBOConstraints — they are
    # NOT in BaseET_BFS and are therefore inactive in the IESA-Opt 1.0 reference.
    #
    # Previously Julia added them all by default, which made the model
    # over-constrained (e.g. H2credNav forces TNB01_07/08 → 0 since
    # techUse_max[TNB01_09]=0, leaving no feasible bunker-navigation supply).
    # Flip the defaults to OPT-IN to mirror IESA-Opt 1.0 BaseET_BFS.  Enable via
    # IESA_POLICY_ENABLE_<NAME>=1, e.g. to reproduce 'Base + RFNBO targets'
    # set IESA_POLICY_ENABLE_CO2_AVI/CO2_NAV/ESAF/SAF/H2_AVI/H2_NAV/SECTOR_NAV=1.
    # ------------------------------------------------------------------------
    enable("ADAPT_NAV")   && _add_Adapt_SectoTarget_BunkerNav_2050!(m, vars, md, mode)
    enable("ADAPT_AVI")   && _add_Adapt_SectoTarget_BunkerAvi_2050!(m, vars, md, mode)
    enable("ADAPT_REFIN") && _add_Adapt_RefinProd_2050!(m, vars, md, mode)
    enable("CO2_AVI")     && _add_CO2_credits_Avi!(m, vars, md)
    enable("CO2_NAV")     && _add_CO2_credits_Nav!(m, vars, md)
    enable("ESAF")        && _add_eSAF_Aviation!(m, vars, md)
    enable("SAF")         && _add_SAF_Aviation!(m, vars, md)
    enable("H2_AVI")      && _add_H2_credits_Avi!(m, vars, md)
    enable("H2_NAV")      && _add_H2_credits_Nav!(m, vars, md)
    enable("SECTOR_NAV")  && _add_SectorTarget_emi_BunkerNav!(m, vars, md, mode)
    # MinLoad is opt-in (IESA-Opt 1.0 lists it in no constraint group)
    enable("MINLOAD")     && _add_MinLoad!(m, vars, md, mode)

    return m
end
