# =============================================================================
# infrastructure.jl — infraVol_H, infraVol_D, infraVol_H_TS, infraVol_D_TS
#
# IESA-Opt 1.0 lines 5144-5202.  The infrastructure-volume constraints bound the
# absolute throughput of each "hourly-infrastructure" activity (`ai_h`) or
# "daily-infrastructure" activity (`ai_d`) by the available infra capacity
# (Σ techStock × cap2act over techs in tech_infraH / tech_infraD whose
# infra_activity == ai_h / ai_d).
#
# The LHS aggregates total **consumption** of that activity by every balancer
# tech `itb` whose activity_balance with `ai_*` is **negative** (i.e. itb
# consumes ai_*), summing the dispatch streams:
#     hourly: tech_useHourly + tech_useDaily/hoursPerDayEffective
#           + tech_use × hourly_profile + deltaS_shed
#     daily : tech_useDaily + Σ_h∈d (tech_useHourly + prof×(tech_use + ΔU_CHP + ΔS_shed))
#
# plus the upward / downward flexible-demand response Σ_tf (ΔQ_UP +
# ΔQ_DW × indicator(tf ∉ tech_fStorage)) × dQ_hourly(tf, ai), and (daily only)
# the buffer-up term Σ_tg ΔB_UP × dB_daily(tg, ai_d).
#
# RHS = − Σ_{iti∈tech_infraH (or D)} techStock × cap2act × (24/hoursPerDayEffective
# for hourly, 1 for daily; for TS use hoursPer_day_cluster).
#
# All four constraints are `>=` after moving RHS to the right (LHS aggregate
# consumption ≥ −infra capacity).  The constraint is added only when:
#   - `tech_infraH`/`tech_infraD` non-empty
#   - `act_infraH`/`act_infraD` non-empty
#   - corresponding hourly variables exist (FH: tech_useHourly/Daily; TS: _TS)
# =============================================================================

const _IJ_INFRA_EPS = 1e-12

# ---------------------------------------------------------------------------
# FH: infraVol_H — hourly infrastructure
# ---------------------------------------------------------------------------
"""
    _add_infraVol_H!(m, vars, md)

IESA-Opt 1.0 line 5144.  Bound hourly-infra activity throughput at each hour.
"""
function _add_infraVol_H!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    isempty(s.tech_infraH) && return
    isempty(s.act_infraH)  && return
    vars.tech_useHourly === nothing && return

    pss = s.periods_solve
    tuH = vars.tech_useHourly
    tuD = vars.tech_useDaily
    tu  = vars.tech_use
    dS  = vars.deltaS_shed
    dqUP = vars.deltaQ_UP
    dqDW = vars.deltaQ_DW
    ts  = vars.techStock

    for ah in s.act_infraH, ps in pss
        # Precompute per-tech terms: which balancer-techs consume ah (ab<0)
        cons_techs = Tuple{Symbol,Float64}[]
        for tb in s.tech_balancers
            ab = get(p.activity_balances, (tb, ah, ps), 0.0)
            if ab < -_IJ_INFRA_EPS
                push!(cons_techs, (tb, ab))
            end
        end
        isempty(cons_techs) && continue

        # RHS techs: infraH techs whose infra_activity == ah; their cap = techStock × cap2act
        rhs_techs = Tuple{Symbol,Float64}[]
        for iti in s.tech_infraH
            if get(p.infra_activity, iti, Symbol("")) == ah
                c2a = get(p.cap2act, iti, 0.0)
                c2a > 0.0 && push!(rhs_techs, (iti, c2a))
            end
        end

        for h in s.hours
            d = get(p.dayPer_hour, h, 0)
            hpdEff = max(get(p.hoursPerDayEffective, h, 24.0 / max(p.hoursPer_day, 1)), 1e-6)
            expr = AffExpr(0.0)
            for (tb, ab) in cons_techs
                # tech_useHourly term (only if tb is hourly-dispatch)
                if tb in s.tech_hourlyDispatch
                    add_to_expression!(expr, ab, tuH[h, tb, ps])
                end
                # tech_useDaily term scaled to per-hour
                if tb in s.tech_dailyDispatch && tuD !== nothing && d > 0
                    add_to_expression!(expr, ab / hpdEff, tuD[d, tb, ps])
                end
                # tech_use × hourly_profile
                prof_t = get(p.profileType_tech, tb, :Flat)
                prof = get(p.hourly_profiles, (h, prof_t), 0.0)
                if abs(prof) > _IJ_INFRA_EPS
                    add_to_expression!(expr, ab * prof, tu[tb, ps])
                end
                # shedding
                if tb in s.tech_shedding && dS !== nothing
                    add_to_expression!(expr, ab, dS[h, tb, ps])
                end
            end
            # Flexible-demand response (over all tech_flexible) — independent of cons_techs
            if dqUP !== nothing
                for tf in s.tech_flexible
                    dq = get(p.dQ_hourly, (tf, ah), 0.0)
                    dq == 0.0 && continue
                    add_to_expression!(expr, dq, dqUP[h, tf, ps])
                    if !(tf in s.tech_fStorage) && dqDW !== nothing
                        add_to_expression!(expr, dq, dqDW[h, tf, ps])
                    end
                end
            end
            # RHS: Σ techStock × cap2act × (24/hpdEff)
            slice_scale = 24.0 / hpdEff
            for (iti, c2a) in rhs_techs
                add_to_expression!(expr, c2a * slice_scale, ts[iti, ps])
            end
            @constraint(m, expr >= 0.0, base_name = "infraVH[$ah,$h,$ps]")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FH: infraVol_D — daily infrastructure
# ---------------------------------------------------------------------------
function _add_infraVol_D!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    isempty(s.tech_infraD) && return
    isempty(s.act_infraD)  && return
    (vars.tech_useDaily === nothing) && return

    pss = s.periods_solve
    tuH = vars.tech_useHourly
    tuD = vars.tech_useDaily
    tu  = vars.tech_use
    dS  = vars.deltaS_shed
    duCHP = vars.deltaU_CHP
    dbUP = vars.deltaB_UP
    ts  = vars.techStock

    for ad in s.act_infraD, ps in pss
        cons_techs = Tuple{Symbol,Float64}[]
        for tb in s.tech_balancers
            ab = get(p.activity_balances, (tb, ad, ps), 0.0)
            if ab < -_IJ_INFRA_EPS
                push!(cons_techs, (tb, ab))
            end
        end
        isempty(cons_techs) && continue

        rhs_techs = Tuple{Symbol,Float64}[]
        for iti in s.tech_infraD
            if get(p.infra_activity, iti, Symbol("")) == ad
                c2a = get(p.cap2act, iti, 0.0)
                c2a > 0.0 && push!(rhs_techs, (iti, c2a))
            end
        end

        # Precompute hours-of-day
        hours_of_day = Dict{Int,Vector{Int}}()
        for h in s.hours
            d = get(p.dayPer_hour, h, 0)
            d == 0 && continue
            push!(get!(() -> Int[], hours_of_day, d), h)
        end

        for d in s.days
            hours_d = get(hours_of_day, d, Int[])
            isempty(hours_d) && continue
            expr = AffExpr(0.0)
            for (tb, ab) in cons_techs
                # daily dispatch units
                if tb in s.tech_dailyDispatch
                    add_to_expression!(expr, ab, tuD[d, tb, ps])
                end
                # Sum over hours of day for hourly dispatch + tech_use*prof + CHP + shed
                for h in hours_d
                    if tb in s.tech_hourlyDispatch && tuH !== nothing
                        add_to_expression!(expr, ab, tuH[h, tb, ps])
                    end
                    prof_t = get(p.profileType_tech, tb, :Flat)
                    prof = get(p.hourly_profiles, (h, prof_t), 0.0)
                    if abs(prof) > _IJ_INFRA_EPS
                        add_to_expression!(expr, ab * prof, tu[tb, ps])
                    end
                    if tb in s.tech_hourlyCHPflex && duCHP !== nothing
                        if abs(prof) > _IJ_INFRA_EPS
                            add_to_expression!(expr, ab * prof, duCHP[h, tb, ps])
                        end
                    end
                    if tb in s.tech_shedding && dS !== nothing
                        if abs(prof) > _IJ_INFRA_EPS
                            add_to_expression!(expr, ab * prof, dS[h, tb, ps])
                        end
                    end
                end
            end
            # Gas-buffer daily contribution
            if dbUP !== nothing
                for tg in s.tech_gasBuffer
                    db = get(p.dB_daily, (tg, ad), 0.0)
                    db == 0.0 && continue
                    add_to_expression!(expr, db, dbUP[d, tg, ps])
                end
            end
            # RHS techs (daily)
            for (iti, c2a) in rhs_techs
                add_to_expression!(expr, c2a, ts[iti, ps])
            end
            @constraint(m, expr >= 0.0, base_name = "infraVD[$ad,$d,$ps]")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TS: infraVol_H_TS — hourly infrastructure (cluster hour)
# ---------------------------------------------------------------------------
function _add_infraVol_H_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    isempty(s.tech_infraH) && return
    isempty(s.act_infraH)  && return
    vars.tech_useHourly_TS === nothing && return
    isempty(s.hours_cluster) && return

    pss  = s.periods_solve
    tuH  = vars.tech_useHourly_TS
    tuD  = vars.tech_useDaily_TS
    tu   = vars.tech_use
    dS   = vars.deltaS_shed_TS
    dqUP = vars.deltaQ_UP_TS
    dqDW = vars.deltaQ_DW_TS
    ts   = vars.techStock
    hpd_c = max(p.hoursPer_day_cluster, 1)

    for ah in s.act_infraH, ps in pss
        cons_techs = Tuple{Symbol,Float64}[]
        for tb in s.tech_balancers
            ab = get(p.activity_balances, (tb, ah, ps), 0.0)
            if ab < -_IJ_INFRA_EPS
                push!(cons_techs, (tb, ab))
            end
        end
        isempty(cons_techs) && continue

        rhs_techs = Tuple{Symbol,Float64}[]
        for iti in s.tech_infraH
            if get(p.infra_activity, iti, Symbol("")) == ah
                c2a = get(p.cap2act, iti, 0.0)
                c2a > 0.0 && push!(rhs_techs, (iti, c2a))
            end
        end

        for hc in s.hours_cluster
            rd = get(p.repDay_of_clusterHour, hc, 0)
            expr = AffExpr(0.0)
            for (tb, ab) in cons_techs
                if tb in s.tech_hourlyDispatch
                    add_to_expression!(expr, ab, tuH[hc, tb, ps])
                end
                if tb in s.tech_dailyDispatch && tuD !== nothing && rd > 0
                    add_to_expression!(expr, ab / hpd_c, tuD[rd, tb, ps])
                end
                prof_t = get(p.profileType_tech, tb, :Flat)
                prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
                if abs(prof) > _IJ_INFRA_EPS
                    add_to_expression!(expr, ab * prof, tu[tb, ps])
                end
                if tb in s.tech_shedding && dS !== nothing
                    add_to_expression!(expr, ab, dS[hc, tb, ps])
                end
            end
            if dqUP !== nothing
                for tf in s.tech_flexible
                    dq = get(p.dQ_hourly, (tf, ah), 0.0)
                    dq == 0.0 && continue
                    add_to_expression!(expr, dq, dqUP[hc, tf, ps])
                    if !(tf in s.tech_fStorage) && dqDW !== nothing
                        add_to_expression!(expr, dq, dqDW[hc, tf, ps])
                    end
                end
            end
            for (iti, c2a) in rhs_techs
                add_to_expression!(expr, c2a, ts[iti, ps])
            end
            @constraint(m, expr >= 0.0, base_name = "infraVH_TS[$ah,$hc,$ps]")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# TS: infraVol_D_TS — daily infrastructure (rep-day)
# ---------------------------------------------------------------------------
function _add_infraVol_D_TS!(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    s, p = md.sets, md.params
    isempty(s.tech_infraD) && return
    isempty(s.act_infraD)  && return
    (vars.tech_useDaily_TS === nothing) && return
    isempty(s.repDays) && return

    pss  = s.periods_solve
    tuH  = vars.tech_useHourly_TS
    tuD  = vars.tech_useDaily_TS
    tu   = vars.tech_use
    dS   = vars.deltaS_shed_TS
    duCHP = vars.deltaU_CHP_TS
    dbUP = vars.deltaB_UP_TS
    ts   = vars.techStock

    # Precompute hours-of-rd
    hours_of_rd = Dict{Int,Vector{Int}}()
    for hc in s.hours_cluster
        rd = get(p.repDay_of_clusterHour, hc, 0)
        rd == 0 && continue
        push!(get!(() -> Int[], hours_of_rd, rd), hc)
    end

    for ad in s.act_infraD, ps in pss
        cons_techs = Tuple{Symbol,Float64}[]
        for tb in s.tech_balancers
            ab = get(p.activity_balances, (tb, ad, ps), 0.0)
            if ab < -_IJ_INFRA_EPS
                push!(cons_techs, (tb, ab))
            end
        end
        isempty(cons_techs) && continue

        rhs_techs = Tuple{Symbol,Float64}[]
        for iti in s.tech_infraD
            if get(p.infra_activity, iti, Symbol("")) == ad
                c2a = get(p.cap2act, iti, 0.0)
                c2a > 0.0 && push!(rhs_techs, (iti, c2a))
            end
        end

        for rd in s.repDays
            hours_d = get(hours_of_rd, rd, Int[])
            isempty(hours_d) && continue
            expr = AffExpr(0.0)
            for (tb, ab) in cons_techs
                if tb in s.tech_dailyDispatch
                    add_to_expression!(expr, ab, tuD[rd, tb, ps])
                end
                for hc in hours_d
                    if tb in s.tech_hourlyDispatch && tuH !== nothing
                        add_to_expression!(expr, ab, tuH[hc, tb, ps])
                    end
                    prof_t = get(p.profileType_tech, tb, :Flat)
                    prof = get(p.hourly_profiles_cluster, (hc, prof_t), 0.0)
                    if abs(prof) > _IJ_INFRA_EPS
                        add_to_expression!(expr, ab * prof, tu[tb, ps])
                    end
                    if tb in s.tech_hourlyCHPflex && duCHP !== nothing
                        if abs(prof) > _IJ_INFRA_EPS
                            add_to_expression!(expr, ab * prof, duCHP[hc, tb, ps])
                        end
                    end
                    if tb in s.tech_shedding && dS !== nothing
                        if abs(prof) > _IJ_INFRA_EPS
                            add_to_expression!(expr, ab * prof, dS[hc, tb, ps])
                        end
                    end
                end
            end
            if dbUP !== nothing
                for tg in s.tech_gasBuffer
                    db = get(p.dB_daily, (tg, ad), 0.0)
                    db == 0.0 && continue
                    add_to_expression!(expr, db, dbUP[rd, tg, ps])
                end
            end
            for (iti, c2a) in rhs_techs
                add_to_expression!(expr, c2a, ts[iti, ps])
            end
            @constraint(m, expr >= 0.0, base_name = "infraVD_TS[$ad,$rd,$ps]")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public entry: orchestrates per-mode
# ---------------------------------------------------------------------------
"""
    add_infrastructure_constraints!(m, vars, md; mode::Symbol = :fh)

Add infrastructure-volume constraints (`infraVol_H` + `infraVol_D` for FH;
`infraVol_H_TS` + `infraVol_D_TS` for TS).  Idempotent and safe to call when
`tech_infraH/D` or `act_infraH/D` are empty.
"""
function add_infrastructure_constraints!(m::JuMP.Model, vars::AnnualVars, md::ModelData;
                                         mode::Symbol = :fh)
    if get(ENV, "IESA_DISABLE_INFRA", "0") == "1"
        @info "add_infrastructure_constraints! ($mode) — DISABLED by IESA_DISABLE_INFRA=1"
        return m
    end
    if mode === :fh
        @info "add_infrastructure_constraints! — FH: infraVol_H + infraVol_D"
        flush(stderr)
        _add_infraVol_H!(m, vars, md)
        _add_infraVol_D!(m, vars, md)
    elseif mode === :ts
        @info "add_infrastructure_constraints! — TS: infraVol_H_TS + infraVol_D_TS"
        flush(stderr)
        _add_infraVol_H_TS!(m, vars, md)
        _add_infraVol_D_TS!(m, vars, md)
    else
        throw(ArgumentError("mode must be :fh or :ts, got $mode"))
    end
    return m
end
