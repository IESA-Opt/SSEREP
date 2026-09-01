# =============================================================================
# sets.jl — Derived set computation
#
# Ported from the legacy IESA-Opt Julia prototype's `src/sets.jl::derive_sets!`
# with adaptations:
#   - Operates on a `ModelData` wrapper (md.sets / md.params) instead of (s, p)
#   - All keys are Symbol instead of String
#   - String-valued IESA-Opt 1.0 enum literals are compared via Symbol comparison
#     (e.g. `:hourly`, `:Hourly_dispatch`); the constants below resolve to the
#     exact values used in the default_data.xlsx Types sheet
#
# Computes every IESA-Opt 1.0-derived subset (tech_balancers ∪ tech_infra,
# tech_hourlyDispatch, tech_flex*, activities_*, etc.) using filters on the
# loaded metadata (`processType_tech`, `flexibilityType_tech`, `dispatchType_act`,
# `flex_range`, `shed_range`, `CHP_range`, etc.).
# =============================================================================

# IESA-Opt 1.0 enum literals as Symbols. Comparisons use these constants to make the
# filters legible and easy to update if the XLSX vocabulary changes.
const PROCESS_HOURLY_DISPATCH = :var"Hourly dispatch"
const PROCESS_DAILY_DISPATCH  = :var"Daily dispatch"
const PROCESS_CHP_FLEXIBLE    = :var"CHP flexible"
const PROCESS_SHEDDING        = :Shedding
const PROCESS_RESERVOIR       = :var"Water Reservoir"
const PROCESS_OPERATION       = :Operation
const PROCESS_HOURLY_IMPORT   = :var"Hourly interconnected import"
const PROCESS_GAS_BUFFER      = :var"Gas buffer"

const ACTIVITY_FIX_ENERGY      = :var"Fix Energy"
const ACTIVITY_ENERGY          = :Energy
const ACTIVITY_EMISSION        = :Emission
const ACTIVITY_EMISSIONREPORT  = :EmissionReport
const ACTIVITY_TARGET          = :Target
const ACTIVITY_MATERIAL_CONV   = :var"Material conversion"
const ACTIVITY_CREDITS         = :Credits
const ACTIVITY_DRIVER          = :Driver

const DISPATCH_HOURLY                = :hourly
const DISPATCH_HOURLY_INTERCONNECTED = :var"hourly-interconnected"
const DISPATCH_DAILY                 = :daily
const DISPATCH_HOURLY_INDIRECT       = :var"hourly-indirect"

const FLEX_DR_SHIFTING = :var"DR shifting"
const FLEX_BE_SHIFTING = :var"BE shifting"
const FLEX_STORAGE     = :Storage
const FLEX_EV_SMART    = :var"EV smart charge"
const FLEX_EV_GRID     = :var"EV P-to-Grid"
const FLEX_NONE        = :none

const RANGE_1HOUR    = :var"1 hour [h]"
const RANGE_1DAY     = :var"1 day [d]"
const RANGE_3DAYS    = :var"3 days [r]"
const RANGE_1WEEK    = :var"1 week [w]"
const RANGE_1MONTH   = :var"1 month [m]"
const RANGE_1SEASON  = :var"1 season [s]"
const RANGE_6MONTHS  = :var"6 months [b]"
const RANGE_1YEAR    = :var"1 year [y]"


"""
    derive_sets!(md::ModelData) -> ModelData

Compute all derived subsets from the loaded base sets + metadata. Idempotent:
clears + rebuilds each derived subset on every call. Always populates the
temporal helper sets (`hours_inDay`, `days`, `weeks`, etc.) regardless of
whether base data is loaded.

Returns the (mutated) `ModelData` for chaining.
"""
function derive_sets!(md::ModelData)
    s = md.sets
    p = md.params

    _build_temporal_sets!(s, p)
    _derive_tech_subsets!(s, p)
    _derive_activity_subsets!(s, p)
    _build_technologies_union!(s)
    _derive_tech_materialConversion!(s, p)
    _derive_infra_subsets!(s, p)
    _derive_retrofit_pairs!(s, p)
    return md
end

# -----------------------------------------------------------------------------
# Temporal sets
# -----------------------------------------------------------------------------
function _build_temporal_sets!(s::ModelSets, p::ModelParams)
    # Hours-in-day
    if p.hoursPer_day > 0 && (isempty(s.hours_inDay) || length(s.hours_inDay) != p.hoursPer_day)
        s.hours_inDay = collect(1:p.hoursPer_day)
    end
    if p.hoursPer_day_cluster > 0 && (isempty(s.hours_inDay_cluster) ||
            length(s.hours_inDay_cluster) != p.hoursPer_day_cluster)
        s.hours_inDay_cluster = collect(1:p.hoursPer_day_cluster)
    end

    # Day/week/month/season/semester sets
    if isempty(s.days)
        s.days = collect(1:365)
    end
    if isempty(s.weeks)
        s.weeks = collect(1:53)
    end
    if isempty(s.months)
        s.months = collect(1:12)
    end
    if isempty(s.seasons)
        s.seasons = collect(1:4)
    end
    if isempty(s.semesters)
        s.semesters = collect(1:2)
    end

    # Active hours for the FH (full-horizon) path.
    #
    # IESA-Opt 1.0 supports `hoursPer_day ∈ {1, 2, 3, 4, 6, 8, 12, 24}`. The raw
    # workbook always provides 8760 hourly profile rows in `s.hours_orig`;
    # `_resolve_hourly_profiles_fh!` aggregates them into `365 × hoursPer_day`
    # FH-resolution buckets. The FH model itself indexes hourly variables/
    # constraints over `s.hours`, which must therefore be the aggregated set
    # `1..(365 * hoursPer_day)` — NOT a verbatim copy of `s.hours_orig`.
    #
    # Always overwrite (idempotent on repeated `derive_sets!` calls after a
    # configuration change). TS-mode sets (`s.hours_cluster`, `s.repDays`,
    # `s.hours_inDay_cluster`, ...) are derived separately and untouched.
    if p.hoursPer_day > 0
        n_hours_fh = 365 * p.hoursPer_day
        if isempty(s.hours) || length(s.hours) != n_hours_fh
            s.hours = collect(1:n_hours_fh)
        end
    end

    # Quarter-hour windows: every `hoursPer_quarter_cluster` hours form one window
    if p.hoursPer_quarter_cluster > 0 && !isempty(s.hours)
        n_quarters_fh = max(div(length(s.hours), p.hoursPer_quarter_cluster), 1)
        if isempty(s.q_hourWindow) || length(s.q_hourWindow) != n_quarters_fh
            s.q_hourWindow = collect(1:n_quarters_fh)
        end
    end

    # r-day rolling window (FH only) — daysPer_range typically 3
    if p.daysPer_range > 0 && !isempty(s.days)
        n_rd_fh = max(cld(length(s.days), p.daysPer_range), 1)
        if isempty(s.r_dayWindow) || length(s.r_dayWindow) != n_rd_fh
            s.r_dayWindow = collect(1:n_rd_fh)
        end
    end

    # Clustering temporal sets (only populated when use_clustering=true and rd>0)
    if p.use_clustering && p.n_repDays > 0
        if isempty(s.repDays) || length(s.repDays) != p.n_repDays
            s.repDays = collect(1:p.n_repDays)
        end
        n_hours_cluster = p.n_repDays * p.hoursPer_day_cluster
        if isempty(s.hours_cluster) || length(s.hours_cluster) != n_hours_cluster
            s.hours_cluster = collect(1:n_hours_cluster)
        end
        if p.hoursPer_quarter_cluster > 0
            n_q_cl = max(div(n_hours_cluster, p.hoursPer_quarter_cluster), 1)
            if isempty(s.q_hourWindow_cluster) || length(s.q_hourWindow_cluster) != n_q_cl
                s.q_hourWindow_cluster = collect(1:n_q_cl)
            end
        end
    end

    # CHP polytope facets — fixed 3-facet polyhedron (IESA-Opt 1.0 f1, f2, f3)
    if isempty(s.CHP_polyFacets)
        s.CHP_polyFacets = [:f1, :f2, :f3]
    end
    return nothing
end
# -----------------------------------------------------------------------------
# Tech subsets
# -----------------------------------------------------------------------------
function _derive_tech_subsets!(s::ModelSets, p::ModelParams)
    process(t) = get(p.processType_tech, t, Symbol(""))
    flex(t)    = get(p.flexibilityType_tech, t, FLEX_NONE)
    flexr(t)   = get(p.flex_range, t, Symbol(""))
    shedr(t)   = get(p.shed_range, t, Symbol(""))
    chpr(t)    = get(p.CHP_range, t, Symbol(""))

    s.tech_hourlyDispatch  = [t for t in s.tech_balancers if process(t) == PROCESS_HOURLY_DISPATCH]
    s.tech_dailyDispatch   = [t for t in s.tech_balancers if process(t) == PROCESS_DAILY_DISPATCH]
    s.tech_hourlyCHPflex   = [t for t in s.tech_balancers if process(t) == PROCESS_CHP_FLEXIBLE]
    s.tech_shedding        = [t for t in s.tech_balancers if process(t) == PROCESS_SHEDDING]
    s.tech_reservoir       = [t for t in s.tech_balancers if process(t) == PROCESS_RESERVOIR]

    # CHP range-based subsets (IESA-Opt 1.0 tech_hourlyCHPflexH/D/W — lines 4659/4673/4687 region)
    s.tech_hourlyCHPflexH = [t for t in s.tech_hourlyCHPflex if chpr(t) == RANGE_1HOUR]
    s.tech_hourlyCHPflexD = [t for t in s.tech_hourlyCHPflex if chpr(t) == RANGE_1DAY]
    s.tech_hourlyCHPflexW = [t for t in s.tech_hourlyCHPflex if chpr(t) == RANGE_1WEEK]

    # Shedding range subsets
    s.tech_shedH = [t for t in s.tech_shedding if shedr(t) in (RANGE_1HOUR, RANGE_1DAY)]
    s.tech_shedW = [t for t in s.tech_shedding if shedr(t) == RANGE_1WEEK]

    # Gas buffers: IESA-Opt 1.0 line 3128 — `tech_gasBuffer = {t | processType_tech(t)='Gas buffer'}`.
    # NOTE: The previous Julia rule `buffer_storage > 0` was wrong — it picked up
    # demand-side techs (Ser01_01, Agr01_01, Res02_01, ...) whose `buffer_storage`
    # column is a tank-size hint but whose process type is `Operation`. The
    # IESA-Opt 1.0 canonical rule is process-type-based.
    s.tech_gasBuffer = [t for t in s.tech_balancers if process(t) == PROCESS_GAS_BUFFER]

    # Flexible demand subsets
    s.tech_flexible = [t for t in s.tech_balancers if flex(t) != FLEX_NONE && flex(t) != Symbol("")]
    s.tech_fStorage = [t for t in s.tech_flexible if flex(t) == FLEX_STORAGE]
    s.tech_fEV      = [t for t in s.tech_flexible if flex(t) in (FLEX_EV_SMART, FLEX_EV_GRID)]
    # IESA-Opt 1.0 tech_fEVcharging / tech_fEVgrid (lines 3306-3309 region)
    s.tech_fEVcharging  = [t for t in s.tech_flexible if flex(t) == FLEX_EV_SMART]
    s.tech_fEVgrid      = [t for t in s.tech_flexible if flex(t) == FLEX_EV_GRID]
    # IESA-Opt 1.0 tech_fDRshifting / tech_fBEshifting (lines 3300/3303 region)
    s.tech_fDRshifting  = [t for t in s.tech_flexible if flex(t) == FLEX_DR_SHIFTING]
    s.tech_fBEshifting  = [t for t in s.tech_flexible if flex(t) == FLEX_BE_SHIFTING]
    s.tech_fWithBattery = [t for t in s.tech_flexible if get(p.flex_storage, t, 0.0) > 0.0]

    # Flex range subsets
    s.tech_flexH  = [t for t in s.tech_flexible if flexr(t) == RANGE_1HOUR]
    s.tech_flexD  = [t for t in s.tech_flexible if flexr(t) == RANGE_1DAY]
    s.tech_flexR  = [t for t in s.tech_flexible if flexr(t) == RANGE_3DAYS]
    s.tech_flexW  = [t for t in s.tech_flexible if flexr(t) == RANGE_1WEEK]
    s.tech_flexM  = [t for t in s.tech_flexible if flexr(t) == RANGE_1MONTH]
    s.tech_flexS  = [t for t in s.tech_flexible if flexr(t) == RANGE_1SEASON]
    s.tech_flexB  = [t for t in s.tech_flexible if flexr(t) == RANGE_6MONTHS]
    s.tech_flexY  = [t for t in s.tech_flexible if flexr(t) == RANGE_1YEAR]
    s.tech_flexLT = [t for t in s.tech_flexible if flex(t) != FLEX_BE_SHIFTING]

    # Emission technologies (where the main activity is of type Emission or EmissionReport)
    s.tech_emission = [t for t in vcat(s.tech_balancers, s.tech_infra)
                       if (a = get(p.activityPer_tech, t, get(p.activityPer_techOrig, t, Symbol("")));
                           a != Symbol("") && get(p.activityType_act, a, Symbol("")) in
                               (ACTIVITY_EMISSION, ACTIVITY_EMISSIONREPORT))]
    return nothing
end

# -----------------------------------------------------------------------------
# tech_Operation (IESA-Opt 1.0 line 2953) — used as the passive base index `tp` in the
# hourly/daily activity-balance constraints. NOT mutually exclusive with
# tech_flexible: a tech with processType_tech='Operation' AND a non-empty
# flexibilityType_tech (e.g. 'DR shifting') appears in BOTH tp and tf.
#
# IESA-Opt 1.0 definition:
#   tech_Operation =
#     {t | processType_tech(t)='Operation'}
#     + {t | (ord(activityPer_tech(t),activities_group)<>0)
#            * (activity_balances(t,activityPer_tech(t),base_year) = -XC_TransmissionLoss)}
#     + {t | (...same...) = -HVtoMV_TransformerLoss}
#     + {t | (...same...) = -MVtoLV_TransformerLoss}
#     + {t | (...same...) = -HVtoLV_TransformerLoss}
#
# When `activities_group` is empty (typical for non-grouped datasets), the union
# reduces to the first clause and the activities-group extension is a NOP.
# TODO: When `activities_group` is non-empty, port the loss-constant matching
# (requires reading HVtoMV/MVtoLV/HVtoLV transformer-loss globals from XLSX).
# -----------------------------------------------------------------------------
function _build_tech_Operation(s::ModelSets, p::ModelParams)::Set{Symbol}
    op = Set{Symbol}()
    for t in s.tech_balancers
        if get(p.processType_tech, t, Symbol("")) == PROCESS_OPERATION
            push!(op, t)
        end
    end
    # Activities-group extension (NOP for current dataset since activities_group is empty).
    # Left unimplemented because the HV/MV/LV transformer-loss globals are not yet
    # read in `data_reading.jl`. If that gap is filled, mirror IESA-Opt 1.0 line 2953
    # exactly. For datasets with non-empty ActGrouping the model would otherwise
    # under-count `tp` membership, so log a warning.
    if !isempty(s.activities_group)
        @warn "_build_tech_Operation: activities_group is non-empty but the IESA-Opt 1.0 \
loss-constant extension (XC/HV/MV/LV transformer losses) is not yet implemented \
in Julia; tech_Operation may under-count grouped-network connector techs. \
See IESA-Opt 1.0 IESA-Opt.ams line 2953."
    end
    return op
end

# -----------------------------------------------------------------------------
# Infrastructure subsets (depend on tech_infra + infra_range + infra_activity)
# IESA-Opt 1.0 lines 5104-5122
# -----------------------------------------------------------------------------
function _derive_infra_subsets!(s::ModelSets, p::ModelParams)
    irange(t) = get(p.infra_range, t, Symbol(""))
    s.tech_infraH = [t for t in s.tech_infra if irange(t) == RANGE_1HOUR]
    s.tech_infraD = [t for t in s.tech_infra if irange(t) == RANGE_1DAY]

    # IESA-Opt 1.0: act_infraH := act_infraH + infra_activity(iti_h) for iti_h in tech_infraH (unique)
    acth = Symbol[]
    seenh = Set{Symbol}()
    for t in s.tech_infraH
        a = get(p.infra_activity, t, Symbol(""))
        a == Symbol("") && continue
        a in seenh || (push!(acth, a); push!(seenh, a))
    end
    s.act_infraH = acth

    actd = Symbol[]
    seend = Set{Symbol}()
    for t in s.tech_infraD
        a = get(p.infra_activity, t, Symbol(""))
        a == Symbol("") && continue
        a in seend || (push!(actd, a); push!(seend, a))
    end
    s.act_infraD = actd
    return nothing
end

# -----------------------------------------------------------------------------
# Activity subsets
# -----------------------------------------------------------------------------
function _derive_activity_subsets!(s::ModelSets, p::ModelParams)
    if isempty(s.activities)
        # `activities` may be empty before `_derive_profile_and_node_maps!` runs.
        # Recompute as union of original ∪ group.
        seen = Set{Symbol}()
        out  = Symbol[]
        for a in s.activities_original
            a in seen || (push!(out, a); push!(seen, a))
        end
        for a in s.activities_group
            a in seen || (push!(out, a); push!(seen, a))
        end
        s.activities = out
    end

    atype(a) = get(p.activityType_act, a, Symbol(""))
    dtype(a) = get(p.dispatchType_act, a, Symbol(""))
    target_bin(a) = get(p.emissionTarget_bin, a, Symbol(""))

    s.activities_balance = [a for a in s.activities
        if !(atype(a) in (ACTIVITY_FIX_ENERGY, ACTIVITY_EMISSION, ACTIVITY_EMISSIONREPORT,
                          ACTIVITY_TARGET, ACTIVITY_MATERIAL_CONV))]
    s.activities_fixEnergy = [a for a in s.activities if atype(a) == ACTIVITY_FIX_ENERGY]
    s.activities_energy    = [a for a in s.activities if atype(a) in (ACTIVITY_ENERGY, ACTIVITY_FIX_ENERGY)]
    s.activities_hour      = [a for a in s.activities if dtype(a) in (DISPATCH_HOURLY, DISPATCH_HOURLY_INTERCONNECTED)]
    s.activities_day       = [a for a in s.activities if dtype(a) == DISPATCH_DAILY]
    s.activities_year      = [a for a in s.activities_energy if !(a in s.activities_hour) && !(a in s.activities_day)]
    s.activities_indirect  = [a for a in s.activities if dtype(a) == DISPATCH_HOURLY_INDIRECT]
    s.activities_target    = [a for a in s.activities if target_bin(a) == Symbol("1")]
    # IESA-Opt 1.0 line ~2240,2247: activities_target_FeedStocks/Bunkers (subsets of activities_solve)
    # filtered by actSolvePer_actOrig == specific original-activity-name.
    _origname(a) = string(get(p.actSolvePer_actOrig, a, a))
    s.activities_target_FeedStocks = [a for a in s.activities if _origname(a) == "CO2 Air Feedstock end of lifetime"]
    s.activities_target_Bunkers    = [a for a in s.activities if _origname(a) == "CO2 Air Int. Transport"]
    s.activities_driver    = [a for a in s.activities if atype(a) == ACTIVITY_DRIVER]
    s.activities_materialConversion = [a for a in s.activities if atype(a) == ACTIVITY_MATERIAL_CONV]
    # IESA-Opt 1.0 line 2225: activities_emission = {as | activityType_act(as)='Emission' and emissionTarget_bin(as)<>1}
    s.activities_emission  = [a for a in s.activities
                              if atype(a) == ACTIVITY_EMISSION && target_bin(a) != Symbol("1")]
    # IESA-Opt 1.0 line ~2218: activities_emissionFix (acf) = {as | activityType_act(as)='Emission'} (no emissionTarget_bin filter)
    s.activities_emissionFix = [a for a in s.activities if atype(a) == ACTIVITY_EMISSION]
    # IESA-Opt 1.0 line 2208: activities_energyNonFixed = {as | activityType_act(as)='Energy'}  (Energy only, excludes Fix Energy)
    s.activities_energyNonFixed = [a for a in s.activities if atype(a) == ACTIVITY_ENERGY]
    # IESA-Opt 1.0 line 2266: activities_emissionReport = {as | activityType_act(as)='EmissionReport'}
    s.activities_emissionReport = [a for a in s.activities if atype(a) == ACTIVITY_EMISSIONREPORT]
    # IESA-Opt 1.0 line 2273: activities_credits = {as | activityType_act(as)='Credits'}
    s.activities_credits        = [a for a in s.activities if atype(a) == ACTIVITY_CREDITS]

    # activities_solve: if grouped → use group representative; else original
    solved = Set{Symbol}()
    out    = Symbol[]
    for a in s.activities_original
        grp = get(p.act_to_group, a, Symbol(""))
        repr_act = grp == Symbol("") ? a : grp
        repr_act in solved || (push!(out, repr_act); push!(solved, repr_act))
    end
    for ag in s.activities_group
        ag in solved || (push!(out, ag); push!(solved, ag))
    end
    s.activities_solve = out
    return nothing
end

# -----------------------------------------------------------------------------
# Final union: technologies = tech_balancers ∪ tech_infra
# -----------------------------------------------------------------------------
function _build_technologies_union!(s::ModelSets)
    seen = Set{Symbol}()
    out  = Symbol[]
    for t in s.tech_balancers
        t in seen || (push!(out, t); push!(seen, t))
    end
    for t in s.tech_infra
        t in seen || (push!(out, t); push!(seen, t))
    end
    s.technologies = out
    return nothing
end

# -----------------------------------------------------------------------------
# tech_materialConversion (depends on technologies + activityPer_tech)
# -----------------------------------------------------------------------------
function _derive_tech_materialConversion!(s::ModelSets, p::ModelParams)
    s.tech_materialConversion = [t for t in s.technologies
        if get(p.activityType_act,
               get(p.activityPer_tech, t, Symbol("")),
               Symbol("")) == ACTIVITY_MATERIAL_CONV]
    return nothing
end

# -----------------------------------------------------------------------------
# Sparse retrofit support: derive `retrofit_pairs` (Vector of (it,jt)) plus
# per-tech in/out adjacency from `retrofit_relations`.
# AIMMS materializes only `retrofit_relations(it,jt)==true` entries before
# sending to Gurobi; the Julia model declared `retrofitting[it,jt,ps]` dense
# over `technologies × technologies`, creating ~|tech|^2 redundant variables
# (Gurobi's presolve must then eliminate them — but the LP that Gurobi sees
# is much larger than AIMMS's, breaking apples-to-apples comparison and
# wasting build time). These derived containers let `add_annual_variables!`
# build a sparse retrofit variable indexed only over active pairs.
# -----------------------------------------------------------------------------
function _derive_retrofit_pairs!(s::ModelSets, p::ModelParams)
    pairs = Tuple{Symbol,Symbol}[]
    in_by  = Dict{Symbol,Vector{Symbol}}()
    out_by = Dict{Symbol,Vector{Symbol}}()

    # Default: sparse mode — only emit the (it,jt) pairs for which
    # `retrofit_relations(it,jt)==true`. This is what AIMMS's NetVarMatrix
    # presolver effectively passes to Gurobi.
    #
    # Set env var `IESA_OPT_DENSE_RETROFIT=1` to materialize the full
    # `technologies × technologies` cross-product instead (AIMMS-comparable
    # mode for LP-size validation). In dense mode every inactive entry is
    # still emitted, and `retrofit_constraint` multiplies the RHS by the
    # relation factor so inactive variables are pinned to 0 (→ Gurobi
    # presolve will drop them, matching AIMMS's behavior).
    dense = get(ENV, "IESA_OPT_DENSE_RETROFIT", "0") == "1"

    if dense && !isempty(s.technologies)
        techs = collect(s.technologies)
        sizehint!(pairs, length(techs)^2)
        for it_ in techs, jt_ in techs
            push!(pairs, (it_, jt_))
        end
        for t_ in techs
            in_by[t_]  = copy(techs)
            out_by[t_] = copy(techs)
        end
    else
        for ((it_, jt_), v) in p.retrofit_relations
            v || continue
            push!(pairs, (it_, jt_))
            push!(get!(in_by,  jt_, Symbol[]), it_)  # (it_, jt_): jt_ has it_ as inbound
            push!(get!(out_by, it_, Symbol[]), jt_)  # (it_, jt_): it_ has jt_ as outbound
        end
    end

    p.retrofit_pairs        = pairs
    p.retrofit_in_by_tech   = in_by
    p.retrofit_out_by_tech  = out_by
    return nothing
end
