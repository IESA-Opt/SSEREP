# =============================================================================
# multi_region.jl — region-cap constraints (project-specific / "Komar")
#
# Port of the AIMMS `Section Komar` constraints from `IESA-Opt.ams`
# (lines 12124–12609 in the colleague's working copy). These are
# *project-specific* extensions that bind regional clones of a tech to the
# national cap stored on a designated "carrier" instance.
#
# Activation: `:multi_region in md.params.extensions`. The UI exposes a
# `MultiRegion` toggle (default OFF). When OFF, NONE of these constraints
# are added — the core model is untouched.
#
# Conventions in the multi-region workbook (e.g. 10Nodes_*.xlsx):
#   - Each tech ID is `CL{n}_{TechFamily}` where `CL{n}` is the regional
#     cluster and `TechFamily` is the AIMMS tech-family code.
#   - For each "logical" tech family (e.g. `PNL01_15`), one regional
#     instance per family carries the *national* `techStock_max(t, p)` (or
#     `techUse_max`) — by AIMMS convention the **first** instance in the
#     technologies set order. The Komar constraints sum the LP variable
#     across all regional siblings sharing the family substring and bound
#     the total by the carrier's cap.
#
# Active constraint families:
#   Stock-cap, exact repeated regional cap:
#     • Comulative_Interconnection_Exports         (PEU01_03)
#     • Comulative_Interconnection_Imports         (PNL04_01)
#     • Comulative_HVNS                            (PNL04_02)
#     • CCUS_Under                                 (Emi01_02)
#     • CCUS_Outside                               (Emi01_03)
#     • Imported_Hydrogen                          (Hyd03_01)
#     • Imported_Ammonia                           (Amm01_07)
#     • Imported_NG                                (Gas01_03)
#     • Ammonia_wCCUS                              (Amm01_02)
#     • Ammonia_eSMR                               (Amm01_08)
#   Stock-cap, exact repeated regional cap over s_TechGroups candidates:
#     • c_TechStock_Limit                          (s_TechGroups)
#   Use-cap, exact repeated regional cap:
#     • all regional `CL{n}_{TechFamily}` families with an exact repeated
#       positive techUse_max in the workbook; PEU01_03 and PNL04_01 are kept
#       as explicit AIMMS-compatible seed candidates.
#   Stock-cap, special:
#     • Comulative_Nuclear   (LHS: tech_name contains "Nuclear" but not
#                             "Borssele"; RHS: techStock_max of FIRST
#                             carrier with substring "PNL01_11")
#     • Comulative_Geothermal (per-tech cap: for each t containing
#                              "LTN01_05", sum stock of all techs in the
#                              same cluster prefix whose name contains
#                              "LTN01_05" / "Agr05_03" / "LTI01_04";
#                              bound by techStock_max(t, p) on each pivot t)
#
# Commented-out in AIMMS (intentionally NOT ported):
#   Comulative_SolarRes / Comulative_SolarSer / Comulative_GMPV /
#   Comulative_Onshore_Wind. These used LAST-carrier semantics. If your
#   workbook needs them, add them analogously.
# =============================================================================

# ----------------------------------------------------------------------------
# AIMMS s_TechGroups — exact copy of `Set s_TechGroups` from AIMMS L12188.
# 131 family substrings. Each substring's stock is summed across all techs
# whose name contains it and bounded by `techStock_max` of the FIRST tech in
# that family.
# ----------------------------------------------------------------------------
const _MR_S_TECH_GROUPS = String[
    "PNL01_04","PNL01_05","PNL01_07","PNL01_08","PNL01_09","PNL01_12","PNL01_13","PNL01_14",
    "PNL01_23","PNL03_01","PNL03_02","PNL03_03","PNL03_04",
    "LTR01_01","LTR01_02","LTR01_03","LTR01_04","LTR01_05","LTR01_06",
    "LTR02_01","LTR02_02","LTR02_03","LTR02_04","LTR02_05",
    "LTS01_01","LTS01_03","LTS01_04","LTS01_05","LTS01_06","LTS01_07","LTS01_08","LTS01_09","LTS01_10",
    "LTN01_01","LTN01_02","LTN01_03","LTN01_04","LTN01_06","LTN01_07","LTN01_08",
    "Agr05_01","Agr05_02","Agr05_04","Agr05_05","Agr05_06","Agr05_07",
    "HTI01_01","HTI01_02","HTI01_03","HTI01_04","HTI01_07","HTI01_08","HTI01_09","HTI01_10",
    "HTI01_13","HTI01_14","HTI01_15","HTI01_16","HTI01_17",
    "LTI01_01","LTI01_02","LTI01_03","LTI01_05",
    "FHI01_01","FHI01_02","FHI01_03","FHI01_04","FHI01_05","FHI01_06",
    "RFP01_01","RFP02_01","RFP03_01","RFP03_02","RFP03_03","RFP03_04","RFP03_05","RFP03_06",
    "RFP03_13","RFP04_01","RFP05_01","RFP06_01",
    "RFB01_01","RFB01_02","RFB01_03","RFB02_01","RFB02_02","RFB03_01","RFB03_03","RFB03_04",
    "RFS01_01","RFS01_02","RFS02_01","RFS02_02","RFS02_03","RFS02_04",
    "RFS03_01","RFS03_05","RFS03_02","RFS03_04","RFS04_01","RFS04_02",
    "Gas01_02","Gas01_04","Gas03_01","Gas04_01","Gas04_02","Gas04_03","Gas04_04","Gas05_01",
    "Hyd01_01","Hyd01_02","Hyd01_04","Hyd01_05","Hyd01_06","Hyd04_01","Hyd04_02",
    "Amm01_01","Amm01_05","Emi01_04",
    "IFG01_01","IFG01_02","IFH01_01","IFH01_02","IFC01_01","IFP01_01","IFL01_01",
    "WAI01_01","WAI01_02","WAI01_03","WAI01_04","WAI01_05",
]

# Extra single-substring stock caps (each is its own AIMMS constraint —
# Comulative_Interconnection_*, Comulative_HVNS, CCUS_*, Imported_*,
# Ammonia_eSMR, Ammonia_wCCUS).
const _MR_EXTRA_FIRST_STOCK = String[
    "Amm01_08","Amm01_02","Gas01_03","Amm01_07","Hyd03_01",
    "Emi01_03","Emi01_02","PNL04_02","PNL04_01","PEU01_03",
]

# AIMMS seed candidates for use caps (Power_to_EU = PEU01_03,
# Power_from_EU = PNL04_01). A seed still needs repeated regional members with
# the exact same positive cap; it is not a bypass around the data rule.
const _MR_FIRST_USE = String["PEU01_03", "PNL04_01"]

# Comulative_Nuclear: LHS = tech_name(t) contains "Nuclear" AND not "Borssele".
const _MR_NUCLEAR_PIVOT     = "PNL01_11"
const _MR_NUCLEAR_NAME_INCL = "Nuclear"
const _MR_NUCLEAR_NAME_EXCL = "Borssele"

# Comulative_Geothermal: per-tech pivot with substring "LTN01_05"; LHS is the
# sum of stocks of techs in the SAME cluster prefix whose name contains any
# of these substrings.
const _MR_GEO_PIVOT  = "LTN01_05"
const _MR_GEO_FAMILY = String["LTN01_05", "Agr05_03", "LTI01_04"]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# AIMMS `ClusterPrefix(t) = SubString(FormatString("%e", t), 1,
# FindString(FormatString("%e", t), "_") - 1)` — the substring before the
# first underscore. For tech IDs of form `CL{n}_{TechFamily}_...` this is the
# regional cluster code (`CL5`, `CL6`, …).
@inline _mr_cluster_prefix(t::Symbol) = first(split(String(t), '_'; limit = 2))

@inline _mr_id_contains(t::Symbol, sub::AbstractString) = occursin(sub, String(t))

function _mr_regional_family(t::Symbol)
    m = match(r"^CL\d+_(.+)$", String(t))
    return m === nothing ? nothing : String(m.captures[1])
end

function _mr_members_with_substring(techs::Vector{Symbol}, sub::AbstractString)
    return [t for t in techs if _mr_id_contains(t, sub)]
end

function _mr_exact_cap_groups(sets::ModelSets,
                              cap_dict::Dict{Tuple{Symbol,Int},Float64},
                              periods::Vector{Int};
                              balancers_only::Bool = false,
                              seed_families::Vector{String} = String[])
    allowed = balancers_only ? Set(sets.tech_balancers) : Set(sets.technologies)
    members_by_family = Dict{String, Vector{Symbol}}()
    candidate_families = Set(seed_families)

    for t in sets.technologies
        t in allowed || continue
        family = _mr_regional_family(t)
        family === nothing && continue
        push!(get!(members_by_family, family, Symbol[]), t)
        push!(candidate_families, family)
    end

    groups = NamedTuple{(:family, :ps, :members, :cap), Tuple{String, Int, Vector{Symbol}, Float64}}[]
    for family in sort!(collect(candidate_families))
        members = get(members_by_family, family, Symbol[])
        length(members) > 1 || continue
        for ps in periods
            cap = get(cap_dict, (first(members), ps), 0.0)
            cap > 0 || continue
            all(get(cap_dict, (t, ps), 0.0) == cap for t in members) || continue
            push!(groups, (family = family, ps = ps, members = members, cap = cap))
        end
    end
    return groups
end

# AIMMS `First({ t | techStock_max(t, p) and FindString(...) > 0 })` —
# returns the first tech in iteration order whose name contains `sub` AND
# has a populated, positive entry in `cap_dict[(t, ps)]`. Returns `nothing`
# if no such carrier exists (constraint is then skipped, matching AIMMS
# semantics where `techStock_max(<empty>, p)` evaluates to 0 → degenerate).
function _mr_first_carrier(techs::Vector{Symbol}, sub::AbstractString,
                            cap_dict::Dict{Tuple{Symbol,Int},Float64}, ps::Int)
    @inbounds for t in techs
        if _mr_id_contains(t, sub) && haskey(cap_dict, (t, ps)) && cap_dict[(t, ps)] > 0
            return t
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Komar AIMMS data convention for techStock_min / techStock_max
# ----------------------------------------------------------------------------
"""
    apply_multi_region_data_convention!(md::ModelData) -> Nothing

Apply the AIMMS Komar (multi-region) sentinel convention to `techStock_min`
and `techStock_max`. No-op when `:multi_region ∉ md.params.extensions`.

In Komar AIMMS the relevant constraints are:

    Constraint maxStock_constraints {
        IndexDomain: (t,ps);
        Definition: (techStock_max(t,ps) <> -1) * (techStock(t,ps) - techStock_max(t,ps)) <= 0;
    }
    Constraint minStock_constraints {
        IndexDomain: (t,ps);
        Definition: (techStock_min(t,ps) <> -1) * (techStock(t,ps) - techStock_min(t,ps)) >= 0;
    }

so blank cells (default to 0 in AIMMS) impose `≤ 0` / `≥ 0`, and `-1`
disables the constraint. Julia's core convention treats blank as
"no constraint" and explicit 0 as "enforce 0", which matches the *new*
AIMMS semantics but **not** the Komar reference workbook.

When the multi-region extension is on, we coerce the dicts to the Komar
convention by:

  1. Removing entries equal to `-1.0` (sentinel for "no constraint").
  2. Filling missing `(t, ps)` cells with `0.0` over `s.technologies × s.periods_solve`.

`techUse_min` / `techUse_max` are intentionally untouched: their AIMMS
constraints (`min_techUse_constraint`, `max_techUse_constraint`) use the
`<>0` rule in both Komar and the current model, so blank/0 already mean
"no constraint" in both.

The function is idempotent.
"""
function apply_multi_region_data_convention!(md::ModelData)
    :multi_region in md.params.extensions || return nothing
    s = md.sets
    p = md.params
    isempty(s.technologies) && return nothing
    periods = isempty(s.periods_solve) ? collect(s.periods) : s.periods_solve
    isempty(periods) && return nothing

    for (dict, name) in ((p.techStock_max, "techStock_max"),
                         (p.techStock_min, "techStock_min"))
        # Step 1: drop -1 sentinels.
        n_dropped = 0
        for (k, v) in collect(dict)
            if v == -1.0
                delete!(dict, k)
                n_dropped += 1
            end
        end
        # Step 2: fill missing (t, ps) with 0.0 over the active grid.
        n_filled = 0
        for t in s.technologies, ps in periods
            key = (t, ps)
            if !haskey(dict, key)
                dict[key] = 0.0
                n_filled += 1
            end
        end
        @info "Komar convention applied" parameter = name dropped_sentinels = n_dropped filled_zeros = n_filled total = length(dict)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Main entry — applies all active Komar constraints
# ----------------------------------------------------------------------------
"""
    apply_multi_region!(m, vars, md; mode::Symbol=:fh) -> Int

Add the regional-cap constraints (AIMMS Komar section, active subset).
Returns the number of constraints added. Reads:

  • `vars.techStock`              — annual stock variable (every build path)
  • `vars.tech_use`               — annual use variable
  • `md.params.techStock_max`     — Dict{(Symbol,Int) → Float64}
  • `md.params.techUse_max`       — Dict{(Symbol,Int) → Float64}
  • `md.params.tech_name`         — Dict{Symbol → String} (for Nuclear filter)
  • `md.sets.technologies`        — Vector{Symbol}
  • `md.sets.periods_solve`       — Vector{Int}

The function is mode-agnostic: it always references annual-level variables
which exist in every build path (`:annual`, `:fh`, `:ts`).
"""
function apply_multi_region!(m::JuMP.Model, vars, md::ModelData; mode::Symbol = :fh)
    sets    = md.sets
    params  = md.params
    techs   = sets.technologies
    periods = sets.periods_solve

    techStock = vars.techStock
    tech_use  = vars.tech_use

    n_added = 0

    # --- Stock caps with exact repeated regional caps -------------------------
    for group in _mr_exact_cap_groups(sets, params.techStock_max, periods;
                                      seed_families = vcat(_MR_S_TECH_GROUPS, _MR_EXTRA_FIRST_STOCK))
        @constraint(m, sum(techStock[t, group.ps] for t in group.members) <= group.cap,
                    base_name = "mr_stockCap_$(group.family)_$(group.ps)")
        n_added += 1
    end

    # --- Use caps with exact repeated regional caps ---------------------------
    for group in _mr_exact_cap_groups(sets, params.techUse_max, periods;
                                      balancers_only = true,
                                      seed_families = _MR_FIRST_USE)
        @constraint(m, sum(tech_use[t, group.ps] for t in group.members) <= group.cap,
                    base_name = "mr_useCap_$(group.family)_$(group.ps)")
        n_added += 1
    end

    # --- Comulative_Nuclear: tech_name "Nuclear" but not "Borssele" -----------
    nuclear_techs = Symbol[]
    if !isempty(params.tech_name)
        for t in techs
            nm = get(params.tech_name, t, "")
            if occursin(_MR_NUCLEAR_NAME_INCL, nm) && !occursin(_MR_NUCLEAR_NAME_EXCL, nm)
                push!(nuclear_techs, t)
            end
        end
    end
    if !isempty(nuclear_techs)
        for ps in periods
            carrier = _mr_first_carrier(techs, _MR_NUCLEAR_PIVOT, params.techStock_max, ps)
            carrier === nothing && continue
            cap = params.techStock_max[(carrier, ps)]
            @constraint(m, sum(techStock[t, ps] for t in nuclear_techs) <= cap,
                        base_name = "mr_nuclear_$(ps)")
            n_added += 1
        end
    end

    # --- Comulative_Geothermal: per-tech regional cap on geothermal family ---
    geo_pivots = [t for t in techs if _mr_id_contains(t, _MR_GEO_PIVOT)]
    if !isempty(geo_pivots)
        # Pre-bucket members by cluster prefix for O(N) lookup
        prefix_members = Dict{String, Vector{Symbol}}()
        for it in techs
            if any(s -> _mr_id_contains(it, s), _MR_GEO_FAMILY)
                push!(get!(prefix_members, _mr_cluster_prefix(it), Symbol[]), it)
            end
        end
        for t in geo_pivots
            members = get(prefix_members, _mr_cluster_prefix(t), Symbol[])
            isempty(members) && continue
            for ps in periods
                haskey(params.techStock_max, (t, ps)) || continue
                cap = params.techStock_max[(t, ps)]
                cap > 0 || continue
                @constraint(m, sum(techStock[it, ps] for it in members) <= cap,
                            base_name = "mr_geo_$(t)_$(ps)")
                n_added += 1
            end
        end
    end

    return n_added
end
