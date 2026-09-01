"""
Core types for IESA-Opt.jl.

These mirror the IESA-Opt 1.0 Set / Parameter / Variable / MathematicalProgram
universe in idiomatic Julia structs.

- `ModelSets`: every IESA-Opt 1.0 Set / index becomes a field (Vector{Int} or Vector{Symbol})
- `ModelParams`: every IESA-Opt 1.0 Parameter becomes a field
  (sparse: `Dict{Tuple{...}, Float64}` for indexed; scalar for unindexed)
- `ModelData`: bundles sets+params for passing into model builders
- `RunResult`: post-solve summary (objective, status, solve time, extracted values)
"""

# ----------------------------------------------------------------------------
# ModelSets — IESA-Opt 1.0 Set declarations + computed subsets
# ----------------------------------------------------------------------------

"""
    ModelSets

All index sets used by the model.

Temporal sets are populated by `read_data` + `derive_sets!`. Tech / activity
subsets are computed from full data by `derive_sets!` (filters on properties
loaded from Excel).

The active `hours` field depends on time-resolution mode (FH / TS / AS):
- FH (full hourly): `hours = 1..(8760 / slotsRatio)` if `hoursPer_day < 24`,
  else `hours = 1..8760`
- TS: `hours` and `hours_cluster` both populated; the model builder uses
  `hours_cluster` for TS constraints
- AS (adaptive segmentation): `hours = 1..n_adaptiveSegments`
"""
mutable struct ModelSets
    # ------------------------------------------------------------------ Temporal
    hours::Vector{Int}                  # active hour set (FH or AS-segmented)
    hours_orig::Vector{Int}             # baseline 1..8760
    hours_cluster::Vector{Int}          # 1..(n_repDays*hoursPer_day_cluster), TS only
    hours_inDay::Vector{Int}            # 1..hoursPer_day
    hours_inDay_cluster::Vector{Int}    # 1..hoursPer_day_cluster (==24 typically)
    repDays::Vector{Int}                # 1..n_repDays
    days::Vector{Int}                   # 1..365
    weeks::Vector{Int}                  # 1..53
    months::Vector{Int}                 # 1..12
    seasons::Vector{Int}                # 1..4
    semesters::Vector{Int}              # 1..2
    periods::Vector{Int}                # e.g. [2022, 2025, 2030, 2035, 2040, 2045, 2050]
    periods_solve::Vector{Int}          # selected for active solve
    q_hourWindow::Vector{Int}           # 4-h windows (FH)
    q_hourWindow_cluster::Vector{Int}   # 4-h windows (TS)
    r_dayWindow::Vector{Int}            # 3-day windows (FH)

    # ------------------------------------------------------------- Tech / activity / node
    technologies::Vector{Symbol}
    activities::Vector{Symbol}
    nodes::Vector{Symbol}
    nodes_IEM::Vector{Symbol}           # interconnected EU market subset
    profile_type::Vector{Symbol}
    process_type::Vector{Symbol}
    activity_type::Vector{Symbol}
    flexibility_type::Vector{Symbol}
    dispatch_type::Vector{Symbol}
    range_type::Vector{Symbol}
    profile_typeRead::Vector{Symbol}    # profile types as read from XLSX (pre-clustering)
    activities_original::Vector{Symbol} # pre-grouping (IESA-Opt 1.0 Activities_original)
    activities_group::Vector{Symbol}    # grouped (IESA-Opt 1.0 Activities_group)
    sectors::Vector{Symbol}             # IESA-Opt 1.0 Sectors
    sectors_kev::Vector{Symbol}         # IESA-Opt 1.0 Sectors_kev (subset)
    node_names::Vector{Symbol}          # IESA-Opt 1.0 NodeName_per_node domain
    energy_labels::Vector{Symbol}       # IESA-Opt 1.0 EnergyLabels
    renewables::Vector{Symbol}          # IESA-Opt 1.0 Renewables (subset of tech)
    CHP_polyFacets::Vector{Symbol}      # IESA-Opt 1.0 CHP_polyFacet domain

    # ------------------------------------------------------------- Tech subsets (derived)
    tech_balancers::Vector{Symbol}
    tech_hourlyDispatch::Vector{Symbol}
    tech_dailyDispatch::Vector{Symbol}
    tech_flexible::Vector{Symbol}
    tech_fStorage::Vector{Symbol}        # IESA-Opt 1.0 tech_fStorage  (index tfb)
    tech_fEV::Vector{Symbol}             # IESA-Opt 1.0 tech_fEV       (index tfv)
    tech_fEVcharging::Vector{Symbol}     # IESA-Opt 1.0 tech_fEVcharging (index tfvc) — 'EV smart charge'
    tech_fEVgrid::Vector{Symbol}         # IESA-Opt 1.0 tech_fEVgrid     (index tfvg) — 'EV P-to-Grid'
    tech_fDRshifting::Vector{Symbol}     # IESA-Opt 1.0 tech_fDRshifting (index tfs)  — 'DR shifting'
    tech_fBEshifting::Vector{Symbol}     # IESA-Opt 1.0 tech_fBEshifting (index tfe)  — 'BE shifting'
    tech_fWithBattery::Vector{Symbol}   # storage + EV (IESA-Opt 1.0 index tfwb)
    tech_flexH::Vector{Symbol}          # hourly-range flex
    tech_flexD::Vector{Symbol}          # daily-range flex
    tech_flexR::Vector{Symbol}          # 3-day-range flex
    tech_flexW::Vector{Symbol}          # weekly-range flex
    tech_flexM::Vector{Symbol}          # monthly
    tech_flexS::Vector{Symbol}          # seasonal
    tech_flexB::Vector{Symbol}          # 6-month (semester)
    tech_flexY::Vector{Symbol}          # IESA-Opt 1.0 tech_flexY  (index tf_y) — '1 year [y]'
    tech_flexLT::Vector{Symbol}         # long-term
    tech_shedding::Vector{Symbol}
    tech_shedH::Vector{Symbol}
    tech_shedW::Vector{Symbol}          # multi-day shed budget
    tech_reservoir::Vector{Symbol}      # PHS
    tech_hourlyCHPflex::Vector{Symbol}
    tech_hourlyCHPflexH::Vector{Symbol}  # IESA-Opt 1.0 tech_hourlyCHPflexH (tk_h) — CHP_range='1 hour [h]'
    tech_hourlyCHPflexD::Vector{Symbol}  # IESA-Opt 1.0 tech_hourlyCHPflexD (tk_d) — CHP_range='1 day [d]'
    tech_hourlyCHPflexW::Vector{Symbol}  # IESA-Opt 1.0 tech_hourlyCHPflexW (tk_w) — CHP_range='1 week [w]'
    tech_gasBuffer::Vector{Symbol}
    tech_emission::Vector{Symbol}
    tech_infra::Vector{Symbol}          # infrastructure (XC, pipelines, grids)
    tech_infraH::Vector{Symbol}         # infra with infra_range == '1 hour [h]' (IESA-Opt 1.0 line 5104)
    tech_infraD::Vector{Symbol}         # infra with infra_range == '1 day [d]'  (IESA-Opt 1.0 line 5111)
    tech_materialConversion::Vector{Symbol}  # tech whose main activity is material conversion

    # ------------------------------------------------------------- Activity subsets (infra-derived)
    act_infraH::Vector{Symbol}          # IESA-Opt 1.0 line 5118: derived as ∪_{ti∈tech_infraH} infra_activity(ti)
    act_infraD::Vector{Symbol}          # IESA-Opt 1.0 line 5122: derived as ∪_{ti∈tech_infraD} infra_activity(ti)

    # ------------------------------------------------------------- Activity subsets (derived)
    activities_solve::Vector{Symbol}
    activities_hour::Vector{Symbol}     # hourly-dispatched
    activities_day::Vector{Symbol}      # daily-dispatched
    activities_indirect::Vector{Symbol} # 'hourly-indirect' (e.g. EV)
    activities_energy::Vector{Symbol}
    activities_fixEnergy::Vector{Symbol}
    activities_balance::Vector{Symbol}
    activities_driver::Vector{Symbol}
    activities_year::Vector{Symbol}
    activities_target::Vector{Symbol}   # CO2 budget targets (emissionTarget_bin == 1)
    activities_target_FeedStocks::Vector{Symbol}  # 'CO2 Air Feedstock end of lifetime' originals
    activities_target_Bunkers::Vector{Symbol}     # 'CO2 Air Int. Transport' originals
    activities_materialConversion::Vector{Symbol}
    activities_emission::Vector{Symbol} # activityType == 'Emission' AND emissionTarget_bin <> 1 (IESA-Opt 1.0 line 2225)
    activities_emissionFix::Vector{Symbol}    # IESA-Opt 1.0 activities_emissionFix (index acf) — activityType == 'Emission'
    activities_energyNonFixed::Vector{Symbol} # activityType == 'Energy' only (IESA-Opt 1.0 line 2208)
    activities_emissionReport::Vector{Symbol} # activityType == 'EmissionReport' (IESA-Opt 1.0 line 2266)
    activities_credits::Vector{Symbol}        # activityType == 'Credits' (IESA-Opt 1.0 line 2273)
end

"""
    ModelSets()

Empty constructor: every field is an empty vector. To be filled by
`read_data` + `derive_sets!`.
"""
function ModelSets()
    ModelSets(
        # temporal (16)
        Int[], Int[], Int[], Int[], Int[],
        Int[], Int[], Int[], Int[], Int[], Int[],
        Int[], Int[], Int[], Int[], Int[],
        # tech/activity/node (19)
        Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
        # tech subsets (34 — added tech_fEVcharging/Grid, tech_fDRshifting,
        #                          tech_fBEshifting, tech_flexY, tech_hourlyCHPflexH/D/W)
        Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[],
        # infra-derived activity subsets (2 — act_infraH, act_infraD)
        Symbol[], Symbol[],
        # activity subsets (18 — added activities_emissionFix)
        Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
        Symbol[], Symbol[], Symbol[], Symbol[], Symbol[], Symbol[],
    )
end

# ----------------------------------------------------------------------------
# ModelParams — IESA-Opt 1.0 Parameter declarations
# ----------------------------------------------------------------------------

"""
    ModelParams

All numerical parameters (loaded + derived).

Loaded parameters come from XLSX. Derived parameters (CRF, decay factors,
`InvMat_lifeTime`, etc.) are computed by `compute_derived_params!`.

Sparse parameters use `Dict{Tuple{...}, Float64}`; access via
`get(params.inv_cost, (tech, 2050), 0.0)`.
"""
Base.@kwdef mutable struct ModelParams
    # -------------------------------------------------------- Scalars / config
    scenario_description::String = ""
    base_year::Int = 2022
    n_repDays::Int = 15
    hoursPer_day::Int = 24
    hoursPer_day_cluster::Int = 24
    hoursPer_quarter_cluster::Int = 4
    daysPer_range::Int = 3
    use_clustering::Bool = true
    use_adaptive::Bool = false

    # Global scalars loaded from Parameters sheet
    XC_TransmissionLoss_global::Float64 = 0.0
    baseload_treshold::Float64 = 0.0
    shedding_inLoad::Float64 = 0.0
    social_discount_rate::Float64 = 0.04
    electricity_trade_ratio::Float64 = 0.0
    electricity_trade_volume::Float64 = 0.0
    ActiveConstraintSet::String = ""

    # Numeric stability
    numeric_epsilon::Float64 = 1e-9

    # Free-form solver attribute bag (overrides default_gurobi_attributes when set)
    solver_options::Dict{String,Any} = Dict{String,Any}()

    # Clustering control
    clustering_approach::Symbol = :kmeans_avg
    clustering_seed::Int = 42
    ts_extremePeriods::Bool = true
    ts_extremeDays_count::Int = 5
    ts_capacityProfile_autoMode::Bool = true
    ts_capacityProfile_autoFloor::Float64 = 0.23
    ts_capacityProfile_autoCap::Float64 = 1.00
    ts_capacityProfile_envelopeMode::Int = 0          # 0=MIN, 1=μ-kσ, 2=MIN↔MEAN
    ts_capacityProfile_minBlend::Float64 = 0.0
    ts_capacityProfile_percentileK::Float64 = 1.282
    ts_capacityProfile_percentileQ::Float64 = 0.10
    ts_capacityProfile_autoFloor_effective::Float64 = 0.23  # precomputed (rd-linear or static)
    ts_boundaryRamping::Bool = true
    ts_extremeWeight::Float64 = 1.0
    dayMix_softness::Float64 = 0.0
    dayMix_weightType::Symbol = :auto
    ts_dayMix_PGD_maxIter::Int = 80
    ts_dayMix_PGD_lr::Float64 = 0.05
    ts_dayMix_PGD_tol::Float64 = 1e-6
    ts_hullClustering_sinkhornFix::Bool = true

    # External clustering override (load mapDay_repDay from IESA-Opt 1.0-exported parquet
    # to bypass Julia's k-means and extreme-day promotion). Used by Julia↔IESA-Opt 1.0
    # validation runs to remove the clustering algorithm as a source of divergence.
    external_clusterMap_path::String = ""

    # Tolerances (mirror IESA-Opt 1.0 defaults)
    Feasibility_tolerance::Float64 = 1e-7
    Optimality_tolerance::Float64 = 1e-7
    BarrierConvergence_tolerance::Float64 = 1e-7

    # ---------------------------------------------------------- Cost data
    inv_cost::Dict{Tuple{Symbol,Int},Float64}  = Dict()
    fom_cost::Dict{Tuple{Symbol,Int},Float64}  = Dict()
    vom_cost::Dict{Tuple{Symbol,Int},Float64}  = Dict()
    salvage_cost::Dict{Tuple{Symbol,Int},Float64} = Dict()
    Salvage_value::Dict{Symbol,Float64}        = Dict()   # raw per-tech salvage fraction from XLSX
    retrofit_cost::Dict{Tuple{Symbol,Symbol,Int},Float64} = Dict()  # (it, jt, p) retrofit cost (broadcast across periods from XLSX)
    WACC::Dict{Symbol,Float64}                 = Dict()
    economic_lifetime::Dict{Symbol,Float64}    = Dict()
    technical_lifetime::Dict{Symbol,Float64}   = Dict()
    construction_time::Dict{Symbol,Float64}    = Dict()
    cap2act::Dict{Symbol,Float64}              = Dict()
    ramping::Dict{Symbol,Float64}              = Dict()

    # Derived financial
    CRF::Dict{Symbol,Float64}                  = Dict()  # capital recovery factor
    social_discount_factor::Dict{Int,Float64}  = Dict()
    InvMat_lifeTime::Dict{Tuple{Symbol,Int,Int},Float64}        = Dict()  # (t, inv_p, solve_p) → {0,1}
    decomMat_NewInv::Dict{Tuple{Symbol,Int,Int},Float64}        = Dict()
    period_span::Dict{Int,Float64}                              = Dict()  # IESA-Opt 1.0: (val(pss)-val(prev))/5  — in 5-year units, real
    transition_interval::Dict{Int,Float64}                      = Dict()  # IESA-Opt 1.0 scalar = last-first+10; key=0 holds the scalar

    # ----------------------------------------------------- Tech metadata
    tech_name::Dict{Symbol,String}             = Dict()
    tech_units::Dict{Symbol,Symbol}            = Dict()
    tech_sector::Dict{Symbol,Symbol}           = Dict()
    tech_sector_kev::Dict{Symbol,Symbol}       = Dict()
    tech_subsector::Dict{Symbol,Symbol}        = Dict()
    tech_category::Dict{Symbol,Symbol}         = Dict()
    processType_tech::Dict{Symbol,Symbol}      = Dict()
    profileType_tech::Dict{Symbol,Symbol}      = Dict()    # resolved (post indirect handling)
    profileType_techRead::Dict{Symbol,Symbol}  = Dict()    # raw from Technologies sheet
    profileType_EVuse::Dict{Symbol,Symbol}     = Dict()    # EV use profile mapping
    flexibilityType_tech::Dict{Symbol,Symbol}  = Dict()
    nodePer_tech::Dict{Symbol,Symbol}          = Dict()
    nodePer_techBal::Dict{Symbol,Symbol}       = Dict()    # node per tech_balancer (derived)
    nodePer_techInfra::Dict{Symbol,Symbol}     = Dict()    # node per tech_infra (derived)
    activityPer_techOrig::Dict{Symbol,Symbol}  = Dict()    # main activity per tech (pre-grouping)
    activityPer_tech::Dict{Symbol,Symbol}      = Dict()    # resolved (post-grouping)
    # IESA-Opt 1.0 line 1973: tech_activity(t) = if t∈tech_balancers then activityPer_tech(t) elseif t∈tech_infra then infra_activity(t)
    tech_activity::Dict{Symbol,Symbol}         = Dict()    # unified activity per tech (derived)
    is_renewable::Dict{Symbol,Float64}         = Dict()
    IEM_sector::Dict{Symbol,Symbol}            = Dict()
    namePer_node::Dict{Symbol,Symbol}          = Dict()
    IEM_node::Dict{Symbol,Symbol}              = Dict()

    # ----------------------------------------------------- Activity metadata
    act_units::Dict{Symbol,Symbol}             = Dict()
    activityType_act::Dict{Symbol,Symbol}      = Dict()
    dispatchType_act::Dict{Symbol,Symbol}      = Dict()
    nodePer_act::Dict{Symbol,Symbol}           = Dict()
    labelPer_act::Dict{Symbol,Symbol}          = Dict()
    emissionTarget_bin::Dict{Symbol,Symbol}    = Dict()
    actChange_maxOrig::Dict{Symbol,Float64}    = Dict()                  # pre-grouping
    actChange_max_act::Dict{Symbol,Float64}    = Dict()                  # post-grouping (per activity)
    act_to_group::Dict{Symbol,Symbol}          = Dict()                  # orig activity → group activity
    actSolvePer_actOrig::Dict{Symbol,Symbol}   = Dict()                  # orig → resolved
    activities_netVolumesOrig::Dict{Tuple{Symbol,Int},Float64} = Dict()  # pre-grouping
    activities_netVolumes::Dict{Tuple{Symbol,Int},Float64}     = Dict()  # post-grouping (a, p) → demand level
    activity_balancesRef::Dict{Tuple{Symbol,Symbol,Int},Float64} = Dict() # raw from EnergyBalance sheet
    activity_balances::Dict{Tuple{Symbol,Symbol,Int},Float64}    = Dict() # post EffImprov adjustment
    activity_EffImprov::Dict{Tuple{Symbol,Symbol,Int},Float64}   = Dict() # efficiency learning per period
    feedstockUse_techOrig::Dict{Tuple{Symbol,Symbol},Float64}    = Dict() # (tech, activity) → fraction

    # ----------------------------------------------------- Profiles
    monthPer_hourOrig::Dict{Int,Int}                          = Dict()  # h_orig → month
    hourly_profilesReadOrig::Dict{Tuple{Int,Symbol},Float64}  = Dict()  # raw 8760 read from XLSX
    hourly_profilesRead::Dict{Tuple{Int,Symbol},Float64}      = Dict()  # post FH aggregation
    hourly_profiles_orig::Dict{Tuple{Int,Symbol},Float64}     = Dict()  # (h_orig=1..8760, yp) → value
    hourly_profiles::Dict{Tuple{Int,Symbol},Float64}          = Dict()  # current FH resolution (handles indirect)
    hourly_profiles_cluster::Dict{Tuple{Int,Symbol},Float64}  = Dict()  # (hc, yp) cluster-resolved
    hourly_profiles_clusterResolved::Dict{Tuple{Int,Symbol},Float64} = Dict()  # (hc, yp)
    hourly_profiles_clusterMedoid::Dict{Tuple{Int,Symbol},Float64}   = Dict()
    hourly_profiles_clusterMin::Dict{Tuple{Int,Symbol},Float64}      = Dict()
    hourly_profiles_clusterMean::Dict{Tuple{Int,Symbol},Float64}     = Dict()
    hourly_profiles_clusterStd::Dict{Tuple{Int,Symbol},Float64}      = Dict()
    hourly_profiles_clusterPercentile::Dict{Tuple{Int,Symbol},Float64} = Dict()  # k-th percentile per (hc, yp)
    hourly_profiles_clusterRankPct::Dict{Tuple{Int,Symbol},Float64}    = Dict()  # rank-interpolation envelope
    hourly_profiles_clusterAutoBlend::Dict{Tuple{Int,Symbol},Float64} = Dict()  # α per (hc, yp)
    hourly_profiles_clusterCapBound::Dict{Tuple{Int,Symbol},Float64}  = Dict()
    interconnectedHourly_pricesOrig::Dict{Tuple{Int,Symbol,Int},Float64} = Dict()  # raw 8760 read
    interconnectedHourly_prices::Dict{Tuple{Int,Symbol,Int},Float64}  = Dict()  # (h, ain, p)
    interconnectedHourly_prices_cluster::Dict{Tuple{Int,Symbol,Int},Float64} = Dict()  # (hc, ain, p)

    # ----------------------------------------------------- Stock & emissions
    existingStock::Dict{Tuple{Symbol,Int},Float64}            = Dict()
    techStock_exist::Dict{Symbol,Float64}                     = Dict()  # raw per-tech existing stock from XLSX (col BN)
    techStock_min::Dict{Tuple{Symbol,Int},Float64}            = Dict()
    techStock_max::Dict{Tuple{Symbol,Int},Float64}            = Dict()
    techUse_min::Dict{Tuple{Symbol,Int},Float64}              = Dict()
    techUse_max::Dict{Tuple{Symbol,Int},Float64}              = Dict()
    techChange_max::Dict{Symbol,Float64}                      = Dict()
    actChange_max::Dict{Symbol,Float64}                       = Dict()
    no_new_invest::Dict{Tuple{Symbol,Int},Bool}               = Dict()
    no_eco_decom::Dict{Tuple{Symbol,Int},Bool}                = Dict()
    decom_planned::Dict{Tuple{Symbol,Int},Float64}            = Dict()
    decom_plannedSel::Dict{Tuple{Symbol,Int},Float64}         = Dict()  # filtered to periods_solve
    retrofit_relations::Dict{Tuple{Symbol,Symbol},Bool}       = Dict()
    # ---- Derived from retrofit_relations by `derive_sets!` -----------------
    # Sparse retrofit support: AIMMS materializes only `retrofit_relations(it,jt)==true`
    # entries before sending to Gurobi, but a dense Julia (it×jt×ps) variable
    # creates ~|tech|^2 redundant cols/rows that Gurobi's presolve must remove.
    # We pre-compute the active pair list and the per-tech in/out adjacency to
    # build a sparse `retrofitting` variable.
    retrofit_pairs::Vector{Tuple{Symbol,Symbol}}              = Tuple{Symbol,Symbol}[]
    retrofit_in_by_tech::Dict{Symbol,Vector{Symbol}}          = Dict()  # t -> [it : (it,t) is a pair]
    retrofit_out_by_tech::Dict{Symbol,Vector{Symbol}}         = Dict()  # t -> [jt : (t,jt) is a pair]
    cumulative_CO2storage::Dict{Symbol,Float64}               = Dict()  # per node
    emissionTargetAir::Dict{Tuple{Symbol,Int},Float64}        = Dict()
    emissionTargetAll::Dict{Tuple{Symbol,Int},Float64}        = Dict()
    emissionTargetBunker::Dict{Tuple{Symbol,Int},Float64}     = Dict()
    emissionTargetFS::Dict{Tuple{Symbol,Int},Float64}         = Dict()
    emissionTarget_inclScope3andFuelex::Dict{Int,Float64}     = Dict()  # derived = NL row of emissionTargetAll
    emissionTarget_cum::Dict{Symbol,Float64}                  = Dict()  # per-node cumulative cap
    CO2_cumulative_budget::Dict{Symbol,Float64}               = Dict()
    max_CO2_storage::Dict{Symbol,Float64}                     = Dict()

    # ----------------------------------------------------- Flex parameters
    flex_capacity::Dict{Tuple{Symbol,Int},Float64}            = Dict()   # derived per (t, p)
    flex_capacity_pct::Dict{Symbol,Float64}                   = Dict()   # raw % read from Technologies sheet (col BA)
    flex_storage::Dict{Symbol,Float64}                        = Dict()   # hours
    flex_range::Dict{Symbol,Symbol}                           = Dict()
    flex_loss_charge::Dict{Symbol,Float64}                    = Dict()
    flex_loss_discharge_eff::Dict{Symbol,Float64}             = Dict()
    flex_losses_legacy::Dict{Symbol,Float64}                  = Dict()   # raw combined loss from sheet (col BD); pre-split
    # Pre-"Storage upgraded" AIMMS applied the loss once (closure UP coefficient 1);
    # the current model applies it on both charge and discharge.
    flex_legacy_roundtrip::Bool                               = false
    flex_standing_loss::Dict{Symbol,Float64}                  = Dict()
    flex_standing_loss_effective::Dict{Symbol,Float64}        = Dict()
    flex_storage_decay_factor::Dict{Tuple{Symbol,Int},Float64} = Dict()  # pre-computed (1-loss)^slice_width
    flex_nnLoad::Dict{Symbol,Float64}                         = Dict()
    flex_backlog_horizon_days::Dict{Symbol,Int}               = Dict()
    flex_activityOrig::Dict{Symbol,Symbol}                    = Dict()   # raw flex activity from sheet (pre-grouping)
    flex_activity::Dict{Symbol,Symbol}                        = Dict()
    avg_speed::Dict{Symbol,Float64}                           = Dict()
    avg_journey::Dict{Symbol,Float64}                         = Dict()

    # IESA-Opt 1.0 flex helper indicators / TS helpers
    is_BEshifting_tech::Dict{Symbol,Bool}                                       = Dict()  # IESA-Opt 1.0 line 3289 — indicator on tech_balancers (true if flexibilityType_tech=='BE shifting')
    cumulativeUP_DW_dQtfe_helper_TS::Dict{Tuple{Int,Symbol,Int},Float64}        = Dict()  # IESA-Opt 1.0 (hc,tfe,p) helper used by cumulativeUP/DW_dQtfe_TS
    cumulative_dQtfe_rhs_TS::Dict{Tuple{Int,Symbol,Int},Float64}                = Dict()  # IESA-Opt 1.0 (qc,tfe,p) RHS aggregated by quarter
    cumulativeS_dQtfv_helper1_TS::Dict{Tuple{Symbol,Int},Float64}               = Dict()  # IESA-Opt 1.0 (tfv,hc) profile-of-EVuse / avg_speed
    cumulativeS_dQtfv_helper2_TS::Dict{Tuple{Symbol,Int},Float64}               = Dict()  # IESA-Opt 1.0 (tfv,hc) currently always 0
    ev_min_soc_fraction_default::Float64                                        = 0.2     # IESA-Opt 1.0 line ~3978 default min SoC fraction
    ev_v2g_power_fraction_default::Float64                                      = 0.8     # IESA-Opt 1.0 line ~3982 default V2G discharge fraction

    # Shedding parameters
    shed_capacity::Dict{Tuple{Symbol,Int},Float64}            = Dict()
    shed_capacity_percentage::Dict{Symbol,Float64}            = Dict()  # raw % from sheet (vs (t,p) above)
    shed_volume::Dict{Symbol,Float64}                         = Dict()
    shed_penalty::Dict{Symbol,Float64}                        = Dict()
    shed_range::Dict{Symbol,Symbol}                           = Dict()
    shed_budget_horizon_days::Dict{Symbol,Int}                = Dict()

    # CHP parameters
    CHP_eta::Dict{Symbol,Float64}                             = Dict()
    CHP_eps::Dict{Tuple{Symbol,Int},Float64}                  = Dict()   # precomputed power/heat ratio
    CHP_range::Dict{Symbol,Symbol}                            = Dict()
    CHP_dev_use::Dict{Symbol,Float64}                         = Dict()
    CHP_dev_PtoH::Dict{Symbol,Float64}                        = Dict()
    CHP_polyFacetA::Dict{Tuple{Symbol,Symbol},Float64}        = Dict()   # (facet, tech)
    CHP_polyFacetB::Dict{Tuple{Symbol,Symbol},Float64}        = Dict()
    CHP_polyFacetRHS::Dict{Tuple{Symbol,Symbol},Float64}      = Dict()
    CHP_prodOrig::Dict{Symbol,Symbol}                         = Dict()   # tech → output activity (pre-grouping)
    CHP_fuelOrig::Dict{Symbol,Symbol}                         = Dict()   # tech → input activity (pre-grouping)
    CHP_prod::Dict{Symbol,Symbol}                             = Dict()
    CHP_fuel::Dict{Symbol,Symbol}                             = Dict()
    dP_electricity::Dict{Tuple{Symbol,Symbol},Float64}        = Dict()   # (tech, activity) → {0,1}
    dP_heat::Dict{Tuple{Symbol,Symbol},Float64}               = Dict()

    # Reservoir/PHS
    phs_capacity::Dict{Symbol,Float64}                        = Dict()
    phs_storage::Dict{Symbol,Float64}                         = Dict()  # IESA-Opt 1.0 phs_storage (PJ) — reservoir-level upper bound multiplier
    phs_Losses::Dict{Symbol,Float64}                          = Dict()
    pumphead_ratio::Dict{Symbol,Float64}                      = Dict()
    reservoir_capacity::Dict{Symbol,Float64}                  = Dict()

    # IESA-Opt 1.0 unit-conversion helper scalars (used in capacity*_dW_TS and capLevelUB_dW_TS)
    GWtoPJ_y::Float64                                         = 3.6 * 8760 / 1000   # IESA-Opt 1.0 GWtoPJ_y — 1 GW · 1 year → PJ
    GWhtoPJ::Float64                                          = 3.6 / 1000          # IESA-Opt 1.0 GWhtoPJ — 1 GWh → PJ

    # Gas buffer
    buffer_storage::Dict{Symbol,Float64}                      = Dict()
    bufferUP_capacity::Dict{Symbol,Float64}                   = Dict()
    bufferDW_capacity::Dict{Symbol,Float64}                   = Dict()
    buffer_activityOrig::Dict{Symbol,Symbol}                  = Dict()   # pre-grouping
    buffer_activity::Dict{Symbol,Symbol}                      = Dict()
    dB_daily::Dict{Tuple{Symbol,Symbol},Float64}              = Dict()

    # Infrastructure (XC, pipelines)
    infra_activityOrig::Dict{Symbol,Symbol}                   = Dict()   # pre-grouping
    infra_activity::Dict{Symbol,Symbol}                       = Dict()
    infra_range::Dict{Symbol,Symbol}                          = Dict()
    XC_TransmissionLoss::Dict{Symbol,Float64}                 = Dict()
    XC_linkPair::Dict{Symbol,Symbol}                          = Dict()   # for linked_*_XC

    # Adaptive Segmentation
    n_adaptiveSegments::Int                                   = 0
    as_segmentsPerDay::Dict{Int,Int}                          = Dict()   # d → segments
    as_segmentStart_inDay::Dict{Tuple{Int,Int},Int}           = Dict()   # (d, seg) → hour-in-day
    as_segmentWidth::Dict{Tuple{Int,Int},Float64}             = Dict()   # (d, seg) → duration
    as_segment_of_origHour::Dict{Int,Int}                     = Dict()   # h_orig → segment index

    # Misc
    dQ_hourly::Dict{Tuple{Symbol,Symbol},Float64}             = Dict()
    dS_hourly::Dict{Tuple{Symbol,Symbol},Float64}             = Dict()
    dW_hourly::Dict{Tuple{Symbol,Symbol},Float64}             = Dict()
    period_weight::Dict{Int,Float64}                          = Dict()
    p_epsilon::Float64                                        = 0.0

    # ----------------------------------------------------- Policy / regulatory targets
    # Hard-coded values from IESA-Opt 1.0 lines 20913-21015. Period-indexed in PJ or fraction.
    ReFuelEU_Aviation_eSAF_target::Dict{Int,Float64}          = Dict()
    ReFuelEU_Aviation_SAF_target::Dict{Int,Float64}           = Dict()
    FuelEU_Maritime_target::Dict{Int,Float64}                 = Dict()

    # ----------------------------------------------------- Index helpers (precomputed)
    dayPer_hour::Dict{Int,Int}                                = Dict()
    weekPer_hour::Dict{Int,Int}                               = Dict()
    monthPer_hour::Dict{Int,Int}                              = Dict()
    seasonPer_hour::Dict{Int,Int}                             = Dict()
    semesterPer_hour::Dict{Int,Int}                           = Dict()
    quarterPer_hour::Dict{Int,Int}                            = Dict()
    rangePer_hour::Dict{Int,Int}                              = Dict()
    rangePer_day::Dict{Int,Int}                               = Dict()
    weekPer_day::Dict{Int,Int}                                = Dict()
    monthPer_day::Dict{Int,Int}                               = Dict()
    seasonPer_day::Dict{Int,Int}                              = Dict()
    semesterPer_day::Dict{Int,Int}                            = Dict()
    hoursindayPer_hour::Dict{Int,Int}                         = Dict()
    lastHourOfDay::Dict{Int,Int}                              = Dict()
    firstHourOfDay::Dict{Int,Int}                             = Dict()
    slice_width_hours::Dict{Int,Float64}                      = Dict()
    hoursPerDayEffective::Dict{Int,Float64}                   = Dict()
    prev_hour::Dict{Int,Int}                                  = Dict()   # h → h-1 (cyclic)
    next_hour::Dict{Int,Int}                                  = Dict()

    # Cluster index helpers (only populated in TS mode)
    dayWeight::Dict{Int,Float64}                              = Dict()   # rd → days assigned
    clusterHourWeight::Dict{Int,Float64}                      = Dict()   # hc → hours represented
    mapDay_repDay::Dict{Int,Int}                              = Dict()   # d → rd
    mapHour_clusterHour::Dict{Int,Int}                        = Dict()   # h → hc
    repDay_of_clusterHour::Dict{Int,Int}                      = Dict()   # hc → rd
    intradaySlot_of_clusterHour::Dict{Int,Int}                = Dict()
    quarterPer_clusterHour::Dict{Int,Int}                     = Dict()

    # Soft day-mix weights (for dayMix_softness>0 or PGD modes)
    dayMix_weight::Dict{Tuple{Int,Int},Float64}               = Dict()   # (d, rd) → weight
    dayMix_weight_soft::Dict{Tuple{Int,Int},Float64}          = Dict()   # raw kernel
    dayMix_weight_soft_active::Dict{Tuple{Int,Int},Float64}   = Dict()   # active after Sinkhorn

    # Temporal window membership (rd → range/week/month/season/semester/year)
    repDayWeight_range::Dict{Tuple{Int,Int},Float64}          = Dict()
    repDayWeight_week::Dict{Tuple{Int,Int},Float64}           = Dict()
    repDayWeight_month::Dict{Tuple{Int,Int},Float64}          = Dict()
    repDayWeight_season::Dict{Tuple{Int,Int},Float64}         = Dict()
    repDayWeight_semester::Dict{Tuple{Int,Int},Float64}       = Dict()
    repDayWeight_year::Dict{Int,Float64}                      = Dict()

    # Full-hourly temporal membership (h → r_dayWindow/week/month/season/semester/day)
    hourInRange::Dict{Tuple{Int,Int},Bool}                    = Dict()
    hourInWeek::Dict{Tuple{Int,Int},Bool}                     = Dict()
    hourInMonth::Dict{Tuple{Int,Int},Bool}                    = Dict()
    hourInSeason::Dict{Tuple{Int,Int},Bool}                   = Dict()
    hourInSemester::Dict{Tuple{Int,Int},Bool}                 = Dict()
    hourInDay::Dict{Tuple{Int,Int},Bool}                      = Dict()

    # Filter sets (precomputed once for IndexDomain conditions)
    closure_days_DR::Dict{Symbol,Vector{Int}}                 = Dict()
    closure_days_BE::Dict{Symbol,Vector{Int}}                 = Dict()
    rolling_window_hours_q::Dict{Int,Vector{Int}}             = Dict()  # q → hours in 4h window
    rolling_window_hours_r::Dict{Int,Vector{Int}}             = Dict()  # r → hours in 3d window

    # =========================================================================
    # Project-specific extensions (opt-in, NOT part of the core model)
    # See src/extensions/README.md for the architecture and removal steps.
    # =========================================================================
    extensions::Set{Symbol}                                   = Set{Symbol}()
end

# ----------------------------------------------------------------------------
# ModelData — bundle for passing into model builders
# ----------------------------------------------------------------------------

"""
    ModelData

Bundle of sets + parameters. Pass-by-reference into model construction
functions and into solver / writer / postprocess functions.
"""
mutable struct ModelData
    sets::ModelSets
    params::ModelParams
end

ModelData() = ModelData(ModelSets(), ModelParams())

# ----------------------------------------------------------------------------
# RunResult — post-solve summary
# ----------------------------------------------------------------------------

"""
    RunResult

Captures one solver invocation. Mirrors the fields tracked by IESA-Opt 1.0
`RecordRunStatistics` and the current run-statistics result table.
"""
struct RunResult
    output_folder::String
    timestamp::DateTime
    mode::Symbol                          # :fh, :ts, :as
    termination_status::String
    primal_status::String
    program_status::String                 # human-readable: "Optimal", "Sub-optimal", "Infeasible"
    objective_value::Float64
    solve_seconds::Float64
    total_seconds::Float64
    n_rows::Int
    n_cols::Int
    n_nnz::Int
    n_iterations::Int
    barrier_iterations::Int
    settings::Dict{String,Any}             # solver attributes used
    weather_year::String
    n_repDays::Int
    hoursPer_day::Int
    clustering_approach::Symbol
end
