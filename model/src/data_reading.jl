# =============================================================================
# data_reading.jl — Read IESA-Opt Excel workbook into ModelSets + ModelParams
#
# Reads the workbook layout used by IESA-Opt into Symbol-keyed Julia data
# structures. The XLSX layout matches the IESA-Opt 1.0 reader
# (`Procedure DataReading` in `MainProject/IESA-Opt.ams`).
#
# Sheet → ModelParams/ModelSets mapping:
#   "IESA-Opt database" → scenario_description
#   "Parameters"        → scalars
#   "Types"             → dispatch/activity/process/flexibility/range/sector
#                         enumerations + node maps
#   "NodeParameters"    → emission targets
#   "Activities"        → activities_original + per-activity metadata + net
#                         volumes
#   "HourlyProfiles"    → hours_orig + profile_typeRead + raw 8760 profiles
#   "Technologies"      → tech_balancers + ~40 per-tech parameters
#   "EnergyBalance"     → activity_balancesRef
#   "Infrastructure"    → tech_infra + infra-specific params (merged into
#                         tech_* dicts)
#   "PriceProfiles"     → interconnectedHourly_pricesOrig
#   "ActGrouping"       → activities_group + act_to_group
#   "EffLearning"       → activity_EffImprov
#   "Feedstocks"        → feedstockUse_techOrig
#   "Retrofitting"      → retrofit_relations, retrofit_cost
# =============================================================================

# -----------------------------------------------------------------------------
# Top-level entry
# -----------------------------------------------------------------------------

"""
    read_data(xlsx_path::AbstractString;
              periods::Vector{Int} = [2022, 2025, 2030, 2035, 2040, 2045, 2050],
              periods_solve::Union{Nothing,Vector{Int}} = nothing) -> ModelData

Open the IESA-Opt Excel workbook at `xlsx_path` and populate a `ModelData`
(bundle of `ModelSets` + `ModelParams`). After reading, derived sets
(`derive_sets!`) and derived parameters (`compute_derived_params!`) are
computed.

If `periods_solve` is `nothing`, defaults to a copy of `periods`.
"""
function read_data(xlsx_path::AbstractString;
                   periods::Vector{Int} = [2022, 2025, 2030, 2035, 2040, 2045, 2050],
                   periods_solve::Union{Nothing,Vector{Int}} = nothing)

    isfile(xlsx_path) || error("XLSX not found: $xlsx_path")

    md = ModelData()
    s, p = md.sets, md.params
    s.periods = copy(periods)
    s.periods_solve = periods_solve === nothing ? copy(periods) : copy(periods_solve)

    @info "read_data: opening workbook" path=xlsx_path
    flush(stderr)
    XLSX.openxlsx(xlsx_path, mode="r") do xf
        _logsheet("IESA-Opt database");   _read_iesa_opt_database!(s, p, xf)
        _logsheet("Parameters");          _read_parameters_sheet!(s, p, xf)
        _logsheet("Types");               _read_types_sheet!(s, p, xf)
        _logsheet("NodeParameters");      _read_node_parameters_sheet!(s, p, xf)
        _logsheet("Activities");          _read_activities_sheet!(s, p, xf)
        _logsheet("HourlyProfiles");      _read_hourly_profiles_sheet!(s, p, xf)
        _logsheet("Technologies");        _read_technologies_sheet!(s, p, xf)
        _logsheet("EnergyBalance");       _read_energy_balance_sheet!(s, p, xf)
        _logsheet("Infrastructure");      _read_infrastructure_sheet!(s, p, xf)
        _logsheet("PriceProfiles");       _read_price_profiles_sheet!(s, p, xf)
        _logsheet("ActGrouping");         _read_act_grouping_sheet!(s, p, xf)
        _logsheet("EffLearning");         _read_eff_learning_sheet!(s, p, xf)
        _logsheet("Feedstocks");          _read_feedstocks_sheet!(s, p, xf)
        _logsheet("Retrofitting");        _read_retrofitting_sheet!(s, p, xf)
    end

    @info "read_data: deriving sets and parameters"
    derive_sets!(md)
    compute_derived_params!(md)

    @info "read_data: complete" technologies=length(s.technologies) activities=length(s.activities_original) periods=length(s.periods)
    return md
end

# -----------------------------------------------------------------------------
# Sheet readers
# -----------------------------------------------------------------------------

function _read_iesa_opt_database!(s::ModelSets, p::ModelParams, xf)
    sh = xf["IESA-Opt database"]
    p.scenario_description = _str(sh["E21"])
    return nothing
end

function _read_parameters_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Parameters"]
    p.XC_TransmissionLoss_global = _float(sh["B5"])
    p.baseload_treshold          = _float(sh["B6"])
    p.shedding_inLoad            = _float(sh["B7"])
    p.social_discount_rate       = _float(sh["B12"])
    p.base_year                  = _int(sh["B13"])
    p.electricity_trade_ratio    = _float(sh["B40"])
    p.electricity_trade_volume   = _float(sh["B41"])
    p.ActiveConstraintSet        = _str(sh["B46"])
    return nothing
end

function _read_types_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Types"]
    last_row = _last_row(sh, "A")
    s.dispatch_type    = _read_column_symbols(sh, "A", 4, last_row)
    s.activity_type    = _read_column_symbols(sh, "B", 4, last_row)
    s.process_type     = _read_column_symbols(sh, "C", 4, last_row)
    s.flexibility_type = _read_column_symbols(sh, "D", 4, last_row)
    s.range_type       = _read_column_symbols(sh, "E", 4, last_row)
    s.sectors          = _read_column_symbols(sh, "F", 4, last_row)
    s.nodes            = _read_column_symbols(sh, "H", 4, last_row)
    s.node_names       = _read_column_symbols(sh, "I", 4, last_row)
    s.energy_labels    = _read_column_symbols(sh, "K", 4, last_row)
    s.sectors_kev      = _read_column_symbols(sh, "N", 4, last_row)

    _read_list_to_sym!(p.IEM_sector,   sh, "F", "G", 4, last_row)
    _read_list_to_sym!(p.namePer_node, sh, "H", "I", 4, last_row)
    _read_list_to_sym!(p.IEM_node,     sh, "H", "J", 4, last_row)
    _read_list_to_float!(p.is_renewable, sh, "K", "L", 4, last_row)
    return nothing
end

function _read_node_parameters_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["NodeParameters"]
    last_row = _last_row(sh, "A")
    nodes_col       = _read_column_symbols(sh, "A", 5, last_row)
    # NodeParameters layout (header row 3):
    #   B..H  (7 cols, 2022-2050)  Emission target air  [MtonCO2eq/yr]
    #   I                          Cumulative emission 2022-2050 [Mton]
    #   J                          Cumulative CO2 storage [Mton]
    #   K..P                       Sector emission targets in 2030 (not used here)
    #   Q                          Data source label
    #   R..X  (7 cols, 2022-2050)  Emission target incl Scope3+FuelEx
    #   Y..AE  (7 cols, 2022-2050)  Emission target Bunker
    #   AF..AL (7 cols, 2022-2050)  Emission target FeedStock (NOT read by IESA-Opt 1.0)
    period_hdr_B    = _read_row_ints(sh, 3, "B", "H")          # B..H = 7 periods
    _read_table_sym_int_to_float!(p.emissionTargetAir, sh, nodes_col, period_hdr_B, "B", 5, last_row;
                                  keep_zeros=true)

    _read_column_to_float_dict_sym!(p.CO2_cumulative_budget,  sh, "A", "I", 5, last_row)
    _read_column_to_float_dict_sym!(p.cumulative_CO2storage,  sh, "A", "J", 5, last_row)

    period_hdr_R = _read_row_ints(sh, 3, "R", "X")
    _read_table_sym_int_to_float!(p.emissionTargetAll, sh, nodes_col, period_hdr_R, "R", 5, last_row;
                                  keep_zeros=true)

    period_hdr_Y = _read_row_ints(sh, 3, "Y", "AE")
    _read_table_sym_int_to_float!(p.emissionTargetBunker, sh, nodes_col, period_hdr_Y, "Y", 5, last_row;
                                  keep_zeros=true)

    # IESA-Opt 1.0 reads emissionTarget_FeedStocks from the SAME "Y5:AE" range as
    # emissionTarget_Bunkers, so the AF..AL column block is never applied.
    _read_table_sym_int_to_float!(p.emissionTargetFS, sh, nodes_col, period_hdr_Y, "Y", 5, last_row;
                                  keep_zeros=true)
    return nothing
end

function _read_activities_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Activities"]
    last_row = _last_row(sh, "A")
    s.activities_original = _read_column_symbols(sh, "A", 9, last_row)

    _read_list_to_sym!(p.act_units,           sh, "A", "B", 9, last_row)
    _read_list_to_float!(p.actChange_maxOrig, sh, "A", "J", 9, last_row)
    _read_list_to_sym!(p.dispatchType_act,    sh, "A", "K", 9, last_row)
    _read_list_to_sym!(p.activityType_act,    sh, "A", "L", 9, last_row)
    _read_list_to_sym!(p.nodePer_act,         sh, "A", "M", 9, last_row)
    _read_list_to_sym!(p.emissionTarget_bin,  sh, "A", "N", 9, last_row)
    _read_list_to_sym!(p.labelPer_act,        sh, "A", "O", 9, last_row)

    period_hdr_C = _read_row_ints(sh, 8, "C", "I")
    _read_table_sym_int_to_float!(p.activities_netVolumesOrig, sh,
        s.activities_original, period_hdr_C, "C", 9, last_row)
    return nothing
end

function _read_hourly_profiles_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["HourlyProfiles"]
    last_row = _last_row(sh, "A")
    last_col = _last_col(sh)
    s.hours_orig = _read_column_ints(sh, "A", 5, last_row)
    s.profile_typeRead = _read_row_symbols(sh, 3, "D", last_col)

    _read_column_to_int_dict_int!(p.monthPer_hourOrig, sh, "A", "C", 5, last_row)
    _read_table_int_sym_to_float!(p.hourly_profilesReadOrig, sh,
        s.hours_orig, s.profile_typeRead, "D", 5, last_row)
    return nothing
end

function _read_technologies_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Technologies"]
    last_row = _last_row(sh, "A")
    s.tech_balancers = _read_column_symbols(sh, "A", 7, last_row)

    _read_list_to_sym!(p.tech_sector_kev,      sh, "A", "B", 7, last_row)
    _read_list_to_sym!(p.tech_category,        sh, "A", "C", 7, last_row)
    _read_list_to_sym!(p.tech_sector,          sh, "A", "D", 7, last_row)
    _read_list_to_sym!(p.tech_subsector,       sh, "A", "E", 7, last_row)
    _read_list_to_sym!(p.activityPer_techOrig, sh, "A", "F", 7, last_row)
    _read_list_to_str!(p.tech_name,            sh, "A", "G", 7, last_row)
    _read_list_to_sym!(p.tech_units,           sh, "A", "H", 7, last_row)

    period_hdr_I = _read_row_ints(sh, 4, "I", "O")
    _read_table_sym_int_to_float!(p.inv_cost, sh, s.tech_balancers, period_hdr_I, "I", 7, last_row)

    _read_list_to_float!(p.Salvage_value, sh, "A", "P", 7, last_row)

    period_hdr_Q = _read_row_ints(sh, 4, "Q", "W")
    _read_table_sym_int_to_float!(p.fom_cost, sh, s.tech_balancers, period_hdr_Q, "Q", 7, last_row)

    period_hdr_X = _read_row_ints(sh, 4, "X", "AD")
    _read_table_sym_int_to_float!(p.vom_cost, sh, s.tech_balancers, period_hdr_X, "X", 7, last_row)

    _read_list_to_float!(p.WACC,                sh, "A", "AE", 7, last_row)
    _read_list_to_float!(p.construction_time,   sh, "A", "AF", 7, last_row)   # stored as Float; will round at use-site
    _read_list_to_float!(p.economic_lifetime,   sh, "A", "AG", 7, last_row)
    _read_list_to_float!(p.technical_lifetime,  sh, "A", "AH", 7, last_row)
    _read_list_to_float!(p.cap2act,             sh, "A", "AI", 7, last_row)
    _read_list_to_sym!(p.processType_tech,      sh, "A", "AJ", 7, last_row)
    _read_list_to_sym!(p.profileType_techRead,  sh, "A", "AK", 7, last_row)
    _read_list_to_float!(p.ramping,             sh, "A", "AL", 7, last_row)

    _read_list_to_sym!(p.CHP_prodOrig,          sh, "A", "AM", 7, last_row)
    _read_list_to_sym!(p.CHP_fuelOrig,          sh, "A", "AN", 7, last_row)
    _read_list_to_float!(p.CHP_eta,             sh, "A", "AO", 7, last_row)
    _read_list_to_sym!(p.CHP_range,             sh, "A", "AP", 7, last_row)
    _read_list_to_float!(p.CHP_dev_use,         sh, "A", "AQ", 7, last_row)
    _read_list_to_float!(p.CHP_dev_PtoH,        sh, "A", "AR", 7, last_row)

    _read_list_to_float!(p.shed_capacity_percentage, sh, "A", "AS", 7, last_row)
    _read_list_to_float!(p.shed_volume,         sh, "A", "AT", 7, last_row)
    _read_list_to_sym!(p.shed_range,            sh, "A", "AU", 7, last_row)

    _read_list_to_float!(p.phs_capacity,        sh, "A", "AV", 7, last_row)
    _read_list_to_float!(p.reservoir_capacity,  sh, "A", "AW", 7, last_row)
    _read_list_to_float!(p.phs_Losses,          sh, "A", "AX", 7, last_row)

    _read_list_to_sym!(p.flexibilityType_tech,  sh, "A", "AY", 7, last_row)
    _read_list_to_sym!(p.flex_activityOrig,     sh, "A", "AZ", 7, last_row)
    _read_list_to_float!(p.flex_capacity_pct,   sh, "A", "BA", 7, last_row)
    _read_list_to_float!(p.flex_storage,        sh, "A", "BB", 7, last_row)
    _read_list_to_sym!(p.flex_range,            sh, "A", "BC", 7, last_row)
    _read_list_to_float!(p.flex_losses_legacy,  sh, "A", "BD", 7, last_row)   # legacy combined; split later
    _read_list_to_float!(p.flex_nnLoad,         sh, "A", "BE", 7, last_row)
    _read_list_to_float!(p.avg_journey,         sh, "A", "BF", 7, last_row)
    _read_list_to_float!(p.avg_speed,           sh, "A", "BG", 7, last_row)

    _read_list_to_sym!(p.buffer_activityOrig,   sh, "A", "BH", 7, last_row)
    _read_list_to_float!(p.bufferUP_capacity,   sh, "A", "BI", 7, last_row)
    _read_list_to_float!(p.bufferDW_capacity,   sh, "A", "BJ", 7, last_row)
    _read_list_to_float!(p.buffer_storage,      sh, "A", "BL", 7, last_row)

    _read_list_to_float!(p.techChange_max,      sh, "A", "BM", 7, last_row)
    _read_list_to_float!(p.techStock_exist,     sh, "A", "BN", 7, last_row)

    period_hdr_BO = _read_row_ints(sh, 5, "BO", "BT")
    _read_table_sym_int_to_float!(p.decom_planned, sh, s.tech_balancers, period_hdr_BO, "BO", 7, last_row)

    period_hdr_BU = _read_row_ints(sh, 5, "BU", "CA")
    _read_table_sym_int_to_float!(p.techStock_min, sh, s.tech_balancers, period_hdr_BU, "BU", 7, last_row)

    period_hdr_CB = _read_row_ints(sh, 5, "CB", "CH")
    _read_aimms_max_stock!(p.techStock_max, sh, s.tech_balancers, period_hdr_CB, "CB", 7, last_row)

    period_hdr_CI = _read_row_ints(sh, 5, "CI", "CO")
    _read_table_sym_int_to_float!(p.techUse_min, sh, s.tech_balancers, period_hdr_CI, "CI", 7, last_row)

    period_hdr_CP = _read_row_ints(sh, 5, "CP", "CV")
    _read_table_sym_int_to_float!(p.techUse_max, sh, s.tech_balancers, period_hdr_CP, "CP", 7, last_row)

    period_hdr_CW = _read_row_ints(sh, 5, "CW", "DC")
    _read_table_sym_int_to_bool!(p.no_new_invest, sh, s.tech_balancers, period_hdr_CW, "CW", 7, last_row)

    period_hdr_DD = _read_row_ints(sh, 5, "DD", "DJ")
    _read_table_sym_int_to_bool!(p.no_eco_decom, sh, s.tech_balancers, period_hdr_DD, "DD", 7, last_row)
    return nothing
end

function _read_energy_balance_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["EnergyBalance"]
    last_row = _last_row(sh, "A")
    last_col = _last_col(sh)
    tech_rows  = _read_column_symbols(sh, "A", 7, last_row)
    act_hdrs   = _read_row_strings(sh, 3, "Q", last_col)
    _read_table_3key_balances!(p.activity_balancesRef, sh, tech_rows, act_hdrs, "Q", 7, last_row)
    return nothing
end

function _read_infrastructure_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Infrastructure"]
    last_row = _last_row(sh, "A")
    s.tech_infra = _read_column_symbols(sh, "A", 6, last_row)

    # Merge infra rows into tech_* dicts (merge=true → don't overwrite existing)
    _read_list_to_sym!(p.tech_sector_kev,  sh, "A", "B", 6, last_row; merge=true)
    _read_list_to_sym!(p.tech_category,    sh, "A", "C", 6, last_row; merge=true)
    _read_list_to_sym!(p.tech_sector,      sh, "A", "D", 6, last_row; merge=true)
    _read_list_to_sym!(p.tech_subsector,   sh, "A", "E", 6, last_row; merge=true)
    _read_list_to_str!(p.tech_name,        sh, "A", "F", 6, last_row; merge=true)
    _read_list_to_sym!(p.tech_units,       sh, "A", "G", 6, last_row; merge=true)

    period_hdr_H = _read_row_ints(sh, 4, "H", "N")
    _read_table_sym_int_to_float!(p.inv_cost, sh, s.tech_infra, period_hdr_H, "H", 6, last_row; merge=true)

    _read_list_to_float!(p.Salvage_value,    sh, "A", "O", 6, last_row; merge=true)

    period_hdr_P = _read_row_ints(sh, 4, "P", "V")
    _read_table_sym_int_to_float!(p.fom_cost, sh, s.tech_infra, period_hdr_P, "P", 6, last_row; merge=true)

    _read_list_to_float!(p.WACC,               sh, "A", "W", 6, last_row; merge=true)
    _read_list_to_float!(p.economic_lifetime,  sh, "A", "X", 6, last_row; merge=true)
    _read_list_to_float!(p.technical_lifetime, sh, "A", "Y", 6, last_row; merge=true)
    _read_list_to_float!(p.cap2act,            sh, "A", "Z", 6, last_row; merge=true)
    _read_list_to_sym!(p.infra_range,          sh, "A", "AA", 6, last_row)
    _read_list_to_sym!(p.infra_activityOrig,   sh, "A", "AB", 6, last_row)
    _read_list_to_float!(p.techChange_max,     sh, "A", "AD", 6, last_row; merge=true)
    _read_list_to_float!(p.techStock_exist,    sh, "A", "AE", 6, last_row; merge=true)

    period_hdr_AF = _read_row_ints(sh, 3, "AF", "AK")
    _read_table_sym_int_to_float!(p.decom_planned, sh, s.tech_infra, period_hdr_AF, "AF", 6, last_row; merge=true)

    period_hdr_AL = _read_row_ints(sh, 3, "AL", "AR")
    _read_table_sym_int_to_float!(p.techStock_min, sh, s.tech_infra, period_hdr_AL, "AL", 6, last_row; merge=true)

    period_hdr_AS = _read_row_ints(sh, 3, "AS", "AY")
    _read_aimms_max_stock!(p.techStock_max, sh, s.tech_infra, period_hdr_AS, "AS", 6, last_row)
    return nothing
end

function _read_price_profiles_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["PriceProfiles"]
    last_row = _last_row(sh, "A")
    last_col = _last_col(sh)
    _read_price_profiles_table!(p.interconnectedHourly_pricesOrig, sh, s.hours_orig, 5, last_row, last_col)
    return nothing
end

function _read_act_grouping_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["ActGrouping"]
    last_row = _last_row(sh, "A")
    s.activities_group = _read_column_symbols(sh, "A", 4, last_row)

    _read_list_to_sym!(p.dispatchType_act,    sh, "A", "B", 4, last_row; merge=true)
    _read_list_to_sym!(p.activityType_act,    sh, "A", "C", 4, last_row; merge=true)
    _read_list_to_sym!(p.nodePer_act,         sh, "A", "D", 4, last_row; merge=true)
    _read_list_to_sym!(p.emissionTarget_bin,  sh, "A", "E", 4, last_row; merge=true)
    _read_list_to_sym!(p.labelPer_act,        sh, "A", "F", 4, last_row; merge=true)

    _read_act_grouping_table!(p.act_to_group, sh, 4, last_row)
    return nothing
end

function _read_eff_learning_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["EffLearning"]
    last_row = _last_row(sh, "C")
    tech_act_pairs = _read_two_col_keys_sym(sh, "C", "D", 4, last_row)
    period_hdr_E   = _read_row_ints(sh, 3, "E", "K")
    _read_table_sym_pair_int_to_float!(p.activity_EffImprov, sh, tech_act_pairs, period_hdr_E, "E", 4, last_row)
    return nothing
end

function _read_feedstocks_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Feedstocks"]
    last_row = _last_row(sh, "C")
    _read_list_2sym_to_float!(p.feedstockUse_techOrig, sh, "C", "D", "H", 4, last_row)
    return nothing
end

function _read_retrofitting_sheet!(s::ModelSets, p::ModelParams, xf)
    sh = xf["Retrofitting"]
    last_row = _last_row(sh, "A")
    _read_list_2sym_to_bool!(p.retrofit_relations, sh, "A", "B", "F", 4, last_row)
    # Retrofit costs are stored by (from_tech, to_tech, period).
    # We broadcast across all configured periods.
    _read_retrofit_cost!(p.retrofit_cost, sh, "A", "B", "G", 4, last_row, s.periods)
    return nothing
end

# =============================================================================
# Low-level XLSX helpers
# =============================================================================

# ---------------------------------------------------------------- coercion --
_str(x)::String = (x === nothing || ismissing(x)) ? "" : string(x)

function _float(x)::Float64
    if x === nothing || ismissing(x)
        return 0.0
    elseif x isa Number
        return Float64(x)
    else
        s = strip(string(x))
        isempty(s) && return 0.0
        s = replace(s, "," => ".")
        v = tryparse(Float64, s)
        return v === nothing ? 0.0 : v
    end
end

_int(x)::Int = (x === nothing || ismissing(x)) ? 0 : Int(round(Float64(x)))

function _sym(x)::Symbol
    s = strip(_str(x))
    return Symbol(s)
end

# Per-sheet progress message; flushes stderr so it shows up immediately even
# when stdout/stderr are piped through PowerShell (block-buffered).
function _logsheet(name::AbstractString)
    @info "  reading sheet" sheet=name
    flush(stderr)
end

# ----------------------------------------------------------- column helpers --
function _col_index(col::AbstractString)::Int
    idx = 0
    for c in uppercase(col)
        idx = idx * 26 + (Int(c) - Int('A') + 1)
    end
    return idx
end

function _col_letter(n::Int)::String
    s = ""
    while n > 0
        rem = (n - 1) % 26
        s = string(Char(Int('A') + rem)) * s
        n = div(n - 1, 26)
    end
    return s
end

function _last_row(sh, col::AbstractString)::Int
    ci = _col_index(col)
    # XLSX's stored dimension is the worksheet's recorded bounding box — use it
    # as the upper bound instead of scanning up to row 10 000 (which is
    # extremely slow for sparse sheets).
    dim_end = try
        XLSX.row_number(XLSX.get_dimension(sh).stop)
    catch
        10000
    end
    last = 1
    consec_empty = 0
    @inbounds for r in 1:dim_end
        v = sh[r, ci]
        if v !== nothing && !ismissing(v) && _str(v) != ""
            last = r
            consec_empty = 0
        else
            consec_empty += 1
            consec_empty > 200 && r > last + 200 && break
        end
    end
    return last
end

function _last_col(sh)::String
    dim = try
        XLSX.get_dimension(sh)
    catch
        nothing
    end
    max_col = dim === nothing ? 1000 : XLSX.column_number(dim.stop)
    max_row = dim === nothing ? 5     : min(5, XLSX.row_number(dim.stop))
    last = 1
    consec_empty = 0
    @inbounds for c in 1:max_col
        found = false
        for r in 1:max_row
            v2 = sh[r, c]
            if v2 !== nothing && !ismissing(v2) && _str(v2) != ""
                found = true; break
            end
        end
        if found
            last = c
            consec_empty = 0
        else
            consec_empty += 1
            consec_empty > 50 && c > last + 50 && break
        end
    end
    return _col_letter(last)
end

# ----------------------------------------------------------- vector readers --
function _read_column_symbols(sh, col::AbstractString, r_start::Int, r_end::Int)::Vector{Symbol}
    ci = _col_index(col)
    out = Symbol[]
    for r in r_start:r_end
        s = _str(sh[r, ci])
        if s != ""
            push!(out, Symbol(strip(s)))
        end
    end
    return out
end

function _read_column_ints(sh, col::AbstractString, r_start::Int, r_end::Int)::Vector{Int}
    ci = _col_index(col)
    out = Int[]
    for r in r_start:r_end
        v = sh[r, ci]
        if v !== nothing && !ismissing(v)
            push!(out, _int(v))
        end
    end
    return out
end

function _read_row_strings(sh, row::Int, col_start::AbstractString, col_end::AbstractString)::Vector{String}
    cs = _col_index(col_start)
    ce = _col_index(col_end)
    out = String[]
    for c in cs:ce
        s = _str(sh[row, c])
        if s != ""
            push!(out, s)
        end
    end
    return out
end

function _read_row_symbols(sh, row::Int, col_start::AbstractString, col_end::AbstractString)::Vector{Symbol}
    return [Symbol(strip(s)) for s in _read_row_strings(sh, row, col_start, col_end)]
end

function _read_row_ints(sh, row::Int, col_start::AbstractString, col_end::AbstractString)::Vector{Int}
    cs = _col_index(col_start)
    ce = _col_index(col_end)
    out = Int[]
    for c in cs:ce
        v = sh[row, c]
        if v !== nothing && !ismissing(v)
            s = strip(_str(v))
            isempty(s) && continue
            n = tryparse(Int, s)
            n === nothing || push!(out, n)
        end
    end
    return out
end

# ------------------------------------------------------------- list readers --
"""
    _read_list_to_sym!(d, sh, col_key, col_val, rs, re; merge=false)

Read column pairs (Symbol key → Symbol value). When `merge=true`, do not
overwrite existing entries (used for the Infrastructure sheet merge).
"""
function _read_list_to_sym!(d::Dict{Symbol,Symbol}, sh, col_key::AbstractString, col_val::AbstractString,
                             r_start::Int, r_end::Int; merge::Bool=false)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        ks = _str(sh[r, ck])
        vs = _str(sh[r, cv])
        ks == "" && continue
        k = Symbol(strip(ks))
        if !merge || !haskey(d, k)
            d[k] = Symbol(strip(vs))
        end
    end
end

function _read_list_to_str!(d::Dict{Symbol,String}, sh, col_key::AbstractString, col_val::AbstractString,
                             r_start::Int, r_end::Int; merge::Bool=false)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        ks = _str(sh[r, ck])
        vs = _str(sh[r, cv])
        ks == "" && continue
        k = Symbol(strip(ks))
        if !merge || !haskey(d, k)
            d[k] = vs
        end
    end
end

function _read_list_to_float!(d::Dict{Symbol,Float64}, sh, col_key::AbstractString, col_val::AbstractString,
                                r_start::Int, r_end::Int; merge::Bool=false)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        ks = _str(sh[r, ck])
        ks == "" && continue
        v = sh[r, cv]
        (v === nothing || ismissing(v)) && continue
        k = Symbol(strip(ks))
        fval = _float(v)
        if !merge || !haskey(d, k)
            d[k] = fval
        end
    end
end

function _read_list_to_int!(d::Dict{Symbol,Int}, sh, col_key::AbstractString, col_val::AbstractString,
                             r_start::Int, r_end::Int; merge::Bool=false)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        ks = _str(sh[r, ck])
        ks == "" && continue
        v = sh[r, cv]
        (v === nothing || ismissing(v)) && continue
        k = Symbol(strip(ks))
        if !merge || !haskey(d, k)
            d[k] = _int(v)
        end
    end
end

function _read_list_2sym_to_float!(d::Dict{Tuple{Symbol,Symbol},Float64}, sh,
                                     col_k1::AbstractString, col_k2::AbstractString, col_val::AbstractString,
                                     r_start::Int, r_end::Int)
    ck1, ck2, cv = _col_index(col_k1), _col_index(col_k2), _col_index(col_val)
    for r in r_start:r_end
        k1s = _str(sh[r, ck1])
        k2s = _str(sh[r, ck2])
        v   = sh[r, cv]
        if k1s != "" && k2s != "" && v !== nothing && !ismissing(v)
            d[(Symbol(strip(k1s)), Symbol(strip(k2s)))] = _float(v)
        end
    end
end

function _read_list_2sym_to_bool!(d::Dict{Tuple{Symbol,Symbol},Bool}, sh,
                                    col_k1::AbstractString, col_k2::AbstractString, col_val::AbstractString,
                                    r_start::Int, r_end::Int)
    ck1, ck2, cv = _col_index(col_k1), _col_index(col_k2), _col_index(col_val)
    for r in r_start:r_end
        k1s = _str(sh[r, ck1])
        k2s = _str(sh[r, ck2])
        v   = sh[r, cv]
        if k1s != "" && k2s != "" && v !== nothing && !ismissing(v)
            d[(Symbol(strip(k1s)), Symbol(strip(k2s)))] = _float(v) != 0.0
        end
    end
end

# ----------------------------------------------------------- table readers --

"""
    _read_table_sym_int_to_float!(d, sh, row_keys, col_period_headers, col_start, rs, re; merge=false, keep_zeros=false)

Read a rectangular 2D block where rows are Symbol keys and columns are Int
periods. Cell `(rk, period)` stored as Float64 in `d`.

When `keep_zeros=false` (default) explicit `0` values are skipped so that the
dict semantically represents "non-default" entries (e.g. for cost parameters
where `0` and `na` are equivalent). For constraint-defining parameters such as
`techStock_max` and `techUse_max`, IESA-Opt 1.0 distinguishes `0` ("must be <= 0")
from `na` ("no upper bound"); pass `keep_zeros=true` to preserve explicit
zero cells in the dict so the constraint loop will pick them up.
"""
function _read_table_sym_int_to_float!(d::Dict{Tuple{Symbol,Int},Float64}, sh,
                                         row_keys::Vector{Symbol}, col_periods::Vector{Int},
                                         col_start::AbstractString,
                                         r_start::Int, r_end::Int;
                                         merge::Bool=false, keep_zeros::Bool=false)
    cs = _col_index(col_start)
    n_rows = min(length(row_keys), r_end - r_start + 1)
    n_cols = length(col_periods)
    (n_rows == 0 || n_cols == 0) && return

    ce = cs + n_cols - 1
    rng = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_rows - 1)
    block = sh[rng]

    @inbounds for ri in 1:n_rows
        rk = row_keys[ri]
        for ci in 1:n_cols
            v = block[ri, ci]
            (v === nothing || ismissing(v)) && continue
            fval = _float(v)
            (!keep_zeros && fval == 0.0) && continue
            key = (rk, col_periods[ci])
            if !merge || !haskey(d, key)
                d[key] = fval
            end
        end
    end
end

function _read_aimms_max_stock!(d::Dict{Tuple{Symbol,Int},Float64}, sh,
                                row_keys::Vector{Symbol}, col_periods::Vector{Int},
                                col_start::AbstractString, r_start::Int, r_end::Int)
    cs = _col_index(col_start)
    n_rows = min(length(row_keys), r_end - r_start + 1)
    n_cols = length(col_periods)
    (n_rows == 0 || n_cols == 0) && return
    ce = cs + n_cols - 1
    range = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_rows - 1)
    block = sh[range]
    @inbounds for row_index in 1:n_rows, column_index in 1:n_cols
        raw = block[row_index, column_index]
        value = (raw === nothing || ismissing(raw)) ? 0.0 : _float(raw)
        value == -1.0 && continue
        d[(row_keys[row_index], col_periods[column_index])] = value
    end
end

function _read_table_sym_int_to_bool!(d::Dict{Tuple{Symbol,Int},Bool}, sh,
                                        row_keys::Vector{Symbol}, col_periods::Vector{Int},
                                        col_start::AbstractString,
                                        r_start::Int, r_end::Int; merge::Bool=false)
    cs = _col_index(col_start)
    n_rows = min(length(row_keys), r_end - r_start + 1)
    n_cols = length(col_periods)
    (n_rows == 0 || n_cols == 0) && return

    ce = cs + n_cols - 1
    rng = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_rows - 1)
    block = sh[rng]

    @inbounds for ri in 1:n_rows
        rk = row_keys[ri]
        for ci in 1:n_cols
            v = block[ri, ci]
            (v === nothing || ismissing(v)) && continue
            b = _float(v) != 0.0
            !b && continue
            key = (rk, col_periods[ci])
            if !merge || !haskey(d, key)
                d[key] = b
            end
        end
    end
end

"""
    _read_table_3key_balances!(d, sh, tech_rows, col_headers, col_start, rs, re)

`activity_balancesRef` reader. The Excel column headers at row 3 are activity
names, optionally with a "|YEAR" or "_YEAR" suffix encoding the period. If no
period suffix is present, the value is broadcast across all default periods
(2022..2050).
"""
function _read_table_3key_balances!(d::Dict{Tuple{Symbol,Symbol,Int},Float64}, sh,
                                      row_keys::Vector{Symbol}, col_headers::Vector{String},
                                      col_start::AbstractString, r_start::Int, r_end::Int;
                                      default_periods::NTuple{N,Int} where N = (2022,2025,2030,2035,2040,2045,2050))
    cs = _col_index(col_start)
    n_rows = min(length(row_keys), r_end - r_start + 1)
    n_cols = length(col_headers)
    (n_rows == 0 || n_cols == 0) && return

    ce = cs + n_cols - 1
    rng = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_rows - 1)
    block = sh[rng]

    @inbounds for ri in 1:n_rows
        rk = row_keys[ri]
        for ci in 1:n_cols
            v = block[ri, ci]
            (v === nothing || ismissing(v)) && continue
            fval = _float(v)
            fval == 0.0 && continue
            header = strip(col_headers[ci])
            m = match(r"^(.*?)[_|](\d{4})$", header)
            if m !== nothing
                act = Symbol(strip(String(m.captures[1])))
                per = parse(Int, m.captures[2])
                key = (rk, act, per)
                d[key] = get(d, key, 0.0) + fval
            else
                act = Symbol(String(header))
                for per in default_periods
                    key = (rk, act, per)
                    d[key] = get(d, key, 0.0) + fval
                end
            end
        end
    end
end

"""
    _read_table_sym_pair_int_to_float!(d, sh, row_pairs, col_periods, col_start, rs, re)

`activity_EffImprov` reader. Rows are (tech, activity) pairs from a two-column
key, columns are periods (Int).
"""
function _read_table_sym_pair_int_to_float!(d::Dict{Tuple{Symbol,Symbol,Int},Float64}, sh,
                                              row_pairs::Vector{Tuple{Symbol,Symbol}},
                                              col_periods::Vector{Int},
                                              col_start::AbstractString,
                                              r_start::Int, r_end::Int)
    cs = _col_index(col_start)
    n_rows = length(row_pairs)
    for (ri, r) in enumerate(r_start:r_end)
        ri > n_rows && break
        (rk1, rk2) = row_pairs[ri]
        for (ci_off, per) in enumerate(col_periods)
            c = cs + ci_off - 1
            v = sh[r, c]
            (v === nothing || ismissing(v)) && continue
            fval = _float(v)
            fval == 0.0 && continue
            d[(rk1, rk2, per)] = fval
        end
    end
end

"""
    _read_table_int_sym_to_float!(d, sh, row_hours, col_yps, col_start, rs, re)

`hourly_profilesReadOrig` reader. Rows are Int hour indices, columns are
Symbol profile-type names.
"""
function _read_table_int_sym_to_float!(d::Dict{Tuple{Int,Symbol},Float64}, sh,
                                         row_hours::Vector{Int}, col_yps::Vector{Symbol},
                                         col_start::AbstractString,
                                         r_start::Int, r_end::Int)
    cs = _col_index(col_start)
    n_rows = min(length(row_hours), r_end - r_start + 1)
    n_cols = length(col_yps)
    n_rows == 0 || n_cols == 0 && return
    # Bulk-read the dense block — orders of magnitude faster than per-cell `sh[r, c]`
    ce = cs + n_cols - 1
    rng = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_rows - 1)
    block = sh[rng]
    @assert size(block) == (n_rows, n_cols)
    @inbounds for ri in 1:n_rows
        h = row_hours[ri]
        for ci in 1:n_cols
            v = block[ri, ci]
            (v === nothing || ismissing(v)) && continue
            d[(h, col_yps[ci])] = _float(v)
        end
    end
end

"""
    _read_price_profiles_table!(d, sh, hours_orig, rs, re, last_col)

`interconnectedHourly_pricesOrig` reader. Column headers in rows 2 (activity)
and 3 (period int); row key is the orig hour.
"""
function _read_price_profiles_table!(d::Dict{Tuple{Int,Symbol,Int},Float64}, sh,
                                       hours_orig::Vector{Int},
                                       r_start::Int, r_end::Int, last_col::AbstractString)
    cs = _col_index("D")
    ce = _col_index(last_col)
    n_h = min(length(hours_orig), r_end - r_start + 1)
    n_c = ce - cs + 1
    (n_h == 0 || n_c == 0) && return

    # Bulk-read the (header rows 2,3 + body) once
    header_rng = string(_col_letter(cs), 2, ":", _col_letter(ce), 3)
    body_rng   = string(_col_letter(cs), r_start, ":", _col_letter(ce), r_start + n_h - 1)
    hdr  = sh[header_rng]                       # 2 × n_c
    body = sh[body_rng]                         # n_h × n_c

    @inbounds for ci in 1:n_c
        act_str = _str(hdr[1, ci])
        per_str = _str(hdr[2, ci])
        per     = tryparse(Int, strip(per_str))
        (act_str == "" || per === nothing) && continue
        act = Symbol(strip(act_str))
        for ri in 1:n_h
            v = body[ri, ci]
            (v === nothing || ismissing(v)) && continue
            d[(hours_orig[ri], act, per)] = _float(v)
        end
    end
end

function _read_column_to_float_dict_sym!(d::Dict{Symbol,Float64}, sh,
                                          col_key::AbstractString, col_val::AbstractString,
                                          r_start::Int, r_end::Int)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        ks = _str(sh[r, ck])
        ks == "" && continue
        v = sh[r, cv]
        (v === nothing || ismissing(v)) && continue
        d[Symbol(strip(ks))] = _float(v)
    end
end

function _read_column_to_int_dict_int!(d::Dict{Int,Int}, sh,
                                         col_key::AbstractString, col_val::AbstractString,
                                         r_start::Int, r_end::Int)
    ck, cv = _col_index(col_key), _col_index(col_val)
    for r in r_start:r_end
        kv = sh[r, ck]
        v  = sh[r, cv]
        kv === nothing && continue
        if v !== nothing && !ismissing(v)
            d[_int(kv)] = _int(v)
        end
    end
end

function _read_two_col_keys_sym(sh, col1::AbstractString, col2::AbstractString,
                                  r_start::Int, r_end::Int)::Vector{Tuple{Symbol,Symbol}}
    c1, c2 = _col_index(col1), _col_index(col2)
    out = Tuple{Symbol,Symbol}[]
    for r in r_start:r_end
        s1 = _str(sh[r, c1])
        s2 = _str(sh[r, c2])
        if s1 != "" && s2 != ""
            push!(out, (Symbol(strip(s1)), Symbol(strip(s2))))
        end
    end
    return out
end

function _read_act_grouping_table!(act_to_group::Dict{Symbol,Symbol}, sh,
                                     r_start::Int, r_end::Int)
    # Columns H = original activity, I = group activity (J is the binary flag)
    ch = _col_index("H")
    ci = _col_index("I")
    for r in r_start:r_end
        orig = _str(sh[r, ch])
        grp  = _str(sh[r, ci])
        if orig != "" && grp != ""
            act_to_group[Symbol(strip(orig))] = Symbol(strip(grp))
        end
    end
end

function _read_retrofit_cost!(d::Dict{Tuple{Symbol,Symbol,Int},Float64}, sh,
                                col_from::AbstractString, col_to::AbstractString, col_val::AbstractString,
                                r_start::Int, r_end::Int, periods::Vector{Int})
    cf, ct, cv = _col_index(col_from), _col_index(col_to), _col_index(col_val)
    for r in r_start:r_end
        sf = _str(sh[r, cf])
        st = _str(sh[r, ct])
        v  = sh[r, cv]
        if sf != "" && st != "" && v !== nothing && !ismissing(v)
            fval = _float(v)
            for per in periods
                d[(Symbol(strip(sf)), Symbol(strip(st)), per)] = fval
            end
        end
    end
end

# =============================================================================
# Post-read derivations (called from `read_data` after all sheets are loaded)
# =============================================================================

"""
    _resolve_activity_names!(md::ModelData)

After `act_to_group` is populated, resolve all "*Orig" name maps to their
grouped form:
- `CHP_prod`, `CHP_fuel` from `CHP_prodOrig`, `CHP_fuelOrig`
- `flex_activity` (already resolved during read, but re-applies grouping)
- `buffer_activity`, `infra_activity` from `*Orig`
- `activityPer_tech` from `activityPer_techOrig`
- `actChange_max_act` from `actChange_maxOrig` via `act_to_group`
"""
function _resolve_activity_names!(md::ModelData)
    s, p = md.sets, md.params

    canonical_activity = Dict{String,Symbol}()
    for a in s.activities
        get!(canonical_activity, lowercase(String(a)), a)
    end
    _canonical_activity(a::Symbol)::Symbol = get(canonical_activity, lowercase(String(a)), a)
    _resolve!(orig::Symbol)::Symbol = _canonical_activity(get(p.act_to_group, orig, orig))

    # Idempotent: clear all derived targets before re-populating. This is
    # critical because some assignments use `+=` (sum over originals feeding a
    # group), so re-running without clearing would double values.
    empty!(p.CHP_prod)
    empty!(p.CHP_fuel)
    empty!(p.buffer_activity)
    empty!(p.infra_activity)
    empty!(p.flex_activity)
    empty!(p.activityPer_tech)
    empty!(p.actChange_max_act)
    empty!(p.activities_netVolumes)

    for (t, orig) in p.CHP_prodOrig
        p.CHP_prod[t] = _resolve!(orig)
    end
    for (t, orig) in p.CHP_fuelOrig
        p.CHP_fuel[t] = _resolve!(orig)
    end
    for (t, orig) in p.buffer_activityOrig
        p.buffer_activity[t] = _resolve!(orig)
    end
    for (t, orig) in p.infra_activityOrig
        p.infra_activity[t] = _resolve!(orig)
    end
    for (t, orig) in p.flex_activityOrig
        p.flex_activity[t] = _resolve!(orig)
    end
    for (t, orig) in p.activityPer_techOrig
        p.activityPer_tech[t] = _resolve!(orig)
    end

    # actChange_max_act: from actChange_maxOrig keyed by activity_orig → group
    for (a_orig, v) in p.actChange_maxOrig
        a = _resolve!(a_orig)
        # Take the max over original activities mapping to the same group
        existing = get(p.actChange_max_act, a, 0.0)
        if v > existing
            p.actChange_max_act[a] = v
        end
    end

    # activities_netVolumes: sum over original activities feeding the same group
    for ((a_orig, per), v) in p.activities_netVolumesOrig
        a = _resolve!(a_orig)
        key = (a, per)
        p.activities_netVolumes[key] = get(p.activities_netVolumes, key, 0.0) + v
    end
    return nothing
end

"""
    _derive_profile_and_node_maps!(md::ModelData)

Build `profileType_tech` (= `profileType_techRead` resolved through any
indirect mapping) and `nodePer_techBal` from per-tech activity → node lookup.
Also seeds `actSolvePer_actOrig` (Identity map; IESA-Opt 1.0 uses this for legacy
reasons; can be overridden by grouping).
"""
function _derive_profile_and_node_maps!(md::ModelData)
    s, p = md.sets, md.params

    # IESA-Opt 1.0 line 3014:
    #   profileType_tech(tb) :=
    #       if (activityPer_tech(tb) in activities_indirect) then activityPer_tech(tb)
    #       elseif (CHP_prod(tb) in activities_indirect) then CHP_prod(tb)
    #       else profileType_techRead(tb)
    # Falls back to profileType_techRead when activities_indirect is not yet
    # built (e.g. when this is called pre-derive_sets!) so re-running
    # compute_derived_params! after derive_sets! settles the indirect mapping.
    ind_set = isempty(s.activities_indirect) ? Set{Symbol}() : Set(s.activities_indirect)
    empty!(p.profileType_tech)
    for (t, ptr) in p.profileType_techRead
        a   = get(p.activityPer_tech, t, get(p.activityPer_techOrig, t, Symbol("")))
        chp = get(p.CHP_prod,         t, get(p.CHP_prodOrig,         t, Symbol("")))
        p.profileType_tech[t] =
            (a   != Symbol("") && a   in ind_set) ? a   :
            (chp != Symbol("") && chp in ind_set) ? chp :
            ptr
    end

    empty!(p.profileType_EVuse)
    for tfv in s.tech_fEV
        ptype = get(p.profileType_tech, tfv, Symbol(""))
        ptype == Symbol("") && continue
        p.profileType_EVuse[tfv] = Symbol(string(ptype), " - Use")
    end

    # nodePer_techBal: node per tech_balancer via activity
    empty!(p.nodePer_techBal)
    for t in s.tech_balancers
        a = get(p.activityPer_tech, t, get(p.activityPer_techOrig, t, Symbol("")))
        n = get(p.nodePer_act, a, Symbol(""))
        if n != Symbol("")
            p.nodePer_techBal[t] = n
        end
    end

    # actSolvePer_actOrig: identity by default
    for a in s.activities_original
        p.actSolvePer_actOrig[a] = get(p.act_to_group, a, a)
    end

    # activities := union of activities_original + activities_group (no duplicates)
    seen = Set{Symbol}()
    out  = Symbol[]
    for a in s.activities_original
        a in seen || (push!(out, a); push!(seen, a))
    end
    for a in s.activities_group
        a in seen || (push!(out, a); push!(seen, a))
    end
    s.activities = out
    return nothing
end
