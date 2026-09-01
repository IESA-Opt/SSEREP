Base.@kwdef struct CampaignBundle
    name::String
    config_path::String
    input_workbook::String
    input_sha256::String
    seed::Int
    rows::Vector{ParameterRow}
    campaigns::Dict{String,Dict{String,Any}}
end

function _campaign_config_value(value)
    value === nothing && return nothing
    value isa JSON3.Object && return Dict{String,Any}(
        String(key) => _campaign_config_value(item) for (key, item) in pairs(value))
    value isa JSON3.Array && return Any[_campaign_config_value(item) for item in value]
    return value
end

function _campaign_parameter_rows(parameter_space)
    columns = String.(parameter_space["columns"])
    required = ("parameter", "subparameter", "sheet", "cell", "type", "min", "max")
    all(name -> name in columns, required) ||
        throw(ArgumentError("Campaign parameter_space is missing required columns."))
    column = Dict(name => index for (index, name) in pairs(columns))
    rows = ParameterRow[]
    for values in parameter_space["rows"]
        length(values) == length(columns) || throw(ArgumentError(
            "Campaign parameter row has $(length(values)) values; expected $(length(columns))."))
        push!(rows, ParameterRow(
            parameter = values[column["parameter"]],
            subparameter = values[column["subparameter"]],
            sheet = values[column["sheet"]],
            cell = values[column["cell"]],
            type = values[column["type"]],
            min = values[column["min"]],
            max = values[column["max"]],
            step = haskey(column, "step") ? values[column["step"]] : nothing,
            notes = "",
        ))
    end
    return rows
end

function _campaign_rows_with_overrides(rows::Vector{ParameterRow}, settings::AbstractDict)
    overrides = get(settings, "range_overrides", Dict{String,Any}())
    isempty(overrides) && return rows
    known_parameters = Set(row.parameter for row in rows)
    unknown = setdiff(Set(String.(keys(overrides))), known_parameters)
    isempty(unknown) || throw(ArgumentError(
        "Range overrides reference unknown parameters: $(join(sort!(collect(unknown)), ", "))."))
    overridden = ParameterRow[]
    for row in rows
        if haskey(overrides, row.parameter)
            bounds = overrides[row.parameter]
            length(bounds) == 2 || throw(ArgumentError(
                "Range override for '$(row.parameter)' must contain [min, max]."))
            minimum, maximum = Float64.(bounds)
            minimum < maximum || throw(ArgumentError(
                "Range override for '$(row.parameter)' must have min < max."))
            push!(overridden, ParameterRow(
                parameter = row.parameter, subparameter = row.subparameter,
                sheet = row.sheet, cell = row.cell, type = row.type,
                min = minimum, max = maximum, step = row.step, notes = row.notes))
        else
            push!(overridden, row)
        end
    end
    return overridden
end

"""
    load_campaign_bundle(path; verify_input=true) -> CampaignBundle

Load a versioned scenario-space campaign bundle. Relative input-workbook
paths are resolved from the repository root (two directories above a bundle
stored under `config/campaigns`). When `verify_input` is true, the workbook's
SHA-256 must match the pinned digest before a campaign can start.
"""
function load_campaign_bundle(path::AbstractString; verify_input::Bool = true)
    config_path = abspath(path)
    isfile(config_path) || throw(ArgumentError("Campaign config not found: $config_path"))
    raw = _campaign_config_value(JSON3.read(read(config_path, String)))
    get(raw, "schema_version", nothing) == 1 ||
        throw(ArgumentError("Unsupported campaign schema version $(get(raw, "schema_version", nothing))."))

    repository_root = normpath(joinpath(dirname(config_path), "..", ".."))
    configured_workbook = String(raw["input_workbook"])
    input_workbook = isabspath(configured_workbook) ?
        normpath(configured_workbook) : normpath(joinpath(repository_root, configured_workbook))
    expected_sha256 = lowercase(String(raw["input_sha256"]))
    if verify_input
        isfile(input_workbook) || throw(ArgumentError(
            "Campaign input workbook not found: $input_workbook"))
        actual_sha256 = bytes2hex(SHA.sha256(read(input_workbook)))
        actual_sha256 == expected_sha256 || throw(ArgumentError(
            "Campaign input workbook SHA-256 mismatch: expected $expected_sha256, got $actual_sha256."))
    end

    campaigns = Dict{String,Dict{String,Any}}(
        String(name) => Dict{String,Any}(String(key) => value for (key, value) in pairs(settings))
        for (name, settings) in pairs(raw["campaigns"])
    )
    return CampaignBundle(
        name = String(raw["name"]),
        config_path = config_path,
        input_workbook = input_workbook,
        input_sha256 = expected_sha256,
        seed = Int(raw["seed"]),
        rows = _campaign_parameter_rows(raw["parameter_space"]),
        campaigns = campaigns,
    )
end

"""
    campaign_spec(bundle, campaign) -> CampaignSpec

Build one grouped `CampaignSpec` from a named campaign arm in `bundle`.
For Morris, `n_variants` must equal `n_trajectories * (k + 1)`, where `k`
is the number of unique grouped parameters.
"""
function campaign_spec(bundle::CampaignBundle, campaign::AbstractString)
    campaign_name = String(campaign)
    haskey(bundle.campaigns, campaign_name) || throw(ArgumentError(
        "Unknown campaign '$campaign_name'. Available campaigns: $(join(sort!(collect(keys(bundle.campaigns))), ", "))."))
    settings = bundle.campaigns[campaign_name]
    method = parse_sampling_method(settings["method"])
    n_variants = Int(settings["n_variants"])
    seed = Int(get(settings, "seed", bundle.seed))
    rows = _campaign_rows_with_overrides(bundle.rows, settings)
    if method === :morris
        n_trajectories = Int(settings["n_trajectories"])
        n_variants == n_trajectories || throw(ArgumentError(
            "Morris n_variants is a trajectory count and must equal n_trajectories ($n_trajectories)."))
        expected_evaluations = n_trajectories *
            (length(unique(String(row.parameter) for row in bundle.rows)) + 1)
        Int(settings["n_evaluations"]) == expected_evaluations || throw(ArgumentError(
            "Morris campaign has $(settings["n_evaluations"]) evaluations; expected $expected_evaluations for $n_trajectories complete trajectories."))
    end
    return CampaignSpec(
        name = "$(bundle.name) - $campaign_name",
        method = method,
        n_variants = n_variants,
        seed = seed,
        rows = rows,
    )
end