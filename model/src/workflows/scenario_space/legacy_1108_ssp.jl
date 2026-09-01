using Distributed

struct Legacy1108ResponseEntry
    field::Symbol
    indices::Tuple
    slope::Float64
end

struct Legacy1108Context
    bundle::CampaignBundle
    base_md::Union{Nothing,ModelData}
    parameter_names::Vector{String}
    direct_targets::Dict{Tuple{String,String},Tuple{Symbol,Tuple}}
    weather_profiles::Dict{Tuple{Int,Int,Symbol},Float64}
    weather_prices::Dict{Tuple{Int,Int},Float64}
    weather_cap2act::Dict{Tuple{Int,Symbol},Float64}
    responses::Dict{String,Tuple{Float64,Vector{Legacy1108ResponseEntry}}}
    cluster_cache_dir::String
end

const _LEGACY_1108_CLUSTER_CACHE_VERSION = 1
const _LEGACY_1108_CONTEXT_CACHE_VERSION = 2
const _LEGACY_1108_WORKER_CLUSTER_CACHE = Dict{Tuple{String,Int,Int},ModelData}()

_legacy_float(value) = value isa Number ? Float64(value) : parse(Float64, String(value))

function _legacy_column_name(column::Int)
    column > 0 || throw(ArgumentError("column must be positive."))
    name = ""
    while column > 0
        column, remainder = divrem(column - 1, 26)
        name = string(Char(Int('A') + remainder), name)
    end
    return name
end

function _legacy_column_values(sheet, column::Int, first_row::Int, last_row::Int)
    name = _legacy_column_name(column)
    return vec(sheet["$(name)$(first_row):$(name)$(last_row)"])
end

function _legacy_cell_row(cell::AbstractString)
    matched = match(r"^[A-Za-z]+([0-9]+)$", String(cell))
    matched === nothing && throw(ArgumentError("Invalid legacy cell coordinate: $cell"))
    return parse(Int, matched.captures[1])
end

function _legacy_direct_targets(bundle::CampaignBundle, workbook::AbstractString)
    targets = Dict{Tuple{String,String},Tuple{Symbol,Tuple}}()
    XLSX.openxlsx(workbook, mode = "r") do xf
        activities = xf["Activities"]
        technologies = xf["Technologies"]
        for row in bundle.rows
            key = (row.sheet, row.cell)
            source_row = _legacy_cell_row(row.cell)
            if row.sheet == "Activities"
                activity = Symbol(String(activities[source_row, 1]))
                targets[key] = (:activities_netVolumesOrig, (activity, 2050))
            elseif row.sheet == "Technologies"
                technology = Symbol(String(technologies[source_row, 1]))
                field = startswith(row.cell, "O") ? :inv_cost :
                        startswith(row.cell, "AD") ? :vom_cost :
                        startswith(row.cell, "CH") ? :techStock_max :
                        throw(ArgumentError("Unsupported 1108 SSP technology coordinate $(row.cell)."))
                targets[key] = (field, (technology, 2050))
            elseif row.sheet == "NodeParameters"
                row.cell == "H5" || throw(ArgumentError(
                    "Unsupported 1108 SSP node coordinate $(row.cell)."))
                targets[key] = (:emissionTargetAir, (:NL, 2050))
            end
        end
    end
    return targets
end

function _legacy_weather_data(workbook::AbstractString)
    profiles = Dict{Tuple{Int,Int,Symbol},Float64}()
    prices = Dict{Tuple{Int,Int},Float64}()
    cap2act = Dict{Tuple{Int,Symbol},Float64}()
    profile_sources = Dict(
        :var"Built Environment" => (("Hourly data helper", 5), ("Hourly data helper", 10),
                                      ("Hourly data helper", 10), ("Hourly data helper", 10),
                                      ("Hourly data helper", 10), ("Hourly data helper", 10),
                                      ("Hourly data helper", 10)),
        :var"Wind Onshore NL" => (("Hourly data helper", 2), ("Hourly data helper", 7),
                                   ("Extreme WY", 2), ("Extreme WY", 5), ("Extreme WY", 8),
                                   ("Extreme WY", 11), ("Extreme WY", 14)),
        :var"Wind Offshore NL" => (("Hourly data helper", 3), ("Hourly data helper", 8),
                                    ("Extreme WY", 3), ("Extreme WY", 6), ("Extreme WY", 9),
                                    ("Extreme WY", 12), ("Extreme WY", 15)),
        :var"Sun NL" => (("Hourly data helper", 4), ("Hourly data helper", 9),
                          ("Extreme WY", 4), ("Extreme WY", 7), ("Extreme WY", 10),
                          ("Extreme WY", 13), ("Extreme WY", 16)),
    )
    price_sources = (("Hourly data helper", 6), ("Hourly data helper", 11),
                     ("Extreme WY", 17), ("Extreme WY", 18), ("Extreme WY", 19),
                     ("Extreme WY", 20), ("Extreme WY", 21))
    technologies = (:PNL01_14, :PNL01_15, :PNL01_16, :PNL01_17)
    XLSX.openxlsx(workbook, mode = "r") do xf
        for state in 1:7
            for (profile, sources) in profile_sources
                sheet, column = sources[state]
                source = xf[sheet]
                values = _legacy_column_values(source, column, 3, 8762)
                for (hour, value) in pairs(values)
                    profiles[(state, hour, profile)] = _legacy_float(value)
                end
            end
            price_sheet, price_column = price_sources[state]
            source = xf[price_sheet]
            values = _legacy_column_values(source, price_column, 3, 8762)
            for (hour, value) in pairs(values)
                prices[(state, hour)] = _legacy_float(value)
            end
            for (index, technology) in pairs(technologies)
                value = if state == 1
                    xf["Hourly data helper"][index + 2, 16]
                elseif state == 2
                    xf["Hourly data helper"][index + 2, 17]
                else
                    xf["Extreme WY"][index + 12, state + 26]
                end
                cap2act[(state, technology)] = _legacy_float(value)
            end
        end
    end
    return profiles, prices, cap2act
end

function _legacy_weather_data_cached(workbook::AbstractString)
    cache_dir = joinpath(dirname(workbook), ".iesa_cache")
    cache_path = joinpath(cache_dir, "1108_weather_v1_$(_workbook_fingerprint(workbook)).bin")
    if isfile(cache_path)
        return open(deserialize, cache_path)
    end
    data = _legacy_weather_data(workbook)
    mkpath(cache_dir)
    temporary = cache_path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        serialize(io, data)
    end
    mv(temporary, cache_path; force = true)
    return data
end

function _legacy_excel_responses(bundle::CampaignBundle)
    path = joinpath(dirname(bundle.config_path), "1108_excel_response.json")
    isfile(path) || throw(ArgumentError("1108 Excel response artifact not found: $path"))
    raw = JSON3.read(read(path, String))
    Int(raw["schema_version"]) == 1 || throw(ArgumentError(
        "Unsupported 1108 Excel response schema $(raw["schema_version"])."))
    responses = Dict{String,Tuple{Float64,Vector{Legacy1108ResponseEntry}}}()
    for (parameter, response) in pairs(raw["parameters"])
        entries = Legacy1108ResponseEntry[]
        for entry in response["entries"]
            indices = Tuple(item isa AbstractString ? Symbol(String(item)) : Int(item)
                            for item in entry["key"])
            push!(entries, Legacy1108ResponseEntry(
                Symbol(String(entry["field"])), indices, Float64(entry["slope"])))
        end
        responses[String(parameter)] = (Float64(response["baseline"]), entries)
    end
    return responses
end

# Cache keys embed this so editing the workbook invalidates every derived cache.
const _WORKBOOK_FINGERPRINTS = Dict{String,String}()

function _workbook_fingerprint(path::AbstractString)
    key = abspath(path)
    return get!(_WORKBOOK_FINGERPRINTS, key) do
        bytes2hex(SHA.sha256(read(key)))[1:12]
    end
end

function _legacy_1108_context_cache_path(bundle::CampaignBundle)
    response_path = joinpath(dirname(bundle.config_path), "1108_excel_response.json")
    workbook_hash = _workbook_fingerprint(bundle.input_workbook)
    config_hash = bytes2hex(SHA.sha256(read(bundle.config_path)))[1:12]
    response_hash = bytes2hex(SHA.sha256(read(response_path)))[1:12]
    cache_dir = joinpath(dirname(bundle.input_workbook), ".iesa_cache")
    filename = "1108_context_v$(_LEGACY_1108_CONTEXT_CACHE_VERSION)_$(workbook_hash)_$(config_hash)_$(response_hash).bin"
    return joinpath(cache_dir, filename)
end

function _load_1108_context_uncached(bundle::CampaignBundle)
    basename(bundle.input_workbook) == "1108 SSP.xlsx" || throw(ArgumentError(
        "The legacy campaign requires Input/1108 SSP.xlsx, got $(bundle.input_workbook)."))
    base_md = read_data_cached(bundle.input_workbook)
    direct_targets = _legacy_direct_targets(bundle, bundle.input_workbook)
    profiles, prices, cap2act = _legacy_weather_data_cached(bundle.input_workbook)
    responses = _legacy_excel_responses(bundle)
    cluster_cache_dir = joinpath(dirname(bundle.input_workbook), ".iesa_cache", "1108_clusters")
    return Legacy1108Context(bundle, base_md, unique_parameters(campaign_spec(bundle, "lhs")),
                             direct_targets, profiles, prices, cap2act, responses,
                             cluster_cache_dir)
end

"""Load the pinned 1108 SSP workbook, responses, and all weather states."""
function load_1108_context(bundle::CampaignBundle)
    path = _legacy_1108_context_cache_path(bundle)
    if isfile(path)
        return open(deserialize, path)
    end
    context = _load_1108_context_uncached(bundle)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        serialize(io, context)
    end
    mv(temporary, path; force = true)
    return context
end

function load_1108_worker_context(bundle::CampaignBundle)
    responses = _legacy_excel_responses(bundle)
    cluster_cache_dir = joinpath(dirname(bundle.input_workbook), ".iesa_cache", "1108_clusters")
    return Legacy1108Context(
        bundle, nothing, unique_parameters(campaign_spec(bundle, "lhs")),
        Dict{Tuple{String,String},Tuple{Symbol,Tuple}}(),
        Dict{Tuple{Int,Int,Symbol},Float64}(),
        Dict{Tuple{Int,Int},Float64}(),
        Dict{Tuple{Int,Symbol},Float64}(), responses, cluster_cache_dir)
end

function _legacy_1108_cluster_cache_path(context::Legacy1108Context,
                                         representative_days::Int,
                                         weather_state::Int)
    1 <= weather_state <= 7 || throw(ArgumentError("weather_state must be in 1:7."))
    representative_days > 0 || throw(ArgumentError("representative_days must be positive."))
    workbook_hash = _workbook_fingerprint(context.bundle.input_workbook)
    filename = "v$(_LEGACY_1108_CLUSTER_CACHE_VERSION)_$(workbook_hash)_wy$(weather_state)_rd$(representative_days).bin"
    return joinpath(context.cluster_cache_dir, filename)
end

function _write_legacy_1108_cluster_cache(path::AbstractString, md::ModelData)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        serialize(io, md)
    end
    mv(temporary, path; force = true)
    return path
end

function build_1108_cluster_cache!(context::Legacy1108Context,
                                   representative_days::Int,
                                   weather_state::Int;
                                   periods::Vector{Int} = [2050],
                                   force::Bool = false)
    path = _legacy_1108_cluster_cache_path(context, representative_days, weather_state)
    isfile(path) && !force && return path
    md = deepcopy(context.base_md)
    _apply_legacy_weather!(md, context, Float64(weather_state), 1.0)
    md.sets.periods_solve = copy(periods)
    md.params.n_repDays = representative_days
    derive_sets!(md)
    compute_derived_params!(md)
    build_temporal_clusters!(md)
    return _write_legacy_1108_cluster_cache(path, md)
end

function load_1108_cluster_cache(context::Legacy1108Context,
                                 representative_days::Int,
                                 weather_state::Int)
    path = _legacy_1108_cluster_cache_path(context, representative_days, weather_state)
    isfile(path) || throw(ArgumentError(
        "1108 cluster cache is missing for weather $weather_state and $representative_days representative days: $path"))
    key = (abspath(path), representative_days, weather_state)
    return get!(_LEGACY_1108_WORKER_CLUSTER_CACHE, key) do
        open(deserialize, path)
    end
end

function verify_1108_cluster_caches(context::Legacy1108Context,
                                    representative_days::Int)
    missing = String[]
    for weather_state in 1:7
        path = _legacy_1108_cluster_cache_path(
            context, representative_days, weather_state)
        isfile(path) || push!(missing, path)
    end
    isempty(missing) || throw(ArgumentError(
        "Missing 1108 weather cluster caches:\n$(join(missing, '\n'))\n" *
        "Run scripts/precompute_1108_clusters.jl first."))
    return true
end

function _legacy_1108_cluster_cache_worker(task_channel::RemoteChannel,
                                           event_channel::RemoteChannel,
                                           config_path::AbstractString,
                                           force::Bool)
    bundle = load_campaign_bundle(config_path; verify_input = false)
    context = load_1108_context(bundle)
    while true
        task = take!(task_channel)
        task === nothing && break
        representative_days, weather_state = task
        started = time()
        try
            path = build_1108_cluster_cache!(context, representative_days,
                weather_state; force = force)
            put!(event_channel, (:completed, Distributed.myid(), representative_days,
                                 weather_state, path, time() - started, nothing))
        catch err
            put!(event_channel, (:failed, Distributed.myid(), representative_days,
                                 weather_state, "", time() - started,
                                 sprint(showerror, err, catch_backtrace())))
        end
    end
    return nothing
end

function _legacy_set_or_multiply!(md::ModelData, field::Symbol, indices::Tuple,
                                  value::Float64, operation::Symbol)
    dictionary = getproperty(md.params, field)
    key = _scalar_key(indices)
    dictionary[key] = operation === :multiply ? dictionary[key] * value : value
    return nothing
end

function _apply_legacy_excel_responses!(md::ModelData, context::Legacy1108Context,
                                        values::Dict{String,Float64})
    excluded = Set(("Electricity Trade Price", "Weather Conditions",
                    "Electricity Trade Ratio", "Electricity Trade Volume"))
    for (parameter, value) in values
        parameter in excluded && continue
        haskey(context.responses, parameter) || throw(ArgumentError(
            "No SSDashboard Excel response is defined for '$parameter'."))
        baseline, entries = context.responses[parameter]
        delta = value - baseline
        delta == 0 && continue
        for entry in entries
            dictionary = getproperty(md.params, entry.field)
            key = _scalar_key(entry.indices)
            dictionary[key] = get(dictionary, key, 0.0) + entry.slope * delta
        end
    end
    return nothing
end

function _apply_legacy_weather!(md::ModelData, context::Legacy1108Context,
                                weather_value::Float64, price_multiplier::Float64)
    state = clamp(floor(Int, weather_value), 1, 7)
    p = md.params
    for ((candidate_state, hour, profile), value) in context.weather_profiles
        candidate_state == state || continue
        p.hourly_profilesReadOrig[(hour, profile)] = value
    end
    for hour in 1:8760
        p.interconnectedHourly_pricesOrig[(hour, :var"Electricity EU", 2050)] =
            context.weather_prices[(state, hour)] * price_multiplier
    end
    for ((candidate_state, technology), value) in context.weather_cap2act
        candidate_state == state || continue
        p.cap2act[technology] = value
    end
    return nothing
end

function _apply_legacy_clustered_price!(md::ModelData, multiplier::Float64)
    for key in keys(md.params.interconnectedHourly_prices_cluster)
        md.params.interconnectedHourly_prices_cluster[key] *= multiplier
    end
    return nothing
end

"""
    prepare_1108_variant(context, sampled_values) -> ModelData

Create one variant from an immutable copy of the pinned 1108 SSP baseline.
`sampled_values` follows the 31-column order in `context.parameter_names`.
No workbook is written or recalculated.
"""
function prepare_1108_variant(context::Legacy1108Context,
                              sampled_values::AbstractVector{<:Real};
                              clustered_template::Union{Nothing,ModelData} = nothing)
    length(sampled_values) == length(context.parameter_names) || throw(DimensionMismatch(
        "Expected $(length(context.parameter_names)) sampled values, got $(length(sampled_values))."))
    values = Dict(name => Float64(value)
                  for (name, value) in zip(context.parameter_names, sampled_values))
    if clustered_template === nothing && context.base_md === nothing
        error("A lightweight 1108 worker context requires a clustered template.")
    end
    md = deepcopy(something(clustered_template, context.base_md))
    _apply_legacy_excel_responses!(md, context, values)

    p = md.params
    p.electricity_trade_ratio = values["Electricity Trade Ratio"]
    p.electricity_trade_volume = values["Electricity Trade Volume"]
    if clustered_template === nothing
        _apply_legacy_weather!(md, context, values["Weather Conditions"],
                               values["Electricity Trade Price"])
    end
    compute_derived_params!(md)
    if clustered_template !== nothing
        _apply_legacy_clustered_price!(md, values["Electricity Trade Price"])
        compute_flex_TS_helpers!(md)
    end
    return md
end

function load_1108_coordinator_context(bundle::CampaignBundle)
    cluster_cache_dir = joinpath(dirname(bundle.input_workbook), ".iesa_cache", "1108_clusters")
    return Legacy1108Context(
        bundle, nothing, unique_parameters(campaign_spec(bundle, "lhs")),
        Dict{Tuple{String,String},Tuple{Symbol,Tuple}}(),
        Dict{Tuple{Int,Int,Symbol},Float64}(),
        Dict{Tuple{Int,Int},Float64}(),
        Dict{Tuple{Int,Symbol},Float64}(),
        Dict{String,Tuple{Float64,Vector{Legacy1108ResponseEntry}}}(),
        cluster_cache_dir)
end