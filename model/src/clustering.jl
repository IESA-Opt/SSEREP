# =============================================================================
# clustering.jl — Phase 5: representative-day clustering (TS mode)
#
# Ports IESA-Opt 1.0 procedure `BuildTemporalClusters` (lines ~6776-7300).
#
# Entry point: `build_temporal_clusters!(md::ModelData) -> Nothing`
#
# Inputs (read from md.params / md.sets):
#   - md.params.hourly_profilesReadOrig  (h_orig=1..8760, profile_type) → value
#   - md.params.n_repDays                 number of rep days to build
#   - md.params.hoursPer_day_cluster      slots per rep-day (typically 24)
#   - md.params.clustering_approach       :kmeans_avg | :kmeans_shape | :hull | :maxdiss
#
# Outputs (written to md):
#   - md.sets.hours_cluster               1..(n_repDays * hoursPer_day_cluster)
#   - md.sets.hours_inDay_cluster         1..hoursPer_day_cluster
#   - md.sets.repDays                     1..n_repDays
#   - md.params.mapDay_repDay             d → rd
#   - md.params.dayWeight                 rd → # cal days assigned (sum=365)
#   - md.params.clusterHourWeight         hc → # real hours represented
#   - md.params.repDay_of_clusterHour     hc → rd
#   - md.params.intradaySlot_of_clusterHour  hc → slot ∈ 1..hoursPer_day_cluster
#   - md.params.dayMix_weight             (d, rd) → weight (hard or soft)
#   - md.params.hourly_profiles_cluster   (hc, yp) → value (mean over assigned days)
#   - md.params.hourly_profiles_clusterMedoid    (hc, yp)
#   - md.params.hourly_profiles_clusterMin       (hc, yp)
#   - md.params.hourly_profiles_clusterMean      (hc, yp)
#   - md.params.hourly_profiles_clusterStd       (hc, yp)
#   - md.params.interconnectedHourly_prices_cluster  (hc, ain, p)
# =============================================================================

using Clustering: kmeans
using Statistics: mean, std, median
using LinearAlgebra: norm
using Parquet2
using DataFrames: DataFrame, eachrow

"""
    build_temporal_clusters!(md::ModelData) -> Nothing

Run the rep-day clustering pipeline and populate all `*_cluster` parameters.
Idempotent: clears prior cluster state before rebuilding.
"""
function build_temporal_clusters!(md::ModelData)
    s, p = md.sets, md.params
    n_rd = p.n_repDays
    hpd_c = p.hoursPer_day_cluster
    n_rd > 0 || error("n_repDays must be > 0")
    hpd_c > 0 || error("hoursPer_day_cluster must be > 0")

    # 1. Reset cluster outputs
    _reset_cluster_state!(md)

    # 2. Index sets
    s.hours_inDay_cluster = collect(1:hpd_c)
    s.repDays             = collect(1:n_rd)
    s.hours_cluster       = collect(1:(n_rd * hpd_c))

    # 3. Index mappings hc ↔ (rd, slot)
    for hc in s.hours_cluster
        rd   = div(hc - 1, hpd_c) + 1
        slot = mod(hc - 1, hpd_c) + 1
        p.repDay_of_clusterHour[hc]        = rd
        p.intradaySlot_of_clusterHour[hc]  = slot
    end

    # 3a. Quarter-hour window set + mapping (IESA-Opt 1.0 line 86 / 113)
    hpc = max(p.hoursPer_quarter_cluster, 1)
    n_qc = max(1, Int(floor(n_rd * hpd_c / hpc)))
    s.q_hourWindow_cluster = collect(1:n_qc)
    empty!(p.quarterPer_clusterHour)
    for hc in s.hours_cluster
        p.quarterPer_clusterHour[hc] = ceil(Int, hc / hpc)
    end

    # 4. Build 365 × (24 × n_profiles) shape matrix from hourly_profilesReadOrig
    profile_types_used = _profile_types_in_profiles(p.hourly_profilesReadOrig)
    isempty(profile_types_used) && @warn "No hourly profiles read — clustering will produce flat rep-days."
    n_days_orig = 365
    n_hours_orig_per_day = 8760 ÷ n_days_orig
    Xmat = _build_day_shape_matrix(p.hourly_profilesReadOrig,
                                    profile_types_used,
                                    n_days_orig,
                                    n_hours_orig_per_day)
    # Xmat: features × days  (features = profiles × intraday-hours)

    # 5. Run k-means OR load external cluster map (IESA-Opt 1.0-exported parquet) when
    #    external_clusterMap_path is set. The external path takes precedence and
    #    skips Julia's k-means AND extreme-day promotion (the IESA-Opt 1.0-side
    #    ApplyExtremePeriods is already baked into the map).
    use_external = !isempty(p.external_clusterMap_path)
    if use_external
        assignments_d_to_rd, n_rd_eff = _load_external_cluster_map(p.external_clusterMap_path,
                                                                    n_days_orig)
        # Rebuild dependent sets with the EFFECTIVE n_rd from the external map
        if n_rd_eff != n_rd
            @info "build_temporal_clusters!: external_clusterMap overrides n_repDays" requested = n_rd loaded = n_rd_eff
            p.n_repDays = n_rd_eff
            n_rd = n_rd_eff
            s.repDays       = collect(1:n_rd)
            s.hours_cluster = collect(1:(n_rd * hpd_c))
            empty!(p.repDay_of_clusterHour)
            empty!(p.intradaySlot_of_clusterHour)
            for hc in s.hours_cluster
                rd   = div(hc - 1, hpd_c) + 1
                slot = mod(hc - 1, hpd_c) + 1
                p.repDay_of_clusterHour[hc]        = rd
                p.intradaySlot_of_clusterHour[hc]  = slot
            end
            hpc2 = max(p.hoursPer_quarter_cluster, 1)
            n_qc2 = max(1, Int(floor(n_rd * hpd_c / hpc2)))
            s.q_hourWindow_cluster = collect(1:n_qc2)
            empty!(p.quarterPer_clusterHour)
            for hc in s.hours_cluster
                p.quarterPer_clusterHour[hc] = ceil(Int, hc / hpc2)
            end
        end
    else
        assignments_d_to_rd = _run_clustering(p, Xmat, profile_types_used,
                                              n_days_orig, n_hours_orig_per_day,
                                              n_rd, p.clustering_approach)
    end

    # 6. Compute mapDay_repDay + dayWeight + clusterHourWeight + dayMix_weight (hard)
    for d in 1:n_days_orig
        rd = assignments_d_to_rd[d]
        p.mapDay_repDay[d]         = rd
        p.dayMix_weight[(d, rd)]   = 1.0
        p.dayWeight[rd]            = get(p.dayWeight, rd, 0.0) + 1.0
    end
    if !use_external
        _repair_empty_and_reorder_clusters!(md, n_days_orig, n_rd)
    end
    _recompute_cluster_weights!(md, n_days_orig, hpd_c)

    # 6b. Extreme periods augmentation (5 peak / 1 dunkelflaute / etc.)
    #     SKIPPED when external_clusterMap_path is set — IESA-Opt 1.0 already
    #     ApplyExtremePeriods is already baked into the loaded mapDay_repDay.
    if p.ts_extremePeriods && !use_external
        _apply_extreme_periods!(md, profile_types_used, n_days_orig, n_hours_orig_per_day, hpd_c)
    end

    # 7. Aggregate profiles per cluster hour (Mean, Std, Min, Medoid, Percentile, RankPct)
    _aggregate_profiles_cluster!(p, profile_types_used, n_days_orig, n_hours_orig_per_day, hpd_c)

    # 7a. Indirect-activity resolution (IESA-Opt 1.0 hourly_profiles_clusterResolved /
    #     MedoidResolved / MinResolved / MeanResolved / StdResolved /
    #     PercentileResolved / RankPctResolved). For every iap ∈ activities_indirect,
    #     write resolved aggregates back into the same dicts under key (hc, iap),
    #     so downstream callsites that look up profileType_tech(tb) — which now
    #     equals the indirect activity name for techs serving an indirect activity
    #     — find the correct blended profile.
    _resolve_indirect_cluster_profiles!(md)

    # 7b. Compute envelope + autoblend α + capBound (MEB autoblend pipeline)
    _compute_envelope_autoblend_capbound!(p, profile_types_used)

    # 8. Aggregate interconnected hourly prices similarly
    _aggregate_prices_cluster!(p, n_days_orig, n_hours_orig_per_day, hpd_c)

    # 9. Cluster-dependent flex helper parameters (IESA-Opt 1.0 RHS aggregates / EV helpers)
    compute_flex_TS_helpers!(md)

    @info "build_temporal_clusters! complete" n_repDays = length(s.repDays) hours_cluster = length(s.hours_cluster) extreme_on = p.ts_extremePeriods
    return nothing
end

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

function _reset_cluster_state!(md::ModelData)
    s, p = md.sets, md.params
    empty!(s.hours_cluster)
    empty!(s.repDays)
    empty!(s.hours_inDay_cluster)
    empty!(p.mapDay_repDay)
    empty!(p.dayWeight)
    empty!(p.clusterHourWeight)
    empty!(p.repDay_of_clusterHour)
    empty!(p.intradaySlot_of_clusterHour)
    empty!(p.dayMix_weight)
    empty!(p.hourly_profiles_cluster)
    empty!(p.hourly_profiles_clusterMedoid)
    empty!(p.hourly_profiles_clusterMin)
    empty!(p.hourly_profiles_clusterMean)
    empty!(p.hourly_profiles_clusterStd)
    empty!(p.hourly_profiles_clusterPercentile)
    empty!(p.hourly_profiles_clusterRankPct)
    empty!(p.hourly_profiles_clusterAutoBlend)
    empty!(p.hourly_profiles_clusterCapBound)
    empty!(p.interconnectedHourly_prices_cluster)
end

function _profile_types_in_profiles(prof::Dict{Tuple{Int,Symbol},Float64})
    yps = Set{Symbol}()
    for ((_, yp), _) in prof
        push!(yps, yp)
    end
    return sort!(collect(yps); by = String)
end

function _build_day_shape_matrix(prof::Dict{Tuple{Int,Symbol},Float64},
                                  yps::Vector{Symbol},
                                  n_days::Int,
                                  hours_per_day::Int)
    n_features = length(yps) * hours_per_day
    n_features = max(1, n_features)
    X = zeros(Float64, n_features, n_days)
    for d in 1:n_days
        for (j, yp) in enumerate(yps)
            for hh in 1:hours_per_day
                h_orig = (d - 1) * hours_per_day + hh
                v = get(prof, (h_orig, yp), 0.0)
                X[(j - 1) * hours_per_day + hh, d] = v
            end
        end
    end
    return X
end

function _run_clustering(p::ModelParams,
                         X::AbstractMatrix{Float64},
                         yps::Vector{Symbol},
                         n_days::Int,
                         hours_per_day::Int,
                         n_clusters::Int,
                         approach::Symbol)
    n_days = size(X, 2)
    if n_clusters >= n_days
        # Degenerate: every day is its own cluster
        return collect(1:n_days)
    end
    if approach == :kmeans_avg
        return _run_iesaopt10_kmeans_avg(p, yps, n_days, hours_per_day, n_clusters)
    end
    # Normalize columns for shape-based clustering
    Xc = approach == :kmeans_shape ? _normalize_columns(X) : copy(X)
    # Use deterministic seed via repeatable init
    res = kmeans(Xc, n_clusters; maxiter = 200, display = :none)
    return res.assignments
end

function _run_iesaopt10_kmeans_avg(p::ModelParams,
                               yps::Vector{Symbol},
                               n_days::Int,
                               hours_per_day::Int,
                               n_clusters::Int)
    active_yps = _iesaopt10_clustering_profile_yps(p, yps, n_days, hours_per_day; avg_family = true)
    isempty(active_yps) && return [mod(d - 1, n_clusters) + 1 for d in 1:n_days]

    day_avg = zeros(Float64, length(active_yps), n_days)
    for d in 1:n_days, (j, yp) in enumerate(active_yps)
        day_avg[j, d] = _daily_profile_avg(p, d, yp, hours_per_day)
    end

    centroids = zeros(Float64, length(active_yps), n_clusters)
    for rd in 1:n_clusters
        init_day = _iesaopt10_initial_cluster_day(rd, n_clusters, n_days)
        centroids[:, rd] .= day_avg[:, init_day]
    end

    assignments = Vector{Int}(undef, n_days)
    for d in 1:n_days
        assignments[d] = _nearest_centroid(day_avg, d, centroids)
    end

    for _iter in 1:50
        changed = false
        for rd in 1:n_clusters
            members = findall(==(rd), assignments)
            isempty(members) && continue
            for j in axes(day_avg, 1)
                centroids[j, rd] = mean(day_avg[j, members])
            end
        end
        for d in 1:n_days
            new_rd = _nearest_centroid(day_avg, d, centroids)
            if assignments[d] != new_rd
                assignments[d] = new_rd
                changed = true
            end
        end
        changed || break
    end
    return assignments
end

function _iesaopt10_initial_cluster_day(rd::Int, n_clusters::Int, n_days::Int)
    return clamp(round(Int, 0.5 + (rd - 0.5) * n_days / n_clusters), 1, n_days)
end

function _nearest_centroid(features::AbstractMatrix{Float64}, day::Int, centroids::AbstractMatrix{Float64})
    best_rd = 1
    best_dist = Inf
    for rd in axes(centroids, 2)
        dist = 0.0
        for j in axes(features, 1)
            delta = features[j, day] - centroids[j, rd]
            dist += delta * delta
        end
        if dist < best_dist
            best_dist = dist
            best_rd = rd
        end
    end
    return best_rd
end

function _repair_empty_and_reorder_clusters!(md::ModelData, n_days::Int, n_clusters::Int)
    p = md.params
    _refresh_day_mix_from_map!(p, n_days)

    for rd in 1:n_clusters
        if get(p.dayWeight, rd, 0.0) == 0.0
            donor_rd = argmax([get(p.dayWeight, candidate, 0.0) for candidate in 1:n_clusters])
            target = _iesaopt10_initial_cluster_day(rd, n_clusters, n_days)
            donor_days = [d for d in 1:n_days if get(p.mapDay_repDay, d, 0) == donor_rd]
            if length(donor_days) > 1
                target = donor_days[argmin(abs.(donor_days .- target))]
            end
            p.mapDay_repDay[target] = rd
            _refresh_day_mix_from_map!(p, n_days)
        end
    end

    mean_pos = Dict{Int,Float64}()
    for rd in 1:n_clusters
        days = [d for d in 1:n_days if get(p.mapDay_repDay, d, 0) == rd]
        mean_pos[rd] = isempty(days) ? rd * n_days / n_clusters : mean(days)
    end
    ordered = sort!(collect(1:n_clusters); by = rd -> (mean_pos[rd], rd))
    remap = Dict(old_rd => new_rd for (new_rd, old_rd) in enumerate(ordered))
    for d in 1:n_days
        old_rd = get(p.mapDay_repDay, d, 0)
        p.mapDay_repDay[d] = get(remap, old_rd, old_rd)
    end
    _refresh_day_mix_from_map!(p, n_days)
    return nothing
end

function _refresh_day_mix_from_map!(p::ModelParams, n_days::Int)
    empty!(p.dayWeight)
    empty!(p.dayMix_weight)
    for d in 1:n_days
        rd = get(p.mapDay_repDay, d, 0)
        rd == 0 && continue
        p.dayMix_weight[(d, rd)] = 1.0
        p.dayWeight[rd] = get(p.dayWeight, rd, 0.0) + 1.0
    end
    return nothing
end

function _recompute_cluster_weights!(md::ModelData, n_days::Int, hpd_c::Int)
    p = md.params
    _refresh_day_mix_from_map!(p, n_days)
    empty!(p.clusterHourWeight)
    rhpc = 24.0 / hpd_c
    for hc in md.sets.hours_cluster
        rd = p.repDay_of_clusterHour[hc]
        p.clusterHourWeight[hc] = get(p.dayWeight, rd, 0.0) * rhpc
    end
    return nothing
end

function _normalize_columns(X::AbstractMatrix{Float64})
    Y = copy(X)
    for j in 1:size(Y, 2)
        c = view(Y, :, j)
        m = mean(c)
        c .-= m
        nrm = norm(c)
        nrm > 1e-12 && (c ./= nrm)
    end
    return Y
end

function _iesaopt10_clustering_profile_yps(p::ModelParams,
                                       yps::Vector{Symbol},
                                       n_days::Int,
                                       hours_per_day::Int;
                                       avg_family::Bool)
    forced = Set([
        "EU Load", "Standard Load", "Built Environment", "Electric Vehicles",
        "Electric LDV", "Electric HDV", "Wind Onshore NL", "Wind Offshore NL",
        "Sun NL",
    ])
    active = Symbol[]
    for yp in yps
        lo = Inf
        hi = -Inf
        if avg_family
            for d in 1:n_days
                total = 0.0
                for hh in 1:hours_per_day
                    h_orig = (d - 1) * hours_per_day + hh
                    total += get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0)
                end
                v = total / hours_per_day
                lo = min(lo, v)
                hi = max(hi, v)
            end
        else
            for d in 1:n_days, hh in 1:hours_per_day
                h_orig = (d - 1) * hours_per_day + hh
                v = get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0)
                lo = min(lo, v)
                hi = max(hi, v)
            end
        end
        if (isfinite(lo) && hi - lo > 0.05) || String(yp) in forced
            push!(active, yp)
        end
    end
    return active
end

function _daily_profile_avg(p::ModelParams, d::Int, yp::Symbol, hours_per_day::Int)
    total = 0.0
    for hh in 1:hours_per_day
        h_orig = (d - 1) * hours_per_day + hh
        total += get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0)
    end
    return total / hours_per_day
end

function _aggregate_profiles_cluster!(p::ModelParams,
                                       yps::Vector{Symbol},
                                       n_days::Int,
                                       hours_per_day::Int,
                                       hpd_c::Int)
    # Group days by rep-day
    rd_to_days = Dict{Int,Vector{Int}}()
    for d in 1:n_days
        rd = get(p.mapDay_repDay, d, 0)
        rd == 0 && continue
        push!(get!(() -> Int[], rd_to_days, rd), d)
    end

    # Find medoid day per rep-day. IESA-Opt 1.0 uses clustering_profile_flag(yp), which
    # combines variability filtering with a forced profile list. For kmeans_avg,
    # maxdiss, and hull-style methods the medoid distance is computed on daily
    # average profile vectors; shape-family methods use the full 24h signature.
    medoid_day_of = Dict{Int,Int}()
    avg_family = p.clustering_approach in (:kmeans_avg, :maxdiss, :hull_convex, :hull_conical, :hierarchical_avg)
    medoid_yps = _iesaopt10_clustering_profile_yps(p, yps, n_days, hours_per_day; avg_family = avg_family)
    for (rd, ds) in rd_to_days
        isempty(ds) && continue
        n_feat = length(medoid_yps) * (avg_family ? 1 : hours_per_day)
        n_feat == 0 && (medoid_day_of[rd] = ds[1]; continue)
        centroid = zeros(Float64, n_feat)
        if avg_family
            for d in ds, (j, yp) in enumerate(medoid_yps)
                centroid[j] += _daily_profile_avg(p, d, yp, hours_per_day)
            end
        else
            for d in ds, (j, yp) in enumerate(medoid_yps), hh in 1:hours_per_day
                h_orig = (d - 1) * hours_per_day + hh
                centroid[(j - 1) * hours_per_day + hh] += get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0)
            end
        end
        centroid ./= length(ds)
        bestd  = typemax(Float64)
        bestid = ds[1]
        for d in ds
            dist2 = 0.0
            if avg_family
                for (j, yp) in enumerate(medoid_yps)
                    delta = _daily_profile_avg(p, d, yp, hours_per_day) - centroid[j]
                    dist2 += delta * delta
                end
            else
                for (j, yp) in enumerate(medoid_yps), hh in 1:hours_per_day
                    h_orig = (d - 1) * hours_per_day + hh
                    v = get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0)
                    delta = v - centroid[(j - 1) * hours_per_day + hh]
                    dist2 += delta * delta
                end
            end
            if dist2 < bestd
                bestd  = dist2
                bestid = d
            end
        end
        medoid_day_of[rd] = bestid
    end

    real_per_slot = max(1, hours_per_day ÷ hpd_c)
    pct_q = clamp(p.ts_capacityProfile_percentileQ, 0.0, 1.0)
    rpQ   = clamp(p.ts_capacityProfile_percentileQ, 0.0, 1.0)  # same Q parameter for RankPct

    for (rd, ds) in rd_to_days
        isempty(ds) && continue
        medoid_d = medoid_day_of[rd]
        for slot in 1:hpd_c
            hc = (rd - 1) * hpd_c + slot
            for yp in yps
                # Real hours covered by this slot, within one day
                real_h_start = (slot - 1) * real_per_slot + 1
                real_h_end   = min(slot * real_per_slot, hours_per_day)
                vals = Float64[]
                for d in ds, hh in real_h_start:real_h_end
                    h_orig = (d - 1) * hours_per_day + hh
                    push!(vals, get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0))
                end
                isempty(vals) && continue
                mu = mean(vals)
                sd = length(vals) > 1 ? std(vals) : 0.0
                mn = minimum(vals)
                p.hourly_profiles_cluster[(hc, yp)]     = mu
                p.hourly_profiles_clusterMean[(hc, yp)] = mu
                p.hourly_profiles_clusterStd[(hc, yp)]  = sd
                p.hourly_profiles_clusterMin[(hc, yp)]  = mn

                # True medoid: value from the medoid calendar day at the matching slot.
                # (Use the first real-hour in the block; for hpd_c=24 there's only one.)
                h_med = (medoid_d - 1) * hours_per_day + real_h_start
                p.hourly_profiles_clusterMedoid[(hc, yp)] =
                    get(p.hourly_profilesReadOrig, (h_med, yp), mu)

                # Percentile envelope: μ − k·σ (Gaussian quantile)
                k = p.ts_capacityProfile_percentileK
                p.hourly_profiles_clusterPercentile[(hc, yp)] = mu - k * sd

                # Rank-Pct envelope: lower-q quantile by rank interpolation
                if length(vals) >= 2
                    sorted = sort(vals)
                    pos = 1.0 + pct_q * (length(sorted) - 1)
                    lo = floor(Int, pos)
                    hi = ceil(Int, pos)
                    if lo == hi
                        p.hourly_profiles_clusterRankPct[(hc, yp)] = sorted[lo]
                    else
                        frac = pos - lo
                        p.hourly_profiles_clusterRankPct[(hc, yp)] =
                            (1 - frac) * sorted[lo] + frac * sorted[hi]
                    end
                else
                    p.hourly_profiles_clusterRankPct[(hc, yp)] = mn
                end
            end
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# MEB autoblend: compute α(hc, yp) ∈ [floor, cap] and capBound = (1-α)·medoid + α·envelope
# IESA-Opt 1.0 lines 7643-7661.
# Envelope is selected by ts_capacityProfile_envelopeMode:
#   0 → Min      (most conservative)
#   1 → Percentile (μ − k·σ Gaussian quantile)
#   2 → RankPct  (rank-interpolated lower-q quantile)
# ----------------------------------------------------------------------------

function _compute_envelope_autoblend_capbound!(p::ModelParams, yps::Vector{Symbol})
    empty!(p.hourly_profiles_clusterAutoBlend)
    empty!(p.hourly_profiles_clusterCapBound)
    p.ts_capacityProfile_autoFloor_effective = p.ts_capacityProfile_autoFloor

    mode   = p.ts_capacityProfile_envelopeMode
    floorα = max(0.0, p.ts_capacityProfile_autoFloor_effective)
    capα   = min(1.0, p.ts_capacityProfile_autoCap)
    # Auto-mode off ⇒ fall back to plain medoid (α=0) so capBound == medoid
    auto   = p.ts_capacityProfile_autoMode

    for (key, medoid) in p.hourly_profiles_clusterMedoid
        envelope = if mode == 1
            get(p.hourly_profiles_clusterPercentile, key, medoid)
        elseif mode == 2
            get(p.hourly_profiles_clusterRankPct, key, medoid)
        else
            get(p.hourly_profiles_clusterMin, key, medoid)
        end

        α = if !auto
            0.0
        else
            denom = max(abs(medoid), 1e-3)
            spread = (medoid - envelope) / denom   # > 0 when envelope < medoid
            spread = clamp(spread, 0.0, 1.0)
            clamp(spread, floorα, capα)
        end
        p.hourly_profiles_clusterAutoBlend[key] = α
        p.hourly_profiles_clusterCapBound[key]  = (1.0 - α) * medoid + α * envelope
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Indirect-activity resolution
#
# IESA-Opt 1.0 lines 428-562: hourly_profiles_clusterResolved (and MedoidResolved /
# MinResolved / PercentileResolved / RankPctResolved). For each indirect
# activity `iap ∈ activities_indirect`, the resolved value is the
# negative-coefficient-weighted average of the consumers' raw profile aggregates:
#
#   resolved(hc, iap) = Σ_{itb | ab(itb,iap,by)<0} agg(hc, ptr(itb)) * ab(itb,iap,by)
#                     / Σ_{itb | ab(itb,iap,by)<0}                       ab(itb,iap,by)
#
# where `ab` is `activity_balances` at the base year and `ptr` is
# `profileType_techRead`.
#
# Julia stores resolved values back into the same dicts under key (hc, iap),
# so downstream callsites that look up by `profileType_tech(tb)` (which equals
# the indirect activity name for techs serving an indirect activity) find the
# blended profile transparently. Mean/Std are also resolved so the percentile
# envelope is correctly recomputed by `_compute_envelope_autoblend_capbound!`.
# ----------------------------------------------------------------------------

function _resolve_indirect_cluster_profiles!(md::ModelData)
    s, p = md.sets, md.params
    isempty(s.activities_indirect) && return nothing
    isempty(s.hours_cluster)       && return nothing
    by = p.base_year

    # Build per-iap consumer list: (itb, profileType_techRead(itb), abs(ab))
    consumers_of = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
    denom_of     = Dict{Symbol, Float64}()
    for iap in s.activities_indirect
        consumers_of[iap] = Tuple{Symbol,Float64}[]
        denom_of[iap]     = 0.0
    end
    for ((tb, a, ps), coef) in p.activity_balances
        ps == by || continue
        coef >= 0 && continue                 # consumers have negative ab
        haskey(consumers_of, a) || continue   # only indirect activities
        ptr = get(p.profileType_techRead, tb, Symbol(""))
        ptr == Symbol("") && continue
        push!(consumers_of[a], (ptr, coef))
        denom_of[a] += coef
    end

    # For each indirect activity, compute resolved aggregate for every cluster hour
    for iap in s.activities_indirect
        cons  = consumers_of[iap]
        denom = denom_of[iap]
        (isempty(cons) || denom == 0.0) && continue
        for hc in s.hours_cluster
            num_mean   = 0.0
            num_min    = 0.0
            num_medoid = 0.0
            num_std    = 0.0
            num_pct    = 0.0
            num_rank   = 0.0
            for (ptr, ab) in cons
                num_mean   += ab * get(p.hourly_profiles_clusterMean,       (hc, ptr), 0.0)
                num_min    += ab * get(p.hourly_profiles_clusterMin,        (hc, ptr), 0.0)
                num_medoid += ab * get(p.hourly_profiles_clusterMedoid,     (hc, ptr), 0.0)
                num_std    += ab * get(p.hourly_profiles_clusterStd,        (hc, ptr), 0.0)
                num_pct    += ab * get(p.hourly_profiles_clusterPercentile, (hc, ptr), 0.0)
                num_rank   += ab * get(p.hourly_profiles_clusterRankPct,    (hc, ptr), 0.0)
            end
            mean_v   = num_mean   / denom
            min_v    = num_min    / denom
            medoid_v = num_medoid / denom
            std_v    = num_std    / denom
            pct_v    = num_pct    / denom
            rank_v   = num_rank   / denom
            p.hourly_profiles_cluster[(hc, iap)]            = mean_v
            p.hourly_profiles_clusterMean[(hc, iap)]        = mean_v
            p.hourly_profiles_clusterMin[(hc, iap)]         = min_v
            p.hourly_profiles_clusterMedoid[(hc, iap)]      = medoid_v
            p.hourly_profiles_clusterStd[(hc, iap)]         = std_v
            p.hourly_profiles_clusterPercentile[(hc, iap)]  = pct_v
            p.hourly_profiles_clusterRankPct[(hc, iap)]     = rank_v
        end
    end
    return nothing
end


# ----------------------------------------------------------------------------
# Extreme periods: peak hourly net-load, dunkelflaute, VRES surplus,
#                  peak hourly demand, and net-load ramp.
# IESA-Opt 1.0 ApplyExtremePeriods (lines ~8484-8668).
#
# Algorithm:
#   1. Identify up to `ts_extremeDays_count` candidate calendar days by ranking
#      365 days on each criterion.
#   2. For each extreme day E, claim the existing rep-day rd* whose centroid is
#      closest to E's daily-average profile, skipping rep-days claimed by prior
#      extremes.
#   3. Evict the other days currently assigned to rd* to their next-best
#      unclaimed rep-day, then assign E to rd*. Recompute weights without
#      changing n_repDays or the rep-day set.
# ----------------------------------------------------------------------------

function _apply_extreme_periods!(md::ModelData,
                                  yps::Vector{Symbol},
                                  n_days::Int,
                                  hours_per_day::Int,
                                  hpd_c::Int)
    s, p = md.sets, md.params
    n_ex = p.ts_extremeDays_count
    n_ex <= 0 && return
    n_ex = min(n_ex, max(0, length(s.repDays) - 2))
    n_ex <= 0 && return

    demand_names = Set([
        "EU Load", "Standard Load", "Built Environment", "Electric Vehicles",
        "Electric LDV", "Electric HDV",
    ])
    res_names = Set(["Wind Onshore NL", "Wind Offshore NL", "Sun NL"])
    yp_demand_list = [yp for yp in yps if String(yp) in demand_names]
    yp_res_list = [yp for yp in yps if String(yp) in res_names]
    isempty(yp_demand_list) && isempty(yp_res_list) && return

    @info "_apply_extreme_periods!: profile detection" demand_n = length(yp_demand_list) res_n = length(yp_res_list)

    hour_peak_netload = fill(-Inf, n_days)
    hour_min_netload = fill(Inf, n_days)
    hour_peak_demand = fill(-Inf, n_days)
    hour_min_res = fill(Inf, n_days)
    for d in 1:n_days
        for hh in 1:hours_per_day
            h_orig = (d - 1) * hours_per_day + hh
            demand = sum(get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0) for yp in yp_demand_list)
            res = sum(get(p.hourly_profilesReadOrig, (h_orig, yp), 0.0) for yp in yp_res_list)
            netload = demand - res
            hour_peak_netload[d] = max(hour_peak_netload[d], netload)
            hour_min_netload[d] = min(hour_min_netload[d], netload)
            hour_peak_demand[d] = max(hour_peak_demand[d], demand)
            hour_min_res[d] = min(hour_min_res[d], res)
        end
    end
    hour_ramp = hour_peak_netload .- hour_min_netload

    active_yps = _iesaopt10_clustering_profile_yps(p, yps, n_days, hours_per_day; avg_family = true)
    isempty(active_yps) && return

    day_avg = Dict{Tuple{Int,Symbol},Float64}()
    for d in 1:n_days, yp in active_yps
        day_avg[(d, yp)] = _daily_profile_avg(p, d, yp, hours_per_day)
    end

    centroids = Dict{Tuple{Int,Symbol},Float64}()
    for rd in s.repDays
        ds = [d for d in 1:n_days if get(p.mapDay_repDay, d, 0) == rd]
        for yp in active_yps
            centroids[(rd, yp)] = isempty(ds) ? 0.0 : mean(day_avg[(d, yp)] for d in ds)
        end
    end

    function distance_day_to_rd(d::Int, rd::Int, centroids)
        total = 0.0
        for yp in active_yps
            diff = day_avg[(d, yp)] - get(centroids, (rd, yp), 0.0)
            total += diff * diff
        end
        return total
    end

    function set_day_rd!(d::Int, rd::Int)
        p.mapDay_repDay[d] = rd
        for k in collect(keys(p.dayMix_weight))
            k[1] == d && delete!(p.dayMix_weight, k)
        end
        p.dayMix_weight[(d, rd)] = 1.0
    end

    claimed_rd = Set{Int}()
    claimed_day = Set{Int}()
    chosen = Tuple{Int,String}[]
    for pass_num in 1:n_ex
        extreme_day = 0
        label = ""
        if pass_num == 1
            extreme_day = _arg_extreme_day(hour_peak_netload, claimed_day; rev = true)
            label = "peak_netload"
        elseif pass_num == 2
            if !isempty(yp_res_list)
                extreme_day = _arg_extreme_day(hour_min_res, claimed_day; rev = false)
                label = "dunkelflaute"
            end
        elseif pass_num == 3
            if !isempty(yp_res_list)
                extreme_day = _arg_extreme_day(hour_min_netload, claimed_day; rev = false)
                label = "VRES_surplus"
            end
        elseif pass_num == 4
            if !isempty(yp_demand_list)
                extreme_day = _arg_extreme_day(hour_peak_demand, claimed_day; rev = true)
                label = "peak_demand"
            end
        else
            extreme_day = _arg_extreme_day(hour_ramp, claimed_day; rev = true)
            label = "max_ramp"
        end
        extreme_day == 0 && continue
        push!(claimed_day, extreme_day)

        candidates = [rd for rd in s.repDays if !(rd in claimed_rd)]
        isempty(candidates) && break
        target_rd = candidates[argmin([distance_day_to_rd(extreme_day, rd, centroids) for rd in candidates])]

        members = [d for d in 1:n_days if get(p.mapDay_repDay, d, 0) == target_rd && d != extreme_day]
        for d in members
            evict_candidates = [rd for rd in s.repDays if !(rd in claimed_rd) && rd != target_rd]
            isempty(evict_candidates) && continue
            new_rd = evict_candidates[argmin([distance_day_to_rd(d, rd, centroids) for rd in evict_candidates])]
            set_day_rd!(d, new_rd)
        end

        set_day_rd!(extreme_day, target_rd)
        for yp in active_yps
            centroids[(target_rd, yp)] = day_avg[(extreme_day, yp)]
        end
        push!(claimed_rd, target_rd)
        push!(chosen, (extreme_day, label))
    end

    isempty(chosen) && return
    @info "_apply_extreme_periods!: promoting extreme days" days = [d for (d, _) in chosen] labels = [label for (_, label) in chosen]

    if p.ts_extremeWeight > 1.0 && !isempty(claimed_rd)
        pulled = Set{Int}()
        pass = 1
        while pass < p.ts_extremeWeight
            for rd in sort!(collect(claimed_rd))
                candidates = [d for d in 1:n_days if !(d in pulled) && !(d in claimed_day) && !(get(p.mapDay_repDay, d, 0) in claimed_rd)]
                isempty(candidates) && continue
                pull_day = candidates[argmin([distance_day_to_rd(d, rd, centroids) for d in candidates])]
                set_day_rd!(pull_day, rd)
                push!(pulled, pull_day)
            end
            pass += 1
        end
    end

    empty!(p.dayWeight)
    for d in 1:n_days
        rd = p.mapDay_repDay[d]
        p.dayWeight[rd] = get(p.dayWeight, rd, 0.0) + 1.0
    end

    empty!(p.clusterHourWeight)
    rhpc = 24.0 / hpd_c
    for hc in s.hours_cluster
        rd = p.repDay_of_clusterHour[hc]
        p.clusterHourWeight[hc] = get(p.dayWeight, rd, 0.0) * rhpc
    end
    return nothing
end

function _arg_extreme_day(values::Vector{Float64}, claimed_day::Set{Int}; rev::Bool)
    best_day = 0
    best_value = rev ? -Inf : Inf
    for d in eachindex(values)
        d in claimed_day && continue
        v = values[d]
        if (rev && v > best_value) || (!rev && v < best_value)
            best_value = v
            best_day = d
        end
    end
    return best_day
end

# Find the first symbol in `candidates` that is present in `yps`. Returns nothing if none match.
function first_existing(yps::Vector{Symbol}, candidates::Vector{Symbol})
    yps_set = Set(yps)
    for c in candidates
        c in yps_set && return c
    end
    return nothing
end

# Case-insensitive substring match against a Symbol's string form.
_contains_ci(yp::Symbol, needle::AbstractString) = occursin(lowercase(needle), lowercase(String(yp)))

function _aggregate_prices_cluster!(p::ModelParams,
                                     n_days::Int,
                                     hours_per_day::Int,
                                     hpd_c::Int)
    isempty(p.interconnectedHourly_pricesOrig) && return
    # Same logic as profiles, but indexed by (h, ain, p_period)
    rd_to_days = Dict{Int,Vector{Int}}()
    for d in 1:n_days
        rd = get(p.mapDay_repDay, d, 0)
        rd == 0 && continue
        push!(get!(() -> Int[], rd_to_days, rd), d)
    end
    real_per_slot = max(1, hours_per_day ÷ hpd_c)
    # Group by (ain, p_period) and aggregate
    ain_periods = Set{Tuple{Symbol,Int}}()
    for ((_, ain, pp), _) in p.interconnectedHourly_pricesOrig
        push!(ain_periods, (ain, pp))
    end
    for (rd, ds) in rd_to_days, slot in 1:hpd_c
        hc = (rd - 1) * hpd_c + slot
        real_h_start = (slot - 1) * real_per_slot + 1
        real_h_end   = min(slot * real_per_slot, hours_per_day)
        for (ain, pp) in ain_periods
            vals = Float64[]
            for d in ds, hh in real_h_start:real_h_end
                h_orig = (d - 1) * hours_per_day + hh
                v = get(p.interconnectedHourly_pricesOrig, (h_orig, ain, pp), 0.0)
                push!(vals, v)
            end
            isempty(vals) && continue
            p.interconnectedHourly_prices_cluster[(hc, ain, pp)] = mean(vals)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# External cluster-map loader (Julia↔IESA-Opt 1.0 validation)
# ---------------------------------------------------------------------------
"""
    _load_external_cluster_map(parquet_path::AbstractString, n_days::Int)
        -> (Vector{Int}, Int)

Reads an IESA-Opt 1.0-exported `cluster_map.parquet` (columns `calendar_day`, `rep_day`)
and returns `(assignments_d_to_rd, n_rd_eff)` where:

  * `assignments_d_to_rd[d]` is the rep-day index for calendar day `d`
    (renumbered to a contiguous 1..n_rd_eff so downstream sets stay tight).
  * `n_rd_eff` is the number of distinct rep-days that appear in the file.

The IESA-Opt 1.0 file uses 1-based calendar day strings (`"1".."365"`) and float
rep-day ids. We coerce both to Int and require complete coverage of all
`1..n_days` days; missing assignments throw an error.

This is the validation-only path that lets Julia reuse the EXACT rep-day
assignment that IESA-Opt 1.0 produced (kmeans seed 42 + ApplyExtremePeriods eviction),
so the LP matrices coincide and any residual objective gap can only come from
constraint or coefficient differences — not clustering.
"""
function _load_external_cluster_map(parquet_path::AbstractString, n_days::Int)
    isfile(parquet_path) || error("external_clusterMap_path not found: $parquet_path")
    df = DataFrame(Parquet2.Dataset(parquet_path))
    "calendar_day" in names(df) || error("cluster_map parquet missing 'calendar_day' column ($parquet_path)")
    "rep_day"      in names(df) || error("cluster_map parquet missing 'rep_day' column ($parquet_path)")

    raw_assign = Dict{Int,Int}()
    for row in eachrow(df)
        cd_raw = row.calendar_day
        rd_raw = row.rep_day
        (cd_raw === missing || rd_raw === missing) && continue
        d_int  = cd_raw isa AbstractString ? parse(Int, cd_raw) : Int(cd_raw)
        rd_int = rd_raw isa AbstractString ? parse(Int, rd_raw) : Int(round(Float64(rd_raw)))
        raw_assign[d_int] = rd_int
    end
    length(raw_assign) == n_days || error("cluster_map parquet has $(length(raw_assign)) rows; expected $n_days")

    # Renumber rep-day ids to 1..k in the order they first appear (sorted by min calendar day)
    used = sort!(collect(Set(values(raw_assign))))
    rd_remap = Dict(rd => i for (i, rd) in enumerate(used))
    assignments = Vector{Int}(undef, n_days)
    for d in 1:n_days
        haskey(raw_assign, d) || error("cluster_map parquet missing calendar_day=$d")
        assignments[d] = rd_remap[raw_assign[d]]
    end
    return assignments, length(used)
end

