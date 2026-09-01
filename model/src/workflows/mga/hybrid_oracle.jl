# ---------------------------------------------------------------------------
# Exact hybrid ORACLE MGA workflow
#
# Isolated from normal single-run and Scenario Space paths. The UI calls these
# helpers only through /api/mga/*.
# ---------------------------------------------------------------------------

Base.@kwdef mutable struct MGAExactConfig
    directions::Int = 12
    cost_slack::Float64 = 5.0
    oracle_iterations::Int = 2
    oracle_batch::Int = 2
    tolerance::Float64 = 0.1
    workers::Int = 1
    threads::Int = 0
    solver::String = "auto"
    solve_method::String = "barrier_crossover"
    mode::Symbol = :ts
    period::Int = 2050
    representative_days::Int = 15
    hours_per_day::Int = 24
    clustering::Symbol = :kmeans_avg
    extreme_periods::Bool = true
    extreme_days::Int = 5
    boundary_ramping::Bool = true
end

const _MGA_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)

function _mga_halton(index::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0
        result += f * (i % base)
        i = fld(i, base)
        f /= base
    end
    return result
end

function _mga_unit_normalize!(weights::Vector{Float64})
    norm_value = sqrt(sum(abs2, weights))
    norm_value <= eps(Float64) && return weights
    for i in eachindex(weights)
        weights[i] = weights[i] / norm_value
    end
    return weights
end

function _mga_ok_status(model::JuMP.Model)
    term = termination_status(model)
    primal = primal_status(model)
    term in (MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return false
    primal in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT) || return false
    return has_values(model)
end

function _mga_apply_highs_method!(attrs::Dict{String,Any}, method::AbstractString)
    # Note: this mirrors `_apply_highs_method!` in `ui_server.jl` but does NOT
    # block the post-crossover simplex cleanup. MGA needs a usable primal
    # solution to read variable values back out for the design point; the
    # single-run preset accepts `primal_status == NO_SOLUTION` from a stalled
    # crossover, but MGA cannot. Letting cleanup complete is slower but yields
    # a real basis solution every time.
    m = lowercase(String(method))
    if m == "barrier"
        attrs["solver"] = "ipm"
        attrs["run_crossover"] = "off"
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "barrier_crossover" || m == "ipm_crossover"
        attrs["solver"] = "ipm"
        attrs["run_crossover"] = "on"
        attrs["primal_feasibility_tolerance"] = 1e-6
        attrs["dual_feasibility_tolerance"] = 1e-6
        attrs["ipm_optimality_tolerance"] = 1e-4
        attrs["start_crossover_tolerance"] = 1e-4
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "concurrent"
        attrs["solver"] = "choose"
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "primal_simplex"
        attrs["solver"] = "simplex"
        attrs["simplex_strategy"] = 4
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    elseif m == "dual_simplex" || m == "simplex"
        attrs["solver"] = "simplex"
        attrs["simplex_strategy"] = 1
        attrs["simplex_iteration_limit"] = Int(typemax(Int32))
    end
    attrs["output_flag"] = false
    attrs["log_to_console"] = false
    return attrs
end

function _mga_apply_gurobi_method!(attrs::Dict{String,Any}, method::AbstractString)
    m = lowercase(String(method))
    if m == "barrier"
        attrs["Method"] = 2
        attrs["Crossover"] = 0
        attrs["BarHomogeneous"] = 1
    elseif m == "barrier_crossover" || m == "ipm_crossover"
        attrs["Method"] = 2
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    elseif m == "concurrent"
        attrs["Method"] = 3
        attrs["Crossover"] = -1
        attrs["BarHomogeneous"] = 0
    elseif m == "primal_simplex"
        attrs["Method"] = 0
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    elseif m == "dual_simplex" || m == "simplex"
        attrs["Method"] = 1
        attrs["Crossover"] = -1
        delete!(attrs, "BarHomogeneous")
    end
    attrs["OutputFlag"] = 0
    attrs["LogToConsole"] = 0
    return attrs
end

function _mga_resolve_solver(cfg::MGAExactConfig)
    id = lowercase(String(cfg.solver))
    id in ("", "auto") || return id
    isdefined(@__MODULE__, :Gurobi) && return "gurobi"
    return "highs"
end

function _mga_optimizer(cfg::MGAExactConfig)
    solver_id = _mga_resolve_solver(cfg)
    # MGA alternatives run sequentially today, so we give every solve the full
    # CPU thread budget. `cfg.workers` is kept for API compatibility but no
    # longer divides the per-solve thread count.
    threads = max(0, cfg.threads)
    if solver_id == "gurobi"
        attrs = default_gurobi_attributes(; threads = threads, rep_days = cfg.mode == :ts ? cfg.representative_days : nothing)
        _mga_apply_gurobi_method!(attrs, cfg.solve_method)
        return gurobi_optimizer(; attrs = attrs), attrs, "Gurobi", threads
    end
    attrs = default_highs_attributes(; threads = threads)
    _mga_apply_highs_method!(attrs, cfg.solve_method)
    return highs_optimizer(; attrs = attrs), attrs, "HiGHS", threads
end

function _mga_prepare_md(md_source::ModelData, cfg::MGAExactConfig)
    md = deepcopy(md_source)
    period = cfg.period in md.sets.periods ? cfg.period : (2050 in md.sets.periods ? 2050 : last(md.sets.periods))
    cfg.period = period
    md.sets.periods_solve = [period]
    md.params.hoursPer_day = cfg.mode == :ts ? 24 : cfg.hours_per_day
    md.params.n_repDays = max(1, cfg.representative_days)
    md.params.hoursPer_day_cluster = 24
    md.params.clustering_approach = cfg.clustering
    md.params.ts_extremePeriods = cfg.extreme_periods
    md.params.ts_extremeDays_count = max(0, cfg.extreme_days)
    md.params.ts_boundaryRamping = cfg.boundary_ramping
    md.params.ts_capacityProfile_autoMode = true
    md.params.ts_capacityProfile_autoFloor = 0.23
    md.params.ts_capacityProfile_autoCap = 1.00
    md.params.ts_capacityProfile_autoFloor_effective = 0.23
    md.params.ts_capacityProfile_envelopeMode = 0
    md.params.dayMix_softness = 0.0
    md.params.dayMix_weightType = :auto
    derive_sets!(md)
    compute_derived_params!(md)
    cfg.mode == :ts && build_temporal_clusters!(md)
    return md
end

# Build a fresh JuMP model on top of an already prepared `md`. The caller is
# responsible for ensuring `md` has been through `_mga_prepare_md` exactly once
# per campaign so the cluster build is not repeated for every alternative.
function _mga_build_model(md::ModelData, cfg::MGAExactConfig)
    optimizer, attrs, solver_label, threads = _mga_optimizer(cfg)
    model = JuMP.Model(optimizer)
    apply_lp_generation_speedups!(model)
    vars = cfg.mode == :fh ? build_fh_lp!(model, md) : build_ts_lp!(model, md)
    return model, vars, attrs, solver_label, threads
end

function mga_design_groups(md::ModelData; limit::Int = 14)
    counts = Dict{String,Int}()
    for tech in md.sets.technologies
        sector = String(get(md.params.tech_sector, tech, Symbol("Unspecified")))
        counts[sector] = get(counts, sector, 0) + 1
    end
    rows = sort!(collect(counts); by = item -> (-item[2], item[1]))
    isempty(rows) && (rows = [("System", max(1, length(md.sets.technologies)))])
    rows = rows[1:min(limit, length(rows))]
    total = max(1, sum(last.(rows)))
    return [Dict{String,Any}(
        "id" => idx,
        "name" => name,
        "count" => count,
        "share" => round(count / total; digits = 6),
    ) for (idx, (name, count)) in enumerate(rows)]
end

function _mga_group_techs(md::ModelData, groups::Vector{Dict{String,Any}})
    out = Vector{Vector{Symbol}}()
    for group in groups
        name = String(group["name"])
        push!(out, [tech for tech in md.sets.technologies if String(get(md.params.tech_sector, tech, Symbol("Unspecified"))) == name])
    end
    return out
end

function _mga_design_exprs(vars::AnnualVars, md::ModelData, groups::Vector{Dict{String,Any}})
    group_techs = _mga_group_techs(md, groups)
    exprs = Vector{AffExpr}()
    for techs in group_techs
        expr = AffExpr(0.0)
        for tech in techs, ps in md.sets.periods_solve
            add_to_expression!(expr, 1.0, vars.techStock[tech, ps])
            add_to_expression!(expr, 0.25, vars.cap_investments[tech, ps])
        end
        push!(exprs, expr)
    end
    return exprs
end

function _mga_design_values(vars::AnnualVars, md::ModelData, groups::Vector{Dict{String,Any}})
    group_techs = _mga_group_techs(md, groups)
    values = Float64[]
    for techs in group_techs
        total = 0.0
        for tech in techs, ps in md.sets.periods_solve
            total += try
                Float64(value(vars.techStock[tech, ps]))
            catch
                0.0
            end
            investment = try
                Float64(value(vars.cap_investments[tech, ps]))
            catch
                0.0
            end
            total += 0.25 * investment
        end
        push!(values, total)
    end
    return values
end

# Capture per-technology investment decisions from a solved MGA model.
# Returns a vector of {tech, name, sector, stock, investment} dicts,
# filtered to technologies with a non-trivial footprint (stock or new
# investment > 1e-6). Used to feed the "possible investments" insight
# tab in the UI.
function _mga_capture_investments(vars::AnnualVars, md::ModelData)
    out = Vector{Dict{String,Any}}()
    for tech in md.sets.technologies
        stock = 0.0
        invest = 0.0
        for ps in md.sets.periods_solve
            stock += try
                Float64(value(vars.techStock[tech, ps]))
            catch
                0.0
            end
            invest += try
                Float64(value(vars.cap_investments[tech, ps]))
            catch
                0.0
            end
        end
        if stock > 1e-6 || invest > 1e-6
            tech_str = String(tech)
            friendly = String(get(md.params.tech_name, tech, tech_str))
            sector = String(get(md.params.tech_sector, tech, Symbol("Unspecified")))
            push!(out, Dict{String,Any}(
                "tech" => tech_str,
                "name" => isempty(friendly) ? tech_str : friendly,
                "sector" => sector,
                "stock" => round(stock; digits = 6),
                "investment" => round(invest; digits = 6),
            ))
        end
    end
    sort!(out; by = row -> -Float64(row["stock"]))
    return out
end

# Aggregate per-technology investments across the baseline + all solved
# alternatives. For each technology we report:
#   - baselineStock / baselineInvestment: the least-cost reference
#   - min / max / mean / std of stock across alternatives
#   - share: fraction of alternatives that built it (stock > 1e-6)
#   - relativeRange: (max - min) / max(mean, baseline, 1)
#   - category: "low-regret" (built in every alternative with tight range),
#               "high-volatility" (large relative spread across alternatives),
#               "optional" (only some alternatives build it),
#               or "stable" (default; built consistently but unremarkable)
function _mga_investment_spread(baseline_invs::Vector{Dict{String,Any}}, results::Vector{Dict{String,Any}})
    tech_meta = Dict{String,NamedTuple{(:name, :sector),Tuple{String,String}}}()
    for row in baseline_invs
        tech_meta[String(row["tech"])] = (name = String(row["name"]), sector = String(row["sector"]))
    end
    baseline_lookup = Dict{String,Tuple{Float64,Float64}}()
    for row in baseline_invs
        baseline_lookup[String(row["tech"])] = (Float64(row["stock"]), Float64(row["investment"]))
    end
    # Collect per-tech stock / investment across solved alternatives.
    solved = [row for row in results if String(get(row, "status", "")) == "solved" && haskey(row, "investments")]
    alt_count = length(solved)
    stocks_by_tech = Dict{String,Vector{Float64}}()
    invest_by_tech = Dict{String,Vector{Float64}}()
    for row in solved
        invs = row["investments"]
        seen = Set{String}()
        for entry in invs
            tech = String(entry["tech"])
            push!(seen, tech)
            push!(get!(stocks_by_tech, tech, Float64[]), Float64(entry["stock"]))
            push!(get!(invest_by_tech, tech, Float64[]), Float64(entry["investment"]))
            if !haskey(tech_meta, tech)
                tech_meta[tech] = (name = String(get(entry, "name", tech)), sector = String(get(entry, "sector", "Unspecified")))
            end
        end
        # Implicit zeros for techs not reported in this alternative.
        for tech in keys(stocks_by_tech)
            tech in seen && continue
            push!(stocks_by_tech[tech], 0.0)
            push!(invest_by_tech[tech], 0.0)
        end
    end
    # Backfill techs that exist only in the baseline.
    for tech in keys(baseline_lookup)
        haskey(stocks_by_tech, tech) && continue
        stocks_by_tech[tech] = fill(0.0, alt_count)
        invest_by_tech[tech] = fill(0.0, alt_count)
    end
    rows = Vector{Dict{String,Any}}()
    for (tech, stocks) in stocks_by_tech
        invs = get(invest_by_tech, tech, Float64[])
        meta = get(tech_meta, tech, (name = tech, sector = "Unspecified"))
        baseline_stock, baseline_invest = get(baseline_lookup, tech, (0.0, 0.0))
        n = length(stocks)
        if n == 0
            mn = mx = mean = std_val = 0.0
            share = 0.0
        else
            mn = minimum(stocks)
            mx = maximum(stocks)
            mean = sum(stocks) / n
            std_val = n > 1 ? sqrt(max(0.0, sum((s - mean)^2 for s in stocks) / (n - 1))) : 0.0
            share = count(s -> s > 1e-6, stocks) / n
        end
        peak = max(mean, baseline_stock, 1e-9)
        rel_range = (mx - mn) / peak
        # Skip techs that no alternative or baseline ever builds.
        if mx <= 1e-6 && baseline_stock <= 1e-6
            continue
        end
        category = if share >= 0.999 && rel_range <= 0.05
            "low-regret"
        elseif rel_range >= 0.40
            "high-volatility"
        elseif share < 0.999
            "optional"
        else
            "stable"
        end
        push!(rows, Dict{String,Any}(
            "tech" => tech,
            "name" => meta.name,
            "sector" => meta.sector,
            "baselineStock" => round(baseline_stock; digits = 6),
            "baselineInvestment" => round(baseline_invest; digits = 6),
            "min" => round(mn; digits = 6),
            "max" => round(mx; digits = 6),
            "mean" => round(mean; digits = 6),
            "std" => round(std_val; digits = 6),
            "range" => round(mx - mn; digits = 6),
            "relativeRange" => round(rel_range; digits = 6),
            "share" => round(share; digits = 6),
            "alternativeCount" => alt_count,
            "category" => category,
        ))
    end
    sort!(rows; by = r -> -Float64(r["max"]))
    return rows
end

function _mga_normalize(values::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64})
    out = similar(values)
    for i in eachindex(values)
        span = upper[i] - lower[i]
        out[i] = span <= 1e-9 ? 0.5 : clamp((values[i] - lower[i]) / span, 0.0, 1.0)
    end
    return out
end

function _mga_denormalize(point::Vector{Float64}, lower::Vector{Float64}, upper::Vector{Float64}, fallback::Vector{Float64})
    raw = similar(point)
    for i in eachindex(point)
        span = upper[i] - lower[i]
        raw[i] = span <= 1e-9 ? fallback[i] : lower[i] + clamp(point[i], 0.0, 1.0) * span
    end
    return raw
end

function _mga_update_bounds!(lower::Vector{Float64}, upper::Vector{Float64}, values::Vector{Float64})
    for i in eachindex(values)
        isfinite(values[i]) || continue
        lower[i] = min(lower[i], values[i])
        upper[i] = max(upper[i], values[i])
    end
    return nothing
end

function _mga_refresh_rows!(rows::Vector{Dict{String,Any}}, groups::Vector{Dict{String,Any}}, lower::Vector{Float64}, upper::Vector{Float64}, baseline_point::Vector{Float64})
    baseline_norm = _mga_normalize(baseline_point, lower, upper)
    for row in rows
        raw = Vector{Float64}(get(row, "rawPoint", Float64[]))
        length(raw) == length(lower) || continue
        all(isfinite, raw) || continue
        point = _mga_normalize(raw, lower, upper)
        deltas = abs.(point .- baseline_norm)
        row["point"] = [round(v; digits = 6) for v in point]
        row["diversityScore"] = round(maximum(deltas); digits = 6)
        if !isempty(deltas)
            row["dominantGroup"] = groups[argmax(deltas)]["name"]
        end
    end
    return nothing
end

function _mga_inner_points(rows::Vector{Dict{String,Any}}, lower::Vector{Float64}, upper::Vector{Float64}, baseline_point::Vector{Float64})
    points = Vector{Vector{Float64}}([_mga_normalize(baseline_point, lower, upper)])
    for row in rows
        raw = Vector{Float64}(get(row, "rawPoint", Float64[]))
        length(raw) == length(lower) || continue
        all(isfinite, raw) || continue
        push!(points, _mga_normalize(raw, lower, upper))
    end
    return points
end

function _mga_weight_rows(groups::Vector{Dict{String,Any}}, weights::Vector{Float64})
    return [Dict("group" => groups[i]["name"], "weight" => round(weights[i]; digits = 6)) for i in eachindex(weights)]
end

function _mga_direction(groups::Vector{Dict{String,Any}}, id::Int, label::String, phase::String, weights::Vector{Float64}; oracle_iteration::Int = 0, max_error = nothing, coverage_gain::Float64 = 0.0)
    weights = copy(weights)
    _mga_unit_normalize!(weights)
    dominant_idx = argmax(abs.(weights))
    return Dict{String,Any}(
        "id" => id,
        "label" => label,
        "phase" => phase,
        "dominantGroup" => groups[dominant_idx]["name"],
        "weights" => _mga_weight_rows(groups, weights),
        "point" => Float64[],
        "oracleIteration" => oracle_iteration,
        "maxError" => max_error,
        "coverageGain" => round(coverage_gain; digits = 6),
    )
end

function _mga_candidate_point(candidate_id::Int, k::Int)
    return [_mga_halton(candidate_id + 3 * j, _MGA_HALTON_BASES[mod1(j, length(_MGA_HALTON_BASES))]) for j in 1:k]
end

function _mga_min_distance(point::Vector{Float64}, inner_points::Vector{Vector{Float64}})
    isempty(inner_points) && return 1.0
    return minimum(maximum(abs.(point .- inner)) for inner in inner_points)
end

function _mga_error_scan(inner_points::Vector{Vector{Float64}}, k::Int; samples::Int = 384)
    best_point = fill(0.0, k)
    best_distance = -Inf
    for idx in 1:samples
        candidate = _mga_candidate_point(idx, k)
        dist = _mga_min_distance(candidate, inner_points)
        if dist > best_distance
            best_distance = dist
            best_point = candidate
        end
    end
    return best_point, max(best_distance, 0.0)
end

function _mga_planned_directions(groups::Vector{Dict{String,Any}}, cfg::MGAExactConfig)
    k = length(groups)
    rows = Vector{Dict{String,Any}}()
    id = 1
    vmm_target = min(2 * k, max(2, floor(Int, cfg.directions * 0.4)))
    for j in 1:k
        (id > cfg.directions || id > vmm_target) && break
        weights = zeros(Float64, k)
        weights[j] = -1.0
        push!(rows, _mga_direction(groups, id, "VMM lower $(groups[j]["name"])", "vmm", weights))
        id += 1
        (id > cfg.directions || id > vmm_target) && break
        weights = zeros(Float64, k)
        weights[j] = 1.0
        push!(rows, _mga_direction(groups, id, "VMM upper $(groups[j]["name"])", "vmm", weights))
        id += 1
    end

    seed_target = min(cfg.directions, max(id - 1, ceil(Int, cfg.directions * 0.65)))
    halton_idx = 1
    while id <= seed_target
        weights = [2 * _mga_halton(halton_idx + 5 * j, _MGA_HALTON_BASES[mod1(j + 2, length(_MGA_HALTON_BASES))]) - 1 for j in 1:k]
        push!(rows, _mga_direction(groups, id, "Parallel seed direction $halton_idx", "parallel-seed", weights))
        id += 1
        halton_idx += 1
    end

    iter = 1
    local_idx = 1
    while id <= cfg.directions && iter <= cfg.oracle_iterations
        weights = [2 * _mga_halton(halton_idx + 11 * j + 17 * iter, _MGA_HALTON_BASES[mod1(j + 4, length(_MGA_HALTON_BASES))]) - 1 for j in 1:k]
        push!(rows, _mga_direction(groups, id, "ORACLE closest-point refinement $iter.$local_idx", "oracle-refine", weights; oracle_iteration = iter))
        id += 1
        local_idx += 1
        if local_idx > cfg.oracle_batch
            iter += 1
            local_idx = 1
        end
    end

    while id <= cfg.directions
        weights = [2 * _mga_halton(halton_idx + 7 * j, _MGA_HALTON_BASES[mod1(j + 6, length(_MGA_HALTON_BASES))]) - 1 for j in 1:k]
        push!(rows, _mga_direction(groups, id, "Coverage fill direction $halton_idx", "parallel-seed", weights))
        id += 1
        halton_idx += 1
    end
    return rows
end

function _mga_preview_certificate(groups, cfg::MGAExactConfig)
    return Dict{String,Any}(
        "mode" => "exact-solver-hybrid-oracle",
        "initialMaxError" => nothing,
        "estimatedMaxError" => nothing,
        "targetTolerance" => round(cfg.tolerance; digits = 6),
        "iterations" => 0,
        "exploratoryDimensions" => length(groups),
        "innerPoints" => 0,
        "converged" => false,
        "baselineCost" => nothing,
        "costCap" => nothing,
    )
end

function mga_hybrid_oracle_preview(md_source::ModelData, cfg::MGAExactConfig)
    md = _mga_prepare_md(md_source, cfg)
    groups = mga_design_groups(md)
    directions = _mga_planned_directions(groups, cfg)
    return Dict{String,Any}(
        "method" => "Exact Hybrid ORACLE MGA",
        "description" => "Preview builds the exact MGA design. Run solves the original IESA-Opt LP, adds a near-optimal system-cost cap, and computes alternatives with solver-backed MGA objectives and ORACLE closest-point LPs.",
        "groups" => groups,
        "directions" => directions,
        "oracleTrace" => Dict{String,Any}[],
        "certificate" => _mga_preview_certificate(groups, cfg),
    )
end

function mga_hybrid_oracle_preview(md::ModelData; directions::Int = 12, cost_slack::Float64 = 5.0, oracle_iterations::Int = 2, oracle_batch::Int = 2, tolerance::Float64 = 0.1)
    cfg = MGAExactConfig(; directions = directions, cost_slack = cost_slack, oracle_iterations = oracle_iterations, oracle_batch = oracle_batch, tolerance = tolerance)
    return mga_hybrid_oracle_preview(md, cfg)
end

function _mga_solve_baseline(md::ModelData, cfg::MGAExactConfig)
    # First attempt: honor the user-selected solve method.
    model, vars, attrs, solver_label, threads = _mga_build_model(md, cfg)
    started = time()
    optimize!(model)
    solve_seconds = round(time() - started; digits = 3)
    used_method = cfg.solve_method
    if !_mga_ok_status(model)
        # Fallback: rebuild with `barrier` (IPM endpoint, no crossover) which
        # always yields a usable primal point for MGA. This keeps MGA reliable
        # even when the user-selected preset (e.g. `barrier_crossover` with
        # blocked simplex cleanup) cannot return a basis solution.
        original_method = cfg.solve_method
        original_term = string(termination_status(model))
        original_primal = string(primal_status(model))
        cfg.solve_method = "barrier"
        try
            model, vars, attrs, solver_label, threads = _mga_build_model(md, cfg)
            started = time()
            optimize!(model)
            solve_seconds = round(time() - started; digits = 3)
            used_method = "barrier"
        finally
            cfg.solve_method = original_method
        end
        _mga_ok_status(model) || error("MGA baseline solve failed: tried $(original_method) (termination_status=$(original_term), primal_status=$(original_primal)) and barrier fallback (termination_status=$(termination_status(model)), primal_status=$(primal_status(model)))")
    end
    groups = mga_design_groups(md)
    point = _mga_design_values(vars, md, groups)
    investments = _mga_capture_investments(vars, md)
    return Dict{String,Any}(
        "model" => model,
        "vars" => vars,
        "md" => md,
        "groups" => groups,
        "cost" => Float64(objective_value(model)),
        "point" => point,
        "investments" => investments,
        "attrs" => attrs,
        "solver" => solver_label,
        "threadsPerSolve" => threads,
        "solveSeconds" => solve_seconds,
        "method" => used_method,
        "rows" => try num_constraints(model; count_variable_in_set_constraints = false) catch; 0 end,
        "columns" => num_variables(model),
        "terminationStatus" => string(termination_status(model)),
        "primalStatus" => string(primal_status(model)),
    )
end

function _mga_weight_vector(direction::Dict{String,Any}, k::Int)
    weights = zeros(Float64, k)
    rows = get(direction, "weights", Any[])
    for i in 1:min(k, length(rows))
        weights[i] = Float64(get(rows[i], "weight", 0.0))
    end
    return weights
end

function _mga_direction_objective(exprs::Vector{AffExpr}, weights::Vector{Float64})
    objective = AffExpr(0.0)
    for i in eachindex(exprs)
        add_to_expression!(objective, -weights[i], exprs[i])
    end
    return objective
end

function _mga_solve_alternative(md::ModelData, cfg::MGAExactConfig, direction::Dict{String,Any}, baseline_cost::Float64, cost_cap::Float64, lower::Vector{Float64}, upper::Vector{Float64}, baseline_point::Vector{Float64}; target_norm::Union{Nothing,Vector{Float64}} = nothing)
    started = time()
    model = nothing
    vars = nothing
    solver_label = ""
    threads = 0
    ok = false
    raw_point = Float64[]
    investments = Vector{Dict{String,Any}}()
    system_cost = NaN
    term_status = "NotSolved"
    prim_status = "NotSolved"
    error_message = ""
    try
        model, vars, _, solver_label, threads = _mga_build_model(md, cfg)
        cost_expr = objective_function(model)
        @constraint(model, cost_expr <= cost_cap)
        groups = mga_design_groups(md)
        exprs = _mga_design_exprs(vars, md, groups)
        if target_norm === nothing
            weights = _mga_weight_vector(direction, length(exprs))
            @objective(model, Min, _mga_direction_objective(exprs, weights))
        else
            target_raw = _mga_denormalize(target_norm, lower, upper, baseline_point)
            @variable(model, mga_distance >= 0)
            for i in eachindex(exprs)
                span = max(upper[i] - lower[i], 1.0)
                @constraint(model, exprs[i] - target_raw[i] <= span * mga_distance)
                @constraint(model, target_raw[i] - exprs[i] <= span * mga_distance)
            end
            @objective(model, Min, mga_distance)
        end
        optimize!(model)
        term_status = string(termination_status(model))
        prim_status = string(primal_status(model))
        ok = _mga_ok_status(model)
        if ok
            raw_point = _mga_design_values(vars, md, groups)
            system_cost = Float64(value(cost_expr))
            investments = _mga_capture_investments(vars, md)
        end
    catch err
        error_message = sprint(showerror, err)
        ok = false
    end
    solve_seconds = round(time() - started; digits = 3)
    slack_used = ok && abs(baseline_cost) > 1e-9 ? 100.0 * (system_cost / baseline_cost - 1) : NaN
    norm_point = ok ? _mga_normalize(raw_point, lower, upper) : Float64[]
    diversity = ok && !isempty(norm_point) ? maximum(abs.(norm_point .- _mga_normalize(baseline_point, lower, upper))) : NaN
    row = Dict{String,Any}(
        "direction" => direction["id"],
        "label" => direction["label"],
        "phase" => direction["phase"],
        "dominantGroup" => direction["dominantGroup"],
        "weights" => direction["weights"],
        "rawPoint" => raw_point,
        "point" => [round(v; digits = 6) for v in norm_point if isfinite(v)],
        "systemCost" => isfinite(system_cost) ? round(system_cost; digits = 6) : nothing,
        "costIndex" => ok && abs(baseline_cost) > 1e-9 ? round(100.0 * system_cost / baseline_cost; digits = 6) : nothing,
        "slackUsed" => isfinite(slack_used) ? round(slack_used; digits = 6) : nothing,
        "diversityScore" => isfinite(diversity) ? round(diversity; digits = 6) : nothing,
        "coverageGain" => direction["coverageGain"],
        "maxError" => direction["maxError"],
        "solveSeconds" => solve_seconds,
        "terminationStatus" => term_status,
        "primalStatus" => prim_status,
        "solver" => solver_label,
        "threadsPerSolve" => threads,
        "worker" => get(direction, "worker", 1),
        "status" => ok ? "solved" : "failed",
        "investments" => investments,
    )
    isempty(error_message) || (row["errorMessage"] = error_message)
    return row
end

function _mga_progress(progress, payload::Dict{String,Any})
    progress === nothing && return nothing
    progress(payload)
    return nothing
end

function mga_hybrid_oracle_run(md_source::ModelData, cfg::MGAExactConfig; progress = nothing)
    _mga_progress(progress, Dict("phase" => "prepare", "message" => "Preparing model data (representative days, extreme periods)", "completed" => 0, "total" => cfg.directions))
    md = _mga_prepare_md(md_source, cfg)
    _mga_progress(progress, Dict("phase" => "baseline", "message" => "Solving least-cost baseline", "completed" => 0, "total" => cfg.directions))
    baseline = _mga_solve_baseline(md, cfg)
    groups = [Dict{String,Any}(group) for group in baseline["groups"]]
    directions = _mga_planned_directions(groups, cfg)
    baseline_cost = Float64(baseline["cost"])
    cost_cap = baseline_cost * (1 + cfg.cost_slack / 100)
    baseline_point = Vector{Float64}(baseline["point"])
    lower = copy(baseline_point)
    upper = copy(baseline_point)
    results = Dict{String,Any}[]
    trace = Dict{String,Any}[]
    _mga_progress(progress, Dict(
        "phase" => "baseline-done",
        "message" => "Baseline solved (cost = $(round(baseline_cost; digits = 3)) in $(baseline["solveSeconds"]) s)",
        "completed" => 0,
        "total" => cfg.directions,
        "baselineCost" => baseline_cost,
        "baselineSolveSeconds" => Float64(baseline["solveSeconds"]),
        "costCap" => cost_cap,
    ))

    seed_dirs = [direction for direction in directions if direction["phase"] != "oracle-refine"]
    for direction in seed_dirs
        direction_id = Int(direction["id"])
        _mga_progress(progress, Dict(
            "phase" => String(direction["phase"]),
            "message" => "Solving $(direction["label"])",
            "completed" => length(results),
            "total" => cfg.directions,
            "directionId" => direction_id,
            "directionLabel" => String(direction["label"]),
            "directionStatus" => "running",
            "directionStartedAt" => time(),
        ))
        row = _mga_solve_alternative(md, cfg, direction, baseline_cost, cost_cap, lower, upper, baseline_point)
        push!(results, row)
        raw = Vector{Float64}(row["rawPoint"])
        _mga_update_bounds!(lower, upper, raw)
        _mga_refresh_rows!(results, groups, lower, upper, baseline_point)
        _mga_progress(progress, Dict(
            "phase" => String(direction["phase"]),
            "message" => "Solved $(direction["label"])",
            "completed" => length(results),
            "total" => cfg.directions,
            "directionId" => direction_id,
            "directionLabel" => String(direction["label"]),
            "directionStatus" => String(row["status"]),
            "directionDurationSeconds" => Float64(get(row, "solveSeconds", 0.0)),
            "directionErrorMessage" => String(get(row, "errorMessage", "")),
            "results" => results,
        ))
    end

    inner_points = _mga_inner_points(results, lower, upper, baseline_point)
    _, initial_error = _mga_error_scan(inner_points, length(groups); samples = max(192, 24 * length(groups)))
    current_error = initial_error

    oracle_dirs = [direction for direction in directions if direction["phase"] == "oracle-refine"]
    oracle_id = 1
    for iter in 1:cfg.oracle_iterations
        isempty(oracle_dirs) && break
        length(results) >= cfg.directions && break
        _mga_progress(progress, Dict("phase" => "oracle", "message" => "ORACLE iteration $iter starting (computing reduced-space gaps)", "completed" => length(results), "total" => cfg.directions, "oracleIteration" => iter, "oracleTrace" => trace))
        trial, before_error = _mga_error_scan(inner_points, length(groups); samples = max(192, 24 * length(groups) + 16 * cfg.oracle_batch))
        current_error = before_error
        accepted = 0
        for local_idx in 1:cfg.oracle_batch
            oracle_id > length(oracle_dirs) && break
            length(results) >= cfg.directions && break
            candidate = local_idx == 1 ? trial : _mga_candidate_point(97 * iter + 11 * local_idx, length(groups))
            direction = oracle_dirs[oracle_id]
            direction_id = Int(direction["id"])
            _mga_progress(progress, Dict(
                "phase" => "oracle",
                "message" => "Solving exact ORACLE closest-point LP $iter.$local_idx",
                "completed" => length(results),
                "total" => cfg.directions,
                "directionId" => direction_id,
                "directionLabel" => String(direction["label"]),
                "directionStatus" => "running",
                "directionStartedAt" => time(),
                "oracleIteration" => iter,
                "oracleTrace" => trace,
            ))
            row = _mga_solve_alternative(md, cfg, direction, baseline_cost, cost_cap, lower, upper, baseline_point; target_norm = candidate)
            push!(results, row)
            raw = Vector{Float64}(row["rawPoint"])
            if all(isfinite, raw) && !isempty(raw)
                _mga_update_bounds!(lower, upper, raw)
                _mga_refresh_rows!(results, groups, lower, upper, baseline_point)
                inner_points = _mga_inner_points(results, lower, upper, baseline_point)
                point = _mga_normalize(raw, lower, upper)
                gain = max(0.0, current_error - _mga_min_distance(point, inner_points))
                row["coverageGain"] = round(gain; digits = 6)
                _, current_error = _mga_error_scan(inner_points, length(groups); samples = max(192, 24 * length(groups)))
                row["maxError"] = round(current_error; digits = 6)
                accepted += 1
            end
            _mga_progress(progress, Dict(
                "phase" => "oracle",
                "message" => "Solved ORACLE closest-point LP $iter.$local_idx",
                "completed" => length(results),
                "total" => cfg.directions,
                "directionId" => direction_id,
                "directionLabel" => String(direction["label"]),
                "directionStatus" => String(row["status"]),
                "directionDurationSeconds" => Float64(get(row, "solveSeconds", 0.0)),
                "directionErrorMessage" => String(get(row, "errorMessage", "")),
                "oracleIteration" => iter,
                "oracleTrace" => trace,
                "results" => results,
            ))
            oracle_id += 1
            current_error <= cfg.tolerance && break
        end
        push!(trace, Dict("iteration" => iter, "trialPoint" => [round(v; digits = 6) for v in trial], "maxErrorBefore" => round(before_error; digits = 6), "maxErrorAfter" => round(current_error; digits = 6), "accepted" => accepted))
        _mga_progress(progress, Dict("phase" => "oracle", "message" => "ORACLE iteration $iter complete", "completed" => length(results), "total" => cfg.directions, "oracleIteration" => iter, "results" => results, "oracleTrace" => trace))
        current_error <= cfg.tolerance && break
    end

    _mga_progress(progress, Dict("phase" => "finalize", "message" => "Computing certificate and assembling results", "completed" => length(results), "total" => cfg.directions))

    _, final_error = _mga_error_scan(inner_points, length(groups); samples = max(384, 32 * length(groups)))
    _mga_refresh_rows!(results, groups, lower, upper, baseline_point)
    sort!(results; by = row -> Int(row["direction"]))
    baseline_investments = Vector{Dict{String,Any}}(get(baseline, "investments", Dict{String,Any}[]))
    investment_spread = _mga_investment_spread(baseline_investments, results)
    certificate = Dict{String,Any}(
        "mode" => "exact-solver-hybrid-oracle",
        "solver" => baseline["solver"],
        "baselineCost" => round(baseline_cost; digits = 6),
        "costCap" => round(cost_cap; digits = 6),
        "costSlackPercent" => round(cfg.cost_slack; digits = 6),
        "initialMaxError" => round(initial_error; digits = 6),
        "estimatedMaxError" => round(final_error; digits = 6),
        "targetTolerance" => round(cfg.tolerance; digits = 6),
        "iterations" => length(trace),
        "exploratoryDimensions" => length(groups),
        "innerPoints" => length(inner_points),
        "converged" => final_error <= cfg.tolerance,
        "period" => cfg.period,
        "modeLabel" => String(cfg.mode),
        "threadsPerSolve" => baseline["threadsPerSolve"],
        "baselineSolveSeconds" => baseline["solveSeconds"],
        "rows" => baseline["rows"],
        "columns" => baseline["columns"],
        "solvedAlternatives" => count(row -> row["status"] == "solved", results),
        "failedAlternatives" => count(row -> row["status"] != "solved", results),
    )
    directions_out = [Dict{String,Any}(
        "id" => row["direction"],
        "label" => row["label"],
        "phase" => row["phase"],
        "dominantGroup" => row["dominantGroup"],
        "weights" => row["weights"],
        "point" => row["point"],
        "maxError" => row["maxError"],
        "coverageGain" => row["coverageGain"],
    ) for row in results]
    return Dict{String,Any}(
        "method" => "Exact Hybrid ORACLE MGA",
        "description" => "Each alternative is an LP over the original IESA-Opt constraints plus a near-optimal system-cost cap. ORACLE refinements use exact closest-point LPs in the reduced design space.",
        "groups" => groups,
        "directions" => directions_out,
        "oracleTrace" => trace,
        "certificate" => certificate,
        "results" => results,
        "baselineInvestments" => baseline_investments,
        "investmentSpread" => investment_spread,
    )
end
