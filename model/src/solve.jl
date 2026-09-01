# =============================================================================
# solve.jl — orchestration: build + solve + extract results
#
# Phase 2 entry points:
#   build_annual_lp!(m::JuMP.Model, md::ModelData) -> AnnualVars
#   solve_annual!(md, optimizer; out_dir=nothing) -> RunResult
#   extract_annual_results(m, vars, md) -> Dict{Symbol,Any}
#
# Phase 3+ will add: build_fh_lp!, build_ts_lp!, etc.
# =============================================================================

using Dates

"""
    apply_lp_generation_speedups!(m::JuMP.Model; keep_names::Bool=false) -> JuMP.Model

Apply JuMP's documented performance-tip settings before adding variables and
constraints. The dominant win is disabling string-name creation: every
`@variable` / `@constraint` in the model passes a `base_name="…"` argument,
which (a) forces string interpolation per call and (b) stores the name in
JuMP's per-model name dictionary. For an IESA-Opt LP with hundreds of
thousands of constraints that overhead is non-trivial — JuMP's own
performance guide recommends turning names off for production runs.
See https://jump.dev/JuMP.jl/stable/tutorials/getting_started/performance_tips/#Disable-string-names

Names are only useful for:
  - reading solver log lines such as "constraint balH_TS[…] has invalid bound"
  - mapping `JuMP.compute_conflict!` IIS members back to source constraints
  - the elastic-relaxation slack report (`report_nonzero_slacks`).

If you need any of those, pass `keep_names = true` (the UI passes this when
"Show violations" is enabled) or set the environment variable
`IESA_OPT_KEEP_NAMES=1`.

Always call this **before** building the model — `set_string_names_on_creation`
only affects subsequently-created variables and constraints.
"""
function apply_lp_generation_speedups!(m::JuMP.Model; keep_names::Bool = false)
    keep = keep_names || get(ENV, "IESA_OPT_KEEP_NAMES", "0") == "1"
    JuMP.set_string_names_on_creation(m, keep)
    return m
end

"""
    build_annual_lp!(m::JuMP.Model, md::ModelData) -> AnnualVars

Build the annual LP (Phase 2 subset) on `m`. Returns the variable container.
"""
function build_annual_lp!(m::JuMP.Model, md::ModelData)
    @info "build_annual_lp! - declaring variables"
    flush(stderr)
    vars = add_annual_variables!(m, md)
    @info "build_annual_lp! - adding stock + investment constraints"
    flush(stderr)
    add_stock_constraints!(m, vars, md)
    @info "build_annual_lp! - adding balance + capacity + emission constraints"
    flush(stderr)
    add_balance_constraints!(m, vars, md)
    @info "build_annual_lp! - setting objective"
    flush(stderr)
    add_objective!(m, vars, md)
    # ---- Extensions hook (opt-in; no-op when params.extensions is empty) ----
    apply_extensions!(m, vars, md; mode = :annual)
    @info "build_annual_lp! - complete"
    flush(stderr)
    return vars
end

"""
    build_fh_lp!(m::JuMP.Model, md::ModelData) -> AnnualVars

Build the full-hourly (FH) LP — Phase 3.  Adds annual variables + stock +
annual balance, then layers on hourly variables + all hourly constraint
families (hourly/daily dispatch, storage, flex, reservoir, CHP, shedding,
backlog, gas buffer) and sets the FH-mode objective.
"""
function build_fh_lp!(m::JuMP.Model, md::ModelData)
    @info "build_fh_lp! - declaring annual variables"
    flush(stderr)
    vars = add_annual_variables!(m, md)
    @info "build_fh_lp! - declaring hourly variables"
    flush(stderr)
    vars = add_hourly_variables!(m, vars, md)
    @info "build_fh_lp! - adding stock + investment constraints"
    flush(stderr)
    add_stock_constraints!(m, vars, md)
    @info "build_fh_lp! - adding annual balance + capacity + emission constraints"
    flush(stderr)
    add_balance_constraints!(m, vars, md)
    @info "build_fh_lp! - adding hourly constraints"
    flush(stderr)
    add_hourly_constraints!(m, vars, md)
    @info "build_fh_lp! - adding infrastructure-volume constraints (FH)"
    flush(stderr)
    add_infrastructure_constraints!(m, vars, md; mode = :fh)
    @info "build_fh_lp! - adding policy constraints (FH)"
    flush(stderr)
    add_policy_constraints!(m, vars, md; mode = :fh)
    if get(ENV, "IESA_ENABLE_EXTRA_CYCLIC_CLOSURES", "0") == "1"
        add_cyclic_closures!(m, vars, md)
    end
    @info "build_fh_lp! - setting objective (with hourly terms)"
    flush(stderr)
    add_objective!(m, vars, md)
    # ---- Extensions hook (opt-in; no-op when params.extensions is empty) ----
    apply_extensions!(m, vars, md; mode = :fh)
    @info "build_fh_lp! - complete"
    flush(stderr)
    return vars
end

"""
    build_ts_lp!(m::JuMP.Model, md::ModelData) -> AnnualVars

Build the time-slice (TS) LP using representative-day clustering — Phase 5/6.

Prerequisites:
  - `build_temporal_clusters!(md)` must have been called to populate
    `md.sets.hours_cluster`, `md.sets.repDays`, and all `*_cluster` parameter
    Dicts.

Layers (in order):
  1. annual variables + stock + annual balance + capacity / emission
  2. TS variables (`*_TS` indexed by `hc`) + calendar-day anchor variables
  3. TS constraint families (mirror of FH, plus cal-day anchors for storage)
  4. Objective with TS-mode hourly terms.
"""
function build_ts_lp!(m::JuMP.Model, md::ModelData)
    @info "build_ts_lp! - declaring annual variables"
    flush(stderr)
    vars = add_annual_variables!(m, md)
    @info "build_ts_lp! - declaring TS (rep-day) variables"
    flush(stderr)
    vars = add_ts_variables!(m, vars, md)
    @info "build_ts_lp! - adding stock + investment constraints"
    flush(stderr)
    add_stock_constraints!(m, vars, md)
    @info "build_ts_lp! - adding annual balance + capacity + emission constraints"
    flush(stderr)
    add_balance_constraints!(m, vars, md)
    @info "build_ts_lp! - adding TS constraints"
    flush(stderr)
    add_ts_constraints!(m, vars, md)
    @info "build_ts_lp! - adding infrastructure-volume constraints (TS)"
    flush(stderr)
    add_infrastructure_constraints!(m, vars, md; mode = :ts)
    @info "build_ts_lp! - adding policy constraints (TS)"
    flush(stderr)
    add_policy_constraints!(m, vars, md; mode = :ts)
    @info "build_ts_lp! - setting objective (with TS hourly terms)"
    flush(stderr)
    add_objective!(m, vars, md)
    # ---- Extensions hook (opt-in; no-op when params.extensions is empty) ----
    apply_extensions!(m, vars, md; mode = :ts)
    @info "build_ts_lp! - complete"
    flush(stderr)
    return vars
end

"""
    solve_annual!(md::ModelData, optimizer; out_dir=nothing, mode=:fh) -> (RunResult, AnnualVars, JuMP.Model)

End-to-end: build + solve + (optionally) write the run summary.
"""
function solve_annual!(md::ModelData, optimizer;
                       out_dir::Union{String,Nothing} = nothing,
                       mode::Symbol = :fh)
    t_build_start = time()
    m = Model(optimizer)
    apply_lp_generation_speedups!(m)
    vars = build_annual_lp!(m, md)
    build_seconds = time() - t_build_start
    @info "build_annual_lp! finished" build_seconds

    t_solve_start = time()
    optimize!(m)
    solve_seconds = time() - t_solve_start

    n_rows = try; num_constraints(m; count_variable_in_set_constraints = false); catch; 0; end
    n_cols = num_variables(m)
    term   = string(termination_status(m))
    primal = string(primal_status(m))
    objval = (termination_status(m) == MOI.OPTIMAL ||
              termination_status(m) == MOI.LOCALLY_SOLVED ||
              termination_status(m) == MOI.ALMOST_OPTIMAL) ? objective_value(m) : NaN

    settings_used = Dict{String,Any}()  # caller can populate from default_*_attributes()
    progstat = _categorize_status(termination_status(m), primal_status(m))

    rr = RunResult(
        out_dir === nothing ? "" : out_dir,
        now(),
        mode,
        term, primal, progstat,
        objval, solve_seconds, build_seconds + solve_seconds,
        n_rows, n_cols, 0,        # nnz unknown without MOI introspection
        0, 0,                       # iteration counts unknown via JuMP generic API
        settings_used,
        "",                          # weather_year not tracked in Phase 2
        md.params.n_repDays, md.params.hoursPer_day,
        md.params.clustering_approach,
    )

    if out_dir !== nothing
        mkpath(out_dir)
        db_path = joinpath(out_dir, IESA_RESULTS_DUCKDB_FILE)
        _remove_duckdb_database!(db_path)
        write_run_statistics_parquet(rr, _duckdb_table_uri(db_path, "run_statistics"))
    end

    return rr, vars, m
end

# Map JuMP termination + primal status to human-readable IESA-Opt 1.0-style string
function _categorize_status(term::MOI.TerminationStatusCode,
                            primal::MOI.ResultStatusCode)::String
    term == MOI.OPTIMAL          && return "Optimal"
    term == MOI.LOCALLY_SOLVED   && return "Optimal"
    term == MOI.ALMOST_OPTIMAL   && return "Sub-optimal"
    term == MOI.INFEASIBLE       && return "Infeasible"
    term == MOI.DUAL_INFEASIBLE  && return "Unbounded"
    term == MOI.TIME_LIMIT       && return "TimeLimit"
    term == MOI.INTERRUPTED      && return "Interrupted"
    if primal == MOI.FEASIBLE_POINT
        return "Sub-optimal"
    end
    return string(term)
end

"""
    extract_annual_results(m::JuMP.Model, vars::AnnualVars, md::ModelData)
        -> Dict{Symbol,Any}

Return a dict of solution arrays keyed by variable name. Each value is itself
a `Dict` indexed by the variable's original axes.
"""
function extract_annual_results(m::JuMP.Model, vars::AnnualVars, md::ModelData)
    out = Dict{Symbol,Any}()
    if has_values(m)
        out[:tech_use]           = Dict((tb, ps) => value(vars.tech_use[tb, ps])
                                        for tb in md.sets.tech_balancers
                                        for ps in md.sets.periods_solve)
        out[:techStock]          = Dict((t, ps) => value(vars.techStock[t, ps])
                                        for t in md.sets.technologies
                                        for ps in md.sets.periods_solve)
        out[:cap_investments]    = Dict((t, ps) => value(vars.cap_investments[t, ps])
                                        for t in md.sets.technologies
                                        for ps in md.sets.periods_solve)
        out[:eco_decommisioning] = Dict((t, ps) => value(vars.eco_decommisioning[t, ps])
                                        for t in md.sets.technologies
                                        for ps in md.sets.periods_solve)
        out[:decomStock]         = Dict((t, ps) => value(vars.decomStock[t, ps])
                                        for t in md.sets.technologies
                                        for ps in md.sets.periods_solve)
        out[:retrofitting]       = Dict(k => value(v) for (k, v) in vars.retrofitting)
    end
    return out
end
