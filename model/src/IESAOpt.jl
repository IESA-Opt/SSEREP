"""
    IESAOpt

IESA-Opt.jl — the Julia/JuMP implementation of the IESA-Opt integrated
energy-system optimization model.

The package exposes data loading, clustering, JuMP model construction, solver
configuration, and result-writing helpers for IESA-Opt.jl.
"""
module IESAOpt

using JuMP
using DataFrames
using XLSX
using Clustering
using Statistics
using LinearAlgebra
using SparseArrays
using Printf
using Random
using Dates
using Logging
using TOML
using JSON3
using SHA
using Parquet2
import MathOptInterface as MOI
using PrecompileTools

# Solver backends (loaded lazily — Gurobi requires a valid license + GUROBI_HOME)
using HiGHS
try
    @eval using Gurobi
catch err
    @warn "Gurobi.jl not available — Gurobi-based runs will fail. Use highs_optimizer() instead." err=err
end

# Parquet2 is still used for legacy cluster-map inputs; DuckDB is used by the UI runtime cache and result database.

# ---------------------------------------------------------------------------
# Core types
include("types.jl")

# ---------------------------------------------------------------------------
# Phase 1: data layer
include("data_reading.jl")
include("data_cache.jl")
include("sets.jl")
include("parameters.jl")
include("data_writing.jl")

# ---------------------------------------------------------------------------
# Solver wiring (Phase 7 production defaults)
include("solver_settings.jl")

# ---------------------------------------------------------------------------
# Phase 2+: model assembly (skeleton includes; bodies land per phase)
include("model/variables.jl")
include("model/stock.jl")
include("model/balance.jl")
include("model/objective.jl")
# Phase 3: full-hourly (FH) constraint families — single consolidated module
include("model/hourly.jl")
# Phase 5/6: time-slice (TS) constraint families
include("model/ts.jl")
# Phase 4 (new): infrastructure-volume + policy / regulatory constraints
include("model/infrastructure.jl")
include("model/policy.jl")
include("model/cyclic_closures.jl")
# include("model/capacity.jl")
# include("model/ramping.jl")
# include("model/emissions.jl")
# include("model/shedding.jl")
# include("model/chp.jl")
# include("model/storage.jl")
# include("model/backlog.jl")
# include("model/reservoir.jl")
# include("model/gasbuffer.jl")
# include("model/interconnect.jl")
# include("model/ts_extras.jl")

# ---------------------------------------------------------------------------
# Phase 5+: clustering
include("clustering.jl")

# ---------------------------------------------------------------------------
# Orchestration (Phase 2+)
include("solve.jl")
# Diagnostics for infeasible / near-infeasible runs (Show violations + IIS)
include("violations.jl")
# include("postprocess.jl")
include("writers.jl")

# ---------------------------------------------------------------------------
# Workflow: Scenario Space exploration
# Phase 1: spec parsing + sampling, no model touch
include("workflows/scenario_space/spec.jl")
include("workflows/scenario_space/campaign_config.jl")
include("workflows/scenario_space/legacy_1108_ssp.jl")
include("workflows/scenario_space/sampling.jl")
# Phase 2: in-place LP mutation (constraint-ref manifest + per-variant apply)
include("workflows/scenario_space/manifest.jl")
include("workflows/scenario_space/variant.jl")
# Phase 3: campaign runner (serial + Distributed.jl worker pool)
include("workflows/scenario_space/runner.jl")
include("workflows/scenario_space/production_campaign.jl")
# Phase 4: high-level orchestrator + result persistence
include("workflows/scenario_space/orchestrator.jl")
include("workflows/scenario_space/persistence.jl")
# Phase 5: analysis helpers (objective_table, sensitivity_scan, pareto_front)
include("workflows/scenario_space/analysis.jl")
include("workflows/scenario_space/gsa.jl")
include("workflows/scenario_space/production_analysis.jl")

# Workflow: MGA reduced-space hybrid ORACLE planning helpers. These utilities
# do not execute during normal single-run solves.
include("workflows/mga/hybrid_oracle.jl")

# ---------------------------------------------------------------------------
# Project-specific extensions (opt-in, NOT part of the core model)
# Each extension is gated on a Symbol in `md.params.extensions::Set{Symbol}`,
# populated by a UI toggle. With an empty set the core model is unchanged.
# See src/extensions/README.md for how to add or remove extensions cleanly.
# ---------------------------------------------------------------------------
include("extensions/multi_region.jl")
include("extensions/extensions.jl")

include("ui_server.jl")
# include("sweeps.jl")

# ---------------------------------------------------------------------------
# Exports
export ModelSets, ModelParams, ModelData, RunResult
export read_data, derive_sets!, compute_derived_params!
export read_data_cached, clear_data_cache
export compute_temporal_helpers!, compute_period_indicators!
export compute_electricity_trade_limits!
export compute_financial_params!, compute_investment_matrices!
export compute_activity_balances!, compute_chp_eps!, compute_activity_indicators!
export compute_decom_planned_sel!, compute_flex_loss_split!
export compute_emission_target_aggregates!
export compute_tech_activity!, init_policy_targets!
export write_sets_dump, write_params_dump, write_run_summary, write_run_statistics
export default_gurobi_attributes, gurobi_tuned_attributes_for_repdays, default_highs_attributes
export gurobi_optimizer, highs_optimizer, apply_solver_attributes!
# Phase 2: annual LP
export AnnualVars, add_annual_variables!
export add_stock_constraints!, add_balance_constraints!, add_objective!
export build_annual_lp!, solve_annual!, extract_annual_results
export apply_lp_generation_speedups!
# Phase 3: full-hourly LP
export add_hourly_variables!, add_hourly_constraints!
export build_fh_lp!
# Phase 5/6: time-slice LP
export build_temporal_clusters!
export add_ts_variables!, add_ts_constraints!
export build_ts_lp!
# Phase 4 (new): infrastructure / policy / cyclic closures
export add_infrastructure_constraints!
export add_policy_constraints!
export add_cyclic_closures!
# Phase 7: writers
export write_parquet_results, write_duckdb_results
export serve_ui!
# Phase 7+: export sweep_ts_postfix

# Scenario-space exploration (Phase 1)
export ParameterRow, CampaignSpec, SampleMatrix
export parse_sampling_method, parse_param_type
export unique_parameters, parameter_bounds, parameter_steps
export validate_spec, spec_from_dict, spec_to_dict
export CampaignBundle, load_campaign_bundle, campaign_spec
export Legacy1108Context, load_1108_context, prepare_1108_variant
export sample_campaign, implied_sample_size
# Scenario-space exploration (Phase 2: in-place LP mutation)
export Mutation, LeafChange
export register_mutation!, is_mutation_registered, registered_mutation_fields
export build_mutations, apply_mutation!, apply_mutations!
export apply_leaf_change!, apply_leaf_changes!, apply_variant!
# Scenario-space exploration (Phase 3: campaign runner)
export VariantResult, run_campaign
export ProductionCampaignSummary, run_1108_production_campaign
# Scenario-space exploration (Phase 3.5: per-variant clustering)
export register_clustering_affecting!, unregister_clustering_affecting!
export is_clustering_affecting, clustering_affecting_fields, variant_affects_clustering
# Scenario-space exploration (Phase 4: orchestrator + persistence)
export LeafTarget, ScenarioSpec, ScenarioResult
export sample_scenario_space, samples_to_changes, run_scenario_space
export save_scenario_results, load_scenario_results
# Scenario-space exploration (Phase 5: analysis helpers)
export objective_table, sensitivity_scan, pareto_front
export delta_sensitivity, morris_sensitivity
export analyze_production_campaign

precompile(_production_worker_loop, (
    Distributed.RemoteChannel, Distributed.RemoteChannel, String, Symbol,
    Int, Symbol, Int, Vector{Int}, Dict{String,Any}, Float64, Int))

const _PRODUCTION_SOLVE_KWARGS = NamedTuple{
    (:solver, :threads, :mode, :representative_days, :periods,
     :solver_attrs, :variant_timeout_seconds, :max_attempts),
    Tuple{Symbol,Int,Symbol,Int,Vector{Int},Dict{String,Any},Float64,Int}}
precompile(Core.kwcall, (
    _PRODUCTION_SOLVE_KWARGS, typeof(_solve_1108_production_variant),
    Legacy1108Context, Vector{Float64}, Int))

# ---------------------------------------------------------------------------
# Module init — populate the scenario-space mutation registry with the
# default leaf-parameter -> constraint-name builders. Idempotent.
function __init__()
    try
        _register_default_mutations!()
    catch err
        @warn "IESAOpt: failed to register default scenario-space mutations" err
    end
    return nothing
end

# ---------------------------------------------------------------------------
# PrecompileTools workload
#
# Pre-bakes the JIT cost of the heaviest runtime code paths (read_data_cached,
# derive_sets!, compute_derived_params!, build_temporal_clusters!, and the
# full TS LP build) into the package precompile cache. Cuts the in-server
# warmup from ~50 s to ~3 s on every UI launch, at the price of adding
# ~60-90 s to the one-time `Pkg.precompile` after code edits, Julia
# upgrades, or dep updates.
#
# Skipped automatically when:
#   - The default workbook is missing (clean checkout, CI without data)
#   - Env var IESA_OPT_SKIP_PRECOMPILE=1 (developer fast-iteration mode —
#     pair with IESA_OPT_SKIP_WARMUP=1 to also skip the in-server warmup)
#
# Failures inside the workload are warned but never abort the package build.
# ---------------------------------------------------------------------------
@setup_workload begin
    _precompile_workbook = normpath(joinpath(@__DIR__, "..", "Input", "1108 SSP.xlsx"))
    _precompile_skip = get(ENV, "IESA_OPT_SKIP_PRECOMPILE", "0") == "1"
    @compile_workload begin
        if !_precompile_skip && isfile(_precompile_workbook)
            try
                _md = read_data_cached(_precompile_workbook)
                if _md !== nothing
                    _md_copy = deepcopy(_md)
                    _periods = collect(_md_copy.sets.periods)
                    if !isempty(_periods)
                        _target = 2050 in _periods ? 2050 : last(_periods)
                        _md_copy.sets.periods_solve = [_target]
                        _md_copy.params.hoursPer_day = 24
                        _md_copy.params.n_repDays = 1
                        _md_copy.params.hoursPer_day_cluster = 24
                        _md_copy.params.clustering_approach = :kmeans_avg
                        _md_copy.params.ts_extremePeriods = false
                        _md_copy.params.ts_extremeDays_count = 0
                        _md_copy.params.ts_boundaryRamping = true
                        _md_copy.params.ts_capacityProfile_autoMode = true
                        _md_copy.params.ts_capacityProfile_autoFloor = 0.23
                        _md_copy.params.ts_capacityProfile_autoCap = 1.00
                        _md_copy.params.ts_capacityProfile_autoFloor_effective = 0.23
                        _md_copy.params.ts_capacityProfile_envelopeMode = 0
                        _md_copy.params.dayMix_softness = 0.0
                        _md_copy.params.dayMix_weightType = :auto
                        derive_sets!(_md_copy)
                        compute_derived_params!(_md_copy)
                        build_temporal_clusters!(_md_copy)
                        _model = JuMP.Model()
                        apply_lp_generation_speedups!(_model)
                        build_ts_lp!(_model, _md_copy)
                        _model = nothing
                    end
                end
            catch err
                @warn "IESA-Opt.jl precompile workload failed (package still loaded successfully)" err
            end
        end
    end
end

end # module
