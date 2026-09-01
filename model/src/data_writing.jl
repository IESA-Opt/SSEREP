# =============================================================================
# data_writing.jl — parquet dumpers for sets, parameters, and run results
#
# Produces flat parquet files for diff and inspection workflows.
#
# Output schemas:
#   write_sets_dump(md, dir)         → <dir>/sets.parquet          [set, member]
#   write_params_dump(md, dir)       → <dir>/params_scalar.parquet [param, value]
#                                       <dir>/params_indexed.parquet [param, key, value]
#   write_run_summary(result, dir)   → <dir>/summary.parquet       [metric, value]
#   write_run_statistics(result, dir)→ <dir>/statistics.parquet    [metric, value]
#
# All writers create `dir` (recursively) if it does not exist.
# =============================================================================

"""
    write_sets_dump(md::ModelData, dir::AbstractString) -> String

Write every populated ModelSets field as `(set_name, member)` rows in
`<dir>/sets.parquet`. Empty sets are skipped. Returns the absolute path to the
written parquet file.
"""
function write_sets_dump(md::ModelData, dir::AbstractString)
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "sets.parquet")
    s = md.sets

    rows = Vector{NamedTuple{(:set_name, :member),Tuple{String,String}}}()
    for name in fieldnames(ModelSets)
        v = getfield(s, name)
        v isa AbstractVector || continue
        isempty(v) && continue
        name_str = String(name)
        for member in v
            push!(rows, (set_name=name_str, member=string(member)))
        end
    end

    _write_parquet_dump(path, isempty(rows) ? DataFrame(set_name=String[], member=String[]) : DataFrame(rows))
    return path
end

"""
    write_params_dump(md::ModelData, dir::AbstractString) -> Tuple{String,String}

Write scalar and indexed parameters to two parquet files:
- `<dir>/params_scalar.parquet`  schema `(param_name, value)`
- `<dir>/params_indexed.parquet` schema `(param_name, key, value)`

Dict-valued parameters become rows with `key` formatted as
`"(a, b)"` or `"a"` (string-joined tuple elements). Returns `(scalar_path,
indexed_path)`.
"""
function write_params_dump(md::ModelData, dir::AbstractString)
    isdir(dir) || mkpath(dir)
    scalar_path  = joinpath(dir, "params_scalar.parquet")
    indexed_path = joinpath(dir, "params_indexed.parquet")
    p = md.params

    scalar_rows  = Vector{NamedTuple{(:param_name, :value),Tuple{String,String}}}()
    indexed_rows = Vector{NamedTuple{(:param_name, :key, :value),Tuple{String,String,String}}}()

    for name in fieldnames(ModelParams)
        v = getfield(p, name)
        name_str = String(name)
        if v isa AbstractDict
            isempty(v) && continue
            for (k, val) in v
                push!(indexed_rows, (param_name=name_str, key=_format_key(k), value=_format_value(val)))
            end
        elseif v isa AbstractVector
            isempty(v) && continue
            for (i, val) in enumerate(v)
                push!(indexed_rows, (param_name=name_str, key=string(i), value=_format_value(val)))
            end
        elseif v isa Union{Number, AbstractString, Symbol, Bool}
            push!(scalar_rows, (param_name=name_str, value=_format_value(v)))
        end
        # Skip everything else (function refs, complex composite types)
    end

    _write_parquet_dump(scalar_path, isempty(scalar_rows) ? DataFrame(param_name=String[], value=String[]) : DataFrame(scalar_rows))
    _write_parquet_dump(indexed_path, isempty(indexed_rows) ? DataFrame(param_name=String[], key=String[], value=String[]) : DataFrame(indexed_rows))
    return scalar_path, indexed_path
end

"""
    write_run_summary(result::RunResult, dir::AbstractString) -> String

Write run-level summary metrics (objective, solve_time, status, etc.) to
`<dir>/summary.parquet`. Returns the absolute path.
"""
function write_run_summary(result::RunResult, dir::AbstractString)
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "summary.parquet")

    rows = NamedTuple{(:metric, :value),Tuple{String,String}}[
        (metric="termination_status", value=result.termination_status),
        (metric="primal_status",      value=result.primal_status),
        (metric="program_status",     value=result.program_status),
        (metric="objective_value",    value=_format_value(result.objective_value)),
        (metric="solve_seconds",      value=_format_value(result.solve_seconds)),
        (metric="total_seconds",      value=_format_value(result.total_seconds)),
        (metric="mode",               value=String(result.mode)),
        (metric="weather_year",       value=result.weather_year),
        (metric="n_repDays",          value=string(result.n_repDays)),
        (metric="hoursPer_day",       value=string(result.hoursPer_day)),
        (metric="clustering_approach",value=String(result.clustering_approach)),
        (metric="timestamp",          value=string(result.timestamp)),
    ]
    _write_parquet_dump(path, DataFrame(rows))
    return path
end

"""
    write_run_statistics(result::RunResult, dir::AbstractString) -> String

Write model-size statistics (n_rows, n_cols, n_nnz, iterations) to
`<dir>/statistics.parquet`. Returns the absolute path.
"""
function write_run_statistics(result::RunResult, dir::AbstractString)
    isdir(dir) || mkpath(dir)
    path = joinpath(dir, "statistics.parquet")

    rows = NamedTuple{(:metric, :value),Tuple{String,String}}[
        (metric="n_rows",             value=string(result.n_rows)),
        (metric="n_cols",             value=string(result.n_cols)),
        (metric="n_nnz",              value=string(result.n_nnz)),
        (metric="n_iterations",       value=string(result.n_iterations)),
        (metric="barrier_iterations", value=string(result.barrier_iterations)),
    ]
    _write_parquet_dump(path, DataFrame(rows))
    return path
end

function _write_parquet_dump(path::AbstractString, df::DataFrame)
    Parquet2.writefile(path, df)
    return path
end

# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
_format_value(x::Float64) = isnan(x) ? "NaN" : (isinf(x) ? string(x) : @sprintf("%.10g", x))
_format_value(x::Real)    = string(x)
_format_value(x::Bool)    = x ? "1" : "0"
_format_value(x::Symbol)  = String(x)
_format_value(x::AbstractString) = String(x)
_format_value(x)          = string(x)

_format_key(k::Tuple) = string("(", join(string.(k), ", "), ")")
_format_key(k)        = string(k)
