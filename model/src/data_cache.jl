# =============================================================================
# data_cache.jl — DuckDB compiled input cache for read_data
#
# IESA-Opt 1.0 reads its XLSX database in ~2s.  Pure XLSX.jl streaming reads of the
# same file take ~60s because cell-by-cell reads of small per-tech lookups
# dominate (~408 techs × dozens of column reads + state-machine cost).
#
# Strategy:
#   - First call to `read_data_cached(xlsx)` runs the full XLSX pipeline and
#     stores the compiled `ModelData` in a DuckDB file next to the workbook.
#   - Subsequent calls compare the XLSX file's `(mtime, size)` against the
#     DuckDB metadata; on match → deserialize the compiled payload.
#   - On XLSX edit (any byte change) the DuckDB input cache is rebuilt.
#
# Usage:
#     md = read_data_cached(xlsx_path)
#     md = read_data_cached(xlsx_path; force_refresh = true)
#     md = read_data_cached(xlsx_path; cache_dir = "/tmp/iesa_cache")
# =============================================================================

using Serialization
using Base64
using DuckDB
import DBInterface

const _IESA_CACHE_FORMAT_VERSION = 10
const _IESA_INPUT_DUCKDB_SCHEMA_VERSION = 2

"""
    read_data_cached(xlsx_path; cache_dir=auto, force_refresh=false, kwargs...)

Wrap `read_data` with a DuckDB compiled input cache keyed on the XLSX file's
mtime+size. Returns the same `ModelData` as `read_data`. The cache file lives at
`cache_dir/<basename_without_ext>.iesa_input.duckdb`.

Keyword args (besides cache-control) are forwarded to `read_data`.
"""
function read_data_cached(xlsx_path::AbstractString;
                          cache_dir::Union{String,Nothing} = nothing,
                          force_refresh::Bool = false,
                          kwargs...)
    isfile(xlsx_path) || error("XLSX not found: $xlsx_path")

    if cache_dir === nothing
        cache_dir = joinpath(dirname(abspath(xlsx_path)), ".iesa_cache")
    end
    mkpath(cache_dir)

    cache_path = _duckdb_input_cache_path(xlsx_path, cache_dir)

    xstat = stat(xlsx_path)
    cache_exists = false
    if !force_refresh
        try
            cache_exists = isfile(cache_path)
        catch err
            @warn "read_data_cached: cannot inspect DuckDB cache, rebuilding from XLSX" cache_path err = err
        end
    end
    if !force_refresh && cache_exists
        try
            t0 = time()
            cached_md = _read_duckdb_input_cache(cache_path, xlsx_path, xstat)
            if cached_md !== nothing
                @info "read_data_cached: DuckDB hit" cache_path elapsed_s = round(time() - t0, digits = 2)
                return cached_md
            end
            @info "read_data_cached: DuckDB cache stale, rebuilding" cache_path
        catch err
            @warn "read_data_cached: failed to read DuckDB cache, rebuilding" err = err
        end
    end

    t0 = time()
    md = read_data(xlsx_path; kwargs...)
    @info "read_data_cached: built from XLSX" elapsed_s = round(time() - t0, digits = 1)

    try
        _write_duckdb_input_cache(cache_path, xlsx_path, xstat, md)
        @info "read_data_cached: wrote DuckDB cache" cache_path size_mb = round(stat(cache_path).size / 1e6, digits = 2)
    catch err
        @warn "read_data_cached: failed to write DuckDB cache (continuing without caching)" err = err
    end

    return md
end

"""
    clear_data_cache(xlsx_path; cache_dir=auto)

Delete the DuckDB cache for `xlsx_path`. Returns true if a cache file existed.
"""
function clear_data_cache(xlsx_path::AbstractString;
                          cache_dir::Union{String,Nothing} = nothing)
    if cache_dir === nothing
        cache_dir = joinpath(dirname(abspath(xlsx_path)), ".iesa_cache")
    end
    removed = false
    for cache_path in (_duckdb_input_cache_path(xlsx_path, cache_dir), _duckdb_input_cache_path(xlsx_path, cache_dir) * ".wal")
        if isfile(cache_path)
            rm(cache_path; force = true)
            @info "clear_data_cache: removed" cache_path
            removed = true
        end
    end
    return removed
end

function _duckdb_input_cache_path(xlsx_path::AbstractString, cache_dir::AbstractString)
    base = first(splitext(basename(xlsx_path)))
    return joinpath(cache_dir, base * ".iesa_input.duckdb")
end

function _read_duckdb_input_cache(cache_path::AbstractString, xlsx_path::AbstractString, xstat::Base.Filesystem.StatStruct)
    con = _duckdb_connect(cache_path; readonly = true)
    try
        metadata = _duckdb_metadata(con)
        get(metadata, "cache_format", "") == string(_IESA_CACHE_FORMAT_VERSION) || return nothing
        get(metadata, "schema_version", "") == string(_IESA_INPUT_DUCKDB_SCHEMA_VERSION) || return nothing
        get(metadata, "xlsx_mtime", "") == string(xstat.mtime) || return nothing
        get(metadata, "xlsx_size", "") == string(xstat.size) || return nothing
        payload = _duckdb_query_df(con, "SELECT payload_base64 FROM model_data WHERE id = 1 LIMIT 1")
        isempty(payload) && return nothing
        bytes = base64decode(String(payload.payload_base64[1]))
        return deserialize(IOBuffer(bytes))
    finally
        DBInterface.close!(con)
        GC.gc()
    end
end

function _write_duckdb_input_cache(cache_path::AbstractString, xlsx_path::AbstractString, xstat::Base.Filesystem.StatStruct, md::ModelData)
    mkpath(dirname(cache_path))
    rm(cache_path; force = true)
    buffer = IOBuffer()
    serialize(buffer, md)
    payload = base64encode(take!(buffer))
    con = _duckdb_connect(cache_path)
    try
        metadata = DataFrames.DataFrame(
            key = ["cache_format", "schema_version", "xlsx_path", "xlsx_mtime", "xlsx_size", "julia_ver", "created_at"],
            value = [string(_IESA_CACHE_FORMAT_VERSION), string(_IESA_INPUT_DUCKDB_SCHEMA_VERSION), abspath(xlsx_path), string(xstat.mtime), string(xstat.size), string(VERSION), string(Dates.now())],
        )
        _duckdb_replace_table!(con, metadata, "metadata")
        _duckdb_replace_table!(con, DataFrames.DataFrame(id = [1], payload_base64 = [payload]), "model_data")
    finally
        DBInterface.close!(con)
        GC.gc()
    end
    return cache_path
end

function _duckdb_metadata(con)
    try
        df = _duckdb_query_df(con, "SELECT key, value FROM metadata")
        return Dict(String(row.key) => String(row.value) for row in eachrow(df))
    catch
        return Dict{String,String}()
    end
end

function _duckdb_connect(path::AbstractString; readonly::Bool = false, attempts::Int = 6)
    last_error = nothing
    for attempt in 1:attempts
        try
            return DBInterface.connect(DuckDB.DB, path; readonly = readonly)
        catch err
            last_error = err
            GC.gc()
            attempt < attempts && sleep(0.15 * attempt)
        end
    end
    throw(last_error)
end

function _duckdb_query_df(con, sql::AbstractString)
    result = DBInterface.execute(con, sql)
    try
        return DataFrames.DataFrame(result)
    finally
        try
            DBInterface.close!(result)
        catch
        end
    end
end

function _duckdb_execute!(con, sql::AbstractString)
    result = DBInterface.execute(con, sql)
    try
        return nothing
    finally
        try
            DBInterface.close!(result)
        catch
        end
    end
end

function _duckdb_replace_table!(con, df::DataFrames.DataFrame, table_name::AbstractString)
    safe_name = _duckdb_quote_identifier(table_name)
    view_name = "__iesa_input_df"
    _duckdb_register_table!(con, df, view_name)
    try
        _duckdb_execute!(con, "DROP TABLE IF EXISTS $(safe_name)")
        _duckdb_execute!(con, "CREATE TABLE $(safe_name) AS SELECT * FROM $(view_name)")
    finally
        _duckdb_unregister_table!(con, view_name)
    end
    return nothing
end

function _duckdb_registered_objects(con)
    if con isa DuckDB.DB
        return con.handle.registered_objects
    elseif con isa DuckDB.Connection
        return con.db.registered_objects
    end
    error("Unsupported DuckDB connection type: $(typeof(con))")
end

function _duckdb_register_table!(con, df::DataFrames.DataFrame, name::AbstractString)
    registered = _duckdb_registered_objects(con)
    registered[String(name)] = DuckDB.Tables.columntable(df)
    name_sql = _duckdb_quote_identifier(name)
    scan_arg = replace(String(name), "'" => "''")
    _duckdb_execute!(con, "CREATE OR REPLACE VIEW $(name_sql) AS SELECT * FROM julia_tbl_scan('$(scan_arg)')")
    return nothing
end

function _duckdb_unregister_table!(con, name::AbstractString)
    registered = _duckdb_registered_objects(con)
    pop!(registered, String(name), nothing)
    _duckdb_execute!(con, "DROP VIEW IF EXISTS $(_duckdb_quote_identifier(name))")
    return nothing
end

function _duckdb_quote_identifier(name::AbstractString)
    return "\"" * replace(String(name), "\"" => "\"\"") * "\""
end
