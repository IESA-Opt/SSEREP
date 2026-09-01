# =============================================================================
# extensions.jl — opt-in extension dispatcher
#
# Project-specific add-on constraints live in src/extensions/*.jl. They are
# disabled by default; activate by adding the extension's name (Symbol) to
# `md.params.extensions::Set{Symbol}`, typically wired through a UI toggle.
#
# This file is part of the **Extensions** category — it is *not* part of the
# IESA-Opt core. Removing the entire category is mechanical:
#
#   1. Delete the `src/extensions/` directory.
#   2. Remove the two `include(...)` lines for the extensions in `IESAOpt.jl`.
#   3. Remove the `extensions::Set{Symbol}` field on `ModelParams` (types.jl).
#   4. Remove the three `apply_extensions!(...)` calls in `solve.jl`.
#   5. Remove any UI toggles that populate `md.params.extensions`.
#
# To add a new extension:
#   1. Create `src/extensions/<your_extension>.jl` defining
#      `apply_<your_extension>!(m, vars, md; mode::Symbol) -> Int`.
#   2. Add `include("extensions/<your_extension>.jl")` to `IESAOpt.jl`.
#   3. Add a dispatch line below in `apply_extensions!`.
#   4. Wire a UI toggle that adds your `:name` Symbol to `md.params.extensions`.
# =============================================================================

"""
    apply_extensions!(m::JuMP.Model, vars, md::ModelData; mode::Symbol) -> Int

Run all enabled extension hooks. Called once at the end of each `build_*_lp!`.
Iterates the *known* extension list and invokes any whose Symbol appears in
`md.params.extensions`. Returns the total number of constraints added.

`mode` is one of `:annual`, `:fh`, `:ts` and is forwarded to extensions in
case they want to specialize per build path.
"""
function apply_extensions!(m::JuMP.Model, vars, md::ModelData; mode::Symbol = :fh)
    isempty(md.params.extensions) && return 0
    n_added = 0

    # ---- Registered extensions -----------------------------------------------
    if :multi_region in md.params.extensions
        n_added += apply_multi_region!(m, vars, md; mode = mode)
    end
    # (add new extensions here — keep them alphabetical for clarity)

    if n_added > 0
        @info "Extensions: applied $(length(md.params.extensions)) hook(s) → $n_added constraints" mode extensions = collect(md.params.extensions)
    end
    return n_added
end
