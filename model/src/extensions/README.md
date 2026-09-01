# `src/extensions/` — project-specific opt-in extensions

This directory holds **opt-in, project-specific** add-on constraints that are
*not* part of the IESA-Opt core model. Anything in here can be removed without
breaking the rest of the package.

Higher-level analysis campaigns live in [`src/workflows/`](../workflows/)
instead. Scenario Space and MGA are workflows because they orchestrate many
model builds or alternative solves; they are not registered here unless they
add constraints directly to one model build.

## How extensions plug in

The dispatcher [`extensions.jl`](extensions.jl) is called once at the end of
each `build_*_lp!` function in [`src/solve.jl`](../solve.jl). Each extension's
hook is gated on a Symbol in `md.params.extensions::Set{Symbol}`, populated by
a UI toggle.

```text
build_*_lp!(m, md)                       (src/solve.jl)
   ├── add_*_variables!
   ├── add_*_constraints!  (core)
   ├── add_objective!
   └── apply_extensions!(m, vars, md; mode = :fh|:ts|:annual)   ← single hook
            └── if :multi_region in md.params.extensions
                    apply_multi_region!(m, vars, md; mode)
```

Activating no extensions (`md.params.extensions == Set()`) makes
`apply_extensions!` a no-op — the core model is bit-exactly unchanged.

## Available extensions

| Symbol           | File                                       | Purpose                                                   | UI toggle    |
|------------------|--------------------------------------------|-----------------------------------------------------------|--------------|
| `:multi_region`  | [`multi_region.jl`](multi_region.jl)       | Regional caps on stock / use ("Komar" Section in AIMMS).  | `MultiRegion` |

## Adding a new extension

1. Create `src/extensions/<your_extension>.jl` defining a hook function:
   ```julia
   function apply_<your_extension>!(m::JuMP.Model, vars, md::ModelData; mode::Symbol = :fh) -> Int
       # ... add your constraints; return the count
   end
   ```
2. Add `include("extensions/<your_extension>.jl")` to
   [`src/IESAOpt.jl`](../IESAOpt.jl) under the
   `Project-specific extensions` block.
3. Add a dispatch line in [`extensions.jl`](extensions.jl):
   ```julia
   if :<your_extension> in md.params.extensions
       n_added += apply_<your_extension>!(m, vars, md; mode = mode)
   end
   ```
4. Wire a UI toggle (HTML + `app.js`) that adds your `:name` Symbol to the run
   payload, then read it in `_normalize_run_config` /
   `_run_ui_job!` in [`src/ui_server.jl`](../ui_server.jl) to populate
   `md.params.extensions`.

## Removing the entire `extensions` category

Mechanical, no surgery on core model files:

1. Delete the `src/extensions/` directory.
2. Remove the two `include(...)` lines for the extensions in
   [`src/IESAOpt.jl`](../IESAOpt.jl).
3. Remove the `extensions::Set{Symbol}` field on `ModelParams` in
   [`src/types.jl`](../types.jl).
4. Remove the three `apply_extensions!(...)` calls in
   [`src/solve.jl`](../solve.jl).
5. Remove any UI elements (HTML toggle + `app.js` payload field +
   `ui_server.jl` parse logic).

That's it — no model/* file is touched.
