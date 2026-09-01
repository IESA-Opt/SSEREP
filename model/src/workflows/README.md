# `src/workflows/` - analysis and solve workflows

This directory contains higher-level workflows that orchestrate one or more IESA-Opt model builds. Workflows are different from model extensions:

- `src/extensions/` contains optional mathematical add-ons that change a single model build, such as `:multi_region` regional cap constraints.
- `src/workflows/` contains campaign logic that runs, mutates, compares, or re-solves models.

## Available workflows

| Directory | Purpose |
|-----------|---------|
| `scenario_space/` | Sampling-based Scenario Space campaigns: parameter definitions, samplers, per-variant mutation, campaign runners, persistence, and analysis helpers. |
| `mga/` | Model-Generated Alternatives workflow: baseline solve, near-optimal cost cap, directional alternatives, ORACLE refinement, and MGA result summaries. |

A workflow may choose to enable a model extension by setting `md.params.extensions`, but the workflow itself should not be registered in `src/extensions/extensions.jl` unless it adds constraints directly to one model build.
