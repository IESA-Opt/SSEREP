# `src/model/` — JuMP constraint families

This folder contains one file per major variable, objective, or constraint
family:

| File | Contents |
|---|---|
| `variables.jl` | JuMP variable declarations |
| `objective.jl` | `totalCosts` objective expression |
| `stock.jl` | technology stock evolution, decomStock, retrofit, economic decommissioning, linked investments |
| `balance.jl` | activity, material-conversion, capacity, and emission-target balances |
| `hourly.jl` | full-hourly dispatch, capacity, ramping, storage, CHP, shedding, and interconnection constraints |
| `ts.jl` | representative-day time-slice constraints and helper logic |
| `cyclic_closures.jl` | annual cyclic closure constraints |
| `infrastructure.jl` | infrastructure-specific constraints |
| `policy.jl` | optional policy constraints |
