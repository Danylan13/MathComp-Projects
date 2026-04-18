# Network Reliability Analysis

![Network paths](outputs/network_paths.png)

## Problem

Route traffic demands across a backbone network while respecting latency and capacity constraints and accounting for link failure probabilities.

## Data

Files:

- `data/network_edges.csv`
- `data/demand_scenarios.csv`

## Method

Each scenario builds a constrained routing problem on a weighted graph. Paths are selected by minimizing a composite score that combines latency and reliability penalty, while rejecting infeasible routes.

Scenario logic includes:

- baseline operation
- maintenance outage on a critical edge
- peak demand with reduced available capacities

After the routing decision is made, the selected paths are also evaluated with a Monte Carlo availability estimate to compare empirical route survivability against the analytical reliability score.

## Key Results

- All six traffic requests remain feasible across the tested operating scenarios
- Reliability-aware routing achieves end-to-end path reliability around `0.94` to `0.95`
- Simulated path availability closely matches analytical reliability, validating the link-failure model

## Benchmarks

| Scenario | Route | Latency (ms) | Reliability | Simulated Availability |
| --- | --- | ---: | ---: | ---: |
| baseline | Kyiv -> Berlin | 39.0 | 0.9487 | 0.9474 |
| maintenance | Dnipro -> Warsaw | 36.0 | 0.9476 | 0.9492 |
| peak | Odessa -> Berlin | 45.0 | 0.9447 | 0.9480 |

The full scenario table is available in [RESULTS.md](RESULTS.md) and `outputs/route_summary.csv`.

## Run

```bash
python 03_network_reliability/src/network_analysis.py
```

## Output

The script reports:

- chosen path per demand
- end-to-end latency
- bottleneck capacity
- path reliability
- Monte Carlo availability estimate and route summary table
- infeasible requests, if any
- saved topology plot in `outputs/network_paths.png`
