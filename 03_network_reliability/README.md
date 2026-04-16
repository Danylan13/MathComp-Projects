# Network Reliability Analysis

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
