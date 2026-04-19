# Network Reliability Results

## Executive Summary

All six traffic requests remain feasible across baseline, maintenance, and peak scenarios. Reliability-aware routing stays close to the best-feasible latency routes while explicitly tracking survivability and degraded-condition resilience.

## Scenario Summary

| Scenario | Route | Latency (ms) | Bottleneck (Mbps) | Reliability | Simulated Availability |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | Kyiv -> Lviv -> Berlin | 39.0 | 920 | 0.9487 | 0.9474 |
| baseline | Odessa -> Lviv -> Warsaw | 38.0 | 760 | 0.9486 | 0.9466 |
| maintenance | Kyiv -> Lviv -> Berlin | 39.0 | 920 | 0.9487 | 0.9474 |
| maintenance | Dnipro -> Kharkiv -> Warsaw | 36.0 | 720 | 0.9476 | 0.9492 |
| peak | Kharkiv -> Dnipro -> Berlin | 44.0 | 480 | 0.9417 | 0.9424 |
| peak | Odessa -> Lviv -> Berlin | 45.0 | 570 | 0.9447 | 0.9480 |

## What Improved

- Reliability-aware route scoring stays stable across all tested operating scenarios
- Monte Carlo availability remains tightly aligned with analytical reliability
- The resilience export adds degraded-path reliability and a redundancy gap against latency-only routing

## Additional Analysis

- `scenario_resilience.csv` summarizes hop count, degraded reliability, and the redundancy gap for each selected route
- `scenario_comparison.png` makes it easier to compare base reliability against a harsher degraded assumption
- `route_summary.csv` keeps the chosen route and the latency-only feasible baseline side by side

## Why This Matters

This project is stronger because it does not stop at “a path exists.” It now asks whether that path remains attractive when the network is stressed or partially degraded, which is much closer to real backbone engineering.

## Limitations

- The graph is intentionally compact, so route diversity is limited
- The degradation penalty is a simplified proxy rather than a full correlated-failure model
- Latency and reliability are combined through a hand-tuned score, not via formal multi-objective optimization

## Artifacts

- `outputs/network_paths.png`
- `outputs/scenario_comparison.png`
- `outputs/route_summary.csv`
- `outputs/scenario_resilience.csv`
