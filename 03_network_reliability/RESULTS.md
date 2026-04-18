# Network Reliability Results

## Scenario Summary

| Scenario | Route | Latency (ms) | Bottleneck (Mbps) | Reliability | Simulated Availability |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | Kyiv -> Lviv -> Berlin | 39.0 | 920 | 0.9487 | 0.9474 |
| baseline | Odessa -> Lviv -> Warsaw | 38.0 | 760 | 0.9486 | 0.9466 |
| maintenance | Kyiv -> Lviv -> Berlin | 39.0 | 920 | 0.9487 | 0.9474 |
| maintenance | Dnipro -> Kharkiv -> Warsaw | 36.0 | 720 | 0.9476 | 0.9492 |
| peak | Kharkiv -> Dnipro -> Berlin | 44.0 | 480 | 0.9417 | 0.9424 |
| peak | Odessa -> Lviv -> Berlin | 45.0 | 570 | 0.9447 | 0.9480 |

## Notes

- Reliability-aware scoring remains stable across baseline, maintenance, and peak scenarios
- Monte Carlo availability closely tracks the analytical path reliability
- Artifacts: `outputs/network_paths.png`, `outputs/route_summary.csv`
