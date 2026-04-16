# Applied Math + Computer Engineering Portfolio

This repository contains five practical mini-projects designed to showcase skills in applied mathematics, algorithms, programming, and engineering.

The portfolio is built to look like a set of compact engineering products rather than classroom exercises. Each project includes:

- a clear business or engineering problem
- a short technical explanation
- reproducible Python code
- a lightweight experiment or simulation
- practical metrics or outputs

## Portfolio Projects

| Project | Theme | Main Skills |
| --- | --- | --- |
| `01_optimization_lab` | Numerical optimization for predictive modeling | gradient descent, feature scaling, regression, model evaluation |
| `02_sensor_anomaly_detection` | Sensor monitoring and anomaly detection | statistics, PCA, signal monitoring, thresholding |
| `03_route_planning` | Graph algorithms for route planning | Dijkstra, A*, heuristics, path reconstruction |
| `04_portfolio_risk_engine` | Monte Carlo risk estimation | stochastic simulation, covariance, VaR, CVaR |
| `05_svd_image_compression` | Linear algebra for data compression | SVD, reconstruction error, rank approximation |

## Why This Repository Works For Internships

The projects intentionally combine:

- applied mathematics: linear algebra, optimization, probability, numerical methods
- computer engineering thinking: simulation, system constraints, monitoring, performance trade-offs
- software engineering hygiene: structure, documentation, reproducibility, clean code

This makes the repository suitable for applications in:

- quantitative internships
- data science and machine learning internships
- software engineering internships with a strong algorithmic component
- embedded, systems, and signal-processing adjacent roles

## Recommended Repository Structure

```text
MathComp-Projects/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- LINKEDIN_PROFILE.md
|-- 01_optimization_lab/
|-- 02_sensor_anomaly_detection/
|-- 03_route_planning/
|-- 04_portfolio_risk_engine/
`-- 05_svd_image_compression/
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run any project from the repository root. Example:

```bash
python 01_optimization_lab/src/optimization_lab.py
python 02_sensor_anomaly_detection/src/sensor_monitor.py
python 03_route_planning/src/route_planner.py
python 04_portfolio_risk_engine/src/risk_engine.py
python 05_svd_image_compression/src/svd_compression.py
```

## Positioning Tips

When publishing this repository on GitHub:

- pin it on your profile
- write a short repository description focused on impact
- keep the README visual and easy to scan
- use consistent commit messages
- add a LinkedIn project entry for one or two of the strongest subprojects

Suggested repository description:

> Portfolio of 5 practical projects in applied mathematics and computer engineering: optimization, anomaly detection, graph algorithms, risk simulation, and SVD-based compression.

## Next Improvements

- add visual outputs as images in each project README
- add unit tests and CI
- deploy one project as a small web demo
- replace synthetic data with public datasets
