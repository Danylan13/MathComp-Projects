from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent


PROJECTS = {
    "Energy Forecasting": {
        "folder": "01_energy_load_forecasting",
        "image": ROOT / "01_energy_load_forecasting" / "outputs" / "forecast_diagnostics.png",
        "extra_image": ROOT / "01_energy_load_forecasting" / "outputs" / "feature_importance.png",
        "results": ROOT / "01_energy_load_forecasting" / "outputs" / "benchmark_summary.csv",
        "summary": "Short-term demand forecasting with ridge regression, feature ablations, and regime-aware diagnostics.",
        "run": "python 01_energy_load_forecasting/src/forecasting_pipeline.py",
        "headline": ("Ridge RMSE", "15.314 MW", "vs lag-24 baseline"),
        "skills": ["time-series modeling", "feature engineering", "regularization", "ablation analysis"],
    },
    "Predictive Maintenance": {
        "folder": "02_predictive_maintenance",
        "image": ROOT / "02_predictive_maintenance" / "outputs" / "maintenance_diagnostics.png",
        "extra_image": ROOT / "02_predictive_maintenance" / "outputs" / "maintenance_tradeoffs.png",
        "results": ROOT / "02_predictive_maintenance" / "outputs" / "model_comparison.csv",
        "summary": "Failure-risk scoring with anomaly detection, logistic regression, business cost, and drift stress tests.",
        "run": "python 02_predictive_maintenance/src/maintenance_model.py",
        "headline": ("Best F1", "0.571", "logistic regression"),
        "skills": ["classification", "anomaly detection", "cost-sensitive evaluation", "robustness testing"],
    },
    "Network Reliability": {
        "folder": "03_network_reliability",
        "image": ROOT / "03_network_reliability" / "outputs" / "network_paths.png",
        "extra_image": ROOT / "03_network_reliability" / "outputs" / "scenario_comparison.png",
        "results": ROOT / "03_network_reliability" / "outputs" / "route_summary.csv",
        "summary": "Reliability-aware routing under latency, capacity, and scenario constraints with resilience summaries.",
        "run": "python 03_network_reliability/src/network_analysis.py",
        "headline": ("Avg reliability", "0.947", "selected routes"),
        "skills": ["graph algorithms", "simulation", "network analysis", "scenario modeling"],
    },
    "Portfolio Optimization": {
        "folder": "04_portfolio_optimization",
        "image": ROOT / "04_portfolio_optimization" / "outputs" / "portfolio_backtest.png",
        "extra_image": ROOT / "04_portfolio_optimization" / "outputs" / "strategy_risk_return.png",
        "results": ROOT / "04_portfolio_optimization" / "outputs" / "strategy_comparison.csv",
        "summary": "Risk-aware allocation, backtesting, rolling rebalancing, and transaction-cost benchmarking.",
        "run": "python 04_portfolio_optimization/src/portfolio_analysis.py",
        "headline": ("Optimized max DD", "-0.0043", "reduced downside"),
        "skills": ["optimization", "risk metrics", "backtesting", "portfolio analytics"],
    },
    "Kalman Sensor Fusion": {
        "folder": "05_kalman_sensor_fusion",
        "image": ROOT / "05_kalman_sensor_fusion" / "outputs" / "trajectory_comparison.png",
        "extra_image": ROOT / "05_kalman_sensor_fusion" / "outputs" / "dropout_sensitivity.png",
        "results": ROOT / "05_kalman_sensor_fusion" / "outputs" / "robustness_summary.csv",
        "summary": "Trajectory estimation with filtering, smoothing, and GPS-dropout robustness analysis.",
        "run": "python 05_kalman_sensor_fusion/src/kalman_fusion.py",
        "headline": ("Smoothed RMSE", "1.206", "best trajectory estimate"),
        "skills": ["state estimation", "kalman filtering", "signal processing", "robustness analysis"],
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dataframe_markdown(rows: list[dict[str, str]], max_rows: int = 8) -> str:
    if not rows:
        return "_No rows available._"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:max_rows]:
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *body])


st.set_page_config(page_title="MathComp Projects", layout="wide")
st.title("MathComp Projects")
st.caption("Interactive showcase for applied mathematics and computer engineering mini-projects.")

overview_col, image_col = st.columns([1.2, 1.0], gap="large")
with overview_col:
    st.subheader("Repository Snapshot")
    st.markdown(
        """
        - 5 data-driven projects with reproducible outputs
        - unified `CLI` entry points for each analysis pipeline
        - diagnostics, stress tests, ablations, and comparative benchmarks
        - root dashboard summarizing the strongest metric from each project
        """
    )
    st.code("python run_all.py", language="bash")
    st.code("streamlit run streamlit_app.py", language="bash")
with image_col:
    st.image(str(ROOT / "outputs" / "overview_dashboard.png"), caption="Repository dashboard")

gallery = st.columns(len(PROJECTS))
for column, (name, details) in zip(gallery, PROJECTS.items()):
    with column:
        st.metric(details["headline"][0], details["headline"][1], details["headline"][2])
        st.caption(name)

selected_project = st.selectbox("Choose a project", list(PROJECTS.keys()))
project = PROJECTS[selected_project]

st.subheader(selected_project)
st.write(project["summary"])
st.caption("Skills: " + " | ".join(project["skills"]))

tab1, tab2, tab3 = st.tabs(["Visuals", "Metrics", "Run"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(str(project["image"]), use_container_width=True)
    with col2:
        st.image(str(project["extra_image"]), use_container_width=True)

with tab2:
    rows = read_csv_rows(project["results"])
    st.markdown(dataframe_markdown(rows), unsafe_allow_html=False)

with tab3:
    st.code(project["run"], language="bash")
    st.markdown(
        f"[Open project README](./{project['folder']}/README.md)  \n"
        f"[Open results file](./{project['folder']}/RESULTS.md)"
    )
