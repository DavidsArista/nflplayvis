#!/usr/bin/env python3
"""Streamlit dashboard for NFL player valuation outputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import streamlit as st

APP_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH: Final[Path] = APP_ROOT / "data" / "nfl_valuation.db"
MODELS_DIR: Final[Path] = APP_ROOT / "models"

METRIC_LABEL_TO_COL: Final[dict[str, str]] = {
    "Context-Adjusted Value / Snap": "context_adjusted_per_snap",
    "Context-Adjusted Total Value": "context_adjusted_total_value",
    "Success Over Expected / Snap": "success_over_expected_per_snap",
    "Success Over Expected / Opportunity": "success_over_expected_per_opp",
    "Value Percentile": "value_percentile",
}

DISPLAY_COLUMNS: Final[list[str]] = [
    "season",
    "position_group",
    "player_name",
    "primary_team",
    "tier_label",
    "is_qualified",
    "opportunities",
    "offense_snaps_resolved",
    "context_adjusted_per_snap",
    "context_adjusted_total_value",
    "success_over_expected_per_snap",
    "success_over_expected_per_opp",
    "value_percentile",
]


def apply_custom_styles() -> None:
    """Inject custom styling for a stakeholder-ready interface."""
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap");

        :root {
          --bg-main: #f4efe6;
          --bg-card: #fffdf8;
          --ink-900: #17212b;
          --ink-700: #354556;
          --accent: #b45f06;
          --accent-soft: #f8e8cf;
          --line: #d9cfbf;
        }

        html, body, [class*="css"] {
          font-family: "Space Grotesk", sans-serif;
        }

        .stApp {
          background:
            radial-gradient(1200px 320px at 10% -10%, #f7dec0 0%, rgba(247, 222, 192, 0) 60%),
            radial-gradient(900px 300px at 100% 0%, #e8f1ff 0%, rgba(232, 241, 255, 0) 60%),
            var(--bg-main);
          color: var(--ink-900);
        }

        .hero-card {
          border: 1px solid var(--line);
          background: linear-gradient(135deg, #fffefb, #fff8ed);
          border-radius: 16px;
          padding: 1.15rem 1.2rem 1rem 1.2rem;
          margin-bottom: 0.9rem;
          animation: riseIn 380ms ease-out;
        }

        .hero-title {
          margin: 0;
          color: var(--ink-900);
          font-weight: 700;
          letter-spacing: 0.15px;
        }

        .hero-sub {
          margin-top: 0.3rem;
          margin-bottom: 0;
          color: var(--ink-700);
          font-size: 0.95rem;
        }

        .kpi-caption {
          font-size: 0.74rem;
          color: var(--ink-700);
          letter-spacing: 0.3px;
          text-transform: uppercase;
        }

        .kpi-value {
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--ink-900);
          font-family: "IBM Plex Mono", monospace;
        }

        [data-testid="stMetricValue"] {
          font-family: "IBM Plex Mono", monospace;
          color: var(--ink-900);
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
          border: 1px solid var(--line);
          border-radius: 12px;
          overflow: hidden;
        }

        @keyframes riseIn {
          from { transform: translateY(6px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_table(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SQLite query and return a DataFrame."""
    connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def load_valuation_data(db_path: str, model_name: str) -> pd.DataFrame:
    """Load valuation rows for a model."""
    query = """
        SELECT
            model_name,
            valuation_run_ts,
            season,
            position_group,
            player_id,
            player_name,
            primary_team,
            opportunities,
            offense_snaps_resolved,
            avg_actual_success_rate,
            avg_expected_success_prob,
            total_success_over_expected,
            success_over_expected_per_opp,
            success_over_expected_per_snap,
            context_adjusted_per_snap,
            context_adjusted_total_value,
            value_percentile,
            value_zscore,
            tier_label,
            tier_rank,
            is_qualified
        FROM player_valuation
        WHERE model_name = ?
        ORDER BY season, position_group, context_adjusted_per_snap DESC;
    """
    frame = load_table(db_path, query, (model_name,))
    numeric_cols = [
        "season",
        "opportunities",
        "offense_snaps_resolved",
        "avg_actual_success_rate",
        "avg_expected_success_prob",
        "total_success_over_expected",
        "success_over_expected_per_opp",
        "success_over_expected_per_snap",
        "context_adjusted_per_snap",
        "context_adjusted_total_value",
        "value_percentile",
        "value_zscore",
        "tier_rank",
        "is_qualified",
    ]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["is_qualified"] = frame["is_qualified"].fillna(0).astype(int)
    return frame


@st.cache_data(show_spinner=False)
def load_validation_summary(db_path: str, model_name: str) -> pd.DataFrame:
    """Load summary validation metrics."""
    query = """
        SELECT
            season,
            position_group,
            n_players,
            n_pro_bowl_total_mapped,
            n_pro_bowl_in_qualified_pool,
            top_k,
            hits_at_k,
            precision_at_k,
            recall_at_k,
            spearman_with_benchmark,
            kendall_with_benchmark,
            pro_bowl_auc
        FROM valuation_validation_summary
        WHERE model_name = ?
        ORDER BY season, position_group;
    """
    return load_table(db_path, query, (model_name,))


@st.cache_data(show_spinner=False)
def load_training_runs(db_path: str, model_name: str) -> pd.DataFrame:
    """Load model-training run diagnostics."""
    query = """
        SELECT
            run_ts,
            train_seasons,
            validation_season,
            test_season,
            target_mode,
            rows_scored,
            feature_count,
            validation_log_loss,
            validation_roc_auc,
            test_log_loss,
            test_roc_auc,
            status
        FROM model_training_runs
        WHERE model_name = ?
        ORDER BY run_ts DESC;
    """
    return load_table(db_path, query, (model_name,))


def load_latest_metrics_json(model_name: str) -> dict | None:
    """Load most recent JSON metrics artifact for a model."""
    metric_files = sorted(MODELS_DIR.glob(f"{model_name}_*_metrics.json"))
    if not metric_files:
        return None
    latest_path = metric_files[-1]
    return json.loads(latest_path.read_text())


def format_player_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format player table for dashboard display."""
    out = frame.copy()
    out["value_percentile"] = (out["value_percentile"] * 100).round(1)
    out = out.rename(
        columns={
            "position_group": "Pos",
            "player_name": "Player",
            "primary_team": "Team",
            "tier_label": "Tier",
            "is_qualified": "Qualified",
            "opportunities": "Opps",
            "offense_snaps_resolved": "Snaps",
            "context_adjusted_per_snap": "Adj Value/Snap",
            "context_adjusted_total_value": "Adj Total Value",
            "success_over_expected_per_snap": "SOE/Snap",
            "success_over_expected_per_opp": "SOE/Opp",
            "value_percentile": "Percentile",
            "season": "Season",
        }
    )
    return out


def render_rankings_tab(valuation: pd.DataFrame) -> None:
    """Render ranking explorer for scouts and personnel users."""
    st.subheader("Player Rankings")
    st.caption(
        "Rank players by context-adjusted value, then filter by season, position, and role qualification."
    )

    seasons = sorted(valuation["season"].dropna().astype(int).unique().tolist())
    positions = sorted(valuation["position_group"].dropna().unique().tolist())
    teams = sorted(valuation["primary_team"].dropna().unique().tolist())

    col1, col2, col3, col4 = st.columns([1.0, 1.0, 1.2, 1.0])
    selected_season = col1.selectbox("Season", options=seasons, index=len(seasons) - 1)
    selected_positions = col2.multiselect(
        "Positions",
        options=positions,
        default=positions,
    )
    selected_metric_label = col3.selectbox(
        "Ranking Metric",
        options=list(METRIC_LABEL_TO_COL.keys()),
        index=0,
    )
    include_non_qualified = col4.checkbox("Include Low-Sample", value=False)

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
    top_n = c1.slider("Top N", min_value=10, max_value=150, value=40, step=5)
    min_snaps = c2.slider("Min Snaps", min_value=0, max_value=900, value=120, step=10)
    selected_teams = c3.multiselect("Team Filter (Optional)", options=teams, default=[])

    filtered = valuation[valuation["season"] == selected_season].copy()
    filtered = filtered[filtered["position_group"].isin(selected_positions)]
    filtered = filtered[filtered["offense_snaps_resolved"] >= min_snaps]
    if not include_non_qualified:
        filtered = filtered[filtered["is_qualified"] == 1]
    if selected_teams:
        filtered = filtered[filtered["primary_team"].isin(selected_teams)]

    ranking_col = METRIC_LABEL_TO_COL[selected_metric_label]
    filtered = filtered.sort_values(ranking_col, ascending=False).reset_index(drop=True)
    ranked = filtered.head(top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Players in Pool", f"{len(filtered):,}")
    kpi2.metric("Top Value/Snap", f"{ranked['context_adjusted_per_snap'].max():.4f}" if not ranked.empty else "N/A")
    kpi3.metric("Median SOE/Snap", f"{filtered['success_over_expected_per_snap'].median():.4f}" if not filtered.empty else "N/A")
    kpi4.metric(
        "Qualified Share",
        f"{(filtered['is_qualified'].mean() * 100):.1f}%" if not filtered.empty else "N/A",
    )

    if ranked.empty:
        st.warning("No players matched the current filters.")
        return

    # Football reasoning:
    # Horizontal ranking bars are easy for non-technical stakeholders to scan in roster meetings.
    bar_fig = px.bar(
        ranked.sort_values(ranking_col, ascending=True),
        x=ranking_col,
        y="player_name",
        color="tier_label",
        orientation="h",
        hover_data=["primary_team", "position_group", "offense_snaps_resolved", "opportunities"],
        title=f"Top {len(ranked)} Players by {selected_metric_label} ({selected_season})",
    )
    bar_fig.update_layout(height=max(460, 16 * len(ranked) + 160), margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(bar_fig, use_container_width=True)

    table_cols = [
        "rank",
        *DISPLAY_COLUMNS,
    ]
    st.dataframe(
        format_player_table(ranked[table_cols]),
        use_container_width=True,
        hide_index=True,
    )


def render_player_explorer_tab(valuation: pd.DataFrame) -> None:
    """Render individual player trend and profile view."""
    st.subheader("Player Trend Explorer")
    st.caption("Track a player across seasons with tier movement, value trend, and context-adjusted outputs.")

    positions = sorted(valuation["position_group"].dropna().unique().tolist())
    selected_position = st.selectbox("Position", positions, index=positions.index("WR") if "WR" in positions else 0)

    pool = valuation[valuation["position_group"] == selected_position].copy()
    pool = pool.sort_values(["player_name", "season"])
    player_options = (
        pool[["player_id", "player_name"]]
        .drop_duplicates()
        .sort_values("player_name")
        .reset_index(drop=True)
    )
    label_map = {
        row["player_id"]: f"{row['player_name']} ({row['player_id']})"
        for _, row in player_options.iterrows()
    }
    selected_player_id = st.selectbox(
        "Player",
        options=player_options["player_id"].tolist(),
        format_func=lambda player_id: label_map[player_id],
    )
    selected_player_name = label_map[selected_player_id].split(" (")[0]

    player_hist = pool[pool["player_id"] == selected_player_id].copy()
    player_hist = player_hist.sort_values("season")

    if player_hist.empty:
        st.warning("No valuation history found for this player.")
        return

    last_row = player_hist.iloc[-1]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Latest Team", str(last_row["primary_team"]))
    p2.metric("Latest Tier", str(last_row["tier_label"]))
    p3.metric("Latest Value/Snap", f"{last_row['context_adjusted_per_snap']:.4f}")
    p4.metric("Latest Percentile", f"{last_row['value_percentile'] * 100:.1f}%")

    trend_fig = px.line(
        player_hist,
        x="season",
        y=["context_adjusted_per_snap", "success_over_expected_per_snap"],
        markers=True,
        title=f"{selected_player_name} Value Trend",
    )
    trend_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), yaxis_title="Per Snap Value")
    st.plotly_chart(trend_fig, use_container_width=True)

    scatter_fig = px.scatter(
        pool,
        x="offense_snaps_resolved",
        y="context_adjusted_per_snap",
        color="tier_label",
        hover_data=["player_name", "primary_team", "season"],
        title=f"{selected_position}: Volume vs Value/Snap",
    )
    highlight = player_hist[player_hist["season"] == player_hist["season"].max()]
    scatter_fig.add_scatter(
        x=highlight["offense_snaps_resolved"],
        y=highlight["context_adjusted_per_snap"],
        mode="markers+text",
        text=highlight["player_name"],
        textposition="top center",
        marker=dict(size=15, color="#B45F06", symbol="star"),
        name="Selected Player",
    )
    scatter_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(scatter_fig, use_container_width=True)

    history_cols = [
        "season",
        "primary_team",
        "tier_label",
        "is_qualified",
        "opportunities",
        "offense_snaps_resolved",
        "context_adjusted_per_snap",
        "context_adjusted_total_value",
        "success_over_expected_per_snap",
        "value_percentile",
    ]
    st.dataframe(
        format_player_table(player_hist[history_cols]),
        use_container_width=True,
        hide_index=True,
    )


def render_validation_tab(validation_summary: pd.DataFrame) -> None:
    """Render validation diagnostics versus external benchmarks."""
    st.subheader("Validation Diagnostics")
    st.caption("How strongly valuation aligns with Pro Bowl selection and external performance proxies.")

    if validation_summary.empty:
        st.warning("Validation summary table is empty.")
        return

    numeric_cols = [
        "precision_at_k",
        "recall_at_k",
        "spearman_with_benchmark",
        "kendall_with_benchmark",
        "pro_bowl_auc",
    ]
    for column in numeric_cols:
        validation_summary[column] = pd.to_numeric(validation_summary[column], errors="coerce")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Precision@K", f"{validation_summary['precision_at_k'].mean():.3f}")
    c2.metric("Mean Spearman", f"{validation_summary['spearman_with_benchmark'].mean():.3f}")
    c3.metric("Mean Pro Bowl AUC", f"{validation_summary['pro_bowl_auc'].mean():.3f}")

    metric_choice = st.selectbox(
        "Heatmap Metric",
        options=[
            "precision_at_k",
            "recall_at_k",
            "spearman_with_benchmark",
            "kendall_with_benchmark",
            "pro_bowl_auc",
        ],
        index=0,
    )
    heatmap_frame = validation_summary.pivot(
        index="position_group",
        columns="season",
        values=metric_choice,
    ).reindex(["QB", "RB", "WR", "TE"])
    heatmap_fig = px.imshow(
        heatmap_frame,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"{metric_choice} by Position and Season",
    )
    heatmap_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(heatmap_fig, use_container_width=True)

    display_summary = validation_summary.copy()
    display_summary["precision_at_k"] = display_summary["precision_at_k"].round(3)
    display_summary["recall_at_k"] = display_summary["recall_at_k"].round(3)
    display_summary["spearman_with_benchmark"] = display_summary["spearman_with_benchmark"].round(3)
    display_summary["kendall_with_benchmark"] = display_summary["kendall_with_benchmark"].round(3)
    display_summary["pro_bowl_auc"] = display_summary["pro_bowl_auc"].round(3)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)


def render_model_tab(
    training_runs: pd.DataFrame,
    metrics_json: dict | None,
) -> None:
    """Render model and feature-importance diagnostics."""
    st.subheader("Model Diagnostics")
    st.caption("Training split quality, holdout metrics, and top feature drivers from the latest model artifact.")

    if training_runs.empty:
        st.warning("No model training runs found.")
        return

    latest = training_runs.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation AUC", f"{latest['validation_roc_auc']:.3f}")
    c2.metric("Test AUC", f"{latest['test_roc_auc']:.3f}")
    c3.metric("Validation LogLoss", f"{latest['validation_log_loss']:.3f}")
    c4.metric("Test LogLoss", f"{latest['test_log_loss']:.3f}")

    run_display = training_runs.copy()
    run_display["run_ts"] = pd.to_datetime(run_display["run_ts"], errors="coerce")
    run_display = run_display.sort_values("run_ts", ascending=False)
    st.dataframe(run_display, use_container_width=True, hide_index=True)

    if metrics_json and "top_feature_importance_gain" in metrics_json:
        importance = pd.DataFrame(metrics_json["top_feature_importance_gain"])
        importance = importance.head(20).sort_values("gain_importance", ascending=True)
        fig = px.bar(
            importance,
            x="gain_importance",
            y="feature",
            orientation="h",
            title="Top 20 Feature Importance (Gain)",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics JSON found in models/ for feature-importance display.")


def main() -> None:
    """Run Streamlit dashboard."""
    st.set_page_config(
        page_title="NFL Player Valuation Dashboard",
        page_icon="🏈",
        layout="wide",
    )
    apply_custom_styles()

    st.markdown(
        """
        <div class="hero-card">
          <h2 class="hero-title">NFL Player Valuation Dashboard</h2>
          <p class="hero-sub">
            Context-adjusted player value for front-office decision support:
            rankings, player trend analysis, and validation diagnostics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        db_path_input = st.text_input("SQLite DB Path", value=str(DEFAULT_DB_PATH))
        db_path = Path(db_path_input).expanduser().resolve()
        if not db_path.exists():
            st.error(f"Database not found: {db_path}")
            st.stop()

        model_options_query = """
            SELECT DISTINCT model_name
            FROM player_valuation
            ORDER BY model_name;
        """
        model_options = load_table(str(db_path), model_options_query)["model_name"].dropna().tolist()
        if not model_options:
            st.error("No model_name values found in player_valuation.")
            st.stop()
        selected_model = st.selectbox("Model Name", options=model_options, index=0)

        st.markdown("---")
        st.caption("Data Sources")
        st.caption("`player_valuation`, `valuation_validation_summary`, `model_training_runs`")

    valuation = load_valuation_data(str(db_path), selected_model)
    validation_summary = load_validation_summary(str(db_path), selected_model)
    training_runs = load_training_runs(str(db_path), selected_model)
    metrics_json = load_latest_metrics_json(selected_model)

    if valuation.empty:
        st.error(f"No valuation data found for model: {selected_model}")
        st.stop()

    tab_rankings, tab_player, tab_validation, tab_model = st.tabs(
        ["Rankings", "Player Explorer", "Validation", "Model"]
    )

    with tab_rankings:
        render_rankings_tab(valuation)

    with tab_player:
        render_player_explorer_tab(valuation)

    with tab_validation:
        render_validation_tab(validation_summary)

    with tab_model:
        render_model_tab(training_runs, metrics_json)


if __name__ == "__main__":
    main()
