#!/usr/bin/env python3
"""Streamlit frontend for NFL player valuation outputs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH: Final[Path] = APP_ROOT / "data" / "nfl_valuation.db"
MODELS_DIR: Final[Path] = APP_ROOT / "models"
POSITION_ORDER: Final[list[str]] = ["QB", "RB", "WR", "TE"]

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

COMPARE_METRICS: Final[list[str]] = [
    "context_adjusted_per_snap",
    "success_over_expected_per_snap",
    "opportunities_per_snap",
    "value_percentile",
    "reliability_weight",
]


def apply_custom_styles() -> None:
    """Inject custom styling for a stakeholder-ready interface."""
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap");

        :root {
          --bg-main: #f3efe7;
          --bg-card: #fffdf8;
          --ink-900: #10212f;
          --ink-700: #354659;
          --accent: #b15a0f;
          --accent-soft: #fce6cd;
          --line: #d8c9b4;
          --navy-soft: #e8f1fb;
        }

        html, body, [class*="css"] {
          font-family: "Sora", sans-serif;
        }

        .stApp {
          background:
            radial-gradient(980px 300px at 8% -10%, #f6dcbf 0%, rgba(246, 220, 191, 0) 62%),
            radial-gradient(840px 280px at 100% 0%, #e4eefc 0%, rgba(228, 238, 252, 0) 62%),
            var(--bg-main);
          color: var(--ink-900);
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        .hero-card {
          border: 1px solid var(--line);
          background: linear-gradient(130deg, #fffefb 0%, #fff7ea 50%, #f6f9ff 100%);
          border-radius: 16px;
          padding: 1.15rem 1.2rem 1rem 1.2rem;
          margin-bottom: 0.95rem;
          box-shadow: 0 2px 12px rgba(16, 33, 47, 0.05);
          animation: fadeRise 360ms ease-out;
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
          font-size: 0.94rem;
          line-height: 1.45;
        }

        [data-testid="stMetric"] {
          border: 1px solid var(--line);
          background: linear-gradient(180deg, #fffefb, #fffaef);
          border-radius: 14px;
          padding: 0.5rem 0.55rem;
        }

        [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
          font-family: "IBM Plex Mono", monospace;
          color: var(--ink-900);
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
          border: 1px solid var(--line);
          border-radius: 12px;
          overflow: hidden;
          background: var(--bg-card);
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #f8f4eb, #f2efe7);
          border-right: 1px solid var(--line);
        }

        div[data-baseweb="tab-list"] {
          gap: 0.4rem;
        }

        button[data-baseweb="tab"] {
          border: 1px solid var(--line);
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.72);
          padding: 0.3rem 0.8rem;
          color: var(--ink-700);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
          background: linear-gradient(180deg, #ffeccf, #ffe3be);
          border-color: #d8b58f;
          color: #3e2d1b;
        }

        @keyframes fadeRise {
          from { transform: translateY(6px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _open_readonly_connection(db_path: str) -> sqlite3.Connection:
    """Return a read-only SQLite connection."""
    resolved_path = Path(db_path).expanduser().resolve()
    return sqlite3.connect(
        f"file:{resolved_path}?mode=ro",
        uri=True,
        timeout=30,
    )


@st.cache_data(show_spinner=False)
def load_table(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    """Run a SQLite query and return a DataFrame."""
    connection = _open_readonly_connection(db_path)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def table_exists(db_path: str, table_name: str) -> bool:
    """Return True if a table exists in SQLite."""
    query = """
        SELECT COUNT(*) AS has_table
        FROM sqlite_master
        WHERE type='table' AND name = ?;
    """
    frame = load_table(db_path, query, (table_name,))
    return int(frame.iloc[0]["has_table"]) > 0


@st.cache_data(show_spinner=False)
def load_model_options(db_path: str) -> list[str]:
    """Load model names available in player_valuation."""
    query = """
        SELECT DISTINCT model_name
        FROM player_valuation
        ORDER BY model_name;
    """
    return load_table(db_path, query)["model_name"].dropna().tolist()


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
            teams_played,
            opportunities,
            offense_snaps,
            offense_snaps_resolved,
            opportunities_per_snap,
            avg_actual_success_rate,
            avg_expected_success_prob,
            total_success_over_expected,
            success_over_expected_per_opp,
            success_over_expected_per_snap,
            reliability_weight,
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
        "offense_snaps",
        "offense_snaps_resolved",
        "opportunities_per_snap",
        "avg_actual_success_rate",
        "avg_expected_success_prob",
        "total_success_over_expected",
        "success_over_expected_per_opp",
        "success_over_expected_per_snap",
        "reliability_weight",
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
    """Load summary validation metrics if available."""
    if not table_exists(db_path, "valuation_validation_summary"):
        return pd.DataFrame()

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
    """Load model-training run diagnostics if available."""
    if not table_exists(db_path, "model_training_runs"):
        return pd.DataFrame()

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
    if "value_percentile" in out.columns:
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
            "teams_played": "Teams",
        }
    )
    return out


def format_number(value: float | int | None, decimals: int = 3) -> str:
    """Format a metric value for display."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def render_overview_tab(valuation: pd.DataFrame) -> None:
    """Render high-level league and market summary."""
    st.subheader("League Overview")
    st.caption(
        "Fast front-office readout across seasons, team value concentration, and position market shape."
    )

    seasons = sorted(valuation["season"].dropna().astype(int).unique().tolist())
    positions = sorted(valuation["position_group"].dropna().unique().tolist())
    latest_season = seasons[-1]

    c1, c2, c3 = st.columns([1.0, 1.1, 1.2])
    selected_season = c1.selectbox("Season", options=seasons, index=len(seasons) - 1, key="overview_season")
    selected_positions = c2.multiselect("Positions", options=positions, default=positions, key="overview_pos")
    metric_label = c3.selectbox(
        "Focus Metric",
        options=list(METRIC_LABEL_TO_COL.keys()),
        index=0,
        key="overview_metric",
    )

    ranking_col = METRIC_LABEL_TO_COL[metric_label]
    filtered = valuation[(valuation["season"] == selected_season) & valuation["position_group"].isin(selected_positions)]
    if filtered.empty:
        st.warning("No players available for the selected filters.")
        return

    qualified = filtered[filtered["is_qualified"] == 1].copy()
    top_pool = qualified if not qualified.empty else filtered
    top_row = top_pool.sort_values(ranking_col, ascending=False).iloc[0]

    top_team_frame = (
        top_pool.groupby("primary_team", as_index=False)[ranking_col]
        .mean()
        .sort_values(ranking_col, ascending=False)
    )
    top_team = top_team_frame.iloc[0]["primary_team"] if not top_team_frame.empty else "N/A"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players Tracked", f"{len(filtered):,}")
    k2.metric("Qualified Players", f"{len(qualified):,}")
    k3.metric("Top Player", f"{top_row['player_name']} ({top_row['primary_team']})")
    k4.metric("Top Team (Avg)", str(top_team))

    team_leader = (
        top_pool.groupby(["primary_team"], as_index=False)
        .agg(avg_metric=(ranking_col, "mean"), players=("player_id", "nunique"))
        .sort_values("avg_metric", ascending=False)
        .head(12)
    )
    fig_team = px.bar(
        team_leader.sort_values("avg_metric", ascending=True),
        x="avg_metric",
        y="primary_team",
        color="players",
        orientation="h",
        title=f"Top Teams by Average {metric_label}",
        labels={"avg_metric": metric_label, "primary_team": "Team", "players": "Players"},
        color_continuous_scale="YlOrBr",
    )
    fig_team.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=430)

    fig_pos = px.box(
        filtered,
        x="position_group",
        y=ranking_col,
        color="position_group",
        category_orders={"position_group": POSITION_ORDER},
        points="outliers",
        title=f"Position Distribution: {metric_label}",
    )
    fig_pos.update_layout(margin=dict(l=20, r=20, t=60, b=20), showlegend=False, height=430)

    chart_left, chart_right = st.columns([1.1, 1.0])
    chart_left.plotly_chart(fig_team, use_container_width=True)
    chart_right.plotly_chart(fig_pos, use_container_width=True)

    tier_mix = (
        filtered.groupby(["position_group", "tier_label"], as_index=False)["player_id"]
        .nunique()
        .rename(columns={"player_id": "players"})
    )
    fig_tier = px.bar(
        tier_mix,
        x="position_group",
        y="players",
        color="tier_label",
        category_orders={"position_group": POSITION_ORDER},
        title=f"Tier Mix by Position ({selected_season})",
    )
    fig_tier.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=390)
    st.plotly_chart(fig_tier, use_container_width=True)

    if selected_season == latest_season:
        st.caption("Displaying latest season snapshot.")
    else:
        st.caption(f"Displaying historical snapshot for {selected_season}.")


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
    selected_season = col1.selectbox("Season", options=seasons, index=len(seasons) - 1, key="rankings_season")
    selected_positions = col2.multiselect(
        "Positions",
        options=positions,
        default=positions,
        key="rankings_pos",
    )
    selected_metric_label = col3.selectbox(
        "Ranking Metric",
        options=list(METRIC_LABEL_TO_COL.keys()),
        index=0,
        key="rankings_metric",
    )
    include_non_qualified = col4.checkbox("Include Low-Sample", value=False, key="rankings_low_sample")

    c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
    top_n = c1.slider("Top N", min_value=10, max_value=150, value=40, step=5, key="rankings_topn")
    min_snaps = c2.slider("Min Snaps", min_value=0, max_value=900, value=120, step=10, key="rankings_snaps")
    selected_teams = c3.multiselect("Team Filter (Optional)", options=teams, default=[], key="rankings_teams")

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
    kpi2.metric("Top Value/Snap", format_number(ranked["context_adjusted_per_snap"].max() if not ranked.empty else None, 4))
    kpi3.metric("Median SOE/Snap", format_number(filtered["success_over_expected_per_snap"].median() if not filtered.empty else None, 4))
    kpi4.metric(
        "Qualified Share",
        f"{(filtered['is_qualified'].mean() * 100):.1f}%" if not filtered.empty else "N/A",
    )

    if ranked.empty:
        st.warning("No players matched the current filters.")
        return

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

    scatter = px.scatter(
        ranked,
        x="offense_snaps_resolved",
        y=ranking_col,
        size="opportunities",
        color="position_group",
        hover_data=["player_name", "primary_team", "tier_label"],
        title="Volume vs Value for Ranked Pool",
    )
    scatter.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
    st.plotly_chart(scatter, use_container_width=True)

    table_cols = ["rank", *DISPLAY_COLUMNS]
    display_ranked = format_player_table(ranked[table_cols])
    st.dataframe(display_ranked, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Rankings CSV",
        data=ranked[table_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"rankings_{selected_season}_{selected_metric_label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_player_explorer_tab(valuation: pd.DataFrame) -> None:
    """Render individual player trend and profile view."""
    st.subheader("Player Trend Explorer")
    st.caption("Track a player across seasons with tier movement, value trend, and context-adjusted outputs.")

    positions = sorted(valuation["position_group"].dropna().unique().tolist())
    default_position = positions.index("WR") if "WR" in positions else 0
    selected_position = st.selectbox("Position", positions, index=default_position, key="explorer_pos")

    pool = valuation[valuation["position_group"] == selected_position].copy()
    pool = pool.sort_values(["player_name", "season"])
    player_options = pool[["player_id", "player_name"]].drop_duplicates().sort_values("player_name").reset_index(drop=True)
    label_map = {
        row["player_id"]: f"{row['player_name']} ({row['player_id']})"
        for _, row in player_options.iterrows()
    }
    selected_player_id = st.selectbox(
        "Player",
        options=player_options["player_id"].tolist(),
        format_func=lambda player_id: label_map[player_id],
        key="explorer_player",
    )
    selected_player_name = label_map[selected_player_id].split(" (")[0]

    player_hist = pool[pool["player_id"] == selected_player_id].copy().sort_values("season")
    if player_hist.empty:
        st.warning("No valuation history found for this player.")
        return

    last_row = player_hist.iloc[-1]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Latest Team", str(last_row["primary_team"]))
    p2.metric("Latest Tier", str(last_row["tier_label"]))
    p3.metric("Latest Value/Snap", format_number(last_row["context_adjusted_per_snap"], 4))
    p4.metric("Latest Percentile", f"{last_row['value_percentile'] * 100:.1f}%")

    trend_fig = px.line(
        player_hist,
        x="season",
        y=["context_adjusted_per_snap", "success_over_expected_per_snap"],
        markers=True,
        title=f"{selected_player_name} Value Trend",
        labels={"value": "Per Snap Value", "variable": "Metric"},
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
        "teams_played",
    ]
    st.dataframe(
        format_player_table(player_hist[history_cols]),
        use_container_width=True,
        hide_index=True,
    )


def render_player_compare_tab(valuation: pd.DataFrame) -> None:
    """Render side-by-side player comparison."""
    st.subheader("Player Compare")
    st.caption("Head-to-head view for trade, extension, and depth-chart decisions.")

    seasons = sorted(valuation["season"].dropna().astype(int).unique().tolist())
    positions = sorted(valuation["position_group"].dropna().unique().tolist())
    latest_season = seasons[-1]

    c1, c2 = st.columns([1.0, 1.0])
    selected_season = c1.selectbox("Season", options=seasons, index=len(seasons) - 1, key="compare_season")
    selected_position = c2.selectbox("Position", options=positions, index=positions.index("WR") if "WR" in positions else 0, key="compare_pos")

    pool = valuation[
        (valuation["season"] == selected_season) & (valuation["position_group"] == selected_position)
    ].copy()
    pool = pool.sort_values("context_adjusted_per_snap", ascending=False)

    if pool.empty:
        st.warning("No players available for this season/position.")
        return

    player_lookup = (
        pool[["player_id", "player_name", "primary_team"]]
        .drop_duplicates()
        .assign(label=lambda x: x["player_name"] + " (" + x["primary_team"] + ")")
    )
    options = player_lookup["player_id"].tolist()
    label_map = dict(zip(player_lookup["player_id"], player_lookup["label"]))

    p1, p2 = st.columns(2)
    player_a = p1.selectbox("Player A", options=options, format_func=lambda p: label_map[p], index=0, key="compare_player_a")
    player_b = p2.selectbox("Player B", options=options, format_func=lambda p: label_map[p], index=1 if len(options) > 1 else 0, key="compare_player_b")

    if player_a == player_b and len(options) > 1:
        st.info("Pick two different players for clearer comparison.")

    row_a = pool[pool["player_id"] == player_a].iloc[0]
    row_b = pool[pool["player_id"] == player_b].iloc[0]

    left, right = st.columns(2)
    left.metric("Player A Value/Snap", format_number(row_a["context_adjusted_per_snap"], 4))
    right.metric("Player B Value/Snap", format_number(row_b["context_adjusted_per_snap"], 4))

    metric_labels = {
        "context_adjusted_per_snap": "Adj Value/Snap",
        "success_over_expected_per_snap": "SOE/Snap",
        "opportunities_per_snap": "Opp/Snap",
        "value_percentile": "Percentile",
        "reliability_weight": "Reliability",
    }

    percentile_frame = pool.copy()
    for metric in COMPARE_METRICS:
        percentile_frame[f"{metric}_pct"] = percentile_frame[metric].rank(pct=True)

    radar_columns = [f"{metric}_pct" for metric in COMPARE_METRICS]
    radar_labels = [metric_labels[metric] for metric in COMPARE_METRICS]
    values_a = percentile_frame[percentile_frame["player_id"] == player_a][radar_columns].iloc[0].tolist()
    values_b = percentile_frame[percentile_frame["player_id"] == player_b][radar_columns].iloc[0].tolist()

    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=values_a,
            theta=radar_labels,
            fill="toself",
            name=label_map[player_a],
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=values_b,
            theta=radar_labels,
            fill="toself",
            name=label_map[player_b],
        )
    )
    radar.update_layout(
        title=f"{selected_position} Comparison Radar ({selected_season})",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=20, r=20, t=60, b=20),
        height=460,
    )
    st.plotly_chart(radar, use_container_width=True)

    compare_table = pd.DataFrame(
        {
            "metric": list(metric_labels.values()),
            label_map[player_a]: [row_a[m] for m in COMPARE_METRICS],
            label_map[player_b]: [row_b[m] for m in COMPARE_METRICS],
        }
    )
    compare_table.loc[compare_table["metric"] == "Percentile", [label_map[player_a], label_map[player_b]]] = (
        compare_table.loc[compare_table["metric"] == "Percentile", [label_map[player_a], label_map[player_b]]] * 100
    )
    st.dataframe(compare_table.round(4), use_container_width=True, hide_index=True)

    if selected_season == latest_season:
        st.caption("Comparing latest season production.")
    else:
        st.caption(f"Comparing historical production in {selected_season}.")


def render_validation_tab(validation_summary: pd.DataFrame) -> None:
    """Render validation diagnostics versus external benchmarks."""
    st.subheader("Validation Diagnostics")
    st.caption("How strongly valuation aligns with Pro Bowl selection and external performance proxies.")

    if validation_summary.empty:
        st.warning("Validation summary table is unavailable or empty.")
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
    c1.metric("Mean Precision@K", format_number(validation_summary["precision_at_k"].mean(), 3))
    c2.metric("Mean Spearman", format_number(validation_summary["spearman_with_benchmark"].mean(), 3))
    c3.metric("Mean Pro Bowl AUC", format_number(validation_summary["pro_bowl_auc"].mean(), 3))

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
        key="validation_metric",
    )
    heatmap_frame = validation_summary.pivot(
        index="position_group",
        columns="season",
        values=metric_choice,
    ).reindex(POSITION_ORDER)
    heatmap_fig = px.imshow(
        heatmap_frame,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=f"{metric_choice} by Position and Season",
    )
    heatmap_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=410)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    trend_fig = px.line(
        validation_summary,
        x="season",
        y="precision_at_k",
        color="position_group",
        markers=True,
        title="Precision@K Trend by Position",
        category_orders={"position_group": POSITION_ORDER},
    )
    trend_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=380)
    st.plotly_chart(trend_fig, use_container_width=True)

    display_summary = validation_summary.copy()
    for column in numeric_cols:
        display_summary[column] = display_summary[column].round(3)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)


def render_model_tab(training_runs: pd.DataFrame, metrics_json: dict | None) -> None:
    """Render model and feature-importance diagnostics."""
    st.subheader("Model Diagnostics")
    st.caption("Training split quality, holdout metrics, and top feature drivers from the latest model artifact.")

    if training_runs.empty:
        st.warning("No model training runs found.")
        return

    latest = training_runs.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation AUC", format_number(latest["validation_roc_auc"], 3))
    c2.metric("Test AUC", format_number(latest["test_roc_auc"], 3))
    c3.metric("Validation LogLoss", format_number(latest["validation_log_loss"], 3))
    c4.metric("Test LogLoss", format_number(latest["test_log_loss"], 3))

    run_display = training_runs.copy()
    run_display["run_ts"] = pd.to_datetime(run_display["run_ts"], errors="coerce")
    run_display = run_display.sort_values("run_ts", ascending=False)

    trend_ready = run_display.dropna(subset=["run_ts"]).sort_values("run_ts")
    if not trend_ready.empty:
        auc_fig = px.line(
            trend_ready,
            x="run_ts",
            y=["validation_roc_auc", "test_roc_auc"],
            markers=True,
            title="AUC by Training Run",
            labels={"value": "AUC", "variable": "Split", "run_ts": "Run Time"},
        )
        auc_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
        st.plotly_chart(auc_fig, use_container_width=True)

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
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=500)
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
          <h2 class="hero-title">NFL Player Valuation Front Office Console</h2>
          <p class="hero-sub">
            Context-adjusted player value for front-office decision support:
            market overview, rankings, trend analysis, model validation, and diagnostics.
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

        if st.button("Refresh Data Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        try:
            model_options = load_model_options(str(db_path))
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Failed to load model names: {exc}")
            st.stop()

        if not model_options:
            st.error("No model_name values found in player_valuation.")
            st.stop()

        selected_model = st.selectbox("Model Name", options=model_options, index=0)

        st.markdown("---")
        st.caption("Data Sources")
        st.caption("`player_valuation`")
        st.caption("`valuation_validation_summary` (optional)")
        st.caption("`model_training_runs` (optional)")

    valuation = load_valuation_data(str(db_path), selected_model)
    validation_summary = load_validation_summary(str(db_path), selected_model)
    training_runs = load_training_runs(str(db_path), selected_model)
    metrics_json = load_latest_metrics_json(selected_model)

    if valuation.empty:
        st.error(f"No valuation data found for model: {selected_model}")
        st.stop()

    tab_overview, tab_rankings, tab_player, tab_compare, tab_validation, tab_model = st.tabs(
        ["Overview", "Rankings", "Player Explorer", "Player Compare", "Validation", "Model"]
    )

    with tab_overview:
        render_overview_tab(valuation)

    with tab_rankings:
        render_rankings_tab(valuation)

    with tab_player:
        render_player_explorer_tab(valuation)

    with tab_compare:
        render_player_compare_tab(valuation)

    with tab_validation:
        render_validation_tab(validation_summary)

    with tab_model:
        render_model_tab(training_runs, metrics_json)


if __name__ == "__main__":
    main()
