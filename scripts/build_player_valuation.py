#!/usr/bin/env python3
"""Build per-player marginal valuation and position tiers.

Inputs:
- `pbp_features` for opportunity player context (position, scheme, usage proxies)
- `play_success_predictions` for model expected success and residuals

Outputs:
- `player_valuation`: player-season value metrics and context-adjusted scores
- `player_valuation_tiers`: tiered rankings by season and position (QB/WR/RB/TE)
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
TABLE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_POSITIONS: Final[list[str]] = ["QB", "RB", "WR", "TE"]

SHRINKAGE_PRIOR_SNAPS: Final[dict[str, int]] = {
    "QB": 260,
    "RB": 180,
    "WR": 210,
    "TE": 170,
}

MIN_SNAPS_BY_POSITION: Final[dict[str, int]] = {
    "QB": 180,
    "RB": 120,
    "WR": 140,
    "TE": 120,
}

MIN_OPPORTUNITIES_BY_POSITION: Final[dict[str, int]] = {
    "QB": 80,
    "RB": 50,
    "WR": 60,
    "TE": 45,
}


@dataclass(frozen=True)
class PositionSeasonModelInfo:
    """Track whether context model fit was possible for position-season groups."""

    season: int
    position_group: str
    fitted_linear_model: bool


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build player marginal value and position-specific valuation tiers."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/nfl_valuation.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default="pbp_features",
        help="Feature source table.",
    )
    parser.add_argument(
        "--predictions-table",
        type=str,
        default="play_success_predictions",
        help="Model predictions table.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xgb_play_success_epa",
        help="Model name in predictions table.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to value (default: 2020 2021 2022 2023 2024).",
    )
    parser.add_argument(
        "--valuation-table",
        type=str,
        default="player_valuation",
        help="Output table for player valuation rows.",
    )
    parser.add_argument(
        "--tiers-table",
        type=str,
        default="player_valuation_tiers",
        help="Output table for position tiers.",
    )
    parser.add_argument(
        "--replace-existing-season",
        action="store_true",
        help="Delete existing rows for selected seasons/model before writing.",
    )
    return parser.parse_args()


def validate_table_name(table_name: str) -> None:
    """Validate table names used in dynamic SQL."""
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table name: {table_name}")


def validate_seasons(seasons: Sequence[int]) -> list[int]:
    """Validate requested season list."""
    normalized = sorted(set(seasons))
    current_year = datetime.now(tz=timezone.utc).year
    for season in normalized:
        if season < 1999 or season > current_year:
            raise ValueError(f"Invalid season {season}. Use 1999-{current_year}.")
    return normalized


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Create SQLite connection."""
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if a SQLite table exists."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    ).fetchone()
    return row is not None


def load_play_level_rows(
    connection: sqlite3.Connection,
    feature_table: str,
    predictions_table: str,
    model_name: str,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """Load play-level rows for valuation from joined features + predictions."""
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT
            p.model_name,
            p.model_run_ts,
            p.split_name,
            p.season,
            p.game_id,
            p.play_id,
            p.target_success_epa,
            p.target_epa,
            p.pred_success_prob,
            p.residual_success,
            f.week,
            f.posteam,
            f.defteam,
            f.opportunity_player_id,
            f.opportunity_player_name,
            f.opportunity_player_position_group,
            f.feat_pass,
            f.feat_no_huddle,
            f.feat_shotgun,
            f.feat_player_prior_avg_usage_rate
        FROM {predictions_table} p
        INNER JOIN {feature_table} f
            ON p.season = f.season
           AND p.game_id = f.game_id
           AND p.play_id = f.play_id
        WHERE p.model_name = ?
          AND p.season IN ({placeholders})
          AND f.opportunity_player_id IS NOT NULL
          AND f.opportunity_player_position_group IN ('QB', 'RB', 'WR', 'TE')
        ORDER BY p.season, p.game_id, p.play_id;
    """
    params = [model_name] + list(seasons)
    frame = pd.read_sql_query(query, connection, params=params)
    if frame.empty:
        raise ValueError("No play-level rows loaded for valuation.")
    return frame


def safe_mode(series: pd.Series) -> str | None:
    """Return deterministic mode for a series."""
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    counts = cleaned.value_counts()
    return str(counts.index[0])


def aggregate_player_play_value(play_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate play-level expected vs actual outcomes at player-season level."""
    working = play_frame.copy()

    numeric_cols = [
        "target_success_epa",
        "target_epa",
        "pred_success_prob",
        "residual_success",
        "feat_pass",
        "feat_no_huddle",
        "feat_shotgun",
        "feat_player_prior_avg_usage_rate",
    ]
    for column in numeric_cols:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    grouped = working.groupby(
        ["model_name", "season", "opportunity_player_id", "opportunity_player_position_group"],
        as_index=False,
    )
    aggregated = grouped.agg(
        player_name=("opportunity_player_name", safe_mode),
        primary_team=("posteam", safe_mode),
        teams_played=("posteam", lambda s: "|".join(sorted(set(s.dropna())))),
        opportunities=("play_id", "count"),
        total_actual_success=("target_success_epa", "sum"),
        total_expected_success=("pred_success_prob", "sum"),
        total_success_over_expected=("residual_success", "sum"),
        avg_actual_success_rate=("target_success_epa", "mean"),
        avg_expected_success_prob=("pred_success_prob", "mean"),
        avg_target_epa=("target_epa", "mean"),
        avg_scheme_pass_rate=("feat_pass", "mean"),
        avg_scheme_no_huddle_rate=("feat_no_huddle", "mean"),
        avg_scheme_shotgun_rate=("feat_shotgun", "mean"),
        avg_player_prior_usage_rate=("feat_player_prior_avg_usage_rate", "mean"),
    )
    aggregated = aggregated.rename(
        columns={
            "opportunity_player_id": "player_id",
            "opportunity_player_position_group": "position_group",
        }
    )
    aggregated["success_over_expected_per_opp"] = (
        aggregated["total_success_over_expected"] / aggregated["opportunities"]
    )
    return aggregated


def build_snap_summary(seasons: Sequence[int]) -> pd.DataFrame:
    """Build season-level offensive snap summary by gsis_id.

    Football rationale:
    - Value per snap should use offensive snaps, not opportunities only.
    - We map pfr IDs to gsis IDs for stable joins with play-level player IDs.
    """
    snap = nfl.import_snap_counts(years=list(seasons))
    ids = nfl.import_ids()[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])

    snap = snap.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    snap = snap.dropna(subset=["gsis_id"])

    snap["offense_snaps"] = pd.to_numeric(snap["offense_snaps"], errors="coerce").fillna(0.0)
    snap["offense_pct"] = pd.to_numeric(snap["offense_pct"], errors="coerce").fillna(0.0)
    snap["position"] = snap["position"].fillna("OTHER").str.upper()
    snap["position_group"] = np.where(
        snap["position"].isin(["HB", "FB"]),
        "RB",
        np.where(snap["position"].isin(SUPPORTED_POSITIONS), snap["position"], "OTHER"),
    )
    snap = snap[snap["position_group"].isin(SUPPORTED_POSITIONS)].copy()

    summary = snap.groupby(["season", "gsis_id"], as_index=False).agg(
        offense_snaps=("offense_snaps", "sum"),
        avg_offense_pct=("offense_pct", "mean"),
    )
    summary = summary.rename(columns={"gsis_id": "player_id"})
    return summary


def merge_snap_context(player_frame: pd.DataFrame, snap_summary: pd.DataFrame) -> pd.DataFrame:
    """Merge snap summary and compute per-snap value metrics."""
    working = player_frame.merge(
        snap_summary,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    working["offense_snaps"] = pd.to_numeric(working["offense_snaps"], errors="coerce").fillna(0.0)
    working["offense_snaps_resolved"] = np.where(
        working["offense_snaps"] > 0,
        working["offense_snaps"],
        working["opportunities"],
    )
    working["snap_data_available"] = (working["offense_snaps"] > 0).astype("int8")
    working["opportunities_per_snap"] = (
        working["opportunities"] / working["offense_snaps_resolved"]
    )
    working["success_over_expected_per_snap"] = (
        working["total_success_over_expected"] / working["offense_snaps_resolved"]
    )
    return working


def apply_bayesian_shrinkage(player_frame: pd.DataFrame) -> pd.DataFrame:
    """Shrink per-snap value toward position-season prior means."""
    working = player_frame.copy()
    working["shrinkage_prior_snaps"] = working["position_group"].map(SHRINKAGE_PRIOR_SNAPS).fillna(200)

    def shrink_group(group: pd.DataFrame) -> pd.DataFrame:
        weighted_mean = np.average(
            group["success_over_expected_per_snap"],
            weights=np.clip(group["offense_snaps_resolved"], a_min=1.0, a_max=None),
        )
        reliability = group["offense_snaps_resolved"] / (
            group["offense_snaps_resolved"] + group["shrinkage_prior_snaps"]
        )
        group = group.copy()
        group["reliability_weight"] = reliability
        group["position_prior_per_snap"] = weighted_mean
        group["shrunk_success_over_expected_per_snap"] = (
            weighted_mean
            + reliability * (group["success_over_expected_per_snap"] - weighted_mean)
        )
        return group

    working = (
        working.groupby(["season", "position_group"], group_keys=False)
        .apply(shrink_group)
        .reset_index(drop=True)
    )
    return working


def apply_context_adjustment(player_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[PositionSeasonModelInfo]]:
    """Control for scheme and usage with position-season linear adjustment."""
    working = player_frame.copy()
    working["context_adjusted_per_snap"] = np.nan
    working["context_baseline_per_snap"] = np.nan

    model_info: list[PositionSeasonModelInfo] = []

    feature_cols = [
        "opportunities_per_snap",
        "avg_player_prior_usage_rate",
        "avg_scheme_pass_rate",
        "avg_scheme_no_huddle_rate",
        "avg_scheme_shotgun_rate",
        "avg_offense_pct",
        "reliability_weight",
    ]
    working["avg_offense_pct"] = pd.to_numeric(working["avg_offense_pct"], errors="coerce").fillna(0.0)

    for (season, position), group in working.groupby(["season", "position_group"]):
        idx = group.index
        group_matrix = group[feature_cols].copy()
        group_matrix = group_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = group["shrunk_success_over_expected_per_snap"].astype(float)

        if len(group) >= 8 and target.nunique() > 1:
            linear_model = LinearRegression()
            linear_model.fit(group_matrix, target)
            baseline = linear_model.predict(group_matrix)
            fitted = True
        else:
            baseline = np.full(shape=len(group), fill_value=float(target.mean()))
            fitted = False

        working.loc[idx, "context_baseline_per_snap"] = baseline
        working.loc[idx, "context_adjusted_per_snap"] = (
            target.to_numpy() - baseline
        )

        model_info.append(
            PositionSeasonModelInfo(
                season=int(season),
                position_group=str(position),
                fitted_linear_model=fitted,
            )
        )

    working["context_adjusted_total_value"] = (
        working["context_adjusted_per_snap"] * working["offense_snaps_resolved"]
    )
    return working, model_info


def compute_position_scores_and_tiers(player_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores, percentiles, and tier labels by position-season."""
    working = player_frame.copy()
    working["min_snaps_threshold"] = working["position_group"].map(MIN_SNAPS_BY_POSITION).fillna(120)
    working["min_opportunities_threshold"] = working["position_group"].map(
        MIN_OPPORTUNITIES_BY_POSITION
    ).fillna(50)
    working["is_qualified"] = (
        (working["offense_snaps_resolved"] >= working["min_snaps_threshold"])
        & (working["opportunities"] >= working["min_opportunities_threshold"])
    ).astype("int8")

    working["value_zscore"] = 0.0
    working["value_percentile"] = 0.5
    working["tier_label"] = "Tier 6 - Low Sample"
    working["tier_rank"] = 6

    for (_, _), group in working.groupby(["season", "position_group"]):
        idx = group.index
        values = group["context_adjusted_per_snap"].astype(float)
        mean_value = values.mean()
        std_value = values.std(ddof=0)
        z = np.zeros(len(group)) if std_value == 0 else (values - mean_value) / std_value
        percentiles = values.rank(method="average", pct=True)

        working.loc[idx, "value_zscore"] = z
        working.loc[idx, "value_percentile"] = percentiles

        qualified = group["is_qualified"] == 1
        tier_labels = np.full(len(group), "Tier 6 - Low Sample", dtype=object)
        tier_ranks = np.full(len(group), 6, dtype=int)

        # Tier cutoffs are percentile-based within position-season.
        # This keeps QB, RB, WR, and TE scales separate and comparable.
        q_percentiles = percentiles.to_numpy()
        tier_labels = np.where(
            qualified & (q_percentiles >= 0.90),
            "Tier 1 - Elite",
            tier_labels,
        )
        tier_ranks = np.where(qualified & (q_percentiles >= 0.90), 1, tier_ranks)

        tier_labels = np.where(
            qualified & (q_percentiles >= 0.70) & (q_percentiles < 0.90),
            "Tier 2 - High-End Starter",
            tier_labels,
        )
        tier_ranks = np.where(
            qualified & (q_percentiles >= 0.70) & (q_percentiles < 0.90),
            2,
            tier_ranks,
        )

        tier_labels = np.where(
            qualified & (q_percentiles >= 0.40) & (q_percentiles < 0.70),
            "Tier 3 - Solid Starter",
            tier_labels,
        )
        tier_ranks = np.where(
            qualified & (q_percentiles >= 0.40) & (q_percentiles < 0.70),
            3,
            tier_ranks,
        )

        tier_labels = np.where(
            qualified & (q_percentiles >= 0.20) & (q_percentiles < 0.40),
            "Tier 4 - Rotational",
            tier_labels,
        )
        tier_ranks = np.where(
            qualified & (q_percentiles >= 0.20) & (q_percentiles < 0.40),
            4,
            tier_ranks,
        )

        tier_labels = np.where(
            qualified & (q_percentiles < 0.20),
            "Tier 5 - Replacement",
            tier_labels,
        )
        tier_ranks = np.where(qualified & (q_percentiles < 0.20), 5, tier_ranks)

        working.loc[idx, "tier_label"] = tier_labels
        working.loc[idx, "tier_rank"] = tier_ranks

    return working


def create_output_tables(
    connection: sqlite3.Connection, valuation_table: str, tiers_table: str
) -> None:
    """Create output tables and indexes if missing."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {valuation_table} (
            model_name TEXT NOT NULL,
            valuation_run_ts TEXT NOT NULL,
            season INTEGER NOT NULL,
            position_group TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT,
            primary_team TEXT,
            teams_played TEXT,
            opportunities REAL NOT NULL,
            offense_snaps REAL NOT NULL,
            offense_snaps_resolved REAL NOT NULL,
            snap_data_available INTEGER NOT NULL,
            opportunities_per_snap REAL NOT NULL,
            avg_actual_success_rate REAL NOT NULL,
            avg_expected_success_prob REAL NOT NULL,
            avg_target_epa REAL,
            total_actual_success REAL NOT NULL,
            total_expected_success REAL NOT NULL,
            total_success_over_expected REAL NOT NULL,
            success_over_expected_per_opp REAL NOT NULL,
            success_over_expected_per_snap REAL NOT NULL,
            shrinkage_prior_snaps REAL NOT NULL,
            reliability_weight REAL NOT NULL,
            position_prior_per_snap REAL NOT NULL,
            shrunk_success_over_expected_per_snap REAL NOT NULL,
            avg_scheme_pass_rate REAL,
            avg_scheme_no_huddle_rate REAL,
            avg_scheme_shotgun_rate REAL,
            avg_player_prior_usage_rate REAL,
            avg_offense_pct REAL,
            context_baseline_per_snap REAL NOT NULL,
            context_adjusted_per_snap REAL NOT NULL,
            context_adjusted_total_value REAL NOT NULL,
            value_zscore REAL NOT NULL,
            value_percentile REAL NOT NULL,
            min_snaps_threshold REAL NOT NULL,
            min_opportunities_threshold REAL NOT NULL,
            is_qualified INTEGER NOT NULL,
            tier_label TEXT NOT NULL,
            tier_rank INTEGER NOT NULL
        );
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {tiers_table} (
            model_name TEXT NOT NULL,
            valuation_run_ts TEXT NOT NULL,
            season INTEGER NOT NULL,
            position_group TEXT NOT NULL,
            tier_rank INTEGER NOT NULL,
            tier_label TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT,
            primary_team TEXT,
            opportunities REAL NOT NULL,
            offense_snaps_resolved REAL NOT NULL,
            context_adjusted_per_snap REAL NOT NULL,
            context_adjusted_total_value REAL NOT NULL,
            value_zscore REAL NOT NULL,
            value_percentile REAL NOT NULL,
            is_qualified INTEGER NOT NULL
        );
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{valuation_table}_season_pos
        ON {valuation_table} (season, position_group);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{valuation_table}_player
        ON {valuation_table} (player_id, season);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{tiers_table}_season_pos_tier
        ON {tiers_table} (season, position_group, tier_rank);
        """
    )


def delete_existing_rows(
    connection: sqlite3.Connection,
    table_name: str,
    model_name: str,
    seasons: Sequence[int],
) -> int:
    """Delete existing rows for model and seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    cursor = connection.execute(
        f"""
        DELETE FROM {table_name}
        WHERE model_name = ?
          AND season IN ({placeholders});
        """,
        [model_name] + list(seasons),
    )
    return int(cursor.rowcount)


def count_existing_rows(
    connection: sqlite3.Connection,
    table_name: str,
    model_name: str,
    seasons: Sequence[int],
) -> int:
    """Count existing rows for model and seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        WHERE model_name = ?
          AND season IN ({placeholders});
        """,
        [model_name] + list(seasons),
    ).fetchone()
    return int(row[0]) if row else 0


def build_tiers_frame(valuation_frame: pd.DataFrame) -> pd.DataFrame:
    """Build tier table from valuation table."""
    columns = [
        "model_name",
        "valuation_run_ts",
        "season",
        "position_group",
        "tier_rank",
        "tier_label",
        "player_id",
        "player_name",
        "primary_team",
        "opportunities",
        "offense_snaps_resolved",
        "context_adjusted_per_snap",
        "context_adjusted_total_value",
        "value_zscore",
        "value_percentile",
        "is_qualified",
    ]
    tiers = valuation_frame[columns].copy()
    tiers = tiers.sort_values(
        ["season", "position_group", "tier_rank", "context_adjusted_per_snap"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return tiers


def main() -> None:
    """Build player valuation and position-tier outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {args.db_path}")

    validate_table_name(args.feature_table)
    validate_table_name(args.predictions_table)
    validate_table_name(args.valuation_table)
    validate_table_name(args.tiers_table)
    seasons = validate_seasons(args.seasons)

    connection = connect_sqlite(args.db_path)
    try:
        play_rows = load_play_level_rows(
            connection=connection,
            feature_table=args.feature_table,
            predictions_table=args.predictions_table,
            model_name=args.model_name,
            seasons=seasons,
        )
        LOGGER.info("Loaded %s play rows for valuation.", f"{len(play_rows):,}")

        player_values = aggregate_player_play_value(play_rows)
        snap_summary = build_snap_summary(seasons=seasons)
        player_values = merge_snap_context(player_values, snap_summary=snap_summary)
        player_values = apply_bayesian_shrinkage(player_values)
        player_values, model_info = apply_context_adjustment(player_values)
        player_values = compute_position_scores_and_tiers(player_values)

        valuation_run_ts = datetime.now(tz=timezone.utc).isoformat()
        player_values["model_name"] = args.model_name
        player_values["valuation_run_ts"] = valuation_run_ts

        valuation_columns = [
            "model_name",
            "valuation_run_ts",
            "season",
            "position_group",
            "player_id",
            "player_name",
            "primary_team",
            "teams_played",
            "opportunities",
            "offense_snaps",
            "offense_snaps_resolved",
            "snap_data_available",
            "opportunities_per_snap",
            "avg_actual_success_rate",
            "avg_expected_success_prob",
            "avg_target_epa",
            "total_actual_success",
            "total_expected_success",
            "total_success_over_expected",
            "success_over_expected_per_opp",
            "success_over_expected_per_snap",
            "shrinkage_prior_snaps",
            "reliability_weight",
            "position_prior_per_snap",
            "shrunk_success_over_expected_per_snap",
            "avg_scheme_pass_rate",
            "avg_scheme_no_huddle_rate",
            "avg_scheme_shotgun_rate",
            "avg_player_prior_usage_rate",
            "avg_offense_pct",
            "context_baseline_per_snap",
            "context_adjusted_per_snap",
            "context_adjusted_total_value",
            "value_zscore",
            "value_percentile",
            "min_snaps_threshold",
            "min_opportunities_threshold",
            "is_qualified",
            "tier_label",
            "tier_rank",
        ]
        valuation_frame = player_values[valuation_columns].copy()
        valuation_frame = valuation_frame.where(pd.notna(valuation_frame), None)
        tiers_frame = build_tiers_frame(valuation_frame)

        create_output_tables(
            connection=connection,
            valuation_table=args.valuation_table,
            tiers_table=args.tiers_table,
        )

        with connection:
            for table_name in [args.valuation_table, args.tiers_table]:
                existing = count_existing_rows(
                    connection=connection,
                    table_name=table_name,
                    model_name=args.model_name,
                    seasons=seasons,
                )
                if existing > 0:
                    if not args.replace_existing_season:
                        raise ValueError(
                            f"{table_name} already has {existing:,} rows for model "
                            f"{args.model_name} and seasons {seasons}. "
                            "Use --replace-existing-season to overwrite."
                        )
                    deleted = delete_existing_rows(
                        connection=connection,
                        table_name=table_name,
                        model_name=args.model_name,
                        seasons=seasons,
                    )
                    LOGGER.info("Deleted %s rows from %s.", f"{deleted:,}", table_name)

            valuation_frame.to_sql(
                name=args.valuation_table,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=1000,
            )
            tiers_frame.to_sql(
                name=args.tiers_table,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=1000,
            )

        LOGGER.info(
            "Player valuation rows written: %s to %s",
            f"{len(valuation_frame):,}",
            args.valuation_table,
        )
        LOGGER.info(
            "Player tier rows written: %s to %s",
            f"{len(tiers_frame):,}",
            args.tiers_table,
        )

        fitted_count = sum(info.fitted_linear_model for info in model_info)
        LOGGER.info(
            "Context adjustment groups fitted with linear model: %s/%s",
            fitted_count,
            len(model_info),
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
