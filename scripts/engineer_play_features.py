#!/usr/bin/env python3
"""Engineer play-level contextual features for player valuation modeling.

This script builds a modeling table (`pbp_features`) from `pbp_clean` with:
- down/distance and field-position context
- game script and clock leverage context
- offensive and defensive team-quality priors (no look-ahead leakage)
- player usage-rate priors
- player snap-count priors from nfl_data_py `import_snap_counts`
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence
from uuid import uuid4

import nfl_data_py as nfl
import numpy as np
import pandas as pd

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_SEASONS: Final[list[int]] = [2020, 2021, 2022, 2023, 2024]
FEATURE_TABLE_DEFAULT: Final[str] = "pbp_features"
FEATURE_RUN_TABLE: Final[str] = "feature_build_runs"

TABLE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

BASE_SELECT_COLUMNS: Final[list[str]] = [
    "season",
    "season_type",
    "week",
    "game_id",
    "play_id",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "quarter",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "quarter_seconds_remaining",
    "goal_to_go",
    "no_huddle",
    "shotgun",
    "pass_location",
    "run_location",
    "play_type",
    "pass",
    "rush",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "qb_scramble",
    "sack",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "epa",
    "wpa",
    "wp",
    "success",
    "drive",
    "passer_player_id",
    "passer_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "rusher_player_id",
    "rusher_player_name",
]

FLAG_COLUMNS: Final[list[str]] = [
    "goal_to_go",
    "no_huddle",
    "shotgun",
    "pass",
    "rush",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "qb_scramble",
    "sack",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "week",
    "quarter",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "quarter_seconds_remaining",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "epa",
    "wpa",
    "wp",
    "success",
    "drive",
]

TARGET_COLUMNS: Final[list[str]] = ["target_epa", "target_success", "target_wpa"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Engineer contextual play-level features into a SQLite feature table."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/nfl_valuation.db"),
        help="SQLite database path containing pbp_clean.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to engineer (default: 2020 2021 2022 2023 2024).",
    )
    parser.add_argument(
        "--feature-table",
        type=str,
        default=FEATURE_TABLE_DEFAULT,
        help="Destination SQLite table name.",
    )
    parser.add_argument(
        "--replace-existing-season",
        action="store_true",
        help="Delete existing rows for selected seasons before writing.",
    )
    return parser.parse_args()


def validate_table_name(table_name: str) -> None:
    """Guard against unsafe dynamic SQL table names."""
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(
            f"Invalid table name '{table_name}'. Use letters, numbers, underscores."
        )


def validate_seasons(seasons: Sequence[int]) -> list[int]:
    """Validate requested seasons and return a sorted unique list."""
    normalized = sorted(set(seasons))
    current_year = datetime.now(tz=timezone.utc).year
    for season in normalized:
        if season < 1999 or season > current_year:
            raise ValueError(f"Invalid season {season}. Use 1999-{current_year}.")
    return normalized


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Create SQLite connection with analytics-oriented pragmas."""
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in SQLite."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_feature_run_table(connection: sqlite3.Connection) -> None:
    """Create metadata table for feature engineering runs."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_RUN_TABLE} (
            run_id TEXT PRIMARY KEY,
            run_ts TEXT NOT NULL,
            seasons TEXT NOT NULL,
            feature_table TEXT NOT NULL,
            rows_written INTEGER NOT NULL,
            feature_count INTEGER NOT NULL,
            replace_existing_season INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT
        );
        """
    )


def load_base_plays(connection: sqlite3.Connection, seasons: Sequence[int]) -> pd.DataFrame:
    """Load modeling-eligible plays from pbp_clean."""
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
        SELECT
            {", ".join(BASE_SELECT_COLUMNS)}
        FROM pbp_clean
        WHERE is_model_play = 1
          AND season IN ({placeholders})
        ORDER BY season, game_id, play_id;
    """
    frame = pd.read_sql_query(query, connection, params=list(seasons))
    if frame.empty:
        raise ValueError("No model-eligible rows were loaded from pbp_clean.")
    return frame


def coerce_base_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply stable dtypes before feature engineering."""
    working = frame.copy()
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in FLAG_COLUMNS:
        working[column] = (
            pd.to_numeric(working[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype("int8")
        )

    working["pass_location"] = working["pass_location"].fillna("unknown").str.lower()
    working["run_location"] = working["run_location"].fillna("unknown").str.lower()
    return working


def add_base_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add down/distance/field/game-script features."""
    working = frame.copy()

    # Targets are stored with explicit names for training clarity.
    working["target_epa"] = working["epa"]
    working["target_success"] = working["success"]
    working["target_wpa"] = working["wpa"]

    # Down and distance.
    working["feat_down_1"] = (working["down"] == 1).astype("int8")
    working["feat_down_2"] = (working["down"] == 2).astype("int8")
    working["feat_down_3"] = (working["down"] == 3).astype("int8")
    working["feat_down_4"] = (working["down"] == 4).astype("int8")
    working["feat_ydstogo"] = working["ydstogo"].fillna(working["ydstogo"].median())
    working["feat_ydstogo_log"] = np.log1p(working["feat_ydstogo"].clip(lower=0))
    working["feat_short_yardage"] = (working["feat_ydstogo"] <= 2).astype("int8")
    working["feat_medium_yardage"] = (
        working["feat_ydstogo"].between(3, 6, inclusive="both")
    ).astype("int8")
    working["feat_long_yardage"] = (working["feat_ydstogo"] >= 7).astype("int8")

    # Field position.
    working["feat_goal_to_go"] = working["goal_to_go"].astype("int8")
    working["feat_yardline_100"] = working["yardline_100"].fillna(
        working["yardline_100"].median()
    )
    working["feat_field_pos_value"] = 100 - working["feat_yardline_100"]
    working["feat_red_zone"] = (working["feat_yardline_100"] <= 20).astype("int8")
    working["feat_inside_ten"] = (working["feat_yardline_100"] <= 10).astype("int8")
    working["feat_goal_line"] = (working["feat_yardline_100"] <= 5).astype("int8")
    working["feat_backed_up"] = (working["feat_yardline_100"] >= 80).astype("int8")
    working["feat_midfield"] = (
        working["feat_yardline_100"].between(40, 60, inclusive="both")
    ).astype("int8")

    # Game script and leverage.
    working["feat_score_diff"] = working["score_differential"].fillna(0.0)
    working["feat_score_diff_abs"] = working["feat_score_diff"].abs()
    working["feat_score_diff_sq"] = working["feat_score_diff"] ** 2
    working["feat_offense_leading"] = (working["feat_score_diff"] > 0).astype("int8")
    working["feat_offense_trailing"] = (working["feat_score_diff"] < 0).astype("int8")
    working["feat_score_tied"] = (working["feat_score_diff"] == 0).astype("int8")
    working["feat_posteam_score"] = working["posteam_score"].fillna(0.0)
    working["feat_defteam_score"] = working["defteam_score"].fillna(0.0)
    working["feat_wp"] = working["wp"].fillna(0.5)
    working["feat_wp_distance_from_neutral"] = (working["feat_wp"] - 0.5).abs()

    # Clock context.
    working["feat_game_seconds_remaining"] = working["game_seconds_remaining"].fillna(0.0)
    working["feat_half_seconds_remaining"] = working["half_seconds_remaining"].fillna(0.0)
    working["feat_quarter_seconds_remaining"] = working["quarter_seconds_remaining"].fillna(
        0.0
    )
    working["feat_game_seconds_pct"] = (
        working["feat_game_seconds_remaining"] / 3600.0
    ).clip(lower=0.0, upper=1.0)
    working["feat_half_seconds_pct"] = (
        working["feat_half_seconds_remaining"] / 1800.0
    ).clip(lower=0.0, upper=1.0)
    working["feat_quarter_seconds_pct"] = (
        working["feat_quarter_seconds_remaining"] / 900.0
    ).clip(lower=0.0, upper=1.0)
    working["feat_two_minute_drill"] = (
        working["feat_half_seconds_remaining"] <= 120
    ).astype("int8")
    working["feat_late_game"] = (working["quarter"] >= 4).astype("int8")
    working["feat_trailing_late"] = (
        (working["quarter"] >= 4) & (working["feat_score_diff"] < 0)
    ).astype("int8")
    working["feat_leading_late"] = (
        (working["quarter"] >= 4) & (working["feat_score_diff"] > 0)
    ).astype("int8")
    working["feat_garbage_time"] = (
        (working["quarter"] >= 4) & (working["feat_score_diff_abs"] >= 17)
    ).astype("int8")

    # Formation/tactical context.
    working["feat_no_huddle"] = working["no_huddle"].astype("int8")
    working["feat_shotgun"] = working["shotgun"].astype("int8")
    working["feat_pass"] = working["pass"].astype("int8")
    working["feat_rush"] = working["rush"].astype("int8")
    working["feat_qb_dropback"] = working["qb_dropback"].astype("int8")
    working["feat_qb_scramble"] = working["qb_scramble"].astype("int8")
    working["feat_sack"] = working["sack"].astype("int8")
    working["feat_early_down"] = (working["down"] <= 2).astype("int8")
    working["feat_money_down"] = (working["down"] >= 3).astype("int8")

    working["feat_air_yards"] = working["air_yards"].fillna(0.0)
    working["feat_yards_after_catch"] = working["yards_after_catch"].fillna(0.0)
    working["feat_deep_pass"] = (
        (working["feat_pass"] == 1) & (working["feat_air_yards"] >= 15)
    ).astype("int8")
    working["feat_screen_proxy"] = (
        (working["feat_pass"] == 1) & (working["feat_air_yards"] <= 0)
    ).astype("int8")
    working["feat_timeout_diff"] = (
        working["posteam_timeouts_remaining"].fillna(0.0)
        - working["defteam_timeouts_remaining"].fillna(0.0)
    )
    working["feat_posteam_timeouts_remaining"] = working[
        "posteam_timeouts_remaining"
    ].fillna(0.0)
    working["feat_defteam_timeouts_remaining"] = working[
        "defteam_timeouts_remaining"
    ].fillna(0.0)

    # Play-direction one-hot context.
    working["feat_pass_location_left"] = (
        working["pass_location"] == "left"
    ).astype("int8")
    working["feat_pass_location_middle"] = (
        working["pass_location"] == "middle"
    ).astype("int8")
    working["feat_pass_location_right"] = (
        working["pass_location"] == "right"
    ).astype("int8")
    working["feat_pass_location_unknown"] = (
        ~working["pass_location"].isin(["left", "middle", "right"])
    ).astype("int8")

    working["feat_run_location_left"] = (
        working["run_location"] == "left"
    ).astype("int8")
    working["feat_run_location_middle"] = (
        working["run_location"] == "middle"
    ).astype("int8")
    working["feat_run_location_right"] = (
        working["run_location"] == "right"
    ).astype("int8")
    working["feat_run_location_unknown"] = (
        ~working["run_location"].isin(["left", "middle", "right"])
    ).astype("int8")

    return working


def add_drive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add within-drive progression features."""
    working = frame.copy()
    sort_cols = ["season", "game_id", "drive", "posteam", "play_id"]
    working = working.sort_values(sort_cols).reset_index(drop=True)

    drive_group = working.groupby(["season", "game_id", "drive", "posteam"], sort=False)
    working["feat_drive_play_index"] = drive_group.cumcount() + 1
    working["feat_drive_total_plays"] = drive_group["play_id"].transform("size")
    working["feat_drive_play_pct"] = (
        working["feat_drive_play_index"] / working["feat_drive_total_plays"]
    )
    drive_start_yardline = drive_group["feat_yardline_100"].transform("first")
    drive_start_score_diff = drive_group["feat_score_diff"].transform("first")
    working["feat_drive_progress_yards"] = drive_start_yardline - working["feat_yardline_100"]
    working["feat_drive_score_swing"] = working["feat_score_diff"] - drive_start_score_diff
    return working


def add_shifted_history_features(
    frame: pd.DataFrame,
    group_cols: list[str],
    order_cols: list[str],
    value_col: str,
    expanding_col: str,
    recent_col: str,
    recent_window: int,
) -> pd.DataFrame:
    """Add shifted expanding and rolling means by group."""
    working = frame.sort_values(group_cols + order_cols).copy()
    grouped = working.groupby(group_cols, sort=False)[value_col]
    working[expanding_col] = grouped.transform(
        lambda series: series.shift(1).expanding().mean()
    )
    working[recent_col] = grouped.transform(
        lambda series: series.shift(1).rolling(recent_window, min_periods=1).mean()
    )
    return working


def build_team_priors(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build offensive and defensive no-look-ahead game-level prior metrics."""
    working = frame.copy()
    working["feat_explosive_play"] = (working["yards_gained"].fillna(0.0) >= 20).astype(
        "int8"
    )

    offense_games = (
        working.groupby(
            ["season", "week", "game_id", "game_date", "posteam"], as_index=False
        )
        .agg(
            off_game_epa_per_play=("target_epa", "mean"),
            off_game_success_rate=("target_success", "mean"),
            off_game_pass_rate=("feat_pass", "mean"),
            off_game_explosive_rate=("feat_explosive_play", "mean"),
        )
        .sort_values(["season", "posteam", "game_date", "week", "game_id"])
    )
    offense_games["feat_off_prior_games"] = offense_games.groupby(
        ["season", "posteam"], sort=False
    ).cumcount()

    offense_games = add_shifted_history_features(
        offense_games,
        group_cols=["season", "posteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="off_game_epa_per_play",
        expanding_col="feat_off_prior_epa_per_play",
        recent_col="feat_off_recent3_epa_per_play",
        recent_window=3,
    )
    offense_games = add_shifted_history_features(
        offense_games,
        group_cols=["season", "posteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="off_game_success_rate",
        expanding_col="feat_off_prior_success_rate",
        recent_col="feat_off_recent3_success_rate",
        recent_window=3,
    )
    offense_games = add_shifted_history_features(
        offense_games,
        group_cols=["season", "posteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="off_game_pass_rate",
        expanding_col="feat_off_prior_pass_rate",
        recent_col="feat_off_recent3_pass_rate",
        recent_window=3,
    )
    offense_games = add_shifted_history_features(
        offense_games,
        group_cols=["season", "posteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="off_game_explosive_rate",
        expanding_col="feat_off_prior_explosive_rate",
        recent_col="feat_off_recent3_explosive_rate",
        recent_window=3,
    )

    defense_games = (
        working.groupby(
            ["season", "week", "game_id", "game_date", "defteam"], as_index=False
        )
        .agg(
            def_game_epa_allowed=("target_epa", "mean"),
            def_game_success_allowed=("target_success", "mean"),
            def_game_yards_allowed=("yards_gained", "mean"),
            def_game_explosive_allowed=("feat_explosive_play", "mean"),
        )
        .sort_values(["season", "defteam", "game_date", "week", "game_id"])
    )
    defense_games["feat_opp_def_prior_games"] = defense_games.groupby(
        ["season", "defteam"], sort=False
    ).cumcount()

    defense_games = add_shifted_history_features(
        defense_games,
        group_cols=["season", "defteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="def_game_epa_allowed",
        expanding_col="feat_opp_def_prior_epa_allowed",
        recent_col="feat_opp_def_recent3_epa_allowed",
        recent_window=3,
    )
    defense_games = add_shifted_history_features(
        defense_games,
        group_cols=["season", "defteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="def_game_success_allowed",
        expanding_col="feat_opp_def_prior_success_allowed",
        recent_col="feat_opp_def_recent3_success_allowed",
        recent_window=3,
    )
    defense_games = add_shifted_history_features(
        defense_games,
        group_cols=["season", "defteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="def_game_yards_allowed",
        expanding_col="feat_opp_def_prior_yards_allowed",
        recent_col="feat_opp_def_recent3_yards_allowed",
        recent_window=3,
    )
    defense_games = add_shifted_history_features(
        defense_games,
        group_cols=["season", "defteam"],
        order_cols=["game_date", "week", "game_id"],
        value_col="def_game_explosive_allowed",
        expanding_col="feat_opp_def_prior_explosive_allowed",
        recent_col="feat_opp_def_recent3_explosive_allowed",
        recent_window=3,
    )

    return offense_games, defense_games


def merge_team_priors(
    frame: pd.DataFrame, offense_games: pd.DataFrame, defense_games: pd.DataFrame
) -> pd.DataFrame:
    """Merge team priors onto each play and apply fallback policy."""
    working = frame.copy()

    offense_keep_cols = [
        "season",
        "game_id",
        "posteam",
        "feat_off_prior_games",
        "feat_off_prior_epa_per_play",
        "feat_off_recent3_epa_per_play",
        "feat_off_prior_success_rate",
        "feat_off_recent3_success_rate",
        "feat_off_prior_pass_rate",
        "feat_off_recent3_pass_rate",
        "feat_off_prior_explosive_rate",
        "feat_off_recent3_explosive_rate",
    ]
    defense_keep_cols = [
        "season",
        "game_id",
        "defteam",
        "feat_opp_def_prior_games",
        "feat_opp_def_prior_epa_allowed",
        "feat_opp_def_recent3_epa_allowed",
        "feat_opp_def_prior_success_allowed",
        "feat_opp_def_recent3_success_allowed",
        "feat_opp_def_prior_yards_allowed",
        "feat_opp_def_recent3_yards_allowed",
        "feat_opp_def_prior_explosive_allowed",
        "feat_opp_def_recent3_explosive_allowed",
    ]

    working = working.merge(
        offense_games[offense_keep_cols],
        on=["season", "game_id", "posteam"],
        how="left",
        validate="many_to_one",
    )
    working = working.merge(
        defense_games[defense_keep_cols],
        on=["season", "game_id", "defteam"],
        how="left",
        validate="many_to_one",
    )

    working["feat_off_no_prior"] = (
        working["feat_off_prior_games"].fillna(0).eq(0)
    ).astype("int8")
    working["feat_opp_def_no_prior"] = (
        working["feat_opp_def_prior_games"].fillna(0).eq(0)
    ).astype("int8")

    # For first games with no priors, use league-level baselines and keep flags.
    fallback_map = {
        "feat_off_prior_epa_per_play": working["target_epa"].mean(),
        "feat_off_recent3_epa_per_play": working["target_epa"].mean(),
        "feat_off_prior_success_rate": working["target_success"].mean(),
        "feat_off_recent3_success_rate": working["target_success"].mean(),
        "feat_off_prior_pass_rate": working["feat_pass"].mean(),
        "feat_off_recent3_pass_rate": working["feat_pass"].mean(),
        "feat_off_prior_explosive_rate": (working["yards_gained"] >= 20).mean(),
        "feat_off_recent3_explosive_rate": (working["yards_gained"] >= 20).mean(),
        "feat_opp_def_prior_epa_allowed": working["target_epa"].mean(),
        "feat_opp_def_recent3_epa_allowed": working["target_epa"].mean(),
        "feat_opp_def_prior_success_allowed": working["target_success"].mean(),
        "feat_opp_def_recent3_success_allowed": working["target_success"].mean(),
        "feat_opp_def_prior_yards_allowed": working["yards_gained"].mean(),
        "feat_opp_def_recent3_yards_allowed": working["yards_gained"].mean(),
        "feat_opp_def_prior_explosive_allowed": (working["yards_gained"] >= 20).mean(),
        "feat_opp_def_recent3_explosive_allowed": (working["yards_gained"] >= 20).mean(),
    }
    for column, fallback in fallback_map.items():
        working[column] = working[column].fillna(float(fallback))

    working["feat_off_prior_games"] = working["feat_off_prior_games"].fillna(0)
    working["feat_opp_def_prior_games"] = working["feat_opp_def_prior_games"].fillna(0)
    return working


def assign_opportunity_player(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign the primary opportunity player for usage/snap context.

    Football rationale:
    - Rush plays are attributed to rushers.
    - Pass plays are attributed to targets when available.
    - If target is missing on a pass, fallback to passer.
    """
    working = frame.copy()
    rush_mask = (working["feat_rush"] == 1) & working["rusher_player_id"].notna()
    target_mask = (working["feat_pass"] == 1) & working["receiver_player_id"].notna()
    passer_fallback_mask = (working["feat_pass"] == 1) & working["passer_player_id"].notna()

    working["opportunity_player_id"] = np.select(
        [rush_mask, target_mask, passer_fallback_mask],
        [
            working["rusher_player_id"],
            working["receiver_player_id"],
            working["passer_player_id"],
        ],
        default=None,
    )
    working["opportunity_player_name"] = np.select(
        [rush_mask, target_mask, passer_fallback_mask],
        [
            working["rusher_player_name"],
            working["receiver_player_name"],
            working["passer_player_name"],
        ],
        default=None,
    )
    working["opportunity_role"] = np.select(
        [rush_mask, target_mask, passer_fallback_mask],
        ["RUSHER", "TARGET", "PASSER"],
        default="UNKNOWN",
    )
    return working


def build_position_map() -> dict[str, str]:
    """Build gsis_id -> normalized position-group map from nflverse IDs."""
    ids = nfl.import_ids()[["gsis_id", "position"]].dropna(subset=["gsis_id"])
    raw_position = ids["position"].fillna("OTHER").str.upper()
    position_group = np.where(
        raw_position.isin(["HB", "FB"]),
        "RB",
        np.where(raw_position.isin(["QB", "RB", "WR", "TE"]), raw_position, "OTHER"),
    )
    ids = ids.assign(position_group=position_group)
    ids = ids.drop_duplicates(subset=["gsis_id"], keep="first")
    return dict(zip(ids["gsis_id"], ids["position_group"]))


def add_player_position_context(frame: pd.DataFrame, position_map: dict[str, str]) -> pd.DataFrame:
    """Attach normalized position group for the opportunity player."""
    working = frame.copy()
    working["opportunity_player_position_group"] = (
        working["opportunity_player_id"].map(position_map).fillna("OTHER")
    )
    working["feat_is_qb_opportunity"] = (
        working["opportunity_player_position_group"] == "QB"
    ).astype("int8")
    working["feat_is_rb_opportunity"] = (
        working["opportunity_player_position_group"] == "RB"
    ).astype("int8")
    working["feat_is_wr_opportunity"] = (
        working["opportunity_player_position_group"] == "WR"
    ).astype("int8")
    working["feat_is_te_opportunity"] = (
        working["opportunity_player_position_group"] == "TE"
    ).astype("int8")
    return working


def build_player_usage_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Build weekly player usage priors from play opportunities."""
    usage_base = frame[frame["opportunity_player_id"].notna()].copy()
    usage_week = (
        usage_base.groupby(
            ["season", "week", "posteam", "opportunity_player_id"], as_index=False
        )
        .agg(player_week_opportunities=("play_id", "count"))
        .sort_values(["season", "opportunity_player_id", "week"])
    )
    team_week = (
        frame.groupby(["season", "week", "posteam"], as_index=False)
        .agg(team_week_model_snaps=("play_id", "count"))
        .sort_values(["season", "posteam", "week"])
    )
    usage_week = usage_week.merge(
        team_week,
        on=["season", "week", "posteam"],
        how="left",
        validate="many_to_one",
    )
    usage_week["player_week_usage_rate"] = (
        usage_week["player_week_opportunities"] / usage_week["team_week_model_snaps"]
    )
    usage_week["feat_player_prior_games"] = usage_week.groupby(
        ["season", "opportunity_player_id"], sort=False
    ).cumcount()

    usage_week = add_shifted_history_features(
        usage_week,
        group_cols=["season", "opportunity_player_id"],
        order_cols=["week"],
        value_col="player_week_opportunities",
        expanding_col="feat_player_prior_avg_weekly_opportunities",
        recent_col="feat_player_prior_4wk_weekly_opportunities",
        recent_window=4,
    )
    usage_week = add_shifted_history_features(
        usage_week,
        group_cols=["season", "opportunity_player_id"],
        order_cols=["week"],
        value_col="player_week_usage_rate",
        expanding_col="feat_player_prior_avg_usage_rate",
        recent_col="feat_player_prior_4wk_usage_rate",
        recent_window=4,
    )
    return usage_week


def build_snap_priors(seasons: Sequence[int]) -> pd.DataFrame:
    """Build weekly prior snap features from nflverse snap counts."""
    snap = nfl.import_snap_counts(years=list(seasons))
    ids = nfl.import_ids()[["gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])

    snap = snap.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    snap = snap.rename(columns={"team": "posteam", "gsis_id": "snap_player_id"})
    snap = snap.dropna(subset=["snap_player_id", "posteam", "season", "week"])
    snap["offense_snaps"] = pd.to_numeric(snap["offense_snaps"], errors="coerce").fillna(0.0)
    snap["offense_pct"] = pd.to_numeric(snap["offense_pct"], errors="coerce")

    snap_week = (
        snap.groupby(["season", "week", "posteam", "snap_player_id"], as_index=False)
        .agg(
            offense_snaps=("offense_snaps", "max"),
            offense_pct=("offense_pct", "max"),
        )
        .sort_values(["season", "snap_player_id", "week"])
    )

    team_week_snaps = (
        snap_week.groupby(["season", "week", "posteam"], as_index=False)
        .agg(team_week_offense_snaps=("offense_snaps", "max"))
    )
    snap_week = snap_week.merge(
        team_week_snaps,
        on=["season", "week", "posteam"],
        how="left",
        validate="many_to_one",
    )
    fallback_offense_pct = np.where(
        snap_week["team_week_offense_snaps"] > 0,
        snap_week["offense_snaps"] / snap_week["team_week_offense_snaps"],
        0.0,
    )
    snap_week["offense_pct"] = snap_week["offense_pct"].where(
        snap_week["offense_pct"].notna(),
        fallback_offense_pct,
    )
    snap_week["snap_prior_games"] = snap_week.groupby(
        ["season", "snap_player_id"], sort=False
    ).cumcount()

    snap_week = add_shifted_history_features(
        snap_week,
        group_cols=["season", "snap_player_id"],
        order_cols=["week"],
        value_col="offense_snaps",
        expanding_col="snap_prior_avg_offense_snaps",
        recent_col="snap_prior_4wk_offense_snaps",
        recent_window=4,
    )
    snap_week = add_shifted_history_features(
        snap_week,
        group_cols=["season", "snap_player_id"],
        order_cols=["week"],
        value_col="offense_pct",
        expanding_col="snap_prior_avg_offense_pct",
        recent_col="snap_prior_4wk_offense_pct",
        recent_window=4,
    )

    keep_cols = [
        "season",
        "week",
        "posteam",
        "snap_player_id",
        "snap_prior_games",
        "snap_prior_avg_offense_snaps",
        "snap_prior_4wk_offense_snaps",
        "snap_prior_avg_offense_pct",
        "snap_prior_4wk_offense_pct",
    ]
    return snap_week[keep_cols].copy()


def merge_usage_and_snap_context(
    frame: pd.DataFrame, usage_priors: pd.DataFrame, snap_priors: pd.DataFrame
) -> pd.DataFrame:
    """Merge usage and snap priors for opportunity players and passers."""
    working = frame.copy()

    usage_cols = [
        "season",
        "week",
        "posteam",
        "opportunity_player_id",
        "feat_player_prior_games",
        "feat_player_prior_avg_weekly_opportunities",
        "feat_player_prior_4wk_weekly_opportunities",
        "feat_player_prior_avg_usage_rate",
        "feat_player_prior_4wk_usage_rate",
    ]
    working = working.merge(
        usage_priors[usage_cols],
        on=["season", "week", "posteam", "opportunity_player_id"],
        how="left",
        validate="many_to_one",
    )
    working["feat_player_no_usage_prior"] = (
        working["feat_player_prior_games"].fillna(0).eq(0)
    ).astype("int8")
    for column in [
        "feat_player_prior_games",
        "feat_player_prior_avg_weekly_opportunities",
        "feat_player_prior_4wk_weekly_opportunities",
        "feat_player_prior_avg_usage_rate",
        "feat_player_prior_4wk_usage_rate",
    ]:
        working[column] = working[column].fillna(0.0)

    # Opportunity-player snap priors.
    opp_snap = snap_priors.rename(
        columns={
            "snap_player_id": "opportunity_player_id",
            "snap_prior_games": "feat_player_snap_prior_games",
            "snap_prior_avg_offense_snaps": "feat_player_snap_prior_avg_offense_snaps",
            "snap_prior_4wk_offense_snaps": "feat_player_snap_prior_4wk_offense_snaps",
            "snap_prior_avg_offense_pct": "feat_player_snap_prior_avg_offense_pct",
            "snap_prior_4wk_offense_pct": "feat_player_snap_prior_4wk_offense_pct",
        }
    )
    working = working.merge(
        opp_snap,
        on=["season", "week", "posteam", "opportunity_player_id"],
        how="left",
        validate="many_to_one",
    )
    working["feat_player_no_snap_prior"] = (
        working["feat_player_snap_prior_games"].fillna(0).eq(0)
    ).astype("int8")
    for column in [
        "feat_player_snap_prior_games",
        "feat_player_snap_prior_avg_offense_snaps",
        "feat_player_snap_prior_4wk_offense_snaps",
        "feat_player_snap_prior_avg_offense_pct",
        "feat_player_snap_prior_4wk_offense_pct",
    ]:
        working[column] = working[column].fillna(0.0)

    # QB snap priors merged separately using passer_player_id.
    qb_snap = snap_priors.rename(
        columns={
            "snap_player_id": "passer_player_id",
            "snap_prior_games": "feat_qb_snap_prior_games",
            "snap_prior_avg_offense_snaps": "feat_qb_snap_prior_avg_offense_snaps",
            "snap_prior_4wk_offense_snaps": "feat_qb_snap_prior_4wk_offense_snaps",
            "snap_prior_avg_offense_pct": "feat_qb_snap_prior_avg_offense_pct",
            "snap_prior_4wk_offense_pct": "feat_qb_snap_prior_4wk_offense_pct",
        }
    )
    working = working.merge(
        qb_snap,
        on=["season", "week", "posteam", "passer_player_id"],
        how="left",
        validate="many_to_one",
    )
    working["feat_qb_no_snap_prior"] = (
        working["feat_qb_snap_prior_games"].fillna(0).eq(0)
    ).astype("int8")
    for column in [
        "feat_qb_snap_prior_games",
        "feat_qb_snap_prior_avg_offense_snaps",
        "feat_qb_snap_prior_4wk_offense_snaps",
        "feat_qb_snap_prior_avg_offense_pct",
        "feat_qb_snap_prior_4wk_offense_pct",
    ]:
        working[column] = working[column].fillna(0.0)

    return working


def validate_feature_keys(frame: pd.DataFrame) -> None:
    """Ensure per-play key uniqueness for the final feature table."""
    key_cols = ["season", "game_id", "play_id"]
    if frame[key_cols].isna().any(axis=1).any():
        raise ValueError("Found null values in feature primary-key columns.")
    duplicates = frame.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate feature keys.")


def build_feature_frame(base_plays: pd.DataFrame, seasons: Sequence[int]) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    working = coerce_base_types(base_plays)
    working = add_base_context_features(working)
    working = add_drive_features(working)

    offense_games, defense_games = build_team_priors(working)
    working = merge_team_priors(working, offense_games=offense_games, defense_games=defense_games)

    working = assign_opportunity_player(working)
    position_map = build_position_map()
    working = add_player_position_context(working, position_map=position_map)

    usage_priors = build_player_usage_priors(working)
    snap_priors = build_snap_priors(seasons=seasons)
    working = merge_usage_and_snap_context(
        working,
        usage_priors=usage_priors,
        snap_priors=snap_priors,
    )

    feature_columns = sorted([column for column in working.columns if column.startswith("feat_")])
    if len(feature_columns) < 40:
        raise ValueError(
            f"Feature count below requirement. Found {len(feature_columns)} features."
        )
    LOGGER.info("Engineered %s contextual features.", len(feature_columns))

    final_columns = [
        "season",
        "season_type",
        "week",
        "game_date",
        "game_id",
        "play_id",
        "home_team",
        "away_team",
        "posteam",
        "defteam",
        "play_type",
        "passer_player_id",
        "passer_player_name",
        "receiver_player_id",
        "receiver_player_name",
        "rusher_player_id",
        "rusher_player_name",
        "opportunity_player_id",
        "opportunity_player_name",
        "opportunity_role",
        "opportunity_player_position_group",
    ] + TARGET_COLUMNS + feature_columns

    final_frame = working[final_columns].copy()
    final_frame["game_date"] = final_frame["game_date"].dt.strftime("%Y-%m-%d")
    final_frame = final_frame.where(pd.notna(final_frame), None)
    validate_feature_keys(final_frame)
    return final_frame


def count_rows_for_seasons(
    connection: sqlite3.Connection, table_name: str, seasons: Sequence[int]
) -> int:
    """Count rows in a table for selected seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    row = connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE season IN ({placeholders});",
        list(seasons),
    ).fetchone()
    return int(row[0]) if row else 0


def delete_rows_for_seasons(
    connection: sqlite3.Connection, table_name: str, seasons: Sequence[int]
) -> int:
    """Delete rows in table for selected seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    cursor = connection.execute(
        f"DELETE FROM {table_name} WHERE season IN ({placeholders});",
        list(seasons),
    )
    return int(cursor.rowcount)


def create_feature_indexes(connection: sqlite3.Connection, feature_table: str) -> None:
    """Create indexes for fast model-building and player queries."""
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{feature_table}_pk
        ON {feature_table} (season, game_id, play_id);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{feature_table}_season_week
        ON {feature_table} (season, week);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{feature_table}_team
        ON {feature_table} (posteam, defteam);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{feature_table}_player
        ON {feature_table} (opportunity_player_id, opportunity_player_position_group);
        """
    )


def insert_feature_run(
    connection: sqlite3.Connection,
    seasons: Sequence[int],
    feature_table: str,
    rows_written: int,
    feature_count: int,
    replace_existing_season: bool,
    status: str,
    message: str,
) -> None:
    """Insert metadata about one feature build run."""
    connection.execute(
        f"""
        INSERT INTO {FEATURE_RUN_TABLE} (
            run_id,
            run_ts,
            seasons,
            feature_table,
            rows_written,
            feature_count,
            replace_existing_season,
            status,
            message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            str(uuid4()),
            datetime.now(tz=timezone.utc).isoformat(),
            ",".join(str(season) for season in seasons),
            feature_table,
            rows_written,
            feature_count,
            int(replace_existing_season),
            status,
            message,
        ),
    )


def write_feature_table(
    connection: sqlite3.Connection,
    feature_table: str,
    seasons: Sequence[int],
    feature_frame: pd.DataFrame,
    replace_existing_season: bool,
) -> tuple[int, int]:
    """Write features to SQLite with replace-safety and indexing."""
    preexisting_rows = 0
    if table_exists(connection, feature_table):
        preexisting_rows = count_rows_for_seasons(connection, feature_table, seasons)
        if preexisting_rows > 0 and not replace_existing_season:
            raise ValueError(
                f"{feature_table} already has {preexisting_rows:,} rows for seasons "
                f"{list(seasons)}. Use --replace-existing-season to overwrite."
            )

    with connection:
        if table_exists(connection, feature_table) and replace_existing_season:
            deleted_rows = delete_rows_for_seasons(connection, feature_table, seasons)
            if deleted_rows > 0:
                LOGGER.info("Deleted %s existing rows from %s.", f"{deleted_rows:,}", feature_table)

        feature_frame.to_sql(
            name=feature_table,
            con=connection,
            if_exists="append",
            index=False,
            chunksize=1_000,
        )
        create_feature_indexes(connection, feature_table)

    rows_written = len(feature_frame)
    feature_count = len([column for column in feature_frame.columns if column.startswith("feat_")])
    return rows_written, feature_count


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    seasons = validate_seasons(args.seasons)
    validate_table_name(args.feature_table)

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    LOGGER.info("Building features for seasons: %s", seasons)
    LOGGER.info("Destination table: %s", args.feature_table)

    connection = connect_sqlite(args.db_path)
    try:
        ensure_feature_run_table(connection)
        base_plays = load_base_plays(connection, seasons)
        LOGGER.info("Loaded %s model plays from pbp_clean.", f"{len(base_plays):,}")

        feature_frame = build_feature_frame(base_plays=base_plays, seasons=seasons)
        rows_written, feature_count = write_feature_table(
            connection=connection,
            feature_table=args.feature_table,
            seasons=seasons,
            feature_frame=feature_frame,
            replace_existing_season=args.replace_existing_season,
        )
        with connection:
            insert_feature_run(
                connection=connection,
                seasons=seasons,
                feature_table=args.feature_table,
                rows_written=rows_written,
                feature_count=feature_count,
                replace_existing_season=args.replace_existing_season,
                status="SUCCESS",
                message="Feature engineering completed successfully.",
            )

        LOGGER.info("Feature engineering complete.")
        LOGGER.info("Rows written: %s", f"{rows_written:,}")
        LOGGER.info("Feature count: %s", feature_count)
        LOGGER.info("SQLite table: %s", args.feature_table)
    except Exception as exc:
        with connection:
            insert_feature_run(
                connection=connection,
                seasons=seasons,
                feature_table=args.feature_table,
                rows_written=0,
                feature_count=0,
                replace_existing_season=args.replace_existing_season,
                status="FAILED",
                message=str(exc),
            )
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
