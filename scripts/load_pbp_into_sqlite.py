#!/usr/bin/env python3
"""Load cleaned NFL play-by-play data from parquet into a local SQLite database.

Step 2 goals:
1) Persist a modeling-ready PBP table with stable football context fields.
2) Record ingestion metadata so every load is auditable and reproducible.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence
from uuid import uuid4

import pandas as pd

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

MIN_SUPPORTED_SEASON: Final[int] = 1999
SPECIAL_TEAMS_PLAY_TYPES: Final[set[str]] = {
    "kickoff",
    "punt",
    "field_goal",
    "extra_point",
}
NON_ACTION_PLAY_TYPES: Final[set[str]] = {
    "no_play",
    "qb_kneel",
    "qb_spike",
}

RAW_COLUMNS: Final[list[str]] = [
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
    "side_of_field",
    "qtr",
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
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "fumble",
    "fumble_lost",
    "penalty",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "touchdown",
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
    "cpoe",
    "success",
    "drive",
    "passer_player_id",
    "passer_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "desc",
]

SOURCE_REQUIRED_COLUMNS: Final[set[str]] = {
    "season",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "play_type",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "epa",
    "success",
}

COLUMN_RENAMES: Final[dict[str, str]] = {
    "qtr": "quarter",
    "desc": "play_description",
}

OUTPUT_COLUMNS: Final[list[str]] = [
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
    "side_of_field",
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
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "fumble",
    "fumble_lost",
    "penalty",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "touchdown",
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
    "cpoe",
    "success",
    "drive",
    "passer_player_id",
    "passer_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "play_description",
    "is_special_teams_play",
    "is_non_action_play",
    "is_scrimmage_play",
    "is_model_play",
    "ingestion_ts",
]

INT_COLUMNS: Final[list[str]] = [
    "season",
    "week",
    "play_id",
    "quarter",
    "down",
    "drive",
]

FLOAT_COLUMNS: Final[list[str]] = [
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
    "cpoe",
    "success",
]

INPUT_FLAG_COLUMNS: Final[list[str]] = [
    "goal_to_go",
    "no_huddle",
    "shotgun",
    "pass",
    "rush",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "fumble",
    "fumble_lost",
    "penalty",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "touchdown",
]


@dataclass(frozen=True)
class SeasonLoadResult:
    """Summary of one season load into SQLite."""

    season: int
    raw_rows: int
    inserted_rows: int
    model_play_rows: int
    replaced_existing: bool
    db_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load parquet PBP seasons into SQLite for valuation modeling."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season(s) to load, e.g. --seasons 2024 or --seasons 2020 2021 2022.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files named pbp_<season>.parquet.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/nfl_valuation.db"),
        help="SQLite database output path.",
    )
    parser.add_argument(
        "--replace-existing-season",
        action="store_true",
        help="Delete existing rows for each season before reloading.",
    )
    return parser.parse_args()


def validate_seasons(seasons: Sequence[int]) -> list[int]:
    """Validate and normalize requested seasons."""
    normalized = sorted(set(seasons))
    current_year = datetime.now(tz=timezone.utc).year
    for season in normalized:
        if season < MIN_SUPPORTED_SEASON or season > current_year:
            raise ValueError(
                f"Season {season} is invalid. Choose {MIN_SUPPORTED_SEASON}-{current_year}."
            )
    return normalized


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection with write settings appropriate for analytics."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """Create database tables and indexes if missing."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS pbp_clean (
            season INTEGER NOT NULL,
            season_type TEXT,
            week INTEGER,
            game_id TEXT NOT NULL,
            play_id INTEGER NOT NULL,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            posteam TEXT,
            defteam TEXT,
            side_of_field TEXT,
            quarter INTEGER,
            down INTEGER,
            ydstogo REAL,
            yardline_100 REAL,
            game_seconds_remaining REAL,
            half_seconds_remaining REAL,
            quarter_seconds_remaining REAL,
            goal_to_go INTEGER,
            no_huddle INTEGER,
            shotgun INTEGER,
            pass_location TEXT,
            run_location TEXT,
            play_type TEXT,
            pass INTEGER,
            rush INTEGER,
            pass_attempt INTEGER,
            rush_attempt INTEGER,
            qb_dropback INTEGER,
            qb_kneel INTEGER,
            qb_spike INTEGER,
            qb_scramble INTEGER,
            sack INTEGER,
            complete_pass INTEGER,
            incomplete_pass INTEGER,
            interception INTEGER,
            fumble INTEGER,
            fumble_lost INTEGER,
            penalty INTEGER,
            first_down INTEGER,
            third_down_converted INTEGER,
            third_down_failed INTEGER,
            fourth_down_converted INTEGER,
            fourth_down_failed INTEGER,
            touchdown INTEGER,
            posteam_score REAL,
            defteam_score REAL,
            score_differential REAL,
            posteam_timeouts_remaining REAL,
            defteam_timeouts_remaining REAL,
            air_yards REAL,
            yards_after_catch REAL,
            yards_gained REAL,
            epa REAL,
            wpa REAL,
            wp REAL,
            cpoe REAL,
            success REAL,
            drive INTEGER,
            passer_player_id TEXT,
            passer_player_name TEXT,
            receiver_player_id TEXT,
            receiver_player_name TEXT,
            rusher_player_id TEXT,
            rusher_player_name TEXT,
            play_description TEXT,
            is_special_teams_play INTEGER NOT NULL,
            is_non_action_play INTEGER NOT NULL,
            is_scrimmage_play INTEGER NOT NULL,
            is_model_play INTEGER NOT NULL,
            ingestion_ts TEXT NOT NULL,
            PRIMARY KEY (season, game_id, play_id)
        );

        CREATE TABLE IF NOT EXISTS ingestion_runs (
            run_id TEXT PRIMARY KEY,
            run_ts TEXT NOT NULL,
            season INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            raw_rows INTEGER NOT NULL,
            inserted_rows INTEGER NOT NULL,
            model_play_rows INTEGER NOT NULL,
            missing_epa_rows INTEGER NOT NULL,
            replaced_existing INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_pbp_clean_season_week
            ON pbp_clean (season, week);
        CREATE INDEX IF NOT EXISTS idx_pbp_clean_posteam
            ON pbp_clean (posteam);
        CREATE INDEX IF NOT EXISTS idx_pbp_clean_defteam
            ON pbp_clean (defteam);
        CREATE INDEX IF NOT EXISTS idx_pbp_clean_players
            ON pbp_clean (passer_player_id, receiver_player_id, rusher_player_id);
        CREATE INDEX IF NOT EXISTS idx_ingestion_runs_season_ts
            ON ingestion_runs (season, run_ts);
        """
    )


def read_source_frame(input_dir: Path, season: int) -> pd.DataFrame:
    """Read one season raw parquet file."""
    source_file = input_dir / f"pbp_{season}.parquet"
    if not source_file.exists():
        raise FileNotFoundError(
            f"Raw source file not found for season {season}: {source_file}"
        )
    return pd.read_parquet(source_file, columns=RAW_COLUMNS)


def assert_required_columns(frame: pd.DataFrame) -> None:
    """Assert required source columns are present."""
    missing = SOURCE_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required source columns: {missing_list}")


def coerce_numeric_columns(frame: pd.DataFrame) -> None:
    """Apply deterministic numeric typing for modeling and SQL safety."""
    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")


def coerce_flag_columns(frame: pd.DataFrame) -> None:
    """Convert NFL event flags to 0/1 integers.

    Football rationale:
    - Many nflfastR flags are stored as nullable floats.
    - For reproducible modeling and SQL filters, explicit binary values are safer.
    """
    for column in INPUT_FLAG_COLUMNS:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype("int8")
        )


def derive_play_flags(frame: pd.DataFrame) -> None:
    """Derive standardized play filters used by downstream valuation models."""
    frame["is_special_teams_play"] = (
        frame["play_type"].isin(SPECIAL_TEAMS_PLAY_TYPES).astype("int8")
    )
    frame["is_non_action_play"] = (
        frame["play_type"].isin(NON_ACTION_PLAY_TYPES)
        | (frame["qb_kneel"] == 1)
        | (frame["qb_spike"] == 1)
    ).astype("int8")
    frame["is_scrimmage_play"] = (
        (frame["pass"] == 1) | (frame["rush"] == 1)
    ).astype("int8")

    # We model true scrimmage downs with valid context and target availability.
    valid_down = frame["down"].fillna(-1).between(1, 4)
    frame["is_model_play"] = (
        (frame["is_scrimmage_play"] == 1)
        & (frame["is_special_teams_play"] == 0)
        & (frame["is_non_action_play"] == 0)
        & valid_down
        & frame["posteam"].notna()
        & frame["defteam"].notna()
        & frame["epa"].notna()
        & frame["success"].notna()
    ).astype("int8")


def standardize_dates(frame: pd.DataFrame) -> None:
    """Store game_date as ISO-8601 text for clean SQLite interoperability."""
    frame["game_date"] = (
        pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    )


def validate_primary_keys(frame: pd.DataFrame) -> None:
    """Ensure season/game/play key integrity before DB writes."""
    null_key_rows = frame[["season", "game_id", "play_id"]].isna().any(axis=1).sum()
    if null_key_rows > 0:
        raise ValueError(f"Found {null_key_rows} rows with null primary key fields.")

    duplicate_rows = frame.duplicated(subset=["season", "game_id", "play_id"]).sum()
    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows} duplicate primary keys in season dataset."
        )


def clean_for_sqlite(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Create a clean modeling base table from raw PBP rows."""
    assert_required_columns(raw_frame)

    frame = raw_frame.rename(columns=COLUMN_RENAMES).copy()
    coerce_numeric_columns(frame)
    coerce_flag_columns(frame)
    derive_play_flags(frame)
    standardize_dates(frame)

    frame["ingestion_ts"] = datetime.now(tz=timezone.utc).isoformat()

    cleaned = frame[OUTPUT_COLUMNS].copy()
    validate_primary_keys(cleaned)

    # SQLite expects Python None for nulls, not pandas NA scalars.
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned


def count_existing_rows(connection: sqlite3.Connection, season: int) -> int:
    """Count existing rows for a season in pbp_clean."""
    row = connection.execute(
        "SELECT COUNT(*) FROM pbp_clean WHERE season = ?;", (season,)
    ).fetchone()
    return int(row[0]) if row else 0


def delete_existing_season(connection: sqlite3.Connection, season: int) -> int:
    """Delete one season to support idempotent reloads."""
    cursor = connection.execute("DELETE FROM pbp_clean WHERE season = ?;", (season,))
    return int(cursor.rowcount)


def insert_ingestion_run(
    connection: sqlite3.Connection,
    season: int,
    source_file: Path,
    raw_rows: int,
    inserted_rows: int,
    model_play_rows: int,
    missing_epa_rows: int,
    replaced_existing: bool,
    status: str,
    message: str | None,
) -> None:
    """Record metadata for one ingestion run."""
    connection.execute(
        """
        INSERT INTO ingestion_runs (
            run_id,
            run_ts,
            season,
            source_file,
            raw_rows,
            inserted_rows,
            model_play_rows,
            missing_epa_rows,
            replaced_existing,
            status,
            message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            str(uuid4()),
            datetime.now(tz=timezone.utc).isoformat(),
            season,
            str(source_file),
            raw_rows,
            inserted_rows,
            model_play_rows,
            missing_epa_rows,
            int(replaced_existing),
            status,
            message,
        ),
    )


def load_one_season(
    connection: sqlite3.Connection,
    input_dir: Path,
    db_path: Path,
    season: int,
    replace_existing_season: bool,
) -> SeasonLoadResult:
    """Load one season into SQLite with transactional safety."""
    source_file = input_dir / f"pbp_{season}.parquet"
    raw_frame = read_source_frame(input_dir=input_dir, season=season)
    cleaned_frame = clean_for_sqlite(raw_frame=raw_frame)

    raw_rows = len(raw_frame)
    model_play_rows = int(cleaned_frame["is_model_play"].sum())
    missing_epa_rows = int(cleaned_frame["epa"].isna().sum())

    existing_rows = count_existing_rows(connection=connection, season=season)
    replaced_existing = False

    with connection:
        if existing_rows > 0:
            if not replace_existing_season:
                message = (
                    f"Season {season} already has {existing_rows:,} rows in pbp_clean. "
                    "Use --replace-existing-season to overwrite."
                )
                insert_ingestion_run(
                    connection=connection,
                    season=season,
                    source_file=source_file,
                    raw_rows=raw_rows,
                    inserted_rows=0,
                    model_play_rows=model_play_rows,
                    missing_epa_rows=missing_epa_rows,
                    replaced_existing=False,
                    status="FAILED",
                    message=message,
                )
                raise ValueError(message)

            deleted_rows = delete_existing_season(connection=connection, season=season)
            replaced_existing = deleted_rows > 0
            LOGGER.info(
                "Deleted %s existing rows for season %s before reload.",
                f"{deleted_rows:,}",
                season,
            )

        cleaned_frame.to_sql(
            name="pbp_clean",
            con=connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2_500,
        )

        inserted_rows = count_existing_rows(connection=connection, season=season)
        insert_ingestion_run(
            connection=connection,
            season=season,
            source_file=source_file,
            raw_rows=raw_rows,
            inserted_rows=inserted_rows,
            model_play_rows=model_play_rows,
            missing_epa_rows=missing_epa_rows,
            replaced_existing=replaced_existing,
            status="SUCCESS",
            message="Season loaded successfully.",
        )

    return SeasonLoadResult(
        season=season,
        raw_rows=raw_rows,
        inserted_rows=inserted_rows,
        model_play_rows=model_play_rows,
        replaced_existing=replaced_existing,
        db_path=db_path,
    )


def log_summary(results: Sequence[SeasonLoadResult]) -> None:
    """Log an overall load summary after all seasons finish."""
    total_rows = sum(result.inserted_rows for result in results)
    total_model_rows = sum(result.model_play_rows for result in results)
    model_rate = 0.0 if total_rows == 0 else total_model_rows / total_rows

    LOGGER.info("Loaded %s season(s).", len(results))
    LOGGER.info("Total rows in loaded seasons: %s", f"{total_rows:,}")
    LOGGER.info("Total model-play rows: %s", f"{total_model_rows:,}")
    LOGGER.info("Model-play share: %.2f%%", model_rate * 100)

    for result in results:
        LOGGER.info(
            (
                "Season %s | rows=%s | model_rows=%s | replaced_existing=%s | "
                "db=%s"
            ),
            result.season,
            f"{result.inserted_rows:,}",
            f"{result.model_play_rows:,}",
            result.replaced_existing,
            result.db_path,
        )


def main() -> None:
    """Entry point for loading one or more seasons into SQLite."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    seasons = validate_seasons(args.seasons)

    connection = connect_sqlite(args.db_path)
    try:
        create_tables(connection)
        results: list[SeasonLoadResult] = []
        for season in seasons:
            LOGGER.info("Loading season %s into %s", season, args.db_path)
            result = load_one_season(
                connection=connection,
                input_dir=args.input_dir,
                db_path=args.db_path,
                season=season,
                replace_existing_season=args.replace_existing_season,
            )
            results.append(result)
        log_summary(results)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
