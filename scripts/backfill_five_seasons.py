#!/usr/bin/env python3
"""Backfill five NFL seasons of PBP data into parquet + SQLite.

This script orchestrates both earlier steps:
1) Pull raw season parquet files from nfl_data_py.
2) Clean and load each season into SQLite.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import pandas as pd

from load_pbp_into_sqlite import connect_sqlite, create_tables, load_one_season
from pull_pbp_data import (
    pull_season_pbp,
    validate_schema as validate_pull_schema,
    validate_season as validate_pull_season,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillSeasonResult:
    """Result summary for one season in the backfill process."""

    season: int
    raw_rows: int
    sqlite_rows: int
    model_play_rows: int
    raw_file: Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill five seasons of NFL PBP into raw parquet files and SQLite "
            "clean table."
        )
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2020,
        help="Start season for backfill window (inclusive). Default: 2020.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="End season for backfill window (inclusive). Default: 2024.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw parquet files are written.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/nfl_valuation.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--no-replace-existing-season",
        action="store_true",
        help=(
            "Do not replace existing season rows in SQLite. By default, this "
            "script replaces each season for deterministic reruns."
        ),
    )
    return parser.parse_args()


def validate_backfill_window(start_season: int, end_season: int) -> list[int]:
    """Validate backfill season range and enforce five-season scope.

    Football rationale:
    - The valuation model in this project uses a five-season sample to balance
      sample size (variance reduction) with recency (scheme/personnel relevance).
    """
    if start_season > end_season:
        raise ValueError("start-season must be <= end-season.")

    seasons = list(range(start_season, end_season + 1))
    for season in seasons:
        validate_pull_season(season)

    if len(seasons) != 5:
        raise ValueError(
            f"Backfill window must contain exactly 5 seasons. Got {len(seasons)}: "
            f"{seasons}."
        )
    return seasons


def write_raw_season_parquet(
    raw_frame: pd.DataFrame, output_dir: Path, season: int
) -> Path:
    """Persist one season raw frame to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pbp_{season}.parquet"
    raw_frame.to_parquet(output_path, index=False)
    return output_path


def run_backfill(
    seasons: Sequence[int],
    input_dir: Path,
    db_path: Path,
    replace_existing_season: bool,
) -> list[BackfillSeasonResult]:
    """Run pull + load for every season in the backfill window."""
    results: list[BackfillSeasonResult] = []

    connection: sqlite3.Connection = connect_sqlite(db_path)
    try:
        create_tables(connection)

        for season in seasons:
            LOGGER.info("Pulling raw PBP for season %s", season)
            raw_frame = pull_season_pbp(season)
            validate_pull_schema(raw_frame)
            raw_path = write_raw_season_parquet(
                raw_frame=raw_frame,
                output_dir=input_dir,
                season=season,
            )
            LOGGER.info(
                "Raw season %s saved to %s (%s rows)",
                season,
                raw_path,
                f"{len(raw_frame):,}",
            )

            LOGGER.info("Loading season %s into SQLite", season)
            load_result = load_one_season(
                connection=connection,
                input_dir=input_dir,
                db_path=db_path,
                season=season,
                replace_existing_season=replace_existing_season,
            )
            results.append(
                BackfillSeasonResult(
                    season=season,
                    raw_rows=len(raw_frame),
                    sqlite_rows=load_result.inserted_rows,
                    model_play_rows=load_result.model_play_rows,
                    raw_file=raw_path,
                )
            )
    finally:
        connection.close()

    return results


def log_backfill_summary(results: Sequence[BackfillSeasonResult], db_path: Path) -> None:
    """Emit a concise summary for operator validation."""
    total_rows = sum(result.sqlite_rows for result in results)
    total_model_rows = sum(result.model_play_rows for result in results)
    model_share = 0.0 if total_rows == 0 else total_model_rows / total_rows

    LOGGER.info("Backfill complete for %s seasons.", len(results))
    LOGGER.info("SQLite total rows across backfilled seasons: %s", f"{total_rows:,}")
    LOGGER.info("SQLite total model-play rows: %s", f"{total_model_rows:,}")
    LOGGER.info("SQLite model-play share: %.2f%%", model_share * 100)
    LOGGER.info("SQLite database path: %s", db_path)

    for result in results:
        LOGGER.info(
            "Season %s | raw_rows=%s | sqlite_rows=%s | model_rows=%s | raw_file=%s",
            result.season,
            f"{result.raw_rows:,}",
            f"{result.sqlite_rows:,}",
            f"{result.model_play_rows:,}",
            result.raw_file,
        )


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    seasons = validate_backfill_window(
        start_season=args.start_season,
        end_season=args.end_season,
    )
    replace_existing_season = not args.no_replace_existing_season

    LOGGER.info(
        "Backfill window: %s (replace_existing_season=%s)",
        seasons,
        replace_existing_season,
    )
    results = run_backfill(
        seasons=seasons,
        input_dir=args.input_dir,
        db_path=args.db_path,
        replace_existing_season=replace_existing_season,
    )
    log_backfill_summary(results=results, db_path=args.db_path)


if __name__ == "__main__":
    main()
