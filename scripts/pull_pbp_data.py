#!/usr/bin/env python3
"""Pull and persist NFL play-by-play (PBP) data for one season.

This script is intentionally strict because downstream valuation work depends on
stable core fields (EPA, success, down/distance, and offensive/defensive team).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Final

import nfl_data_py as nfl
import pandas as pd

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
MIN_SUPPORTED_SEASON: Final[int] = 1999
REQUIRED_COLUMNS: Final[set[str]] = {
    "season",
    "game_id",
    "play_id",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "epa",
    "success",
    "play_type",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download one NFL season of play-by-play data from nfl_data_py."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season to pull (e.g., 2024).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--file-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Storage format for raw data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def validate_season(season: int) -> None:
    """Validate season boundaries before making a network call."""
    current_year = datetime.now().year
    if season < MIN_SUPPORTED_SEASON or season > current_year:
        msg = (
            f"Season {season} is invalid. Choose a season between "
            f"{MIN_SUPPORTED_SEASON} and {current_year}."
        )
        raise ValueError(msg)


def pull_season_pbp(season: int) -> pd.DataFrame:
    """Fetch one season of PBP data from nfl_data_py.

    Football rationale:
    - We keep every play event for now (no filtering), because later valuation
      work needs to decide whether to include/exclude kneels, spikes, penalties,
      and special teams based on explicit model assumptions.
    """
    LOGGER.info("Downloading play-by-play data for season %s...", season)
    pbp_df = nfl.import_pbp_data(
        years=[season],
        downcast=True,
        cache=False,
    )
    if pbp_df.empty:
        raise ValueError(f"No play-by-play rows returned for season {season}.")
    return pbp_df


def validate_schema(pbp_df: pd.DataFrame) -> None:
    """Ensure required columns exist for Step 1 and future modeling."""
    missing_columns = REQUIRED_COLUMNS.difference(pbp_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Play-by-play data is missing required columns: " f"{missing_list}"
        )


def write_output(
    pbp_df: pd.DataFrame,
    output_dir: Path,
    season: int,
    file_format: str,
    overwrite: bool,
) -> Path:
    """Persist the pulled season as a raw artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pbp_{season}.{file_format}"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )

    if file_format == "parquet":
        pbp_df.to_parquet(output_path, index=False)
    elif file_format == "csv":
        pbp_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return output_path


def log_summary(pbp_df: pd.DataFrame, season: int, output_path: Path) -> None:
    """Log pull summary so the run can be validated quickly."""
    total_rows = len(pbp_df)
    total_games = pbp_df["game_id"].nunique()
    total_teams = pbp_df["posteam"].dropna().nunique()
    epa_not_null = pbp_df["epa"].notna().sum()

    LOGGER.info("Season pulled: %s", season)
    LOGGER.info("Rows pulled: %s", f"{total_rows:,}")
    LOGGER.info("Unique games: %s", f"{total_games:,}")
    LOGGER.info("Unique offensive teams: %s", total_teams)
    LOGGER.info("Rows with EPA available: %s", f"{epa_not_null:,}")
    LOGGER.info("Saved raw dataset to: %s", output_path)


def main() -> None:
    """Run the one-season data pull pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    validate_season(args.season)
    pbp_df = pull_season_pbp(args.season)
    validate_schema(pbp_df)
    output_path = write_output(
        pbp_df=pbp_df,
        output_dir=args.output_dir,
        season=args.season,
        file_format=args.file_format,
        overwrite=args.overwrite,
    )
    log_summary(pbp_df=pbp_df, season=args.season, output_path=output_path)


if __name__ == "__main__":
    main()
