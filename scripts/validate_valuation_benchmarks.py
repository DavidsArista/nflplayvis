#!/usr/bin/env python3
"""Validate player valuation against Pro Bowl and public benchmark proxies.

This step creates two validation layers:
1) Alignment with external honors (Pro Bowl selections).
2) Rank correlation with public "PFF-style" performance proxies.

If an external benchmark CSV is provided (e.g., manually curated PFF-like grades),
it will be used as primary benchmark where available.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Sequence
from uuid import uuid4

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
SUPPORTED_POSITIONS: Final[list[str]] = ["QB", "RB", "WR", "TE"]
TABLE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

VALIDATION_RUN_TABLE: Final[str] = "valuation_validation_runs"
DEFAULT_SUMMARY_TABLE: Final[str] = "valuation_validation_summary"
DEFAULT_PLAYER_TABLE: Final[str] = "valuation_validation_player"
DEFAULT_MODEL_NAME: Final[str] = "xgb_play_success_epa"


@dataclass(frozen=True)
class ProBowlMappingStats:
    """Track mapping quality for Pro Bowl scraped names."""

    total_entries: int
    mapped_entries: int
    ambiguous_entries: int
    unresolved_entries: int


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate valuation outputs against Pro Bowl and benchmark proxies."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/nfl_valuation.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--valuation-table",
        type=str,
        default="player_valuation",
        help="Source valuation table.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name in valuation table.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024],
        help="Seasons to validate (default: 2020 2021 2022 2023 2024).",
    )
    parser.add_argument(
        "--external-benchmark-csv",
        type=Path,
        default=None,
        help=(
            "Optional external benchmark CSV with columns "
            "season,player_id,position_group,benchmark_score (or pff_grade)."
        ),
    )
    parser.add_argument(
        "--summary-table",
        type=str,
        default=DEFAULT_SUMMARY_TABLE,
        help="Output summary table name.",
    )
    parser.add_argument(
        "--player-table",
        type=str,
        default=DEFAULT_PLAYER_TABLE,
        help="Output player-detail table name.",
    )
    parser.add_argument(
        "--replace-existing-season",
        action="store_true",
        help="Replace existing validation rows for selected seasons/model.",
    )
    return parser.parse_args()


def validate_table_name(table_name: str) -> None:
    """Validate dynamic SQL table names."""
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError(f"Invalid table name: {table_name}")


def validate_seasons(seasons: Sequence[int]) -> list[int]:
    """Validate and normalize season list."""
    normalized = sorted(set(seasons))
    current_year = datetime.now(tz=timezone.utc).year
    for season in normalized:
        if season < 1999 or season > current_year:
            raise ValueError(f"Invalid season {season}. Use 1999-{current_year}.")
    return normalized


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Create SQLite connection with WAL."""
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if table exists."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    ).fetchone()
    return row is not None


def canonicalize_name(name: str) -> str:
    """Canonicalize names for fuzzy deterministic joins."""
    cleaned = name.lower()
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def map_position_group(position_label: str) -> str | None:
    """Map position labels to supported valuation groups."""
    text = str(position_label).upper()
    text = re.sub(r"[^A-Z0-9/ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    tokens = set(text.replace("/", " ").split())
    if "QB" in tokens or "QUARTERBACK" in tokens:
        return "QB"
    if "WR" in tokens or ("WIDE" in tokens and "RECEIVER" in tokens):
        return "WR"
    if "TE" in tokens or ("TIGHT" in tokens and "END" in tokens):
        return "TE"
    if (
        "RB" in tokens
        or "HB" in tokens
        or "FB" in tokens
        or ("RUNNING" in tokens and "BACK" in tokens)
        or "HALFBACK" in tokens
        or "FULLBACK" in tokens
    ):
        return "RB"
    return None


def normalize_column_name(column: object) -> str:
    """Normalize table column name for roster-table detection."""
    if isinstance(column, tuple):
        joined = " ".join(str(item) for item in column if str(item) != "nan")
    else:
        joined = str(column)
    joined = re.sub(r"\s+", " ", joined).strip().lower()
    return joined


def split_player_cell(cell_value: object) -> list[str]:
    """Split roster cell into individual player names."""
    if pd.isna(cell_value):
        return []
    text = str(cell_value)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace(" and ", "\n")
    text = text.replace(" / ", "\n")
    text = text.replace(";", "\n")
    text = text.replace("|", "\n")
    parts = [part.strip() for part in text.split("\n")]
    names = []
    for part in parts:
        if not part:
            continue
        if part.lower() in {"nan", "none", "-", "vacant"}:
            continue
        names.append(part)
    return names


def fetch_wikipedia_html(page_title: str) -> str | None:
    """Fetch a Wikipedia page by title and return HTML or None if missing."""
    url = f"https://en.wikipedia.org/wiki/{page_title}"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (nflplayvis-validation-bot)"},
        timeout=30,
    )
    if response.status_code != 200:
        return None
    html = response.text
    if "Wikipedia does not have an article with this exact name" in html:
        return None
    return html


def scrape_pro_bowl_for_season(season: int) -> pd.DataFrame:
    """Scrape Pro Bowl player names from Wikipedia season page."""
    html: str | None = None
    for title in [f"{season}_Pro_Bowl_Games", f"{season}_Pro_Bowl"]:
        html = fetch_wikipedia_html(title)
        if html is not None:
            LOGGER.info("Using Wikipedia Pro Bowl page: %s", title)
            break
    if html is None:
        LOGGER.warning("No Pro Bowl page found for season %s.", season)
        return pd.DataFrame(
            columns=[
                "season",
                "position_group",
                "pro_bowl_player_name",
                "pro_bowl_name_canon",
                "selection_bucket",
            ]
        )

    tables = pd.read_html(html)
    rows: list[dict[str, object]] = []

    for table in tables:
        normalized_cols = {column: normalize_column_name(column) for column in table.columns}
        position_col = None
        for column, normalized in normalized_cols.items():
            if "position" in normalized:
                position_col = column
                break
        if position_col is None:
            continue

        selected_player_cols = []
        for column, normalized in normalized_cols.items():
            if "starter" in normalized or "reserve" in normalized:
                selected_player_cols.append(column)
        if not selected_player_cols:
            continue

        for _, row in table.iterrows():
            position_group = map_position_group(str(row[position_col]))
            if position_group is None:
                continue
            for column in selected_player_cols:
                bucket = normalize_column_name(column)
                for player_name in split_player_cell(row[column]):
                    rows.append(
                        {
                            "season": season,
                            "position_group": position_group,
                            "pro_bowl_player_name": player_name,
                            "pro_bowl_name_canon": canonicalize_name(player_name),
                            "selection_bucket": bucket,
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "position_group",
                "pro_bowl_player_name",
                "pro_bowl_name_canon",
                "selection_bucket",
            ]
        )

    result = pd.DataFrame(rows).drop_duplicates(
        subset=["season", "position_group", "pro_bowl_name_canon"]
    )
    return result.reset_index(drop=True)


def scrape_pro_bowl_all_seasons(seasons: Sequence[int]) -> pd.DataFrame:
    """Scrape Pro Bowl entries for all seasons."""
    frames = [scrape_pro_bowl_for_season(season) for season in seasons]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        subset=["season", "position_group", "pro_bowl_name_canon"]
    ).reset_index(drop=True)


def build_player_name_lookup() -> tuple[
    dict[tuple[str, str], set[str]],
    dict[str, set[str]],
    dict[str, list[str]],
    list[str],
]:
    """Build canonical-name lookup to gsis_id."""
    players = nfl.import_players()[["gsis_id", "display_name", "position"]].dropna(
        subset=["gsis_id", "display_name"]
    )
    ids = nfl.import_ids()[["gsis_id", "name", "position"]].dropna(
        subset=["gsis_id", "name"]
    )

    players = players.rename(columns={"display_name": "source_name"})
    ids = ids.rename(columns={"name": "source_name"})
    combined = pd.concat([players[["gsis_id", "source_name", "position"]], ids], ignore_index=True)

    combined["position_group"] = combined["position"].map(map_position_group)
    combined["name_canon"] = combined["source_name"].map(canonicalize_name)
    combined = combined[
        combined["name_canon"].ne("") & combined["position_group"].isin(SUPPORTED_POSITIONS)
    ].copy()

    lookup_name_pos: dict[tuple[str, str], set[str]] = {}
    lookup_name: dict[str, set[str]] = {}
    for _, row in combined.iterrows():
        key_pos = (str(row["name_canon"]), str(row["position_group"]))
        lookup_name_pos.setdefault(key_pos, set()).add(str(row["gsis_id"]))
        lookup_name.setdefault(str(row["name_canon"]), set()).add(str(row["gsis_id"]))
    names_by_pos: dict[str, list[str]] = {}
    for (name_canon, position_group) in lookup_name_pos:
        names_by_pos.setdefault(position_group, []).append(name_canon)
    names_all = list(lookup_name.keys())
    return lookup_name_pos, lookup_name, names_by_pos, names_all


def map_pro_bowl_to_player_ids(pro_bowl_frame: pd.DataFrame) -> tuple[pd.DataFrame, ProBowlMappingStats]:
    """Map scraped Pro Bowl names to gsis player IDs."""
    if pro_bowl_frame.empty:
        empty = pd.DataFrame(
            columns=[
                "season",
                "position_group",
                "player_id",
                "pro_bowl_player_name",
                "mapping_status",
            ]
        )
        stats = ProBowlMappingStats(0, 0, 0, 0)
        return empty, stats

    (
        lookup_name_pos,
        lookup_name,
        names_by_pos,
        names_all,
    ) = build_player_name_lookup()
    mapped_rows: list[dict[str, object]] = []
    mapped = 0
    ambiguous = 0
    unresolved = 0

    for _, row in pro_bowl_frame.iterrows():
        season = int(row["season"])
        position_group = str(row["position_group"])
        name_canon = str(row["pro_bowl_name_canon"])
        raw_name = str(row["pro_bowl_player_name"])

        candidates = set(lookup_name_pos.get((name_canon, position_group), set()))
        if len(candidates) == 0:
            candidates = set(lookup_name.get(name_canon, set()))

        # Substring fallback for cells that include jersey numbers, teams, and notes.
        if len(candidates) == 0 and name_canon:
            padded_name = f" {name_canon} "
            pos_name_matches = []
            for candidate_name in names_by_pos.get(position_group, []):
                if len(candidate_name) < 5:
                    continue
                if f" {candidate_name} " in padded_name:
                    pos_name_matches.append(candidate_name)

            if pos_name_matches:
                max_len = max(len(name) for name in pos_name_matches)
                longest_matches = [
                    name for name in pos_name_matches if len(name) == max_len
                ]
                for candidate_name in longest_matches:
                    candidates.update(
                        lookup_name_pos.get((candidate_name, position_group), set())
                    )

        if len(candidates) == 0 and name_canon:
            padded_name = f" {name_canon} "
            global_name_matches = []
            for candidate_name in names_all:
                if len(candidate_name) < 5:
                    continue
                if f" {candidate_name} " in padded_name:
                    global_name_matches.append(candidate_name)
            if global_name_matches:
                max_len = max(len(name) for name in global_name_matches)
                longest_matches = [
                    name for name in global_name_matches if len(name) == max_len
                ]
                for candidate_name in longest_matches:
                    candidates.update(lookup_name.get(candidate_name, set()))

        if len(candidates) == 1:
            mapped += 1
            player_id = next(iter(candidates))
            mapped_rows.append(
                {
                    "season": season,
                    "position_group": position_group,
                    "player_id": player_id,
                    "pro_bowl_player_name": raw_name,
                    "mapping_status": "mapped",
                }
            )
        elif len(candidates) > 1:
            ambiguous += 1
            for player_id in sorted(candidates):
                mapped_rows.append(
                    {
                        "season": season,
                        "position_group": position_group,
                        "player_id": player_id,
                        "pro_bowl_player_name": raw_name,
                        "mapping_status": "ambiguous",
                    }
                )
        else:
            unresolved += 1

    mapped_df = pd.DataFrame(mapped_rows)
    if mapped_df.empty:
        mapped_df = pd.DataFrame(
            columns=[
                "season",
                "position_group",
                "player_id",
                "pro_bowl_player_name",
                "mapping_status",
            ]
        )
    mapped_df = mapped_df.drop_duplicates(subset=["season", "position_group", "player_id"])
    stats = ProBowlMappingStats(
        total_entries=len(pro_bowl_frame),
        mapped_entries=mapped,
        ambiguous_entries=ambiguous,
        unresolved_entries=unresolved,
    )
    return mapped_df, stats


def load_valuation_rows(
    connection: sqlite3.Connection,
    valuation_table: str,
    model_name: str,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """Load valuation rows for target model and seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    query = f"""
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
            context_adjusted_per_snap,
            context_adjusted_total_value,
            value_percentile,
            is_qualified
        FROM {valuation_table}
        WHERE model_name = ?
          AND season IN ({placeholders})
          AND position_group IN ('QB', 'RB', 'WR', 'TE');
    """
    params = [model_name] + list(seasons)
    frame = pd.read_sql_query(query, connection, params=params)
    if frame.empty:
        raise ValueError("No valuation rows loaded for selected seasons/model.")
    return frame


def build_public_proxy_benchmark(seasons: Sequence[int]) -> pd.DataFrame:
    """Build public PFF-style benchmark proxies from weekly nflverse metrics."""
    weekly = nfl.import_weekly_data(years=list(seasons))
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly = weekly[weekly["position_group"].isin(SUPPORTED_POSITIONS)].copy()

    numeric_cols = [
        "attempts",
        "carries",
        "targets",
        "passing_epa",
        "rushing_epa",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
        "dakota",
    ]
    for column in numeric_cols:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce")

    grouped = weekly.groupby(["season", "player_id", "position_group"], as_index=False).agg(
        attempts=("attempts", "sum"),
        carries=("carries", "sum"),
        targets=("targets", "sum"),
        passing_epa=("passing_epa", "sum"),
        rushing_epa=("rushing_epa", "sum"),
        receiving_epa=("receiving_epa", "sum"),
        target_share=("target_share", "mean"),
        air_yards_share=("air_yards_share", "mean"),
        wopr=("wopr", "mean"),
        dakota=("dakota", "mean"),
    )

    grouped["pass_epa_per_att"] = np.where(
        grouped["attempts"] > 0,
        grouped["passing_epa"] / grouped["attempts"],
        0.0,
    )
    grouped["rush_epa_per_carry"] = np.where(
        grouped["carries"] > 0,
        grouped["rushing_epa"] / grouped["carries"],
        0.0,
    )
    grouped["recv_epa_per_target"] = np.where(
        grouped["targets"] > 0,
        grouped["receiving_epa"] / grouped["targets"],
        0.0,
    )

    qb_score = (
        grouped["pass_epa_per_att"]
        + 0.35 * grouped["rush_epa_per_carry"]
        + 0.02 * grouped["dakota"].fillna(0.0)
    )
    rb_score = (
        grouped["rush_epa_per_carry"]
        + 0.55 * grouped["recv_epa_per_target"]
        + 0.10 * grouped["target_share"].fillna(0.0)
    )
    wr_te_score = (
        grouped["recv_epa_per_target"]
        + 0.25 * grouped["wopr"].fillna(0.0)
        + 0.15 * grouped["air_yards_share"].fillna(0.0)
    )

    grouped["benchmark_score_public"] = np.select(
        [
            grouped["position_group"] == "QB",
            grouped["position_group"] == "RB",
            grouped["position_group"].isin(["WR", "TE"]),
        ],
        [qb_score, rb_score, wr_te_score],
        default=np.nan,
    )
    grouped["benchmark_source"] = "public_proxy"

    return grouped[
        ["season", "player_id", "position_group", "benchmark_score_public", "benchmark_source"]
    ].copy()


def load_external_benchmark(external_csv: Path | None) -> pd.DataFrame:
    """Load optional external benchmark CSV."""
    if external_csv is None:
        return pd.DataFrame(
            columns=["season", "player_id", "position_group", "benchmark_score_external"]
        )
    if not external_csv.exists():
        raise FileNotFoundError(f"External benchmark CSV not found: {external_csv}")

    frame = pd.read_csv(external_csv)
    required_base = {"season", "player_id", "position_group"}
    missing_base = required_base.difference(frame.columns)
    if missing_base:
        raise ValueError(f"External benchmark CSV missing columns: {sorted(missing_base)}")

    if "benchmark_score" in frame.columns:
        score_col = "benchmark_score"
    elif "pff_grade" in frame.columns:
        score_col = "pff_grade"
    else:
        raise ValueError(
            "External benchmark CSV must include benchmark_score or pff_grade column."
        )

    frame = frame.copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["position_group"] = frame["position_group"].astype(str).str.upper()
    frame["benchmark_score_external"] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame.dropna(subset=["season", "player_id", "position_group", "benchmark_score_external"])
    frame["season"] = frame["season"].astype(int)
    frame = frame[frame["position_group"].isin(SUPPORTED_POSITIONS)]
    frame = frame.drop_duplicates(subset=["season", "player_id", "position_group"], keep="first")
    return frame[
        ["season", "player_id", "position_group", "benchmark_score_external"]
    ].copy()


def merge_validation_inputs(
    valuation: pd.DataFrame,
    pro_bowl_mapped: pd.DataFrame,
    benchmark_public: pd.DataFrame,
    benchmark_external: pd.DataFrame,
) -> pd.DataFrame:
    """Merge valuation, pro bowl labels, and benchmarks."""
    working = valuation.copy()
    working["season"] = pd.to_numeric(working["season"], errors="coerce").astype(int)
    working["is_qualified"] = (
        pd.to_numeric(working["is_qualified"], errors="coerce").fillna(0).astype(int)
    )
    working["valuation_score"] = pd.to_numeric(
        working["context_adjusted_per_snap"], errors="coerce"
    )

    pro = pro_bowl_mapped.copy()
    if not pro.empty:
        pro = pro[pro["mapping_status"] == "mapped"].copy()
        pro["is_pro_bowl"] = 1
        pro = pro[["season", "player_id", "position_group", "is_pro_bowl"]].drop_duplicates()
        working = working.merge(
            pro,
            on=["season", "player_id", "position_group"],
            how="left",
            validate="many_to_one",
        )
    else:
        working["is_pro_bowl"] = 0
    working["is_pro_bowl"] = working["is_pro_bowl"].fillna(0).astype(int)

    working = working.merge(
        benchmark_public,
        on=["season", "player_id", "position_group"],
        how="left",
        validate="many_to_one",
    )
    if not benchmark_external.empty:
        working = working.merge(
            benchmark_external,
            on=["season", "player_id", "position_group"],
            how="left",
            validate="many_to_one",
        )
    else:
        working["benchmark_score_external"] = np.nan

    working["benchmark_score"] = np.where(
        working["benchmark_score_external"].notna(),
        working["benchmark_score_external"],
        working["benchmark_score_public"],
    )
    working["benchmark_source"] = np.where(
        working["benchmark_score_external"].notna(),
        "external_pff_like",
        "public_proxy",
    )

    # Rank by season + position among qualified players only.
    working["valuation_rank"] = np.nan
    working["benchmark_rank"] = np.nan
    for (_, _), group in working.groupby(["season", "position_group"]):
        idx = group.index
        qualified_mask = group["is_qualified"] == 1

        if qualified_mask.any():
            qualified = group.loc[qualified_mask]
            val_rank = qualified["valuation_score"].rank(
                ascending=False, method="min", na_option="bottom"
            )
            working.loc[qualified.index, "valuation_rank"] = val_rank

            bench_rank = qualified["benchmark_score"].rank(
                ascending=False, method="min", na_option="bottom"
            )
            working.loc[qualified.index, "benchmark_rank"] = bench_rank
        else:
            working.loc[idx, "valuation_rank"] = np.nan
            working.loc[idx, "benchmark_rank"] = np.nan

    return working


def evaluate_group_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Compute validation metrics for one season-position group."""
    qualified = group[group["is_qualified"] == 1].copy()
    n_players = len(qualified)
    n_with_benchmark = int(qualified["benchmark_score"].notna().sum())
    n_pro_bowl_total = int(group["is_pro_bowl"].sum())
    n_pro_bowl_in_pool = int(qualified["is_pro_bowl"].sum())

    top_k = max(n_pro_bowl_in_pool, 1)
    top_players = qualified.nsmallest(top_k, "valuation_rank") if n_players > 0 else qualified
    hits_at_k = int(top_players["is_pro_bowl"].sum()) if n_players > 0 else 0
    precision_at_k = float(hits_at_k / top_k) if top_k > 0 else float("nan")
    recall_at_k = (
        float(hits_at_k / n_pro_bowl_in_pool) if n_pro_bowl_in_pool > 0 else float("nan")
    )

    pro_bowl_ranks = qualified.loc[qualified["is_pro_bowl"] == 1, "valuation_rank"]
    mean_pro_bowl_rank = float(pro_bowl_ranks.mean()) if not pro_bowl_ranks.empty else float("nan")
    median_pro_bowl_rank = (
        float(pro_bowl_ranks.median()) if not pro_bowl_ranks.empty else float("nan")
    )

    bench_subset = qualified[qualified["benchmark_score"].notna()].copy()
    if len(bench_subset) >= 5 and bench_subset["valuation_score"].nunique() > 1:
        spearman = float(
            bench_subset["valuation_score"].corr(bench_subset["benchmark_score"], method="spearman")
        )
        kendall = float(
            bench_subset["valuation_score"].corr(bench_subset["benchmark_score"], method="kendall")
        )
    else:
        spearman = float("nan")
        kendall = float("nan")

    if qualified["is_pro_bowl"].nunique() > 1 and qualified["valuation_score"].nunique() > 1:
        pro_bowl_auc = float(roc_auc_score(qualified["is_pro_bowl"], qualified["valuation_score"]))
    else:
        pro_bowl_auc = float("nan")

    return {
        "n_players": int(n_players),
        "n_players_with_benchmark": int(n_with_benchmark),
        "n_pro_bowl_total_mapped": int(n_pro_bowl_total),
        "n_pro_bowl_in_qualified_pool": int(n_pro_bowl_in_pool),
        "top_k": int(top_k),
        "hits_at_k": int(hits_at_k),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mean_pro_bowl_rank": mean_pro_bowl_rank,
        "median_pro_bowl_rank": median_pro_bowl_rank,
        "spearman_with_benchmark": spearman,
        "kendall_with_benchmark": kendall,
        "pro_bowl_auc": pro_bowl_auc,
    }


def build_validation_tables(
    merged: pd.DataFrame, model_name: str, run_ts: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build player-detail and summary validation tables."""
    player_cols = [
        "season",
        "position_group",
        "player_id",
        "player_name",
        "primary_team",
        "is_qualified",
        "is_pro_bowl",
        "opportunities",
        "offense_snaps_resolved",
        "valuation_score",
        "valuation_rank",
        "value_percentile",
        "benchmark_score",
        "benchmark_rank",
        "benchmark_source",
        "context_adjusted_total_value",
    ]
    player_frame = merged[player_cols].copy()
    player_frame["model_name"] = model_name
    player_frame["validation_run_ts"] = run_ts
    player_frame = player_frame[
        [
            "model_name",
            "validation_run_ts",
            *player_cols,
        ]
    ]

    summary_rows: list[dict[str, object]] = []
    for (season, position_group), group in merged.groupby(["season", "position_group"]):
        metrics = evaluate_group_metrics(group)
        summary_rows.append(
            {
                "model_name": model_name,
                "validation_run_ts": run_ts,
                "season": int(season),
                "position_group": position_group,
                **metrics,
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame = summary_frame.sort_values(["season", "position_group"]).reset_index(drop=True)
    return player_frame, summary_frame


def create_output_tables(
    connection: sqlite3.Connection, summary_table: str, player_table: str
) -> None:
    """Create output tables and indexes."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {summary_table} (
            model_name TEXT NOT NULL,
            validation_run_ts TEXT NOT NULL,
            season INTEGER NOT NULL,
            position_group TEXT NOT NULL,
            n_players INTEGER NOT NULL,
            n_players_with_benchmark INTEGER NOT NULL,
            n_pro_bowl_total_mapped INTEGER NOT NULL,
            n_pro_bowl_in_qualified_pool INTEGER NOT NULL,
            top_k INTEGER NOT NULL,
            hits_at_k INTEGER NOT NULL,
            precision_at_k REAL,
            recall_at_k REAL,
            mean_pro_bowl_rank REAL,
            median_pro_bowl_rank REAL,
            spearman_with_benchmark REAL,
            kendall_with_benchmark REAL,
            pro_bowl_auc REAL
        );
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {player_table} (
            model_name TEXT NOT NULL,
            validation_run_ts TEXT NOT NULL,
            season INTEGER NOT NULL,
            position_group TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT,
            primary_team TEXT,
            is_qualified INTEGER NOT NULL,
            is_pro_bowl INTEGER NOT NULL,
            opportunities REAL,
            offense_snaps_resolved REAL,
            valuation_score REAL,
            valuation_rank REAL,
            value_percentile REAL,
            benchmark_score REAL,
            benchmark_rank REAL,
            benchmark_source TEXT,
            context_adjusted_total_value REAL
        );
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {VALIDATION_RUN_TABLE} (
            run_id TEXT PRIMARY KEY,
            run_ts TEXT NOT NULL,
            model_name TEXT NOT NULL,
            seasons TEXT NOT NULL,
            summary_table TEXT NOT NULL,
            player_table TEXT NOT NULL,
            pro_bowl_entries INTEGER NOT NULL,
            pro_bowl_mapped INTEGER NOT NULL,
            pro_bowl_ambiguous INTEGER NOT NULL,
            pro_bowl_unresolved INTEGER NOT NULL,
            status TEXT NOT NULL,
            message TEXT
        );
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{summary_table}_model_season_pos
        ON {summary_table} (model_name, season, position_group);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{player_table}_model_season_pos
        ON {player_table} (model_name, season, position_group);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{player_table}_player
        ON {player_table} (player_id, season);
        """
    )


def count_existing_rows(
    connection: sqlite3.Connection, table_name: str, model_name: str, seasons: Sequence[int]
) -> int:
    """Count rows for model+seasons in table."""
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


def delete_existing_rows(
    connection: sqlite3.Connection, table_name: str, model_name: str, seasons: Sequence[int]
) -> int:
    """Delete rows for model+seasons in table."""
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


def write_run_metadata(
    connection: sqlite3.Connection,
    run_ts: str,
    model_name: str,
    seasons: Sequence[int],
    summary_table: str,
    player_table: str,
    mapping_stats: ProBowlMappingStats,
    status: str,
    message: str,
) -> None:
    """Write run metadata row."""
    connection.execute(
        f"""
        INSERT INTO {VALIDATION_RUN_TABLE} (
            run_id,
            run_ts,
            model_name,
            seasons,
            summary_table,
            player_table,
            pro_bowl_entries,
            pro_bowl_mapped,
            pro_bowl_ambiguous,
            pro_bowl_unresolved,
            status,
            message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            str(uuid4()),
            run_ts,
            model_name,
            ",".join(str(season) for season in seasons),
            summary_table,
            player_table,
            mapping_stats.total_entries,
            mapping_stats.mapped_entries,
            mapping_stats.ambiguous_entries,
            mapping_stats.unresolved_entries,
            status,
            message,
        ),
    )


def main() -> None:
    """Run validation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {args.db_path}")

    validate_table_name(args.valuation_table)
    validate_table_name(args.summary_table)
    validate_table_name(args.player_table)
    seasons = validate_seasons(args.seasons)

    run_ts = datetime.now(tz=timezone.utc).isoformat()
    connection = connect_sqlite(args.db_path)
    try:
        create_output_tables(
            connection=connection,
            summary_table=args.summary_table,
            player_table=args.player_table,
        )

        valuation = load_valuation_rows(
            connection=connection,
            valuation_table=args.valuation_table,
            model_name=args.model_name,
            seasons=seasons,
        )
        LOGGER.info("Loaded %s valuation rows.", f"{len(valuation):,}")

        pro_bowl_scraped = scrape_pro_bowl_all_seasons(seasons)
        pro_bowl_mapped, mapping_stats = map_pro_bowl_to_player_ids(pro_bowl_scraped)
        LOGGER.info(
            (
                "Pro Bowl mapping | entries=%s | mapped=%s | ambiguous=%s | "
                "unresolved=%s"
            ),
            mapping_stats.total_entries,
            mapping_stats.mapped_entries,
            mapping_stats.ambiguous_entries,
            mapping_stats.unresolved_entries,
        )

        benchmark_public = build_public_proxy_benchmark(seasons)
        benchmark_external = load_external_benchmark(args.external_benchmark_csv)
        if not benchmark_external.empty:
            LOGGER.info(
                "Loaded external benchmark rows: %s",
                f"{len(benchmark_external):,}",
            )
        else:
            LOGGER.info("No external benchmark provided; using public proxy only.")

        merged = merge_validation_inputs(
            valuation=valuation,
            pro_bowl_mapped=pro_bowl_mapped,
            benchmark_public=benchmark_public,
            benchmark_external=benchmark_external,
        )
        player_frame, summary_frame = build_validation_tables(
            merged=merged,
            model_name=args.model_name,
            run_ts=run_ts,
        )
        player_frame = player_frame.where(pd.notna(player_frame), None)
        summary_frame = summary_frame.where(pd.notna(summary_frame), None)

        with connection:
            for table_name in [args.summary_table, args.player_table]:
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

            summary_frame.to_sql(
                name=args.summary_table,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=1000,
            )
            player_frame.to_sql(
                name=args.player_table,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=1000,
            )
            write_run_metadata(
                connection=connection,
                run_ts=run_ts,
                model_name=args.model_name,
                seasons=seasons,
                summary_table=args.summary_table,
                player_table=args.player_table,
                mapping_stats=mapping_stats,
                status="SUCCESS",
                message="Valuation validation completed.",
            )

        LOGGER.info(
            "Validation summary rows written: %s to %s",
            f"{len(summary_frame):,}",
            args.summary_table,
        )
        LOGGER.info(
            "Validation player rows written: %s to %s",
            f"{len(player_frame):,}",
            args.player_table,
        )
    except Exception as exc:
        with connection:
            create_output_tables(
                connection=connection,
                summary_table=args.summary_table,
                player_table=args.player_table,
            )
            write_run_metadata(
                connection=connection,
                run_ts=datetime.now(tz=timezone.utc).isoformat(),
                model_name=args.model_name,
                seasons=seasons,
                summary_table=args.summary_table,
                player_table=args.player_table,
                mapping_stats=ProBowlMappingStats(0, 0, 0, 0),
                status="FAILED",
                message=str(exc),
            )
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
