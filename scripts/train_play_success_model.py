#!/usr/bin/env python3
"""Train an XGBoost play-success model and persist per-play predictions.

Model objective:
- Predict probability that a play is successful under an EPA-based definition.
- EPA-based success target defaults to: 1 if target_epa > 0 else 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_TRAIN_SEASONS: Final[list[int]] = [2020, 2021, 2022]
DEFAULT_VALIDATION_SEASON: Final[int] = 2023
DEFAULT_TEST_SEASON: Final[int] = 2024
DEFAULT_MODEL_NAME: Final[str] = "xgb_play_success_epa"
DEFAULT_FEATURE_TABLE: Final[str] = "pbp_features"
DEFAULT_PREDICTIONS_TABLE: Final[str] = "play_success_predictions"
MODEL_RUN_TABLE: Final[str] = "model_training_runs"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost play-success model from engineered play features."
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
        default=DEFAULT_FEATURE_TABLE,
        help="Source feature table name.",
    )
    parser.add_argument(
        "--predictions-table",
        type=str,
        default=DEFAULT_PREDICTIONS_TABLE,
        help="Destination table for per-play predicted success probability.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model identifier used in artifact names and prediction table.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SEASONS,
        help="Training seasons (default: 2020 2021 2022).",
    )
    parser.add_argument(
        "--validation-season",
        type=int,
        default=DEFAULT_VALIDATION_SEASON,
        help="Validation season (default: 2023).",
    )
    parser.add_argument(
        "--test-season",
        type=int,
        default=DEFAULT_TEST_SEASON,
        help="Test season (default: 2024).",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["epa_positive", "target_success"],
        default="epa_positive",
        help="Training target. epa_positive uses 1(target_epa > 0).",
    )
    parser.add_argument(
        "--replace-existing-predictions",
        action="store_true",
        help="Replace existing predictions for this model and selected seasons.",
    )
    return parser.parse_args()


def validate_split_config(
    train_seasons: Sequence[int], validation_season: int, test_season: int
) -> list[int]:
    """Validate season split configuration."""
    train_unique = sorted(set(train_seasons))
    if validation_season in train_unique or test_season in train_unique:
        raise ValueError("Validation/test seasons must not overlap with training seasons.")
    if validation_season == test_season:
        raise ValueError("Validation season and test season must differ.")
    all_seasons = train_unique + [validation_season, test_season]
    if len(set(all_seasons)) != len(all_seasons):
        raise ValueError("Duplicate seasons detected in split config.")
    return all_seasons


def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite with WAL settings."""
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    return connection


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists."""
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def load_feature_columns(connection: sqlite3.Connection, feature_table: str) -> list[str]:
    """Load ordered feature columns (`feat_*`) from source table."""
    if not table_exists(connection, feature_table):
        raise ValueError(f"Feature table not found: {feature_table}")
    schema = connection.execute(f"PRAGMA table_info('{feature_table}');").fetchall()
    feature_columns = [row[1] for row in schema if row[1].startswith("feat_")]
    if len(feature_columns) < 40:
        raise ValueError(
            f"Expected >= 40 feature columns in {feature_table}, found {len(feature_columns)}."
        )
    return feature_columns


def load_training_frame(
    connection: sqlite3.Connection,
    feature_table: str,
    all_seasons: Sequence[int],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Load keyed rows for requested seasons and selected columns."""
    placeholders = ",".join(["?"] * len(all_seasons))
    cols = [
        "season",
        "game_id",
        "play_id",
        "target_epa",
        "target_success",
    ] + list(feature_columns)
    query = f"""
        SELECT {", ".join(cols)}
        FROM {feature_table}
        WHERE season IN ({placeholders})
        ORDER BY season, game_id, play_id;
    """
    frame = pd.read_sql_query(query, connection, params=list(all_seasons))
    if frame.empty:
        raise ValueError("No rows loaded from feature table for requested seasons.")
    return frame


def build_target(frame: pd.DataFrame, target_mode: str) -> pd.Series:
    """Construct binary target labels."""
    if target_mode == "epa_positive":
        return (pd.to_numeric(frame["target_epa"], errors="coerce") > 0).astype(int)
    return (
        pd.to_numeric(frame["target_success"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )


def prepare_feature_matrix(
    frame: pd.DataFrame, feature_columns: Sequence[str], impute_values: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Cast features to numeric and fill missing values."""
    matrix = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if impute_values is None:
        impute_values = matrix.median()
    matrix = matrix.fillna(impute_values)
    return matrix, impute_values


def evaluate_binary_classification(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Compute classification metrics for probability model."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "rows": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def top_feature_importance(
    model: xgb.XGBClassifier, feature_columns: Sequence[str], top_n: int = 25
) -> list[dict[str, float | str]]:
    """Return top features by gain importance."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    rows: list[dict[str, float | str]] = []
    for feature_name in feature_columns:
        score = float(scores.get(feature_name, 0.0))
        rows.append({"feature": feature_name, "gain_importance": score})
    rows = sorted(rows, key=lambda row: row["gain_importance"], reverse=True)
    return rows[:top_n]


def ensure_model_run_table(connection: sqlite3.Connection) -> None:
    """Create model run metadata table if missing."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MODEL_RUN_TABLE} (
            run_id TEXT PRIMARY KEY,
            run_ts TEXT NOT NULL,
            model_name TEXT NOT NULL,
            feature_table TEXT NOT NULL,
            predictions_table TEXT NOT NULL,
            train_seasons TEXT NOT NULL,
            validation_season INTEGER NOT NULL,
            test_season INTEGER NOT NULL,
            target_mode TEXT NOT NULL,
            rows_scored INTEGER NOT NULL,
            feature_count INTEGER NOT NULL,
            validation_log_loss REAL,
            validation_roc_auc REAL,
            test_log_loss REAL,
            test_roc_auc REAL,
            status TEXT NOT NULL,
            message TEXT
        );
        """
    )


def ensure_predictions_table(connection: sqlite3.Connection, predictions_table: str) -> None:
    """Create predictions table if missing."""
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {predictions_table} (
            model_name TEXT NOT NULL,
            model_run_ts TEXT NOT NULL,
            split_name TEXT NOT NULL,
            season INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            play_id INTEGER NOT NULL,
            target_success_epa INTEGER NOT NULL,
            target_epa REAL,
            target_success REAL,
            pred_success_prob REAL NOT NULL,
            residual_success REAL NOT NULL
        );
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{predictions_table}_model_season
        ON {predictions_table} (model_name, season);
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{predictions_table}_play_key
        ON {predictions_table} (season, game_id, play_id);
        """
    )


def delete_existing_predictions(
    connection: sqlite3.Connection,
    predictions_table: str,
    model_name: str,
    seasons: Sequence[int],
) -> int:
    """Delete existing predictions for model and seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    cursor = connection.execute(
        f"""
        DELETE FROM {predictions_table}
        WHERE model_name = ?
          AND season IN ({placeholders});
        """,
        [model_name] + list(seasons),
    )
    return int(cursor.rowcount)


def count_existing_predictions(
    connection: sqlite3.Connection,
    predictions_table: str,
    model_name: str,
    seasons: Sequence[int],
) -> int:
    """Count existing predictions for model and seasons."""
    placeholders = ",".join(["?"] * len(seasons))
    row = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {predictions_table}
        WHERE model_name = ?
          AND season IN ({placeholders});
        """,
        [model_name] + list(seasons),
    ).fetchone()
    return int(row[0]) if row else 0


def add_split_labels(
    frame: pd.DataFrame,
    train_seasons: Sequence[int],
    validation_season: int,
    test_season: int,
) -> pd.Series:
    """Assign split names per row."""
    split = np.where(
        frame["season"].isin(train_seasons),
        "train",
        np.where(frame["season"] == validation_season, "validation", "test"),
    )
    split_series = pd.Series(split, index=frame.index, dtype="object")
    unknown = ~split_series.isin(["train", "validation", "test"])
    if unknown.any():
        raise ValueError("Unknown split assignment encountered.")
    if (frame.loc[split_series == "test", "season"] != test_season).any():
        raise ValueError("Test split assignment mismatch.")
    return split_series


def main() -> None:
    """Train model, save artifacts, and persist predictions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {args.db_path}")

    all_seasons = validate_split_config(
        train_seasons=args.train_seasons,
        validation_season=args.validation_season,
        test_season=args.test_season,
    )
    run_ts = datetime.now(tz=timezone.utc).isoformat()
    run_ts_file = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    connection = connect_sqlite(args.db_path)
    try:
        ensure_model_run_table(connection)
        ensure_predictions_table(connection, args.predictions_table)

        feature_columns = load_feature_columns(connection, args.feature_table)
        raw_frame = load_training_frame(
            connection=connection,
            feature_table=args.feature_table,
            all_seasons=all_seasons,
            feature_columns=feature_columns,
        )
        raw_frame["target_success_epa"] = build_target(raw_frame, args.target_mode)
        raw_frame["split_name"] = add_split_labels(
            raw_frame,
            train_seasons=args.train_seasons,
            validation_season=args.validation_season,
            test_season=args.test_season,
        )

        train_df = raw_frame[raw_frame["split_name"] == "train"].copy()
        val_df = raw_frame[raw_frame["split_name"] == "validation"].copy()
        test_df = raw_frame[raw_frame["split_name"] == "test"].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("One or more splits are empty. Check season configuration.")

        LOGGER.info(
            "Split rows | train=%s | validation=%s | test=%s",
            f"{len(train_df):,}",
            f"{len(val_df):,}",
            f"{len(test_df):,}",
        )

        X_train, impute_values = prepare_feature_matrix(train_df, feature_columns)
        y_train = train_df["target_success_epa"].astype(int)
        X_val, _ = prepare_feature_matrix(val_df, feature_columns, impute_values)
        y_val = val_df["target_success_epa"].astype(int)
        X_test, _ = prepare_feature_matrix(test_df, feature_columns, impute_values)
        y_test = test_df["target_success_epa"].astype(int)

        # Football rationale:
        # The model needs nonlinear interactions between situation context and
        # usage/scheme features. Gradient-boosted trees capture that cleanly.
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        val_metrics = evaluate_binary_classification(y_true=y_val, y_prob=val_prob)
        test_metrics = evaluate_binary_classification(y_true=y_test, y_prob=test_prob)

        LOGGER.info("Validation metrics: %s", val_metrics)
        LOGGER.info("Test metrics: %s", test_metrics)

        X_all, _ = prepare_feature_matrix(raw_frame, feature_columns, impute_values)
        raw_frame["pred_success_prob"] = model.predict_proba(X_all)[:, 1]
        raw_frame["residual_success"] = (
            raw_frame["target_success_epa"] - raw_frame["pred_success_prob"]
        )
        raw_frame["model_name"] = args.model_name
        raw_frame["model_run_ts"] = run_ts

        predictions = raw_frame[
            [
                "model_name",
                "model_run_ts",
                "split_name",
                "season",
                "game_id",
                "play_id",
                "target_success_epa",
                "target_epa",
                "target_success",
                "pred_success_prob",
                "residual_success",
            ]
        ].copy()
        predictions = predictions.where(pd.notna(predictions), None)

        with connection:
            existing_count = count_existing_predictions(
                connection=connection,
                predictions_table=args.predictions_table,
                model_name=args.model_name,
                seasons=all_seasons,
            )
            if existing_count > 0:
                if not args.replace_existing_predictions:
                    raise ValueError(
                        f"{args.predictions_table} already has {existing_count:,} rows for "
                        f"model={args.model_name} seasons={all_seasons}. "
                        "Use --replace-existing-predictions to overwrite."
                    )
                deleted_rows = delete_existing_predictions(
                    connection=connection,
                    predictions_table=args.predictions_table,
                    model_name=args.model_name,
                    seasons=all_seasons,
                )
                LOGGER.info(
                    "Deleted %s existing rows from %s for model %s.",
                    f"{deleted_rows:,}",
                    args.predictions_table,
                    args.model_name,
                )

            predictions.to_sql(
                name=args.predictions_table,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=2_000,
            )

            connection.execute(
                f"""
                INSERT INTO {MODEL_RUN_TABLE} (
                    run_id,
                    run_ts,
                    model_name,
                    feature_table,
                    predictions_table,
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
                    status,
                    message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(uuid4()),
                    run_ts,
                    args.model_name,
                    args.feature_table,
                    args.predictions_table,
                    ",".join(str(season) for season in sorted(args.train_seasons)),
                    args.validation_season,
                    args.test_season,
                    args.target_mode,
                    len(predictions),
                    len(feature_columns),
                    val_metrics["log_loss"],
                    val_metrics["roc_auc"],
                    test_metrics["log_loss"],
                    test_metrics["roc_auc"],
                    "SUCCESS",
                    "Model trained and predictions written.",
                ),
            )

        args.model_dir.mkdir(parents=True, exist_ok=True)
        model_artifact_path = args.model_dir / f"{args.model_name}_{run_ts_file}.joblib"
        metrics_path = args.model_dir / f"{args.model_name}_{run_ts_file}_metrics.json"

        model_payload = {
            "model": model,
            "feature_columns": list(feature_columns),
            "impute_values": impute_values.to_dict(),
            "target_mode": args.target_mode,
            "train_seasons": sorted(args.train_seasons),
            "validation_season": args.validation_season,
            "test_season": args.test_season,
            "model_name": args.model_name,
            "run_ts": run_ts,
        }
        joblib.dump(model_payload, model_artifact_path)

        metrics_payload = {
            "model_name": args.model_name,
            "run_ts": run_ts,
            "target_mode": args.target_mode,
            "feature_count": len(feature_columns),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_feature_importance_gain": top_feature_importance(
                model=model,
                feature_columns=feature_columns,
                top_n=25,
            ),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        LOGGER.info("Training complete.")
        LOGGER.info("Model artifact: %s", model_artifact_path)
        LOGGER.info("Metrics artifact: %s", metrics_path)
        LOGGER.info(
            "Predictions written: %s rows to %s",
            f"{len(predictions):,}",
            args.predictions_table,
        )
    except Exception as exc:
        with connection:
            ensure_model_run_table(connection)
            connection.execute(
                f"""
                INSERT INTO {MODEL_RUN_TABLE} (
                    run_id,
                    run_ts,
                    model_name,
                    feature_table,
                    predictions_table,
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
                    status,
                    message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(uuid4()),
                    datetime.now(tz=timezone.utc).isoformat(),
                    args.model_name,
                    args.feature_table,
                    args.predictions_table,
                    ",".join(str(season) for season in sorted(args.train_seasons)),
                    args.validation_season,
                    args.test_season,
                    args.target_mode,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    "FAILED",
                    str(exc),
                ),
            )
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
