# NFL Player Valuation Model

End-to-end project for NFL player valuation with play-level context, expected
outcomes modeling, and position-level marginal value analysis.

## Step 1: Environment Setup + First Season Pull

### 1) Create and activate a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Why Python 3.11: it is mature and well-supported across the analytics stack
(`nfl_data_py`, `xgboost`, `scikit-learn`, `streamlit`) with fewer dependency
edge cases than the newest interpreter versions.

Why `pandas==1.5.3` and `numpy==1.26.4`: `nfl_data_py==0.3.3` currently
requires `pandas<2.0` and `numpy<2.0`, so these pins keep the environment
fully resolver-safe and reproducible.

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Pull one season of play-by-play data (example: 2024)

```bash
python scripts/pull_pbp_data.py --season 2024 --output-dir data/raw --file-format parquet
```

### 4) Expected successful output

You should see log lines including:

- `Downloading play-by-play data for season 2024...`
- `Season pulled: 2024`
- `Rows pulled: ...`
- `Unique games: ...`
- `Saved raw dataset to: data/raw/pbp_2024.parquet`

If successful, the file `data/raw/pbp_2024.parquet` will exist.

## Step 2: Clean + Load Into SQLite

### 1) Load season parquet into SQLite

```bash
python scripts/load_pbp_into_sqlite.py \
  --seasons 2024 \
  --input-dir data/raw \
  --db-path data/nfl_valuation.db
```

If you are re-running the same season:

```bash
python scripts/load_pbp_into_sqlite.py \
  --seasons 2024 \
  --input-dir data/raw \
  --db-path data/nfl_valuation.db \
  --replace-existing-season
```

### 2) Expected successful output

You should see log lines including:

- `Loading season 2024 into data/nfl_valuation.db`
- `Loaded 1 season(s).`
- `Total rows in loaded seasons: ...`
- `Model-play share: ...`
- `Season 2024 | rows=... | model_rows=... | replaced_existing=...`

### 3) Validate DB contents

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/nfl_valuation.db")
row = conn.execute(
    """
    SELECT season, COUNT(*) AS plays, SUM(is_model_play) AS model_plays
    FROM pbp_clean
    GROUP BY season
    ORDER BY season;
    """
).fetchall()
runs = conn.execute(
    "SELECT COUNT(*) FROM ingestion_runs WHERE status = 'SUCCESS';"
).fetchone()[0]
print("season_rows=", row)
print("successful_ingestion_runs=", runs)
conn.close()
PY
```

## Step 8: Streamlit Dashboard for Scouts & GM Views

### 1) Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

### 2) Expected dashboard sections

- `Overview`: league snapshot by season, team market leaders, and position value distributions.
- `Rankings`: filter by season/position/team, sort by value metric, and download top-player cuts.
- `Player Explorer`: inspect season-over-season value/tier trends for any player.
- `Player Compare`: side-by-side comparison with head-to-head radar metrics.
- `Validation`: review Pro Bowl alignment and benchmark-correlation diagnostics.
- `Model`: check training runs, holdout metrics, and top feature importance.

### 3) Confirm dashboard is using your SQLite outputs

Use the sidebar:

- `SQLite DB Path` should point to `data/nfl_valuation.db`.
- `Model Name` should include `xgb_play_success_epa`.
- `Refresh Data Cache` can be used after pipeline writes to SQLite.

If data is connected correctly:

- Rankings table should populate immediately.
- Validation heatmap should render for seasons 2020-2024.
- Model tab should show validation/test AUC and feature-importance chart.

## Step 7: Validate Against Pro Bowl + PFF-Style Benchmarks

### 1) Build validation outputs

```bash
python scripts/validate_valuation_benchmarks.py \
  --db-path data/nfl_valuation.db \
  --valuation-table player_valuation \
  --model-name xgb_play_success_epa \
  --seasons 2020 2021 2022 2023 2024 \
  --summary-table valuation_validation_summary \
  --player-table valuation_validation_player \
  --replace-existing-season
```

Optional external benchmark (if you have PFF-like grades in CSV):

```bash
python scripts/validate_valuation_benchmarks.py \
  --db-path data/nfl_valuation.db \
  --valuation-table player_valuation \
  --model-name xgb_play_success_epa \
  --seasons 2020 2021 2022 2023 2024 \
  --summary-table valuation_validation_summary \
  --player-table valuation_validation_player \
  --external-benchmark-csv data/external/pff_like_benchmarks.csv \
  --replace-existing-season
```

External CSV format:

- Required columns: `season`, `player_id`, `position_group`, and either `benchmark_score` or `pff_grade`.

### 2) Expected successful output

You should see log lines including:

- `Loaded ... valuation rows.`
- `Using Wikipedia Pro Bowl page: ...`
- `Pro Bowl mapping | entries=... | mapped=... | ambiguous=... | unresolved=...`
- `Validation summary rows written: ... to valuation_validation_summary`
- `Validation player rows written: ... to valuation_validation_player`

### 3) Validate quality of validation outputs

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/nfl_valuation.db")
summary_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM valuation_validation_summary
    WHERE model_name='xgb_play_success_epa'
      AND season BETWEEN 2020 AND 2024;
    """
).fetchone()[0]

player_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM valuation_validation_player
    WHERE model_name='xgb_play_success_epa'
      AND season BETWEEN 2020 AND 2024;
    """
).fetchone()[0]

latest_run = conn.execute(
    """
    SELECT status, pro_bowl_entries, pro_bowl_mapped, pro_bowl_ambiguous, pro_bowl_unresolved
    FROM valuation_validation_runs
    ORDER BY run_ts DESC
    LIMIT 1;
    """
).fetchone()

sample_metrics = conn.execute(
    """
    SELECT season, position_group, ROUND(precision_at_k,3), ROUND(spearman_with_benchmark,3)
    FROM valuation_validation_summary
    WHERE model_name='xgb_play_success_epa'
    ORDER BY season, position_group
    LIMIT 8;
    """
).fetchall()

conn.close()

assert summary_count == 20, f"Expected 20 summary rows, got {summary_count}"
assert player_count > 2000, f"Expected >2000 player rows, got {player_count}"
assert latest_run is not None and latest_run[0] == "SUCCESS", "Latest validation run failed"
assert latest_run[2] > 0, "No Pro Bowl entries were mapped to player IDs"

print("PASS")
print("summary_count=", summary_count)
print("player_count=", player_count)
print("latest_run=", latest_run)
print("sample_metrics=", sample_metrics)
PY
```

## Step 6: Build Player Marginal Value + Position Tiers

### 1) Build valuation tables from model residuals

```bash
python scripts/build_player_valuation.py \
  --db-path data/nfl_valuation.db \
  --feature-table pbp_features \
  --predictions-table play_success_predictions \
  --model-name xgb_play_success_epa \
  --seasons 2020 2021 2022 2023 2024 \
  --valuation-table player_valuation \
  --tiers-table player_valuation_tiers \
  --replace-existing-season
```

### 2) Expected successful output

You should see log lines including:

- `Loaded ... play rows for valuation.`
- `Player valuation rows written: ... to player_valuation`
- `Player tier rows written: ... to player_valuation_tiers`
- `Context adjustment groups fitted with linear model: .../...`

### 3) Validate valuation and tiers

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/nfl_valuation.db")
season_pos_counts = conn.execute(
    """
    SELECT season, position_group, COUNT(*) AS n_players, SUM(is_qualified) AS qualified
    FROM player_valuation
    WHERE model_name = 'xgb_play_success_epa'
    GROUP BY season, position_group
    ORDER BY season, position_group;
    """
).fetchall()

tier_counts = conn.execute(
    """
    SELECT season, position_group, tier_label, COUNT(*) AS n
    FROM player_valuation_tiers
    WHERE model_name = 'xgb_play_success_epa'
      AND season = 2024
      AND is_qualified = 1
    GROUP BY season, position_group, tier_label
    ORDER BY position_group, tier_label;
    """
).fetchall()
conn.close()

positions = {(season, pos) for season, pos, _, _ in season_pos_counts}
for season in [2020, 2021, 2022, 2023, 2024]:
    for pos in ["QB", "RB", "WR", "TE"]:
        assert (season, pos) in positions, f"Missing valuation rows for {season} {pos}"

assert len(season_pos_counts) == 20, f"Expected 20 season-position rows, got {len(season_pos_counts)}"
assert len(tier_counts) > 0, "No qualified tier rows found for 2024"

print("PASS")
print("season_pos_counts=", season_pos_counts[:8], "... total", len(season_pos_counts))
print("sample_2024_tier_counts=", tier_counts[:10])
PY
```

## Step 5: Train XGBoost Play Success Model

### 1) Train model and write per-play expected-success probabilities

```bash
python scripts/train_play_success_model.py \
  --db-path data/nfl_valuation.db \
  --feature-table pbp_features \
  --predictions-table play_success_predictions \
  --model-name xgb_play_success_epa \
  --model-dir models \
  --train-seasons 2020 2021 2022 \
  --validation-season 2023 \
  --test-season 2024 \
  --target-mode epa_positive \
  --replace-existing-predictions
```

### 2) Expected successful output

You should see log lines including:

- `Split rows | train=... | validation=... | test=...`
- `Validation metrics: ...`
- `Test metrics: ...`
- `Training complete.`
- `Predictions written: ... rows to play_success_predictions`

### 3) Validate model outputs and prediction table

```bash
python - <<'PY'
import sqlite3
from pathlib import Path

conn = sqlite3.connect("data/nfl_valuation.db")
split_rows = conn.execute(
    """
    SELECT split_name, COUNT(*) AS n
    FROM play_success_predictions
    WHERE model_name = 'xgb_play_success_epa'
    GROUP BY split_name
    ORDER BY split_name;
    """
).fetchall()

latest = conn.execute(
    """
    SELECT status, validation_roc_auc, test_roc_auc
    FROM model_training_runs
    WHERE model_name = 'xgb_play_success_epa'
    ORDER BY run_ts DESC
    LIMIT 1;
    """
).fetchone()
conn.close()

metrics_files = sorted(Path("models").glob("xgb_play_success_epa_*_metrics.json"))
assert metrics_files, "Missing metrics artifact in models/"
assert latest is not None and latest[0] == "SUCCESS", "Latest model run failed"
assert latest[1] > 0.70, f"Validation AUC too low: {latest[1]}"
assert latest[2] > 0.70, f"Test AUC too low: {latest[2]}"
assert sum(n for _, n in split_rows) > 150000, "Prediction row count too low"

print("PASS")
print("split_rows=", split_rows)
print("latest_run=", latest)
print("latest_metrics_file=", metrics_files[-1])
PY
```

## Step 4: Engineer 40+ Contextual Features

### 1) Build play-level feature table in SQLite

```bash
python scripts/engineer_play_features.py \
  --db-path data/nfl_valuation.db \
  --seasons 2020 2021 2022 2023 2024 \
  --feature-table pbp_features \
  --replace-existing-season
```

### 2) Expected successful output

You should see log lines including:

- `Loaded 175,387 model plays from pbp_clean.`
- `Engineered ... contextual features.`
- `Feature engineering complete.`
- `Rows written: 175,387`
- `SQLite table: pbp_features`

### 3) Validate feature table quality

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/nfl_valuation.db")
season_rows = conn.execute(
    """
    SELECT season, COUNT(*) AS rows
    FROM pbp_features
    GROUP BY season
    ORDER BY season;
    """
).fetchall()

feature_columns = conn.execute(
    "PRAGMA table_info('pbp_features');"
).fetchall()
feature_count = sum(1 for c in feature_columns if c[1].startswith("feat_"))

last_run = conn.execute(
    """
    SELECT status, rows_written, feature_count
    FROM feature_build_runs
    ORDER BY run_ts DESC
    LIMIT 1;
    """
).fetchone()

conn.close()

assert len(season_rows) == 5, f"Expected 5 seasons, got {len(season_rows)}"
assert feature_count >= 40, f"Expected >= 40 features, got {feature_count}"
assert last_run is not None and last_run[0] == "SUCCESS", "Last feature run failed"
assert last_run[1] > 150000, f"Unexpected row count: {last_run[1]}"

print("PASS")
print("season_rows=", season_rows)
print("feature_count=", feature_count)
print("last_run=", last_run)
PY
```

## Step 3: Backfill 5 Seasons (2020-2024)

### 1) Pull and load full five-season window

```bash
python scripts/backfill_five_seasons.py \
  --start-season 2020 \
  --end-season 2024 \
  --input-dir data/raw \
  --db-path data/nfl_valuation.db
```

### 2) Expected successful output

You should see log lines including:

- `Backfill window: [2020, 2021, 2022, 2023, 2024]`
- `Pulling raw PBP for season 2020` ... through `2024`
- `Loading season 2020 into SQLite` ... through `2024`
- `Backfill complete for 5 seasons.`
- `SQLite total rows across backfilled seasons: ...`

### 3) Validate all 5 seasons exist in SQLite

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/nfl_valuation.db")
rows = conn.execute(
    """
    SELECT season, COUNT(*) AS plays, SUM(is_model_play) AS model_plays
    FROM pbp_clean
    WHERE season BETWEEN 2020 AND 2024
    GROUP BY season
    ORDER BY season;
    """
).fetchall()
print("season_rows=", rows)
print("season_count=", len(rows))
conn.close()
PY
```
