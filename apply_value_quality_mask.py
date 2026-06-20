from pathlib import Path
import numpy as np
import pandas as pd


# =====================
# CONFIG
# =====================

TIMESTAMP_COL = "timestamp"

INPUT_FILES = {
    "train": "model_data/train_filtered.csv",
    "val": "model_data/val_filtered.csv",
    "test": "model_data/test_filtered.csv",
}

OUTPUT_DIR = Path("model_data_qc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_REASONABLE_VALUE = 1500
MIN_REASONABLE_VALUE = 0
REPLACE_INVALID_WITH_NAN = True
SAVE_VALID_MASKS = True
SAVE_INVALID_POINTS = True
DIAGNOSTIC_THRESHOLDS = [1000, 1500, 2000, 3000]


# =====================
# FUNCTIONS
# =====================

def load_split(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]

    if len(sensors) == 0:
        raise ValueError(f"No sensor columns found in {path}")

    for sensor in sensors:
        df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

    return df, sensors


def build_valid_mask(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    X = df[sensors]

    valid = pd.DataFrame(
        True,
        index=df.index,
        columns=sensors,
    )

    valid &= X.notna()
    valid &= X >= MIN_REASONABLE_VALUE
    valid &= X <= MAX_REASONABLE_VALUE

    return valid


def make_qc_dataframe(
    df: pd.DataFrame,
    sensors: list[str],
    valid_mask: pd.DataFrame,
) -> pd.DataFrame:
    qc_df = df.copy()

    if REPLACE_INVALID_WITH_NAN:
        for sensor in sensors:
            qc_df.loc[~valid_mask[sensor], sensor] = np.nan

    return qc_df


def summarize_split(
    split_name: str,
    df: pd.DataFrame,
    sensors: list[str],
    valid_mask: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    X = df[sensors]

    for sensor in sensors:
        values = X[sensor]

        n_total = len(values)
        n_nan_original = int(values.isna().sum())

        invalid_low = values < MIN_REASONABLE_VALUE
        invalid_high = values > MAX_REASONABLE_VALUE
        invalid_any = ~valid_mask[sensor]

        row = {
            "split": split_name,
            "sensor": sensor,
            "n_total": n_total,
            "n_nan_original": n_nan_original,
            "n_invalid_total": int(invalid_any.sum()),
            "invalid_rate": float(invalid_any.mean()),
            "n_below_min": int(invalid_low.sum()),
            "n_above_max": int(invalid_high.sum()),
            "min_value": float(values.min(skipna=True)) if values.notna().any() else np.nan,
            "max_value": float(values.max(skipna=True)) if values.notna().any() else np.nan,
            "mean_value": float(values.mean(skipna=True)) if values.notna().any() else np.nan,
            "median_value": float(values.median(skipna=True)) if values.notna().any() else np.nan,
            "p99_value": float(values.quantile(0.99)) if values.notna().any() else np.nan,
            "p999_value": float(values.quantile(0.999)) if values.notna().any() else np.nan,
        }

        for threshold in DIAGNOSTIC_THRESHOLDS:
            row[f"n_gt_{threshold}"] = int((values > threshold).sum())
            row[f"rate_gt_{threshold}"] = float((values > threshold).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def build_invalid_points_table(
    split_name: str,
    df: pd.DataFrame,
    sensors: list[str],
    valid_mask: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    X = df[sensors]

    invalid_positions = ~valid_mask

    for sensor in sensors:
        invalid_idx = np.where(invalid_positions[sensor].to_numpy())[0]

        for idx in invalid_idx:
            value = X[sensor].iloc[idx]

            if pd.isna(value):
                reason = "original_nan"
            elif value < MIN_REASONABLE_VALUE:
                reason = "below_min"
            elif value > MAX_REASONABLE_VALUE:
                reason = "above_max"
            else:
                reason = "unknown_invalid"

            rows.append({
                "split": split_name,
                "timestamp": df[TIMESTAMP_COL].iloc[idx],
                "row_idx": int(idx),
                "sensor": sensor,
                "value": value,
                "reason": reason,
            })

    return pd.DataFrame(rows)


def save_outputs(
    split_name: str,
    qc_df: pd.DataFrame,
    valid_mask: pd.DataFrame,
    invalid_points: pd.DataFrame,
) -> None:
    qc_output = OUTPUT_DIR / f"{split_name}_filtered_qc.csv"
    qc_df.to_csv(qc_output, index=False)

    if SAVE_VALID_MASKS:
        mask_output = OUTPUT_DIR / f"{split_name}_valid_value_mask.csv"
        valid_mask.to_csv(mask_output, index=False)

    if SAVE_INVALID_POINTS:
        invalid_output = OUTPUT_DIR / f"{split_name}_invalid_points.csv"
        invalid_points.to_csv(invalid_output, index=False)

    print(f"Saved QC data: {qc_output}")


def main() -> None:
    all_summary_rows = []
    all_invalid_points = []

    for split_name, input_path in INPUT_FILES.items():
        print(f"\nProcessing split: {split_name}")
        print(f"Input: {input_path}")

        df, sensors = load_split(input_path)

        valid_mask = build_valid_mask(
            df=df,
            sensors=sensors,
        )

        qc_df = make_qc_dataframe(
            df=df,
            sensors=sensors,
            valid_mask=valid_mask,
        )

        summary = summarize_split(
            split_name=split_name,
            df=df,
            sensors=sensors,
            valid_mask=valid_mask,
        )

        invalid_points = build_invalid_points_table(
            split_name=split_name,
            df=df,
            sensors=sensors,
            valid_mask=valid_mask,
        )

        save_outputs(
            split_name=split_name,
            qc_df=qc_df,
            valid_mask=valid_mask,
            invalid_points=invalid_points,
        )

        all_summary_rows.append(summary)
        all_invalid_points.append(invalid_points)

        n_values = len(df) * len(sensors)
        n_invalid = int((~valid_mask).to_numpy().sum())

        print(f"Rows: {len(df)}")
        print(f"Sensors: {len(sensors)}")
        print(f"Total values: {n_values}")
        print(f"Invalid values: {n_invalid}")
        print(f"Invalid rate: {n_invalid / n_values:.6f}")

        print("\nTop sensors by invalid values:")
        print(
            summary.sort_values("n_invalid_total", ascending=False)
            .head(10)[
                [
                    "sensor",
                    "n_invalid_total",
                    "invalid_rate",
                    "max_value",
                    "p99_value",
                    "p999_value",
                    f"n_gt_{MAX_REASONABLE_VALUE}",
                ]
            ]
        )

    report = pd.concat(all_summary_rows, ignore_index=True)
    invalid_all = pd.concat(all_invalid_points, ignore_index=True)

    report_output = OUTPUT_DIR / "value_outlier_report_by_sensor.csv"
    invalid_all_output = OUTPUT_DIR / "invalid_points_all.csv"

    report.to_csv(report_output, index=False)
    invalid_all.to_csv(invalid_all_output, index=False)

    split_summary = (
        report
        .groupby("split")
        .agg(
            n_sensors=("sensor", "nunique"),
            n_total_values=("n_total", "sum"),
            n_invalid_total=("n_invalid_total", "sum"),
            n_original_nan=("n_nan_original", "sum"),
            n_above_max=("n_above_max", "sum"),
            n_below_min=("n_below_min", "sum"),
            max_value=("max_value", "max"),
        )
        .reset_index()
    )

    split_summary["invalid_rate"] = (
        split_summary["n_invalid_total"] / split_summary["n_total_values"]
    )

    split_summary_output = OUTPUT_DIR / "value_outlier_report_by_split.csv"
    split_summary.to_csv(split_summary_output, index=False)

    print("\nFinal split summary")
    print("-------------------")
    print(split_summary)

    print(f"\nSaved report by sensor to: {report_output}")
    print(f"Saved report by split to:  {split_summary_output}")
    print(f"Saved invalid points to:   {invalid_all_output}")
    print(f"\nQC files saved in:         {OUTPUT_DIR}")


if __name__ == "__main__":
    main()