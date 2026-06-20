#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT = "combined_sensors_2022-10-01_2022-12-31.csv"
OUTDIR = Path("profiling_outputs")
OUTDIR.mkdir(exist_ok=True)

TIMESTAMP_COL = "timestamp"

# Przy danych 10-minutowych:
SAMPLES_PER_HOUR = 6
SAMPLES_PER_DAY = 144


def longest_true_run(mask) -> int:
    """Length of the longest consecutive True run."""
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0:
        return 0

    padded = np.r_[False, arr, False]
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return 0

    return int((ends - starts).max())


def true_run_lengths(mask):
    """Return all consecutive True run lengths."""
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0:
        return []

    padded = np.r_[False, arr, False]
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return list((ends - starts).astype(int))


def check_timestamp_grid(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df[TIMESTAMP_COL])
    diffs = ts.diff().dropna()

    grid_report = pd.DataFrame({
        "metric": [
            "start",
            "end",
            "n_rows",
            "most_common_step",
            "n_unique_steps",
            "n_duplicate_timestamps",
            "n_missing_timestamps_on_10min_grid",
        ],
        "value": [
            str(ts.min()),
            str(ts.max()),
            len(ts),
            str(diffs.mode().iloc[0]) if len(diffs) else None,
            diffs.nunique(),
            ts.duplicated().sum(),
            len(pd.date_range(ts.min(), ts.max(), freq="10min").difference(ts)),
        ],
    })

    return grid_report


def classify_sensor(row):
    if row["missing_rate"] == 1.0:
        return "all_nan"
    if row["zero_rate"] == 1.0:
        return "all_zero"
    if row["missing_rate"] > 0.20:
        return "high_missing"
    if row["zero_rate"] > 0.90:
        return "mostly_zero_90"
    if row["zero_rate"] > 0.80:
        return "mostly_zero_80"
    if row["n_unique"] <= 3:
        return "low_variability"
    if row["longest_zero_run"] >= SAMPLES_PER_DAY:
        return "long_zero_run_ge_1day"
    return "usable"


def build_sensor_profile(X: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sensor in X.columns:
        x = X[sensor]

        missing = x.isna()
        zero = x.eq(0).fillna(False)

        rows.append({
            "sensor": sensor,
            "n": len(x),
            "n_missing": int(missing.sum()),
            "missing_rate": float(missing.mean()),
            "n_zero": int(zero.sum()),
            "zero_rate": float(zero.mean()),
            "n_unique": int(x.nunique(dropna=True)),
            "mean": float(x.mean(skipna=True)) if x.notna().any() else np.nan,
            "median": float(x.median(skipna=True)) if x.notna().any() else np.nan,
            "std": float(x.std(skipna=True)) if x.notna().any() else np.nan,
            "min": float(x.min(skipna=True)) if x.notna().any() else np.nan,
            "p01": float(x.quantile(0.01)) if x.notna().any() else np.nan,
            "p05": float(x.quantile(0.05)) if x.notna().any() else np.nan,
            "p95": float(x.quantile(0.95)) if x.notna().any() else np.nan,
            "p99": float(x.quantile(0.99)) if x.notna().any() else np.nan,
            "max": float(x.max(skipna=True)) if x.notna().any() else np.nan,
            "longest_nan_run": longest_true_run(missing),
            "longest_zero_run": longest_true_run(zero),
            "n_nan_runs": len(true_run_lengths(missing)),
            "n_zero_runs": len(true_run_lengths(zero)),
            "n_zero_runs_ge_1h": sum(l >= SAMPLES_PER_HOUR for l in true_run_lengths(zero)),
            "n_zero_runs_ge_6h": sum(l >= 6 * SAMPLES_PER_HOUR for l in true_run_lengths(zero)),
            "n_zero_runs_ge_1day": sum(l >= SAMPLES_PER_DAY for l in true_run_lengths(zero)),
        })

    profile = pd.DataFrame(rows)
    profile["status"] = profile.apply(classify_sensor, axis=1)

    return profile.sort_values(
        ["status", "missing_rate", "zero_rate", "longest_zero_run"],
        ascending=[True, False, False, False],
    )


def build_gap_table(X: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    condition:
      - 'nan'
      - 'zero'
    """
    rows = []

    for sensor in X.columns:
        x = X[sensor]

        if condition == "nan":
            mask = x.isna()
        elif condition == "zero":
            mask = x.eq(0).fillna(False)
        else:
            raise ValueError("condition must be 'nan' or 'zero'")

        lengths = true_run_lengths(mask)

        for length in lengths:
            rows.append({
                "sensor": sensor,
                "condition": condition,
                "length": int(length),
                "duration_hours": length / SAMPLES_PER_HOUR,
                "duration_days": length / SAMPLES_PER_DAY,
            })

    return pd.DataFrame(rows)


def plot_missing_by_time(df: pd.DataFrame, X: pd.DataFrame):
    tmp = pd.DataFrame({
        "timestamp": df[TIMESTAMP_COL],
        "missing_fraction": X.isna().mean(axis=1),
        "zero_fraction": X.eq(0).mean(axis=1),
    })

    plt.figure(figsize=(14, 4))
    plt.plot(tmp["timestamp"], tmp["missing_fraction"], label="NaN fraction")
    plt.plot(tmp["timestamp"], tmp["zero_fraction"], label="Zero fraction")
    plt.legend()
    plt.title("Fraction of sensors with NaN / zero over time")
    plt.xlabel("Time")
    plt.ylabel("Fraction of sensors")
    plt.tight_layout()
    plt.savefig(OUTDIR / "missing_zero_fraction_over_time.png", dpi=200)
    plt.close()


def plot_sensor_rates(profile: pd.DataFrame):
    p = profile.sort_values("missing_rate", ascending=False)

    plt.figure(figsize=(14, 4))
    plt.bar(p["sensor"], p["missing_rate"])
    plt.xticks(rotation=90, fontsize=6)
    plt.title("Missing rate per sensor")
    plt.ylabel("NaN rate")
    plt.tight_layout()
    plt.savefig(OUTDIR / "missing_rate_per_sensor.png", dpi=200)
    plt.close()

    p = profile.sort_values("zero_rate", ascending=False)

    plt.figure(figsize=(14, 4))
    plt.bar(p["sensor"], p["zero_rate"])
    plt.xticks(rotation=90, fontsize=6)
    plt.title("Zero rate per sensor")
    plt.ylabel("Zero rate")
    plt.tight_layout()
    plt.savefig(OUTDIR / "zero_rate_per_sensor.png", dpi=200)
    plt.close()


def plot_gap_histogram(gaps: pd.DataFrame, condition: str):
    if gaps.empty:
        return

    g = gaps[gaps["condition"] == condition]
    if g.empty:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(g["length"], bins=50)
    plt.yscale("log")
    plt.title(f"Distribution of {condition} run lengths")
    plt.xlabel("Run length [samples]")
    plt.ylabel("Count, log scale")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{condition}_run_length_histogram.png", dpi=200)
    plt.close()


def plot_missing_by_hour_day(df: pd.DataFrame, X: pd.DataFrame):
    tmp = pd.DataFrame({
        "timestamp": df[TIMESTAMP_COL],
        "missing_fraction": X.isna().mean(axis=1),
        "zero_fraction": X.eq(0).mean(axis=1),
    })

    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["dayofweek"] = tmp["timestamp"].dt.dayofweek

    by_hour = tmp.groupby("hour")[["missing_fraction", "zero_fraction"]].mean()
    by_day = tmp.groupby("dayofweek")[["missing_fraction", "zero_fraction"]].mean()

    by_hour.to_csv(OUTDIR / "missing_zero_by_hour.csv")
    by_day.to_csv(OUTDIR / "missing_zero_by_dayofweek.csv")

    plt.figure(figsize=(8, 4))
    plt.plot(by_hour.index, by_hour["missing_fraction"], marker="o", label="NaN")
    plt.plot(by_hour.index, by_hour["zero_fraction"], marker="o", label="zero")
    plt.legend()
    plt.title("Average NaN / zero fraction by hour")
    plt.xlabel("Hour")
    plt.ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(OUTDIR / "missing_zero_by_hour.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(by_day.index, by_day["missing_fraction"], marker="o", label="NaN")
    plt.plot(by_day.index, by_day["zero_fraction"], marker="o", label="zero")
    plt.legend()
    plt.title("Average NaN / zero fraction by day of week")
    plt.xlabel("Day of week, Monday=0")
    plt.ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(OUTDIR / "missing_zero_by_dayofweek.png", dpi=200)
    plt.close()


def summarize_gaps(gaps: pd.DataFrame, condition: str) -> pd.DataFrame:
    g = gaps[gaps["condition"] == condition]

    if g.empty:
        return pd.DataFrame({
            "condition": [condition],
            "n_runs": [0],
        })

    return pd.DataFrame({
        "condition": [condition],
        "n_runs": [len(g)],
        "mean_length": [g["length"].mean()],
        "median_length": [g["length"].median()],
        "p75_length": [g["length"].quantile(0.75)],
        "p90_length": [g["length"].quantile(0.90)],
        "p95_length": [g["length"].quantile(0.95)],
        "p99_length": [g["length"].quantile(0.99)],
        "max_length": [g["length"].max()],
        "n_ge_1h": [(g["length"] >= SAMPLES_PER_HOUR).sum()],
        "n_ge_6h": [(g["length"] >= 6 * SAMPLES_PER_HOUR).sum()],
        "n_ge_1day": [(g["length"] >= SAMPLES_PER_DAY).sum()],
    })


def main():
    df = pd.read_csv(INPUT, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]
    X = df[sensors].apply(pd.to_numeric, errors="coerce")

    print(f"Rows: {len(df)}")
    print(f"Sensors: {len(sensors)}")
    print(f"Start: {df[TIMESTAMP_COL].min()}")
    print(f"End: {df[TIMESTAMP_COL].max()}")

    grid_report = check_timestamp_grid(df)
    grid_report.to_csv(OUTDIR / "timestamp_grid_report.csv", index=False)

    profile = build_sensor_profile(X)
    profile.to_csv(OUTDIR / "sensor_quality_profile.csv", index=False)

    nan_gaps = build_gap_table(X, "nan")
    zero_gaps = build_gap_table(X, "zero")
    gaps = pd.concat([nan_gaps, zero_gaps], ignore_index=True)
    gaps.to_csv(OUTDIR / "gap_runs_nan_and_zero.csv", index=False)

    gap_summary = pd.concat([
        summarize_gaps(gaps, "nan"),
        summarize_gaps(gaps, "zero"),
    ], ignore_index=True)
    gap_summary.to_csv(OUTDIR / "gap_summary.csv", index=False)

    status_counts = profile["status"].value_counts().rename_axis("status").reset_index(name="n_sensors")
    status_counts.to_csv(OUTDIR / "sensor_status_counts.csv", index=False)

    usable_sensors = profile.loc[profile["status"] == "usable", "sensor"]
    usable_sensors.to_csv(OUTDIR / "usable_sensors.csv", index=False)

    plot_missing_by_time(df, X)
    plot_sensor_rates(profile)
    plot_gap_histogram(gaps, "nan")
    plot_gap_histogram(gaps, "zero")
    plot_missing_by_hour_day(df, X)

    print("\nSensor status counts:")
    print(status_counts)

    print("\nWorst sensors by missing rate:")
    print(profile.sort_values("missing_rate", ascending=False).head(15)[
        ["sensor", "status", "missing_rate", "zero_rate", "longest_nan_run", "longest_zero_run"]
    ])

    print("\nWorst sensors by zero rate:")
    print(profile.sort_values("zero_rate", ascending=False).head(15)[
        ["sensor", "status", "missing_rate", "zero_rate", "longest_nan_run", "longest_zero_run"]
    ])

    print(f"\nSaved profiling outputs to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()