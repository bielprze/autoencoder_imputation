from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

TIMESTAMP_COL = "timestamp"

TEST_CLEAN = "model_data_qc/test_filtered_qc.csv"
TEST_MASK_MAP = "masked_data/scenario_A_empirical_outage_qc/test/mask_map.csv"
TEST_SAMPLED_OUTAGES = "masked_data/scenario_A_empirical_outage_qc/test/sampled_outages.csv"

DAE_IMPUTED = "dae_results/scenario_A_empirical_outage_qc/test_imputed_dae.csv"

EDGES_FILE = "edges_filtered.csv"
EDGE_SOURCE_COL = "sensor_id"
EDGE_TARGET_COL = "neighbor_id"

GRAPH_DAE_IMPUTED = "pyg_gdae_results/scenario_A_empirical_outage_qc/test_imputed_pyg_graph_dae.csv"
OUTPUT_DIR = Path("pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 30

UNDIRECTED_GRAPH = True


# =====================
# DATA LOADING
# =====================

def load_data():
    clean_df = pd.read_csv(TEST_CLEAN, parse_dates=[TIMESTAMP_COL])
    dae_df = pd.read_csv(DAE_IMPUTED, parse_dates=[TIMESTAMP_COL])
    graph_df = pd.read_csv(GRAPH_DAE_IMPUTED, parse_dates=[TIMESTAMP_COL])
    mask_map_df = pd.read_csv(TEST_MASK_MAP)
    outages_df = pd.read_csv(
        TEST_SAMPLED_OUTAGES,
        parse_dates=["start_timestamp", "end_timestamp"],
    )

    clean_df = clean_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    dae_df = dae_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    graph_df = graph_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in clean_df.columns if c != TIMESTAMP_COL]

    for name, df in [
        ("DAE", dae_df),
        ("Graph DAE", graph_df),
    ]:
        if [c for c in df.columns if c != TIMESTAMP_COL] != sensors:
            raise ValueError(f"{name}: sensor columns differ from clean data.")

        if not clean_df[TIMESTAMP_COL].equals(df[TIMESTAMP_COL]):
            raise ValueError(f"{name}: timestamps differ from clean data.")

    if list(mask_map_df.columns) != sensors:
        raise ValueError("mask_map columns differ from clean data sensors.")

    clean_X = clean_df[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    dae_X = dae_df[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    graph_X = graph_df[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mask = mask_map_df.to_numpy(dtype=bool)

    return clean_df, sensors, clean_X, dae_X, graph_X, mask, outages_df


def load_graph_degrees(sensors: list[str]) -> pd.DataFrame:
    edges = pd.read_csv(EDGES_FILE)

    if EDGE_SOURCE_COL not in edges.columns or EDGE_TARGET_COL not in edges.columns:
        if len(edges.columns) < 2:
            raise ValueError("Edges file must contain at least two columns.")

        print(
            f"WARNING: columns {EDGE_SOURCE_COL}/{EDGE_TARGET_COL} not found. "
            f"Using first two columns: {edges.columns[0]}, {edges.columns[1]}"
        )

        source_col = edges.columns[0]
        target_col = edges.columns[1]
    else:
        source_col = EDGE_SOURCE_COL
        target_col = EDGE_TARGET_COL

    sensors_set = set(sensors)

    degree = {sensor: 0 for sensor in sensors}

    for _, row in edges.iterrows():
        src = row[source_col]
        dst = row[target_col]

        if src not in sensors_set or dst not in sensors_set:
            continue

        degree[src] += 1

        if UNDIRECTED_GRAPH:
            degree[dst] += 1

    degree_df = pd.DataFrame({
        "sensor": list(degree.keys()),
        "graph_degree": list(degree.values()),
    })

    return degree_df


# =====================
# METRICS
# =====================

def metrics_from_errors(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return {
            "n_points": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs_error": np.nan,
            "p90_abs_error": np.nan,
            "max_abs_error": np.nan,
            "sum_squared_error": 0.0,
        }

    abs_errors = np.abs(errors)

    return {
        "n_points": int(len(errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_abs_error": float(np.median(abs_errors)),
        "p90_abs_error": float(np.quantile(abs_errors, 0.90)),
        "max_abs_error": float(np.max(abs_errors)),
        "sum_squared_error": float(np.sum(errors ** 2)),
    }


def build_point_comparison(
    clean_df: pd.DataFrame,
    sensors: list[str],
    clean_X: np.ndarray,
    dae_X: np.ndarray,
    graph_X: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    valid_eval = (
        mask
        & np.isfinite(clean_X)
        & np.isfinite(dae_X)
        & np.isfinite(graph_X)
    )

    rows_idx, cols_idx = np.where(valid_eval)

    y_true = clean_X[rows_idx, cols_idx]

    dae_pred = dae_X[rows_idx, cols_idx]
    graph_pred = graph_X[rows_idx, cols_idx]

    dae_error = dae_pred - y_true
    graph_error = graph_pred - y_true

    dae_abs_error = np.abs(dae_error)
    graph_abs_error = np.abs(graph_error)

    point_df = pd.DataFrame({
        "timestamp": clean_df[TIMESTAMP_COL].iloc[rows_idx].to_numpy(),
        "row_idx": rows_idx,
        "sensor": np.array(sensors)[cols_idx],
        "sensor_idx": cols_idx,
        "y_true": y_true,

        "dae_pred": dae_pred,
        "dae_error": dae_error,
        "dae_abs_error": dae_abs_error,
        "dae_squared_error": dae_error ** 2,

        "graph_pred": graph_pred,
        "graph_error": graph_error,
        "graph_abs_error": graph_abs_error,
        "graph_squared_error": graph_error ** 2,
    })

    point_df["abs_error_delta_graph_minus_dae"] = (
        point_df["graph_abs_error"] - point_df["dae_abs_error"]
    )

    point_df["squared_error_delta_graph_minus_dae"] = (
        point_df["graph_squared_error"] - point_df["dae_squared_error"]
    )

    point_df["graph_better_abs_error"] = (
        point_df["graph_abs_error"] < point_df["dae_abs_error"]
    )

    point_df["graph_better_squared_error"] = (
        point_df["graph_squared_error"] < point_df["dae_squared_error"]
    )

    point_df["hour"] = pd.to_datetime(point_df["timestamp"]).dt.hour
    point_df["date"] = pd.to_datetime(point_df["timestamp"]).dt.strftime("%Y-%m-%d")

    return point_df


def aggregate_by_sensor(point_df: pd.DataFrame, degree_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sensor, group in point_df.groupby("sensor"):
        dae_metrics = metrics_from_errors(group["dae_error"].to_numpy())
        graph_metrics = metrics_from_errors(group["graph_error"].to_numpy())

        rows.append({
            "sensor": sensor,
            "n_points": dae_metrics["n_points"],

            "dae_mae": dae_metrics["mae"],
            "dae_rmse": dae_metrics["rmse"],
            "dae_median_abs_error": dae_metrics["median_abs_error"],
            "dae_p90_abs_error": dae_metrics["p90_abs_error"],
            "dae_max_abs_error": dae_metrics["max_abs_error"],
            "dae_sum_squared_error": dae_metrics["sum_squared_error"],

            "graph_mae": graph_metrics["mae"],
            "graph_rmse": graph_metrics["rmse"],
            "graph_median_abs_error": graph_metrics["median_abs_error"],
            "graph_p90_abs_error": graph_metrics["p90_abs_error"],
            "graph_max_abs_error": graph_metrics["max_abs_error"],
            "graph_sum_squared_error": graph_metrics["sum_squared_error"],

            "delta_mae_graph_minus_dae": graph_metrics["mae"] - dae_metrics["mae"],
            "delta_rmse_graph_minus_dae": graph_metrics["rmse"] - dae_metrics["rmse"],
            "delta_sse_graph_minus_dae": (
                graph_metrics["sum_squared_error"] - dae_metrics["sum_squared_error"]
            ),

            "graph_better_point_share_abs": float(group["graph_better_abs_error"].mean()),
            "graph_better_point_share_squared": float(group["graph_better_squared_error"].mean()),
        })

    sensor_df = pd.DataFrame(rows)

    sensor_df = sensor_df.merge(
        degree_df,
        on="sensor",
        how="left",
    )

    total_dae_sse = sensor_df["dae_sum_squared_error"].sum()
    total_graph_sse = sensor_df["graph_sum_squared_error"].sum()

    sensor_df["dae_sse_share"] = sensor_df["dae_sum_squared_error"] / total_dae_sse
    sensor_df["graph_sse_share"] = sensor_df["graph_sum_squared_error"] / total_graph_sse

    return sensor_df.sort_values("delta_rmse_graph_minus_dae").reset_index(drop=True)


def aggregate_by_outage(point_df: pd.DataFrame, outages_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, outage in outages_df.iterrows():
        sensor = outage["sensor"]
        start_idx = int(outage["start_idx"])
        end_idx = int(outage["end_idx"])

        subset = point_df[
            (point_df["sensor"] == sensor)
            & (point_df["row_idx"] >= start_idx)
            & (point_df["row_idx"] <= end_idx)
        ]

        if subset.empty:
            continue

        dae_metrics = metrics_from_errors(subset["dae_error"].to_numpy())
        graph_metrics = metrics_from_errors(subset["graph_error"].to_numpy())

        rows.append({
            "sensor": sensor,
            "outage_id": outage["outage_id"],
            "length": int(outage["length"]),
            "duration_hours": float(outage["duration_hours"]),
            "duration_days": float(outage["duration_days"]),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_timestamp": outage["start_timestamp"],
            "end_timestamp": outage["end_timestamp"],

            "n_points": dae_metrics["n_points"],

            "dae_mae": dae_metrics["mae"],
            "dae_rmse": dae_metrics["rmse"],
            "dae_sum_squared_error": dae_metrics["sum_squared_error"],
            "dae_max_abs_error": dae_metrics["max_abs_error"],

            "graph_mae": graph_metrics["mae"],
            "graph_rmse": graph_metrics["rmse"],
            "graph_sum_squared_error": graph_metrics["sum_squared_error"],
            "graph_max_abs_error": graph_metrics["max_abs_error"],

            "delta_mae_graph_minus_dae": graph_metrics["mae"] - dae_metrics["mae"],
            "delta_rmse_graph_minus_dae": graph_metrics["rmse"] - dae_metrics["rmse"],
            "delta_sse_graph_minus_dae": (
                graph_metrics["sum_squared_error"] - dae_metrics["sum_squared_error"]
            ),

            "graph_better_point_share_abs": float(subset["graph_better_abs_error"].mean()),
            "graph_better_point_share_squared": float(subset["graph_better_squared_error"].mean()),
        })

    outage_df = pd.DataFrame(rows)

    if outage_df.empty:
        raise ValueError("No outage-level comparisons were created.")

    return outage_df.sort_values("delta_rmse_graph_minus_dae").reset_index(drop=True)


def aggregate_by_degree(sensor_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()

    bins = [-1, 0, 1, 2, 4, 8, 9999]
    labels = ["0", "1", "2", "3-4", "5-8", "9+"]

    df["degree_bucket"] = pd.cut(
        df["graph_degree"],
        bins=bins,
        labels=labels,
    )

    degree_df = df.groupby("degree_bucket", observed=False).agg(
        n_sensors=("sensor", "count"),
        n_points=("n_points", "sum"),
        mean_delta_mae=("delta_mae_graph_minus_dae", "mean"),
        mean_delta_rmse=("delta_rmse_graph_minus_dae", "mean"),
        median_delta_mae=("delta_mae_graph_minus_dae", "median"),
        median_delta_rmse=("delta_rmse_graph_minus_dae", "median"),
        mean_graph_better_point_share_abs=("graph_better_point_share_abs", "mean"),
    ).reset_index()

    return degree_df


def build_summary(point_df: pd.DataFrame, sensor_df: pd.DataFrame, outage_df: pd.DataFrame) -> pd.DataFrame:
    dae_global = metrics_from_errors(point_df["dae_error"].to_numpy())
    graph_global = metrics_from_errors(point_df["graph_error"].to_numpy())

    summary = pd.DataFrame([
        {
            "level": "global",
            "n_points": dae_global["n_points"],
            "dae_mae": dae_global["mae"],
            "dae_rmse": dae_global["rmse"],
            "graph_mae": graph_global["mae"],
            "graph_rmse": graph_global["rmse"],
            "delta_mae_graph_minus_dae": graph_global["mae"] - dae_global["mae"],
            "delta_rmse_graph_minus_dae": graph_global["rmse"] - dae_global["rmse"],
            "graph_better_point_share_abs": float(point_df["graph_better_abs_error"].mean()),
            "graph_better_point_share_squared": float(point_df["graph_better_squared_error"].mean()),
            "n_sensors_graph_better_mae": int((sensor_df["delta_mae_graph_minus_dae"] < 0).sum()),
            "n_sensors_graph_worse_mae": int((sensor_df["delta_mae_graph_minus_dae"] > 0).sum()),
            "n_outages_graph_better_mae": int((outage_df["delta_mae_graph_minus_dae"] < 0).sum()),
            "n_outages_graph_worse_mae": int((outage_df["delta_mae_graph_minus_dae"] > 0).sum()),
        }
    ])

    return summary


# =====================
# PLOTS
# =====================

def plot_top_sensor_improvements(sensor_df: pd.DataFrame) -> None:
    top = pd.concat([
        sensor_df.sort_values("delta_mae_graph_minus_dae").head(10),
        sensor_df.sort_values("delta_mae_graph_minus_dae").tail(10),
    ]).copy()

    top = top.sort_values("delta_mae_graph_minus_dae")

    plt.figure(figsize=(10, 7))
    plt.barh(top["sensor"], top["delta_mae_graph_minus_dae"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Delta MAE: Graph DAE - DAE")
    plt.ylabel("Sensor")
    plt.title("Graph DAE improvement/worsening by sensor")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_delta_mae_top_bottom.png", dpi=200)
    plt.close()


def plot_degree_vs_delta(sensor_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(sensor_df["graph_degree"], sensor_df["delta_mae_graph_minus_dae"], alpha=0.8)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Graph degree")
    plt.ylabel("Delta MAE: Graph DAE - DAE")
    plt.title("Does graph degree relate to Graph DAE improvement?")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "degree_vs_delta_mae.png", dpi=200)
    plt.close()


def plot_outage_delta(outage_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(outage_df["length"], outage_df["delta_mae_graph_minus_dae"], alpha=0.8)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Outage length [samples]")
    plt.ylabel("Delta MAE: Graph DAE - DAE")
    plt.title("Graph DAE improvement by outage length")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outage_length_vs_delta_mae.png", dpi=200)
    plt.close()


# =====================
# MAIN
# =====================

def main() -> None:
    clean_df, sensors, clean_X, dae_X, graph_X, mask, outages_df = load_data()
    degree_df = load_graph_degrees(sensors)

    point_df = build_point_comparison(
        clean_df=clean_df,
        sensors=sensors,
        clean_X=clean_X,
        dae_X=dae_X,
        graph_X=graph_X,
        mask=mask,
    )

    sensor_df = aggregate_by_sensor(
        point_df=point_df,
        degree_df=degree_df,
    )

    outage_df = aggregate_by_outage(
        point_df=point_df,
        outages_df=outages_df,
    )

    degree_bucket_df = aggregate_by_degree(sensor_df)

    summary_df = build_summary(
        point_df=point_df,
        sensor_df=sensor_df,
        outage_df=outage_df,
    )

    point_df.to_csv(OUTPUT_DIR / "point_comparison_dae_vs_graph_dae.csv", index=False)
    sensor_df.to_csv(OUTPUT_DIR / "sensor_comparison_dae_vs_graph_dae.csv", index=False)
    outage_df.to_csv(OUTPUT_DIR / "outage_comparison_dae_vs_graph_dae.csv", index=False)
    degree_bucket_df.to_csv(OUTPUT_DIR / "degree_bucket_comparison.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_dae_vs_graph_dae.csv", index=False)

    sensor_df.sort_values("delta_mae_graph_minus_dae").head(TOP_N).to_csv(
        OUTPUT_DIR / "top_sensors_graph_better.csv",
        index=False,
    )

    sensor_df.sort_values("delta_mae_graph_minus_dae").tail(TOP_N).to_csv(
        OUTPUT_DIR / "top_sensors_graph_worse.csv",
        index=False,
    )

    outage_df.sort_values("delta_mae_graph_minus_dae").head(TOP_N).to_csv(
        OUTPUT_DIR / "top_outages_graph_better.csv",
        index=False,
    )

    outage_df.sort_values("delta_mae_graph_minus_dae").tail(TOP_N).to_csv(
        OUTPUT_DIR / "top_outages_graph_worse.csv",
        index=False,
    )

    plot_top_sensor_improvements(sensor_df)
    plot_degree_vs_delta(sensor_df)
    plot_outage_delta(outage_df)

    print("DAE vs Graph DAE analysis")
    print("-------------------------")

    print("\nGlobal summary:")
    print(summary_df.T)

    print("\nTop sensors where Graph DAE improves MAE:")
    print(
        sensor_df.sort_values("delta_mae_graph_minus_dae")
        .head(10)[
            [
                "sensor",
                "graph_degree",
                "n_points",
                "dae_mae",
                "graph_mae",
                "delta_mae_graph_minus_dae",
                "dae_rmse",
                "graph_rmse",
                "delta_rmse_graph_minus_dae",
            ]
        ]
    )

    print("\nTop sensors where Graph DAE worsens MAE:")
    print(
        sensor_df.sort_values("delta_mae_graph_minus_dae")
        .tail(10)[
            [
                "sensor",
                "graph_degree",
                "n_points",
                "dae_mae",
                "graph_mae",
                "delta_mae_graph_minus_dae",
                "dae_rmse",
                "graph_rmse",
                "delta_rmse_graph_minus_dae",
            ]
        ]
    )

    print("\nTop outages where Graph DAE improves MAE:")
    print(
        outage_df.sort_values("delta_mae_graph_minus_dae")
        .head(10)[
            [
                "sensor",
                "length",
                "duration_hours",
                "dae_mae",
                "graph_mae",
                "delta_mae_graph_minus_dae",
                "dae_rmse",
                "graph_rmse",
                "delta_rmse_graph_minus_dae",
            ]
        ]
    )

    print("\nTop outages where Graph DAE worsens MAE:")
    print(
        outage_df.sort_values("delta_mae_graph_minus_dae")
        .tail(10)[
            [
                "sensor",
                "length",
                "duration_hours",
                "dae_mae",
                "graph_mae",
                "delta_mae_graph_minus_dae",
                "dae_rmse",
                "graph_rmse",
                "delta_rmse_graph_minus_dae",
            ]
        ]
    )

    print("\nDegree bucket comparison:")
    print(degree_bucket_df)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()