from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer


# =====================
# CONFIG
# =====================

TIMESTAMP_COL = "timestamp"

TRAIN_CLEAN = "model_data_qc/train_filtered_qc.csv"
VAL_CLEAN = "model_data_qc/val_filtered_qc.csv"
TEST_CLEAN = "model_data_qc/test_filtered_qc.csv"

SCENARIO_DIR = Path("masked_data/scenario_A_empirical_outage_qc")

OUTPUT_DIR = Path("baseline_results/scenario_A_empirical_outage_qc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MASKED = SCENARIO_DIR / "train" / "masked.csv"
TRAIN_MASK_MAP = SCENARIO_DIR / "train" / "mask_map.csv"

VAL_MASKED = SCENARIO_DIR / "val" / "masked.csv"
VAL_MASK_MAP = SCENARIO_DIR / "val" / "mask_map.csv"

TEST_MASKED = SCENARIO_DIR / "test" / "masked.csv"
TEST_MASK_MAP = SCENARIO_DIR / "test" / "mask_map.csv"


RESULTS_OUTPUT = OUTPUT_DIR / "baseline_metrics.csv"

SAVE_IMPUTED_FILES = True
IMPUTED_OUTPUT_DIR = OUTPUT_DIR / "imputed_files"

KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"  # "uniform" albo "distance"


# =====================
# FUNCTIONS
# =====================

def load_clean_csv(path: str | Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]

    X = df[sensors].apply(pd.to_numeric, errors="coerce")

    return df, sensors, X


def load_masked_csv(path: str | Path, expected_sensors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]

    if sensors != expected_sensors:
        raise ValueError(
            f"Sensor columns in masked file do not match expected sensors: {path}"
        )

    X = df[sensors].apply(pd.to_numeric, errors="coerce")

    return df, X


def load_mask_map(path: str | Path, expected_sensors: list[str]) -> pd.DataFrame:
    mask = pd.read_csv(path)

    if list(mask.columns) != expected_sensors:
        raise ValueError(
            f"Mask map columns do not match expected sensors: {path}"
        )

    mask = mask.astype(bool)

    if not mask.any().any():
        raise ValueError(f"Mask map has no masked positions: {path}")

    return mask


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape}, "
            f"y_pred shape {y_pred.shape}"
        )

    if y_true.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape}, "
            f"mask shape {mask.shape}"
        )

    valid_eval = (
        mask
        & np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    n_requested_masked = int(mask.sum())
    n_valid_eval = int(valid_eval.sum())
    n_skipped = n_requested_masked - n_valid_eval

    if n_valid_eval == 0:
        raise ValueError(
            "No valid evaluation positions found. "
            "Check mask_map and QC-filtered clean data."
        )

    errors = y_pred[valid_eval] - y_true[valid_eval]
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "n_masked_values": n_requested_masked,
        "n_valid_eval_values": n_valid_eval,
        "n_skipped_invalid_eval_values": n_skipped,
        "mae": float(mae),
        "rmse": float(rmse),
    }


def make_output_df(
    timestamps: pd.Series,
    sensors: list[str],
    X_imputed: np.ndarray,
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({TIMESTAMP_COL: timestamps}),
            pd.DataFrame(X_imputed, columns=sensors),
        ],
        axis=1,
    )


def impute_mean(
    X_masked: pd.DataFrame,
    train_means: pd.Series,
) -> np.ndarray:
    return X_masked.fillna(train_means).to_numpy(dtype=float)


def impute_median(
    X_masked: pd.DataFrame,
    train_medians: pd.Series,
) -> np.ndarray:
    return X_masked.fillna(train_medians).to_numpy(dtype=float)


def impute_ffill_bfill(
    X_masked: pd.DataFrame,
) -> np.ndarray:
    X_imputed = X_masked.ffill().bfill()

    if X_imputed.isna().any().any():
        raise ValueError("ffill/bfill failed: NaN values remain.")

    return X_imputed.to_numpy(dtype=float)


def impute_linear_interpolation(
    X_masked: pd.DataFrame,
) -> np.ndarray:
    X_imputed = X_masked.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )

    if X_imputed.isna().any().any():
        raise ValueError("linear interpolation failed: NaN values remain.")

    return X_imputed.to_numpy(dtype=float)


def fit_knn_imputer(train_X: pd.DataFrame) -> KNNImputer:
    imputer = KNNImputer(
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
    )

    imputer.fit(train_X.to_numpy(dtype=float))

    return imputer


def impute_knn(
    X_masked: pd.DataFrame,
    knn_imputer: KNNImputer,
) -> np.ndarray:
    X_imputed = knn_imputer.transform(X_masked.to_numpy(dtype=float))

    if np.isnan(X_imputed).any():
        raise ValueError("KNN imputer failed: NaN values remain.")

    return X_imputed


def evaluate_split(
    split_name: str,
    clean_path: str | Path,
    masked_path: str | Path,
    mask_map_path: str | Path,
    expected_sensors: list[str],
    train_means: pd.Series,
    train_medians: pd.Series,
    knn_imputer: KNNImputer,
) -> list[dict]:
    clean_df, sensors, X_clean = load_clean_csv(clean_path)

    if sensors != expected_sensors:
        raise ValueError(f"{split_name}: clean sensors do not match train sensors.")

    masked_df, X_masked = load_masked_csv(masked_path, expected_sensors)
    mask_map = load_mask_map(mask_map_path, expected_sensors)

    if len(clean_df) != len(masked_df):
        raise ValueError(f"{split_name}: clean and masked files have different row counts.")

    if not clean_df[TIMESTAMP_COL].equals(masked_df[TIMESTAMP_COL]):
        raise ValueError(f"{split_name}: timestamps differ between clean and masked files.")

    y_true = X_clean.to_numpy(dtype=float)
    mask = mask_map.to_numpy(dtype=bool)

    methods = {}

    methods["mean_per_sensor"] = impute_mean(
        X_masked=X_masked,
        train_means=train_means,
    )

    methods["median_per_sensor"] = impute_median(
        X_masked=X_masked,
        train_medians=train_medians,
    )

    methods["ffill_bfill"] = impute_ffill_bfill(
        X_masked=X_masked,
    )

    methods["linear_interpolation"] = impute_linear_interpolation(
        X_masked=X_masked,
    )

    methods["knn_imputer"] = impute_knn(
        X_masked=X_masked,
        knn_imputer=knn_imputer,
    )

    rows = []

    for method_name, X_imputed in methods.items():
        metrics = compute_metrics(
            y_true=y_true,
            y_pred=X_imputed,
            mask=mask,
        )

        row = {
            "split": split_name,
            "method": method_name,
            **metrics,
        }

        rows.append(row)

        if SAVE_IMPUTED_FILES:
            split_outdir = IMPUTED_OUTPUT_DIR / split_name
            split_outdir.mkdir(parents=True, exist_ok=True)

            imputed_df = make_output_df(
                timestamps=clean_df[TIMESTAMP_COL],
                sensors=sensors,
                X_imputed=X_imputed,
            )

            imputed_df.to_csv(
                split_outdir / f"{method_name}_imputed.csv",
                index=False,
            )

    return rows


def main() -> None:
    train_clean_df, train_sensors, train_X = load_clean_csv(TRAIN_CLEAN)

    train_means = train_X.mean(axis=0)
    train_medians = train_X.median(axis=0)

    print("Fitting KNN Imputer on clean training data...")
    knn_imputer = fit_knn_imputer(train_X)

    all_rows = []

    splits = [
        {
            "split_name": "train",
            "clean_path": TRAIN_CLEAN,
            "masked_path": TRAIN_MASKED,
            "mask_map_path": TRAIN_MASK_MAP,
        },
        {
            "split_name": "val",
            "clean_path": VAL_CLEAN,
            "masked_path": VAL_MASKED,
            "mask_map_path": VAL_MASK_MAP,
        },
        {
            "split_name": "test",
            "clean_path": TEST_CLEAN,
            "masked_path": TEST_MASKED,
            "mask_map_path": TEST_MASK_MAP,
        },
    ]

    for split in splits:
        print(f"\nEvaluating split: {split['split_name']}")

        rows = evaluate_split(
            split_name=split["split_name"],
            clean_path=split["clean_path"],
            masked_path=split["masked_path"],
            mask_map_path=split["mask_map_path"],
            expected_sensors=train_sensors,
            train_means=train_means,
            train_medians=train_medians,
            knn_imputer=knn_imputer,
        )

        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    results = results.sort_values(["split", "rmse", "mae"]).reset_index(drop=True)
    results.to_csv(RESULTS_OUTPUT, index=False)

    print("\nBaseline results")
    print("----------------")
    print(results)

    print(f"\nSaved metrics to: {RESULTS_OUTPUT}")

    if SAVE_IMPUTED_FILES:
        print(f"Saved imputed files to: {IMPUTED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()