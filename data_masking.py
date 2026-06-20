from pathlib import Path
import numpy as np
import pandas as pd


# =====================
# CONFIG
# =====================

TRAIN_DATA = "model_data_qc/train_filtered_qc.csv"
VAL_DATA = "model_data_qc/val_filtered_qc.csv"
TEST_DATA = "model_data_qc/test_filtered_qc.csv"

GAP_RUNS_FILE = "profiling_outputs/gap_runs_nan_and_zero.csv"

OUTPUT_DIR = Path("masked_data/scenario_A_empirical_outage_qc")

TIMESTAMP_COL = "timestamp"

# 6 samples = 1 hour for 10-minute data
# 144 samples = 1 day for 10-minute data
MIN_GAP_LENGTH = 6
MAX_GAP_LENGTH = 144

GAP_CONDITION = "zero"

RANDOM_SEED = 42

# One outage per sensor.
OUTAGES_PER_SENSOR = 1

# How many times to try sampling a valid outage before failing.
MAX_SAMPLING_ATTEMPTS_PER_OUTAGE = 500

# If True, script fails when it cannot create the requested number of outages.
# If False, it skips sensors where no valid block can be found.
REQUIRE_OUTAGE_PER_SENSOR = True


# =====================
# FUNCTIONS
# =====================

def load_empirical_gap_lengths() -> np.ndarray:
    gaps = pd.read_csv(GAP_RUNS_FILE)

    required_cols = {"condition", "length"}
    missing_cols = required_cols - set(gaps.columns)

    if missing_cols:
        raise ValueError(f"Missing columns in gap runs file: {sorted(missing_cols)}")

    selected = gaps[
        (gaps["condition"] == GAP_CONDITION)
        & (gaps["length"] >= MIN_GAP_LENGTH)
        & (gaps["length"] <= MAX_GAP_LENGTH)
    ].copy()

    if selected.empty:
        raise ValueError("No gap lengths selected. Check MIN_GAP_LENGTH / MAX_GAP_LENGTH.")

    lengths = selected["length"].astype(int).to_numpy()

    return lengths


def save_gap_distribution(lengths: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distribution = pd.DataFrame({
        "length": lengths,
        "duration_hours": lengths / 6,
        "duration_days": lengths / 144,
    })

    distribution.to_csv(OUTPUT_DIR / "gap_distribution_used.csv", index=False)

    summary = pd.DataFrame({
        "metric": [
            "n_gaps",
            "min_length",
            "median_length",
            "mean_length",
            "p75_length",
            "p90_length",
            "p95_length",
            "p99_length",
            "max_length",
        ],
        "value": [
            len(lengths),
            np.min(lengths),
            np.median(lengths),
            np.mean(lengths),
            np.quantile(lengths, 0.75),
            np.quantile(lengths, 0.90),
            np.quantile(lengths, 0.95),
            np.quantile(lengths, 0.99),
            np.max(lengths),
        ],
    })

    summary.to_csv(OUTPUT_DIR / "gap_distribution_summary.csv", index=False)


def find_valid_start_indices(valid_positions: np.ndarray, length: int) -> np.ndarray:
    """
    Finds all start indices where a contiguous block of given length
    contains only valid values.

    valid_positions:
        True  = value can be used as ground truth
        False = original NaN / physically invalid value / already masked
    """
    n_rows = len(valid_positions)

    if length > n_rows:
        return np.array([], dtype=int)

    valid_int = valid_positions.astype(int)

    # rolling sum over windows of size `length`
    window_sums = np.convolve(
        valid_int,
        np.ones(length, dtype=int),
        mode="valid",
    )

    starts = np.where(window_sums == length)[0]

    return starts.astype(int)


def sample_valid_outage(
    valid_positions: np.ndarray,
    gap_lengths: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """
    Samples one outage length and start index such that the whole outage
    falls only on valid positions.

    Returns:
        length, start_idx, end_idx
    """
    n_rows = len(valid_positions)

    possible_gap_lengths = gap_lengths[gap_lengths <= n_rows]

    if len(possible_gap_lengths) == 0:
        raise ValueError("No possible gap lengths for this split.")

    for _ in range(MAX_SAMPLING_ATTEMPTS_PER_OUTAGE):
        length = int(rng.choice(possible_gap_lengths))

        starts = find_valid_start_indices(
            valid_positions=valid_positions,
            length=length,
        )

        if len(starts) == 0:
            continue

        start_idx = int(rng.choice(starts))
        end_idx = start_idx + length - 1

        return length, start_idx, end_idx

    feasible_lengths = []

    for length in np.unique(possible_gap_lengths):
        starts = find_valid_start_indices(
            valid_positions=valid_positions,
            length=int(length),
        )

        if len(starts) > 0:
            feasible_lengths.append(int(length))

    if len(feasible_lengths) == 0:
        raise ValueError("No valid contiguous segment found for any selected gap length.")

    length = int(rng.choice(feasible_lengths))

    starts = find_valid_start_indices(
        valid_positions=valid_positions,
        length=length,
    )

    start_idx = int(rng.choice(starts))
    end_idx = start_idx + length - 1

    return length, start_idx, end_idx


def generate_empirical_outage_mask(
    df: pd.DataFrame,
    X: pd.DataFrame,
    gap_lengths: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.DataFrame]:
    sensors = [c for c in df.columns if c != TIMESTAMP_COL]
    n_rows = len(df)

    mask = np.zeros((n_rows, len(sensors)), dtype=bool)
    outage_rows = []

    for sensor_idx, sensor in enumerate(sensors):
        for outage_id in range(OUTAGES_PER_SENSOR):
            # Valid means:
            # - not NaN in QC data,
            # - not already selected as artificial missing for this sensor.
            valid_positions = X[sensor].notna().to_numpy()
            valid_positions = valid_positions & (~mask[:, sensor_idx])

            try:
                length, start_idx, end_idx = sample_valid_outage(
                    valid_positions=valid_positions,
                    gap_lengths=gap_lengths,
                    rng=rng,
                )
            except ValueError as exc:
                message = (
                    f"Could not sample valid outage for split sensor={sensor}, "
                    f"outage_id={outage_id}. Reason: {exc}"
                )

                if REQUIRE_OUTAGE_PER_SENSOR:
                    raise ValueError(message)

                print(f"WARNING: {message}")
                continue

            mask[start_idx:start_idx + length, sensor_idx] = True

            outage_rows.append({
                "sensor": sensor,
                "outage_id": outage_id,
                "length": length,
                "duration_hours": length / 6,
                "duration_days": length / 144,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_timestamp": df[TIMESTAMP_COL].iloc[start_idx],
                "end_timestamp": df[TIMESTAMP_COL].iloc[end_idx],
            })

    sampled_outages = pd.DataFrame(outage_rows)

    return mask, sampled_outages


def apply_mask_and_save(
    input_path: str,
    split_name: str,
    gap_lengths: np.ndarray,
    seed_offset: int,
) -> None:
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)

    df = pd.read_csv(input_path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]

    X = df[sensors].apply(pd.to_numeric, errors="coerce")

    n_original_nan = int(X.isna().sum().sum())

    mask, sampled_outages = generate_empirical_outage_mask(
        df=df,
        X=X,
        gap_lengths=gap_lengths,
        rng=rng,
    )

    original_values = X.to_numpy(dtype=float).copy()
    masked_values = original_values.copy()

    masked_values[mask] = np.nan

    masked_df = pd.concat(
        [
            df[[TIMESTAMP_COL]],
            pd.DataFrame(masked_values, columns=sensors),
        ],
        axis=1,
    )

    mask_map_df = pd.DataFrame(mask.astype(int), columns=sensors)

    removed_values = np.where(mask, original_values, np.nan)

    removed_values_df = pd.concat(
        [
            df[[TIMESTAMP_COL]],
            pd.DataFrame(removed_values, columns=sensors),
        ],
        axis=1,
    )

    rows_idx, cols_idx = np.where(mask)

    removed_long_df = pd.DataFrame({
        TIMESTAMP_COL: df[TIMESTAMP_COL].iloc[rows_idx].to_numpy(),
        "sensor": np.array(sensors)[cols_idx],
        "value": original_values[rows_idx, cols_idx],
    })

    if removed_long_df["value"].isna().any():
        raise ValueError(
            f"{split_name}: synthetic mask selected NaN values. "
            "This should not happen."
        )

    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    masked_df.to_csv(split_dir / "masked.csv", index=False)
    mask_map_df.to_csv(split_dir / "mask_map.csv", index=False)
    removed_values_df.to_csv(split_dir / "removed_values.csv", index=False)
    removed_long_df.to_csv(split_dir / "removed_values_long.csv", index=False)
    sampled_outages.to_csv(split_dir / "sampled_outages.csv", index=False)

    n_artificial_masked = int(mask.sum())
    total_values = mask.size

    artificial_missing_rate = n_artificial_masked / total_values
    original_nan_rate = n_original_nan / total_values
    total_nan_after_masking = int(np.isnan(masked_values).sum())
    total_nan_rate_after_masking = total_nan_after_masking / total_values

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"input:                    {input_path}")
    print(f"rows:                     {len(df)}")
    print(f"sensors:                  {len(sensors)}")
    print(f"original NaN/QC invalid:  {n_original_nan}")
    print(f"original NaN rate:        {original_nan_rate:.4%}")
    print(f"artificial masked values: {n_artificial_masked}")
    print(f"artificial missing rate:  {artificial_missing_rate:.4%}")
    print(f"total NaN after masking:  {total_nan_after_masking}")
    print(f"total NaN rate after:     {total_nan_rate_after_masking:.4%}")
    print(f"sampled outages:          {len(sampled_outages)}")
    print(f"output dir:               {split_dir}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gap_lengths = load_empirical_gap_lengths()
    save_gap_distribution(gap_lengths)

    print("Scenario A: empirical outage masking with QC-aware valid positions")
    print("------------------------------------------------------------------")
    print(f"gap condition:        {GAP_CONDITION}")
    print(f"min gap length:       {MIN_GAP_LENGTH}")
    print(f"max gap length:       {MAX_GAP_LENGTH}")
    print(f"empirical gaps used:  {len(gap_lengths)}")
    print(f"median length:        {np.median(gap_lengths):.1f}")
    print(f"mean length:          {np.mean(gap_lengths):.2f}")
    print(f"p90 length:           {np.quantile(gap_lengths, 0.90):.1f}")
    print(f"p95 length:           {np.quantile(gap_lengths, 0.95):.1f}")
    print(f"p99 length:           {np.quantile(gap_lengths, 0.99):.1f}")

    apply_mask_and_save(
        input_path=TRAIN_DATA,
        split_name="train",
        gap_lengths=gap_lengths,
        seed_offset=0,
    )

    apply_mask_and_save(
        input_path=VAL_DATA,
        split_name="val",
        gap_lengths=gap_lengths,
        seed_offset=1000,
    )

    apply_mask_and_save(
        input_path=TEST_DATA,
        split_name="test",
        gap_lengths=gap_lengths,
        seed_offset=2000,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()