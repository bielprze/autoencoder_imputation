from pathlib import Path
import pandas as pd


# =====================
# CONFIG
# =====================

OCT_DEC_DATA = "combined_sensors_2022-10-01_2022-12-31_filtered.csv"
JAN_DATA = "combined_sensors_2023-01-01_2023-01-31_filtered.csv"

OUTPUT_DIR = Path("model_data")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_OUTPUT = OUTPUT_DIR / "train_filtered.csv"
VAL_OUTPUT = OUTPUT_DIR / "val_filtered.csv"
TEST_OUTPUT = OUTPUT_DIR / "test_filtered.csv"

TIMESTAMP_COL = "timestamp"

TRAIN_START = "2022-10-01 00:00:00"
TRAIN_END = "2022-11-30 23:50:00"

VAL_START = "2022-12-01 00:00:00"
VAL_END = "2022-12-31 23:50:00"

TEST_START = "2023-01-01 00:00:00"
TEST_END = "2023-01-31 23:50:00"

EXPECTED_FREQ = "10min"


# =====================
# FUNCTIONS
# =====================

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def slice_by_time(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    sliced = df[
        (df[TIMESTAMP_COL] >= start)
        & (df[TIMESTAMP_COL] <= end)
    ].copy()

    return sliced.reset_index(drop=True)


def check_time_grid(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name}: dataframe is empty.")

    ts = df[TIMESTAMP_COL]

    expected_grid = pd.date_range(
        start=ts.min(),
        end=ts.max(),
        freq=EXPECTED_FREQ,
    )

    missing_timestamps = expected_grid.difference(ts)
    duplicated_timestamps = ts[ts.duplicated()]

    diffs = ts.diff().dropna()
    most_common_step = diffs.mode().iloc[0] if len(diffs) > 0 else None

    print(f"\n{name}")
    print("-" * len(name))
    print(f"start:                  {ts.min()}")
    print(f"end:                    {ts.max()}")
    print(f"rows:                   {len(df)}")
    print(f"expected rows on grid:  {len(expected_grid)}")
    print(f"missing timestamps:     {len(missing_timestamps)}")
    print(f"duplicated timestamps:  {len(duplicated_timestamps)}")
    print(f"most common step:       {most_common_step}")
    print(f"unique time steps:      {diffs.nunique()}")

    if len(missing_timestamps) > 0:
        raise ValueError(f"{name}: missing timestamps detected.")

    if len(duplicated_timestamps) > 0:
        raise ValueError(f"{name}: duplicated timestamps detected.")

    if len(df) != len(expected_grid):
        raise ValueError(f"{name}: row count does not match expected grid.")


def check_columns(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_cols = list(train.columns)
    val_cols = list(val.columns)
    test_cols = list(test.columns)

    if train_cols != val_cols:
        raise ValueError("Train and validation columns differ or are in different order.")

    if train_cols != test_cols:
        raise ValueError("Train and test columns differ or are in different order.")

    print("\nColumn check")
    print("------------")
    print("Train/val/test columns are identical and in the same order.")
    print(f"Number of sensors: {len(train_cols) - 1}")


def check_values(df: pd.DataFrame, name: str) -> None:
    sensors = [c for c in df.columns if c != TIMESTAMP_COL]
    X = df[sensors].apply(pd.to_numeric, errors="coerce")

    nan_rate = X.isna().mean().mean()
    zero_rate = X.eq(0).mean().mean()
    min_value = X.min().min()
    max_value = X.max().max()

    print(f"\n{name} values")
    print("-" * (len(name) + 7))
    print(f"global NaN rate:   {nan_rate:.6f}")
    print(f"global zero rate:  {zero_rate:.6f}")
    print(f"min value:         {min_value}")
    print(f"max value:         {max_value}")

    if nan_rate > 0:
        raise ValueError(f"{name}: NaN values detected after filtering.")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# =====================
# MAIN
# =====================

def main() -> None:
    oct_dec = load_csv(OCT_DEC_DATA)
    jan = load_csv(JAN_DATA)

    train = slice_by_time(
        oct_dec,
        start=TRAIN_START,
        end=TRAIN_END,
    )

    val = slice_by_time(
        oct_dec,
        start=VAL_START,
        end=VAL_END,
    )

    test = slice_by_time(
        jan,
        start=TEST_START,
        end=TEST_END,
    )

    check_columns(train, val, test)

    check_time_grid(train, "train")
    check_time_grid(val, "validation")
    check_time_grid(test, "test")

    check_values(train, "train")
    check_values(val, "validation")
    check_values(test, "test")

    save_csv(train, TRAIN_OUTPUT)
    save_csv(val, VAL_OUTPUT)
    save_csv(test, TEST_OUTPUT)

    print("\nDone.")


if __name__ == "__main__":
    main()