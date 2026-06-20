from pathlib import Path
import pandas as pd


# =====================
# CONFIG
# =====================

PROFILE_PATH = "profiling_outputs/sensor_quality_profile.csv"

INPUT_PATH = "combined_sensors_2022-10-01_2022-12-31.csv"
OUTPUT_PATH = "combined_sensors_2022-10-01_2022-12-31_filtered.csv"

# INPUT_PATH = "combined_sensors_2023-01-01_2023-01-31.csv"
# OUTPUT_PATH = "combined_sensors_2023-01-01_2023-01-31_filtered.csv"

PANEL_OUTPUT_PATH = "conservative_sensors.txt"

TIMESTAMP_COL = "timestamp"

MAX_ZERO_RATE = 0.20
MAX_LONGEST_ZERO_RUN = 144

STRICT = True


# =====================
# FUNCTIONS
# =====================

def load_conservative_panel(profile_path: str) -> list[str]:
    profile = pd.read_csv(profile_path)

    required_cols = {
        "sensor",
        "status",
        "zero_rate",
        "longest_zero_run",
    }

    missing_cols = required_cols - set(profile.columns)
    if missing_cols:
        raise ValueError(
            f"Profile file is missing required columns: {sorted(missing_cols)}"
        )

    selected = profile[
        (profile["status"] == "usable")
        & (profile["zero_rate"] <= MAX_ZERO_RATE)
        & (profile["longest_zero_run"] < MAX_LONGEST_ZERO_RUN)
    ].copy()

    sensors = selected["sensor"].astype(str).tolist()

    if not sensors:
        raise ValueError("No sensors selected. Check filtering thresholds.")

    return sensors


def filter_csv_to_panel(
    input_path: str,
    output_path: str,
    sensors: list[str],
) -> list[str]:
    df = pd.read_csv(input_path)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Timestamp column '{TIMESTAMP_COL}' not found in input file."
        )

    available_sensors = [s for s in sensors if s in df.columns]
    missing_sensors = [s for s in sensors if s not in df.columns]

    if missing_sensors and STRICT:
        raise ValueError(
            "Some selected sensors are missing from the input CSV: "
            + ", ".join(missing_sensors)
        )

    if not available_sensors:
        raise ValueError("None of the selected sensors are present in the input CSV.")

    filtered = df[[TIMESTAMP_COL] + available_sensors].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    return missing_sensors


def save_sensor_panel(sensors: list[str], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sensors) + "\n", encoding="utf-8")


# =====================
# MAIN
# =====================

def main() -> None:
    sensors = load_conservative_panel(PROFILE_PATH)

    save_sensor_panel(sensors, PANEL_OUTPUT_PATH)

    missing_sensors = filter_csv_to_panel(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        sensors=sensors,
    )

    print(f"Selected sensors from profile: {len(sensors)}")
    print(f"Wrote filtered CSV to: {OUTPUT_PATH}")
    print(f"Columns in output: 1 timestamp + {len(sensors) - len(missing_sensors)} sensors")
    print(f"Wrote sensor panel to: {PANEL_OUTPUT_PATH}")

    if missing_sensors:
        print("\n[warning] These selected sensors were not present in input CSV:")
        for sensor in missing_sensors:
            print(f"  - {sensor}")


if __name__ == "__main__":
    main()