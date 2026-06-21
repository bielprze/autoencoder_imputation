# Traffic Sensor Data Imputation with Denoising Autoencoders and PyTorch Geometric

This repository contains a complete experiment pipeline for traffic sensor data imputation.

The project compares classical imputation baselines, a standard denoising autoencoder, and a graph-based denoising autoencoder implemented with **PyTorch Geometric**. The final graph model uses a manually prepared sparse road-network graph and `GCNConv` layers to propagate information between neighboring traffic sensors.

---

## Project goal

The goal of the project is to reconstruct missing traffic sensor measurements under a realistic block-missing scenario.

Instead of evaluating models only on random missing points, the project uses empirical outage lengths derived from traffic data profiling. Artificial gaps are then applied to valid reference values, so the models can be evaluated against known ground truth.

The project includes:

1. data profiling,
2. sensor filtering based on profiling,
3. train/validation/test split preparation,
4. value-level quality control,
5. realistic missing-data scenario generation,
6. classical baseline imputation,
7. denoising autoencoder training,
8. graph denoising autoencoder training with PyTorch Geometric,
9. global and local model comparison.

---

## Repository structure

The repository is organized as a flat project directory. All scripts use configuration variables at the top of the file and assume paths relative to the repository root.

```text
.
├── README.md
├── requirements.txt
│
├── data_profiling.py
├── filter_sensors_based_on_profiling.py
├── prepare_model_data.py
├── apply_value_quality_mask.py
├── data_masking.py
├── baseline_imputation.py
├── basic_autoencoder.py
├── pyg_graph_autoencoder.py
├── analyze_dae_vs_graph_dae.py
│
├── combined_sensors_2022-10-01_2022-12-31.csv
├── combined_sensors_2022-10-01_2022-12-31_filtered.csv
├── combined_sensors_2023-01-01_2023-01-31.csv
├── combined_sensors_2023-01-01_2023-01-31_filtered.csv
│
├── sensor_quality_profile.csv
├── gap_runs_nan_and_zero.csv
└── edges.csv
```

---

## Data files

### `combined_sensors_2022-10-01_2022-12-31.csv`

Initial traffic sensor data for October, November, and December 2022.


### `combined_sensors_2023-01-01_2023-01-31.csv`

Initial traffic sensor data for January 2023.

January 2023 is used as the final held-out test period.


### `sensor_quality_profile.csv`

Sensor quality profile generated during the profiling stage.

It documents sensor-level quality indicators such as:

- missing-value rate,
- zero-value rate,
- longest NaN run,
- longest zero run,
- all-NaN sensors,
- all-zero sensors,
- mostly-zero sensors,
- sensors with long zero-runs.


### `combined_sensors_2022-10-01_2022-12-31_filtered.csv`

Filtered October-December 2022 data containing the final selected sensor panel.


### `combined_sensors_2023-01-01_2023-01-31_filtered.csv`

Filtered January 2023 data containing the same selected sensor panel.


### `gap_runs_nan_and_zero.csv`

Run-length file generated during profiling.

It contains detected NaN and zero runs. The project uses this file to derive empirical outage lengths for the synthetic missing-data scenario.

The final missing-data scenario samples zero-runs with:

```text
length >= 6
length <= 144
```

At 10-minute resolution, this corresponds to outage lengths from 1 hour to 1 day.


### `edges.csv`

Sparse road-network graph used by the PyTorch Geometric model.

The graph was prepared manually in QGIS and represents local road-network connections between sensors.


---

## Data split

The final experiment uses the following temporal split:

| Split | Period | Purpose |
|---|---|---|
| Train | 2022-10-01 to 2022-11-30 | Model training |
| Validation | 2022-12-01 to 2022-12-31 | Model validation |
| Test | 2023-01-01 to 2023-01-31 | Final held-out evaluation |

The final selected sensor panel contains 109 traffic sensors.

---

## Workflow

### 0. Profile the raw data

Script:

```bash
python data_profiling.py
```

Inputs:

```text
combined_sensors_2022-10-01_2022-12-31.csv
combined_sensors_2023-01-01_2023-01-31.csv
```

Outputs:

```text
sensor_quality_profile.csv
gap_runs_nan_and_zero.csv
```

Purpose:

This step analyzes sensor quality and missing/zero run lengths. It provides the profiling artifacts used later for sensor selection, value-quality decisions, and empirical outage-length sampling.

Important note:

Long zero-runs are not automatically treated as missing values in the original data. They are used only to estimate realistic outage durations. Synthetic gaps with these durations are later applied to valid reference values.

---

### 1. Filter sensors based on profiling

Script:

```bash
python filter_sensors_based_on_profiling.py
```

Inputs:

```text
combined_sensors_2022-10-01_2022-12-31.csv
combined_sensors_2023-01-01_2023-01-31.csv
sensor_quality_profile.csv
```

Outputs:

```text
combined_sensors_2022-10-01_2022-12-31_filtered.csv
combined_sensors_2023-01-01_2023-01-31_filtered.csv
```

Purpose:

This step creates the final stable sensor panel used in the experiment.

The project uses conservative filtering based on the profiling results. The final selected panel contains 109 sensors.

The filtering criteria are based on sensor quality status and zero-run behavior. The goal is to keep sensors suitable for controlled imputation experiments and exclude sensors with severe quality issues.

---

### 2. Prepare train, validation, and test data

Script:

```bash
python prepare_model_data.py
```

Inputs:

```text
combined_sensors_2022-10-01_2022-12-31_filtered.csv
combined_sensors_2023-01-01_2023-01-31_filtered.csv
```

Outputs:

```text
model_data/train_filtered.csv
model_data/val_filtered.csv
model_data/test_filtered.csv
```

Purpose:

This script creates the final temporal split from already filtered sensor-panel files.

It also performs sanity checks:

- train/validation/test columns are identical,
- timestamps follow a complete 10-minute grid,
- there are no duplicated timestamps,
- there are no missing timestamps,
- there are no NaN values before value-level QC.

---

### 3. Apply value-level quality control

Script:

```bash
python apply_value_quality_mask.py
```

Inputs:

```text
model_data/train_filtered.csv
model_data/val_filtered.csv
model_data/test_filtered.csv
```

Outputs:

```text
model_data_qc/train_filtered_qc.csv
model_data_qc/val_filtered_qc.csv
model_data_qc/test_filtered_qc.csv

model_data_qc/train_valid_value_mask.csv
model_data_qc/val_valid_value_mask.csv
model_data_qc/test_valid_value_mask.csv

model_data_qc/invalid_points_all.csv
model_data_qc/value_outlier_report_by_sensor.csv
model_data_qc/value_outlier_report_by_split.csv
```

Purpose:

This step removes physically implausible traffic measurements from the valid ground truth. Invalid values are replaced with `NaN` and excluded from artificial masking and evaluation.

The project focuses on missing-data imputation, not anomaly correction. Therefore, physically implausible values are not treated as valid reference values.

---

### 4. Generate synthetic missing-data scenario

Script:

```bash
python data_masking.py
```

Inputs:

```text
model_data_qc/train_filtered_qc.csv
model_data_qc/val_filtered_qc.csv
model_data_qc/test_filtered_qc.csv
gap_runs_nan_and_zero.csv
```

Outputs:

```text
masked_data/scenario_A_empirical_outage_qc/train/masked.csv
masked_data/scenario_A_empirical_outage_qc/train/mask_map.csv
masked_data/scenario_A_empirical_outage_qc/train/removed_values.csv
masked_data/scenario_A_empirical_outage_qc/train/removed_values_long.csv
masked_data/scenario_A_empirical_outage_qc/train/sampled_outages.csv

masked_data/scenario_A_empirical_outage_qc/val/masked.csv
masked_data/scenario_A_empirical_outage_qc/val/mask_map.csv
masked_data/scenario_A_empirical_outage_qc/val/removed_values.csv
masked_data/scenario_A_empirical_outage_qc/val/removed_values_long.csv
masked_data/scenario_A_empirical_outage_qc/val/sampled_outages.csv

masked_data/scenario_A_empirical_outage_qc/test/masked.csv
masked_data/scenario_A_empirical_outage_qc/test/mask_map.csv
masked_data/scenario_A_empirical_outage_qc/test/removed_values.csv
masked_data/scenario_A_empirical_outage_qc/test/removed_values_long.csv
masked_data/scenario_A_empirical_outage_qc/test/sampled_outages.csv

masked_data/scenario_A_empirical_outage_qc/gap_distribution_used.csv
masked_data/scenario_A_empirical_outage_qc/gap_distribution_summary.csv
```

Purpose:

This script creates realistic artificial outage masks. The generated missing values are applied only to valid reference values.

Important files:

| File | Meaning |
|---|---|
| `masked.csv` | Data with artificial missing values. |
| `mask_map.csv` | Binary mask of artificially removed values. |
| `sampled_outages.csv` | List of sampled outages with sensor, start, end, and length. |
| `gap_distribution_used.csv` | Gap lengths used for sampling. |

---

### 5. Run classical imputation baselines

Script:

```bash
python baseline_imputation.py
```

Inputs:

```text
model_data_qc/train_filtered_qc.csv
model_data_qc/val_filtered_qc.csv
model_data_qc/test_filtered_qc.csv

masked_data/scenario_A_empirical_outage_qc/
```

Outputs:

```text
baseline_results/scenario_A_empirical_outage_qc/baseline_metrics.csv
baseline_results/scenario_A_empirical_outage_qc/imputed_files/
```

Implemented methods:

- mean per sensor,
- median per sensor,
- forward/backward fill,
- linear interpolation,
- KNN Imputer.

---

### 6. Train the standard Denoising Autoencoder

Script:

```bash
python basic_autoencoder.py
```

Inputs:

```text
model_data_qc/train_filtered_qc.csv
model_data_qc/val_filtered_qc.csv
model_data_qc/test_filtered_qc.csv

masked_data/scenario_A_empirical_outage_qc/val/masked.csv
masked_data/scenario_A_empirical_outage_qc/val/mask_map.csv

masked_data/scenario_A_empirical_outage_qc/test/masked.csv
masked_data/scenario_A_empirical_outage_qc/test/mask_map.csv

masked_data/scenario_A_empirical_outage_qc/gap_distribution_used.csv
```

Outputs:

```text
dae_results/scenario_A_empirical_outage_qc/dae_model.pt
dae_results/scenario_A_empirical_outage_qc/dae_metrics.csv
dae_results/scenario_A_empirical_outage_qc/training_history.csv
dae_results/scenario_A_empirical_outage_qc/test_imputed_dae.csv
```

Final DAE configuration:

```text
WINDOW_SIZE = 72
LATENT_DIM = 128
OBSERVED_LOSS_WEIGHT = 0.02
```

---

### 7. Train the PyTorch Geometric Graph DAE

Script:

```bash
python pyg_graph_autoencoder.py
```

Inputs:

```text
model_data_qc/train_filtered_qc.csv
model_data_qc/val_filtered_qc.csv
model_data_qc/test_filtered_qc.csv

masked_data/scenario_A_empirical_outage_qc/val/masked.csv
masked_data/scenario_A_empirical_outage_qc/val/mask_map.csv

masked_data/scenario_A_empirical_outage_qc/test/masked.csv
masked_data/scenario_A_empirical_outage_qc/test/mask_map.csv

masked_data/scenario_A_empirical_outage_qc/gap_distribution_used.csv

edges.csv
```

Outputs:

```text
pyg_gdae_results/scenario_A_empirical_outage_qc/pyg_graph_dae_model.pt
pyg_gdae_results/scenario_A_empirical_outage_qc/pyg_graph_dae_metrics.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/pyg_graph_dae_training_history.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/test_imputed_pyg_graph_dae.csv
```

Purpose:

This script trains the final graph-based denoising autoencoder.

The model uses:

```text
GCNConv
GCNConv
MLP autoencoder
```

The graph model incorporates local road-network context from `edges.csv`.

---

### 8. Analyze DAE vs PyG Graph DAE

Script:

```bash
python analyze_dae_vs_graph_dae.py
```

Inputs:

```text
model_data_qc/test_filtered_qc.csv

masked_data/scenario_A_empirical_outage_qc/test/mask_map.csv
masked_data/scenario_A_empirical_outage_qc/test/sampled_outages.csv

dae_results/scenario_A_empirical_outage_qc/test_imputed_dae.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/test_imputed_pyg_graph_dae.csv

edges.csv
```

Outputs:

```text
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/summary_dae_vs_graph_dae.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/point_comparison_dae_vs_graph_dae.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/sensor_comparison_dae_vs_graph_dae.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/outage_comparison_dae_vs_graph_dae.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/degree_bucket_comparison.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/top_sensors_graph_better.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/top_sensors_graph_worse.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/top_outages_graph_better.csv
pyg_gdae_results/scenario_A_empirical_outage_qc/dae_vs_pyg_graph_dae_analysis/top_outages_graph_worse.csv
```

Purpose:

This script compares the standard DAE and the PyG Graph DAE:

- globally,
- per sensor,
- per outage,
- by node degree in the graph.

---

## Evaluation

All models are evaluated only on artificially masked values, where the true reference values are known.

Metrics:

- MAE: mean absolute error,
- RMSE: root mean squared error.

MAPE is not used because traffic values may contain zeros.

---

## Final results

| Method | Test MAE | Test RMSE |
|---|---:|---:|
| Mean per sensor | 117.40 | 161.15 |
| Median per sensor | 116.76 | 163.26 |
| Forward/backward fill | 93.38 | 158.70 |
| Linear interpolation | 58.12 | 89.90 |
| KNN Imputer | 33.22 | 62.86 |
| Denoising Autoencoder | 34.44 | 61.26 |
| Graph DAE, PyTorch Geometric GCNConv | 33.03 | 59.58 |

The final PyTorch Geometric Graph DAE achieved the best overall performance.

Compared with the standard DAE, the graph model reduced:

```text
MAE  by 1.40
RMSE by 1.68
```

Local analysis showed that the graph model improved 59 out of 109 sensor-level outages and performed better on 52.6% of masked points.

---

## Installation

A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Minimal required packages:

```text
numpy
pandas
scikit-learn
matplotlib
torch
torch-geometric
networkx
```

---
