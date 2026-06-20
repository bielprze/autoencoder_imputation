from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv


# =====================
# CONFIG
# =====================

TIMESTAMP_COL = "timestamp"

TRAIN_CLEAN = "model_data_qc/train_filtered_qc.csv"
VAL_CLEAN = "model_data_qc/val_filtered_qc.csv"
TEST_CLEAN = "model_data_qc/test_filtered_qc.csv"

SCENARIO_DIR = Path("masked_data/scenario_A_empirical_outage_qc")

VAL_MASKED = SCENARIO_DIR / "val" / "masked.csv"
VAL_MASK_MAP = SCENARIO_DIR / "val" / "mask_map.csv"

TEST_MASKED = SCENARIO_DIR / "test" / "masked.csv"
TEST_MASK_MAP = SCENARIO_DIR / "test" / "mask_map.csv"

GAP_DISTRIBUTION = SCENARIO_DIR / "gap_distribution_used.csv"

EDGES_FILE = "edges_filtered.csv"
EDGE_SOURCE_COL = "sensor_id"
EDGE_TARGET_COL = "neighbor_id"

OUTPUT_DIR = Path("pyg_gdae_results/scenario_A_empirical_outage_qc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT = OUTPUT_DIR / "pyg_graph_dae_model.pt"
METRICS_OUTPUT = OUTPUT_DIR / "pyg_graph_dae_metrics.csv"
HISTORY_OUTPUT = OUTPUT_DIR / "pyg_graph_dae_training_history.csv"
TEST_IMPUTED_OUTPUT = OUTPUT_DIR / "test_imputed_pyg_graph_dae.csv"

RANDOM_SEED = 42

WINDOW_SIZE = 72
TRAIN_STRIDE = 12
EVAL_STRIDE = 12

BATCH_SIZE = 32
N_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

GCN_HIDDEN_DIM_1 = 8
GCN_HIDDEN_DIM_2 = 4

HIDDEN_DIM_1 = 2048
HIDDEN_DIM_2 = 512
LATENT_DIM = 128
DROPOUT = 0.10

OUTAGES_PER_WINDOW = 6
OBSERVED_LOSS_WEIGHT = 0.02

MAX_MASK_SAMPLING_ATTEMPTS = 200

UNDIRECTED_GRAPH = True
ADD_SELF_LOOPS_IN_GCN = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# REPRODUCIBILITY
# =====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================
# DATA LOADING
# =====================

def load_clean_data(path: str | Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in df.columns if c != TIMESTAMP_COL]

    if len(sensors) == 0:
        raise ValueError(f"No sensor columns found in {path}")

    X = df[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    return df, sensors, X


def load_masked_data(
    masked_path: str | Path,
    mask_map_path: str | Path,
    expected_sensors: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    masked_df = pd.read_csv(masked_path, parse_dates=[TIMESTAMP_COL])
    masked_df = masked_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    sensors = [c for c in masked_df.columns if c != TIMESTAMP_COL]

    if sensors != expected_sensors:
        raise ValueError(f"Sensor columns do not match expected sensors in {masked_path}")

    X_masked = masked_df[sensors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    mask_map_df = pd.read_csv(mask_map_path)

    if list(mask_map_df.columns) != expected_sensors:
        raise ValueError(f"Mask map columns do not match expected sensors in {mask_map_path}")

    artificial_mask = mask_map_df.to_numpy(dtype=bool)

    if not artificial_mask.any():
        raise ValueError(f"No artificial missing positions found in {mask_map_path}")

    return masked_df, X_masked, artificial_mask


def load_gap_lengths(path: str | Path) -> np.ndarray:
    gaps = pd.read_csv(path)

    if "length" not in gaps.columns:
        raise ValueError("gap_distribution_used.csv must contain column: length")

    lengths = gaps["length"].astype(int).to_numpy()

    if len(lengths) == 0:
        raise ValueError("No gap lengths found.")

    return lengths


# =====================
# GRAPH
# =====================

def load_edge_index(
    edges_path: str | Path,
    sensors: list[str],
) -> torch.Tensor:
    edges = pd.read_csv(edges_path)

    if EDGE_SOURCE_COL in edges.columns and EDGE_TARGET_COL in edges.columns:
        source_col = EDGE_SOURCE_COL
        target_col = EDGE_TARGET_COL
    else:
        if len(edges.columns) < 2:
            raise ValueError("Edges file must contain at least two columns.")

        print(
            f"WARNING: columns {EDGE_SOURCE_COL}/{EDGE_TARGET_COL} not found. "
            f"Using first two columns: {edges.columns[0]}, {edges.columns[1]}"
        )

        source_col = edges.columns[0]
        target_col = edges.columns[1]

    sensor_to_idx = {sensor: idx for idx, sensor in enumerate(sensors)}

    edge_pairs = []
    skipped_edges = 0

    for _, row in edges.iterrows():
        src = row[source_col]
        dst = row[target_col]

        if src not in sensor_to_idx or dst not in sensor_to_idx:
            skipped_edges += 1
            continue

        i = sensor_to_idx[src]
        j = sensor_to_idx[dst]

        edge_pairs.append((i, j))

        if UNDIRECTED_GRAPH:
            edge_pairs.append((j, i))

    if len(edge_pairs) == 0:
        raise ValueError("No graph edges matched selected sensors.")

    edge_pairs = sorted(set(edge_pairs))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    n_sensors = len(sensors)

    degree = np.zeros(n_sensors, dtype=int)
    for src, dst in edge_pairs:
        degree[src] += 1

    print("\nGraph")
    print("-----")
    print(f"edges file:          {edges_path}")
    print(f"directed edge_index: {edge_index.shape[1]}")
    print(f"skipped edges:       {skipped_edges}")
    print(f"sensors/nodes:       {n_sensors}")
    print(f"undirected:          {UNDIRECTED_GRAPH}")
    print(f"GCN self loops:      {ADD_SELF_LOOPS_IN_GCN}")
    print(f"degree min:          {degree.min()}")
    print(f"degree median:       {np.median(degree):.1f}")
    print(f"degree mean:         {degree.mean():.2f}")
    print(f"degree max:          {degree.max()}")

    return edge_index


def make_batched_edge_index(
    base_edge_index: torch.Tensor,
    num_graphs: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Repeats the same graph num_graphs times with node index offsets.

    One graph = one time slice from one sample.
    If batch has B windows and each has T timestamps, then num_graphs = B * T.
    """
    base_edge_index = base_edge_index.to(device)

    edge_count = base_edge_index.shape[1]

    offsets = (
        torch.arange(num_graphs, device=device, dtype=torch.long)
        .repeat_interleave(edge_count)
        * num_nodes
    )

    repeated = base_edge_index.repeat(1, num_graphs)
    repeated = repeated + offsets.unsqueeze(0)

    return repeated


# =====================
# SCALING
# =====================

class StandardScalerPerSensor:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)

        if np.isnan(self.mean_).any():
            bad = np.where(np.isnan(self.mean_))[0]
            raise ValueError(f"Some sensors have only NaN values in train data: {bad}")

        self.std_[np.isnan(self.std_)] = 1.0
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return (X_scaled * self.std_ + self.mean_).astype(np.float32)


# =====================
# WINDOWING AND MASKING
# =====================

def make_window_starts(n_rows: int, window_size: int, stride: int) -> list[int]:
    if n_rows < window_size:
        raise ValueError("n_rows is smaller than WINDOW_SIZE.")

    starts = list(range(0, n_rows - window_size + 1, stride))

    last_start = n_rows - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def find_valid_start_indices(valid_positions: np.ndarray, length: int) -> np.ndarray:
    if length > len(valid_positions):
        return np.array([], dtype=int)

    valid_int = valid_positions.astype(int)

    window_sums = np.convolve(
        valid_int,
        np.ones(length, dtype=int),
        mode="valid",
    )

    starts = np.where(window_sums == length)[0]

    return starts.astype(int)


def create_training_mask(
    valid_target_mask: np.ndarray,
    gap_lengths: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    window_size, n_sensors = valid_target_mask.shape
    mask = np.zeros((window_size, n_sensors), dtype=bool)

    possible_lengths = gap_lengths[gap_lengths <= window_size]

    if len(possible_lengths) == 0:
        raise ValueError("No possible gap lengths for current WINDOW_SIZE.")

    created_outages = 0
    attempts = 0

    while created_outages < OUTAGES_PER_WINDOW and attempts < MAX_MASK_SAMPLING_ATTEMPTS:
        attempts += 1

        sensor_idx = int(rng.integers(0, n_sensors))
        length = int(rng.choice(possible_lengths))

        available_positions = valid_target_mask[:, sensor_idx] & (~mask[:, sensor_idx])

        starts = find_valid_start_indices(
            valid_positions=available_positions,
            length=length,
        )

        if len(starts) == 0:
            continue

        start_idx = int(rng.choice(starts))
        end_idx = start_idx + length

        mask[start_idx:end_idx, sensor_idx] = True
        created_outages += 1

    if not mask.any():
        valid_positions = np.where(valid_target_mask)

        if len(valid_positions[0]) == 0:
            return mask

        chosen = int(rng.integers(0, len(valid_positions[0])))
        row_idx = valid_positions[0][chosen]
        sensor_idx = valid_positions[1][chosen]
        mask[row_idx, sensor_idx] = True

    return mask


class PyGGraphDenoisingWindowDataset(Dataset):
    def __init__(
        self,
        X_scaled: np.ndarray,
        gap_lengths: np.ndarray,
        window_size: int,
        stride: int,
        seed: int,
    ):
        self.X_scaled = X_scaled
        self.gap_lengths = gap_lengths
        self.window_size = window_size
        self.starts = make_window_starts(
            n_rows=len(X_scaled),
            window_size=window_size,
            stride=stride,
        )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        end = start + self.window_size

        clean = self.X_scaled[start:end].copy()

        valid_target_mask = np.isfinite(clean)

        artificial_missing_mask = create_training_mask(
            valid_target_mask=valid_target_mask,
            gap_lengths=self.gap_lengths,
            rng=self.rng,
        )

        observed_mask = valid_target_mask & (~artificial_missing_mask)

        corrupted = clean.copy()
        corrupted[~observed_mask] = 0.0

        target = clean.copy()
        target[~valid_target_mask] = 0.0

        input_features = np.stack(
            [
                corrupted,
                observed_mask.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        target_flat = target.reshape(-1).astype(np.float32)
        artificial_missing_flat = artificial_missing_mask.reshape(-1).astype(np.float32)
        observed_flat = observed_mask.reshape(-1).astype(np.float32)
        valid_target_flat = valid_target_mask.reshape(-1).astype(np.float32)

        return (
            torch.from_numpy(input_features),
            torch.from_numpy(target_flat),
            torch.from_numpy(artificial_missing_flat),
            torch.from_numpy(observed_flat),
            torch.from_numpy(valid_target_flat),
        )


# =====================
# MODEL
# =====================

class PyGGraphDenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_sensors: int,
        edge_index: torch.Tensor,
    ):
        super().__init__()

        self.window_size = window_size
        self.n_sensors = n_sensors

        self.register_buffer("base_edge_index", edge_index.long())

        self.gcn_1 = GCNConv(
            in_channels=2,
            out_channels=GCN_HIDDEN_DIM_1,
            add_self_loops=ADD_SELF_LOOPS_IN_GCN,
            normalize=True,
        )

        self.gcn_2 = GCNConv(
            in_channels=GCN_HIDDEN_DIM_1,
            out_channels=GCN_HIDDEN_DIM_2,
            add_self_loops=ADD_SELF_LOOPS_IN_GCN,
            normalize=True,
        )

        flattened_graph_dim = window_size * n_sensors * GCN_HIDDEN_DIM_2
        output_dim = window_size * n_sensors

        self.encoder_decoder = nn.Sequential(
            nn.Linear(flattened_graph_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM_2, LATENT_DIM),
            nn.ReLU(),

            nn.Linear(LATENT_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM_1, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            B x T x N x 2

        We treat every timestamp in every batch item as a separate graph
        with the same edge_index.
        """
        batch_size, window_size, n_sensors, n_features = x.shape

        if window_size != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {window_size}")

        if n_sensors != self.n_sensors:
            raise ValueError(f"Expected n_sensors={self.n_sensors}, got {n_sensors}")

        num_graphs = batch_size * window_size

        x_nodes = x.reshape(num_graphs * n_sensors, n_features)

        batched_edge_index = make_batched_edge_index(
            base_edge_index=self.base_edge_index,
            num_graphs=num_graphs,
            num_nodes=n_sensors,
            device=x.device,
        )

        h = self.gcn_1(x_nodes, batched_edge_index)
        h = torch.relu(h)

        h = self.gcn_2(h, batched_edge_index)
        h = torch.relu(h)

        h = h.reshape(batch_size, window_size, n_sensors, GCN_HIDDEN_DIM_2)
        h = h.reshape(batch_size, -1)

        out = self.encoder_decoder(h)

        return out


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    artificial_missing_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    valid_target_mask: torch.Tensor,
) -> torch.Tensor:
    missing_loss_mask = artificial_missing_mask * valid_target_mask
    observed_loss_mask = observed_mask * valid_target_mask

    missing_count = missing_loss_mask.sum().clamp(min=1.0)
    observed_count = observed_loss_mask.sum().clamp(min=1.0)

    missing_loss = (
        ((prediction - target) ** 2) * missing_loss_mask
    ).sum() / missing_count

    observed_loss = (
        ((prediction - target) ** 2) * observed_loss_mask
    ).sum() / observed_count

    return missing_loss + OBSERVED_LOSS_WEIGHT * observed_loss


# =====================
# TRAINING
# =====================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
) -> pd.DataFrame:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            (
                inputs,
                targets,
                artificial_missing_masks,
                observed_masks,
                valid_target_masks,
            ) = batch

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            artificial_missing_masks = artificial_missing_masks.to(DEVICE)
            observed_masks = observed_masks.to(DEVICE)
            valid_target_masks = valid_target_masks.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = masked_mse_loss(
                prediction=predictions,
                target=targets,
                artificial_missing_mask=artificial_missing_masks,
                observed_mask=observed_masks,
                valid_target_mask=valid_target_masks,
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        epoch_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
        })

        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:03d} | train_loss = {epoch_loss:.6f}")

    return pd.DataFrame(history)


# =====================
# EVALUATION
# =====================

def prepare_eval_input(window_scaled_with_nan: np.ndarray) -> np.ndarray:
    observed_mask = np.isfinite(window_scaled_with_nan)

    corrupted = window_scaled_with_nan.copy()
    corrupted[~observed_mask] = 0.0

    input_features = np.stack(
        [
            corrupted,
            observed_mask.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    return input_features


def reconstruct_full_series(
    model: nn.Module,
    X_masked_scaled_with_nan: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    n_rows, n_sensors = X_masked_scaled_with_nan.shape

    starts = make_window_starts(
        n_rows=n_rows,
        window_size=window_size,
        stride=stride,
    )

    pred_sum = np.zeros((n_rows, n_sensors), dtype=np.float32)
    pred_count = np.zeros((n_rows, n_sensors), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for start in starts:
            end = start + window_size

            window = X_masked_scaled_with_nan[start:end]
            input_features = prepare_eval_input(window)

            input_tensor = torch.from_numpy(input_features).unsqueeze(0).to(DEVICE)

            pred_flat = model(input_tensor).cpu().numpy()[0]
            pred_window = pred_flat.reshape(window_size, n_sensors)

            pred_sum[start:end] += pred_window
            pred_count[start:end] += 1.0

    if (pred_count == 0).any():
        raise ValueError("Some positions were not covered by evaluation windows.")

    return pred_sum / pred_count


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

    n_masked_values = int(mask.sum())
    n_valid_eval_values = int(valid_eval.sum())
    n_skipped_invalid_eval_values = n_masked_values - n_valid_eval_values

    if n_valid_eval_values == 0:
        raise ValueError(
            "No valid evaluation positions found. "
            "Check mask_map and QC-filtered clean data."
        )

    errors = y_pred[valid_eval] - y_true[valid_eval]
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "n_masked_values": n_masked_values,
        "n_valid_eval_values": n_valid_eval_values,
        "n_skipped_invalid_eval_values": n_skipped_invalid_eval_values,
        "mae": float(mae),
        "rmse": float(rmse),
    }


def evaluate_split(
    split_name: str,
    model: nn.Module,
    clean_X: np.ndarray,
    masked_X: np.ndarray,
    artificial_mask: np.ndarray,
    scaler: StandardScalerPerSensor,
) -> tuple[dict, np.ndarray]:
    masked_scaled = scaler.transform(masked_X)

    pred_scaled = reconstruct_full_series(
        model=model,
        X_masked_scaled_with_nan=masked_scaled,
        window_size=WINDOW_SIZE,
        stride=EVAL_STRIDE,
    )

    pred_original = scaler.inverse_transform(pred_scaled)

    metrics = compute_metrics(
        y_true=clean_X,
        y_pred=pred_original,
        mask=artificial_mask,
    )

    metrics = {
        "split": split_name,
        "method": "pyg_graph_denoising_autoencoder",
        **metrics,
    }

    return metrics, pred_original


def make_imputed_dataframe(
    timestamps: pd.Series,
    sensors: list[str],
    masked_X: np.ndarray,
    pred_X: np.ndarray,
    artificial_mask: np.ndarray,
) -> pd.DataFrame:
    imputed_X = masked_X.copy()

    # Fill only artificial missing values.
    # QC-invalid NaNs remain NaN.
    imputed_X[artificial_mask] = pred_X[artificial_mask]

    return pd.concat(
        [
            pd.DataFrame({TIMESTAMP_COL: timestamps}),
            pd.DataFrame(imputed_X, columns=sensors),
        ],
        axis=1,
    )


# =====================
# MAIN
# =====================

def main() -> None:
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    train_df, sensors, train_X = load_clean_data(TRAIN_CLEAN)
    val_df, val_sensors, val_clean_X = load_clean_data(VAL_CLEAN)
    test_df, test_sensors, test_clean_X = load_clean_data(TEST_CLEAN)

    if sensors != val_sensors or sensors != test_sensors:
        raise ValueError("Train/val/test sensors differ.")

    n_sensors = len(sensors)

    print(f"Train rows: {len(train_X)}")
    print(f"Val rows:   {len(val_clean_X)}")
    print(f"Test rows:  {len(test_clean_X)}")
    print(f"Sensors:    {n_sensors}")

    print("\nQC NaN counts")
    print("-------------")
    print(f"train NaN values: {int(np.isnan(train_X).sum())}")
    print(f"val NaN values:   {int(np.isnan(val_clean_X).sum())}")
    print(f"test NaN values:  {int(np.isnan(test_clean_X).sum())}")

    edge_index = load_edge_index(
        edges_path=EDGES_FILE,
        sensors=sensors,
    )

    gap_lengths = load_gap_lengths(GAP_DISTRIBUTION)

    scaler = StandardScalerPerSensor()
    scaler.fit(train_X)

    train_X_scaled = scaler.transform(train_X)

    train_dataset = PyGGraphDenoisingWindowDataset(
        X_scaled=train_X_scaled,
        gap_lengths=gap_lengths,
        window_size=WINDOW_SIZE,
        stride=TRAIN_STRIDE,
        seed=RANDOM_SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = PyGGraphDenoisingAutoencoder(
        window_size=WINDOW_SIZE,
        n_sensors=n_sensors,
        edge_index=edge_index,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel")
    print("-----")
    print(f"trainable parameters: {n_params:,}")
    print(f"window size:          {WINDOW_SIZE}")
    print(f"GCN hidden dims:      {GCN_HIDDEN_DIM_1}, {GCN_HIDDEN_DIM_2}")
    print(f"latent dim:           {LATENT_DIM}")

    print("\nTraining PyG Graph Denoising Autoencoder")
    print("----------------------------------------")

    history = train_model(
        model=model,
        train_loader=train_loader,
    )

    history.to_csv(HISTORY_OUTPUT, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sensors": sensors,
            "window_size": WINDOW_SIZE,
            "edge_index": edge_index,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.std_,
            "config": {
                "WINDOW_SIZE": WINDOW_SIZE,
                "TRAIN_STRIDE": TRAIN_STRIDE,
                "EVAL_STRIDE": EVAL_STRIDE,
                "GCN_HIDDEN_DIM_1": GCN_HIDDEN_DIM_1,
                "GCN_HIDDEN_DIM_2": GCN_HIDDEN_DIM_2,
                "LATENT_DIM": LATENT_DIM,
                "OUTAGES_PER_WINDOW": OUTAGES_PER_WINDOW,
                "N_EPOCHS": N_EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LEARNING_RATE": LEARNING_RATE,
                "OBSERVED_LOSS_WEIGHT": OBSERVED_LOSS_WEIGHT,
                "ADD_SELF_LOOPS_IN_GCN": ADD_SELF_LOOPS_IN_GCN,
            },
        },
        MODEL_OUTPUT,
    )

    print(f"\nSaved model to: {MODEL_OUTPUT}")

    val_masked_df, val_masked_X, val_artificial_mask = load_masked_data(
        masked_path=VAL_MASKED,
        mask_map_path=VAL_MASK_MAP,
        expected_sensors=sensors,
    )

    test_masked_df, test_masked_X, test_artificial_mask = load_masked_data(
        masked_path=TEST_MASKED,
        mask_map_path=TEST_MASK_MAP,
        expected_sensors=sensors,
    )

    if not val_df[TIMESTAMP_COL].equals(val_masked_df[TIMESTAMP_COL]):
        raise ValueError("Val timestamps differ between clean and masked data.")

    if not test_df[TIMESTAMP_COL].equals(test_masked_df[TIMESTAMP_COL]):
        raise ValueError("Test timestamps differ between clean and masked data.")

    print("\nEvaluating")
    print("----------")

    val_metrics, _ = evaluate_split(
        split_name="val",
        model=model,
        clean_X=val_clean_X,
        masked_X=val_masked_X,
        artificial_mask=val_artificial_mask,
        scaler=scaler,
    )

    test_metrics, test_pred_X = evaluate_split(
        split_name="test",
        model=model,
        clean_X=test_clean_X,
        masked_X=test_masked_X,
        artificial_mask=test_artificial_mask,
        scaler=scaler,
    )

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(METRICS_OUTPUT, index=False)

    print(metrics_df)
    print(f"\nSaved metrics to: {METRICS_OUTPUT}")

    test_imputed_df = make_imputed_dataframe(
        timestamps=test_masked_df[TIMESTAMP_COL],
        sensors=sensors,
        masked_X=test_masked_X,
        pred_X=test_pred_X,
        artificial_mask=test_artificial_mask,
    )

    test_imputed_df.to_csv(TEST_IMPUTED_OUTPUT, index=False)
    print(f"Saved test imputed data to: {TEST_IMPUTED_OUTPUT}")


if __name__ == "__main__":
    main()