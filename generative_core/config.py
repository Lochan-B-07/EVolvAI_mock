"""
generative_core/config.py
=========================
Single source of truth for every tunable constant, file path, and scenario
definition in the GCD-VAE pipeline.

Changing a value here propagates to every module automatically.
Do NOT hardcode any of these values elsewhere in the codebase.
"""

import os

# ─── Project Root ─────────────────────────────────────────────────────────────
# Resolved from this file's location so the code is CWD-independent.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── Grid Topology ────────────────────────────────────────────────────────────
NUM_NODES = 32
"""int: Number of spatial load nodes in the power-grid graph (IEEE 33-Bus, 
excluding substation node 1). Changing this requires retraining from scratch."""

SEQ_LEN = 24
"""int: Temporal resolution – one profile = 24 hourly slots (midnight to 23:00)."""

# ─── Physics Penalties ────────────────────────────────────────────────────────
LAMBDA_VOLT    = 0.0
LAMBDA_THERMAL = 0.0
LAMBDA_XFMR   = 0.0
"""float: Physics penalty weights — set to 0.0 during isolation training
to prevent the decoder collapsing to zero-output local minima.
Re-enable (1.0, 2.0, 2.0) in a second training phase once the VAE converges."""

NUM_WEATHER_FEATURES = 4
"""int: Number of weather/environment channels appended to each charging sample.
  Channel 0 – temperature (°C)
  Channel 1 – precipitation (mm/hr)
  Channel 2 – wind speed (m/s)
  Channel 3 – traffic index (0.0=empty, 1.0=gridlock)
"""

NUM_FEATURES = NUM_NODES + NUM_WEATHER_FEATURES
"""int: Total feature dimension fed into the TCN encoder.
= NUM_NODES (charging demand per node) + NUM_WEATHER_FEATURES."""


# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_SAMPLES = 1000
"""int: Synthetic sample count used when no real parquet is available."""

BATCH_SIZE = 64
"""int: Mini-batch size for the training DataLoader.
Raised to 64 — larger batches provide more stable gradient estimates
for the bigger model and give ~31 batches/epoch with 2000 training days."""

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_data.parquet")
"""str: Absolute path to the preprocessed training parquet.
Expected schema (one row per node per hour):
  date (str/date), hour (int 0-23), node_id (str), demand_kw (float),
  temperature_c (float), precipitation_mm (float), wind_mps (float)
"""


# ─── Model Architecture ───────────────────────────────────────────────────────
TCN_CHANNELS = [128, 256, 256, 256, 256]
"""list[int]: Output channel width for each TCN residual block.
Dilations are [1, 2, 4, 8, 16] — 5 blocks give a receptive field of
(2-1)*(1+2+4+8+16) = 31 steps, covering the full 24-hour window.
With 256 channels the parameter count jumps from ~280k to ~8M,
requiring GPU for comfortable training."""

KERNEL_SIZE = 3
"""int: Causal convolution kernel size.  3 + dilated stacking increases
receptive field coverage compared to kernel-2 at the same depth."""

DROPOUT = 0.15
"""float: Dropout probability after each activation in TCN blocks.
Slightly lower than before because the larger model is already regularised
by its depth; keep at 0.2 if overfitting on small real datasets."""

LATENT_DIM = 128
"""int: Dimensionality of the VAE latent space Z.
Raised from 16 → 128 to give the encoder enough bandwidth to represent
complex spatial loading patterns across 32 grid nodes without blurring."""

COND_DIM = 6
"""int: Length of the condition vector C injected into the decoder.
Must equal the length of every `condition` list in SCENARIOS and BASELINE_CONDITION.
  C[0] – temperature anomaly (float)
  C[1] – EV electrification multiplier (float, 1.0 = today's fleet)
  C[2] – solar generation availability (float, 0.0–1.0)
  C[3] – weekend flag (0 or 1)
  C[4] – holiday flag (0 or 1)
  C[5] – traffic index (float, 0.0 = empty roads, 1.0 = rush-hour gridlock)
"""

DECODER_HIDDEN = 512
"""int: Hidden layer width in the decoder fully-connected block.
Raised from 128 → 512 to match the wider TCN and latent space."""


# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE = 1e-3
"""float: Adam initial learning rate.  Drop to 1e-4 if training loss explodes."""

EPOCHS = 150
"""int: Training epochs.  150 for isolation phase; raise to 1000+ for full physics run."""

KLD_WEIGHT = 1.0
"""float: β in the β-VAE loss: L = MSE + β * KLD.
Raise to 2.0 if generated scenarios look blurry / lack diversity.
Lower to 0.5 if the model collapses to a point estimate."""

GRAD_CLIP_NORM = 1.0
"""float: Maximum L2 norm for gradient clipping.
Prevents loss explosion on early epochs with random data."""


# ─── Baseline Condition ───────────────────────────────────────────────────────
BASELINE_CONDITION = [0.0, 1.0, 1.0, 0.0, 0.0, 0.5]
"""list[float]: Default condition used *during training*.
Represents a typical weekday with no weather anomaly, today's fleet size,
and average traffic (0.5 = midday level).
Length must equal COND_DIM."""


# ─── Counterfactual Scenarios ─────────────────────────────────────────────────
SCENARIOS = {
    "extreme_winter_storm": {
        "description": "Extreme winter storm + 2.5x fleet electrification surge",
        # temp anomaly=1.0 (severe cold), EV mult=2.5, no solar, weekday, not holiday, rush hour
        "condition": [1.0, 2.5, 0.0, 0.0, 0.0, 0.85],
    },
    "summer_peak": {
        "description": "High summer temperatures + 1.5x electrification",
        # mild heat anomaly=0.5, EV mult=1.5, full solar, weekday, PM rush
        "condition": [0.5, 1.5, 1.0, 0.0, 0.0, 0.90],
    },
    "full_electrification": {
        "description": "Normal weather + 3.0x full fleet electrification",
        # no temp anomaly, 3x fleet (full ICE→EV conversion), bright day, moderate traffic
        "condition": [0.0, 3.0, 1.0, 0.0, 0.0, 0.65],
    },
    "extreme_winter_v2": {
        "description": "Winter storm + full electrification + weekend",
        # worst-case: cold storm, 2.5x fleet, no solar, weekend demand pattern, low traffic
        "condition": [1.0, 2.5, 0.0, 1.0, 0.0, 0.30],
    },
    "rush_hour_gridlock": {
        "description": "Peak rush hour with 2x fleet electrification",
        # no weather anomaly, 2x fleet, partial solar, weekday, max traffic
        "condition": [0.0, 2.0, 0.5, 0.0, 0.0, 1.0],
    },
}
"""dict: Named counterfactual scenarios.
Each entry maps a scenario name to:
  description (str) – human-readable label used in UI and reports.
  condition (list[float], len=COND_DIM) – the intervention trigger vector.

Every .npy output file in output/ corresponds to one scenario key here.
To add a scenario, append an entry and re-run `python run.py generate`.
"""


# ─── Output Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
"""str: Directory where all generated tensors and checkpoints are written."""

MOCK_TENSOR_PATH = os.path.join(OUTPUT_DIR, "mock_demand_tensor.npy")
"""str: Path for the fast-handoff mock tensor ([24, NUM_NODES] float32 kW array)."""

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "gcvae_model.pt")
"""str: Path for the trained VAE state-dict checkpoint."""


# ─── Traffic Data ─────────────────────────────────────────────────────────────
TRAFFIC_DATA_PATH = os.path.join(OUTPUT_DIR, "..", "data", "processed", "traffic_index_tensor.npy")
"""str: Path to the preprocessed traffic index tensor [24, NUM_NODES] float32."""

BOULDER_BBOX = {
    "north": 40.0950,
    "south": 39.9530,
    "east": -105.1780,
    "west": -105.3010,
}
"""dict: Bounding box for Boulder, CO (WGS84).  Shared with traffic_preprocess."""
