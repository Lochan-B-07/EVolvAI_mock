"""
generative_core/data_loader.py
================================
PyTorch Dataset + DataLoader for the EVolvAI physics-informed TCN-VAE.

Two operating modes (auto-detected):
  1. REAL   — reads data/processed/train_data.parquet produced by
              data_pipeline/preprocess.py
  2. SYNTH  — falls back to Lochan's EV schedule generator when no
              parquet exists (same schedule generator used in train.py)

In both cases every batch yields:
    x     : FloatTensor [B, NUM_FEATURES, SEQ_LEN]   (Conv1d channel-first)
    cond  : FloatTensor [B, COND_DIM]                 (dynamic per date)

Condition vector layout (COND_DIM = 6):
    C[0]  temperature_anomaly   float  normalised deviation from monthly mean
    C[1]  ev_multiplier         float  1.0 = today's fleet size
    C[2]  solar_availability    float  0=night/cloudy, 1=full sun
    C[3]  weekend               0/1
    C[4]  holiday               0/1
    C[5]  traffic_index         float  0=empty roads, 1=gridlock rush hour
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import config

log = logging.getLogger(__name__)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _znorm(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise an array, safe for zero-variance inputs."""
    std = arr.std()
    if std < 1e-8:
        return arr - arr.mean()
    return (arr - arr.mean()) / (std + 1e-8)


def _date_to_condition(date_str: str, temp_anomaly: float = 0.0) -> list:
    """
    Derive a 6-D condition vector from a date string (YYYY-MM-DD).

    Fields computed deterministically so training is reproducible:
      weekend flag   – 1 if Saturday or Sunday
      solar          – approximated from declination (summer = more solar)
      traffic        – weekday mornings = high, weekends = low
      holiday        – not computed here (always 0; extend with holidays lib)
      temp_anomaly   – fed in from weather data if available, else 0.0
      ev_multiplier  – always 1.0 during training (only changed at generation)
    """
    import datetime
    try:
        d = datetime.date.fromisoformat(date_str)
    except Exception:
        return [0.0, 1.0, 0.5, 0.0, 0.0, 0.5]

    dow       = d.weekday()         # 0=Mon … 6=Sun
    is_wknd   = float(dow >= 5)

    # Solar availability: rough proxy using day-of-year (summer = high)
    doy       = d.timetuple().tm_yday
    solar     = 0.5 + 0.5 * np.sin((doy - 80) * 2 * np.pi / 365)

    # Traffic: high on weekday rush hours (simplified to day-level proxy)
    traffic   = 0.85 if dow < 5 else 0.30

    return [
        float(np.clip(temp_anomaly, -1.0, 1.0)),   # C[0] temperature anomaly
        1.0,                                         # C[1] EV multiplier (baseline)
        float(np.clip(solar, 0.0, 1.0)),            # C[2] solar availability
        is_wknd,                                     # C[3] weekend flag
        0.0,                                         # C[4] holiday flag
        float(traffic),                              # C[5] traffic index
    ]


# ─── Real parquet loader ──────────────────────────────────────────────────────

def _load_parquet(num_nodes: int, seq_len: int) -> Optional[Tuple[np.ndarray, list]]:
    """
    Load and validate the preprocessed parquet.

    Returns
    -------
    (demand, dates)
        demand : float32 ndarray [N_days, seq_len, num_nodes]  (kW)
        dates  : list[str]  one entry per day
    or None on failure.
    """
    if not os.path.isfile(config.DATA_PATH):
        log.info("Parquet not found at %s — using synthetic fallback.", config.DATA_PATH)
        return None

    try:
        import pandas as pd
        df = pd.read_parquet(config.DATA_PATH)

        required = {"date", "hour", "node_id", "demand_kw"}
        missing  = required - set(df.columns)
        if missing:
            log.warning("Parquet missing columns %s — falling back.", missing)
            return None

        # Pivot: rows = (date, hour), columns = node_id
        pivot = df.pivot_table(
            index=["date", "hour"], columns="node_id", values="demand_kw",
            aggfunc="sum", fill_value=0.0,
        )

        if pivot.shape[1] != num_nodes:
            log.warning(
                "Parquet has %d nodes but NUM_NODES=%d — falling back.",
                pivot.shape[1], num_nodes,
            )
            return None

        # Group by date → [N, seq_len, num_nodes]
        dates_in_pivot = pivot.index.get_level_values("date").unique().tolist()
        days = []
        valid_dates = []
        for date in sorted(dates_in_pivot):
            try:
                day = pivot.loc[date]                     # [24, num_nodes]
                if len(day) != seq_len:
                    continue                              # skip incomplete days
                days.append(day.values.astype(np.float32))
                valid_dates.append(date)
            except Exception:
                continue

        if len(days) < 2:
            log.warning("Too few valid days in parquet — falling back.")
            return None

        demand = np.stack(days, axis=0)                  # [N, 24, num_nodes]
        log.info(
            "Parquet loaded: %d days × %d h × %d nodes",
            len(valid_dates), seq_len, num_nodes,
        )
        return demand, valid_dates

    except Exception as exc:
        log.warning("Parquet load failed (%s) — falling back.", exc)
        return None


# ─── Synthetic fallback (Lochan generator) ────────────────────────────────────

def _generate_synthetic(num_samples: int, num_nodes: int,
                         seq_len: int) -> Tuple[np.ndarray, list]:
    """Lochan-style synthetic demand + dummy date list."""
    import datetime
    rng      = np.random.default_rng(42)
    base_dt  = datetime.date(2022, 1, 1)

    base    = rng.uniform(10, 100, (num_samples, seq_len, num_nodes))
    diurnal = np.clip(
        [1 + np.sin((h - 12) * np.pi / 12) for h in range(seq_len)],
        0.5, 2.0,
    ).reshape(1, seq_len, 1)
    demand = (base * diurnal).astype(np.float32)

    dates = [(base_dt + datetime.timedelta(days=i)).isoformat()
             for i in range(num_samples)]

    log.info("Synthetic fallback: %d samples generated.", num_samples)
    return demand, dates


# ─── Dataset class ────────────────────────────────────────────────────────────

class EVDemandDataset(Dataset):
    """
    PyTorch Dataset of 24-hour EV demand + weather profiles.

    Each sample is a tuple (x, cond):
        x    : FloatTensor [NUM_FEATURES, SEQ_LEN]   channel-first for Conv1d
        cond : FloatTensor [COND_DIM]                dynamic per-date condition

    Automatically reads the real parquet if available, otherwise falls back
    to the synthetic generator — the rest of the pipeline sees no difference.
    """

    def __init__(self,
                 num_nodes:   int = config.NUM_NODES,
                 seq_len:     int = config.SEQ_LEN,
                 num_samples: int = config.NUM_SAMPLES):

        result = _load_parquet(num_nodes, seq_len)

        if result is not None:
            demand, self._dates = result
            self._source = "real"
        else:
            demand, self._dates = _generate_synthetic(num_samples, num_nodes, seq_len)
            self._source = "synthetic"

        n_days = demand.shape[0]

        # Build weather channels: [N, seq_len, NUM_WEATHER_FEATURES]
        # Real weather integration: replace this block with a weather parquet join.
        rng     = np.random.default_rng(0)
        weather = rng.uniform(-10, 40,
                              (n_days, seq_len, config.NUM_WEATHER_FEATURES)).astype(np.float32)

        try:
            import pandas as pd
            import os
            weather_parquet_path = os.path.join(config.PROJECT_ROOT, "weather_data.parquet")
            if os.path.exists(weather_parquet_path):
                df_w = pd.read_parquet(weather_parquet_path)
                feat_cols = ['temperature_c', 'precipitation_mm', 'solar_availability', 'traffic_index']
                if all(col in df_w.columns for col in ['date', 'hour'] + feat_cols):
                    df_w = df_w.sort_values(by=['date', 'hour'])
                    for i, d in enumerate(self._dates):
                        day_weather = df_w[df_w['date'] == d]
                        if len(day_weather) == seq_len:
                            weather[i] = day_weather[feat_cols].values.astype(np.float32)
                    log.info("Integrated real weather data from weather_data.parquet")
        except Exception as e:
            log.warning(f"Failed to integrate real weather data from parquet: {e}")

        # Z-score normalise demand and weather separately
        demand_n  = _znorm(demand)
        weather_n = _znorm(weather)

        # [N, seq_len, num_nodes + NUM_WEATHER_FEATURES]
        self._data = np.concatenate([demand_n, weather_n], axis=-1).astype(np.float32)

        # Pre-compute condition vectors for every day
        self._conds = np.array(
            [_date_to_condition(d) for d in self._dates],
            dtype=np.float32,
        )  # [N, COND_DIM]

        log.info(
            "EVDemandDataset ready [%s]: %d samples  shape=%s  cond_dim=%d",
            self._source, n_days, self._data.shape, self._conds.shape[1],
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x    : FloatTensor [NUM_FEATURES, SEQ_LEN]   (channel-first)
        cond : FloatTensor [COND_DIM]
        """
        x    = torch.from_numpy(self._data[idx])       # [seq_len, features]
        x    = x.permute(1, 0)                         # [features, seq_len] ← Conv1d
        cond = torch.from_numpy(self._conds[idx])
        return x, cond

    @property
    def source(self) -> str:
        return self._source


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloader(batch_size: int = config.BATCH_SIZE,
                   num_nodes:  int = config.NUM_NODES,
                   shuffle:    bool = True) -> DataLoader:
    """
    Build and return a shuffled training DataLoader.

    Returns
    -------
    DataLoader yielding (x, cond) tuples:
        x    : [B, NUM_FEATURES, SEQ_LEN]
        cond : [B, COND_DIM]
    """
    dataset = EVDemandDataset(num_nodes=num_nodes)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=0, pin_memory=False)
    log.info("DataLoader: source=%s  batches/epoch=%d", dataset.source, len(loader))
    return loader
