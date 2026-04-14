"""
data_pipeline/preprocess.py
============================
Real-world EV charging data pipeline using the Caltech ACN-Data API.

Outputs
-------
  data/processed/train_data.parquet
  Schema: date (str), hour (int 0-23), node_id (str), demand_kw (float)

Usage
-----
  # Download via API (requires free token from ev.caltech.edu):
  python data_pipeline/preprocess.py --api-token YOUR_TOKEN

  # Parse a previously downloaded ACN session CSV:
  python data_pipeline/preprocess.py --csv data/raw/acndata_sessions.csv

  # Quick synthetic fallback (no credentials needed, for testing):
  python data_pipeline/preprocess.py --synthetic --days 500

The output parquet is automatically picked up by generative_core/data_loader.py.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR     = REPO_ROOT / "data" / "raw"
PROC_DIR    = REPO_ROOT / "data" / "processed"
OUT_PARQUET = PROC_DIR / "train_data.parquet"

NUM_NODES   = 32     # IEEE 33-bus load nodes

# ─── ACN-Data API ─────────────────────────────────────────────────────────────
ACN_BASE_URL = "https://ev.caltech.edu/api/v1/sessions/caltech"

ACN_FIELDS = [
    "connectionTime", "disconnectTime",
    "kWhDelivered", "userID",
]


def download_acn_sessions(api_token: str, site: str = "caltech",
                          max_results: int = 50_000) -> pd.DataFrame:
    """
    Download EV charging sessions from the Caltech ACN-Data API.

    Parameters
    ----------
    api_token : str
        Free token from https://ev.caltech.edu  (register → API Keys)
    site : str
        'caltech' or 'jpl'
    max_results : int
        Upper cap on sessions fetched (API paginates at 1000/page).

    Returns
    -------
    pd.DataFrame with columns:
        connectionTime (UTC ISO), disconnectTime (UTC ISO), kWhDelivered (float),
        userID (str)
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    url     = f"https://ev.caltech.edu/api/v1/sessions/{site}"
    headers = {"Authorization": f"Bearer {api_token}"}
    params  = {"page": 1, "page_size": 1000}

    all_sessions = []
    log.info("Fetching ACN sessions from %s …", url)

    while len(all_sessions) < max_results:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise PermissionError(
                "ACN API 401 Unauthorized. Check your token at ev.caltech.edu."
            )
        resp.raise_for_status()
        data = resp.json()

        docs = data.get("_items", [])
        if not docs:
            break
        all_sessions.extend(docs)
        log.info("  fetched %d / target %d", len(all_sessions), max_results)

        if not data.get("_links", {}).get("next"):
            break
        params["page"] += 1
        time.sleep(0.1)   # be polite to the API

    log.info("Total sessions downloaded: %d", len(all_sessions))
    df = pd.DataFrame(all_sessions)
    df = df.rename(columns={
        "_id": "session_id",
        "connectionTime": "connectionTime",
        "disconnectTime":  "disconnectTime",
        "kWhDelivered":    "kWhDelivered",
        "userID":          "userID",
    })
    return df


def load_acn_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a pre-downloaded ACN session CSV.

    ACN CSVs typically have headers like:
        Session ID, Connection Time, Disconnect Time, kWh Delivered, User ID, ...

    We normalise whatever header names are present.
    """
    log.info("Loading ACN CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Flexible column-name mapping
    col_map = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "")
        if "connection" in lc:
            col_map[col] = "connectionTime"
        elif "disconnect" in lc:
            col_map[col] = "disconnectTime"
        elif "kwh" in lc or "energy" in lc:
            col_map[col] = "kWhDelivered"
        elif "user" in lc or "ev" in lc:
            col_map[col] = "userID"

    df = df.rename(columns=col_map)
    log.info("CSV loaded: %d sessions", len(df))
    return df


# ─── Core aggregation ─────────────────────────────────────────────────────────

def _stable_node(user_id, num_nodes: int = NUM_NODES) -> int:
    """Deterministically map a userID to a grid node [0, num_nodes-1]."""
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    return h % num_nodes


def sessions_to_hourly_demand(df: pd.DataFrame,
                               num_nodes: int = NUM_NODES) -> pd.DataFrame:
    """
    Aggregate per-session records into 24-hour hourly kW demand profiles
    mapped across num_nodes grid nodes.

    Strategy
    --------
    1. Parse connection timestamps → extract date and start hour.
    2. Approximate average charging power = kWhDelivered / session_duration_h.
    3. Assign each session to a node via stable hash of userID.
    4. Pivot: one row per (date, hour, node_id), value = total kW.
    5. Forward-fill missing hours (no EV plugged in → 0 kW).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: connectionTime, disconnectTime, kWhDelivered, userID

    Returns
    -------
    pd.DataFrame  columns: date (str), hour (int), node_id (str), demand_kw (float)
    """
    df = df.copy()

    # Parse timestamps
    df["connectionTime"]  = pd.to_datetime(df["connectionTime"],  utc=True, errors="coerce")
    df["disconnectTime"]  = pd.to_datetime(df["disconnectTime"],  utc=True, errors="coerce")
    df["kWhDelivered"]    = pd.to_numeric(df["kWhDelivered"],     errors="coerce")

    df = df.dropna(subset=["connectionTime", "disconnectTime", "kWhDelivered"])
    df = df[df["kWhDelivered"] > 0]

    # Session duration in hours (clip at 0.5 h minimum to avoid /0 or spikes)
    df["duration_h"] = (
        (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    ).clip(lower=0.5)

    df["avg_kw"]  = df["kWhDelivered"] / df["duration_h"]
    df["date"]    = df["connectionTime"].dt.date.astype(str)
    df["hour"]    = df["connectionTime"].dt.hour
    df["node_id"] = df["userID"].apply(lambda u: f"node_{_stable_node(u, num_nodes):02d}")

    # Aggregate: sum charging power per (date, hour, node)
    agg = (
        df.groupby(["date", "hour", "node_id"])["avg_kw"]
        .sum()
        .reset_index()
        .rename(columns={"avg_kw": "demand_kw"})
    )

    # Ensure every node has all 24 hours (fill 0 for quiet hours)
    all_dates = agg["date"].unique()
    all_nodes = [f"node_{i:02d}" for i in range(num_nodes)]
    full_idx  = pd.MultiIndex.from_product(
        [all_dates, range(24), all_nodes],
        names=["date", "hour", "node_id"],
    )
    agg = (
        agg.set_index(["date", "hour", "node_id"])
           .reindex(full_idx, fill_value=0.0)
           .reset_index()
    )

    log.info(
        "Aggregated: %d unique dates × 24 h × %d nodes = %d rows",
        len(all_dates), num_nodes, len(agg),
    )
    return agg


# ─── Synthetic fallback ───────────────────────────────────────────────────────

def generate_synthetic_parquet(num_days: int = 500,
                                num_nodes: int = NUM_NODES,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset when no ACN credentials are available.

    Uses a diurnal sine-wave pattern (peak at hour 18) with per-node
    Dirichlet weights and Gaussian noise.  Produces the same schema
    as the real ACN pipeline so the DataLoader sees no difference.
    """
    log.info("Generating synthetic parquet (%d days, %d nodes)…", num_days, num_nodes)
    rng = np.random.default_rng(seed)

    base_date = pd.Timestamp("2022-01-01")
    records   = []

    for d in range(num_days):
        date_str = (base_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        weights  = rng.dirichlet(np.ones(num_nodes) * 0.8)
        for h in range(24):
            # Diurnal profile: peak ~18:00
            diurnal = 0.5 + 1.5 * max(0.0, np.sin((h - 8) * np.pi / 14))
            fleet   = 45.0 * diurnal                          # kW for whole fleet
            noise   = rng.normal(1.0, 0.12)
            node_kw = (fleet * noise * weights).clip(min=0.0)
            for i in range(num_nodes):
                records.append({
                    "date":      date_str,
                    "hour":      h,
                    "node_id":   f"node_{i:02d}",
                    "demand_kw": float(node_kw[i]),
                })

    df = pd.DataFrame(records)
    log.info("Synthetic parquet: %d rows", len(df))
    return df


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1e6
    log.info("Saved → %s  (%.2f MB)", OUT_PARQUET, size_mb)
    log.info("Schema: %s", list(df.columns))
    log.info("Dates:  %d unique", df["date"].nunique())
    log.info("kW range: [%.3f, %.3f]", df["demand_kw"].min(), df["demand_kw"].max())


# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="EVolvAI — EV demand data pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--api-token", metavar="TOKEN",
                       help="Caltech ACN-Data API Bearer token (ev.caltech.edu)")
    group.add_argument("--csv", metavar="PATH",
                       help="Path to a downloaded ACN session CSV file")
    group.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic data (no credentials needed)")

    p.add_argument("--site",     default="caltech",
                   choices=["caltech", "jpl"],
                   help="ACN site (API mode only)")
    p.add_argument("--max-sessions", type=int, default=50_000,
                   help="Max sessions to pull from API")
    p.add_argument("--days",     type=int, default=500,
                   help="Days to synthesise (--synthetic mode only)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--nodes",    type=int, default=NUM_NODES,
                   help="Number of grid nodes to map sessions to")
    return p.parse_args()


def main():
    args = get_args()

    # ── Acquire raw session data ────────────────────────────────────────────
    if args.synthetic:
        df = generate_synthetic_parquet(args.days, args.nodes, args.seed)
    elif args.csv:
        raw = load_acn_csv(args.csv)
        df  = sessions_to_hourly_demand(raw, args.nodes)
    else:
        # API download
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        sessions = download_acn_sessions(args.api_token, args.site, args.max_sessions)
        # Cache raw download
        raw_path = RAW_DIR / f"acn_{args.site}_raw.csv"
        sessions.to_csv(raw_path, index=False)
        log.info("Raw sessions cached → %s", raw_path)
        df = sessions_to_hourly_demand(sessions, args.nodes)

    save_parquet(df)
    log.info("✅  Pipeline complete. Run train.py to start training.")


if __name__ == "__main__":
    # Make sure repo root is importable
    sys.path.insert(0, str(REPO_ROOT))
    main()
