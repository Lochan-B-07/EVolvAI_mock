import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.traffic_preprocess import build_hourly_traffic_tensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_PARQUET = PROC_DIR / "train_data.parquet"
NUM_NODES = 32

def parse_acn_data(csv_path: str) -> pd.DataFrame:
    """Load ACN data and parse to sessions with date and start hour."""
    log.info("Loading ACN CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    col_map = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "")
        lc = col.lower().replace(" ", "")
        if "connect" in lc and "time" in lc and "disconnect" not in lc: col_map[col] = "connectionTime"
        elif "disconnect" in lc and "time" in lc: col_map[col] = "disconnectTime"
        elif "kwh" in lc or "energy" in lc: col_map[col] = "kWhDelivered"
        elif "driver" in lc or "user" in lc or "ev" in lc: col_map[col] = "userID"
    df = df.rename(columns=col_map)
    
    # Try to find a date column if "Charging Date" (ACN style) isn't there
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    
    if date_col:
        # Check if we have the columns and they are strings (not already datetimes)
        for col in ["connectionTime", "disconnectTime"]:
            if col in df.columns:
                # If it's a Series (single column) and looks like just time (no date separator)
                series = df[col]
                if not isinstance(series, pd.Series): # Handle multi-column naming collision just in case
                    series = series.iloc[:, 0]
                
                if series.dtype == object:
                    sample = str(series.dropna().iloc[0]) if not series.dropna().empty else ""
                    if ":" in sample and "-" not in sample and "/" not in sample:
                        df[col] = df[date_col].astype(str) + " " + series.astype(str)
            
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, errors="coerce")
    df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")
    
    # Adjust disconnect time for cross-midnight sessions
    cross_midnight = df["disconnectTime"] < df["connectionTime"]
    df.loc[cross_midnight, "disconnectTime"] += pd.Timedelta(days=1)
    
    df = df.dropna(subset=["connectionTime", "disconnectTime", "kWhDelivered"])
    df = df[df["kWhDelivered"] > 0]
    
    df["duration_h"] = ((df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600).clip(lower=0.5)
    df["avg_kw"] = df["kWhDelivered"] / df["duration_h"]
    df["start_date"] = df["connectionTime"].dt.date
    df["start_hour"] = df["connectionTime"].dt.hour
    
    log.info("Parsed %d valid ACN sessions.", len(df))
    return df

def generate_mock_acn_data(days=100) -> pd.DataFrame:
    """Generate fake ACN-like sessions if CSV is missing."""
    log.info("Generating mock ACN data for %d historical days...", days)
    rng = np.random.default_rng(42)
    base_date = pd.Timestamp("2021-01-01")
    records = []
    for d in range(days):
        date_str = (base_date + pd.Timedelta(days=d)).date()
        n_sessions = int(rng.normal(150, 30))
        for _ in range(max(10, n_sessions)):
            hour = int(rng.normal(12, 4)) % 24
            duration = rng.uniform(1.0, 8.0)
            kwh = rng.uniform(5.0, 50.0)
            records.append({
                "start_date": date_str,
                "start_hour": hour,
                "duration_h": duration,
                "avg_kw": kwh / duration,
                "kWhDelivered": kwh
            })
    return pd.DataFrame(records)

def bootstrap_daily_scenarios(historical_df: pd.DataFrame, num_scenarios: int = 5000, num_nodes: int = NUM_NODES) -> pd.DataFrame:
    """Bootstrap 5000 daily scenarios grouped by day, distributing via traffic index."""
    log.info("Group Caltech ACN data by DAY...")
    # Group by historical day
    daily_groups = dict(tuple(historical_df.groupby("start_date")))
    historical_days = list(daily_groups.keys())
    
    log.info("Building hourly traffic tensor for distribution...")
    # traffic_index: (24, NUM_NODES)
    traffic_index = build_hourly_traffic_tensor(num_nodes=num_nodes, try_real_data=False)
    # Convert traffic index to probabilities per hour
    # For each hour, prob of node = traffic / sum(traffic)
    hour_node_probs = np.zeros_like(traffic_index)
    for h in range(24):
        s = traffic_index[h].sum()
        if s > 0:
            hour_node_probs[h] = traffic_index[h] / s
        else:
            hour_node_probs[h] = 1.0 / num_nodes
            
    log.info("Bootstrapping %d daily scenarios...", num_scenarios)
    rng = np.random.default_rng(1337)
    
    # Pre-allocate output arrays for speed
    # We need 1 record per (day, hour, node)
    out_dates = []
    out_hours = []
    out_nodes = []
    out_kw = []
    
    base_gen_date = pd.Timestamp("2025-01-01")
    
    nodes = np.arange(num_nodes)
    
    for i in range(num_scenarios):
        # Sample a historical day uniformly
        sampled_day = rng.choice(historical_days)
        sessions_df = daily_groups[sampled_day]
        
        gen_date_str = (base_gen_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Accumulate demand for this day
        # shape: (24, 32)
        day_demand = np.zeros((24, num_nodes))
        
        # Distribute individual sessions across nodes
        for _, row in sessions_df.iterrows():
            sh = int(row["start_hour"])
            dur = int(round(row["duration_h"]))
            kw = row["avg_kw"]
            
            # Select node based on traffic index at start hour
            chosen_node = rng.choice(nodes, p=hour_node_probs[sh])
            
            # Distribute power across duration
            for h_offset in range(dur):
                h = (sh + h_offset) % 24
                day_demand[h, chosen_node] += kw
                
        # Append to output
        for h in range(24):
            for n in range(num_nodes):
                out_dates.append(gen_date_str)
                out_hours.append(h)
                out_nodes.append(f"node_{n:02d}")
                out_kw.append(day_demand[h, n])
                
        if (i+1) % 500 == 0:
            log.info("  Generated %d / %d scenarios", i+1, num_scenarios)
            
    log.info("Compiling daily scenarios into DataFrame...")
    final_df = pd.DataFrame({
        "date": out_dates,
        "hour": out_hours,
        "node_id": out_nodes,
        "demand_kw": out_kw
    })
    
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to ACN data CSV")
    parser.add_argument("--scenarios", type=int, default=5000, help="Number of scenarios to generate")
    args = parser.parse_args()
    
    if args.csv and os.path.exists(args.csv):
        df = parse_acn_data(args.csv)
    else:
        if args.csv:
            log.warning("CSV not found: %s. Falling back to mock data.", args.csv)
        df = generate_mock_acn_data()
        
    final_df = bootstrap_daily_scenarios(df, num_scenarios=args.scenarios, num_nodes=NUM_NODES)
    
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / 1e6
    log.info("Saved bootstrapped dataset → %s  (%.2f MB)", OUT_PARQUET, size_mb)
    log.info("Shape: %s, Dates: %d, Nodes: %d", final_df.shape, final_df["date"].nunique(), final_df["node_id"].nunique())
    
if __name__ == "__main__":
    main()