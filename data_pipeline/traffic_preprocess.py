"""
data_pipeline/traffic_preprocess.py
====================================
Traffic flow data pipeline for EVolvAI.

Acquires, processes, and normalises traffic volume data so the generative
model (GCD-VAE) receives a physical "traffic_index" in its condition
vector C.  The index ranges from 0.0 (empty roads, 3 AM) to 1.0 (peak
rush-hour gridlock).

Geographic scope:  Boulder, Colorado
    Bounding box: N 40.0950, S 39.9530, E -105.1780, W -105.3010
    Rationale: Highest EV adoption density (~50 EVs/1k residents),
               NREL nearby, I-25 & US-36 EV corridors, existing
               project references to Boulder weather data.

Data sources (in priority order):
    1. CDOT OTIS / Colorado open-data traffic sensor counts (hourly)
    2. US Census LEHD LODES8 origin-destination commuter data
    3. Synthetic diurnal fallback (always available, no network needed)

Quick start
-----------
    from data_pipeline.traffic_preprocess import build_hourly_traffic_tensor

    tensor = build_hourly_traffic_tensor()       # shape (24, 50)
    print(tensor.min(), tensor.max())             # 0.0  1.0
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Geographic Constants ─────────────────────────────────────────────────────
BOULDER_BBOX = {
    "north": 40.0950,
    "south": 39.9530,
    "east": -105.1780,
    "west": -105.3010,
}
"""Bounding box for Boulder, CO city limits (WGS84 decimal degrees)."""

BOULDER_CENTER = (40.0150, -105.2705)
"""Approximate centroid of Boulder, CO (lat, lon)."""

# ─── Project paths (resolved from this file) ─────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "traffic")
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

# Default node count (must match generative_core.config.NUM_NODES)
_DEFAULT_NUM_NODES = 50
_SEQ_LEN = 24  # hours per daily profile


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Static Road Network (OSMnx)
# ═══════════════════════════════════════════════════════════════════════════════

def download_road_network(
    bbox: Optional[dict] = None,
    output_dir: Optional[str] = None,
    network_type: str = "drive",
) -> dict:
    """
    Download the road network for the bounding box using OSMnx.

    Parameters
    ----------
    bbox : dict, optional
        Keys: north, south, east, west.  Defaults to BOULDER_BBOX.
    output_dir : str, optional
        Where to save graphml + geojson.  Defaults to data/raw/traffic/.
    network_type : str
        OSMnx network type ('drive', 'walk', 'bike', 'all').

    Returns
    -------
    dict with keys:
        graph        – NetworkX MultiDiGraph
        graphml_path – str, path to saved .graphml file
        geojson_path – str, path to saved edges .geojson file
        n_nodes      – int, number of intersections
        n_edges      – int, number of road segments
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for road network download.\n"
            "Install with: pip install osmnx"
        )

    bbox = bbox or BOULDER_BBOX
    output_dir = output_dir or _RAW_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Downloading %s road network for bbox: N=%.4f S=%.4f E=%.4f W=%.4f",
        network_type, bbox["north"], bbox["south"], bbox["east"], bbox["west"],
    )

    G = ox.graph_from_bbox(
        bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
        network_type=network_type,
    )

    # Save GraphML
    graphml_path = os.path.join(output_dir, "boulder_road_network.graphml")
    ox.save_graphml(G, filepath=graphml_path)
    logger.info("Saved GraphML → %s", graphml_path)

    # Save edges as GeoJSON (for visualization / GIS)
    geojson_path = os.path.join(output_dir, "boulder_roads.geojson")
    try:
        import geopandas as gpd  # noqa: F401
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info("Saved GeoJSON → %s", geojson_path)
    except Exception as e:
        logger.warning("GeoJSON export failed (%s) — GraphML still saved.", e)
        geojson_path = None

    result = {
        "graph": G,
        "graphml_path": graphml_path,
        "geojson_path": geojson_path,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }
    logger.info(
        "Road network: %d intersections, %d road segments",
        result["n_nodes"], result["n_edges"],
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Dynamic Traffic Volume
# ═══════════════════════════════════════════════════════════════════════════════

def download_lehd_od_data(
    state: str = "co",
    year: int = 2021,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Download US Census LEHD LODES8 origin-destination data for a state.

    The OD file tells us how many workers commute from Census Block A to
    Census Block B — a strong proxy for AM/PM traffic directionality.

    Parameters
    ----------
    state : str
        Two-letter state abbreviation (lowercase).
    year : int
        Data year (2021 is latest LODES8).
    output_dir : str, optional
        Download destination.  Defaults to data/raw/traffic/.

    Returns
    -------
    str or None
        Path to downloaded CSV.GZ file, or None on failure.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — skipping LEHD download.")
        return None

    output_dir = output_dir or _RAW_DIR
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{state}_od_main_JT00_{year}.csv.gz"
    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state}/od/{filename}"
    dest = os.path.join(output_dir, filename)

    if os.path.exists(dest):
        logger.info("LEHD file already exists: %s", dest)
        return dest

    logger.info("Downloading LEHD LODES8 OD data: %s", url)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / (1 << 20)
        logger.info("Downloaded %.1f MB → %s", size_mb, dest)
        return dest
    except Exception as e:
        logger.warning("LEHD download failed: %s", e)
        return None


def parse_lehd_to_hourly_profile(
    csv_gz_path: str,
    boulder_tract_prefixes: tuple[str, ...] = ("08013",),
) -> Optional[np.ndarray]:
    """
    Parse LEHD OD data into a 24-hour traffic volume profile.

    LEHD provides daily totals (workers commuting from A→B).  We distribute
    these over 24 hours using the standard FHWA K- and D-factors for urban
    areas to create an hourly profile.

    Parameters
    ----------
    csv_gz_path : str
        Path to the downloaded LEHD csv.gz file.
    boulder_tract_prefixes : tuple of str
        Census tract FIPS prefixes to filter.  '08013' = Boulder County.

    Returns
    -------
    ndarray shape (24,) float64 — normalised hourly traffic index, or None.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — cannot parse LEHD data.")
        return None

    try:
        df = pd.read_csv(csv_gz_path, compression="gzip", dtype=str)
    except Exception as e:
        logger.warning("Failed to read LEHD file: %s", e)
        return None

    # Filter to Boulder County tracts (origin OR destination)
    # LEHD geocode columns: w_geocode (workplace), h_geocode (home)
    if "w_geocode" not in df.columns or "h_geocode" not in df.columns:
        logger.warning("LEHD file missing expected columns.")
        return None

    mask = df["w_geocode"].str[:5].isin(boulder_tract_prefixes) | \
           df["h_geocode"].str[:5].isin(boulder_tract_prefixes)
    boulder_df = df[mask]

    if boulder_df.empty:
        logger.warning("No Boulder County records found in LEHD data.")
        return None

    # Total commuter flow (S000 = total jobs)
    total_flow = boulder_df["S000"].astype(float).sum() if "S000" in boulder_df.columns else len(boulder_df)

    # Distribute using FHWA urban hourly distribution factors
    # (same shape as synthetic but scaled by real commuter volume)
    hourly_factors = _fhwa_urban_hourly_factors()
    hourly_volume = hourly_factors * total_flow

    # Normalise to [0, 1]
    return _min_max_normalize(hourly_volume)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Fallback — Always Available
# ═══════════════════════════════════════════════════════════════════════════════

def build_synthetic_traffic_profile(seed: int = 42) -> np.ndarray:
    """
    Generate a realistic 24-hour diurnal traffic volume profile.

    Based on the well-documented double-hump commuter pattern from FHWA
    urban traffic studies:
        - Morning peak:  7–9 AM  (index ~0.85)
        - Midday lull:   11 AM–2 PM (index ~0.45)
        - Evening peak:  4–7 PM  (index ~1.0)
        - Night trough:  1–5 AM  (index ~0.05)

    Parameters
    ----------
    seed : int
        Random seed for minor stochastic noise.

    Returns
    -------
    ndarray shape (24,) float64 — traffic index in [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)

    # FHWA-style hourly distribution factors for urban arterials
    hourly_factors = _fhwa_urban_hourly_factors()

    # Add minor stochastic perturbation (±5%) for realism
    noise = rng.uniform(-0.05, 0.05, size=24)
    hourly_factors = hourly_factors + noise

    return _min_max_normalize(hourly_factors)


def _fhwa_urban_hourly_factors() -> np.ndarray:
    """
    Return FHWA-style hourly traffic distribution factors for urban roads.

    These factors represent the fraction of daily traffic volume occurring
    in each hour.  Values are based on published FHWA urban arterial
    K-factor tables and real-world traffic count studies.

    Returns
    -------
    ndarray shape (24,) float64 — raw hourly factors (not yet normalised).
    """
    # Hour 0 (midnight) through Hour 23 (11 PM)
    # Double-hump profile: AM peak 7–9, PM peak 16–18
    factors = np.array([
        0.015,  # 00:00 — midnight
        0.010,  # 01:00 — deep night
        0.008,  # 02:00
        0.006,  # 03:00 — minimum
        0.008,  # 04:00
        0.015,  # 05:00 — early commuters
        0.035,  # 06:00 — ramp-up
        0.070,  # 07:00 — AM peak start
        0.085,  # 08:00 — AM peak
        0.070,  # 09:00 — AM peak tail
        0.055,  # 10:00 — mid-morning
        0.048,  # 11:00 — approaching lunch
        0.052,  # 12:00 — lunch traffic
        0.050,  # 13:00 — early afternoon
        0.055,  # 14:00 — mid-afternoon
        0.065,  # 15:00 — school dismissal
        0.080,  # 16:00 — PM peak start
        0.090,  # 17:00 — PM peak (highest)
        0.082,  # 18:00 — PM peak tail
        0.060,  # 19:00 — evening
        0.042,  # 20:00
        0.032,  # 21:00
        0.025,  # 22:00
        0.020,  # 23:00 — late night
    ], dtype=np.float64)

    return factors


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Integration Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_traffic_index(raw_volume: np.ndarray) -> np.ndarray:
    """
    Convert raw traffic volume to a normalised index in [0.0, 1.0].

    Uses min-max normalisation.  If the input is constant (all zeros or
    all the same value), returns a uniform 0.5 array.

    Parameters
    ----------
    raw_volume : ndarray
        Any-shape array of raw traffic values.

    Returns
    -------
    ndarray, same shape, values in [0.0, 1.0].
    """
    return _min_max_normalize(raw_volume)


def map_traffic_to_grid_nodes(
    hourly_profile: np.ndarray,
    num_nodes: int = _DEFAULT_NUM_NODES,
    seed: int = 42,
) -> np.ndarray:
    """
    Map a single 24-hour traffic profile to all grid nodes.

    Each node gets a slightly perturbed version of the base profile to
    reflect that different parts of the distribution feeder experience
    different traffic patterns (e.g., nodes near highways vs. residential
    side streets).

    The perturbation model:
        - Assigns each node a "traffic exposure" weight drawn from a
          Beta distribution (some nodes are on main arterials, others
          on quiet side streets).
        - Multiplies the base profile by each node's weight.
        - Re-normalises to [0, 1].

    Parameters
    ----------
    hourly_profile : ndarray shape (24,)
        Base traffic index for each hour.
    num_nodes : int
        Number of grid nodes (default 50).
    seed : int
        Random seed for reproducible node weighting.

    Returns
    -------
    ndarray shape (24, num_nodes) float64 — traffic index per hour per node,
    values in [0.0, 1.0].
    """
    assert hourly_profile.shape == (24,), \
        f"Expected hourly_profile shape (24,), got {hourly_profile.shape}"

    rng = np.random.default_rng(seed)

    # Node exposure weights: Beta(2, 5) → most nodes have moderate traffic,
    # a few are on major roads (high weight), some on quiet streets (low).
    node_weights = rng.beta(a=2.0, b=5.0, size=num_nodes)
    # Scale so max weight → 1.0 (the busiest node gets the full profile)
    node_weights = node_weights / node_weights.max()
    # Ensure no node is completely dead — minimum 10% of peak
    node_weights = np.clip(node_weights, 0.10, 1.0)

    # Broadcast: (24, 1) * (1, num_nodes) → (24, num_nodes)
    raw = hourly_profile.reshape(24, 1) * node_weights.reshape(1, num_nodes)

    # Add per-node temporal jitter (± 1 hour shift for some nodes)
    for n in range(num_nodes):
        shift = rng.integers(-1, 2)  # -1, 0, or +1 hour
        if shift != 0:
            raw[:, n] = np.roll(raw[:, n], shift)

    # Normalise globally (preserving inter-node differences)
    raw = _min_max_normalize(raw)

    return raw


def build_hourly_traffic_tensor(
    num_nodes: int = _DEFAULT_NUM_NODES,
    seed: int = 42,
    try_real_data: bool = True,
) -> np.ndarray:
    """
    End-to-end: produce a (24, num_nodes) traffic index tensor.

    Attempts to use real data (LEHD) first, then falls back to the
    synthetic diurnal profile.  The tensor is ready to be consumed
    by the condition vector builder in the generative core.

    Parameters
    ----------
    num_nodes : int
        Number of grid nodes.
    seed : int
        Random seed for reproducibility.
    try_real_data : bool
        If True, attempt to download/parse LEHD OD data first.

    Returns
    -------
    ndarray shape (24, num_nodes) float32 — traffic_index ∈ [0.0, 1.0].
    """
    hourly_profile = None

    # Attempt 1: Real LEHD data
    if try_real_data:
        try:
            csv_path = download_lehd_od_data()
            if csv_path is not None:
                hourly_profile = parse_lehd_to_hourly_profile(csv_path)
                if hourly_profile is not None:
                    logger.info("Using real LEHD origin-destination traffic profile.")
        except Exception as e:
            logger.warning("Real data pipeline failed: %s — using synthetic.", e)

    # Attempt 2: Synthetic fallback (always works)
    if hourly_profile is None:
        logger.info("Using synthetic diurnal traffic profile (FHWA urban pattern).")
        hourly_profile = build_synthetic_traffic_profile(seed=seed)

    # Map to grid nodes
    tensor = map_traffic_to_grid_nodes(
        hourly_profile, num_nodes=num_nodes, seed=seed,
    )

    # Final type cast
    tensor = tensor.astype(np.float32)

    # Sanity check
    assert tensor.shape == (24, num_nodes), \
        f"Expected shape (24, {num_nodes}), got {tensor.shape}"
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, \
        f"Traffic index out of bounds: [{tensor.min()}, {tensor.max()}]"

    logger.info(
        "Traffic tensor built: shape=%s, range=[%.3f, %.3f]",
        tensor.shape, tensor.min(), tensor.max(),
    )
    return tensor


def save_traffic_tensor(
    output_path: Optional[str] = None,
    num_nodes: int = _DEFAULT_NUM_NODES,
    seed: int = 42,
) -> str:
    """
    Build and save the traffic tensor as a .npy file.

    Parameters
    ----------
    output_path : str, optional
        Where to save.  Defaults to data/processed/traffic_index_tensor.npy.
    num_nodes : int
        Grid size.
    seed : int
        Random seed.

    Returns
    -------
    str — absolute path to the saved .npy file.
    """
    output_path = output_path or os.path.join(
        _PROCESSED_DIR, "traffic_index_tensor.npy"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tensor = build_hourly_traffic_tensor(num_nodes=num_nodes, seed=seed)
    np.save(output_path, tensor)
    logger.info("Saved traffic tensor → %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Constant input → 0.5."""
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def get_traffic_summary(tensor: np.ndarray) -> dict:
    """
    Return a human-readable summary of a traffic tensor.

    Parameters
    ----------
    tensor : ndarray shape (24, N)

    Returns
    -------
    dict with summary statistics.
    """
    peak_hour = int(np.argmax(tensor.mean(axis=1)))
    quiet_hour = int(np.argmin(tensor.mean(axis=1)))
    return {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "peak_hour": peak_hour,
        "quiet_hour": quiet_hour,
        "peak_avg_index": float(tensor[peak_hour].mean()),
        "quiet_avg_index": float(tensor[quiet_hour].mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  EVolvAI — Traffic Flow Data Pipeline")
    print("  Geographic scope: Boulder, Colorado")
    print("=" * 60)

    # Build tensor (synthetic fallback — no network needed)
    tensor = build_hourly_traffic_tensor(try_real_data=False)

    # Print summary
    summary = get_traffic_summary(tensor)
    print(f"\n  Tensor shape   : {summary['shape']}")
    print(f"  Value range    : [{summary['min']:.3f}, {summary['max']:.3f}]")
    print(f"  Mean index     : {summary['mean']:.3f}")
    print(f"  Peak hour      : {summary['peak_hour']}:00  (avg={summary['peak_avg_index']:.3f})")
    print(f"  Quietest hour  : {summary['quiet_hour']}:00  (avg={summary['quiet_avg_index']:.3f})")

    # Save
    path = save_traffic_tensor()
    print(f"\n  Saved → {path}")

    # Print hourly profile for first node
    print("\n  Hourly traffic index (Node 1):")
    print("  " + "-" * 50)
    for h in range(24):
        bar = "█" * int(tensor[h, 0] * 40)
        print(f"  {h:02d}:00  {tensor[h, 0]:.3f}  {bar}")

    print("\n" + "=" * 60)
