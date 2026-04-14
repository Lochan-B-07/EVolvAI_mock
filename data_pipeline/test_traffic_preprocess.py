#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  EVolvAI — Traffic Preprocessing Test Suite                          ║
║  Validates diurnal profiles, normalisation, grid mapping, and        ║
║  end-to-end tensor generation.                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pytest

from data_pipeline.traffic_preprocess import (
    build_synthetic_traffic_profile,
    normalize_traffic_index,
    map_traffic_to_grid_nodes,
    build_hourly_traffic_tensor,
    get_traffic_summary,
    _fhwa_urban_hourly_factors,
    _min_max_normalize,
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
N_NODES = 50
SEED = 42


# ─────────────────────────────────────────────────────────────────────
# §1  FHWA hourly factors
# ─────────────────────────────────────────────────────────────────────
class TestFHWAFactors:
    def test_shape(self):
        factors = _fhwa_urban_hourly_factors()
        assert factors.shape == (24,)

    def test_all_positive(self):
        factors = _fhwa_urban_hourly_factors()
        assert np.all(factors > 0)

    def test_sum_approximately_one(self):
        """FHWA factors should sum to ~1.0 (fractions of daily volume)."""
        factors = _fhwa_urban_hourly_factors()
        assert factors.sum() == pytest.approx(1.0, abs=0.10)

    def test_pm_peak_higher_than_night(self):
        """Hour 17 (5 PM peak) should be much higher than hour 3 (3 AM)."""
        factors = _fhwa_urban_hourly_factors()
        assert factors[17] > factors[3] * 5


# ─────────────────────────────────────────────────────────────────────
# §2  Synthetic traffic profile
# ─────────────────────────────────────────────────────────────────────
class TestSyntheticProfile:
    def test_shape(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        assert profile.shape == (24,)

    def test_range_zero_to_one(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        assert profile.min() >= 0.0
        assert profile.max() <= 1.0

    def test_contains_zero_and_one(self):
        """After min-max normalisation, min should map to 0 and max to 1."""
        profile = build_synthetic_traffic_profile(seed=SEED)
        assert profile.min() == pytest.approx(0.0, abs=1e-10)
        assert profile.max() == pytest.approx(1.0, abs=1e-10)

    def test_deterministic(self):
        """Same seed → identical profile."""
        p1 = build_synthetic_traffic_profile(seed=123)
        p2 = build_synthetic_traffic_profile(seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        """Different seeds → different profiles (noise differs)."""
        p1 = build_synthetic_traffic_profile(seed=1)
        p2 = build_synthetic_traffic_profile(seed=2)
        assert not np.array_equal(p1, p2)

    def test_pm_peak_higher_than_am_trough(self):
        """Evening rush (hour 17) should be higher than 3 AM."""
        profile = build_synthetic_traffic_profile(seed=SEED)
        assert profile[17] > profile[3]


# ─────────────────────────────────────────────────────────────────────
# §3  Normalisation
# ─────────────────────────────────────────────────────────────────────
class TestNormalization:
    def test_basic_range(self):
        raw = np.array([10.0, 50.0, 100.0, 30.0])
        result = normalize_traffic_index(raw)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_input(self):
        """Constant array → all 0.5."""
        raw = np.full(10, 42.0)
        result = normalize_traffic_index(raw)
        np.testing.assert_allclose(result, 0.5)

    def test_zero_input(self):
        """All zeros → all 0.5."""
        raw = np.zeros(5)
        result = normalize_traffic_index(raw)
        np.testing.assert_allclose(result, 0.5)

    def test_negative_values(self):
        """Min-max should handle negative values gracefully."""
        raw = np.array([-10.0, 0.0, 10.0])
        result = normalize_traffic_index(raw)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_preserves_shape(self):
        raw = np.random.rand(24, 50)
        result = _min_max_normalize(raw)
        assert result.shape == (24, 50)


# ─────────────────────────────────────────────────────────────────────
# §4  Grid node mapping
# ─────────────────────────────────────────────────────────────────────
class TestGridMapping:
    def test_output_shape(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        mapped = map_traffic_to_grid_nodes(profile, num_nodes=N_NODES, seed=SEED)
        assert mapped.shape == (24, N_NODES)

    def test_output_range(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        mapped = map_traffic_to_grid_nodes(profile, num_nodes=N_NODES, seed=SEED)
        assert mapped.min() >= 0.0
        assert mapped.max() <= 1.0

    def test_different_nodes_differ(self):
        """Nodes should have different profiles due to exposure weights + jitter."""
        profile = build_synthetic_traffic_profile(seed=SEED)
        mapped = map_traffic_to_grid_nodes(profile, num_nodes=N_NODES, seed=SEED)
        # At least some nodes should differ
        node_means = mapped.mean(axis=0)
        assert node_means.std() > 0.01

    def test_single_node(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        mapped = map_traffic_to_grid_nodes(profile, num_nodes=1, seed=SEED)
        assert mapped.shape == (24, 1)

    def test_33_nodes(self):
        """IEEE 33-bus topology."""
        profile = build_synthetic_traffic_profile(seed=SEED)
        mapped = map_traffic_to_grid_nodes(profile, num_nodes=33, seed=SEED)
        assert mapped.shape == (24, 33)

    def test_deterministic(self):
        profile = build_synthetic_traffic_profile(seed=SEED)
        m1 = map_traffic_to_grid_nodes(profile, num_nodes=N_NODES, seed=SEED)
        m2 = map_traffic_to_grid_nodes(profile, num_nodes=N_NODES, seed=SEED)
        np.testing.assert_array_equal(m1, m2)


# ─────────────────────────────────────────────────────────────────────
# §5  End-to-end tensor
# ─────────────────────────────────────────────────────────────────────
class TestHourlyTensor:
    def test_shape(self):
        tensor = build_hourly_traffic_tensor(
            num_nodes=N_NODES, seed=SEED, try_real_data=False,
        )
        assert tensor.shape == (24, N_NODES)

    def test_dtype(self):
        tensor = build_hourly_traffic_tensor(
            num_nodes=N_NODES, seed=SEED, try_real_data=False,
        )
        assert tensor.dtype == np.float32

    def test_range(self):
        tensor = build_hourly_traffic_tensor(
            num_nodes=N_NODES, seed=SEED, try_real_data=False,
        )
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_peak_exceeds_trough(self):
        """Average traffic at peak hours should exceed off-peak."""
        tensor = build_hourly_traffic_tensor(
            num_nodes=N_NODES, seed=SEED, try_real_data=False,
        )
        hourly_means = tensor.mean(axis=1)
        # PM peak window (16-18) vs night window (2-4)
        pm_peak = hourly_means[16:19].mean()
        night = hourly_means[2:5].mean()
        assert pm_peak > night

    def test_33_bus_variant(self):
        """Works with IEEE 33-bus node count."""
        tensor = build_hourly_traffic_tensor(
            num_nodes=33, seed=SEED, try_real_data=False,
        )
        assert tensor.shape == (24, 33)


# ─────────────────────────────────────────────────────────────────────
# §6  Summary helper
# ─────────────────────────────────────────────────────────────────────
class TestSummary:
    def test_returns_dict(self):
        tensor = build_hourly_traffic_tensor(
            num_nodes=N_NODES, seed=SEED, try_real_data=False,
        )
        summary = get_traffic_summary(tensor)
        assert isinstance(summary, dict)
        assert "peak_hour" in summary
        assert "quiet_hour" in summary
        assert summary["peak_hour"] != summary["quiet_hour"]


# ─────────────────────────────────────────────────────────────────────
# §7  Config integration
# ─────────────────────────────────────────────────────────────────────
class TestConfigIntegration:
    def test_cond_dim_is_six(self):
        from generative_core.config import COND_DIM
        assert COND_DIM == 6

    def test_baseline_condition_length(self):
        from generative_core.config import BASELINE_CONDITION, COND_DIM
        assert len(BASELINE_CONDITION) == COND_DIM

    def test_all_scenarios_correct_length(self):
        from generative_core.config import SCENARIOS, COND_DIM
        for name, scenario in SCENARIOS.items():
            assert len(scenario["condition"]) == COND_DIM, \
                f"Scenario '{name}' has {len(scenario['condition'])} dims, expected {COND_DIM}"

    def test_traffic_index_in_baseline(self):
        """C[5] should be the traffic index in baseline condition."""
        from generative_core.config import BASELINE_CONDITION
        assert 0.0 <= BASELINE_CONDITION[5] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
