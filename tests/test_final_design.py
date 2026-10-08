"""Non-experimental tests: handmade scalar arrays/tables, no images or models."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets.synthetic import DEFAULT_MANIFEST, expected_synthetic_records
from experiments.build_report import curve_data
from experiments.inventory import validate_synthetic_inventory
from experiments.protocol import control_combinations, primary_combinations
from experiments.runner import config_strategies
from experiments.uncertainty import cluster_summary
from features import CONTROL_CONFIG
from photometric import PHOTOMETRIC_VALUES, apply_photometric, synthetic_grid
from transformations import Transformation, transformation_grid


class PhotometricArrayTests(unittest.TestCase):
    def test_brightness_clip_no_mutation_and_shape(self):
        source = np.array([[0, 30, 128, 240, 255]], np.uint8)
        before = source.copy()
        result, info = apply_photometric(source, Transformation("brightness", 30))
        np.testing.assert_array_equal(result, [[30, 60, 158, 255, 255]])
        np.testing.assert_array_equal(source, before)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(info["photometric_clipped_fraction"], .4)
        self.assertEqual(info["photometric_saturated_fraction"], .4)

    def test_fixed_contrast_centre_round_to_even(self):
        source = np.array([[0, 127, 128, 255]], np.uint8)
        result, _ = apply_photometric(source, Transformation("contrast", .6))
        np.testing.assert_array_equal(result, [[51, 127, 128, 204]])
        ties, _ = apply_photometric(np.array([[127, 128]], np.uint8), Transformation("contrast", 2))
        np.testing.assert_array_equal(ties, [[126, 128]])

    def test_gamma_uses_exponent_not_reciprocal(self):
        result, info = apply_photometric(np.array([[0, 64, 128, 255]], np.uint8), Transformation("gamma", 2))
        np.testing.assert_array_equal(result, [[0, 16, 64, 255]])
        self.assertEqual(info["photometric_clipped_fraction"], 0)
        self.assertEqual(info["photometric_saturated_fraction"], .5)
        for setting in (Transformation("gamma", 0), Transformation("contrast", -1), Transformation("brightness", np.nan)):
            with self.assertRaises(ValueError):
                apply_photometric(np.array([[1]], np.uint8), setting)


class FinalInventoryTests(unittest.TestCase):
    def fixture(self, auxiliary=False):
        combinations = control_combinations() if auxiliary else primary_combinations()
        frozen = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["sources"]
        rows = []
        for relative in frozen:
            source = str(Path("fixture") / relative)
            sequence = Path(source).parent.name
            for setting in synthetic_grid():
                for method, strategy in combinations:
                    pair = sequence + "__" + setting.name
                    rows.append({"source_image": source, "sequence": sequence, "method": method,
                                 "matching_strategy": strategy, "dataset": "synthetic", "is_primary": not auxiliary,
                                 "transformation_type": setting.kind, "transformation_strength": setting.parameter,
                                 "transformation_family": "photometric" if setting.kind in PHOTOMETRIC_VALUES else "geometry",
                                 "category": setting.kind, "pair": pair, "synthetic_pair_id": pair,
                                 "identity_baseline_id": sequence + "__identity_0"})
        return pd.DataFrame(rows)

    def test_exact_matrix_and_auxiliary_counts(self):
        self.assertEqual(len(transformation_grid()), 28)
        self.assertEqual(len(synthetic_grid()), 40)
        self.assertEqual(sum(s.kind == "identity" for s in synthetic_grid()), 1)
        self.assertEqual(expected_synthetic_records(), 5440)
        frame, auxiliary = self.fixture(), self.fixture(True)
        validate_synthetic_inventory(frame)
        validate_synthetic_inventory(auxiliary, auxiliary=True)
        self.assertEqual(frame.transformation_family.value_counts().to_dict(), {"geometry": 3808, "photometric": 1632})
        self.assertEqual(len(auxiliary), 640)
        self.assertEqual(580*17 + len(frame), 15300)
        self.assertEqual(580*2 + 2*2 + len(auxiliary), 1804)
        self.assertEqual(config_strategies(CONTROL_CONFIG, SimpleNamespace()), ["ratio", "crosscheck"])

    def test_reject_duplicate_wrong_level_strategy_source_and_baseline(self):
        frame = self.fixture()
        for key, value in (("matching_strategy", "bad"), ("transformation_strength", 999),
                           ("source_image", "fixture/i_replacement/1.ppm"), ("identity_baseline_id", "fake")):
            changed = frame.copy()
            changed.loc[0, key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_synthetic_inventory(changed)
        changed = frame.copy()
        changed.iloc[0] = changed.iloc[1]
        with self.assertRaises(ValueError):
            validate_synthetic_inventory(changed)
        with self.assertRaises(ValueError):
            validate_synthetic_inventory(frame.iloc[:-1])
        with self.assertRaises(ValueError):
            validate_synthetic_inventory(self.fixture(True).iloc[:-1], auxiliary=True)

    def test_curves_reference_one_baseline_without_mutating_raw(self):
        raw = self.fixture()
        original_count = len(raw)
        curves = curve_data(raw)
        self.assertEqual(len(raw), original_count)
        self.assertEqual(len(raw[raw.transformation_type == "identity"]), 8*17)
        for kind, baseline in (("brightness", 0), ("contrast", 1), ("gamma", 1), ("rotation", 0)):
            selected = curves[(curves.curve == kind) & curves.baseline_reference]
            self.assertEqual(len(selected), 8*17)
            self.assertTrue(selected.curve_parameter.eq(baseline).all())
            self.assertTrue(selected.transformation_type.eq("identity").all())


class ClusterUncertaintyTests(unittest.TestCase):
    def test_clusters_not_rows_replicated_and_deterministic(self):
        frame = pd.DataFrame({"method": ["m"]*4, "source": ["a", "a", "b", "b"], "metric": [0., 0., 1., 1.]})
        a = cluster_summary(frame, ["method"], ["metric"], "source", replicates=100, seed=0)
        b = cluster_summary(pd.concat([frame]*5, ignore_index=True), ["method"], ["metric"], "source", replicates=100, seed=0)
        self.assertEqual(a.cluster_count.iloc[0], 2)
        self.assertEqual(a["mean"].iloc[0], .5)
        np.testing.assert_array_equal(a[["ci95_low", "ci95_high"]], b[["ci95_low", "ci95_high"]])
        self.assertEqual(a.leave_one_cluster_out_min.iloc[0], 0)
        self.assertEqual(a.leave_one_cluster_out_max.iloc[0], 1)

    def test_nan_counts_and_single_valid_cluster_no_ci(self):
        frame = pd.DataFrame({"method": ["m"]*3, "sequence": ["a", "a", "b"], "metric": [1., np.nan, np.nan]})
        table = cluster_summary(frame, ["method"], ["metric"], "sequence", replicates=20)
        self.assertEqual(table.valid_count.iloc[0], 1)
        self.assertEqual(table.valid_cluster_count.iloc[0], 1)
        self.assertTrue(np.isnan(table.ci95_low.iloc[0]))
