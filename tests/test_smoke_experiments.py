"""OPT-IN actual smoke experiments. Authoring status: NOT RUN.

Only the user should set IACV_RUN_SMOKE_TESTS=1. Default unit discovery skips all.
All writes are inside a unique tmp/smoke_tests_* folder, never formal results.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.protocol import FORMAL, ROOT


@unittest.skipUnless(os.environ.get("IACV_RUN_SMOKE_TESTS") == "1", "NOT RUN: manual opt-in smoke experiments only")
class SmokeExperimentTests(unittest.TestCase):
    def options(self, output):
        protocol = {**FORMAL, "warmup": 0, "repetitions": 1, "max_keypoints": 128}
        return SimpleNamespace(**protocol, device="cpu", figures=False, formal=False,
                               routed=False, sift_control=False, output=output,
                               methods=["sift", "orb"], bf_strategies=["ratio", "crosscheck"])

    def temporary_output(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="smoke_tests_", dir=ROOT / "tmp")

    def test_bf_shared_graf(self):
        from datasets.graf import iter_graf_pairs
        from experiments.runner import run_experiment
        with self.temporary_output() as directory:
            options = self.options(Path(directory) / "bf")
            pairs = list(iter_graf_pairs(ROOT / "data/graf"))[:1]
            rows = run_experiment(pairs, options)
            self.assertEqual(len(rows), 4)
            for method in options.methods:
                selected = [r for r in rows if r["method"] == method]
                self.assertEqual(selected[0]["feature_sha256_image1"], selected[1]["feature_sha256_image1"])
                self.assertEqual(selected[0]["extraction_id_image2"], selected[1]["extraction_id_image2"])

    def learned(self, method):
        from datasets.graf import iter_graf_pairs
        from experiments.runner import run_experiment
        from learned_features import require_checkpoint
        kinds = ["superpoint"] if method == "superpoint_superglue" else (["superpoint", method] if method.startswith("superpoint") else [method])
        if method == "superpoint_superglue":
            from superglue_adapter import require_official_files
            require_official_files()
        for kind in kinds:
            require_checkpoint(kind)  # Opt-in smoke fails on missing models; never downloads.
        with self.temporary_output() as directory:
            options = self.options(Path(directory) / method)
            options.methods = [method]
            options.bf_strategies = []
            options.sift_control = method == "sift_lightglue"
            rows = run_experiment(list(iter_graf_pairs(ROOT / "data/graf"))[:1], options)
            self.assertEqual(len(rows), 3 if options.sift_control else 1)
            self.assertEqual(rows[0]["matching_strategy"], "superglue" if method == "superpoint_superglue" else "lightglue")

    def test_superpoint_lightglue(self):
        self.learned("superpoint_lightglue")

    def test_superpoint_superglue(self):
        self.learned("superpoint_superglue")

    def test_sift_lightglue(self):
        self.learned("sift_lightglue")

    def test_synthetic(self):
        from datasets.synthetic import DEFAULT_MANIFEST
        from experiments.run_synthetic import run_synthetic
        with self.temporary_output() as directory:
            options = self.options(Path(directory) / "synthetic")
            options.data_root = ROOT / "data/hpatches"
            options.source_manifest = DEFAULT_MANIFEST
            options.limit_sources = 1
            options.settings = ["identity_0", "rotation_30", "perspective_0.06"]
            rows = run_synthetic(options)
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r["transformation_type"] == "identity" for r in rows), 4)

    def test_photometric_with_shared_matchers_and_sift_controls(self):
        from datasets.synthetic import DEFAULT_MANIFEST
        from experiments.run_synthetic import run_synthetic
        from features import MAIN_METHODS
        with self.temporary_output() as directory:
            options = self.options(Path(directory) / "photometric_all")
            options.methods = MAIN_METHODS
            options.sift_control = True
            options.data_root = ROOT / "data/hpatches"
            options.source_manifest = DEFAULT_MANIFEST
            options.limit_sources = 1
            options.settings = ["identity_0", "brightness_30", "contrast_0.8", "gamma_1.25"]
            rows = run_synthetic(options)
            self.assertEqual(sum(r["is_primary"] for r in rows), 4 * 17)
            self.assertEqual(sum(not r["is_primary"] for r in rows), 4 * 2)
            self.assertEqual(sum(r["transformation_type"] == "identity" for r in rows), 19)
            import numpy as np
            import json
            for row in rows:
                np.testing.assert_array_equal(json.loads(row["H_gt"]), np.eye(3))
                self.assertEqual(row["width_image1"], row["width_image2"])
                self.assertEqual(row["height_image1"], row["height_image2"])
            for pair in {r["pair"] for r in rows}:
                sp = [r for r in rows if r["pair"] == pair and r["method"].startswith("superpoint")]
                self.assertEqual(len(sp), 4)
                for key in ("feature_sha256_image1", "feature_sha256_image2", "extraction_id_image1", "extraction_id_image2"):
                    self.assertEqual(len({r[key] for r in sp}), 1)
