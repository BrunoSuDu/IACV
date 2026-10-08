"""CLI defaults, output protection and formal benchmark identities."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from features import BF_METHODS, CLASSICAL_METHODS, LIGHTGLUE_METHODS, SUPERGLUE_METHODS


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "part1-17x40-v1"
RESULTS_ROOT = ROOT / "results" / PROTOCOL_VERSION
FORMAL = {"max_keypoints": 2048, "warmup": 2, "repetitions": 5, "seed": 0,
          "threads": 4, "ratio": 0.8, "correctness_threshold": 3.0,
          "ransac_threshold": 3.0, "ransac_max_iters": 5000, "ransac_confidence": 0.995}


def add_protocol_arguments(parser):
    for name in ("max_keypoints", "warmup", "repetitions", "seed", "threads", "ransac_max_iters"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=FORMAL[name])
    for name in ("ratio", "correctness_threshold", "ransac_threshold", "ransac_confidence"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=FORMAL[name])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--figures", action="store_true")


def validate_options(options):
    if options.warmup < 0 or options.repetitions < 1 or options.threads < 1 or options.max_keypoints < 1:
        raise ValueError("Require warmup>=0, repetitions>=1, threads>=1, max-keypoints>=1")
    if (not 0 < options.ratio < 1
            or any(not math.isfinite(v) or v <= 0 for v in (options.correctness_threshold, options.ransac_threshold))):
        raise ValueError("Require ratio in (0,1) and positive geometric thresholds")
    if options.ransac_max_iters < 1 or not 0 < options.ransac_confidence < 1:
        raise ValueError("Invalid RANSAC settings")


def formal_options(options):
    for key, expected in FORMAL.items():
        if getattr(options, key) != expected:
            raise ValueError(f"Formal experiment requires {key}={expected}")


def check_output(path):
    path = Path(path).resolve()
    protected = [ROOT / name for name in ("src", "tests", "configs", "data", "models", ".venv", ".git")]
    if path == ROOT or any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError(f"Output overlaps project source/data/models: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing nonempty output: {path}. Choose a fresh directory; no overwrite/resume.")
    return path


def reserve_outputs(paths):
    paths = sorted({Path(path).resolve() for path in paths})
    for path in paths:
        check_output(path)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def result_folder(root, dataset, method, strategy):
    root = Path(root)
    stage = strategy
    if dataset == "graf":
        suffix = "graf_control" if method == "sift_compatible_bf" else "graf"
    elif dataset == "hpatches":
        if method == "sift_compatible_bf":
            suffix = "hpatches_sift_control"
        elif method == "sift_lightglue":
            suffix = "hpatches_sift"
        elif method.startswith("superpoint"):
            suffix = "hpatches_superpoint"
        else:
            suffix = "hpatches_classical"
    else:
        raise ValueError(f"Unsupported routed dataset: {dataset}")
    # Compatible BF is an auxiliary LightGlue control, independent of ratio totals.
    if method == "sift_compatible_bf":
        stage = "lightglue"
    return root / stage / suffix


def validate_pair_inventory(pairs, dataset, formal=True):
    if not pairs or any(p.dataset != dataset for p in pairs):
        raise ValueError("Missing pairs or inconsistent dataset")
    if len({p.name for p in pairs}) != len(pairs):
        raise ValueError("Duplicate image pairs")
    if dataset == "hpatches" and formal:
        categories = {c: sum(p.category == c for p in pairs) for c in ("illumination", "viewpoint")}
        sequences = {p.sequence for p in pairs}
        if len(pairs) != 580 or len(sequences) != 116 or categories != {"illumination": 285, "viewpoint": 295}:
            raise ValueError(f"Incomplete HPatches: {len(pairs)} pairs, {len(sequences)} sequences, {categories}")
        for sequence in sequences:
            if {p.target_index for p in pairs if p.sequence == sequence} != {2, 3, 4, 5, 6}:
                raise ValueError(f"Missing target index in {sequence}")
    if dataset == "graf" and formal and {p.target_index for p in pairs} != {2, 4}:
        raise ValueError("GRAF requires the two supplied pairs")


def code_identity():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "run_part1.py"]
    paths += sorted((ROOT / "configs").glob("*.json"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_manifest(paths):
    records = []
    for path in sorted({Path(p).resolve() for p in paths}):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    return records


def primary_combinations():
    return [(m, f"bf_{s}") for m in BF_METHODS for s in ("ratio", "crosscheck")] + [
        (m, "lightglue") for m in LIGHTGLUE_METHODS] + [(m, "superglue") for m in SUPERGLUE_METHODS]


def expected_hpatches_records(pair_count=580):
    return {"ratio": len(BF_METHODS) * pair_count, "crosscheck": len(BF_METHODS) * pair_count,
            "lightglue": len(LIGHTGLUE_METHODS) * pair_count, "superglue": len(SUPERGLUE_METHODS) * pair_count}


def control_combinations():
    return [("sift_compatible_bf", "bf_ratio"), ("sift_compatible_bf", "bf_crosscheck")]
