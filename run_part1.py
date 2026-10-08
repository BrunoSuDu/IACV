"""Sequential Part 1 reproduction. Running this file starts experiments: manual use only."""
import argparse
from copy import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from datasets.graf import iter_graf_pairs
from datasets.hpatches import iter_hpatches_pairs
from datasets.synthetic import DEFAULT_MANIFEST, select_sources
from experiments.protocol import (RESULTS_ROOT, add_protocol_arguments, check_output, formal_options,
                                  validate_options, validate_pair_inventory)
from experiments.runner import planned_outputs, run_experiment
from features import BF_METHODS, LIGHTGLUE_METHODS, MAIN_METHODS, SUPERGLUE_METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_protocol_arguments(parser)
    parser.set_defaults(figures=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false", help="Explicitly skip qualitative figures")
    parser.add_argument("--only", choices=["all", "experiments", "bf", "ratio", "crosscheck", "lightglue", "superglue",
                                          "synthetic", "supplementary", "report"], default="all")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--hpatches-root", type=Path, default=ROOT / "data/hpatches")
    parser.add_argument("--graf-root", type=Path, default=ROOT / "data/graf")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    validate_options(args)
    formal_options(args)
    if args.only == "report":
        from experiments.build_report import build_report
        if not build_report(args.results_root):
            raise SystemExit(2)
        return
    args.formal = True
    args.routed = True
    args.sift_control = False
    full = args.only in ("all", "experiments")
    bf_stage = full or args.only in ("bf", "ratio", "crosscheck")
    lg_stage = full or args.only == "lightglue"
    sg_stage = full or args.only == "superglue"
    args.methods = (BF_METHODS if bf_stage else []) + (LIGHTGLUE_METHODS if lg_stage else []) + (SUPERGLUE_METHODS if sg_stage else [])
    args.bf_strategies = [args.only] if args.only in ("ratio", "crosscheck") else ["ratio", "crosscheck"]
    args.sift_control = lg_stage
    plans = []
    outputs = []
    # Inspect every input and every target before any experimental writes.
    if args.methods:
        for dataset, root, iterator in (("hpatches", args.hpatches_root, iter_hpatches_pairs),
                                        ("graf", args.graf_root, iter_graf_pairs)):
            local = copy(args)
            local.dataset = dataset
            local.data_root = root
            pairs = list(iterator(root))
            validate_pair_inventory(pairs, dataset, formal=True)
            _, routes = planned_outputs(pairs, local)
            outputs.extend(routes.values())
            plans.append((pairs, local))
    synthetic = None
    if full or args.only == "synthetic":
        synthetic = copy(args)
        synthetic.data_root = args.hpatches_root
        synthetic.output = args.results_root / "synthetic"
        synthetic.methods = MAIN_METHODS
        synthetic.bf_strategies = ["ratio", "crosscheck"]
        synthetic.sift_control = True
        synthetic.limit_sources = None
        synthetic.settings = None
        from experiments.run_synthetic import selected_plan
        selected_plan(synthetic)
        outputs.append(synthetic.output)
    supplementary = None
    if full or args.only == "supplementary":
        supplementary = copy(args)
        supplementary.output = args.results_root / "supplementary"
        list(iter_graf_pairs(args.graf_root))
        list(iter_hpatches_pairs(args.hpatches_root, ["i_ajuntament", "v_adam"]))
        outputs.append(supplementary.output)
    if args.only == "all":
        outputs.append(args.results_root / "comparison")
    for output in set(outputs):
        check_output(output)
    # Fail before any dataset experiment if required local models are absent.
    methods = set(args.methods) | (set(MAIN_METHODS) if synthetic else set())
    if any(method.startswith("superpoint") or method == "sift_lightglue" for method in methods):
        from learned_features import require_checkpoint, verify_pinned_installation
        verify_pinned_installation()
        for kind in ("superpoint", "superpoint_lightglue", "sift_lightglue"):
            required = kind in methods or (kind == "superpoint" and any(m.startswith("superpoint") for m in methods))
            if required:
                require_checkpoint(kind)
    if "superpoint_superglue" in methods:
        from superglue_adapter import require_official_files, require_shared_superpoint
        require_official_files()
        require_shared_superpoint(require_checkpoint("superpoint"))
    import cv2
    import numpy as np
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    for pairs, local in plans:
        run_experiment(pairs, local)
    if synthetic:
        from experiments.run_synthetic import run_synthetic
        run_synthetic(synthetic)
    if supplementary:
        from experiments.run_supplementary import run_supplementary
        run_supplementary(supplementary)
    if args.only == "all":
        from experiments.build_report import build_report
        if not build_report(args.results_root):
            raise RuntimeError("Full run completed stages but report inventory is incomplete")


if __name__ == "__main__":
    main()
