"""Independent manual dataset entry; defaults route into the formal results layout."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

from datasets.graf import iter_graf_pairs
from datasets.hpatches import iter_hpatches_pairs
from experiments.protocol import RESULTS_ROOT, add_protocol_arguments, validate_options
from experiments.runner import run_experiment
from features import CLASSICAL_METHODS, CONFIGS, LEARNED_MATCHERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ExplicitMatchingStrategy(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.matching_strategy_explicit = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_protocol_arguments(parser)
    parser.add_argument("--dataset", choices=["graf", "hpatches"], default="graf")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--methods", nargs="+", choices=list(CONFIGS), default=CLASSICAL_METHODS)
    parser.set_defaults(matching_strategy_explicit=False)
    parser.add_argument("--matching-strategy", choices=["ratio", "crosscheck"], default="ratio",
                        action=ExplicitMatchingStrategy,
                        help="Default BF strategy is ratio; explicit BF flags reject learned matcher methods")
    parser.add_argument("--bf-shared", action="store_true", help="Run ratio and crosscheck on one extraction")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--sift-control", action="store_true")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, help="Isolated output, e.g. tmp/smoke_bf_ratio")
    options = parser.parse_args()
    validate_options(options)
    if len(set(options.methods)) != len(options.methods):
        parser.error("Duplicate methods")
    if options.matching_strategy_explicit and any(CONFIGS[m].matcher in LEARNED_MATCHERS for m in options.methods):
        parser.error("Learned matchers do not accept --matching-strategy ratio/crosscheck; omit the flag")
    if options.bf_shared and (options.matching_strategy_explicit or any(CONFIGS[m].matcher in LEARNED_MATCHERS for m in options.methods)):
        parser.error("--bf-shared requires BF methods and no explicit single strategy")
    if options.limit_pairs is not None and options.limit_pairs < 1:
        parser.error("limit-pairs must be positive")
    options.bf_strategies = ["ratio", "crosscheck"] if options.bf_shared else [options.matching_strategy]
    options.formal = False
    options.routed = options.output is None
    options.data_root = options.data_root or PROJECT_ROOT / "data" / options.dataset
    cv2.setNumThreads(options.threads)
    cv2.setRNGSeed(options.seed)
    np.random.seed(options.seed)
    pairs = list(iter_graf_pairs(options.data_root)) if options.dataset == "graf" else list(
        iter_hpatches_pairs(options.data_root, options.sequences))
    if options.limit_pairs is not None:
        pairs = pairs[:options.limit_pairs]
    run_experiment(pairs, options)


if __name__ == "__main__":
    main()
