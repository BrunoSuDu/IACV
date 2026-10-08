"""一次运行 Part 1 的主实验、补充实验和汇总。"""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
LEARNED = ["superpoint", "superpoint_lightglue"]


def run(arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-classical", action="store_true",
                        help="Reuse an already completed 2048-point classical benchmark")
    args = parser.parse_args()
    common = ["--max-keypoints", "2048", "--warmup", "2", "--repetitions", "5", "--figures"]
    if not args.skip_classical:
        run(["src/main.py", "--dataset", "hpatches", *common,
             "--output", "results/hpatches_classical"])
    else:
        import pandas as pd
        table = pd.read_csv(ROOT / "results/hpatches_classical/part1_raw_results.csv")
        expected = {"sift", "orb", "kaze", "fast_brief", "fast_brisk", "fast_freak"}
        if set(table.method) != expected or not (table.groupby("method").size() == 580).all():
            raise ValueError("Cannot skip: classical results are incomplete")
    run(["src/main.py", "--dataset", "graf", "--methods", "sift", "orb", "kaze",
         "fast_brief", "fast_brisk", "fast_freak", *LEARNED, *common,
         "--output", "results/graf"])
    run(["src/main.py", "--dataset", "hpatches", "--methods", *LEARNED, *common,
         "--output", "results/hpatches_learned"])
    run(["src/run_supplementary.py"])
    run(["src/experiments/build_report.py"])


if __name__ == "__main__":
    main()
