"""验证全量实验覆盖率，并从真实 CSV 生成简明报告。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from experiments.reporting import save_summary
from features import CONFIGS


def markdown_table(frame):
    # 避免仅为 to_markdown 再安装 tabulate。
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    folders = [ROOT / "results/hpatches_classical", ROOT / "results/hpatches_learned"]
    for folder in folders:
        options = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["options"]
        for field, expected in {"max_keypoints": 2048, "warmup": 2, "repetitions": 5,
                                "ratio": 0.8, "correctness_threshold": 3.0,
                                "ransac_threshold": 3.0, "threads": 4}.items():
            if options[field] != expected:
                raise ValueError(f"Unexpected {field} in {folder}: {options[field]}")
    frame = pd.concat([pd.read_csv(folder / "part1_raw_results.csv") for folder in folders],
                      ignore_index=True)
    if set(frame.method) != set(CONFIGS) or frame.duplicated(["method", "pair"]).any():
        raise ValueError("Missing methods or duplicate pairs")
    for method, group in frame.groupby("method"):
        if group.category.value_counts().to_dict() != {"viewpoint": 295, "illumination": 285}:
            raise ValueError(f"Incomplete categories for {method}")
        if group.sequence.nunique() != 116 or len(group) != 580:
            raise ValueError(f"Incomplete sequences for {method}")
    save_summary(frame.to_dict("records"), ROOT / "results")
    groups = frame.groupby(["method", "category"], sort=False)
    quality = groups[["detector_repeatability", "repeatability", "pmr", "precision",
                      "matching_score", "recall"]].mean().reset_index()
    counts = groups[["features_image1_total", "features_image2_total", "n_features",
                     "raw_putative_matches", "n_putative", "n_correct",
                     "n_correspondences"]].mean().reset_index()
    costs = groups[["detection_ms_image1", "detection_ms_image2", "description_ms_image1",
                    "description_ms_image2", "joint_extraction_ms_image1", "joint_extraction_ms_image2",
                    "matching_ms", "bytes_per_descriptor"]].mean().reset_index()
    costs["extraction_pair_ms"] = (
        costs.detection_ms_image1 + costs.detection_ms_image2
        + costs.description_ms_image1 + costs.description_ms_image2
    ).fillna(costs.joint_extraction_ms_image1 + costs.joint_extraction_ms_image2)
    costs = costs[["method", "category", "extraction_pair_ms", "matching_ms", "bytes_per_descriptor"]]
    costs["pair_stage_total_ms"] = costs.extraction_pair_ms + costs.matching_ms
    homography = groups.agg(
        inliers=("ransac_inliers", "mean"), residual_px=("ransac_reprojection_error_px", "mean"),
        corner_mean_px=("homography_error_px", "mean"), corner_median_px=("homography_error_px", "median"),
        valid_corner_errors=("homography_error_px", "count"),
        estimation_success=("homography_success", "mean"),
    ).reset_index()
    for threshold in (3, 5):
        frame[f"corner_within_{threshold}px"] = frame.homography_error_px.le(threshold)
    success = frame.groupby(["method", "category"], sort=False)[
        ["corner_within_3px", "corner_within_5px"]
    ].mean().reset_index()
    homography = homography.merge(success, on=["method", "category"])
    quality.to_csv(ROOT / "results/part1_matching_table.csv", index=False)
    counts.to_csv(ROOT / "results/part1_counts_table.csv", index=False)
    costs.to_csv(ROOT / "results/part1_runtime_table.csv", index=False)
    homography.to_csv(ROOT / "results/part1_homography_table.csv", index=False)
    sections = [
        "# Part 1 实验结果\n",
        "已完成 8 种配置 × 580 对 = 4640 行 HPatches 主实验。每方法包含 285 对光照变化、295 对视角变化。",
        "全分辨率图像；共同最大预算 2048 点；CPU 4 线程；预热 2 次、重复 5 次；GT 阈值 3 px、ratio 0.8。",
        "分母、硬件、模型和计时边界见 README 与各运行目录 metadata.json。所有表格直接从 CSV 计算。",
        "## 匹配与重复性（每对宏平均；比例 0–1）\n", markdown_table(quality),
        "detector_repeatability 使用描述子计算前的选定点集；repeatability 使用描述子保留点集。",
        "## 平均计数与分母\n", markdown_table(counts),
        "本表是每对计数的平均值；上表是逐对比值的宏平均，不能简单用本表两列相除复现。逐对分子分母见 raw CSV。",
        "## 计算成本\n", markdown_table(costs),
        "extraction_pair_ms 为两图提取时间之和；传统方法合并检测和描述，SuperPoint 使用联合提取时间。",
        "stage total 排除 I/O、CPU/GPU 传输、模型加载、GT 和 RANSAC，不能当成实际端到端帧率。",
        "## Homography\n", markdown_table(homography),
        "均值与中位数同时给出，以显示灾难性错误对均值的影响。阈值成功率将缺失 H 计为失败。",
        "## 观察\n",
    ]
    for category in ("illumination", "viewpoint"):
        subset = quality[quality.category == category]
        best_score = subset.loc[subset.matching_score.idxmax()]
        best_precision = subset.loc[subset.precision.idxmax()]
        fastest = costs[costs.category == category].sort_values("pair_stage_total_ms").iloc[0]
        sections.append(
            f"- {category}：matching score 最高为 {best_score.method} ({best_score.matching_score:.4f})；"
            f"precision 最高为 {best_precision.method} ({best_precision.precision:.4f})；"
            f"所测阶段总耗时最低为 {fastest.method} ({fastest.pair_stage_total_ms:.2f} ms)。"
        )
    sections += [
        "\n这些排名只适用于本次实现、阈值、点数预算和设备；不能把 GPU 与 CPU 差异归因于算法本身。",
        "视角变化同时影响局部形变、可见范围和尺度；描述子具备方向归一化也不能保证任意视角下正确。",
        "BRIEF 固定采样对不做方向或尺度归一化；BRISK/FREAK 的方向处理、SIFT/KAZE 的尺度空间是不同设计。",
        "SuperPoint 与 SuperPoint+LightGlue 的特征配置相同，因此比较它们主要反映 matcher 的差异。",
        "RANSAC 内点多或残差低仍可能来自错误的重复纹理模型；GT 角点误差能揭示这种情况。",
        "实时方案需要同时满足延迟和精度；可先从本机时间表较快的方法试起，并加入真实 I/O、传输、RANSAC 再测端到端延迟。",
        "精度优先不能只看 precision：应结合 matching score、recall、正确匹配数和 H 成功率。",
        "当前数据没有将 scale 和 rotation 独立控制，不能据此得出独立的旋转／尺度不变性曲线。",
        "\n补充实验：results/supplementary/fast_parameters.csv 与 brief_sampling.csv 只覆盖 12 对选定图像。",
        "GRAF：results/graf/；代表图：各主实验目录 figures/。旧 smoke_* 只作运行检查，hpatches_uncapped_pilot 为中止的旧参数试跑。",
        "\nPart 2 拼接和整份作业的最终报告仍需另行完成。本文是 Part 1 结果说明，不冒充整份作业报告。",
    ]
    (ROOT / "PART1_RESULTS.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("Validated 4640 HPatches rows; wrote combined tables and PART1_RESULTS.md")


if __name__ == "__main__":
    main()
