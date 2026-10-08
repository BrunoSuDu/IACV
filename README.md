# IACV Assignment 1 — Part 1

代码按「读图 → 检测 → 描述 → 匹配 → 评估 → 保存结果」组织。
目前工作范围是 Part 1，尚未实现 Part 2 panorama。

## 建议阅读顺序

1. `src/main.py`：选择数据集、方法和实验参数。
2. `src/features.py`：查看每种方法由哪个 detector、descriptor、matcher 组成。
3. `src/classical_pipeline.py`：看一张图如何提取特征，两张图如何匹配。
4. `src/matching.py`：kNN、ratio test、一对一过滤。
5. `src/evaluation.py`：利用 GT 判断正确匹配、计算分母和指标。
6. `src/homography.py`：用候选匹配估计 H，再评价 H。
7. `src/experiments/runner.py`：把上述函数应用到整个数据集。

`datasets/` 只提供文件路径；算法函数不认识 GRAF 或 HPatches。
`learned_features.py` 单独包装 PyTorch 模型，向评估模块提供相同的数据格式。
少量 dataclass 只用于装数据，不使用继承层级或插件框架。

## 安装

已验证环境使用 Python 3.10.11 和 opencv-contrib-python 4.13.0.92。
在本目录打开 PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

传统方法只需 CPU。学习方法可用 CPU，也可用 CUDA；本机 CPU 为 Intel Core i7-12700H，
GPU 为 NVIDIA GeForce RTX 3070 Ti Laptop GPU。
CUDA 12.6 的 PyTorch 单独安装，避免普通 requirements 意外装成不合适的构建：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-learned.txt
```

CPU 环境可将 PyTorch index 改成 `https://download.pytorch.org/whl/cpu`，运行时加 `--device cpu`。
LightGlue 通过固定 Git commit 安装，需要 Git。`--no-deps` 是为保留已有的
`opencv-contrib-python`，避免 LightGlue 再装一份提供相同 `cv2` 模块的 `opencv-python`。
这会让 `pip check` 报 LightGlue 的 opencv-python 包名依赖缺失；运行所需的 cv2 API
由 contrib 包提供。其余依赖由上述两个 requirements 和 PyTorch 安装命令提供。
模型首次运行会从官方 release 下载，缓存在 `models/checkpoints/`。

## 数据

保留原图分辨率。Windows 中文路径使用 `np.fromfile` + `cv2.imdecode` 读取。

```text
data/graf/img1.ppm, img2.ppm, img4.ppm, H1to2p.txt, H1to4p.txt
data/hpatches/i_*/1.ppm ... 6.ppm, H_1_2 ... H_1_6
data/hpatches/v_*/1.ppm ... 6.ppm, H_1_2 ... H_1_6
```

本次下载包含 57 个 illumination 和 59 个 viewpoint 序列，每序列 5 对，共 580 对。
GRAF 用课程提供的文件。HPatches 使用完整图像序列，不能用裁剪的 descriptor patches
替代；参见 [HPatches 官方项目](https://github.com/hpatches/hpatches-dataset)。
缺少任何图像或 H 文件时直接报错，不静默跳过。

## 运行

全部安装和数据准备完成后，可用一个命令依次执行主实验、GRAF、补充实验和最终汇总：

```powershell
.\.venv\Scripts\python.exe run_part1.py
```

也可以按下面的命令分步骤运行，便于理解和检查结果。

快速检查（单次计时只用于检查能否运行）：

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset graf --max-keypoints 0 --warmup 0 --repetitions 1 --output results/smoke_graf
.\.venv\Scripts\python.exe src/main.py --dataset hpatches --sequences i_ajuntament v_adam --warmup 0 --repetitions 1 --output results/smoke_hpatches
```

传统方法全量实验（默认包含 SIFT、ORB、KAZE、FAST+BRIEF、FAST+BRISK、FAST+FREAK）：

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset hpatches --warmup 2 --repetitions 5 --figures --output results/hpatches_classical
```

学习方法全量实验：

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset hpatches --methods superpoint superpoint_lightglue --warmup 2 --repetitions 5 --figures --output results/hpatches_learned
```

单独跑一种方法可用 `--methods sift`。默认计时参数是预热 2 次、重复 10 次；
以上正式命令显式指定 5 次。不同实验使用不同 `--output` 目录，重复指定同一目录会覆盖表格。
`--figures` 每种方法只画每个类别遇到的第一对图，选择规则固定，不按结果挑图。

FAST 参数与简化 BRIEF 采样实验（12 对选定图像，不能称为全量 HPatches）：

```powershell
.\.venv\Scripts\python.exe src/run_supplementary.py
```

几何正确性测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 评估协议

**GT 只用于评估。** descriptor matching 和 RANSAC 均不能先用 GT 筛选候选匹配。

传统方法与 SuperPoint 描述子匹配使用 BF kNN（k=2），保留
`best_distance < 0.8 * second_distance`，然后按距离从小到大贪心保留一对一匹配。
严格小于沿用原代码；第二近邻距离为零时拒绝，少于两个候选也拒绝。
SIFT/KAZE/SuperPoint 用 L2；ORB/BRIEF/BRISK/FREAK 用 Hamming。
LightGlue 用其预训练匹配器和默认 confidence threshold=0.1，不叠加 Lowe ratio test。

匹配评估沿用原项目的 feature 集合：**成功计算描述子的关键点**。
原 detector 点数另存为 `detected_keypoints_image1/2`，描述子保留数为
`features_image1/2_total`，不能混称为检测点数。

| 字段 | 定义 |
| --- | --- |
| `raw_putative_matches` | GT 过滤前的一对一候选匹配总数，也是 RANSAC 输入数 |
| `n_features` | Image 1 的 feature，经 H_gt 投影后落在 Image 2 的 `0 <= x < width, 0 <= y < height` 内 |
| `n_putative` | query 属于上述可见区域的候选匹配数 |
| `n_correct` | 上述匹配中，GT 投影位置与匹配位置的欧氏距离 ≤ 3 px |
| `n_correspondences` | 只用几何建立 ≤ 3 px 的边，再求最大一对一二分图匹配的数量 |
| `pmr` | n_putative / n_features |
| `precision` | n_correct / n_putative；乘 100 即正确匹配百分比 |
| `matching_score` | n_correct / n_features |
| `recall` | n_correct / n_correspondences |
| `repeatability` | n_correspondences / n_features；descriptor-retained feature 集合 |
| `detector_repeatability` | 用计算描述子前的 detector 点重新计算上述 repeatability |

这是项目明确选择的**有方向、基于点距离的协议**，不是声称复现 HPatches 官方全部评估，
也不是基于椭圆区域重叠的 repeatability。`detector_n_features` 与
`detector_n_correspondences` 保存纯检测指标的分母和分子。
为节省内存，几何近邻搜索使用 cKDTree；阈值与原始 cdist 稠密图定义相同，并有对照测试。
分母为零时保存 NaN；摘要提供 count，不把未定义值当成零。

每行检查 `n_correct <= n_putative <= n_features`、
`n_correct <= n_correspondences <= n_features` 和有效比值范围。
PMR 与 Precision 都有定义时，验证 `matching_score = pmr * precision`。

## Homography

`estimate_homography()` 使用全部 raw putative matches，RANSAC 阈值 3 px、
最多 5000 次、confidence=0.995、seed=0。RANSAC 从最小点集提出 H，
根据残差寻找共识集，降低错误匹配对最终 H 的影响；阈值越大通常接受更多点，
也可能接受误匹配。这里的 3 px 是显式选择的初始容差，和 GT 正确性阈值是独立参数。

- `ransac_inlier_ratio` 的分母是 **raw_putative_matches**。
- `ransac_reprojection_error_px` 是内点在 **H_est** 下的平均前向残差。
- `mean_correct_gt_error_px` 是正确匹配在 **H_gt** 下的平均距离。
- `homography_error_px` 是 Image 1 四角分别经 H_est、H_gt 投影后位置差的均值。
- 少于 4 个匹配或求解失败时 `homography_success=False`，误差留 NaN。

角点误差也衡量非重叠区域的外推，不能与内点残差混为一谈。
RANSAC 返回矩阵不保证准确，必须同时看角点误差。报告失败数和有效误差样本数，
不能只看丢弃失败行后的平均误差。

## 计时、内存和公平性

主实验每种方法最多 2048 点。SIFT/KAZE/FAST 按 response 降序选前 2048 个；
ORB 的 nfeatures 改为 2048，之后同样执行上限筛选；SuperPoint 使用自身 score top-k。
提取描述子时仍可能删掉边缘点，所以最终数量不一定等于 2048。
检测计时包含筛选。`detected_before_limit_image*` 记录外部筛选前点数；
ORB 内部已经限额，此列不是其内部全部候选角点数。SuperPoint 未暴露未截断数量，留 NaN。
检测 repeatability 衡量的是实际选择的点集。这是共同最大预算，并不保证每种方法点数相同。
此上限是实验选择，不是老师规定的常数，也不消除所有算法／硬件差异。
使用 `--max-keypoints 0` 可恢复传统方法默认行为（其中 ORB 自身默认 500 点）。
已有 `results/smoke_graf` 是此前默认点数的单次运行；不用于正式时间比较。

CPU 算法固定 4 个 OpenCV 线程。每项预热后重复测量，输出平均值与总体标准差。

除点数上限外保留 OpenCV 默认参数，重要数值为：SIFT contrastThreshold=0.04、
edgeThreshold=10、sigma=1.6、nOctaveLayers=3；FAST threshold=10、
nonmaxSuppression=True、TYPE_9_16；KAZE threshold=0.001、nOctaves=4、
nOctaveLayers=4、extended=False、upright=False；ORB scaleFactor=1.2、nlevels=8、
WTA_K=2、patchSize=31、fastThreshold=20。OpenCV BRIEF 为 32 bytes、
use_orientation=False；BRISK patternScale=1；FREAK 默认进行方向和尺度归一化。
学习模型的完整默认参数保存在各运行目录 `models/*.json` 中。
同一序列的参考图只提取一次，它的计时会在 5 行 pair 结果中重复出现；
这不是 5 次独立图像测量。摘要中的 std 是不同 pair 之间的样本标准差。
文件读写、建对象、模型加载、画图、GT evaluation 和 RANSAC 不计入 feature/matching 时间。

SuperPoint 同时产生关键点和描述子，记录 `joint_extraction_ms_image1/2`，
独立 detection/description 列为 NaN。GPU 计时前后调用 cuda.synchronize。
输入张量准备和 CPU/GPU 传输不计入模型时间，因此这些数字不是端到端应用延迟。
SuperPoint+BF 的 BF 在 CPU 上，SuperPoint+LightGlue 的 matcher 在选定设备上。
不要把 CPU 与 GPU 时间解释成纯算法的硬件无关排名。

描述子维度按数组每行元素数报告：例如 SIFT 是 128 个 float32，ORB 是 32 个 uint8
（256 bits）。`bytes_per_descriptor` 和两个 `descriptor_memory_bytes_image*`
仅计描述子数组载荷，不含关键点、Python 对象、模型权重、网络激活或 GPU 显存峰值。

## 结果文件

每个实验目录包含：

- `part1_raw_results.csv`：每种方法、每对图像一行。
- `part1_summary.csv`：按 dataset / method / category 聚合的 mean、std、count。
- `part1_computational_cost.csv`：运行时间、描述子存储等字段。
- `metadata.json`：运行参数与环境。
- `progress.csv`：执行中逐行刷新；中断后可检查已完成部分，不代表完整 benchmark。
- `figures/`：关键点、候选／正确／错误匹配、RANSAC 内点。
- `models/*.json`：学习模型配置、版本、checkpoint SHA-256。

`putative.png` 画的是 co-visible 区域内参与评估的候选匹配；
`ransac_inliers.png` 则来自全部 raw 候选的 RANSAC 估计。每幅最多画 100 条线，
标题中的 total 是该集合的完整数量，不是画出的线数。

摘要是每对图像指标的宏平均，不是把所有匹配计数合并后计算的微平均。
illumination/viewpoint 分开统计，二者分别应有 285/295 行每方法。
真实结果的解释和完成状态见 `PART1_RESULTS.md`（实验结束后生成）。

## 来源与范围

作业依据为本目录 `Assignment_1_Python_Keypoints_Descriptors_Stitching.pdf`：
§2.7–2.8 为定量指标，§4 为 minimum recommended configurations，§7 为可复现性要求。
SIFT 是 principal classical baseline；SURF optional。
§2.3 要求研究 FAST threshold/non-max suppression。
§2.5 鼓励自行实现简化 BRIEF，并写明至少研究两种采样策略；补充实验采用均匀与 Gaussian。
SIFT+LightGlue 是鼓励扩展；本次主实验选择 SuperPoint+LightGlue。

- [OpenCV matching 文档](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html)
- [LightGlue 官方实现](https://github.com/cvg/LightGlue)，固定 commit 见 requirements-learned.txt。
- [BRIEF 原论文](https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/CalonderLOTSF12.pdf)，§3.1–3.2。

简化 BRIEF 使用固定 seed=0、31×31 patch、256 bits、7×7 Gaussian 平滑，
采样取整并截断到 patch 范围，不做方向／尺度归一化。它是教学对照，不替代主表的 OpenCV BRIEF。
FAST 参数和 BRIEF 对照只覆盖选定 12 对，不据此声称全数据集结论。
提交时排除 `.venv/`、完整数据集、模型权重与临时文件，附获取说明即可。
