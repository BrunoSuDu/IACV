# IACV Assignment 1 — Part 1 最终实验框架

当前状态：**code prepared；tests NOT RUN；smoke experiments NOT RUN；benchmark NOT RUN**。
本轮只进行了源码改造、课程 PDF/已安装官方源码阅读、AST 语法与 JSON 检查。
语法检查不代表功能、模型兼容性或性能已经通过运行验证。
原 results 的旧 CSV/figures/metadata 和旧生成报告已按用户授权清理。
Part 2 panorama、blending、RANSAC threshold sensitivity 不在本轮范围。

## 项目结构

```text
run_part1.py                   顺序运行正式阶段；默认全部并汇总
configs/synthetic_sources.json 固定八张原图的 manifest
src/main.py                    独立 HPatches/GRAF 入口，也供 smoke 使用
src/features.py                十种主配置、独立 BF control、ImageFeatures
src/detectors.py, descriptors.py OpenCV 标准创建函数
src/classical_pipeline.py      检测/描述计时；同一提取结果可交给两种 BF 策略
src/matching.py                ratio / crosscheck，一对一匹配
src/learned_features.py        共享 SuperPoint / SIFT-compatible 与 learned matchers
src/superglue_adapter.py       固定官方 SuperGlue 源码/权重身份、输入与输出适配
src/transformations.py         28 个唯一设置、精确 H、warp、valid mask
src/feature_masks.py           同步过滤 keypoints/descriptors/native fields，特征哈希
src/evaluation.py              GT correctness、分母、最大一对一 GT 对应、可选 masks
src/homography.py              raw matches 的 RANSAC 与四角 GT 误差
src/timing.py                  CPU/GPU 同步计时
src/datasets/                  图像路径与 frozen synthetic sources
src/experiments/protocol.py    统一参数、输出保护、输入/代码身份、完整性规则
src/experiments/runner.py      共享提取、独立匹配、逐行进度、完成标记
src/experiments/run_synthetic.py 独立合成实验
src/experiments/run_supplementary.py FAST 与教学 BRIEF
src/experiments/build_report.py 只读真实完成的 CSV 并生成新表格/曲线
src/run_supplementary.py       保留原 supplementary 入口
src/visualization.py           keypoints、matches、synthetic context
src/image_io.py                中文路径读图
src/brief.py                   固定 pattern 的教学 BRIEF
```

## 十种主配置

Detector 找关键点；descriptor 描述局部外观；matcher 建立两图对应。

| Method | Detector | Descriptor | Matcher / distance |
|---|---|---|---|
| sift | OpenCV SIFT | 原始 SIFT | BF / L2 |
| orb | ORB | ORB | BF / Hamming |
| kaze | KAZE | KAZE | BF / L2 |
| fast_brief | FAST | OpenCV BRIEF | BF / Hamming |
| fast_brisk | FAST | BRISK | BF / Hamming |
| fast_freak | FAST | FREAK | BF / Hamming |
| superpoint | 官方 SuperPoint | SuperPoint 256维 | BF / L2 |
| superpoint_lightglue | 相同 SuperPoint | 相同 SuperPoint | 官方 LightGlue |
| sift_lightglue | 官方 SIFT-compatible OpenCV backend | RootSIFT 128维 | 官方 SIFT LightGlue |
| superpoint_superglue | 相同 SuperPoint | 相同 SuperPoint | 官方 pretrained SuperGlue outdoor |

前七种 BF methods 支持两种策略：

- `bf_ratio`：`knnMatch(k=2)`，严格 `best < 0.8*second`，按距离贪心一对一；沿用原定义。
- `bf_crosscheck`：`BFMatcher(crossCheck=True).match()`，双向最近邻；不调用 kNN/ratio。
- `lightglue`：官方 pretrained matcher，使用默认模型参数；不叠加 BF 过滤。
- `superglue`：固定官方 outdoor pretrained matcher；100 Sinkhorn iterations、match threshold=.2，不叠加 BF 过滤。

CLI 的 `--matching-strategy ratio|crosscheck` 默认 ratio。LightGlue/SuperGlue 方法自动选择注册的 matcher；
**显式给 LightGlue/SuperGlue 传 BF strategy 会报错**。`--bf-shared` 仅用于 BF 方法。
每行保存 method/detector/descriptor/matcher/matching_strategy/distance_metric/ratio_threshold。
非 ratio 的 ratio_threshold 留 NaN；learned matcher 的 distance_metric 标记为 learned_confidence。

## 共享提取与 SIFT+LightGlue

完整运行或 `--only bf`：每个 classical source、每张 reference 只提取一次，target 每 pair 一次，
同一 `ImageFeatures` 对象传给 ratio/crosscheck，分别测量 matching time。KAZE 不会为两种 BF 策略重提取。
计时预热/重复会多次调用提取函数，但只产生一组供 matcher 使用的最终结果和一组提取时间测量。
同序列 reference 的测量在五条 pair 记录中复用；它们不是五次独立测量。

完整运行中 SuperPoint 的 BF ratio/crosscheck、LightGlue 和 SuperGlue 共享一组 native features。
记录 `feature_sha256_image1/2`、`extraction_id_image1/2`、run_id、缓存和共享字段。
分别执行 `--only ratio` / `--only crosscheck` 会独立提取；report 必须验证特征哈希相同，
并标记是否确为同一次提取测量。正式策略对照优先 `--only bf`。

本地已静态核对官方 commit **eb42fee2d71449efb0aa5c10549752b5d75384d8**：
`sift.py`、`utils.py`、`lightglue.py`、`superpoint.py` 和安装 direct_url.json。
SIFT-compatible 使用官方 `SIFT` 的 OpenCV backend、过滤、top-k 和 RootSIFT normalization：
L1 normalization → clamp eps=1e-6 → sqrt → L2 normalization。
EmptySafeSIFT 仅补齐官方 OpenCV 路径的空结果形状，继承官方 forward 的 RootSIFT 处理。

LightGlue 输入是同一设备上的：keypoints `[1,N,2]`、descriptors `[1,N,128]`、
scales `[1,N]`（OpenCV kp.size）、oris `[1,N]`（radians）、image_size `[1,2]`（width,height）。
`LightGlue(features="sift")` 自动启用 128维投影和 scale/orientation encoding。
输出 matches/scores 的长度、索引范围和一对一性质均检查。

**原 SIFT baseline 保留不变**：contrastThreshold=.04、nOctaveLayers=3、原始描述子。
官方兼容 extractor 是 detection_threshold=.0066667、num_octaves=4、edge_threshold=10、rootsift=True、nms_radius=0。
因此原 SIFT BF 与 SIFT+LightGlue **不是严格 matcher-only comparison**。
默认另写 `sift_compatible_bf` control，使用和 SIFT+LightGlue 同一次提取的完全相同 RootSIFT features，
只将 matcher 换成 BF ratio。该 control 不属于十种主配置，也不计入 9860/2240。

Compatible SIFT 在 CPU 提取，LightGlue 在所选 CPU/GPU；该 extractor 联合执行 detectAndCompute 与处理，
记录 joint_extraction_ms，独立 detection/description 留 NaN。SuperPoint 同样记录联合时间。

## 环境、数据和本地模型

保留现有环境和依赖文件；没有安装、升级或下载。
原环境快照：Python 3.10.11，OpenCV contrib 4.13.0.92；NumPy 2.2.6、SciPy 1.15.3、
pandas 2.3.3、torch 2.14.1+cu126、torchvision 0.29.1+cu126（版本依据现有 lock，非本轮运行验证）。
继续使用 `.venv`；contrib 提供 BRIEF/FREAK。不要同时装覆盖 cv2 的 opencv-python。
`requirements-learned.txt` 已固定 kornia 和官方 LightGlue commit，未新增额外依赖。

若需要在另一机器重建环境，用户自行执行，CUDA/CPU wheel 源应与目标机器一致：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-learned.txt
```

完整历史依赖快照见 requirements-lock.txt；requirements.txt 的既有非模型包约束没有升级。
LightGlue 上游声明 opencv-python 包名，而此项目由 contrib 提供 cv2；不要以重复安装 cv2 解决包名检查。
安装 wheel 的可获取性、CPU wheel 对应版本与当前硬件兼容性需用户环境中确认。

```text
data/graf/img1.ppm img2.ppm img4.ppm H1to2p.txt H1to4p.txt
data/hpatches/i_*/1.ppm ... 6.ppm H_1_2 ... H_1_6
data/hpatches/v_*/1.ppm ... 6.ppm H_1_2 ... H_1_6
```

HPatches 使用 full-resolution image sequences，不是裁剪的 descriptor patches。
正式库存必须是 57 illumination、59 viewpoint，116 sequences、580 pairs；缺图/H 会报错。
GRAF 使用课程两对原图与 GT，作为 supplementary qualitative/geometric example。

LightGlue/SuperPoint 模型仅从本地 `models/checkpoints/` 加载，缺少文件即报错，**runner 不隐式下载**。
正式运行记录官方 release URL、commit、模型参数、device、GPU 和 checkpoint SHA-256，
并检查 checkpoint 覆盖模型的所有 trained parameters（上游 strict=False 不足以保证这一点）。
本轮观察到 SuperPoint 与其 LightGlue checkpoint 已存在；**SIFT LightGlue checkpoint 尚缺失**。
需要你手动下载，以下命令只作为说明，没有被执行；不会覆盖现有文件：

```powershell
New-Item -ItemType Directory -Force models/checkpoints | Out-Null
$release = 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv'
$models = @(
  @('superpoint_v1.pth', 'superpoint_v1.pth'),
  @('superpoint_lightglue.pth', 'superpoint_lightglue_v0-1_arxiv.pth'),
  @('sift_lightglue.pth', 'sift_lightglue_v0-1_arxiv.pth')
)
foreach ($model in $models) {
  $destination = Join-Path 'models/checkpoints' $model[1]
  if (-not (Test-Path -LiteralPath $destination)) {
    Invoke-WebRequest -Uri "$release/$($model[0])" -OutFile $destination
  }
}
```

CUDA 不可用时显式使用 `--device cpu`，不静默切换设备。
设置 seed=0、OpenCV 4线程、cuDNN deterministic、torch deterministic algorithms、
CUBLAS_WORKSPACE_CONFIG=:4096:8；不支持确定性 kernel 时直接报错，由用户检查设备/运行库。
这些设置不保证不同硬件、OpenCV/PyTorch 版本的浮点结果逐位一致。

## SuperGlue 的固定来源与共享特征协议

采用 [Magic Leap 官方实现](https://github.com/magicleap/SuperGluePretrainedNetwork)，固定 commit
`ddcf11f42e7e0732a0c4607648f9448ea8d73590`，官方 `superglue_outdoor.pth`，不训练、不搜索权重。
[官方 README](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/README.md)
说明 indoor 来自 ScanNet、outdoor 来自 MegaDepth。本项目预先选 outdoor 作为自然场景/平面图像对的起点；
**没有用 HPatches 调参，也没有证据声称它最优**。保留官方 GNN、100次 Sinkhorn 和 .2匹配阈值。

官方文件由用户放在 `models/superglue_official/`，不复制进 `src/`，不安装第三方同名 pip package。
请查看该版本的[官方许可](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/LICENSE)，
其中限定 academic/nonprofit noncommercial research 用途。手动准备命令如下；本轮未执行：

```powershell
$sgCommit = 'ddcf11f42e7e0732a0c4607648f9448ea8d73590'
$sgBase = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/$sgCommit"
$sgFiles = @('models/superglue.py', 'models/weights/superglue_outdoor.pth', 'LICENSE')
foreach ($sgFile in $sgFiles) {
  $sgDestination = Join-Path 'models/superglue_official' $sgFile
  if (-not (Test-Path -LiteralPath $sgDestination)) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $sgDestination) | Out-Null
    Invoke-WebRequest -Uri "$sgBase/$sgFile" -OutFile $sgDestination
  }
}
```

Adapter 在导入官方源码/加载模型之前检查以下 Git blob SHA-1（包含 Git `blob <bytes>\0` header，
不是普通文件 SHA-1）；同时记录实际文件 SHA-256。错误或不完整下载会明确失败，不自动替换。

| 固定官方文件 | Git blob SHA-1 |
|---|---|
| models/superglue.py | 5a89b0348075bcb918eab123bc988c7102137a3d |
| models/weights/superglue_outdoor.pth | 5ca3392bd5349790d7bc1c246961fc041b88c1e8 |
| LICENSE | afc1ed1db5d14d18e546de546762dd45fc78ef1d |

已只读计算本地 SuperPoint checkpoint 的 SHA-256：
`52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`，
其 Git blob 与官方 SuperGlue 的 `superpoint_v1.pth` 相同。无需第二份 SuperPoint checkpoint。
完整运行只使用现有 pinned LightGlue SuperPoint extractor：相同灰度图、full resolution、cap2048、
NMS4、threshold=.0005、border4，三个 matcher 使用同一份 keypoints/descriptors/**真实 scores**。
这不同于官方 SuperGlue outdoor demo 推荐的 resize1600/NMS3；也不同于旧 SuperPoint 类默认threshold=.005。
本项目是控制提取器后的 matcher 对照，不声称复现官方 demo 的完整配置。

[固定官方 forward](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/superglue.py)
输入：keypoints `[1,N,2]`（像素 x,y），descriptors `[1,256,N]`，scores `[1,N]`。
Adapter 只将共享 `[1,N,256]` 描述子转置并 contiguous；不重提取、不归一化 scores、不预归一化坐标。
模型内部用 `(xy-[W,H]/2)/(0.7*max(W,H))` 归一化。该固定 forward 只读取 image.shape；
用单个零标量 expand 出 `[1,1,H,W]` shape carrier，无需复制真实像素或额外分配整张图。
布局适配、验证与数据搬运在 matcher 计时外，模型 forward 在计时内。

输出 matches0/1、matching_scores0/1 均检查长度、整数索引、范围、有限 scores、双向一致性；
-1 unmatched 丢弃；`DMatch.distance=1-confidence` 仅供统一展示，不是描述子距离。
空特征直接记录零 matching time，模型初始化不计时。模型加载/推理兼容性尚未验证。

报告提供 BF ratio–LightGlue、BF ratio–SuperGlue、LightGlue–SuperGlue 的 matching quality、
homography（success/failure、error mean/median、3/5px accuracy）与 matching runtime 配对比较，
分别输出 `learned_comparisons.csv`、`graf_learned_comparisons.csv`、`synthetic_learned_comparisons.csv`。
校验 fingerprint 覆盖 keypoints/descriptors/scores/image sizes；extraction IDs 判断是否同一次提取测量。
BF ratio–crosscheck 对照另有独立配对表。Runtime 表分列 extraction/matching；重复引用提取时间不能当独立样本。

## Synthetic 数据、矩阵与 masks

固定 manifest 在 `configs/synthetic_sources.json`：
`i_ajuntament,i_autannes,i_bologna,i_books,v_abstract,v_adam,v_apprentices,v_artisans` 的 `1.ppm`。
规则是当前已安装快照中各类别按名称排序的前四个序列，已冻结路径，不按结果选图、不随机替换。
**尚未视觉确认场景多样性**，请你在正式运行前查看八张图；如果决定更换，应先冻结新 manifest，
保留四张 illumination/四张 viewpoint，并按同一 manifest 从零运行。
缺任何一张图直接失败，不补选。正式运行保存图像尺寸、原图 SHA-256 和 source-selection 说明。

| 操作 | 参数 | H 的定义 |
|---|---|---|
| rotation | -90,-60,-30,0,30,60,90 degrees | 图像中心旋转，OpenCV 正角为画面逆时针 |
| scale | .5,.75,1,1.25,1.5,2 | 图像中心的均匀缩放 |
| translation_x | -.25,-.1,0,.1,.25 | dx=parameter*width |
| translation_y | -.25,-.1,.1,.25 | dy=parameter*height；单独变化，不做 Cartesian combinations |
| horizontal shear | -.25,-.125,0,.125,.25 | x'=x+s*(y-cy), y'=y |
| perspective | -.12,-.06,0,.06,.12 | 明确的四角规则，见下文 |

中心 `(cx,cy)=((w-1)/2,(h-1)/2)`。Rotation、scale、translation、shear 都直接构造 float64 3x3 H。
Perspective strength s 无单位；令 d=s*(w-1)，按 TL,TR,BR,BL 顺序将角点 x 偏移 `[+d,-d,+d,-d]`，
y 不变。正 s 使上边变窄、下边变宽。用 float64 八方程解 H，验证有限性、非奇异、
角点分母同号且为正、polygon convex 和非零面积。不用 RANSAC 构造 GT。
所有 H 的方向是 **original → transformed**；warp 使用和保存的 H 同一矩阵。

各曲线的 identity 合并为一个 `identity_0`：1+6+5+4+4+4+4=28 unique settings/source。
八张原图共224 pairs，十配置共2240 rows。曲线可引用同一 identity 行，绝不增加 raw measurement。

固定 canvas 为原图 width/height，原点(0,0)，不扩展、不追加 canvas 平移；内容可能裁切。
图像用 INTER_LINEAR 和 BORDER_REFLECT_101，减少黑色填充的突变。
相同 H/canvas 将全1 float mask 用 INTER_LINEAR、constant-zero border warp：
coverage>=1-1e-6 才是有效内容，额外1px erosion 处理插值边界；反射填充像素始终无效。
GT 与图像坐标不因 mask 变化。

所有方法在提取并执行 cap 后，用同一规则后过滤：关键点到无效像素/画布外的距离
必须 `>max(48px,3*kp.size)`。同步过滤 detector points、descriptor keypoints、descriptors、
scales、oris、keypoint_scores、native_indices；保留 image_size。不会为过滤掉的点补齐预算。
该规则也应用于 original image 的外边缘。SuperPoint 没有显式尺度，kp.size=1。

GT 只在 evaluation 判断双向 co-visibility：query 在原图安全区域且其 GT 投影在 target 安全区域；
target 的逆投影也必须落在原图安全区域。默认安全区域为 valid mask 再 erosion 48px。
描述子按各自 keypoint-size support 进一步过滤。无 masks 的 HPatches 保持原有分母定义。
matching 和 RANSAC 不接受 H_gt，也不使用 GT 筛选后的 matches。

`overlap_ratio` 是原图像素中，经实际变换仍有几何可见内容的比例；
`target_valid_ratio` 是插值边界处理后 target 的有效像素比例；
`support_overlap_ratio` 进一步考虑双方48px安全区域，以整张 source 面积为分母。
保存这三者、完整 H、canvas/interpolation/border 定义及 retained counts。
不要将 overlap/cropping 导致的匹配数下降直接归因于 transformation invariance。

**限制**：后过滤不能完全消除 padding 对 Gaussian/nonlinear scale spaces、CNN receptive fields、
orientation 和 top-k 选择的影响；3*size 是统一的有限支持近似，不是各 descriptor 的严格支持证明。
原图边缘也会损失特征，小尺度时有效区域可明显减少；report 必须结合 overlap 和计数解释。

## 正式协议与指标

正式参数统一为2048 keypoints、warmup=2、repetitions=5、seed=0、OpenCV threads=4，
ratio=.8、GT correctness<=3px、RANSAC threshold=3px/maxIters=5000/confidence=.995。
2048 是共同预算，不是优化所得最优值；正式 runner 拒绝其他参数。
Smoke 入口允许少量方法/图像/单次计时，不作为正式统计来源。

除 cap 外保留 OpenCV defaults：SIFT contrast=.04, edge=10, sigma=1.6, octave layers=3；
ORB nfeatures=2048, scaleFactor=1.2, nlevels=8, WTA_K=2, patchSize=31, edgeThreshold=31,
fastThreshold=20, HARRIS_SCORE；KAZE threshold=.001, nOctaves=4, nOctaveLayers=4,
extended=False, upright=False, DIFF_PM_G2；FAST threshold=10, nonmax=True, TYPE_9_16；
BRIEF 32 bytes, use_orientation=False；BRISK patternScale=1；FREAK direction/scale normalization=True。
具体 defaults 与全部实验参数也写入 metadata。

GT 只用于评估；RANSAC 用完整 raw descriptor matches。主要有方向协议：

- n_features：descriptor-retained query 中 GT 投影位于 target 有效区域的数量。
- n_putative：参与评估的一对一候选；n_correct：GT 前向误差<=3px；n_incorrect 为差。
- n_correspondences：几何阈值图中最大一对一二分图对应数，与 descriptor 无关。
- PMR=n_putative/n_features；precision=n_correct/n_putative；recall=n_correct/n_correspondences。
- matching_score=n_correct/n_features=PMR*precision；feature repeatability=n_correspondences/n_features。
- detector repeatability 单独用计算描述子前的检测点集评价，保存自身分子/分母。

零分母留 NaN，summary 有 mean/std/valid count；这是点距离协议，不声称复现椭圆 overlap repeatability。
RANSAC inlier ratio 的分母是 raw_putative_matches；inlier residual 根据 H_est，
GT correctness 根据 H_gt，二者不能混为一谈。
Homography error 是 source 四角经 H_est 与 H_gt 投影的 mean transfer error；
失败记录 False、原因与 NaN error，行保留。Report 同时列 mean、median、valid count、
failure count、success rate、3/5px accuracy；NaN 在 threshold accuracy 中算失败。

CPU 各阶段重复计时；GPU 在计时前后 synchronize。timing 排除I/O、建对象/加载模型、
CPU/GPU transfer、post-mask filter、GT、RANSAC 和画图。传统方法 detection 包含 cap 筛选，
learned/compatible extractor 记录联合时间。stage total 不是端到端延迟。
Descriptor memory 仅数组载荷，排除模型、激活、Python对象与显存峰值。

## 结果目录、数量与防覆盖

```text
results/
  ratio/{hpatches_classical,hpatches_superpoint,graf}/
  crosscheck/{hpatches_classical,hpatches_superpoint,graf}/
  lightglue/{hpatches_superpoint,hpatches_sift,graf}/
  superglue/{hpatches_superpoint,graf}/
  lightglue/hpatches_sift_control/   额外580行，不计主表
  lightglue/graf_control/           额外2行
  synthetic/{raw,summary,figures}/
  supplementary/{fast_parameters.csv,brief_sampling.csv,metadata.json,complete.json}
  comparison/                      新报告、配对表、runtime/memory、homography、synthetic curves
```

| 主实验 | 规模 |
|---|---|
| HPatches BF ratio | 7*580=4060 |
| HPatches BF crosscheck | 7*580=4060 |
| HPatches LightGlue | 2*580=1160 |
| HPatches SuperGlue | 1*580=580 |
| HPatches 合计 | **9860** |
| Synthetic | 8*28*10=**2240** |
| 两类主要合计 | **12100** |
| GRAF supplementary | 17 method/strategy combinations*2=34 |

主表有十种 method、17种 method/strategy组合；每组合 illumination=285、viewpoint=295。
BF control、GRAF、FAST/BRIEF 均独立统计，不改变主表计数。
FAST thresholds10/20/40 × nonmaxTrue/False，加 SIFT detector baseline；
教学 BRIEF uniform/Gaussian、256bits、31px patch、seed0。选定GRAF2对与HPatches10对；不做额外参数网格。

每个主目录写 progress.csv（逐行flush）、metadata.json、models/*.json（适用时）、
part1_raw_results.csv、part1_summary.csv、part1_computational_cost.csv、complete.json。
Synthetic 另存 sources.json、pairs.json、robustness.csv 和 context figures。
metadata 有代码/config SHA-256、输入图像/GT SHA-256、run_id、配置与预期行数。
程序遇到失败会停止；Homography 正常估计失败保留行。没有 silent skip 或 resume。

**默认拒绝非空目标目录，所有正式输出先预检查。没有 --overwrite，也不会自动清理。**
中断后的 progress 可检查，但不是完成数据；重试请选择新 `--results-root`，
或由用户审查后清理对应目标。不要清理 data/models。
Smoke 必须用独立 `tmp/smoke_*`；若已非空，改用新的名字。
`.gitignore` 使用 `/models/`，以保留 results 中的 model manifests。

Report 只读上述新路径。CSV库存、重复、类别、配置、参数、代码/输入身份、模型manifests、
特征配对和完成标记有校验；缺文件返回 PENDING/exit2，不创建假表或宣称 validation passed。
结果从真实CSV计算，不硬编码算法排名。所有比例采用pair宏平均，std为pair/source间sample std。
Qualitative figures 每集合最多100条线，标题记录全量数量；定量指标使用完整match集合。
最终新报告位于 `results/comparison/PART1_RESULTS.md`，不再覆盖项目根目录的旧报告。

## 手动执行顺序（以下命令均未由 Codex 执行）

在 `D:\桌面\IACV\Assigment1` 打开 PowerShell。先确认数据、本地模型与上述manifest。
默认GPU；需要CPU时在实验命令中把 `--device cuda` 改为 `--device cpu`。

### Step 1 — Unit tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原 tests 保留；其中原 geometry 测试会读一张GRAF图并做小型RANSAC验证。
新 learned contract tests 不加载checkpoint；真实模型integration默认skip，须显式选择：

```powershell
$env:IACV_RUN_MODEL_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_learned_contract.py -v
Remove-Item Env:IACV_RUN_MODEL_TESTS
```

上述第二条会执行模型推理，只由用户手动运行。测试永不下载模型；部分模型测试缺少checkpoint则skip，SuperGlue的缺文件/哈希不一致直接失败，
skip不是成功验证。所有当前测试状态：**NOT RUN — user will execute manually**。

另有默认skip的实际smoke测试文件，可在准备好数据/模型后手动运行（会处理图片和推理）：

```powershell
$env:IACV_RUN_SMOKE_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_smoke_experiments.py -v
Remove-Item Env:IACV_RUN_SMOKE_TESTS
```

### Step 2 — BF smoke（一个GRAF pair，SIFT/ORB，共享提取）

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset graf --methods sift orb --bf-shared --limit-pairs 1 --warmup 0 --repetitions 1 --device cpu --figures --output tmp/smoke_bf
```

应写4行ratio/crosscheck；同method的两行feature hashes和extraction IDs相同，matching time各自测量。
也可单独使用 `--matching-strategy ratio` 或 `--matching-strategy crosscheck`，改用两个独立tmp目录。

### Step 3 — Learned smoke

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset graf --methods superpoint_lightglue --limit-pairs 1 --warmup 0 --repetitions 1 --device cuda --figures --output tmp/smoke_superpoint_lightglue
.\.venv\Scripts\python.exe src/main.py --dataset graf --methods sift_lightglue --sift-control --limit-pairs 1 --warmup 0 --repetitions 1 --device cuda --figures --output tmp/smoke_sift_lightglue
```

第二条应有一个主配置和一个独立control，共2行；不加BF strategy参数。

三个 SuperPoint matcher 的共享提取 smoke（仅由用户手动执行，预期3行）：

```powershell
.\.venv\Scripts\python.exe src/main.py --dataset graf --methods superpoint superpoint_lightglue superpoint_superglue --limit-pairs 1 --warmup 0 --repetitions 1 --device cuda --figures --output tmp/smoke_superpoint_three_matchers
```

不要加 `--matching-strategy`；BF 自动 ratio，另两种自动注册的 learned matcher。

### Step 4 — Synthetic smoke

```powershell
.\.venv\Scripts\python.exe src/experiments/run_synthetic.py --limit-sources 1 --settings identity_0 rotation_30 scale_0.75 translation_x_0.1 shear_0.125 perspective_0.06 --methods sift orb --warmup 0 --repetitions 1 --device cpu --figures --output tmp/smoke_synthetic
```

应有1*6*2=12行。检查H方向、valid mask、overlap、特征过滤和figure；不是正式结果。

### Step 5 — Output/report validation

```powershell
.\.venv\Scripts\python.exe src/experiments/build_report.py --validate-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_results_runner.py -v
```

正式结果尚缺时第一条应打印PENDING并exit2，这是预期的缺结果状态，不代表正式数据已验证。
第二条检查临时目录防覆盖、共享runner mocks、missing-result判定和paired hash约束。
重复Step2同一输出目录应明确拒绝，保留已有文件；不要用正式results做smoke。

### Step 6 — 从零正式运行（先实验，后单独report）

```powershell
.\.venv\Scripts\python.exe run_part1.py --only experiments --device cuda --figures
```

顺序执行HPatches、GRAF、synthetic、supplementary；不启动并行进程。
从零完成所有阶段且直接生成最终比较报告的**一条推荐命令**为：

```powershell
.\.venv\Scripts\python.exe run_part1.py --device cuda --figures
```

两条是二选一；不要先跑前一条再跑默认all，以免触发防覆盖。
正式部分按默认协议，不需要手动重复指定2048/2/5等。

### Step 7 — 汇总真实正式结果

如果Step6使用 `--only experiments`：

```powershell
.\.venv\Scripts\python.exe run_part1.py --only report
.\.venv\Scripts\python.exe src/experiments/build_report.py --validate-only
```

如果Step6使用默认all，报告已经生成，只需 `--validate-only`。
需要重新生成report时请选择新目录，避免覆盖已有报告：

```powershell
.\.venv\Scripts\python.exe src/experiments/build_report.py --output results/comparison_review
```

### 各正式阶段独立调用

```powershell
.\.venv\Scripts\python.exe run_part1.py --only bf --device cuda --figures
.\.venv\Scripts\python.exe run_part1.py --only lightglue --device cuda --figures
.\.venv\Scripts\python.exe run_part1.py --only superglue --device cuda --figures
.\.venv\Scripts\python.exe run_part1.py --only synthetic --device cuda --figures
.\.venv\Scripts\python.exe run_part1.py --only supplementary
.\.venv\Scripts\python.exe run_part1.py --only report
```

这组也是默认all的替代方案。BF、LightGlue、SuperGlue分开运行时SuperPoint独立提取，report会检查feature内容；
严格共用SuperPoint提取测量需默认all或 `--only experiments`。
单策略阶段也支持，优先共享bf阶段：

```powershell
.\.venv\Scripts\python.exe run_part1.py --only ratio --device cuda --figures
.\.venv\Scripts\python.exe run_part1.py --only crosscheck --device cuda --figures
```

所有实验可用 `--results-root results/run_02` 选择全新根目录，report同样传该参数。
分阶段必须使用相同device、代码、参数和数据；修改源码/config后不能混合已有阶段结果。

## 静态审查依据、测试与尚未确认项

读到的课程资料是本地 `Assignment_1_Python_Keypoints_Descriptors_Stitching.pdf`
与 `CVC-Assign1-LightGlue-colab-demo.pdf`。前者§2.7–2.8规定指标，§2.3/2.5规定FAST/BRIEF，
§2.6介绍SuperGlue和LightGlue，官方LightGlue支持SIFT/SuperPoint pretrained配置；demo PDF提供Colab链接及两配置说明。
没有找到独立Slides文件，也未访问demo链接中的notebook内容；未声称已核对不存在的资料。

新增测试：matching strategies、synthetic geometry/masks、learned tensors/pinned API/opt-in inference、
evaluation公式/失败/GT隔离、results输出保护/runner共享/完整性。原test_geometry和test_brief保留。
所有unit、smoke、integration：**NOT RUN**；所有正式benchmark：**NOT RUN**。

仍需人工确认：八图场景多样性；SIFT checkpoint及SuperGlue官方文件的手动准备与实际加载；本机CUDA确定性kernel、
显存与full-resolution推理；OpenCV各descriptor实际保留的indices；warp/interpolation的可视质量；
CSV/figures的真实生成与全库存完成。AST语法检查无法证明这些运行性质。
源码职责与静态核查记录另见 `docs/PART1_IMPLEMENTATION.md`。
