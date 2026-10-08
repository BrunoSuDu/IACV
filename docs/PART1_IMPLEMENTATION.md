# Part 1 实现交付与静态审查记录

状态：代码与文档已准备；所有 unit/integration/smoke tests **NOT RUN**；所有 benchmark **NOT RUN**。
语法检查仅解析 Python AST 与 JSON，没有导入项目、提取特征、运行匹配/RANSAC或加载模型。
完整人工执行命令及前置条件见项目 README。

## 修改文件与职责

| 文件 | 变更职责 |
|---|---|
| run_part1.py | 顺序正式入口；all/experiments/bf/ratio/crosscheck/lightglue/superglue/synthetic/supplementary/report；预检查输入、模型、输出 |
| src/main.py | 独立数据集入口；ratio默认、显式不支持组合报错、bf-shared与smoke输出 |
| src/features.py | 十种方法注册、BF/LightGlue/SuperGlue分组、独立SIFT-compatible control |
| src/matching.py | 沿用ratio定义，加入真正mutual NN crosscheck，空结果与一对一 |
| src/classical_pipeline.py | 策略参数传递与独立matching计时；原检测/描述参数保留 |
| src/learned_features.py | 官方pinned SIFT/SuperPoint适配、空SIFT输出、RootSIFT、scale/ori、device、manifest、checkpoint本地预检查与参数覆盖 |
| src/evaluation.py | 可选source/target masks，双向co-visibility，索引范围检查；无mask原协议保留 |
| src/homography.py | 统一RANSAC参数、估计异常/退化失败原因及NaN，不使用GT估计 |
| src/timing.py | 将函数默认重复次数统一为5；保留CPU/GPU同步计时边界 |
| src/experiments/runner.py | 按feature source共享提取、reference缓存、分别match/evaluate、hash/测量ID、逐行progress与完整性 |
| src/experiments/reporting.py | 策略分组、synthetic分组、代码身份、参数/计时语义metadata |
| src/experiments/build_report.py | 新路径库存/配置/模型/配对校验，真实CSV统计，learned control对比，synthetic曲线，pending而非假报告 |
| src/experiments/run_supplementary.py | 接入新runner、统一参数、输出保护、输入manifest；保留FAST/BRIEF范围 |
| src/visualization.py | 新增original/transformed/valid-mask context；完整数量与100条展示上限 |
| .gitignore | models/改为/models/，保留实验model manifests |
| README.md | 完整实验设计、结果目录、计数、限制、人工PowerShell步骤 |

## 新增文件

| 文件 | 职责 |
|---|---|
| configs/synthetic_sources.json | 固定4+4 reference paths与独立选图规则，视觉多样性待人工确认 |
| src/transformations.py | 28 unique settings；float64 original→transformed H；有效角点/polygon；fixed canvas和valid mask |
| src/feature_masks.py | 统一support过滤并同步native索引；evaluation安全区域、特征内容SHA-256 |
| src/datasets/synthetic.py | manifest读取、缺图失败、SyntheticPair与配置计数 |
| src/experiments/protocol.py | 正式参数、stage目录、保护源文件/非空输出、输入/code hashes、正式pair库存 |
| src/experiments/run_synthetic.py | 224pairs/2240主记录、共享SuperPoint、identity去重、progress/raw/summary/pair metadata/context |
| tests/test_matching_strategies.py | ratio兼容、k=2、strict threshold、一对一、crosscheck mutual NN/空描述子/norm/共享输入 |
| tests/test_synthetic_geometry.py | 已知点投影、六种H、逆矩阵、corner rule、mask/canvas/overlap、support/native alignment、co-visibility、28与固定sources |
| tests/test_learned_contract.py | batched shapes、device、RootSIFT/pinned参数、LightGlue索引、无下载、拒绝BF；真实模型测试opt-in |
| tests/test_evaluation_protocol.py | PMR/precision/recall/score、<=3px、空分母、错误索引、RANSAC输入方向与失败 |
| tests/test_results_runner.py | 十方法/17组合/12100计数、output隔离/保护、pending、共享runner mocks、paired feature hash |
| src/superglue_adapter.py | 固定官方源码/权重 Git blob 校验；真实scores、转置descriptors、尺寸与像素坐标、mutual输出检查 |
| tests/test_superglue_contract.py | 文件身份、缺文件拒绝、输入布局/原值、空输出、索引/一对一、scores/image_size哈希；无推理 |
| tests/test_smoke_experiments.py | 默认skip；显式opt-in才运行BF、两种LightGlue、SuperGlue和synthetic小样本，输出unique tmp |
| docs/PART1_IMPLEMENTATION.md | 本交付记录 |

原 detectors/descriptors/brief/image_io/datasets/hpatches/graf 及原 tests 保留，不进行不必要重写。
requirements 保持原有版本与约束；requirements-learned.txt 只增加 SuperGlue 手动安装来源注释，没有升级或增加 pip 依赖。

## 静态检查要点

- Ratio仍是两个最近邻的严格比值测试加greedy one-to-one；crosscheck是match互查，不混用ratio。
- full/bf共享同一ImageFeatures；matching分别计时；内容hash和extraction ID区分同内容与同测量。
- 原SIFT不改；官方compatible SIFT的threshold/octaves/RootSIFT不同，新增独立BF control隔离matcher。
- 官方输入keypoints/128D RootSIFT/scales/oris/image_size在同设备，索引过滤同步，输出索引检查。
- float64 H精确保存，warp直接使用该H；没有GT估计器。固定canvas导致裁切，overlap单独记录。
- 有效mask与warp同坐标，constant-zero coverage避免反射padding算真内容；共同support后过滤。
- GT只在评价可见性/正确性/最大一对一几何对应中使用；RANSAC继续用未GT过滤的raw matches。
- 零分母NaN、H失败保留行；report列valid/failure/success及mean/median，threshold accuracy将NaN算失败。
- HPatches 9860、synthetic 2240严格分开；control不改变主配置计数；GRAF只有两对。
- progress逐行flush；complete标记只在库存校验后写；非空目录拒绝，没有resume/隐式覆盖。
- 本轮已检查并清理旧results的21 CSV、8 JSON、144 PNG及旧根目录生成报告；未动data/models。

## 具体未确认事项

1. 当前本地缺少 sift_lightglue_v0-1_arxiv.pth，以及 SuperGlue 固定官方源码、LICENSE、outdoor checkpoint；README 提供手动准备命令，runner不会代为下载。
2. 仅静态核对官方安装commit和API，没有运行模型或验证checkpoint与本机推理性能。
3. frozen sources按稳定名称选择，未视觉确认texture/architecture/nature等场景覆盖。
4. 后过滤不是完全消除padding artifacts的证明；尺度空间、CNN和top-k仍可能受影响，README说明该限制。
5. full-resolution的显存、CUDA确定性kernel及时间成本尚未运行验证；不静默改变分辨率或device。
6. 没有找到单独Slides，也未打开demo的远程Colab notebook；只读取了两份本地课程PDF。
7. AST语法通过不代表unit、smoke、模型兼容性、实际CSV生成或全实验完成。所有这些状态均NOT RUN。

请按README的unit→BF smoke→learned smoke→synthetic smoke→output validation→full experiment→report顺序手动执行。

## 本次恢复工作后的重新审查与修复

- 第十主配置 `superpoint_superglue` 已并入同一 registry/feature grouping/HPatches/GRAF/synthetic runners；不是另建实验框架。
- 官方 SuperGlue 固定 commit `ddcf11f42e7e0732a0c4607648f9448ea8d73590`，outdoor/MegaDepth 权重，参数保持官方默认（100 Sinkhorn、.2 threshold）。选择是预先设定，未作最优性或性能声称。
- 仅在线读取官方源码、README、LICENSE 与 Git tree metadata；没有下载模型或将第三方源码复制进项目。
- 已只读哈希确认现有 SuperPoint checkpoint 与官方 SuperGlue 所用文件一致。BF/LG/SG 共用现有 pinned extractor 的原始 scores、像素 keypoints、L2 descriptors；SG只转置描述子。官方归一化保留在模型内部。
- 特征指纹原先遗漏 native scores/image_size，现已覆盖，且包含字段名、形状、dtype；report可拒绝表面相同、实际matcher输入不同的对照。
- 补齐 runner mock 缺少的 n_correct/n_putative/homography_error_px 字段，避免日志格式化导致测试提前 KeyError；增加三matcher共享提取的纯mock测试，没有降低断言。
- RANSAC metadata 改为记录 options 中的实际 max_iters/confidence；以前会对自定义轻量运行错误记录正式默认值。
- report 增加 SG版本/权重/文件身份检查，更新十方法、17组合、9860/2240/12100库存，GRAF为34主行。
- 三组SuperPoint matcher配对比较覆盖三个数据集；检查完整配对、特征内容和提取ID，记录quality/H success/error/median/accuracy及matching时间。布尔success差值显式转float，避免布尔减法错误。
- report 补强指标精确分母、整型计数、布尔值、实际stage组合、设备/环境、synthetic逐pair方法集合、BF策略、类别前缀和补充实验逐设置库存；FAST mixed boolean/NaN列显式解析。
- GRAF的SIFT-compatible BF control现在也纳入独立验收与对比；不改变主计数。
- 拒绝重复/空methods、非有限几何阈值；处理OpenCV descriptor空keypoints的返回形式。
- synthetic metadata明确仅BF ratio、不包含SIFT control，避免继承主入口的额外策略标记。

静态检查：38个Python文件的AST解析与内存中语法编译通过，1个JSON配置解析通过；没有执行或导入这些项目模块。
`git diff --check`通过（Git仅提示已有LF/CRLF策略）；新文件另外做空白与冲突标记检查。
所有单元、integration、smoke、HPatches、GRAF、synthetic、FAST/BRIEF运行状态仍是 **NOT RUN — requires manual execution**。
没有Git commit/push，没有修改原始data、checkpoint或课程PDF。

运行限制仍需人工检查：官方旧版SuperGlue与本地torch/CUDA实际加载，full-resolution/2048点的显存，CUDA确定性kernel，图像/CSV/figure实际产物。
不确定性不能由静态语法检查消除。下一步先按README准备缺失文件，再由用户自行运行测试和正式入口。
