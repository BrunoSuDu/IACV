# Part 1 实验结果


已完成 8 种配置 × 580 对 = 4640 行 HPatches 主实验。每方法包含 285 对光照变化、295 对视角变化。

全分辨率图像；共同最大预算 2048 点；CPU 4 线程；预热 2 次、重复 5 次；GT 阈值 3 px、ratio 0.8。

分母、硬件、模型和计时边界见 README 与各运行目录 metadata.json。所有表格直接从 CSV 计算。

## 匹配与重复性（每对宏平均；比例 0–1）


| method | category | detector_repeatability | repeatability | pmr | precision | matching_score | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sift | illumination | 0.3691 | 0.3691 | 0.2278 | 0.7186 | 0.1924 | 0.4256 |
| sift | viewpoint | 0.3775 | 0.3775 | 0.2535 | 0.7449 | 0.2150 | 0.5104 |
| orb | illumination | 0.4961 | 0.4961 | 0.1866 | 0.6868 | 0.1586 | 0.2573 |
| orb | viewpoint | 0.5086 | 0.5086 | 0.1733 | 0.6853 | 0.1454 | 0.2548 |
| kaze | illumination | 0.4704 | 0.4704 | 0.2647 | 0.7668 | 0.2283 | 0.4026 |
| kaze | viewpoint | 0.4575 | 0.4575 | 0.2196 | 0.7403 | 0.1870 | 0.3553 |
| fast_brief | illumination | 0.5074 | 0.5064 | 0.2938 | 0.7725 | 0.2545 | 0.4473 |
| fast_brief | viewpoint | 0.4909 | 0.4831 | 0.1661 | 0.5850 | 0.1341 | 0.2493 |
| fast_brisk | illumination | 0.5074 | 0.5066 | 0.2001 | 0.7355 | 0.1721 | 0.2918 |
| fast_brisk | viewpoint | 0.4909 | 0.4878 | 0.1086 | 0.5429 | 0.0839 | 0.1488 |
| fast_freak | illumination | 0.5074 | 0.5065 | 0.1986 | 0.6633 | 0.1615 | 0.2720 |
| fast_freak | viewpoint | 0.4909 | 0.4844 | 0.1283 | 0.5316 | 0.0922 | 0.1664 |
| superpoint | illumination | 0.5327 | 0.5327 | 0.3599 | 0.8307 | 0.3113 | 0.5286 |
| superpoint | viewpoint | 0.4890 | 0.4890 | 0.3203 | 0.7724 | 0.2763 | 0.4967 |
| superpoint_lightglue | illumination | 0.5327 | 0.5327 | 0.5923 | 0.8310 | 0.5003 | 0.9182 |
| superpoint_lightglue | viewpoint | 0.4890 | 0.4890 | 0.5481 | 0.8065 | 0.4572 | 0.9050 |

detector_repeatability 使用描述子计算前的选定点集；repeatability 使用描述子保留点集。

## 平均计数与分母


| method | category | features_image1_total | features_image2_total | n_features | raw_putative_matches | n_putative | n_correct | n_correspondences |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift | illumination | 1665.2456 | 1608.4947 | 1665.0596 | 355.9193 | 355.9158 | 295.1895 | 590.7474 |
| sift | viewpoint | 1909.2712 | 1875.3492 | 1771.3254 | 461.0102 | 456.9153 | 391.2949 | 674.2847 |
| orb | illumination | 1956.8596 | 1940.2105 | 1956.8596 | 362.5439 | 362.5439 | 307.3439 | 963.8772 |
| orb | viewpoint | 2005.1525 | 2006.5051 | 1888.6915 | 340.1017 | 337.4441 | 284.5085 | 967.5186 |
| kaze | illumination | 1396.1404 | 1406.8737 | 1396.1298 | 351.9018 | 351.9018 | 297.0070 | 638.5895 |
| kaze | viewpoint | 1797.0508 | 1798.1085 | 1671.1288 | 379.0949 | 376.8034 | 323.2034 | 784.0441 |
| fast_brief | illumination | 1824.4211 | 1798.4912 | 1824.4211 | 533.2316 | 533.2316 | 460.9509 | 918.7088 |
| fast_brief | viewpoint | 1900.5593 | 1906.7119 | 1765.9559 | 303.2271 | 300.2915 | 244.8441 | 853.5322 |
| fast_brisk | illumination | 1921.8246 | 1896.7930 | 1921.7614 | 382.4175 | 382.4175 | 328.7965 | 968.7684 |
| fast_brisk | viewpoint | 1988.2881 | 1980.3864 | 1829.6373 | 206.8542 | 204.0407 | 158.5966 | 887.1254 |
| fast_freak | illumination | 1853.3860 | 1828.4070 | 1853.3860 | 365.7544 | 365.7544 | 296.9754 | 933.7053 |
| fast_freak | viewpoint | 1925.4237 | 1928.9186 | 1784.6305 | 237.6508 | 233.3559 | 168.4102 | 863.5220 |
| superpoint | illumination | 1688.5263 | 1691.0035 | 1688.1368 | 617.0140 | 616.9930 | 530.4035 | 891.3509 |
| superpoint | viewpoint | 1981.6610 | 1974.0441 | 1817.1559 | 596.1017 | 594.8373 | 513.4203 | 888.3119 |
| superpoint_lightglue | illumination | 1688.5263 | 1691.0035 | 1688.1368 | 1007.4842 | 1007.4632 | 840.8982 | 891.3509 |
| superpoint_lightglue | viewpoint | 1981.6610 | 1974.0441 | 1817.1559 | 997.3627 | 997.1186 | 835.0373 | 888.3119 |

本表是每对计数的平均值；上表是逐对比值的宏平均，不能简单用本表两列相除复现。逐对分子分母见 raw CSV。

## 计算成本


| method | category | extraction_pair_ms | matching_ms | bytes_per_descriptor | pair_stage_total_ms |
| --- | --- | --- | --- | --- | --- |
| sift | illumination | 403.4325 | 13.9284 | 512.0000 | 417.3609 |
| sift | viewpoint | 373.9336 | 13.6666 | 512.0000 | 387.6002 |
| orb | illumination | 22.4023 | 10.1981 | 32.0000 | 32.6004 |
| orb | viewpoint | 29.7135 | 10.6297 | 32.0000 | 40.3432 |
| kaze | illumination | 1170.1121 | 5.1628 | 256.0000 | 1175.2748 |
| kaze | viewpoint | 1459.0577 | 7.4975 | 256.0000 | 1466.5552 |
| fast_brief | illumination | 13.5880 | 9.1104 | 32.0000 | 22.6984 |
| fast_brief | viewpoint | 18.8343 | 9.4086 | 32.0000 | 28.2429 |
| fast_brisk | illumination | 28.7381 | 10.0985 | 64.0000 | 38.8366 |
| fast_brisk | viewpoint | 34.0410 | 10.6290 | 64.0000 | 44.6700 |
| fast_freak | illumination | 23.1513 | 9.3567 | 64.0000 | 32.5080 |
| fast_freak | viewpoint | 28.4932 | 10.0493 | 64.0000 | 38.5425 |
| superpoint | illumination | 72.1416 | 23.6272 | 1024.0000 | 95.7689 |
| superpoint | viewpoint | 54.3200 | 34.1969 | 1024.0000 | 88.5169 |
| superpoint_lightglue | illumination | 73.1058 | 21.5838 | 1024.0000 | 94.6896 |
| superpoint_lightglue | viewpoint | 57.2979 | 20.9943 | 1024.0000 | 78.2922 |

extraction_pair_ms 为两图提取时间之和；传统方法合并检测和描述，SuperPoint 使用联合提取时间。

stage total 排除 I/O、CPU/GPU 传输、模型加载、GT 和 RANSAC，不能当成实际端到端帧率。

## Homography


| method | category | inliers | residual_px | corner_mean_px | corner_median_px | valid_corner_errors | estimation_success | corner_within_3px | corner_within_5px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sift | illumination | 310.6772 | 0.4922 | 23.5205 | 0.7018 | 285 | 1.0000 | 0.8632 | 0.9333 |
| sift | viewpoint | 389.7085 | 0.8169 | 103.9005 | 2.2663 | 295 | 1.0000 | 0.5763 | 0.6847 |
| orb | illumination | 320.3719 | 0.7233 | 43.3213 | 1.3698 | 284 | 0.9965 | 0.7123 | 0.8246 |
| orb | viewpoint | 286.1051 | 1.1322 | 118.3265 | 4.1387 | 293 | 0.9932 | 0.3797 | 0.5559 |
| kaze | illumination | 318.1965 | 0.6205 | 258.5594 | 1.1463 | 283 | 0.9930 | 0.7754 | 0.8632 |
| kaze | viewpoint | 321.3356 | 1.0498 | 140.9828 | 3.5450 | 295 | 1.0000 | 0.4576 | 0.6305 |
| fast_brief | illumination | 483.9930 | 0.7453 | 12.1451 | 0.9498 | 285 | 1.0000 | 0.8386 | 0.9333 |
| fast_brief | viewpoint | 245.0814 | 0.9702 | 138.2164 | 4.9954 | 290 | 0.9831 | 0.3153 | 0.4915 |
| fast_brisk | illumination | 340.7263 | 0.4655 | 42.0915 | 0.5911 | 285 | 1.0000 | 0.8947 | 0.9474 |
| fast_brisk | viewpoint | 159.5559 | 0.7805 | 269.2630 | 4.6069 | 290 | 0.9831 | 0.3593 | 0.5017 |
| fast_freak | illumination | 307.1053 | 0.5069 | 24.7508 | 0.6524 | 285 | 1.0000 | 0.8456 | 0.9333 |
| fast_freak | viewpoint | 168.7932 | 0.8126 | 186.3612 | 4.4111 | 293 | 0.9932 | 0.3525 | 0.5390 |
| superpoint | illumination | 558.8281 | 1.0396 | 1.6579 | 0.9407 | 285 | 1.0000 | 0.8737 | 0.9579 |
| superpoint | viewpoint | 511.4136 | 1.1684 | 37.3619 | 3.1173 | 287 | 0.9729 | 0.4746 | 0.6271 |
| superpoint_lightglue | illumination | 881.1684 | 1.2088 | 1.5430 | 1.1985 | 285 | 1.0000 | 0.8982 | 0.9789 |
| superpoint_lightglue | viewpoint | 826.0475 | 1.2775 | 24.8120 | 2.2140 | 292 | 0.9898 | 0.6102 | 0.7695 |

均值与中位数同时给出，以显示灾难性错误对均值的影响。阈值成功率将缺失 H 计为失败。

## 观察


- illumination：matching score 最高为 superpoint_lightglue (0.5003)；precision 最高为 superpoint_lightglue (0.8310)；所测阶段总耗时最低为 fast_brief (22.70 ms)。

- viewpoint：matching score 最高为 superpoint_lightglue (0.4572)；precision 最高为 superpoint_lightglue (0.8065)；所测阶段总耗时最低为 fast_brief (28.24 ms)。


这些排名只适用于本次实现、阈值、点数预算和设备；不能把 GPU 与 CPU 差异归因于算法本身。

视角变化同时影响局部形变、可见范围和尺度；描述子具备方向归一化也不能保证任意视角下正确。

BRIEF 固定采样对不做方向或尺度归一化；BRISK/FREAK 的方向处理、SIFT/KAZE 的尺度空间是不同设计。

SuperPoint 与 SuperPoint+LightGlue 的特征配置相同，因此比较它们主要反映 matcher 的差异。

RANSAC 内点多或残差低仍可能来自错误的重复纹理模型；GT 角点误差能揭示这种情况。

实时方案需要同时满足延迟和精度；可先从本机时间表较快的方法试起，并加入真实 I/O、传输、RANSAC 再测端到端延迟。

精度优先不能只看 precision：应结合 matching score、recall、正确匹配数和 H 成功率。

当前数据没有将 scale 和 rotation 独立控制，不能据此得出独立的旋转／尺度不变性曲线。


补充实验：results/supplementary/fast_parameters.csv 与 brief_sampling.csv 只覆盖 12 对选定图像。

GRAF：results/graf/；代表图：各主实验目录 figures/。旧 smoke_* 只作运行检查，hpatches_uncapped_pilot 为中止的旧参数试跑。


Part 2 拼接和整份作业的最终报告仍需另行完成。本文是 Part 1 结果说明，不冒充整份作业报告。
