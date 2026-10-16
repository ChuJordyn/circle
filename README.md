# 环形激光内径测量算法验证工程

本工程实现环形激光图像的软件测量流程，并保留原V5线激光验证。当前完整流程为：

```text
灰度预处理与环形 ROI 定位
→ 极坐标展开与周期接缝处理
→ 局部背景与噪声估计、主条纹连续路径筛选
→ 边界感知的二次拟合引导与法线灰度重心
→ 按条纹宽度分组的批量 Hessian/Steger Taylor 精修
→ 置信度判断、回退与缺失点标记
→ 亚像素坐标反变换、畸变与激光平面标定重建
→ 圆/椭圆拟合、内径与最小二乘圆基准圆度
→ 共同基准下的多截面锥度、中心偏移和直线度分析
```

图像处理和几何计算已经接通。当前以仿真标定验证；真实相机标定、采集与编码器同步仍需设备输入。

当前算法版本为 **V5**。本次改变了主条纹选择和法线引导模型：最多五个局部峰参与二阶连续路径筛选，法线由边界感知的局部二次拟合估计；同时加入信噪比和原图局部峰证据门控。Steger 仍使用 Hessian/Taylor 模型，以批量计算替代逐点重复转换和采样。

V5中心线核心面向每列一条目标中心线的线激光或展开图。新增 `src/ring_measurement/` 自动展开环形图并反变换；缺失区插值只辅助追踪，不计为有效测量。环形后端局部绑定原V5函数，只加速分位数、卷积、采样及等价路径计算，不修改原核心模块或其历史结果。

## 目录结构

```text
circle/
├─ docs/                    技术路线、论文调研、说明文档
│  └─ reference/            PDF、参考资料
├─ result/                  算法输出结果（由 .gitignore 忽略）
├─ scripts/                 可直接运行的实验脚本
├─ src/centerline/          可复用算法模块
│  ├─ preprocess.py         灰度化、背景扣除、ROI 和有效区域
│  ├─ gray_centroid.py      像素峰值与普通灰度重心
│  ├─ normal_guided_centroid.py
│  │                         法线估计和法线方向灰度重心
│  ├─ normal_guided_steger.py
│  │                         Hessian 特征分解和局部 Steger 精修
│  ├─ ridge_support.py      局部信号、连续路径和二次拟合引导
│  ├─ steger_batch.py       按尺度分组的批量精修
│  └─ pipeline.py           完整处理链
├─ tests/                   核心数学函数单元测试
├─ README.md                工程说明
├─ requirements.txt         Python 依赖
└─ .gitignore               Git 忽略规则
```

## 当前已完成

- 整理环形激光内径测量技术路线；
- 整理中心线提取相关论文调研；
- 实现预处理、ROI 和有效区域；
- 实现像素级峰值、普通灰度重心和法线引导灰度重心；
- 实现可按需调用的局部 Hessian/Steger、Taylor 位移、置信度门控和回退；
- 生成仿真线激光/曲线激光条纹；
- 默认在线执行完整法线重心加 Steger，保留关闭精修的配置；
- 报告完整链路误差、覆盖率、精修及回退点数、运行时间和超时次数；
- 增加核心数学函数单元测试。

## 运行方式

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

项目书实时工况（50 mm/s + 10 mm 步长）：

```bash
python scripts/accuracy_benchmark.py
```

该工况需要 5 FPS，即单帧预算 200 ms。`RealtimePipelineConfig` 默认 `refine_with_steger=True`；设置为 `False` 可只输出法线灰度重心。精修按支持宽度量化尺度，范围 0.75–2.0 px、系数 0.2，属于本工程参数策略。基准结果保存到：

```text
result/v5_benchmark/summary.json
```

基准使用 512×512 图像、九类场景和五组随机种子，每图预热 2 次、计时 10 次。不同种子生成不同噪声；重复计时使用同一张图。报告保存代码和图像摘要、完整配置、逐列误差、RMSE、偏差、最大误差、有效点比例和耗时。无效点不计入 RMSE，必须同时看覆盖率；仍有效的 Steger 回退点计入误差。

开发检查同时要求 RMSE≤0.1 px、覆盖率≥95%、所有计时样本≤200 ms。前两项是开发门槛，**不是项目书的物理验收要求**。误差是估计位置处垂直残差沿真值法线的投影，不是精确最短曲线距离。使用 `--require-pass` 可在任一场景未通过时返回非零退出码。

`scripts/realtime_benchmark.py` 调用同一协议。原先只判断耗时的 CSV 基准已被替换，历史 PASS 不代表精度达标。

技术方案要求系统误差≤±0.01 mm、圆度精度≤0.005 mm等物理指标。像素 RMSE 需经过标定、环形展开、拟合及完整误差预算才能关联这些要求。本次计时不包括采集、编码器同步、展开、内径拟合和报告生成，不构成整机验收。

## 当前 V5 验证记录

更新日期：2026 年 10 月 12 日。直接替换当前 V5 记录；算法代码及参数未变，汇总数据为 `result/v5_benchmark/current_summary.json`。

九类图统一为单因素仿真：普通直线为基准，背景20、主条纹幅度160、法向宽度σ=2.6 px、基础高斯噪声σ=5。同一种子配对同一噪声场；曲线和大曲率只改变形状，弱光只降低条纹幅度至42%，反光只增加局部亮斑，断裂只移除指定区间条纹信号。强噪声仅提高噪声σ至18，参数在测量算法前确定，因为原σ=24会让部分种子额外出现饱和。所有图共同采用uint8截断与量化。

单侧干扰采用幅度24的副条纹（比例15%、间距8 px、σ=2 px），评估主峰一侧不对称；纯饱和保持水平直线，仅将条纹幅度提高至280后截断到255。最终饱和、大曲率、反光、15%干扰图已核对一致。旧版输出、混合因素场景、诊断中间文件及文档预览缓存已清理。

五个随机种子共45张图片已导出并核对，图像清单为 result/v5_benchmark/image_manifest.json，原始PNG位于 result/v5_benchmark/test_images/。下表是重新测量种子7的九张512×512图，每图预热2次、计时10次；没有声称完成45张图的算法测量。输入变化前后的数字不能用来宣称算法改进。

| 场景 | RMSE px | 有效点 | 平均 ms | 最大 ms | 联合门槛 |
|---|---:|---:|---:|---:|---|
| 普通直线 | 0.03048 | 512/512 | 96.7 | 99.1 | 通过 |
| 曲线 | 0.03958 | 512/512 | 120.4 | 125.9 | 通过 |
| 大曲率 | 0.16989 | 512/512 | 152.7 | 177.5 | 未通过 |
| 弱光 | 0.06957 | 512/512 | 100.5 | 103.6 | 通过 |
| 噪声 | 0.10514 | 512/512 | 101.0 | 105.8 | 未通过 |
| 反光 | 0.11009 | 512/512 | 100.5 | 104.7 | 未通过 |
| 纯饱和 | 0.01618 | 512/512 | 87.9 | 91.0 | 通过 |
| 单侧干扰 15% | 0.03161 | 512/512 | 94.0 | 95.9 | 通过 |
| 断裂 | 0.03725 | 457/512 | 96.1 | 98.5 | 未通过 |

联合开发门槛为 RMSE≤0.1 px、覆盖率≥95%、所有计时样本≤200 ms。本表通过 5/9 项，90次计时均未超时；43项回归检查通过。大曲率、强噪声、反光精度超标，断裂覆盖率不足。像素误差指标与整机物理验收不同，计时不包括采集、环形展开、标定或内径拟合。

## 最终 V5 文件

算法位于 `src/centerline/`；生成和验证脚本位于 `scripts/`，回归检查位于 `tests/`。`line_grayscale_centroid_demo.py` 保留为仿真生成器所依赖的兼容模块。项目原始参考资料和结果记录保留在 `docs/`。

`result/v5_benchmark/` 保存唯一的最终测量汇总 `current_summary.json`、45张原始PNG、图像参数与摘要清单 `image_manifest.json`、中心线真值 `ground_truth.npz`、九类图总览和最终文件校验清单 `final_manifest.json`。真值文件按场景名称取数组，数组第x项为该列中心y坐标，五个种子的几何真值相同。最终汇总仅包含种子7的九图测量，其他种子仅完成数据集检查。

重新导出数据集：

```bash
python scripts/export_single_factor_images.py
```

复现已保存的单组测量，可另存到默认 `summary.json`，最终记录仍以 `current_summary.json` 为准：

```bash
python scripts/accuracy_benchmark.py --seeds 7 --warmup 2 --repeats 10 --save-images
```

如需测量全部45张图，使用前面的默认基准命令。重新测量后的时间受运行环境影响，不能把新耗时与最终记录混为同一次运行。

## 环形完整流程

运行环形仿真验证：

```bash
python scripts/ring_measurement_demo.py
```

默认五个种子、九类环形图、每圈3600个角度样本，每图预热2次、计时5次。角度从图像右侧开始顺时针增加；半径步长默认1 px。技术方案的1°/点与3600点存在矛盾，这里采用3600点，即0.1°/点，不能把它解释为3600个独立相机像素。

最终结果位于 `result/v5_ring_measurement/`：45张原始图、解析真值、逐点原图/毫米坐标与有效性和拟合诊断标记、完整测量汇总、过程示例，以及6米多截面报告和三维点云。仅对观测完整的法线和二维导数邻域输出有效点。圆度与椭圆度使用全部观测点，稳健拟合单列诊断，不静默删除真实磨损。

处理外部图片，可不提供标定而只输出像素结果：

```bash
python scripts/measure_ring.py picture.png --output result/v5_ring_measurement/external
python scripts/measure_ring.py picture.png --calibration calibration.json --output result/v5_ring_measurement/external
```

可用 `--center CX CY --radius-range MIN MAX` 指定已校准坐标系中的环形ROI。未指定时程序从图像估计。带畸变时展开坐标位于校正图像，取样与导出原图点位映射回畸变图；毫米点通过相机光线与激光平面求交，在平面正交基底中拟合。

标定JSON参照 `simulation_calibration.json`，仿真参数不能当成设备实测参数。支持 `camera_laser_plane` 和适用于等比例已校正平面的 `planar_scale`；后者可通过 `fit_scale_from_gauges` 从环规像素直径拟合。世界坐标由标定姿态定义；`pose_reference=zero_encoder` 表示姿态不包含扫描位移，`encoder_position` 表示已包含。多截面必须使用相同frame_id和姿态约定，避免重复添加编码器位移。

多截面API使用每帧的编码器位置和一个固定基准：

```python
from ring_measurement.pipeline import run_ring_measurement
from ring_measurement.sweep import analyze_sweep

sections = []
for image, position_mm in frames_with_encoder_positions:
    section = run_ring_measurement(image, calibration=calibration)
    section["position_mm"] = position_mm
    sections.append(section)
profile = analyze_sweep(sections, reference_origin_mm=[0, 0, 400],
                       reference_direction=[0, 0, 1])
```

上例基准原点是仿真装置参数，设备使用时必须替换为真实共同基准。源点、回退点和拟合诊断保持可追溯。磨损量需要公称内径；剩余壁厚还需要外径或壁厚信息，不能仅凭内壁图像推断。

当前完整流程74项测试通过，45张环形图225次计时最慢186.7 ms。普通圆环原图法向RMSE为0.0424–0.0442 px，五个种子的内径范围0.004432 mm。联合像素/覆盖率/时间门槛通过34/45图；强噪声、部分弱光精度及断裂覆盖率仍不满足要求。

仿真采用512×512传感器和0.5 mm/px尺度。反光及副条纹会令毫米内径误差超出±0.01 mm；本组圆度测量误差均未达到0.005 mm。这些结果明确不能作为技术方案的整机达标结论。完整流程代码实现完成，真实计量验收仍需提高像素精度、合理的光学分辨率、真实标定和环规数据。

## 仍需设备输入的工作

- 真实相机采集与编码器同步，测量姿态和全局基准；
- 相机内参、畸变、激光平面及标准环规的实际标定；
- 温漂和探杆变形的实测补偿参数；
- 整机物理精度、重复性及真实运行速度验证。
