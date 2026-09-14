# DRMNet 三维模块融合实验

目标：在 LUNA16 上评估检测召回率，运行设备为用户的 NVIDIA L20。
本次交付是结构和实验入口；没有训练结果，也没有测得 L20 显存、速度或召回率。

## 1. 四个可独立运行的版本

| `--variant` | 编码器 | 解码器 |
| --- | --- | --- |
| `baseline` | 原来的共享单向 Mamba | 原来的上采样、拼接、卷积 |
| `mamba` | 第 3、4 级改为独立的六方向三维模块 | 原结构 |
| `freqfusion` | 原结构 | 第 2 级（1/4 分辨率）加入三维频率融合 |
| `fusion` | 第 3、4 级六方向模块 | 第 2 级三维频率融合 |

不指定版本且从头训练时使用 `baseline`。断点续训和评估自动读取检查点中的
`model_config`。如果显式版本与检查点不符，会报错；没有配置元数据的历史检查点
按 `baseline` 处理。四组实验应从头训练、使用相同数据划分、随机种子与训练预算。
`--ckpt` 只用于同结构断点续训，不是跨结构的部分权重初始化。

编码器输出仍为 64 通道，默认输入 128³ 时四级分辨率是 64³、32³、16³、8³。
RPN 输入仍是 128 通道、32³、步长 4，RCNN 使用的浅层特征仍是 64 通道、步长 2。
检测头、锚框、阈值、损失和数据增强未因版本切换而改变。

## 2. 实现及与参考仓库的区别

### MultiDirectionMamba3d

文件：`net/fusion_modules.py`；接入位置：`net/resnet.py` 的 `_context`。

* 采用三种空间排列 `(D,H,W)`、`(H,W,D)`、`(W,D,H)`，每种正向和反向各一次。
* 各扫描结果先逆序、恢复到原始 D/H/W 坐标，再用可学习 softmax 权重融合。
* 一个模块内部六路共享一个 `mamba_ssm.Mamba`；不同编码器级别的模块彼此独立。
* 增加深度可分离三维局部分支和可学习残差缩放，初值 0.01。
* 新模块的 SSM 路径使用 FP32，支持在常规 FP32 参数模型外使用 AMP；不要直接
  对整个模型调用 `.half()`。保留的原始 Mamba 路径使用原来的精度策略。
* 默认开启新模块的 activation checkpointing；以重计算换取训练显存。

这是受 MedSegMamba 的三维扫描、局部卷积和残差结构启发的独立实现，**不是
MedSegMamba 的完整复现**。没有移植其固定大小蛇形/Hilbert 索引，也没有复制其
SS3D 内部的方向特定投影；这里复用 DRMNet 已依赖的 Mamba 序列模块。
模块本身支持非立方体；完整 FeatureNet 的输入各空间维仍需按原网络要求填充至
16 的倍数，以匹配转置卷积和跳跃连接。

### FreqFusion3d

文件：`net/fusion_modules.py`；接入位置：`net/feature_net.py` 的 `comb2` 拼接之前。

* 在原上采样和 Mamba 之后，使用浅层与深层特征共同生成逐体素滤波权重。
* 自适应低通平滑深层特征，自适应高通残差增强浅层特征，两个分支分别门控。
* 默认压缩通道为 16，滤波核为 3³，使用归一化 Hamming 窗约束。
* 滤波权重在通道间共享；逐邻域位置累加，避免显式构造
  `B×C×27×D×H×W` 的完整邻域张量。实际速度和显存仍需在 L20 上测量。
* 边界使用 replicate padding，常数特征不会在体积边界产生高通伪响应。

这是 **FreqFusion 启发的简化三维版本**：保留自适应高/低通思想，增加门控，
没有实现原论文完整 CARAFE 上采样、偏移采样或迭代 mask 生成。默认接入时，
两路已经通过原来的解码路径对齐分辨率。不能将该版本报告为原论文模块的严格复现。

参考仓库不作为运行时依赖；训练无需把两个外部项目添加到 PYTHONPATH，
也无需它们的 MRI/ImageNet 权重、MMCV、Hilbert 或 pyzorder 依赖。

## 3. 来源记录

* [MedSegMamba](https://github.com/aaroncao06/MedSegMamba)，本地参考
  `/Users/lay/Data/Intern/gkry/Project/MedSegMamba`，commit
  `7f86b43cd484f44ab4ebc8803342012ff52e2a16`。
  阅读了 `models/SS3D.py`、`models/VSS3D.py` 和 README。
  论文：[arXiv:2409.08307](https://arxiv.org/abs/2409.08307)。
* [FreqFusion](https://github.com/Linwei-Chen/FreqFusion)，本地参考
  `/Users/lay/Data/Intern/gkry/Project/FreqFusion`，commit
  `3fb0c70637a3c194fb74294d3ce4681958b26241`。
  阅读了根目录 `FreqFusion.py`，对应 *Frequency-aware Feature Fusion for Dense Image Prediction*。

外部仓库未修改。上述三维适配以机制为参考独立编写，未直接搬入上游完整源码。
本项目原 README 所述非商业研究许可继续适用；论文中应引用 DRMNet 和上述参考工作，
并清楚说明适配与新增结构。

## 4. 在 L20 上先做真实 CUDA 检查

在原 DRMNet 已能运行的 Linux/CUDA 环境中执行。继续使用与服务器 PyTorch/CUDA
匹配的 `mamba_ssm`、原有检测依赖，不要用本次 macOS 临时
测试环境替换服务器环境。新模块不需要额外第三方运行库。
从零搭建环境见 [L20 环境与运行命令](l20_setup.md)。原 `box` 扩展不可用时，
`utils/pybox.py` 现在使用 NumPy CPU 备用实现；不需要运行旧的 `torch.utils.ffi`
编译脚本。它保留连续中心尺寸框的 IoU 和 `IoU >= threshold` 抑制规则，
候选很多时会比编译实现慢。

在 DRMNet 根目录执行：

```bash
python scripts/smoke_fusion.py --variant baseline --shape 128 128 128 --batch-size 1
python scripts/smoke_fusion.py --variant fusion --shape 128 128 128 --batch-size 1
python scripts/smoke_fusion.py --variant fusion --shape 128 128 128 --batch-size 1 --amp
python -m pytest tests/test_fusion.py -q
```

smoke 脚本使用真实 CUDA Mamba，先预热再运行三次合成训练步，检查新增参数梯度、
RPN 输出形状并打印显存与耗时。没有 CUDA 时明确退出，不会自动换成替代算子。
它只覆盖 **FeatureNet + RPN**，不覆盖数据加载、候选框生成、RCNN、NMS，也不表示
整例 CT 推理速度。小尺寸排错可以使用 `--shape 32 32 32`。

训练入口目前仍沿用原工程的 FP32 训练循环；smoke 的 `--amp` 仅用于数值测试，
不会自动给 `train.py` 启用混合精度。默认新模块使用 checkpointing。

## 5. LUNA16 单数据集训练示例

下列 `/data/LUNA16/...` 均为示例，必须换成实际路径。已有文件需要符合原
`BboxReader` 的预处理图像、框标注及 filelist 格式；官方原始 CT 不能直接传入。
使用按患者隔离的官方 folds，训练/验证/测试文件列表互不重叠。

```bash
python train.py --variant fusion \
  --train-set /data/LUNA16/fold0/train.npy \
  --val-set /data/LUNA16/fold0/val.npy \
  --train-data-dir /data/LUNA16/preprocessed \
  --val-data-dir /data/LUNA16/preprocessed \
  --train-anno /data/LUNA16/fold0/train_annos.csv \
  --val-anno /data/LUNA16/fold0/val_annos.csv \
  --batch-size 1 \
  --out-dir ./results/luna16_fold0_fusion
```

将 `--variant` 和独立输出目录改为 `baseline`、`mamba`、`freqfusion` 即可进行其余三组。
batch size 1 是首次排错起点，所有组保持相同实际 batch size。确认显存余量后可增加。
单数据集时，`StackDataset + train_collate` 每个 batch 只提供该数据集样本；
多源配置会按源数扩展实际 batch size。

续训示例：使用相同数据参数，增加 `--ckpt ./results/luna16_fold0_fusion/model/021.ckpt`。
省略 `--variant` 时自动恢复检查点结构；指定不同结构会报错。

## 6. 评估与召回率

```bash
python test.py --mode eval \
  --weight ./results/luna16_fold0_fusion/model/080.ckpt \
  --test-set-name /data/LUNA16/fold0/test.npy \
  --preprocessed-data-dir /data/LUNA16/preprocessed \
  --test-anno /data/LUNA16/fold0/test_annos.csv \
  --froc-annotations /data/LUNA16/evaluation/annotations.csv \
  --froc-excluded /data/LUNA16/evaluation/annotations_excluded.csv \
  --froc-seriesuids /data/LUNA16/fold0/test_seriesuids.csv \
  --out-dir ./results/luna16_fold0_fusion
```

`--test-anno` 是数据读取器的标注；`--froc-*` 是原 FROC 评估器使用的标注与病例列表。
二者格式和坐标可能不同，必须与本项目预处理、预测框导出坐标匹配；不能只因文件名相似
就互换。此次没有数据可核验整个坐标转换流程。FROC seriesuids 只能包含当前测试划分。

原评估脚本分别输出 RPN、RCNN、ensemble 的 FROC。优先比较固定 1、2、4 FP/scan
下的召回率和七点 CPM，再分析 RCNN 是否过滤掉 RPN 已找到的结节。
各组保持相同候选阈值和 NMS 设置；阈值扫描只在验证集做，不用测试集选结构或权重。
结节大小分组需使用物理尺寸，本次未新增分组评估器。

## 7. 本次验证边界与保留的问题

* CPU 测试验证了六方向坐标与因果扫描顺序、非立方体输入、滤波数值和自动微分、
  常数保持、脉冲的平滑/增强、checkpoint 前后输出与梯度一致，以及四个版本的
  FeatureNet/RPN/RCNN head 张量接口。
* CPU 集成测试显式注入 `CausalTestMixer`，不是 Mamba，也不是生产环境后备算子。
  CUDA 测试仅在 NVIDIA CUDA 与 `mamba_ssm` 同时可用时运行。
* 基线重构另外与修改前 Git HEAD 对照：在相同 CPU 测试 mixer 下，state_dict 键、
  同种子初始权重、严格加载和前向输出一致。真实 Mamba 历史权重仍需服务器验证。
* FeatureNet 和两个检测 head 被拆到独立文件，`net.sanet` 仍导出相同类名，
  `SANet.feature_net/rpn/rcnn_head` 参数前缀不变。
* 修复了构造器忽略传入 cfg、以及 RPN reshape 读取全局锚框数量的问题，
  确保选中的结构配置和检查点配置实际生效。
* 原 RADA 的采样写法、全图随机低秩正则、推理期间额外 SVD 都保留，避免把损失修正
  与结构增益混在一起。本次 `baseline` 是代码基线，不是先前建议的“修正后基线 B”。
  后续应将标准重参数化和区域采样作为单独实验。
* 没有训练 LUNA16、运行完整检测器或测量 L20，不能声称召回率已提升。
