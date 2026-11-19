# DRMNet 四组消融

## 跨尺度融合实验

旧 `fusion` 在解码器先做转置卷积和单向 Mamba，再对已经同分辨率的两路特征做滤波。新版本 `fusion_cross_scale` 保留第 3、4 级多方向编码器，将第 2 级的上采样交给频率融合模块。每个步长 4 的输出体素根据浅层特征和插值后的深层特征预测权重，从步长 8 的原始网格重组 3×3×3 邻域。门控控制重组结果与三线性插值的混合比例，浅层分支保留高通残差。这个设计让同一个粗网格体素的八个子位置可以选取不同邻域权重，检测头仍在步长 4 上工作。

RPN 的旧采样器每个裁块只随机保留一个正锚框。在含两个结节、每个结节各有两个合格锚框的合成裁块上，旧设置保留一个；设置 `--rpn-positives-per-nodule 3` 后两个结节各保留两个。新设置优先给尽可能多的结节分配不同的重叠锚框，其余名额按 IoU 填充。每个结节的首个匹配可低于前景阈值，后续匹配需达到阈值。该上限写入 checkpoint，训练、续训和裁块验证使用同一设置。

### 在 subset1 上分开检验两项改动

以下 Bash 命令在训练服务器的 DRMNet 根目录执行。三个实验均使用 subset2–9 的 710 例训练，subset1 的 89 例做整例验证。运行器检查病例数、文件清单重复和划分交集；若提供原始 subset 目录，还会核对扫描 UID。只保留预处理文件时，病例数与去重检查无法识别等数量的子集互换。

```bash
conda activate drmnet-l20
export LUNA_ROOT=/data/home/zpc/jc/data/LUNA16
export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_fusion.py --variant fusion_cross_scale --shape 32 32 32

run_validation_experiment() {
  local variant="$1" quota="$2" out="$3"
  python scripts/run_luna16_ablation.py train-val --luna-root "$LUNA_ROOT" \
    --variant "$variant" --out-dir "$out" --epochs 80 \
    --auxiliary-losses off --scale-augmentation off --rpn-positives-per-nodule "$quota"
}

run_validation_experiment fusion             0 results/luna16_original_targets
run_validation_experiment fusion             3 results/luna16_per_nodule
run_validation_experiment fusion_cross_scale 3 results/luna16_per_nodule
```

第一、二组的差异估计正样本分配的作用，第二、三组的差异估计跨尺度解码器的作用。三个实验使用相同训练预算、种子、检测头和学习率安排。若已经有与第一组设置相同的旧权重，可以直接执行 `val` 复用它，运行器会从 `--out-dir/<variant>/model/080.ckpt` 加载。此处选择关闭辅助损失是为了复用仓库之前的纯检测训练设置。如果需要对照启用了辅助损失的历史实验，应让三个组使用相同开关。

主要比较各输出目录的 `<variant>_val_080_strict.csv` 中 `head=ensemble` 的 `cpm_raw` 和七个 FP/scan 点的灵敏度。RPN 与 RCNN 行可帮助判断差距出现在候选生成还是候选重评分。案例固定第 80 轮；如要选轮次，可先为每组约定相同的候选轮次，再用 `val --epochs N` 按 subset1 的 ensemble CPM 选择。架构、权重、检测头和推理设置固定后，将同一条 `val` 命令的阶段改为 `test`，对 subset0 的 89 例做最终评估。

### 与独立 FreqFusion 和 MedSegMamba 比较

仓库内的 `freqfusion` 和 `mamba` 是 DRMNet 的单模块消融。相邻两个项目分别使用 2.5D Faster R-CNN 和 3D 分割候选提取，三方自带评估器对边界命中、FROC 取点的处理有差异。下面复用 FreqFusion 的评估器，在同一份 subset1 清单、参考标注、排除标注和每例最多 100 个候选的条件下重算三份世界坐标 CSV。请先确认三方推理日志均覆盖全部 89 例，包括零检出扫描。路径采用各项目默认输出目录。

```bash
DRM_VAL=results/luna16_per_nodule/fusion_cross_scale/val_predictions/res/80/FROC_world_strict
score_val() {
  python ../FreqFusion/tools/luna16/evaluate.py --split val \
    --prepared "$LUNA_ROOT/freqfusion_2p5d" --max-candidates 100 \
    --predictions "$1" --output-dir "results/luna16_common_val/$2"
}
score_val "$DRM_VAL/submission_ensemble.csv" fusion_cross_scale
score_val ../FreqFusion/work_dirs/luna16_freqfusion/val_candidates.csv freqfusion
score_val ../MedSegMamba/runs/luna16/val/predictions.csv medsegmamba
```

在三个 `metrics.json` 中比较 `cpm` 和 `sensitivity_at_fp_per_scan`。如果独立项目已调整输出路径或排除标注位置，请使用实际路径。新结构的收益需要真实 CT 验证，本地合成裁块只验证采样、重组和梯度链路。

## 原有四组对照

| 实验 | `--variant` | 多方向模块 | 高低通融合 |
| --- | --- | --- | --- |
| 基线 DRMNet | `baseline` | 无新增模块 | 无 |
| 仅多方向改进 | `mamba` | 编码器第 3、4 级 | 无 |
| 仅融合改进 | `freqfusion` | 无新增模块 | 解码器第 2 级 |
| 完整方法 | `fusion` | 编码器第 3、4 级 | 解码器第 2 级 |

四种结构共用 [模型配置](../net/model_config.py)，由 [编码器](../net/resnet.py) 和
[特征金字塔](../net/feature_net.py) 接入 [三维模块](../net/fusion_modules.py)。

## 数据和环境

在安装好 PyTorch、CUDA Mamba 和项目依赖的 Linux 环境中运行，见
[L20 环境配置](l20_setup.md)。命令沿用 LUNA16 的固定划分，subsets 2–9 训练，
subset 1 验证，subset 0 测试。

```text
LUNA16/
  annotations.csv
  evaluationScript/.../annotations_excluded.csv
  drmnet_preprocessed/
    <seriesuid>_clean.npy
    <seriesuid>_meta.npz
    filelists/
      train.npy, val.npy, test.npy
      train_annos.csv, val_annos.csv, test_annos.csv
```

如有不同的预处理目录，传 `--data-dir /实际路径`。若官方排除标注放在其他位置，
传 `--excluded /实际路径/annotations_excluded.csv`。评估用元数据将裁剪体素坐标还原为世界坐标
（毫米），并核对参考结节和完整病例清单。

## 运行

在训练服务器的仓库根目录执行，按实际情况修改数据路径。

```bash
conda activate drmnet-l20
export CUDA_VISIBLE_DEVICES=0
export LUNA_ROOT=/data/home/zpc/jc/data/LUNA16
export RUN_DIR="$PWD/results/luna16_ablation"

# 训练并测试全部四组
python scripts/run_luna16_ablation.py all --luna-root "$LUNA_ROOT" --out-dir "$RUN_DIR"
```

可以分开运行训练和测试。测试会读取每组在同一轮次保存的检查点。

```bash
python scripts/run_luna16_ablation.py train --luna-root "$LUNA_ROOT" --out-dir "$RUN_DIR"
python scripts/run_luna16_ablation.py test --luna-root "$LUNA_ROOT" --out-dir "$RUN_DIR"
```

只测试 `freqfusion` 时给测试命令加 `--variant freqfusion`。
评估中断后可用该参数分别完成尚未生成 `summary.json` 的组。
给任一命令加 `--dry-run` 可打印实际的 `train.py`、`test.py` 和 FROC 命令。
训练中断后，可参考打印的对应阶段命令，用 `train.py --ckpt` 和该组最近保存的检查点续训。

默认每组从头训练 80 轮，以第 80 轮为统一测试点。前 20 轮训练 RPN，第 21 轮启用
RCNN。同组两阶段分别使用初始学习率 `1e-3` 和 `1e-4`，并沿用仓库的学习率衰减。
检测头使用 LeakyReLU，启用 DRMNet 辅助损失，关闭缩放增强，保留随机裁剪。
种子默认 `35202`，四组共享每轮的数据顺序和裁剪，验证裁剪固定。CUDA 算子仍可能产生数值差异。
如果实验沿用之前的纯检测训练设置，可在所有组统一指定 `--auxiliary-losses off`。

## 测试输出

- `$RUN_DIR/<variant>/model/080.ckpt` 是对应组的最终权重。
- `$RUN_DIR/<variant>/test_predictions/res/80/FROC_world_strict/` 包含每组的 FROC 曲线、
  `metrics.txt` 和 `summary.json`。
- `$RUN_DIR/ablation_test_080_strict.csv` 在四组评估结束后生成，包含 RPN、RCNN、ensemble
  的七个固定 FP/scan 点召回率、原始 CPM 和 bootstrap CPM。对照实验表可取四条
  `head=ensemble` 的记录。

默认采用严格半径匹配。需要按项目历史容差评估时，四组统一传 `--matching-rule legacy`，
结果写入带 `legacy` 的评估目录和汇总文件。

## 检查命令

```bash
python -m pytest -q tests/test_fusion.py tests/test_luna_eval.py tests/test_luna16_ablation.py

(
  set -e
  for variant in baseline mamba freqfusion fusion; do
    python scripts/smoke_fusion.py --variant "$variant" --shape 32 32 32 --batch-size 1
  done
)
```

CUDA 冒烟测试覆盖真实 Mamba、特征金字塔、RPN 和前反向梯度。整例检测与召回率由
上面的 `test` 命令计算。
