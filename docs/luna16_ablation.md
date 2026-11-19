# DRMNet 四组消融

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
