# L20 环境搭建、训练与测试

下面针对 Linux x86_64 + NVIDIA L20 + Bash，假设已安装 Conda，且 `nvidia-smi`
正常显示 GPU。需要支持 CUDA 12.1 的 NVIDIA 驱动。以下为根据官方版本说明与本地
源码整理的建议环境，尚未在用户 L20 上执行完整安装或训练。

## 环境

```bash
nvidia-smi
conda create -n drmnet-l20 python=3.10 -y
conda activate drmnet-l20

# CUDA 编译工具；无需通过该命令安装或更换系统驱动。
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit=12.1.1 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
export MPLBACKEND=Agg
nvcc --version
g++ --version

python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

# 修改成服务器上同步后的 DRMNet 根目录。
export DRMNET_DIR=/data/DRMNet
cd "$DRMNET_DIR"
python -m pip install -r "$DRMNET_DIR/requirements-l20.txt"
python -m pip install causal-conv1d==1.2.0.post2 --no-build-isolation
python -m pip install mamba-ssm==1.2.2 --no-build-isolation
python -m pip check
```

如果没有 `g++`，Ubuntu 管理员可先执行 `sudo apt-get update` 和
`sudo apt-get install -y build-essential`；CUDA 12.1 源码编译使用其支持的
主机编译器（例如 GCC 11/12），不要强行使用超出支持范围的新版 GCC。
如果安装已有匹配 wheel，未必触发源码编译。

这组包优先使用作者发布的匹配 wheel。如果 wheel 下载失败且安装包不包含完整 CUDA
源码，可以改用作者 Git 仓库的相同 tag 安装（需要 git 与 nvcc）：

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.0"
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation \
  "git+https://github.com/state-spaces/mamba.git@v1.2.2"
```

仅在上面的常规安装失败后使用源码安装，并记录最终实际安装版本。
PyTorch 自带的 CUDA 运行库不等于 `nvcc` 编译工具；不要根据 `nvidia-smi` 的
CUDA Version 字段判断系统是否有 nvcc。新开终端后先激活 Conda 环境并恢复上面的
`CUDA_HOME`、`PATH`、`MPLBACKEND` 和项目路径变量。

## 环境检查

```bash
python - <<'PY'
import torch
import mamba_ssm
import causal_conv1d
from torch.utils.data import StackDataset
from utils.pybox import BOX_BACKEND, torch_nms, torch_overlap
from net.sanet import SANet

assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动与 PyTorch 安装'
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Mamba:', mamba_ssm.__version__)
print('causal-conv1d:', causal_conv1d.__version__)
print('Box backend:', BOX_BACKEND)
boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
assert torch.allclose(torch_overlap(boxes, boxes), torch.ones(1, 1))
dets = torch.cat([torch.ones(1, 1), boxes], dim=1)
assert len(torch_nms(dets, 0.1)[0]) == 1
print('Detector imports and box operations OK')
PY
python "$DRMNET_DIR/train.py" --help
python "$DRMNET_DIR/test.py" --help
python -m pytest "$DRMNET_DIR/tests" -q
CUDA_VISIBLE_DEVICES=0 python "$DRMNET_DIR/scripts/smoke_fusion.py" \
  --variant fusion --shape 32 32 32 --batch-size 1
CUDA_VISIBLE_DEVICES=0 python "$DRMNET_DIR/scripts/smoke_fusion.py" \
  --variant fusion --shape 128 128 128 --batch-size 1
```

原来的 `box` 编译扩展缺失时，当前代码自动使用经过几何单元测试的 NumPy CPU 备用
实现。不需要执行 `net/lib` 中使用旧 `torch.utils.ffi` 的编译脚本，也不需要安装名为
`box` 的无关 PyPI 包。备用实现候选数量很大时较慢；原有效编译扩展仍优先使用。
2026-09-14 新增备用实现的 3 个 NumPy 单元测试通过；Torch 包装接口留由上述服务器检查验证。
网络 smoke 只检查特征提取器和 RPN，不代表完整检测器已经通过训练验证。

## 数据要求

```bash
export LUNA_ROOT=/data/LUNA16
export RUN_DIR="$DRMNET_DIR/results/luna16_fold0_fusion"
```

预期示例结构：

```text
/data/LUNA16/
  preprocessed/<seriesuid>_clean.npy
  fold0/train.npy
  fold0/val.npy
  fold0/test.npy
  fold0/train_annos.csv
  fold0/val_annos.csv
  fold0/test_annos.csv
  fold0/test_seriesuids.csv
  evaluation/annotations.csv
  evaluation/annotations_excluded.csv
```

* clean.npy 为 `[1,D,H,W]` 预处理数组。数据读取器将其按 `(image-128)/128` 归一化，
  不应直接把未经处理的 HU 体积或原始 `.mhd/.raw` 当作 clean.npy。
* train/val/test.npy 为病例 ID 列表，与 clean.npy 文件名前缀一致。
* train/val/test_annos.csv 的必要列：`pid,zmin,zmax,ymin,ymax,xmin,xmax`，
  坐标对应预处理体积中的体素框，pid 与列表中的 ID 一致。
* FROC 两个 annotations 文件使用 `seriesuid,coordX,coordY,coordZ,diameter_mm` 格式；
  `test_seriesuids.csv` 每行一个测试病例 ID，不含表头。
* 当前原评估导出路径直接写出预处理坐标，没有自动逆变换到官方世界坐标。
  因此 FROC 标注必须与导出的坐标和尺度一致；如果使用官方原始世界坐标标注，
  需要先补齐预测坐标的 origin/spacing/crop 逆变换，不能直接比较。
* 训练、验证、测试按患者隔离。单纯设置 `--fold-num` 不会自动生成数据划分。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python "$DRMNET_DIR/train.py" \
  --variant fusion \
  --train-set "$LUNA_ROOT/fold0/train.npy" \
  --val-set "$LUNA_ROOT/fold0/val.npy" \
  --train-data-dir "$LUNA_ROOT/preprocessed" \
  --val-data-dir "$LUNA_ROOT/preprocessed" \
  --train-anno "$LUNA_ROOT/fold0/train_annos.csv" \
  --val-anno "$LUNA_ROOT/fold0/val_annos.csv" \
  --batch-size 1 --num-workers 4 \
  --optimizer SGD --init-lr 0.01 \
  --epochs 80 --epoch-rcnn 21 \
  --out-dir "$RUN_DIR"
```

第 1–20 epoch 训练 RPN，第 21 epoch 起加入 RCNN。batch size 1 用于先跑通，确认
显存后再决定是否增加。当前 train.py 使用 FP32；不要给它传 smoke 专用的 `--amp`。
输入训练 patch 仍为 config.py 中的 128³。

消融时将 `--variant` 分别改成 `baseline`、`mamba`、`freqfusion`，每组同时换一个
独立 `--out-dir`，其他训练设置保持一致。

断点续训：完整复用以上训练命令，增加：

```text
--ckpt "$RUN_DIR/model/040.ckpt"
```

这里 `--epochs 80` 表示总共训练到第 80 轮，不是再训练 80 轮。检查点文件名仅为示例，
应选择实际已保存的文件。

## 测试 / FROC

确认 FROC 标注与预测处于相同坐标系后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python "$DRMNET_DIR/test.py" \
  --mode eval \
  --weight "$RUN_DIR/model/080.ckpt" \
  --test-set-name "$LUNA_ROOT/fold0/test.npy" \
  --preprocessed-data-dir "$LUNA_ROOT/preprocessed" \
  --test-anno "$LUNA_ROOT/fold0/test_annos.csv" \
  --froc-annotations "$LUNA_ROOT/evaluation/annotations.csv" \
  --froc-excluded "$LUNA_ROOT/evaluation/annotations_excluded.csv" \
  --froc-seriesuids "$LUNA_ROOT/fold0/test_seriesuids.csv" \
  --out-dir "$RUN_DIR"
```

架构从检查点自动恢复。`080.ckpt` 只是命令示例，正式报告应使用验证集选择的轮次。
评估子目录为 `$RUN_DIR/res/80/FROC/`（其他轮次对应其他数字）。
原汇总 CSV 代码使用字符串拼接输出在 `${RUN_DIR}froc_include_fp.csv` 和
`${RUN_DIR}froc_simple.csv`，即与 RUN_DIR 同级、带运行目录名前缀的文件。
重点比较 RPN/RCNN/ensemble 的七点 CPM，以及 1、2、4 FP/scan 时的召回率。
test.py 按整例预处理 CT 推理；大体积显存占用需实际验证，当前没有滑窗推理入口。

## 版本依据

* [PyTorch 2.1.2 官方安装命令](https://pytorch.org/get-started/previous-versions/#v212)
* [Mamba v1.2.2 官方仓库](https://github.com/state-spaces/mamba/tree/v1.2.2)
* [causal-conv1d 1.2.0.post2](https://pypi.org/project/causal-conv1d/1.2.0.post2/)
* [NVIDIA CUDA 12.1.1 Conda 安装说明](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-installation-guide-linux/index.html#conda-installation)
