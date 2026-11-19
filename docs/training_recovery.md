# 已确认的检测头梯度阻断与恢复训练

更新：恢复训练已运行至第 8 轮，RPN 隐藏层梯度恢复，但分类损失仍接近 0.693，
尚未表现出明确分离能力。暂停完整续训，先按 [固定裁剪诊断](rpn_overfit_probe.md)
执行检查。下面保留初始恢复方案；其中第 6 轮启用 RCNN 的原安排已推迟至第 21 轮，
也不应在诊断完成前直接继续。

两个真实训练裁块、train/eval 两种特征模式的对照显示：RPN 的 ReLU 输出约
99.995%–100% 为零，RCNN fc2 的 ReLU 输出全部为零；RCNN fc1/fc2 权重梯度为零。
LeakyReLU 恢复了梯度，但沿用旧权重时 RCNN 回归损失达到约 692–848，梯度范数也很大。
因此本次恢复不沿用已经退化的检测头，也不加载旧优化器动量。

这确认了被测试裁块上的梯度阻断；尚未证明整个特征提取器质量正常，或恢复训练能提高
验证集召回率。后续必须用验证集检查，不能把激活非零直接当作检测成功。

## 恢复策略

- `--init-from` 严格加载完整模型权重，随后 `--reset-heads` 仅重置 RPN 与 RCNN head。
  骨干、解码器、融合模块、ROI 特征提取器及辅助分支保留权重，并在后续训练中正常更新。
- 检测头使用 `--head-activation leaky_relu`，负半轴斜率 0.1。隐藏层采用 Kaiming
  初始化；分类输出权重 std=0.01，回归输出权重 std=0.001，输出偏置归零。
- 新建 SGD 优化器，lr=1e-4，梯度范数上限 5。第 1–5 轮仅 RPN，第 6 轮起加入 RCNN。
- train/valid patch 的 RPN 候选在 NMS 前最多 2000、NMS 后最多 300，以免重置后候选
  过多导致第二阶段难以运行。这两项不改变全 CT eval/test 的候选规则。
- 首次运行设置 `--epochs 80 --stop-after-epoch 3`：实际只运行到第 3 轮，但学习率仍按
  80 轮计划计算。第 3 轮必定保存检查点。恢复结果写到独立目录，原结果保留。

```bash
conda activate drmnet-l20
cd /data/home/zpc/jc/DRMNet
python -m unittest discover -s tests -p test_training_recovery.py -v

export DATA_DIR=/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed
export RECOVERY_DIR=/data/home/zpc/jc/DRMNet/results/luna16_fusion_recovery
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=0 python train.py \
  --init-from /data/home/zpc/jc/DRMNet/results/luna16_fusion/model/080.ckpt \
  --reset-heads --head-activation leaky_relu \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --val-set "$DATA_DIR/filelists/val.npy" \
  --train-data-dir "$DATA_DIR" --val-data-dir "$DATA_DIR" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --val-anno "$DATA_DIR/filelists/val_annos.csv" \
  --batch-size 1 --num-workers 4 \
  --optimizer SGD --init-lr 0.0001 --max-grad-norm 5 \
  --rpn-train-pre-topk 2000 --rpn-train-post-topk 300 \
  --epochs 80 --epoch-rcnn 6 --stop-after-epoch 3 \
  --report-head-health --out-dir "$RECOVERY_DIR"
```

每 200 个训练 batch 输出 `[head_health ...]`：激活零值比例、检测头梯度范数、正／负
anchor 平均概率及裁剪前总梯度范数；启用 RCNN 后还输出实际正负候选数和回归目标范围。
RPN 的 TPR/TNR 使用 logit=0（概率 0.5）作为界限；它是 anchor 分类统计，不是 FROC
结节召回率。不要仅凭 TPR 高判断成功，要同时看 TNR、正负分数区分和验证分类损失。

## 验证之后续训

先确认短程训练中损失有限、隐藏层梯度不再持续为零、正负 anchor 开始区分，再继续：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --ckpt "$RECOVERY_DIR/model/003.ckpt" \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --val-set "$DATA_DIR/filelists/val.npy" \
  --train-data-dir "$DATA_DIR" --val-data-dir "$DATA_DIR" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --val-anno "$DATA_DIR/filelists/val_annos.csv" \
  --batch-size 1 --num-workers 4 \
  --report-head-health --stop-after-epoch 8 --out-dir "$RECOVERY_DIR"
```

该命令恢复新检查点中的激活配置、优化器状态、lr=1e-4、80 轮学习率计划、第 6 轮启用
RCNN 和梯度裁剪，从第 4 轮运行至第 8 轮。续训时不要再传 `--init-from` 或
`--reset-heads`。RCNN 阶段要检查背景候选是否长期为零，回归目标是否长期全零。
停止点不保存为训练设置；删除 `--stop-after-epoch` 后可继续到计划结束。

若短程检查仍然长期无区分，不应消耗完整 80 轮。保留报告后检查上游特征和训练数据，
必要时将低学习率、LeakyReLU、梯度裁剪方案用于全新初始化的独立实验。

新检查点保存 `training_settings` 和含激活配置的 `model_config`，测试时自动恢复。
新模型的检测性能仍应以验证集选轮次、最终在原 89 例测试集按同一协议计算 FROC。
