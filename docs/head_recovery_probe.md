# 检测头退化检查

第 80 轮 checkpoint 的 RCNN bias-only 前景概率为 0.9235160947，与导出的全部
RCNN 分数 0.9235161 一致；前景回归偏置约 1e-9。参数没有非有限值，也没有全零。
这些证据说明有效特征贡献接近零，但不能只凭静态参数断定 fc2 的 ReLU 全部死亡：
两个类别共有的特征贡献也会在 softmax 中抵消，需要实际激活与梯度确认。

本实验不训练完整模型、不评估测试集、不保存权重。它从训练列表选择两个不同扫描的
阳性裁块，关闭增强；每个裁块只读取一次，因此所有对照共享完全相同的图像和目标。
裁块本身仍由原 BboxReader 随机裁剪，不同次脚本调用不保证相同裁块。

分别用 eval 和 train 模式提取冻结的特征；后者使用 batch 统计，所有 BatchNorm
缓冲区会在下一次提取前恢复，原始检查点不被修改。随后分别从同一组权重构造
ReLU 和 LeakyReLU 检测头，固定输入特征与目标，用 SGD lr=1e-4、无动量／衰减，
梯度范数上限 5，执行 20 步。检测头以外的参数不更新。

RCNN 的受控候选含真实框、沿 z 偏移 0.2 倍框深度的候选和图像角部候选，
经原始 RCNN sampler 标注。其目的是避免仅对真实框自身回归的简单目标，并报告
实际正负例数量；不是实际 RPN 候选召回率，也不用于报告模型检测性能。

```bash
conda activate drmnet-l20
cd /data/home/zpc/jc/DRMNet
python -m unittest discover -s tests -p test_head_activation.py -v
python -m unittest discover -s tests -p test_rpn_classification_stats.py -v

CUDA_VISIBLE_DEVICES=0 python scripts/probe_detector_heads.py \
  --checkpoint /data/home/zpc/jc/DRMNet/results/luna16_fusion/model/080.ckpt \
  --data-dir /data/home/zpc/jc/data/LUNA16/drmnet_preprocessed \
  --train-set /data/home/zpc/jc/data/LUNA16/drmnet_preprocessed/filelists/train.npy \
  --train-anno /data/home/zpc/jc/data/LUNA16/drmnet_preprocessed/filelists/train_annos.csv \
  --samples 2 --steps 20 --lr 0.0001 \
  --out /data/home/zpc/jc/DRMNet/results/luna16_fusion/head_probe.json
```

终端给出 8 组摘要（两个裁块 × 两种特征模式 × 两种激活）。完整 JSON 包含所有中间
层范围、零值比例、特征判别贡献、梯度与损失，以及实际 RCNN 正负样本和回归目标。

判断要结合多个证据：

- ReLU 隐藏输出全零、对应梯度为零，而 LeakyReLU 在相同输入下恢复非零梯度，支持激活截断。
- 仅 eval 模式退化而 train 模式正常，需要进一步排查 BatchNorm 统计与输入分布。
- 两种激活均失败且输入特征也退化，不能只更换检测头，应继续检查骨干／解码器与训练数据。
- 短程损失下降只证明这条固定特征训练路径可更新；不证明实际召回率恢复，更不能替代验证集评估。

`net/heads.py` 默认仍使用 ReLU，旧权重键保持不变。LeakyReLU 仅在显式配置
`head_activation='leaky_relu'` 时启用，脚本在内存中传入该配置。
