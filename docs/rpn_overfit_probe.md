# 恢复训练第 8 轮后的固定裁剪诊断

本次日志中第 8 轮训练 RPN 分类损失 0.693768，验证 0.688949；采样正样本平均
分数约 0.49928，随机采样负样本约 0.49553。隐藏层有梯度，但分类仍未明显改善。
先暂停完整续训与 RCNN 阶段，使用 `008.ckpt` 检查少量训练样本的可学习性。

`scripts/probe_rpn_overfit.py` 使用两个不同训练扫描的固定裁剪，不做随机缩放等增强。
沿用现有数据读取、归一化、坐标变换、anchor 匹配和 RPN 损失；正负 anchor 只采样一次。
绕过数据读取器按墙上时间设置随机种子的行为，报告记录裁剪与目标的 SHA256。
两组实验从相同 checkpoint 权重开始，使用相同裁剪与 anchor 标签：

- `head_only`：缓存特征，只更新 RPN。
- `end_to_end`：更新特征网络和 RPN。

两组都只优化 RPN 分类和回归，省略 KL、低秩辅助损失和 RCNN。采用新的 SGD 动量，
lr=1e-4、momentum=0.9、weight_decay=1e-4、梯度裁剪 5；每组共 300 次更新，
交替使用两个裁剪（每个裁剪 150 次）。默认使用 train 模式的单裁剪 BN 统计；
所有 BN 缓冲及权重改动只存在于诊断进程内，不写 checkpoint。

```bash
conda activate drmnet-l20
cd /data/home/zpc/jc/DRMNet
export MPLBACKEND=Agg
export DATA_DIR=/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed
export RECOVERY_DIR=/data/home/zpc/jc/DRMNet/results/luna16_fusion_recovery

python -m unittest discover -s tests -p test_rpn_overfit_probe.py -v
CUDA_VISIBLE_DEVICES=0 python scripts/probe_rpn_overfit.py \
  --checkpoint "$RECOVERY_DIR/model/008.ckpt" \
  --data-dir "$DATA_DIR" \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --samples 2 --steps 300 --lr 0.0001 \
  --out "$RECOVERY_DIR/rpn_overfit_008.json"
```

输出必须是尚不存在的 JSON，重复实验需另选文件名。每 50 步报告所有固定裁剪的损失、
正样本概率、800 个固定采样负 anchor 的平均概率，以及从其中挖掘的困难负样本概率。
还记录特征空间标准差、特征零值比例和梯度范数。梯度为裁剪前的值。

重点比较 step=0 和 step=300 的同一裁剪：分类损失是否明显下降，正样本是否高于
困难负样本，超过正样本分数的负 anchor 数量是否减少。

- 只训练头已能明显拟合：固定特征包含可利用信号，进一步检查原训练的增强、目标随机性
  和辅助损失；此结果不能单独确定是哪一项导致平台期。
- 只有端到端能拟合：更新特征网络对这组裁剪有帮助，继续受控检查优化和原目标函数。
- 两组均无法拟合：此学习率和更新次数下的结果不足以定位根因；接着检查特征退化、
  裁剪与标注对齐、优化步长，并与全新初始化作对照，不应直接断言标签错误。

这些是训练裁剪的学习能力结果，不是泛化或结节检测召回率，不替代验证集 FROC。
固定裁剪也不能检验原训练的随机缩放链路。

## 1e-3 对照后的局部检查

相同裁剪与目标哈希下，提高学习率后，端到端实验在第一个裁剪上将分类损失降至
0.384198、正样本概率升至 0.923637，但第二个裁剪仍为 0.693736、0.508592。
这说明步长影响第一个裁剪的学习，尚不足以解释第二个裁剪的停滞。
已记录的第 250 步总梯度范数为 5.405773，发生梯度裁剪；它不是非有限数错误。

脚本新增 `--steps 0`，只检查原始 `008.ckpt` 的同一批固定裁剪，不更新权重。
注意：此前 300 步实验没有保存权重，所以该模式不能恢复实验第 300 步的网络状态。
在报告 `end_to_end.local_features` 中，对每个正 anchor 检查：

- `positive_feature_zero_fraction` / `positive_feature_l2`：该空间位置的特征，而非全局均值。
- `positive_logit_grad_after_relu_l2` / `positive_logit_grad_before_relu_l2`：正 anchor 的
  logit 对解码器最后一个 ReLU 两侧特征的导数，避免用回归梯度掩盖分类问题。
- `zero_input_probability`：把该位置输入特征置零时，当前 RPN 对该 anchor 类型的分数。
- `same_kind_hard_negative_feature_distances`：同 anchor 类型的困难负样本与正样本特征
  的 L2 距离；小于 1e-6 记为近乎相同。RPN 的卷积均为 1×1×1，相同输入特征和
  anchor 类型会给出相同分类分数。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_rpn_overfit.py \
  --checkpoint "$RECOVERY_DIR/model/008.ckpt" \
  --data-dir "$DATA_DIR" \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --samples 2 --steps 0 \
  --out "$RECOVERY_DIR/rpn_local_features_008.json"
```

若某正样本位置特征全部为零，且 ReLU 后导数非零、之前导数为零，才有证据支持
该位置的局部 ReLU 梯度阻断。仅凭全局特征标准差或这次损失曲线，不能下此结论。
局部检查会临时注册 hook 克隆 ReLU 输入，保持前向数值与梯度链路并保留激活前值；
检查结束移除 hook，不改变正式网络代码或 checkpoint。

## 局部检查结果与全新初始化对照

`008.ckpt` 的两个正 anchor 位置均未出现末端 ReLU 完全阻断：特征范数分别为
5.21113、5.21182，零值比例 45.31%、42.97%，ReLU 前正 logit 的梯度范数为
0.00842、0.00975；被检查的同类型困难负 anchor 与正 anchor 的特征无近乎相同项。
这排除了被测试位置、该检查点、train 模式下的完全阻断假设；不能据此排除其他层、
其他位置或后续优化步骤的问题，也不能用特征距离大小证明可分类。

下一步使用同一模型结构、数据、固定目标、学习率 1e-3 和 300 次更新，比较全新初始化。
`--initialization fresh` 使用 FeatureNet 和 RpnHead 构造器的初始化，实验模型不加载
checkpoint 权重；checkpoint 仍提供结构配置，并保持原固定裁剪与 anchor 采样过程。
每组开始前重新设定随机种子。此处是完整特征网络与 RPN 的初始化对照，不能单独归因于
骨干、融合模块或检测头中的某一个组件。

`--scope end_to_end` 跳过随机特征下的 head-only 实验。`--export-crops` 将固定裁剪
在 GT 中心的三个正交切片及局部放大图写入与 JSON 同名的目录，不覆盖已有目录。
图中青框为带 bbox_border 的训练目标，红框为正 anchor，黄色十字为 GT 中心；
显示的是预处理灰度（不是原始 HU），供核对坐标与标注，不是医学诊断。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_rpn_overfit.py \
  --checkpoint "$RECOVERY_DIR/model/008.ckpt" \
  --data-dir "$DATA_DIR" \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --initialization fresh --scope end_to_end --export-crops \
  --samples 2 --steps 300 --lr 0.001 \
  --out "$RECOVERY_DIR/rpn_overfit_fresh_lr1e3.json"
```

若全新初始化对两个裁剪的分类均明显改善，支持继续评估独立从头训练方案；若只有
第一个改善或两个都不改善，结合导出的坐标叠加图检查数据链路，并评估优化预算。
单个随机种子的 300 步结果不能证明模型结构本身不可学习，不用于选择最终测试集成绩。
