# 固定裁剪实验之后的从头训练

同一批固定裁剪和目标、相同 lr=1e-3 / 300 步的端到端对照中，全新初始化将两个正
anchor 的分数升至 0.982988 / 0.984425；旧 008.ckpt 起点为 0.923637 / 0.508592。
第二个裁剪的分类损失从旧权重实验的 0.693736 降为 0.420911。第一例的分类损失
为 0.406911，略高于旧权重实验的 0.384198，因此不能声称所有指标均改善。
全新初始化实验中，困难负样本均分仍为 0.549092 / 0.562184，高于 0.5；这是固定训练
裁剪上的排序改善，不代表检测召回率或低假阳性率已达标。只比较了一种随机初始化。

两张三视图的 GT 中心与对应亮点位置一致，未见这两个无缩放裁剪有明显轴顺序或平移
错误；这不验证整个数据集或缩放增强。青色训练框包含 bbox_border，范围大于亮点正常。

下一步在新的输出目录从头训练融合模型。维持 LeakyReLU 检测头、SGD lr=1e-3、
momentum=0.9、weight_decay=1e-4、梯度裁剪 5。固定裁剪实验中记录的裁剪前总梯度
范数为 31.09–396.40，不能去掉裁剪。

新增开关均存入 model_config，续训、测试自动恢复：

- `--auxiliary-losses off`：不执行 mu/logvar、随机采样、SVD 或 KL 分支，只优化检测
  损失。默认不传保留原行为和参数键，不是把已计算出的异常损失乘零。
- `--scale-augmentation off`：暂时关闭缩放，保持与已验证裁剪链路一致；随机裁剪仍
  保留。原缩放分支尚未完成独立验证，不据此断言它是旧训练失败的根因。

训练数据仍为 subsets2–9，验证为 subset1，测试 subset0 不参与参数选择。
第一次只到第 3 轮，RCNN 保持关闭。这里验证的是全训练集随机裁剪下的学习能力，
不是再做两个固定样本的实验。

```bash
conda activate drmnet-l20
cd /data/home/zpc/jc/DRMNet
export MPLBACKEND=Agg
export DATA_DIR=/data/home/zpc/jc/data/LUNA16/drmnet_preprocessed
export FRESH_DIR=/data/home/zpc/jc/DRMNet/results/luna16_fusion_fresh_det

python -m unittest discover -s tests -p test_detector_only_training.py -v
CUDA_VISIBLE_DEVICES=0 python train.py \
  --variant fusion --head-activation leaky_relu \
  --auxiliary-losses off --scale-augmentation off \
  --train-set "$DATA_DIR/filelists/train.npy" \
  --val-set "$DATA_DIR/filelists/val.npy" \
  --train-data-dir "$DATA_DIR" --val-data-dir "$DATA_DIR" \
  --train-anno "$DATA_DIR/filelists/train_annos.csv" \
  --val-anno "$DATA_DIR/filelists/val_annos.csv" \
  --batch-size 1 --num-workers 4 \
  --optimizer SGD --init-lr 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --max-grad-norm 5 \
  --rpn-train-pre-topk 2000 --rpn-train-post-topk 300 \
  --epochs 80 --epoch-rcnn 21 --stop-after-epoch 3 \
  --report-head-health --out-dir "$FRESH_DIR"
```

不要加入 `--ckpt`、`--init-from` 或 `--reset-heads`。使用独立目录；新训练遇到已有
checkpoint 的目录会报错以免覆盖。代码默认 initial_checkpoint 为 None；启动日志应
显示 `start_epoch 1`、`use_rcnn: False`、`disable_auxiliary_losses: True`、`scale: False`。

在新日志中同时检查正样本与困难负样本均分、训练/验证分类损失、回归损失与梯度范数，
不以 anchor TPR 单独判断成功。训练健康日志新增困难负样本均分和最大分数。
确认全数据短程训练有进展后再选择是否延长 RPN 阶段，RCNN 和辅助损失另作阶段性验证。
新辅助开关需要同步 `net/sanet.py` 到训练与测试环境，否则旧代码会忽略该开关。
