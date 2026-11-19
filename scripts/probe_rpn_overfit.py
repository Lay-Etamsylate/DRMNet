"""Fixed training-crop RPN overfit comparison; writes JSON, never checkpoints.

Both cases start from identical weights and use identical fixed anchors/targets.
The objective is RPN classification + regression only (no RCNN or auxiliary loss).
This is a learnability diagnostic, not validation or FROC evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from dataset.bbox_reader import BboxReader
from net.feature_net import FeatureNet
from net.heads import RpnHead
from net.model_config import checkpoint_model_config
from net.layer.rpn_nms import make_rpn_windows
from net.layer.rpn_target import make_rpn_target
from net.layer.rpn_loss import rpn_loss


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def build_modules(cfg, state):
    feature = FeatureNet(cfg)
    head = RpnHead(cfg, 128)
    if state is not None:
        for prefix, module in [('feature_net.', feature), ('rpn.', head)]:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)
    return feature.cuda(), head.cuda()


def export_crop_views(sample, metadata, path):
    """Orthogonal views in cropped voxel coordinates, without medical rescaling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    volume = sample['image'][0, 0].numpy() * 128 + 128
    positive = (sample['targets'][0].reshape(-1) == 1).nonzero().reshape(-1)
    assignment = int(sample['targets'][1].reshape(-1)[positive[0]])
    truth = np.asarray(metadata['truth_boxes'][assignment])
    anchor = np.asarray(metadata['positive_windows'][0])
    center = truth[:3]
    planes = [(0, 1, 2, 'Axial: x/y'), (1, 0, 2, 'Coronal: x/z'), (2, 0, 1, 'Sagittal: y/z')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    try:
        for col, (fixed, vertical, horizontal, title) in enumerate(planes):
            plane = int(np.clip(np.rint(center[fixed]), 0, volume.shape[fixed] - 1))
            view = np.take(volume, plane, axis=fixed)
            for row in range(2):
                ax = axes[row, col]
                ax.imshow(view, cmap='gray', vmin=0, vmax=255, origin='upper')
                for box, color in [(truth, 'cyan'), (anchor, 'red')]:
                    lo = box[:3] - box[3:] / 2
                    ax.add_patch(Rectangle((lo[horizontal], lo[vertical]), box[horizontal + 3],
                                           box[vertical + 3], fill=False, edgecolor=color, linewidth=1))
                ax.plot(center[horizontal], center[vertical], '+', color='yellow', markersize=8)
                ax.set_title(f'{title}; slice={plane}' + ('; zoom' if row else ''))
                if row:
                    ax.set_xlim(max(-.5, center[horizontal] - 24), min(view.shape[1] - .5, center[horizontal] + 24))
                    ax.set_ylim(min(view.shape[0] - .5, center[vertical] + 24), max(-.5, center[vertical] - 24))
        fig.suptitle(metadata['seriesuid'] + '\nCyan: training GT (includes bbox border); red: positive anchor; yellow: GT center', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def predictions(head, feature, targets, cfg):
    logits, deltas = head(feature)
    logits, deltas = logits.reshape(1, -1, 1), deltas.reshape(1, -1, 6)
    labels, _, weights, boxes, box_weights = targets
    cls, reg, _ = rpn_loss(logits, deltas, labels, weights, boxes, box_weights,
                         cfg, mode='train')
    with torch.no_grad():
        p = logits.sigmoid().reshape(-1)
        positive = p[labels.reshape(-1) == 1]
        negative = p[(labels.reshape(-1) == 0) & (weights.reshape(-1) > 0)]
        k = min(len(negative), int(cfg['num_hard']) * len(positive))
        hard = negative.topk(k).values if k > 0 else negative
        record = dict(rpn_cls=float(cls), rpn_reg=float(reg),
                      positive_probability=float(positive.mean()),
                      sampled_negative_probability=float(negative.mean()),
                      hard_negative_probability=float(hard.mean()),
                      hard_negative_max=float(hard.max()),
                      sampled_negatives_above_positive=int((negative > positive.mean()).sum()),
                      positive_count=len(positive), sampled_negative_count=len(negative),
                      feature_spatial_std=float(feature.flatten(2).std(dim=2, unbiased=False).mean()),
                      feature_zero_fraction=float((feature == 0).float().mean()))
    return cls + reg, record


def grad_norm(module):
    norms = [p.grad.detach().norm().square() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else None


def inspect_local_features(feature, head, sample, cfg):
    """Inspect positive sites, not whole-volume averages; never update weights.

    RPN is pointwise: flattened anchor i uses spatial feature i // A.
    Logit derivatives on the two sides of the decoder's final ReLU expose
    local gating without confusing regression gradients with classification.
    """
    captured = {}

    def capture(module, inputs):
        # Clone because back2's ReLU is in-place. The clone remains on the
        # differentiable path and prevents the original input being overwritten.
        captured['pre_relu'] = inputs[0]
        return (inputs[0].clone(),)

    handle = feature.back2[-1].register_forward_pre_hook(capture)
    try:
        f = feature(sample['image'].cuda())[0][-1]
    finally:
        handle.remove()
    pre = captured['pre_relu']
    logits, _ = head(f)
    logits = logits.reshape(-1)
    labels, _, weights, _, _ = sample['targets']
    positive_ids = (labels.reshape(-1) == 1).nonzero().reshape(-1)
    negative_ids = ((labels.reshape(-1) == 0) & (weights.reshape(-1) > 0)).nonzero().reshape(-1)
    after_grad, before_grad = torch.autograd.grad(logits[positive_ids].sum(), (f, pre))
    # B=1, flattening is z,y,x with anchor index varying fastest.
    anchors = len(cfg['anchors'])
    vectors = f.detach().flatten(2)[0].T
    pre_vectors = pre.detach().flatten(2)[0].T
    after_grad = after_grad.detach().flatten(2)[0].T
    before_grad = before_grad.detach().flatten(2)[0].T
    spatial_shape = tuple(f.shape[2:])
    result = []
    with torch.no_grad():
        zero = torch.zeros((1, f.shape[1], 1, 1, 1), device=f.device, dtype=f.dtype)
        zero_logits = head(zero)[0].reshape(-1)
        for anchor_id_tensor in positive_ids:
            anchor_id = int(anchor_id_tensor)
            site, kind = divmod(anchor_id, anchors)
            same_kind = negative_ids[negative_ids % anchors == kind]
            k = min(len(same_kind), max(1, int(cfg['num_hard'])))
            hard_ids = same_kind[logits[same_kind].topk(k).indices]
            v = vectors[site]
            distances = (vectors[hard_ids // anchors] - v).norm(dim=1)
            item = dict(anchor_flat_index=anchor_id, anchor_kind=kind,
                        feature_site_zyx=[int(x) for x in np.unravel_index(site, spatial_shape)],
                        positive_probability=float(logits[anchor_id].sigmoid()),
                        zero_input_probability=float(zero_logits[kind].sigmoid()),
                        positive_feature_l2=float(v.norm()),
                        positive_feature_zero_fraction=float((v == 0).float().mean()),
                        pre_relu_min=float(pre_vectors[site].min()),
                        pre_relu_max=float(pre_vectors[site].max()),
                        positive_logit_grad_after_relu_l2=float(after_grad[site].norm()),
                        positive_logit_grad_before_relu_l2=float(before_grad[site].norm()),
                        same_kind_sampled_negative_count=len(same_kind),
                        same_kind_hard_negative_probabilities=logits[hard_ids].sigmoid().tolist(),
                        same_kind_hard_negative_feature_distances=distances.tolist(),
                        same_kind_hard_negative_near_identical_count=int((distances < 1e-6).sum()))
            result.append(item)
    return result


def run_case(cfg, state, samples, args, scope, report, save):
    seed(args.seed)
    feature, head = build_modules(cfg, None if args.initialization == 'fresh' else state)
    # train uses per-crop BN statistics in both cases; running buffers are disposable.
    feature.train(args.feature_mode == 'train')
    head.train()
    feature.requires_grad_(scope == 'end_to_end')
    cached = []
    if scope == 'head_only':
        with torch.no_grad():
            cached = [feature(sample['image'].cuda())[0][-1].detach() for sample in samples]
    parameters = list(head.parameters())
    if scope == 'end_to_end':
        parameters += list(feature.parameters())
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=.9, weight_decay=.0001)
    case = dict(scope=scope, initialization=args.initialization, history=[])
    report['cases'].append(case)
    if args.steps == 0 and scope == 'end_to_end':
        case['local_features'] = []
        for index, sample in enumerate(samples):
            item = dict(sample=index, positives=inspect_local_features(feature, head, sample, cfg))
            case['local_features'].append(item)
            print(json.dumps(dict(scope=scope, local_features=item), allow_nan=False), flush=True)
        save()
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            # Measure every fixed crop at the same parameter state, with no update.
            with torch.no_grad():
                for index, sample in enumerate(samples):
                    f = cached[index] if cached else feature(sample['image'].cuda())[0][-1]
                    _, metrics = predictions(head, f, sample['targets'], cfg)
                    item = dict(step=step, sample=index, **metrics)
                    case['history'].append(item)
                    print(json.dumps(dict(scope=scope, **item), allow_nan=False), flush=True)
            save()
        if step == args.steps:
            break
        index = step % len(samples)
        sample = samples[index]
        optimizer.zero_grad(set_to_none=True)
        f = cached[index] if cached else feature(sample['image'].cuda())[0][-1]
        loss, _ = predictions(head, f, sample['targets'], cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(f'{scope}: non-finite loss at step {step}')
        loss.backward()
        if step % args.log_every == 0:
            gradients = dict(step=step, sample=index, head_grad_norm=grad_norm(head),
                             feature_grad_norm=grad_norm(feature))
            case.setdefault('gradients', []).append(gradients)
            print(json.dumps(dict(scope=scope, **gradients), allow_nan=False), flush=True)
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        if step % args.log_every == 0:
            gradients['total_grad_norm_before_clip'] = float(norm)
        optimizer.step()
    case['complete'] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--train-set', type=Path, required=True)
    parser.add_argument('--train-anno', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=5.)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=35202)
    parser.add_argument('--feature-mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--initialization', choices=['checkpoint', 'fresh'], default='checkpoint',
                        help='fresh uses model constructor initialization; checkpoint still defines architecture/data targets')
    parser.add_argument('--scope', choices=['both', 'head_only', 'end_to_end'], default='both')
    parser.add_argument('--export-crops', action='store_true', help='save orthogonal GT/anchor overlays next to the JSON')
    args = parser.parse_args()
    if min(args.samples, args.log_every) < 1 or args.steps < 0:
        parser.error('samples/log-every must be positive; steps must be nonnegative')
    if not all(np.isfinite(v) and v > 0 for v in (args.lr, args.max_grad_norm)):
        parser.error('lr and max-grad-norm must be finite and positive')
    if args.out.suffix.lower() != '.json' or args.out.exists():
        parser.error('--out must be a new .json file; existing files are preserved')
    crop_dir = args.out.with_suffix('')
    if args.export_crops and crop_dir.exists():
        parser.error(f'Crop output directory already exists: {crop_dir}')
    if not torch.cuda.is_available():
        parser.error('CUDA and the installed Mamba backend are required')
    seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint_model_config(config, checkpoint)
    cfg['batch_size'] = 1
    dataset = BboxReader(str(args.data_dir), str(args.train_set), str(args.train_anno), cfg, 0, mode='val')
    feature, head = build_modules(cfg, checkpoint['state_dict'])
    feature.eval()
    samples, metadata, seen = [], [], set()
    for index, box in enumerate(dataset.bboxes):
        scan = int(box[0])
        if scan in seen:
            continue
        # BboxReader normally reseeds NumPy from wall time. Override only its
        # module-local time reference so these diagnostic crops are repeatable.
        fixed_time = 1000.12345 + len(samples) * .01
        with patch('dataset.bbox_reader.time', SimpleNamespace(time=lambda: fixed_time)):
            image, boxes, labels = dataset[index]
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        if not len(boxes):
            raise RuntimeError(f'Annotated crop {index} contains no truth boxes')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 3:] <= 0):
            raise RuntimeError(f'Invalid truth boxes in crop {index}')
        image = image.unsqueeze(0)
        with torch.no_grad():
            windows = make_rpn_windows(feature(image.cuda())[0][-1], cfg)
            targets = make_rpn_target(cfg, 'train', image.cuda(), windows, [boxes], [labels])
        if not (targets[0] == 1).any() or not ((targets[0] == 0) & (targets[2] > 0)).any():
            raise RuntimeError('Diagnostic requires both positive and negative anchors')
        digest = hashlib.sha256(image.numpy().tobytes())
        for tensor in targets:
            digest.update(tensor.cpu().numpy().tobytes())
        metadata.append(dict(seriesuid=str(dataset.filenames[scan]), truth_boxes=boxes.tolist(),
                             image_shape=list(image.shape), image_min=float(image.min()),
                             image_max=float(image.max()), crop_target_sha256=digest.hexdigest(),
                             positive_windows=windows[(targets[0].cpu().numpy().reshape(-1) == 1)].tolist()))
        samples.append(dict(image=image, targets=targets))
        seen.add(scan)
        if len(samples) == args.samples:
            break
    if len(samples) != args.samples:
        parser.error(f'Only {len(samples)} distinct annotated scans available')
    del feature, head
    report = dict(checkpoint=str(args.checkpoint), epoch=checkpoint.get('epoch'),
                  weight_source=('model constructors, no checkpoint weights loaded into experiment cases'
                                 if args.initialization == 'fresh' else str(args.checkpoint)),
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  scope='fixed training crops/targets; RPN cls+reg only; NOT recall; no checkpoints saved',
                  samples=metadata, cases=[], complete=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    if args.export_crops:
        crop_dir.mkdir(parents=True)
        report['crop_views'] = []
        for index, (sample, info) in enumerate(zip(samples, metadata)):
            path = crop_dir / f'sample_{index}_views.png'
            export_crop_views(sample, info, path)
            report['crop_views'].append(str(path))
    save()
    scopes = ('head_only', 'end_to_end') if args.scope == 'both' else (args.scope,)
    for scope in scopes:
        run_case(cfg, checkpoint['state_dict'], samples, args, scope, report, save)
    report['complete'] = True
    save()
    print(f'Report: {args.out}. No checkpoint was written or changed.', flush=True)


if __name__ == '__main__':
    main()
