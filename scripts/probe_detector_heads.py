"""Controlled head-only experiment on real training crops; never saves weights.

Compares original ReLU and opt-in LeakyReLU on identical frozen features and
targets, including GT/jitter/background ROIs. These losses are NOT test recall.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from dataset.bbox_reader import BboxReader
from net.heads import RpnHead, RcnnHead
from net.model_config import checkpoint_model_config
from net.sanet import SANet
from net.layer.rpn_nms import make_rpn_windows
from net.layer.rpn_target import make_rpn_target
from net.layer.rcnn_target import make_rcnn_target
from net.layer.rpn_loss import rpn_loss
from net.layer.rcnn_loss import rcnn_loss
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes


def stats(x):
    a = x.detach().float()
    finite = torch.isfinite(a)
    good = a[finite]
    return dict(shape=list(a.shape), nonfinite=int((~finite).sum()),
                min=float(good.min()) if good.numel() else None,
                max=float(good.max()) if good.numel() else None,
                std=float(good.std(unbiased=False)) if good.numel() else None,
                mean=float(good.mean()) if good.numel() else None,
                zero_fraction=float((a == 0).float().mean()) if a.numel() else None,
                nonpositive_fraction=float((a <= 0).float().mean()) if a.numel() else None)


def controlled_proposals(boxes, shape):
    candidates = []
    for box in boxes:
        shifted = box.copy()
        shifted[0] += .2 * shifted[3]
        shifted[:3] = np.clip(shifted[:3], 0, np.asarray(shape) - 1)
        candidates.append([0., 1., *shifted])
    from itertools import product
    for center in product(*[(16., float(n - 16)) for n in shape]):
        candidates.append([0., 1., *center, 12., 12., 12.])
    return torch.as_tensor(np.asarray(candidates, dtype=np.float32), device='cuda')


def run_case(cfg, state, features, roi, rpn_targets, rcnn_labels, rcnn_targets,
             activation, steps, lr):
    case_cfg = dict(cfg, head_activation=activation)
    rpn = RpnHead(case_cfg, 128).cuda()
    rcnn = RcnnHead(case_cfg, 64).cuda()
    rpn.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith('rpn.')}, strict=True)
    rcnn.load_state_dict({k[len('rcnn_head.'):]: v for k, v in state.items() if k.startswith('rcnn_head.')}, strict=True)
    rpn.train()
    rcnn.train()
    parameters = list(rpn.parameters()) + list(rcnn.parameters())
    optimizer = torch.optim.SGD(parameters, lr=lr)
    captured = {}
    handles = []
    for name, module in [('rpn_before_activation', rpn.conv[0]), ('rpn_after_activation', rpn.conv[1]),
                         ('rcnn_fc1_before_activation', rcnn.fc1), ('rcnn_fc1_after_activation', rcnn.activation1),
                         ('rcnn_fc2_before_activation', rcnn.fc2), ('rcnn_fc2_after_activation', rcnn.activation2)]:
        def hook(module, inputs, output, key=name):
            captured[key] = stats(output)
        handles.append(module.register_forward_hook(hook))
    history = []
    first = {}
    try:
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits, deltas = rpn(features)
            logits = logits.reshape(1, -1, 1)
            deltas = deltas.reshape(1, -1, 6)
            rc_logits, rc_deltas = rcnn(roi)
            labels, assigns, weights, targets, target_weights = rpn_targets
            cls, reg, _ = rpn_loss(logits, deltas, labels, weights, targets, target_weights, cfg, mode='train')
            rc_cls, rc_reg, _ = rcnn_loss(rc_logits, rc_deltas, rcnn_labels, rcnn_targets)
            loss = cls + reg + rc_cls + rc_reg
            if not torch.isfinite(loss):
                raise RuntimeError(f'{activation}: non-finite probe loss at step {step}')
            if step in (0, steps):
                history.append(dict(step=step, rpn_cls=float(cls.detach()), rpn_reg=float(reg.detach()),
                                    rcnn_cls=float(rc_cls.detach()), rcnn_reg=float(rc_reg.detach())))
            if step == 0:
                first['activations'] = dict(captured)
                probabilities = logits.detach().sigmoid().reshape(-1)
                first['rpn_positive_probabilities'] = stats(probabilities[labels.reshape(-1) > 0])
                first['rpn_negative_probabilities'] = stats(probabilities[(labels.reshape(-1) == 0) & (weights.reshape(-1) > 0)])
                first['rcnn_foreground_probabilities'] = stats(rc_logits.detach().softmax(-1)[:, 1])
                hidden = rcnn.activation2(rcnn.fc2(rcnn.activation1(rcnn.fc1(roi.flatten(1))))).detach()
                projection = torch.nn.functional.linear(hidden, rcnn.logit.weight.detach(), bias=None)
                first['rcnn_foreground_minus_background_feature_contribution'] = stats(projection[:, 1] - projection[:, 0])
                # Only the first forward needs instrumentation.
                for handle in handles:
                    handle.remove()
            if step < steps or step == 0:
                loss.backward()
                if step == 0:
                    first['gradient_norms'] = {}
                    for prefix, module in [('rpn', rpn), ('rcnn', rcnn)]:
                        for name, parameter in module.named_parameters():
                            g = parameter.grad
                            first['gradient_norms'][prefix + '.' + name] = float(g.norm()) if g is not None and torch.isfinite(g).all() else None
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                if step == 0:
                    first['gradient_norm_before_clip'] = float(norm)
            if step < steps:
                optimizer.step()
    finally:
        for handle in handles:
            handle.remove()
    return dict(activation=activation, initial=first, loss_history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--train-set', type=Path, required=True)
    parser.add_argument('--train-anno', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU with the installed Mamba backend is required')
    if args.samples < 1 or args.steps < 0 or args.lr <= 0:
        parser.error('samples/lr must be positive and steps nonnegative')
    random.seed(35202)
    np.random.seed(35202)
    torch.manual_seed(35202)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint_model_config(config, checkpoint)
    cfg['batch_size'] = 1
    dataset = BboxReader(str(args.data_dir), str(args.train_set), str(args.train_anno), cfg, 0, mode='val')
    model = SANet(cfg).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.requires_grad_(False)
    buffers = {name: tensor.detach().clone() for name, tensor in model.feature_net.named_buffers()}
    selected, seen = [], set()
    for index, box in enumerate(dataset.bboxes):
        scan_index = int(box[0])
        if scan_index not in seen:
            selected.append(index)
            seen.add(scan_index)
        if len(selected) >= args.samples:
            break
    if not selected:
        parser.error('No annotated training crops found')
    report = dict(checkpoint=str(args.checkpoint), epoch=checkpoint.get('epoch'),
                  scope='fixed training crops; frozen backbone/ROI extractor; head-only updates; NOT recall',
                  lr=args.lr, steps=args.steps, samples=[])
    for index in selected:
        image, boxes, labels = dataset[index]
        if not len(boxes):
            raise RuntimeError('Selected crop has no ground truth; inspect preprocessing/cropping')
        x = image.unsqueeze(0).cuda()
        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        uid = str(dataset.filenames[int(dataset.bboxes[index][0])])
        sample = dict(seriesuid=uid, image=stats(x), truth_boxes=boxes.tolist(), modes=[])
        for mode in ('eval', 'train'):
            for name, tensor in model.feature_net.named_buffers():
                tensor.copy_(buffers[name])
            model.feature_net.train(mode == 'train')
            print(f'Probe {uid}: feature mode={mode}', flush=True)
            random.seed(35202)
            np.random.seed(35202)
            torch.manual_seed(35202)
            with torch.no_grad():
                features, _ = model.feature_net(x)
                windows = make_rpn_windows(features[-1], cfg)
                rpn_targets = make_rpn_target(cfg, 'train', x, windows, [boxes], [labels])
                proposals = controlled_proposals(boxes, x.shape[2:])
                proposals, rc_labels, _, rc_targets = make_rcnn_target(cfg, 'train', x, proposals, [boxes], [labels])
                corners = proposals[:, [0, 2, 3, 4, 5, 6, 7]].cpu().numpy().copy()
                corners[:, 1:] = center_box_to_coord_box(corners[:, 1:])
                corners = corners.astype(np.int64)
                corners[:, 1:] = clip_boxes(ext2factor(corners[:, 1:], 4), x.shape[2:])
                roi = model.rcnn_crop([f.unsqueeze(0) for f in features], x, torch.from_numpy(corners).cuda()).detach()
            item = dict(feature_mode=mode, pyramid=stats(features[-1]), roi=stats(roi),
                        rcnn_positives=int((rc_labels > 0).sum()), rcnn_negatives=int((rc_labels == 0).sum()),
                        rcnn_targets=stats(rc_targets), cases=[])
            for activation in ('relu', 'leaky_relu'):
                item['cases'].append(run_case(cfg, checkpoint['state_dict'], features[-1].detach(), roi,
                                               rpn_targets, rc_labels, rc_targets, activation, args.steps, args.lr))
                case = item['cases'][-1]
                taps = case['initial']['activations']
                gradients = case['initial']['gradient_norms']
                print(json.dumps(dict(activation=activation,
                    hidden_zero_fraction={name: taps[name]['zero_fraction'] for name in
                        ('rpn_after_activation', 'rcnn_fc1_after_activation', 'rcnn_fc2_after_activation')},
                    gradient_norms={name: gradients[name] for name in
                        ('rpn.conv.0.weight', 'rcnn.fc1.weight', 'rcnn.fc2.weight')},
                    losses=case['loss_history'])), flush=True)
            sample['modes'].append(item)
            del features, roi, rpn_targets, proposals, rc_labels, rc_targets
        report['samples'].append(sample)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Probe report: {args.out}. No checkpoint was written or changed.')


if __name__ == '__main__':
    main()
