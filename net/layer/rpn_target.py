try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap
import random
from scipy.optimize import linear_sum_assignment
from net.layer.rpn_nms import rpn_encode


def assign_per_nodule(overlap, truth_label, threshold, limit):
    """Reserve distinct overlapping anchors first, then fill each nodule's quota."""
    remaining = np.ones(len(overlap), dtype=bool)
    owners = overlap.argmax(axis=1)
    valid = np.flatnonzero(truth_label > 0)
    candidates = np.flatnonzero((overlap[:, valid] > 0).any(axis=1))
    if candidates.size:
        iou = overlap[np.ix_(candidates, valid)].T
        # One extra covered nodule outweighs every possible gain in total IoU.
        scores = np.where(iou > 0, len(valid) + iou, 0.)
        gt_row, anchor_col = linear_sum_assignment(scores, maximize=True)
        matched = iou[gt_row, anchor_col] > 0
        reserved = candidates[anchor_col[matched]]
        owners[reserved] = valid[gt_row[matched]]
        remaining[reserved] = False

    for gt in valid:
        candidates = np.flatnonzero(remaining & (owners == gt) & (overlap[:, gt] >= threshold))
        ranked = np.argsort(-overlap[candidates, gt], kind='stable')
        remaining[candidates[ranked[:limit - 1]]] = False
    return np.flatnonzero(~remaining), owners


def make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label):
    """
    Generate region proposal targets for one batch

    cfg: dict, for hyper-parameters
    mode: string, which phase/mode is used currently
    input: 5D torch tensor of [batch, channel, z, y, x], original input to the network
    window: list of anchor bounding boxes, [z, y, x, d, h, w]
    truth_box: list of ground truth bounding boxes, [z, y, x, d, h, w]
    truth_label: list of grount truth class label for each object in the correponding truth_box

    return torch tensors
    label: positive or negative (1 or 0) for each anchor box
    label_assign: index of the ground truth box, to which the anchor box is matched to
    label_weight: class weight for each sample, zero means current sample is protected,
                  and won't contribute to loss
    target: bounding box regression terms
    target_weight: weight for each regression term, by default it should all be ones
    """

    num_neg = cfg['num_neg']
    num_window = len(window)
    label = np.zeros((num_window, ), np.float32)
    label_assign = np.zeros((num_window, ), np.int32) - 1
    label_weight = np.zeros((num_window, ), np.float32)
    target = np.zeros((num_window, 6), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)


    num_truth_box = len(truth_box)
    if num_truth_box:

        overlap = np.asarray(torch_overlap(window, truth_box))
        label_assign[:] = overlap.argmax(axis=1)
        best_iou = overlap[np.arange(num_window), label_assign]
        label_weight[best_iou < cfg['rpn_train_bg_thresh_high']] = 1
        threshold = cfg['rpn_train_fg_thresh_low']
        limit = cfg.get('rpn_positives_per_nodule', 0)
        if limit:
            fg_index, owners = assign_per_nodule(overlap, truth_label, threshold, limit)
            label_assign[fg_index] = owners[fg_index]
        else:
            # Historical experiments sample one positive across the whole patch.
            tied, gt = np.where(overlap == overlap.max(axis=0))
            label_assign[tied] = gt
            candidates = np.union1d(np.flatnonzero(best_iou >= threshold), tied)
            label_weight[candidates] = 0
            fg_index = candidates[random.sample(range(len(candidates)), 1)]
        label[fg_index] = 1
        label_weight[fg_index] = 1

        # Prepare regression terms for each positive anchor
        fg_index = np.where(label != 0)[0]
        target_window = window[fg_index]
        target_truth_box = truth_box[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box, cfg['box_reg_weight'])
        target_weight[fg_index] = 1

        # This should by no means be used, left just in case
        invalid_truth_label = np.where(truth_label < 0)[0]
        invalid_index = np.isin(label_assign, invalid_truth_label) & (label!=0)
        label_weight[invalid_index] = 0
        target_weight[invalid_index] = 0

        if mode in ['train']:
            fg_index = np.where( (label_weight!=0) & (label!=0))[0]
            bg_index = np.where( (label_weight!=0) & (label==0))[0]

            # Random sample num_neg negative anchor boxes first
            # This is very strange, but it works well in practice
            # It makes the use of hard negative example mining loss, not 
            # actually hard negative example mining.
            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]

            # Calculate weight for class balance
            num_fg = max(1, len(fg_index))
            num_bg = len(bg_index)
            if num_bg:
                label_weight[bg_index] = float(num_fg)/num_bg

        target_weight[fg_index] = label_weight[fg_index]
    else:
        # if there is no ground truth box in this batch
        label_weight[...] = 1

        if mode in ['train']:
            bg_index = np.where((label_weight!=0) & (label==0))[0]

            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]
            if len(bg_index):
                label_weight[bg_index] = 1.0 / len(bg_index)


    return tuple(torch.as_tensor(value, device=input.device)
                 for value in (label, label_assign, label_weight, target, target_weight))


def make_rpn_target(cfg, mode, inputs, window, truth_boxes, truth_labels):
    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(inputs)
    for b in range(batch_size):
        input = inputs[b]
        truth_box   = np.array(list(truth_boxes[b]), dtype=np.float64)
        truth_label   = np.array(list(truth_labels[b]), dtype=np.float64)

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1, -1))
        rpn_label_assigns.append(rpn_label_assign.view(1, -1))
        rpn_label_weights.append(rpn_label_weight.view(1, -1))
        rpn_targets.append(rpn_target.view(1, -1, 6))
        rpn_targets_weights.append(rpn_targets_weight.view(1, -1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_assigns   = torch.cat(rpn_label_assigns, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights
