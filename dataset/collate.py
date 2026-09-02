import torch
import numpy as np

def train_collate(batch):
    """Collate samples from one dataset (for example, LUNA16 only)."""
    inputs = torch.stack([sample[0] for sample in batch], 0)
    bboxes = [sample[1] for sample in batch]
    labels = [sample[2] for sample in batch]

    return [inputs, bboxes, labels]


def domain_train_collate(batch):
    """Flatten samples returned by ``StackDataset`` for multi-domain training."""
    inputs = []
    bboxes = []
    labels = []
    for domain_samples in batch:
        for sample in domain_samples:
            inputs.append(sample[0])
            bboxes.append(sample[1])
            labels.append(sample[2])
    return [torch.stack(inputs, 0), bboxes, labels]


def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]

    return [inputs, bboxes, labels]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
