"""Detector heads, independent of proposal/NMS extension imports."""

from torch import nn
from torch.nn import functional as F


def head_activation(cfg, inplace=False):
    """Default preserves old checkpoints; leaky_relu is an opt-in experiment."""
    name = cfg.get('head_activation', 'relu')
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    if name == 'leaky_relu':
        slope = float(cfg.get('head_negative_slope', .1))
        if not 0 < slope < 1:
            raise ValueError('head_negative_slope must be between 0 and 1')
        return nn.LeakyReLU(slope, inplace=inplace)
    raise ValueError('Unknown head_activation: %s' % name)


class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.num_anchors = len(config['anchors'])
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    head_activation(config))
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], self.num_anchors, 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], self.num_anchors, 6)
        

        return logits, deltas

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.activation1 = head_activation(cfg, inplace=True)
        self.activation2 = head_activation(cfg, inplace=True)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas
