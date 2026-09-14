import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from net.fusion_modules import MultiDirectionMamba3d, make_mamba

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size=None,
                 sample_duration=None,
                 shortcut_type='B',
                 num_classes=400,
                 multidirection_stages=(),
                 fusion_checkpoint=True,
                 mamba_factory=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # self.preBlock = nn.Sequential(
        #         nn.Conv3d(1, 64, kernel_size = 3, padding = 1, stride=2),
        #         nn.BatchNorm3d(32),
        #         nn.ReLU(inplace = True),
        #         nn.Conv3d(64, 64, kernel_size = 3, padding = 1),
        #         nn.BatchNorm3d(64),
        #         nn.ReLU(inplace = True))

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 64, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 64, layers[3], shortcut_type, stride=2)
        self.ln = nn.LayerNorm(64)
        factory = mamba_factory or make_mamba
        self.mamba = factory(
                        d_model=64,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if any(stage not in (1, 2, 3, 4) for stage in multidirection_stages):
            raise ValueError('multidirection_stages must contain encoder stages 1..4')
        # Construct after legacy initialization so the original path and its
        # checkpoint keys stay intact, and local block initialization is kept.
        self.multidirection = nn.ModuleDict({
            str(stage): MultiDirectionMamba3d(
                64, use_checkpoint=fusion_checkpoint, mamba_factory=factory)
            for stage in multidirection_stages
        })

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # if (i == 5 and blocks == 6) or (i == 3 and blocks == 6) or (i == 1 and blocks == 6) or \
            #         (i == 3 and blocks == 4) or (i == 1 and blocks == 4):
            #     # layers.append(cgnl.SpatialCGNL(
            #     #     self.inplanes,
            #     #     int(self.inplanes / 2),
            #     #     use_scale=False,
            #     #     groups=4))
            #     layers.append(nn.LayerNorm(self.inplanes))
            #     layers.append(Mamba(
            #             d_model=self.inplanes,
            #             d_state=16,  
            #             d_conv=4,    
            #             expand=2,   
            #         ))

            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _context(self, x, stage):
        if str(stage) in self.multidirection:
            return self.multidirection[str(stage)](x)
        b, c = x.shape[:2]
        shape = x.shape[2:]
        tokens = x.reshape(b, c, -1).transpose(1, 2)
        tokens = self.mamba(self.ln(tokens))
        return tokens.transpose(1, 2).reshape(b, c, *shape)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        out1 = self._context(self.layer1(x), 1)
        out2 = self._context(self.layer2(out1), 2)
        out3 = self._context(self.layer3(out2), 3)
        out4 = self._context(self.layer4(out3), 4)
        return x, out1, out2, out3, out4


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
