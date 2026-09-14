"""Feature pyramid shared by the original detector and fusion ablations."""

import torch
from torch import nn

from net import resnet
from net.fusion_modules import FreqFusion3d, make_mamba

bn_momentum = 0.1

class FeatureNet(nn.Module):
    def __init__(self, config, mamba_factory=None):
        super(FeatureNet, self).__init__()
        factory = mamba_factory or make_mamba
        self.resnet50 = resnet.resnet50(
            multidirection_stages=config.get('multidirection_stages', ()),
            fusion_checkpoint=config.get('fusion_checkpoint', True),
            mamba_factory=factory)

        self.back1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.reduce1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        self.reduce4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))
        #mamba
        self.ln1 = nn.LayerNorm(64)
        self.mamba1 = factory(
                        d_model=64,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )
        self.ln2 = nn.LayerNorm(128)
        self.mamba2 = factory(
                        d_model=128,
                        d_state=16,  
                        d_conv=4,    
                        expand=2,   
                    )

        stages = config.get('freqfusion_stages', ())
        if any(stage not in (2, 3) for stage in stages):
            raise ValueError('freqfusion_stages must contain decoder stages 2 or 3')
        self.freqfusion = nn.ModuleDict({
            str(stage): FreqFusion3d(
                hr_channels=64, lr_channels=64,
                compressed_channels=config.get('freqfusion_compressed_channels', 16),
                kernel_size=config.get('freqfusion_kernel_size', 3),
                use_checkpoint=config.get('fusion_checkpoint', True))
            for stage in stages
        })

    def forward(self, x):
        x1, out1, out2, out3, out4 = self.resnet50(x)
        out4 = self.reduce1(out4)
        rev3 = self.path1(out4)
        B, C = rev3.shape[:2]
        n_tokens = rev3.shape[2:].numel()
        img_dims = rev3.shape[2:]
        rev3_feature_flat = rev3.view(B, C, n_tokens).transpose(-1, -2)
        rev3_feature_flat = self.ln1(rev3_feature_flat)
        rev3 = self.mamba1(rev3_feature_flat)
        rev3 = rev3.transpose(-1, -2).view(B, C, *img_dims)
        
        out3 = self.reduce2(out3)
        if '3' in self.freqfusion:
            out3, rev3 = self.freqfusion['3'](out3, rev3)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        
        rev2 = self.path2(comb3)
        B, C = rev2.shape[:2]
        n_tokens = rev2.shape[2:].numel()
        img_dims = rev2.shape[2:]
        rev2_feature_flat = rev2.view(B, C, n_tokens).transpose(-1, -2)
        rev2_feature_flat = self.ln1(rev2_feature_flat)
        rev2 = self.mamba1(rev2_feature_flat)
        rev2 = rev2.transpose(-1, -2).view(B, C, *img_dims)
        
        out2 = self.reduce3(out2)
        if '2' in self.freqfusion:
            out2, rev2 = self.freqfusion['2'](out2, rev2)
        comb2 = self.back2(torch.cat((rev2, out2), 1))
       
        return [x1, out1, comb2], out1
