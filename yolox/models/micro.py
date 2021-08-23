import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.nn import AdaptiveAvgPool2d

def _init_conv_weight(op, weight_init="kaiming_normal"):
    assert weight_init in [None, "kaiming_normal"]
    if weight_init is None:
        return
    if weight_init == "kaiming_normal":
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(op, "bias") and op.bias is not None:
            nn.init.constant_(op.bias, 0.0)





class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class AvgPool(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 stride=None,
                 **kwargs):
        super(AvgPool, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        return self.avgpool(x)

class AdaAvgPool(nn.Module):
    def __init__(self,
                 output_size = (1, 1),
                 **kwargs):
        super(AdaAvgPool, self).__init__()
        self.adavgpool = torch.nn.AdaptiveAvgPool2d(output_size = output_size)
    
    def forward(self, x):
        return self.adavgpool(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, **kwargs):
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.maxpool(x)


class ConvBlock(nn.Module):
    # conv-bn-act block
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 groups=1,
                 bias=True,
                 bn=True,
                 act='relu',
                 zero_gamma=False):
        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, \
                         kernel_size=kernel_size, stride=stride, \
                         padding=padding, groups=groups, bias=bias)
        _init_conv_weight(conv)
        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

        if zero_gamma:
            nn.init.constant_(self.bn.weight, 0.0)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True) 
        else:
            self.act = None

    def forward(self, x):
        if x.dtype != self.conv.weight.dtype and self.conv.weight.dtype == torch.float16:
            x = x.half()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
        return x * y


class FBNetV2Block(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 stride,
                 expansion,
                 groups=1, 
                 use_se=False,
                 act='relu'): 
    
        super(FBNetV2Block, self).__init__()        

        mid_channel = int(out_channels * expansion)
       
        # 1x1 conv
        self.conv_bn_relu1 = ConvBlock(in_channels=in_channels, \
                             out_channels=mid_channel, bias=False, \
                             kernel_size=1, stride=1, padding=0, act='relu')
        # 3x3 conv
        self.conv_bn_relu2 = ConvBlock(in_channels=mid_channel, \
                             out_channels=mid_channel, \
                             kernel_size=kernel_size, stride=stride, \
                             padding=kernel_size//2, groups=groups, \
                             bias=False, act='relu')
        # se module
        self.se = SEModule(mid_channel) if use_se else None
        # 1x1 conv
        self.res_flag = (in_channels == out_channels) and (stride == 1) 
        self.conv_bn_relu3 = ConvBlock(in_channels=mid_channel, \
                             out_channels=out_channels, bias=False, \
                             kernel_size=1, stride=1, padding=0, act='none')
        if self.res_flag:
            self.downsample = Identity()
        else:
            self.downsample = ConvBlock(in_channels=in_channels, \
                       out_channels=out_channels, bias=False, \
                       kernel_size=1, stride=stride, padding=0, act='none')
        self.final_act = nn.ReLU(inplace=True) 
        
    def forward(self, x):
        res = self.downsample(x)
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        if self.se is not None:
            out = self.se(out) 
        out = self.conv_bn_relu3(out)
        out = self.final_act(out + res)
        return out
