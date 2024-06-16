# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from ultralytics.utils import LOGGER

__all__ = (
    "Conv",
    "Conv_Max_Pooling_Dropout_Attn",
    "Conv2",
    "Nothing",
    "GhostConv_Attn",
    "Conv_Down_Up",
    "GhostConv_Without_BN_Act",
    "GhostConv_Attn_Avg_Pool",
    "LightConv",
    "CBAM_Conv_Avg_Pooling",
    "Conv_sliceSamp_Attn",
    "Conv_Spatial_Attn",
    "Conv_Prune",
    "CBAM_Conv_Fractional_Max_Pooling",
    "Conv_S3Pool_Attn",
    "Conv_SP",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Conv_Dropout",
    "Focus",
    "GhostConv",
    "Conv_Max_Pooling_Dropout",
    "Conv_Fractional_Max_Pooling",
    "Conv_Fractional_Max_Pooling_Attn",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "CBAM_Module",
    "Concat",
    "Concat_Feature_Map",
    "RepConv",
    "Conv_Avg_Pooling",
    "Avg_Pooling_Conv",
    "Conv_3",
    "CBAM_Conv_Avg_Pooling",
    "Conv_Avg_Pooling_Attn",
    "Conv_Mix_Pooling_Dropout_Attn",
    "Conv_Attn",
    "Conv_Avg_Pooling_Attn_Dropout",
    "CBAM_Conv",
    "Conv_Avg_Pooling_Spatial_Attn",
    "Conv_Avg_Pooling_Attnv2",
    "DS_Conv_Attn",
    "DS_Conv",
    "GhostConv_Modification"
    "ChannelAttention_Pool",
    "Conv_DownSampleAttn",
    "sliceSamp_Conv",
    "Conv_Weighted_Pooling"
)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class sliceSamp(nn.Module):
    default_act = nn.GELU()
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False, act=True):
        super(sliceSamp, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        h,w = x.size()[2:]
        x_slice = []
        x_slice.append(x[:,:h//2,:w//2])
        x_slice.append(x[:,:h//2,w//2:w])
        x_slice.append(x[:,h//2:,:w//2])
        x_slice.append(x[:,h//2:,w//2:w])
        out = self.depthwise(torch.cat(x_slice, 1))
        out = self.bn(out)
        out = self.act(out)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        return out
    
class Conv_sliceSamp_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.slice_samp = sliceSamp(c2, c2, kernel_size = 3, padding = 1, bias=False, act=True)  # GAP layer
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.ca(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.slice_samp(x)
        x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.ca(x)
        x = self.act(self.conv(x))
        x = self.slice_samp(x)
        x = self.sa(x)
        return x
    
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class sliceSamp_Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1 * 2, c2 * 2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        h,w = x.size()[2:]
        # Slice the input tensor into 4 parts
        x_slice = []
        x_slice.append(x[:,:h//2,:w//2])
        x_slice.append(x[:,:h//2,w//2:w])
        x_slice.append(x[:,h//2:,:w//2])
        x_slice.append(x[:,h//2:,w//2:w])
        
        # Concatenate along the channel dimension
        x_cat = torch.cat(x_slice, 1)
        
        # Apply convolution, batch normalization, and activation
        out = self.conv(x_cat)
        
        # Slice the output tensor back into 4 parts
        # out_h, out_w = out.shape[1:]
        out_slices = torch.chunk(out, 4, dim=1)
        
        # Combine the slices to form the original spatial dimensions
        out1 = torch.cat([out_slices[0], out_slices[2]], dim=1)
        out2 = torch.cat([out_slices[1], out_slices[3]], dim=1)
        out_combined = torch.cat([out1, out2], dim=2)
        out = self.bn(out_combined)
        out = self.act(out)
        
        return out    
class Conv_Down_Up(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.a = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.u = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.u(self.act(self.bn(self.conv(self.a(x)))))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.u(self.act(self.conv(self.a(x))))

class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 9, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostConv_Modification(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, k1=7, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k1, s, None, math.gcd(c1,c_), act=act)
        self.cv2 = Conv(c_, c_, k, 1, None, g, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
    
class GhostConv_Attn_Avg_Pool(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 1, 1, None, c_, act=act)
        self.ca2 = ChannelAttention(c_)
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()
        self.avg_pool = nn.AvgPool2d(3, stride=2)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        x = self.ca(x)
        y = self.cv1(x)
        z = torch.cat((y, self.ca2(self.cv2(y))), 1)
        z = self.avg_pool(z)
        return self.sa(z)
    
class GhostConv_Attn(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 1, 1, None, c_, act=act)
        self.ca2 = ChannelAttention(c_)
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        x = self.ca(x)
        y = self.cv1(x)
        z = torch.cat((y, self.ca2(self.cv2(y))), 1)
        return self.sa(z)
    
class GhostConv_Without_BN_Act(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()
    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        x = self.ca(x)
        y = self.cv1(x)
        z = torch.cat((y, self.sa(y)), 1)
        return z

class Conv_Prune(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = prune.ln_structured(nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=1, dilation=d, bias=False), name="weight", amount=0.2, dim=1, n=float('-inf'))
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class Conv_3(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv2 = nn.Conv2d(c2, c2, k, s)
        self.conv3 = nn.Conv2d(c2, c2, k, s)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv3(self.conv2(self.conv(x)))))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv3(self.conv2(self.conv(x))))
        return x

    
class Conv_Max_Pooling_Dropout(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.drop = nn.Dropout(0.2)
        self.max_pool = nn.MaxPool2d(3, stride=2)  # GAP layer
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.drop(x)
        x = self.max_pool(x)
        x = self.conv(x)
        x = self.act(self.bn(x))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.drop(x)
        x = self.max_pool(x)
        x = self.conv(x)
        x = self.act(x)
        return x
    
class Conv_Weighted_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.drop = nn.Dropout(0.5)
        self.sum_pool = nn.AvgPool2d(3, stride=2, divisor_override=1) # GAP layer
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.drop(x)
        x = self.sum_pool(x)
        x = self.conv(x)
        x = self.act(self.bn(x))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.drop(x)
        x = self.sum_pool(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class Conv_Max_Pooling_Dropout_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.max_pool = nn.MaxPool2d(3, stride=2)  # GAP layer
        self.sa = SpatialAttention()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(self.max_pool(self.sa(x)))))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(self.max_pool(self.sa(x))))
        return x
    
class Avg_Pooling_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)  # GAP layer

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.avg_pool(x)
        x = self.act(self.bn(self.conv(x)))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.avg_pool(x)
        x = self.act(self.conv(x))
        return x
    
class Conv_Avg_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(2, stride=2)  # GAP layer

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(self.avg_pool(x))))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.avg_pool(x)
        return x
    

class mixedPool(nn.Module):
    def __init__(self,kernel_size, stride, padding=0, alpha=0.5):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (
                    1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x
     
class Conv_Mix_Pooling_Dropout_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity() # GAP layer
        self.mix_pool = mixedPool(3, 2)
        self.dropout = nn.Dropout(p=0.0009)
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.ca(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.dropout(x)
        x = self.mix_pool(x)
        x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.ca(x)
        x = self.act(self.conv(x))
        x = self.dropout(x)
        x = self.mix_pool(x)
        x = self.sa(x)
        return x

class Conv_Avg_Pooling_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(2, stride=2)  # GAP layer
        # self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # x = self.ca(x)
        x = self.act(self.bn(self.conv(self.avg_pool(self.sa(x)))))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        # x = self.ca(x)
        x = self.act(self.conv(self.avg_pool(self.sa(x))))
        return x

class Conv_DownSampleAttn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(2, stride=2)  # GAP layer
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()
        # self.conv_stride = nn.Conv2d(c2, c2, 3, 2)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x)
        x = self.sa(x)
        x = self.avg_pool(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.ca(x)
        x = self.sa(x)
        x = self.avg_pool(x)
        return x
    
class Conv_Avg_Pooling_Attnv2(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2)  # GAP layer
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.ca(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.sa(x)
        x = self.avg_pool(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.ca(x)
        x = self.act(self.conv(x))
        x = self.sa(x)
        x = self.avg_pool(x)
        return x
    
class CBAM_Conv_Avg_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2)  # GAP layer
        self.cbam = CBAM(c1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.cbam(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.avg_pool(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.cbam(x)
        x = self.act(self.conv(x))
        x = self.avg_pool(x)
        return x
    
class Conv_Avg_Pooling_Spatial_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2)  # GAP layer
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.avg_pool(x)
        x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.avg_pool(x)
        x = self.sa(x)
        return x
    
class Conv_Spatial_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.sa(x)
        return x
    
class Conv_Avg_Pooling_Attn_Dropout(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2)
        self.dropout = nn.Dropout(p=0.0009)  # GAP layer
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.ca(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.ca(x)
        x = self.act(self.conv(x))
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = self.sa(x)
        return x
class DS_Conv_Attn(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(DS_Conv_Attn, self).__init__()
        self.depthwise = nn.Conv2d(c1, c2, k, s,padding=autopad(k, p, d), groups=math.gcd(c1, c2), dilation=d, bias=False)
        self.pointwise = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.cbam = CBAM(c2)

    def forward(self, x):
        x = self.act(self.bn(self.pointwise(self.depthwise(x))))
        x = self.cbam(x)
        return x
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.pointwise(self.depthwise(x)))
        x = self.cbam(x)
        return x
    
class DS_Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(DS_Conv, self).__init__()
        self.depthwise = nn.Conv2d(c1, c2, k, s,padding=autopad(k, p, d), groups=math.gcd(c1, c2), dilation=d, bias=False)
        self.pointwise = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        x = self.act(self.bn(self.pointwise(self.depthwise(x))))
        return x
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.pointwise(self.depthwise(x)))
        return x
    
class Conv_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.ca = ChannelAttention(c1)
        # self.sa = SpatialAttention()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.ca(x)
        x = self.act(self.bn(self.conv(x)))
        # x = self.sa(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.ca(x)
        x = self.act(self.conv(x))
        # x = self.sa(x)
        return x
    
class Conv_Dropout(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.dropout = nn.Dropout(p=0.2) # GAP layer

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.dropout(x)
        return x

class Conv_Fractional_Max_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.fractional_max_pool = nn.FractionalMaxPool2d(3, output_ratio=(0.25, 0.25))

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.fractional_max_pool(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.fractional_max_pool(x)
        return x
    

    
class CBAM_Conv_Fractional_Max_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.fractional_max_pool = nn.FractionalMaxPool2d(3, output_ratio=(0.25, 0.25))
        self.cbam = CBAM(c2)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.fractional_max_pool(x)
        x = self.cbam(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.fractional_max_pool(x)
        x = self.cbam(x)
        return x
    
class CBAM_Conv_Avg_Pooling(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.avg_pool = nn.AvgPool2d(3, stride=2)
        self.cbam = CBAM(c2)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.avg_pool(x)
        x = self.cbam(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.avg_pool(x)
        x = self.cbam(x)
        return x
    
class CBAM_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.cbam = CBAM(c1)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.cbam(x)
        x = self.act(self.bn(self.conv(x)))
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.cbam(x)
        x = self.act(self.conv(x))
        return x
    
class Conv_SP(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.sattn = SpatialAttention(kernel_size=3)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.act(self.bn(self.conv(x)))
        x = self.sattn(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.act(self.conv(x))
        x = self.sattn(x)
        return x
class StochasticPool2DLayer(nn.Module):
    def __init__(self, pool_size=2, maxpool=True, training=False, grid_size=None, **kwargs):
        super(StochasticPool2DLayer, self).__init__(**kwargs)
        self.rng = torch.cuda.manual_seed_all(123) # this changed in Pytorch for working
        self.pool_size = pool_size
        self.maxpool_flag = maxpool
        self.training = training
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = pool_size

        self.Maxpool = torch.nn.MaxPool2d(kernel_size=self.pool_size, stride=1)
        self.Avgpool = torch.nn.AvgPool2d(kernel_size=self.pool_size,
                                          stride=self.pool_size,
                                          padding=self.pool_size//2,)
        self.padding = nn.ConstantPad2d((0,1,0,1),0)

    def forward(self, x, **kwargs):
        if self.maxpool_flag:
            x = self.Maxpool(x)
            x = self.padding(x)
        if not self.training:
            # print(x.size())
            x = self.Avgpool(x)
            return x
            # return x[:, :, ::self.pool_size, ::self.pool_size]       
        else:
            w, h = x.data.shape[2:]
            n_w, n_h = w//self.grid_size, h//self.grid_size
            n_sample_per_grid = self.grid_size//self.pool_size
            # print('===========================')
            idx_w = []
            idx_h = []
            if w>2 and h>2:
                for i in range(n_w):
                    offset = self.grid_size * i
                    if i < n_w - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[2] - offset
                    
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])
                    idx_w.append(offset + this_idx)
                for i in range(n_h):
                    offset = self.grid_size * i
                    if i < n_h - 1:
                        this_n = self.grid_size
                    else:
                        this_n = x.data.shape[3] - offset
                    this_idx, _ = torch.sort(torch.randperm(this_n)[:n_sample_per_grid])

                    idx_h.append(offset + this_idx)
                idx_w = torch.cat(idx_w, dim=0)
                idx_h = torch.cat(idx_h, dim=0)
            else:
                idx_w = torch.LongTensor([0])
                idx_h = torch.LongTensor([0])

            output = x[:, :, idx_w.cpu()][:, :, :, idx_h.cpu()]
            return output
        
class Conv_S3Pool_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.s3pool = StochasticPool2DLayer(pool_size=4, maxpool=True)
        self.cattn = ChannelAttention(c1)
        self.sattn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.cattn(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.s3pool(x)
        x = self.sattn(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.cattn(x)
        x = self.act(self.conv(x))
        x = self.s3pool(x)
        x = self.sattn(x)
        return x
    
class Conv_Fractional_Max_Pooling_Attn(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.fractional_max_pool = nn.FractionalMaxPool2d(3, output_ratio=(0.25, 0.25))
        self.cattn = ChannelAttention(c1)
        self.sattn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.cattn(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.fractional_max_pool(x)
        x = self.sattn(x)
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.cattn(x)
        x = self.act(self.conv(x))
        x = self.fractional_max_pool(x)
        x = self.sattn(x)
        return x
    
# class InvertedResidual_BN_SL(Conv):
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
#         self.conv = InvertedResidual(c1, c2, s)
        


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))





class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))

class ChannelAttention_Pool(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, c1 ) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x *self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
    
class CBAM_Module(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
    
class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct = False):
        """

        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = 1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)
        return tmpZ


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
    

class Concat_Feature_Map(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        self.pool1 = nn.MaxPool2d(3,2,1)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat((x[0], self.pool1(x[1])), self.d)
    
class Nothing(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return x

