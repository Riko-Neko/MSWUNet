from math import sqrt
from typing import Dict

import pytorch_wavelets.dwt.lowlevel as lowlevel
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

"""
Temporary tools for debugging.
"""


def plot_tensor(tensor):
    """
    Plot tensor（tensor[0, 0, :, :]）
    """

    import matplotlib.pyplot as plt
    image = tensor[0, 0].cpu().detach().numpy()  # [H, W]
    plt.imshow(image, cmap='gray')
    plt.title("Tensor[0, 0, :, :]")
    plt.axis('off')
    plt.show()


"""
These experimental methods of DWT2D, IDWT2D, and SWT 
are from the PyTorch Wavelets library.

The DWT2D and IDWT2D classes are used to perform 2D DWT and IDWT respectively.
The SWT class is used to perform 2D Stationary Wavelet Transform (SWT).

The DWT2D and IDWT2D classes are based on the PyTorch Wavelets library, which
is a wrapper around the pywt library. The SWT class is based on the pywt library
itself.

To use these classes, you can simply import them and create an instance of them
with the desired parameters. For example:

```
from model.DWTNet import DWT2D, IDWT2D, SWT

# Create a DWT2D object with 2 levels of decomposition and the 'db1' wavelet
dwt = DWT2D(J=2, wave='db1')

# Create an IDWT2D object with the 'db1' wavelet
idwt = IDWT2D(wave='db1')

# Create a SWT object with 2 levels of decomposition and the 'db1' wavelet
swt = SWT(J=2, wave='db1')
"""


class DWT2D(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """

    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # plot_tensor(ll)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        # plot_tensor(ll)

        return ll, yh


class IDWT2D(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # plot_tensor(ll)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)

            # plot_tensor(ll)

        return ll


class SWT(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """

    def __init__(self, J=1, wave='db1', mode='periodization'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        for j in range(self.J):
            # Do 1 level of the transform
            y = lowlevel.afb2d_atrous(ll, filts, self.mode, 2 ** j)
            coeffs.append(y)
            ll = y[:, :, 0]

        return coeffs


"""
These experimental methods of DWT2DL, IDWT2DL, and SWT2DL are based on the PyTorch Wavelets library.

They are designed to be fully learnable, with learnable thresholds for the highpass coefficients 
and learnable orthogonal basis of wavelet transform, which is a combination of the analysis and synthesis filters.

Attention: These methods are still under development and may not work properly.
The loss of some functions in the wavelet transform of the learnable orthogonal wavelet machine 
needs to guide these wavelet machines to develop in the orthogonal direction. 
Therefore, please preferably use the wavelet loss function we provide to guide.

with the desired parameters. For example:

```
from model.DWTNet import DWT2DL, IDWT2DL, SWT2DL

# Create a DWT2DL object with 2 levels of decomposition and the 'db4' wavelet
dwt = DWT2DL(wavelet_name='db4', level=2, alpha=50.0)

# Create an IDWT2DL object with the 'db4' wavelet
idwt = IDWT2DL(wavelet_name='db4', level=2, alpha=50.0)

# Create a SWT2DL object with 2 levels of decomposition and the 'db4' wavelet
swt = SWT2DL(wavelet_name='db4', level=2, alpha=50.0)
"""


class DWT2DL(nn.Module):
    pass


class IDWT2DL(nn.Module):
    pass


"""
Basic blocks for DWTNet.
"""


class ResBlock(nn.Module):
    """
    A classic residual block.
    """

    def __init__(self, in_chans, out_chans, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

        if in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_chans),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SeparableConvBlock(nn.Module):
    """
    A residual block with depthwise separable convolutions.
    """

    def __init__(self, in_chans, out_chans, kernel_size):
        super().__init__()
        # Depthwise separable convolution 1
        self.depthwise1 = nn.Conv2d(in_chans, in_chans, kernel_size, stride=1, padding=1, groups=in_chans, bias=False)
        self.pointwise1 = nn.Conv2d(in_chans, out_chans, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)

        # Depthwise separable convolution 2
        self.depthwise2 = nn.Conv2d(out_chans, out_chans, kernel_size, stride=1, padding=1, groups=out_chans,
                                    bias=False)
        self.pointwise2 = nn.Conv2d(out_chans, out_chans, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.relu = nn.ReLU(inplace=True)

        if in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_chans),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        # First depthwise separable convolution
        out = self.depthwise1(x)
        out = self.pointwise1(out)
        out = self.bn1(out)
        out = self.relu(out)

        # Second depthwise separable convolution
        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)


class UpConvBlock(nn.Module):
    """
    Upsample(optional) convolution block.
    """

    def __init__(self, in_chans, out_chans, up=False):
        super(UpConvBlock, self).__init__()
        self.up = up

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.conv(x)


class ConvBlock1D(nn.Module):
    """2D conv wrapper that treats time as a spatial dim but mostly focuses on freq axis.
    Kernel shape (1,k) is typical to avoid mixing time and frequency too aggressively.
    Input: (B, C, T, F) -> we will use nn.Conv2d with kernel (1,k_freq).
    """

    def __init__(self, in_ch, out_ch, k_freq=3, padding_freq=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=(1, k_freq), padding=(0, padding_freq))]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


"""
Regressive heads for 1D-frequency object detection.

Design goals (implemented):
- 1D-frequency object detection: predict start and end frequency of each object, and classify each object.
"""


class CoordAtt(nn.Module):
    """
    Coordinate Attention
    Reference idea: encode H and W positional information separately and generate channel attention.
    Implementation follows the common open-source variant.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # keep T, collapse F
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # keep F, collapse T
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h = self.pool_h(x)  # (B, C, T, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, F) -> (B, C, F, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H + F, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h_att = self.conv_h(y[:, :, :H, :])  # (B, C, T, 1)
        x_w_att = self.conv_w(y[:, :, H:, :]).permute(0, 1, 3, 2)  # (B, C, F, 1) -> (B, C, 1, F)
        attn = torch.sigmoid(x_h_att + x_w_att)  # broadcasting to (B,C,H,W)
        return x * attn


class FreqRegressionDetector(nn.Module):
    """
    Input: x (B, in_channels, T, F)
    Output: dict with keys:
        - 'f_start': Tensor (B, N)  (raw logits)
        - 'f_end'  : Tensor (B, N)
        - 'class_logits': Tensor (B, N, num_classes)
        - 'confidence'  : Tensor (B, N)  (logits)
    Args:
        - fchans: number of frequency bins
        - in_channels: number of input channels
        - N: Fixed number of output items
        - num_classes: number of classes
        - feat_channels: base channels in CNN
        - hidden_dim: hidden units in FC (not used in linear head)
        - backbone_downsample: downsampling layers
        - dropout: dropout rate
    Usage:
        model = FreqRegressionDetector(in_channels=1, N=8, num_classes=2)
        out = model(x)  # out is a dict
    """

    def __init__(self, fchans: int = 1024, in_channels: int = 1, N: int = 5, num_classes: int = 2,
                 feat_channels: int = 64, backbone_downsample: int = 4, dropout: float = 0.0):
        super().__init__()
        self.N = N
        self.num_classes = num_classes
        self.fchans = fchans
        self.backbone_downsample = backbone_downsample
        self.bottleneck = feat_channels
        backbone_coord_atts = True
        coord_att_reduction = 32
        strides = [(2, 2), (3, 2), (2, 2), (3, 2)]
        neck_dim_T = 2

        convs = []
        ch = in_channels
        for i in range(backbone_downsample):
            out_ch = feat_channels * (2 ** i)
            stride = strides[i]
            convs.append(nn.Conv2d(ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.ReLU(inplace=True))
            if dropout > 0:
                convs.append(nn.Dropout2d(p=dropout))
            ch = out_ch
            if backbone_coord_atts:
                convs.append(CoordAtt(out_ch, out_ch, reduction=coord_att_reduction))

        convs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False))
        convs.append(nn.BatchNorm2d(ch))
        convs.append(nn.ReLU(inplace=True))
        self.conv_backbone = nn.Sequential(*convs)

        self.freq_dim = self._get_F_dim(self.fchans, self.backbone_downsample)  # compute F'
        self.time_freq_pool = nn.AdaptiveAvgPool2d((neck_dim_T, self.freq_dim))  # T' -> 2

        self.bottleneck_conv = nn.Conv2d(ch, self.bottleneck, kernel_size=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(self.bottleneck)
        self.bottleneck_relu = nn.ReLU(inplace=True)
        self.coord_att_bottleneck = CoordAtt(self.bottleneck, self.bottleneck, reduction=coord_att_reduction)

        neck_dim = self.bottleneck * neck_dim_T * self.freq_dim
        out_dim = N * (2 + num_classes + 1)  # f_start, f_end, class_logits, confidence
        # hidden_dim = 1 << (int(sqrt(neck_dim * out_dim)).bit_length() - 1)  # hidden_dim ≈ sqrt(in_dim × out_dim)
        hidden_dim = 1 << (int(sqrt(neck_dim * out_dim)) - 1).bit_length()
        self.linear_head = nn.Sequential(nn.Linear(neck_dim, hidden_dim), nn.Linear(hidden_dim, out_dim))

        self._init_weights()

    def _get_F_dim(self, fchans: int, backbone_downsample: int) -> int:
        f = fchans
        for i in range(backbone_downsample):
            f = (f + 1) // 2
        return int(f)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return raw logits for f_start, f_end, class_logits, and confidence.
        Input: x (B, in_channels, T, F)
        """
        B = x.size(0)
        feat = self.conv_backbone(x)  # (B, ch, T', F')
        pooled = self.time_freq_pool(feat)  # (B, ch, 2, freq_dim)
        feat = self.bottleneck_relu(self.bottleneck_bn(self.bottleneck_conv(pooled)))  # (B, bottleneck, 2, freq_dim)
        feat = self.coord_att_bottleneck(feat)  # (B, bottleneck, 2, freq_dim)
        pooled = feat.view(B, -1)  # (B, bottleneck * 2 * freq_dim)
        out = self.linear_head(pooled)  # (B, N * (2 + num_classes + 1))
        out = out.view(B, self.N, -1)  # (B, N, 2 + num_classes + 1)

        # split outputs
        f_start = out[..., 0]  # (B, N)
        f_end = out[..., 1]  # (B, N)
        class_logits = out[..., 2:2 + self.num_classes]  # (B, N, num_classes)
        confidence = out[..., -1]  # (B, N)

        return {"f_start": f_start, "f_end": f_end, "class_logits": class_logits, "confidence": confidence}


"""
Main model for DWTNet.
"""


class DWTNet(nn.Module):
    def __init__(self, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4', extension_mode='reflect',
                 fchans=1024, N=5, num_classes=2, feat_channels=64, dropout=0.0):
        super(DWTNet, self).__init__()
        self.level = 1  # Do not change DWT internal level!
        filters = [dim, dim * levels[0], dim * levels[1], dim * levels[2], dim * levels[3]]

        # DWT
        self.dwt1 = DWT2D(J=self.level, wave=wavelet_name, mode=extension_mode)  # 使用Periodization模式(防止像素扩张)
        self.dwt2 = DWT2D(J=self.level, wave=wavelet_name, mode=extension_mode)
        self.dwt3 = DWT2D(J=self.level, wave=wavelet_name, mode=extension_mode)
        self.dwt4 = DWT2D(J=self.level, wave=wavelet_name, mode=extension_mode)

        # IDWT
        self.idwt1 = IDWT2D(wave=wavelet_name, mode=extension_mode)
        self.idwt2 = IDWT2D(wave=wavelet_name, mode=extension_mode)
        self.idwt3 = IDWT2D(wave=wavelet_name, mode=extension_mode)
        self.idwt4 = IDWT2D(wave=wavelet_name, mode=extension_mode)

        # Encoder
        self.enc1 = ResBlock(in_chans, filters[0], kernel_size=3)
        self.enc2 = ResBlock(filters[0], filters[1], kernel_size=3)
        self.enc3 = ResBlock(filters[1], filters[2], kernel_size=3)
        self.enc4 = ResBlock(filters[2], filters[3], kernel_size=3)

        # Bottleneck
        self.bottleneck = nn.Sequential(SeparableConvBlock(filters[3], filters[4], kernel_size=3),
                                        SeparableConvBlock(filters[4], filters[4], kernel_size=3),
                                        SeparableConvBlock(filters[4], filters[3], kernel_size=3))

        # Decoder
        self.dec4 = ResBlock(filters[4], filters[3], kernel_size=3)
        self.dec3 = ResBlock(filters[3], filters[2], kernel_size=3)
        self.dec2 = ResBlock(filters[2], filters[1], kernel_size=3)
        self.dec1 = ResBlock(filters[1], filters[0], kernel_size=3)

        self.dec4_c = UpConvBlock(filters[3], filters[2])
        self.dec3_c = UpConvBlock(filters[2], filters[1])
        self.dec2_c = UpConvBlock(filters[1], filters[0])
        self.dec1_c = UpConvBlock(filters[0], filters[0])

        # Denoising 分支
        self.deno_out = nn.Conv2d(filters[0], in_chans, kernel_size=1)

        # Regressions 分支
        self.detector = FreqRegressionDetector(fchans=fchans, in_channels=in_chans, N=N, num_classes=num_classes,
                                               feat_channels=feat_channels, dropout=dropout)

    def forward(self, x):
        lls = []
        Hs = []

        # Encoder
        e1 = self.enc1(x)  # (B, D, H, W)
        ll, H = self.dwt1(e1)  # lls[-4]: (B, D, H//2, W//2)
        lls.append(ll)
        Hs.append(H)

        e2 = self.enc2(ll)  # (B, 2D, H//2, W//2)
        ll, H = self.dwt2(e2)  # lls[-3]: (B, 2D, H//4, W//4)
        lls.append(ll)
        Hs.append(H)

        e3 = self.enc3(ll)  # (B, 4D, H//4, W//4)
        ll, H = self.dwt3(e3)  # lls[-2]: (B, 4D, H//8, W//8)
        lls.append(ll)
        Hs.append(H)

        e4 = self.enc4(ll)  # (B, 8D, H//8, W//8)
        ll, H = self.dwt4(e4)  # lls[-1]: (B, 8D, H//16, W//16)
        lls.append(ll)
        Hs.append(H)

        # Bottleneck
        bn = self.bottleneck(ll)  # (B, 8D, H//16, W//16)

        # Denoising Branch Decoder (With Skip Connections)
        bn = self._interp_if(bn, lls[-1])
        d4 = torch.cat([bn, lls[-1]], dim=1)  # (B, 16D, H//16, W//16)
        d4 = self.dec4(d4)  # (B, 8D, H//16, W//16)
        d4 = self.idwt4([d4, Hs[-1]])  # (B, 8D, H//8, W//8)
        d4 = self.dec4_c(d4)  # (B, 4D, H//8, W//8)

        d4 = self._interp_if(d4, lls[-2])
        d3 = torch.cat([d4, lls[-2]], dim=1)  # (B, 8D, H//8, W//8)
        d3 = self.dec3(d3)  # (B, 4D, H//8, W//8)
        d3 = self.idwt3([d3, Hs[-2]])  # (B, 4D, H//4, W//4)
        d3 = self.dec3_c(d3)  # (B, 2D, H//4, W//4)

        d3 = self._interp_if(d3, lls[-3])
        d2 = torch.cat([d3, lls[-3]], dim=1)  # (B, 4D, H//4, W//4)
        d2 = self.dec2(d2)  # (B, 2D, H//4, W//4)
        d2 = self.idwt2([d2, Hs[-3]])  # (B, 2D, H//2, W//2)
        d2 = self.dec2_c(d2)  # (B, D, H//2, W//2)

        d2 = self._interp_if(d2, lls[-4])
        d1 = torch.cat([d2, lls[-4]], dim=1)  # (B, 2D, H//2, W//2)
        d1 = self.dec1(d1)  # (B, D, H//2, W//2)
        d1 = self.idwt1([d1, Hs[-4]])  # (B, D, H, W)
        ou = self.dec1_c(d1)  # (B, D, H, W)

        # deno = self._re_TF(self.deno_out(ou), orig_t, orig_f)  # (B, 1, H, W)
        deno = self.deno_out(ou)  # (B, 1, H, W)
        # print(f" --- Denoised shape: {deno.shape} --- ")

        regs = self.detector(deno)  # dict with keys: f_start, f_end, class_logits, confidence, logits

        return deno, regs

    @staticmethod
    def _interp_if(d, ll):
        _, _, H, W = ll.shape
        return F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)


"""
Testing code.
"""
if __name__ == '__main__':
    test_mode = 'model'
    # test_mode = 'det'

    if test_mode == 'model':
        print(f"[\033[32mInfo\033[0m] Testing for DWTNet:")
        model = DWTNet(in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4', extension_mode='reflect',
                       fchans=1024, N=5, num_classes=2, feat_channels=64, dropout=0.0)
        print(model)

        # (batch_size, channels, time channels, freq channels)
        B = 1
        C = 1
        H = 116
        W = 1024
        tile_size = 16
        row = torch.arange(H).unsqueeze(1)
        col = torch.arange(W).unsqueeze(0)
        checkerboard = ((row // tile_size + col // tile_size) % 2).float()
        wave = (torch.sin(row / 10.0) + torch.cos(col / 10.0)) * 0.15 + 0.15
        line = torch.zeros_like(wave)
        thickness = 3
        k = 0.5
        b = W // 4
        for y in range(H):
            x_center = int(k * y + b)
            if 0 <= x_center < W:
                x_start = max(0, x_center - thickness)
                x_end = min(W, x_center + thickness + 1)
                line[y, x_start:x_end] = 1.0

        # mode
        # mode = "checkerboard"
        mode = "wave_line"
        # mode = "noise"

        if mode == "checkerboard":
            test_input = checkerboard.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).clone()
        elif mode == "wave_line":
            combined = wave.clone()
            combined[line > 0] = 255
            test_input = combined.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).clone()
        else:
            test_input = torch.randn((B, C, H, W))

        denoised, detections = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Denoised shape: {denoised.shape}")
        print(f"Detections: {detections}")
        print(f"[\033[32mInfo\033[0m] Generating summary for DWTNet:")
        summary(model, input_size=(1, 1, 116, 1024))

    if test_mode == 'det':
        print(f"[\033[32mInfo\033[0m] Testing for FPNDetector:")
        B = 4
        T = 128
        F = 256
        x = torch.randn(B, 1, T, F)
        model = FreqRegressionDetector(in_channels=1, N=5, num_classes=2)
        out = model(x)
        dec = model.decode(out, apply_sigmoid=True, apply_softmax_class=False)
        print("f_start shape:", dec["f_start"].shape)  # (B, N)
        print("f_end shape:", dec["f_end"].shape)  # (B, N)
        print("class shape:", dec["class"].shape)  # (B, N, C) if softmax else logits
        print("confidence shape:", dec["confidence"].shape)  # (B, N)
