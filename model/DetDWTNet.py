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
FreqDetector with frequency-only FPN / PAN fusion for 1D-frequency object detection.

Design goals (implemented):
- Only upsample/downsample the FREQUENCY axis (last spatial dim), preserve TIME axis.
- Top-down lateral fusion (like YOLO): deep -> upsample 2x -> concat with next shallow
- Optional PAN bottom-up path-aggregation to strengthen low-level features
- SPP on deepest layer to enlarge receptive field in frequency axis (optional)
- Heads attached to multiple pyramid levels; each head expects (B,C,T,F)
- Example usage and shape checks at bottom

Assumptions:
- feat_list is ordered from shallow->deep: [C3, C4, C5], but you can pass more/less levels.
- Each tensor has shape (B, C, T, F_i). We assume TIME dimension T may differ across levels; the
  forward now performs pairwise time alignment at fusion points.
- in_channels_list must match feat_list channels.

Notes:
- This is a self-contained module you can drop into your codebase. It does not depend on other project files.
- The detection head is imported from your earlier implementation or you can use the included simple head.
"""


def align_time(x: torch.Tensor, T_tgt: int, mode: str = 'auto') -> torch.Tensor:
    """
    Align time dimension of x to T_tgt.

    x: (B, C, T_src, F)
    mode: 'auto'|'avg'|'linear'
      - 'auto': if T_src > T_tgt -> avg pool (downsample), else -> linear interpolate (upsample)
    Returns x_aligned: (B, C, T_tgt, F)
    """
    B, C, T_src, F_len = x.shape
    if T_src == T_tgt:
        return x
    if mode == 'auto':
        mode = 'avg' if T_src > T_tgt else 'linear'

    # reshape to (B*F, C, T_src) to use 1D ops
    x_perm = x.permute(0, 3, 1, 2).reshape(B * F_len, C, T_src)  # (B*F, C, T_src)

    if mode == 'avg':
        # adaptive avg pool gives robust downsampling
        x_al = F.adaptive_avg_pool1d(x_perm, output_size=T_tgt)  # This operation is not supported by mps backend yet!!!
    else:
        # linear interpolation for upsampling or explicit linear downsample
        x_al = F.interpolate(x_perm, size=T_tgt, mode='linear', align_corners=False)

    # reshape back to (B, C, T_tgt, F)
    x_al = x_al.view(B, F_len, C, T_tgt).permute(0, 2, 3, 1).contiguous()
    return x_al


def upsample_F(x: torch.Tensor, F_tgt: int) -> torch.Tensor:
    """
    Upsample only frequency axis from F_src to F_tgt.

    x: (B, C, T, F_src)
    returns: (B, C, T, F_tgt)
    Implementation: reshape to (B*T, C, F_src) and use interpolate(mode='linear').
    """
    B, C, T, F_src = x.shape
    if F_src == F_tgt:
        return x
    x_perm = x.permute(0, 2, 1, 3).reshape(B * T, C, F_src)  # (B*T, C, F_src)
    x_up = F.interpolate(x_perm, size=F_tgt, mode='linear', align_corners=False)
    x_up = x_up.view(B, T, C, F_tgt).permute(0, 2, 1, 3).contiguous()
    return x_up


def downsample_F(x: torch.Tensor, F_tgt: int) -> torch.Tensor:
    """Downsample only frequency axis to F_tgt using linear interpolation (or avg pool).
    Keep shape (B, C, T, F_tgt).
    """
    B, C, T, F_src = x.shape
    if F_src == F_tgt:
        return x
    x_perm = x.permute(0, 2, 1, 3).reshape(B * T, C, F_src)
    x_down = F.interpolate(x_perm, size=F_tgt, mode='linear', align_corners=False)
    x_down = x_down.view(B, T, C, F_tgt).permute(0, 2, 1, 3).contiguous()
    return x_down


class SPP1D(nn.Module):
    """
    SPP-like block on frequency axis using 1D pooling.

    Input: (B, C, T, F)
    Output: (B, C_out, T, F) where C_out == C (we concat pooled features then conv)
    """

    def __init__(self, in_ch: int, pool_sizes=(5, 9, 13)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.conv = nn.Conv2d(in_ch * (1 + len(pool_sizes)), in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F) => transform to (B*T, C, F) to do 1D pooling
        B, C, T, F_len = x.shape
        x_rt = x.permute(0, 2, 1, 3).reshape(B * T, C, F_len)  # (B*T, C, F)

        pooled = [x_rt]
        for k in self.pool_sizes:
            # pad so pooling keeps same length (use replicate pad)
            pad = (k - 1) // 2
            x_p = F.max_pool1d(F.pad(x_rt, (pad, k - 1 - pad), mode='replicate'), kernel_size=k, stride=1)
            pooled.append(x_p)

        cat = torch.cat(pooled, dim=1)  # (B*T, C*(1+len), F)
        out = self.conv(cat.unsqueeze(2)).squeeze(2)  # conv expects 4D, we use (N, C, 1, F)
        out = out.view(B, T, C, F_len).permute(0, 2, 1, 3).contiguous()  # (B, C, T, F)
        return out


class FreqRegressionDetector(nn.Module):
    """
    Input: x (B, in_channels, T, F)
    Output: dict with keys:
        - 'f_start': Tensor (B, N)  (values in [0,1] after sigmoid if use_sigmoid=True in decode)
        - 'f_end'  : Tensor (B, N)
        - 'class_logits': Tensor (B, N, num_classes)
        - 'confidence'  : Tensor (B, N)  (logits; apply sigmoid to get [0,1])
    Args:
        - in_channels: number of input channels
        - N: Fixed number of output items (f_start, f_end, class_logits, confidence)
        - num_classes: number of classes for classification
        - feat_channels: number of channels in feature maps
        - hidden_dim: number of hidden units in FC layers
        - backbone_downsample: number of downsampling layers in backbone
        - dropout: dropout rate in FC layers
    Usage:
        model = RegressionDetector(in_channels=1, N=8, num_classes=2)
        out = model(x)  # out is a dict
    """

    def __init__(self, in_channels: int = 1, N: int = 8, num_classes: int = 2, feat_channels: int = 64,
                 hidden_dim: int = 256, backbone_downsample: int = 4, dropout: float = 0.0, ):
        super().__init__()
        self.N = N
        self.num_classes = num_classes

        convs = []
        ch = in_channels
        for i in range(backbone_downsample):
            out_ch = feat_channels if i == 0 else feat_channels * (1 + i // 2)
            convs.append(nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.ReLU(inplace=True))
            ch = out_ch

        # 追加一层 3x3 conv 增加感受野（不改变尺寸）
        convs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False))
        convs.append(nn.BatchNorm2d(ch))
        convs.append(nn.ReLU(inplace=True))
        self.conv_backbone = nn.Sequential(*convs)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, ch, 1, 1)
        feat_dim = ch  # channels after backbone

        # Shared FC layers
        self.fc_shared = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(), )

        # Output: N * (2 (start,end) + num_classes + 1 (confidence))
        out_dim_per_item = 2 + num_classes + 1
        self.fc_out = nn.Linear(hidden_dim, N * out_dim_per_item)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        """
        # x: (B, C, T, F)
        B = x.size(0)
        feat = self.conv_backbone(x)  # (B, ch, T', F')
        pooled = self.global_pool(feat).view(B, -1)  # (B, feat_dim)
        hid = self.fc_shared(pooled)  # (B, hidden_dim)
        out = self.fc_out(hid)  # (B, N * out_dim_per_item)
        out = out.view(B, self.N, -1)  # (B, N, out_dim_per_item)

        # 切分
        f_start = out[..., 0]  # (B, N)  raw (logit) -> 推荐 sigmoid to [0,1]
        f_end = out[..., 1]  # (B, N)
        class_logits = out[..., 2:2 + self.num_classes]  # (B, N, C) logits
        confidence = out[..., -1]  # (B, N)  raw logits -> sigmoid for prob

        return {"f_start": f_start, "f_end": f_end, "class_logits": class_logits, "confidence": confidence}


"""
Main model for DWTNet.
"""


class DWTNet(nn.Module):
    def __init__(self, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4', extension_mode='reflect',
                 N=10, num_classes=2, dropout=0.0):
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
        self.detector = FreqRegressionDetector(in_channels=in_chans, N=N, num_classes=num_classes,
                                               feat_channels=filters[0], dropout=dropout)

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
        model = DWTNet(in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
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
        print(f"Detections shape: {[p.shape for p in detections]}")
        print(f"[\033[32mInfo\033[0m] Generating summary for DWTNet:")
        summary(model, input_size=(1, 1, 116, 1024))

    if test_mode == 'det':
        print(f"[\033[32mInfo\033[0m] Testing for FPNDetector:")
        B = 4
        T = 128
        F = 256
        x = torch.randn(B, 1, T, F)
        model = FreqRegressionDetector(in_channels=1, N=6, num_classes=2)
        out = model(x)
        dec = model.decode(out, apply_sigmoid=True, apply_softmax_class=False)
        print("f_start shape:", dec["f_start"].shape)  # (B, N)
        print("f_end shape:", dec["f_end"].shape)  # (B, N)
        print("class shape:", dec["class"].shape)  # (B, N, C) if softmax else logits
        print("confidence shape:", dec["confidence"].shape)  # (B, N)
