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


class DWTNet(nn.Module):
    def __init__(self, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4'):
        super(DWTNet, self).__init__()
        self.level = 1  # Do not change DWT internal level!
        filters = [dim, dim * levels[0], dim * levels[1], dim * levels[2], dim * levels[3]]

        # DWT
        self.dwt1 = DWT2D(J=self.level, wave=wavelet_name, mode='periodization')  # 必须使用Periodization模式(防止像素扩张)
        self.dwt2 = DWT2D(J=self.level, wave=wavelet_name, mode='periodization')
        self.dwt3 = DWT2D(J=self.level, wave=wavelet_name, mode='periodization')
        self.dwt4 = DWT2D(J=self.level, wave=wavelet_name, mode='periodization')

        # IDWT
        self.idwt1 = IDWT2D(wave=wavelet_name, mode='periodization')
        self.idwt2 = IDWT2D(wave=wavelet_name, mode='periodization')
        self.idwt3 = IDWT2D(wave=wavelet_name, mode='periodization')
        self.idwt4 = IDWT2D(wave=wavelet_name, mode='periodization')

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

        # Denoising分支
        self.deno_out = nn.Conv2d(filters[0], in_chans, kernel_size=1)

    def _pad_TF(self, x):
        """
        Pad the time dimension (T) and frequency dimension (F) to the nearest power of 2
        x: [B, C, T, F]
        return: (x_padded, orig_t, orig_f)
        """
        orig_t, orig_f = x.shape[2], x.shape[3]

        # Nearest power of 2
        target_t = 1 << (orig_t - 1).bit_length()
        target_f = 1 << (orig_f - 1).bit_length()
        pad_t = target_t - orig_t
        pad_f = target_f - orig_f
        if pad_t > 0 or pad_f > 0:
            # F.pad order is (left, right, top, bottom, front, back, ...)
            # Here padding (F axis, T axis)
            x = F.pad(x, (0, pad_f, 0, pad_t), mode="reflect")

        return x, orig_t, orig_f

    def _re_TF(self, x, orig_t, orig_f):
        """
        Restore to the original T and F
        x: [B, C, T, F]
        """
        return x[:, :, :orig_t, :orig_f]

    def forward(self, x):
        # orig_t, orig_f = x.size(1), x.size(2)
        x, orig_t, orig_f = self._pad_TF(x)

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
        assert bn.shape == lls[-1].shape, f"features must match ll_-1. ({bn.shape} != {lls[-1].shape})"
        d4 = torch.cat([bn, lls[-1]], dim=1)  # (B, 16D, H//16, W//16)
        d4 = self.dec4(d4)  # (B, 8D, H//16, W//16)
        d4 = self.idwt4([d4, Hs[-1]])  # (B, 8D, H//8, W//8)
        d4 = self.dec4_c(d4)  # (B, 4D, H//8, W//8)

        assert d4.shape == lls[-2].shape, f"features must match ll_-2. ({d4.shape} != {lls[-2].shape})"
        d3 = torch.cat([d4, lls[-2]], dim=1)  # (B, 8D, H//8, W//8)
        d3 = self.dec3(d3)  # (B, 4D, H//8, W//8)
        d3 = self.idwt3([d3, Hs[-2]])  # (B, 4D, H//4, W//4)
        d3 = self.dec3_c(d3)  # (B, 2D, H//4, W//4)

        assert d3.shape == lls[-3].shape, f"features must match ll_-3. ({d3.shape} != {lls[-3].shape})"
        d2 = torch.cat([d3, lls[-3]], dim=1)  # (B, 4D, H//4, W//4)
        d2 = self.dec2(d2)  # (B, 2D, H//4, W//4)
        d2 = self.idwt2([d2, Hs[-3]])  # (B, 2D, H//2, W//2)
        d2 = self.dec2_c(d2)  # (B, D, H//2, W//2)

        assert d2.shape == lls[-4].shape, f"features must match ll_-4. ({d2.shape} != {lls[-4].shape})"
        d1 = torch.cat([d2, lls[-4]], dim=1)  # (B, 2D, H//2, W//2)
        d1 = self.dec1(d1)  # (B, D, H//2, W//2)
        d1 = self.idwt1([d1, Hs[-4]])  # (B, D, H, W)
        ou = self.dec1_c(d1)  # (B, D, H, W)

        deno = self._re_TF(self.deno_out(ou), orig_t, orig_f)  # (B, 1, H, W)
        # print(f" --- Denoised shape: {deno.shape} --- ")

        return deno


if __name__ == '__main__':
    model = DWTNet(in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
    print(model)

    # (batch_size, channels, time channels, freq channels)
    B, C, H, W = 1, 1, 116, 1024
    tile_size = 16
    row = torch.arange(H).unsqueeze(1)
    col = torch.arange(W).unsqueeze(0)
    checkerboard = ((row // tile_size + col // tile_size) % 2).float()
    test_input = checkerboard.unsqueeze(0).unsqueeze(0).expand(B, C, H, W).clone()

    denoised = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Denoised shape: {denoised.shape}")
    summary(model, input_size=(1, 1, 116, 1024))
