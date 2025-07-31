import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class LearnableDWT2D(nn.Module):
    """2D可学习离散小波变换模块，用于下采样"""

    def __init__(self, wavelet_name='db4', levels=2, alpha=50.0):
        super(LearnableDWT2D, self).__init__()
        self.levels = levels
        self.alpha = alpha
        wavelet = pywt.Wavelet(wavelet_name)
        dec_lo, dec_hi = np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)
        filter_len = len(dec_lo)
        dec_ll = np.outer(dec_lo, dec_lo)
        dec_lh = np.outer(dec_lo, dec_hi)
        dec_hl = np.outer(dec_hi, dec_lo)
        dec_hh = np.outer(dec_hi, dec_hi)
        w_init = np.stack([dec_ll, dec_lh, dec_hl, dec_hh], axis=0).astype(np.float32)
        self.dec_conv_weight = nn.Parameter(torch.tensor(w_init))
        self.theta_pos = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.theta_neg = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, x):
        coeffs = []
        current = x
        for lvl in range(self.levels):
            pad = self.dec_conv_weight.shape[-1] // 2
            x_pad = F.pad(current, (pad, pad, pad, pad), mode='reflect')
            out = F.conv2d(x_pad, self.dec_conv_weight, stride=2)
            LL = out[:, 0:1]
            LH = out[:, 1:2]
            HL = out[:, 2:3]
            HH = out[:, 3:4]
            high_coeffs = torch.cat([LH, HL, HH], dim=1)
            pos_mask = torch.sigmoid(self.alpha * (high_coeffs - self.theta_pos.view(1, -1, 1, 1)))
            neg_mask = torch.sigmoid(self.alpha * (-high_coeffs - self.theta_neg.view(1, -1, 1, 1)))
            mask = (pos_mask + neg_mask).clamp(max=1.0)
            high_denoised = high_coeffs * mask
            LH_d = high_denoised[:, 0:1]
            HL_d = high_denoised[:, 1:2]
            HH_d = high_denoised[:, 2:3]
            coeffs.append((LH_d, HL_d, HH_d))
            current = LL
        coeffs.append(current)
        coeffs_out = coeffs[::-1]
        return coeffs_out


class LearnableIDWT2D(nn.Module):
    """2D可学习逆离散小波变换模块，用于上采样"""

    def __init__(self, wavelet_name='db4', levels=2):
        super(LearnableIDWT2D, self).__init__()
        self.levels = levels
        wavelet = pywt.Wavelet(wavelet_name)
        rec_lo, rec_hi = np.array(wavelet.rec_lo), np.array(wavelet.rec_hi)
        rec_ll = np.outer(rec_lo, rec_lo)
        rec_lh = np.outer(rec_lo, rec_hi)
        rec_hl = np.outer(rec_hi, rec_lo)
        rec_hh = np.outer(rec_hi, rec_hi)
        w_rec_init = np.stack([rec_ll, rec_lh, rec_hl, rec_hh], axis=0).astype(np.float32)
        self.rec_conv_weight = nn.Parameter(torch.tensor(w_rec_init))
        self.pad = max(0, (w_rec_init.shape[-1] - 2) // 2)

    def forward(self, coeffs, original_size):
        current = coeffs[0]
        idx = 1
        for lvl in range(self.levels):
            LH, HL, HH = coeffs[idx]
            idx += 1
            LL_pad = F.pad(current, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            LH_pad = F.pad(LH, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            HL_pad = F.pad(HL, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            HH_pad = F.pad(HH, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            combined = torch.cat([LL_pad, LH_pad, HL_pad, HH_pad], dim=1)
            up = F.conv_transpose2d(combined, self.rec_conv_weight, stride=2, padding=self.pad)
            current = up
        if current.size(2) != original_size[0] or current.size(3) != original_size[1]:
            current = F.interpolate(current, size=original_size, mode='bilinear', align_corners=True)
        return current


class MaskDecoder(nn.Module):
    """Mask专用解码器分支"""

    def __init__(self, in_channels, out_channels, levels):
        super(MaskDecoder, self).__init__()
        self.levels = levels
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()

        # 创建上采样层和卷积层
        for i in range(levels):
            self.upsamples.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            ))
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)
            ))
            in_channels = in_channels // 2

        # 最终输出层
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_connections):
        for i in range(self.levels):
            x = self.upsamples[i](x)
            # 拼接跳接连接
            if i < len(skip_connections):
                x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.convs[i](x)
        return self.sigmoid(self.final_conv(x))


class DWTNet(nn.Module):
    """改进版DWTNet，瓶颈层单独引出一条支路输出mask"""

    def __init__(self, wavelet_name='db4', levels=3, base_channels=16, att_type='WLAM'):
        super(DWTNet, self).__init__()
        self.levels = levels
        self.base_channels = base_channels

        # DWT和IDWT层
        self.dwt_layers = nn.ModuleList([
            LearnableDWT2D(wavelet_name=wavelet_name, levels=1)
            for _ in range(levels)
        ])
        self.idwt_layers = nn.ModuleList([
            LearnableIDWT2D(wavelet_name=wavelet_name, levels=1)
            for _ in range(levels)
        ])

        # 编码器卷积
        self.enc_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(levels)
        ])

        # 瓶颈层处理
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 主解码器路径
        self.main_dec_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels * 2 if i == 0 else base_channels,
                          base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for i in range(levels)
        ])

        # Mask解码器路径
        self.mask_decoder = MaskDecoder(
            in_channels=base_channels * 2,
            out_channels=1,
            levels=levels
        )

        # 注意力模块
        self.att_dec = nn.ModuleList([
            AttentionModule(channels=base_channels, att_type=att_type)
            for _ in range(levels)
        ])

        # 最终重建卷积
        self.reconstruct_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        sizes = []
        highs = []
        skip_connections = []
        current = x

        # 编码器
        for i in range(self.levels):
            sizes.append((current.size(2), current.size(3)))
            coeffs = self.dwt_layers[i](current)
            LL, (LH, HL, HH) = coeffs[0], coeffs[1]
            highs.append((LH, HL, HH))
            current = self.enc_convs[i](LL)
            skip_connections.append(current)  # 保存跳接连接

        # 瓶颈层
        bottleneck_feat = self.bottleneck(current)

        # 解码器
        main_dec = bottleneck_feat[:, :self.base_channels]  # 取前半部分作为主路径输入
        for i in range(self.levels - 1, -1, -1):
            LH, HL, HH = highs[i]
            main_dec = self.main_dec_convs[i](main_dec)
            main_dec = self.idwt_layers[i]([main_dec, (LH, HL, HH)], sizes[i])
            main_dec = self.att_dec[i](main_dec)
        reconstructed = self.reconstruct_conv(main_dec)

        # Mask解码器路径
        mask_feat = bottleneck_feat[:, self.base_channels:]  # 取后半部分作为mask路径输入
        mask = self.mask_decoder(mask_feat, skip_connections[::-1])

        return reconstructed, mask


if __name__ == '__main__':
    model = DWTNet(wavelet_name='db4',
                   levels=3,
                   base_channels=64,
                   att_type='WLAM')
    sample = torch.randn(16, 1, 224, 224)
    denoised, rfi_mask = model(sample)
    summary(model, input_size=(16, 1, 224, 224))
    print("Denoised shape:", denoised.shape)
    print("RFI Mask shape:", rfi_mask.shape)