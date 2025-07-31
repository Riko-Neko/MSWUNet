import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import pywt
from typing import Tuple, List


class DWT2d(nn.Module):
    """可微分的二维离散小波变换(DWT)模块"""

    def __init__(self, wavelet_name: str = 'db4', levels: int = 2):
        super(DWT2d, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        dec_lo, dec_hi, _, _ = self.wavelet.filter_bank

        # 构造 2D 卷积核：外积
        def outer(a, b):
            return torch.tensor(np.outer(a, b), dtype=torch.float32)

        self.register_buffer('dec_ll', outer(dec_lo, dec_lo)[None, None])
        self.register_buffer('dec_lh', outer(dec_lo, dec_hi)[None, None])
        self.register_buffer('dec_hl', outer(dec_hi, dec_lo)[None, None])
        self.register_buffer('dec_hh', outer(dec_hi, dec_hi)[None, None])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, channels, height, width = x.size()
        coeffs = []
        current = x

        for _ in range(self.levels):
            pad = self.dec_ll.shape[-1] // 2
            current_padded = F.pad(current, (pad, pad, pad, pad), mode='reflect')

            ll = F.conv2d(current_padded, self.dec_ll, stride=2, groups=1)
            lh = F.conv2d(current_padded, self.dec_lh, stride=2, groups=1)
            hl = F.conv2d(current_padded, self.dec_hl, stride=2, groups=1)
            hh = F.conv2d(current_padded, self.dec_hh, stride=2, groups=1)

            coeffs.append((lh, hl, hh))  # 注意：低频部分留给下一层
            current = ll

        coeffs.append(current)  # 最后的低频部分
        return coeffs[::-1]  # [LL_n, (LH_n, HL_n, HH_n), ..., (LH1, HL1, HH1)]


class IDWT2d(nn.Module):
    """可微分的二维逆离散小波变换(IDWT)模块"""

    def __init__(self, wavelet_name: str = 'db1', levels: int = 2):
        super(IDWT2d, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        _, _, rec_lo, rec_hi = self.wavelet.filter_bank

        def outer(a, b):
            return torch.tensor(np.outer(a, b), dtype=torch.float32)

        self.register_buffer('rec_ll', outer(rec_lo, rec_lo)[None, None])
        self.register_buffer('rec_lh', outer(rec_lo, rec_hi)[None, None])
        self.register_buffer('rec_hl', outer(rec_hi, rec_lo)[None, None])
        self.register_buffer('rec_hh', outer(rec_hi, rec_hi)[None, None])

    def forward(self, coeffs: List[torch.Tensor], original_size: Tuple[int, int]) -> torch.Tensor:
        current = coeffs[0]  # 最低频部分：LL_n
        batch_size, channels, _, _ = current.size()

        for (lh, hl, hh) in coeffs[1:]:
            current_padded = F.pad(current, (1, 1, 1, 1), mode='reflect')
            lh_padded = F.pad(lh, (1, 1, 1, 1), mode='reflect')
            hl_padded = F.pad(hl, (1, 1, 1, 1), mode='reflect')
            hh_padded = F.pad(hh, (1, 1, 1, 1), mode='reflect')

            up_ll = F.conv_transpose2d(current_padded, self.rec_ll, stride=2)
            up_lh = F.conv_transpose2d(lh_padded, self.rec_lh, stride=2)
            up_hl = F.conv_transpose2d(hl_padded, self.rec_hl, stride=2)
            up_hh = F.conv_transpose2d(hh_padded, self.rec_hh, stride=2)

            # 对齐维度
            min_h = min(up_ll.shape[2], up_lh.shape[2], up_hl.shape[2], up_hh.shape[2])
            min_w = min(up_ll.shape[3], up_lh.shape[3], up_hl.shape[3], up_hh.shape[3])
            current = up_ll[:, :, :min_h, :min_w] + up_lh[:, :, :min_h, :min_w] \
                      + up_hl[:, :, :min_h, :min_w] + up_hh[:, :, :min_h, :min_w]

        if current.size(2) != original_size[0] or current.size(3) != original_size[1]:
            current = F.interpolate(current, size=original_size, mode='bilinear', align_corners=True)

        return current


class DWT1d(nn.Module):
    """可微分的一维离散小波变换(DWT)模块"""

    def __init__(self, wavelet_name: str = 'db1', levels: int = 2):
        super(DWT1d, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        dec_lo, dec_hi, _, _ = self.wavelet.filter_bank
        self.register_buffer('dec_lo', torch.tensor(dec_lo).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('dec_hi', torch.tensor(dec_hi).float().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, channels, time, freq = x.size()
        coeffs = []
        current = x

        for _ in range(self.levels):
            current_reshaped = current.reshape(-1, 1, current.size(3))
            pad = self.dec_lo.shape[-1] // 2
            cA_reshaped = F.conv1d(
                F.pad(current_reshaped, (pad, pad), mode='reflect'),
                self.dec_lo, stride=2
            )
            cD_reshaped = F.conv1d(
                F.pad(current_reshaped, (pad, pad), mode='reflect'),
                self.dec_hi, stride=2
            )
            freq_new = cA_reshaped.size(2)
            cA = cA_reshaped.view(batch_size, time, channels, freq_new).permute(0, 2, 1, 3)
            cD = cD_reshaped.view(batch_size, time, channels, freq_new).permute(0, 2, 1, 3)
            coeffs.append(cD)
            current = cA

        coeffs.append(current)
        return coeffs[::-1]


class IDWT1d(nn.Module):
    """可微分的一维逆离散小波变换(IDWT)模块"""

    def __init__(self, wavelet_name: str = 'db1', levels: int = 2):
        super(IDWT1d, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        _, _, rec_lo, rec_hi = self.wavelet.filter_bank
        self.register_buffer('rec_lo', torch.tensor(rec_lo).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('rec_hi', torch.tensor(rec_hi).float().unsqueeze(0).unsqueeze(0))

    def forward(self, coeffs: List[torch.Tensor], original_size: Tuple[int, int]) -> torch.Tensor:
        current = coeffs[0]
        batch_size, channels, time, _ = current.size()

        for cD in coeffs[1:]:
            current_reshaped = current.reshape(-1, 1, current.size(3))
            cD_reshaped = cD.reshape(-1, 1, cD.size(3))
            pad = self.rec_lo.shape[-1] // 2

            current_up = F.conv_transpose1d(
                current_reshaped, self.rec_lo, stride=2,
                padding=pad, output_padding=1
            )
            cD_up = F.conv_transpose1d(
                cD_reshaped, self.rec_hi, stride=2,
                padding=pad, output_padding=1
            )

            min_length = min(current_up.size(2), cD_up.size(2))
            current_reshaped = current_up[:, :, :min_length] + cD_up[:, :, :min_length]
            freq_new = current_reshaped.size(2)
            current = current_reshaped.view(batch_size, time, channels, freq_new).permute(0, 2, 1, 3)

        if current.size(2) != original_size[0] or current.size(3) != original_size[1]:
            current = F.interpolate(current, size=original_size, mode='bilinear', align_corners=True)

        return current


class SeparableConv2d(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = SeparableConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DWTNet(nn.Module):
    def __init__(self, wavelet_name: str = 'db4', levels: int = 3,
                 base_channels: int = 16, use_multibranch: bool = False):
        super(DWTNet, self).__init__()
        self.levels = levels
        self.use_multibranch = use_multibranch
        # self.dwt = DWT1d(wavelet_name, levels)
        # self.idwt = IDWT1d(wavelet_name, levels)
        self.dwt = DWT2d(wavelet_name, levels)
        self.idwt = IDWT2d(wavelet_name, levels)

        # 计算展平后的系数数量 (1个LL + 每级3个高频)
        self.num_flat_coeffs = 1 + 3 * levels

        # 多分支处理
        if use_multibranch:
            self.multibranch = nn.ModuleList()
            for _ in range(self.num_flat_coeffs):
                branch = nn.Sequential(
                    ResidualBlock(1),
                    ResidualBlock(1),
                    nn.Conv2d(1, base_channels, kernel_size=1)  # 扩展通道数
                )
                self.multibranch.append(branch)
            fusion_channels = base_channels * self.num_flat_coeffs
        else:
            fusion_channels = self.num_flat_coeffs

        # 编码器路径
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            nn.Conv2d(fusion_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            nn.MaxPool2d(2)
        ))

        self.encoder.append(nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
            nn.MaxPool2d(2)
        ))

        self.encoder.append(nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            nn.MaxPool2d(2)
        ))

        self.encoder.append(nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            nn.MaxPool2d(2)
        ))

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

        # 解码器路径
        self.decoder = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        ))
        self.skip_connections.append(nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=1))

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        ))
        self.skip_connections.append(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1))

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ))
        self.skip_connections.append(nn.Conv2d(base_channels, base_channels, kernel_size=1))

        self.decoder.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        ))
        self.skip_connections.append(nn.Conv2d(base_channels, base_channels, kernel_size=1))

        # 输出层（去噪图像） - 为每个展平系数创建输出卷积
        self.output_convs = nn.ModuleList()
        for _ in range(self.num_flat_coeffs):
            self.output_convs.append(nn.Conv2d(base_channels, 1, kernel_size=1))

        # RFI掩码生成层
        self.rfi_mask_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = (x.size(2), x.size(3))
        coeffs = self.dwt(x)

        # 展平系数列表 (LL_n, LH_n, HL_n, HH_n, LH_{n-1}, ...)
        flat_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):  # 高频分量 (LH, HL, HH)
                flat_coeffs.extend(c)
            else:  # 低频分量 LL
                flat_coeffs.append(c)

        # 处理小波系数
        processed_coeffs = []
        if self.use_multibranch:
            for coeff, branch in zip(flat_coeffs, self.multibranch):
                # 添加通道维度 (B,1,H,W)
                if coeff.dim() == 3:
                    coeff = coeff.unsqueeze(1)
                processed = branch(coeff)
                processed = F.interpolate(processed, size=original_size, mode='bilinear', align_corners=True)
                processed_coeffs.append(processed)
        else:
            for coeff in flat_coeffs:
                # 添加通道维度 (B,1,H,W)
                if coeff.dim() == 3:
                    coeff = coeff.unsqueeze(1)
                coeff = F.interpolate(coeff, size=original_size, mode='bilinear', align_corners=True)
                processed_coeffs.append(coeff)

        # 拼接特征
        features = torch.cat(processed_coeffs, dim=1)

        # 编码路径
        skip_connections = []
        for encoder_block in self.encoder:
            features = encoder_block(features)
            skip_connections.append(features)

        # 瓶颈层
        features = self.bottleneck(features)

        # 解码路径（反向处理跳跃连接）
        for i, decoder_block in enumerate(self.decoder):
            features = decoder_block(features)

            # 获取对应的跳跃连接（从后往前）
            skip_index = len(skip_connections) - i - 2
            if 0 <= skip_index < len(skip_connections):
                skip = skip_connections[skip_index]
                skip_connection = self.skip_connections[i](skip)

                # 确保尺寸匹配
                if features.size()[2:] != skip_connection.size()[2:]:
                    skip_connection = F.interpolate(
                        skip_connection, size=features.size()[2:],
                        mode='bilinear', align_corners=True
                    )
                features = features + skip_connection

        # 输出层（去噪图像）
        output_coeffs = []
        for i, conv in enumerate(self.output_convs):
            coeff_pred = conv(features)

            # 计算目标尺寸 (不同级别有不同尺寸)
            if i == 0:  # LL_n
                level = self.levels
            else:  # 高频分量
                level = self.levels - (i - 1) // 3

            target_height = original_size[0] // (2 ** level)
            target_width = original_size[1] // (2 ** level)

            coeff_pred = F.interpolate(
                coeff_pred,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=True
            )
            output_coeffs.append(coeff_pred)

        # 重组系数结构供IDWT使用 [LL, (LH,HL,HH), ...]
        idwt_coeffs = [output_coeffs[0]]  # LL_n
        idx = 1
        for _ in range(self.levels):
            # 每组取3个高频分量
            group = (
                output_coeffs[idx],
                output_coeffs[idx + 1],
                output_coeffs[idx + 2]
            )
            idwt_coeffs.append(group)
            idx += 3

        # 逆小波变换生成去噪图像
        denoised = self.idwt(idwt_coeffs, original_size)

        # 生成RFI掩码
        rfi_mask = self.rfi_mask_conv(features)
        rfi_mask = self.sigmoid(rfi_mask)
        rfi_mask = F.interpolate(rfi_mask, size=original_size, mode='bilinear', align_corners=True)

        return denoised, rfi_mask


if __name__ == '__main__':
    model = DWTNet(use_multibranch=True, levels=6, base_channels=128)
    sample = torch.randn(16, 1, 224, 224)
    denoised, rfi_mask = model(sample)
    summary(model, input_size=(16, 1, 224, 224))
    print("Denoised shape:", denoised.shape)
    print("RFI Mask shape:", rfi_mask.shape)
