import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class LearnableDWT2D(nn.Module):
    """
    2D 可学习离散小波变换 (DWT) 模块，用于下采样。
    将输入分解为多个等级的 LL、LH、HL、HH 子带，并对高频子带应用可学习阈值去噪。
    """

    def __init__(self, wavelet_name='db4', levels=1, alpha=50.0):
        super(LearnableDWT2D, self).__init__()
        self.levels = levels
        self.alpha = alpha  # 控制阈值函数陡峭度的系数
        # 初始化小波滤波器系数
        wavelet = pywt.Wavelet(wavelet_name)
        dec_lo, dec_hi = np.array(wavelet.dec_lo), np.array(wavelet.dec_hi)
        filter_len = len(dec_lo)
        # 构造二维分解滤波器 (4 个输出通道对应 LL, LH, HL, HH)
        dec_ll = np.outer(dec_lo, dec_lo)  # 低频x低频
        dec_lh = np.outer(dec_lo, dec_hi)  # 低频x高频
        dec_hl = np.outer(dec_hi, dec_lo)  # 高频x低频
        dec_hh = np.outer(dec_hi, dec_hi)  # 高频x高频
        w_init = np.stack([dec_ll, dec_lh, dec_hl, dec_hh], axis=0).astype(np.float32)  # (4, filter_len, filter_len)
        # 将滤波器作为可学习参数（4x1卷积核，相当于每个输入通道应用同一组小波滤波器）
        self.dec_conv_weight = nn.Parameter(torch.tensor(w_init))
        # 为三个高频子带 (LH, HL, HH) 设置可学习的正负阈值参数
        self.theta_pos = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.theta_neg = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # 注意：不使用偏置，保持小波变换的线性特性

    def forward(self, x):
        """
        对输入张量 x 进行多层离散小波分解。
        返回系数列表格式：[LL_n, (LH_n, HL_n, HH_n), ..., (LH_1, HL_1, HH_1)]，LL_n 为最低频子带。
        """
        coeffs = []
        current = x
        # 逐层进行 DWT 分解
        for lvl in range(self.levels):
            # 边界填充以减少卷积引入的边缘效应
            pad = self.dec_conv_weight.shape[-1] // 2
            x_pad = F.pad(current, (pad, pad, pad, pad), mode='reflect')
            # 应用可学习小波滤波器进行卷积，下采样步长为2
            out = F.conv2d(x_pad, self.dec_conv_weight, stride=2)  # 输出形状: (B, 4, H/2, W/2)
            # 分离低频和高频子带
            LL = out[:, 0:1]  # 低频子带
            LH = out[:, 1:2]
            HL = out[:, 2:3]
            HH = out[:, 3:4]  # 高频子带
            # 对高频子带应用可学习硬阈值激活，保留重要细节系数
            high_coeffs = torch.cat([LH, HL, HH], dim=1)  # 拼接高频系数 (B, 3, H/2, W/2)
            # 计算阈值掩膜：对正负方向分别使用 theta_pos 和 theta_neg 作为阈值
            pos_mask = torch.sigmoid(self.alpha * (high_coeffs - self.theta_pos.view(1, -1, 1, 1)))
            neg_mask = torch.sigmoid(self.alpha * (-high_coeffs - self.theta_neg.view(1, -1, 1, 1)))
            mask = (pos_mask + neg_mask).clamp(max=1.0)  # 掩膜取值约束在 [0,1]
            # 应用掩膜抑制低幅值系数
            high_denoised = high_coeffs * mask
            # 拆分阈值处理后的高频子带
            LH_d = high_denoised[:, 0:1]
            HL_d = high_denoised[:, 1:2]
            HH_d = high_denoised[:, 2:3]
            coeffs.append((LH_d, HL_d, HH_d))
            current = LL  # 下一层以当前层的LL作为输入继续分解
        coeffs.append(current)  # 最后一层的 LL_n 系数
        # 将系数列表倒序，形成从最低频 LL_n 到高频细节的输出顺序
        coeffs_out = coeffs[::-1]
        return coeffs_out


class LearnableIDWT2D(nn.Module):
    """
    2D 可学习逆离散小波变换 (IDWT) 模块，用于上采样重建。
    根据提供的各级子带系数逐层重构出高分辨率图像。
    """

    def __init__(self, wavelet_name='db4', levels=2):
        super(LearnableIDWT2D, self).__init__()
        self.levels = levels
        # 初始化小波重构滤波器系数
        wavelet = pywt.Wavelet(wavelet_name)
        rec_lo, rec_hi = np.array(wavelet.rec_lo), np.array(wavelet.rec_hi)
        rec_ll = np.outer(rec_lo, rec_lo)  # LL 重构滤波器
        rec_lh = np.outer(rec_lo, rec_hi)  # LH 重构滤波器
        rec_hl = np.outer(rec_hi, rec_lo)  # HL 重构滤波器
        rec_hh = np.outer(rec_hi, rec_hi)  # HH 重构滤波器
        w_rec_init = np.stack([rec_ll, rec_lh, rec_hl, rec_hh], axis=0).astype(np.float32)
        # 将重构滤波器作为可学习参数（4x1转置卷积核，将4个子带合并为1个输出）
        self.rec_conv_weight = nn.Parameter(torch.tensor(w_rec_init))
        # 计算转置卷积所需的填充，使输出尺寸匹配 IDWT 重构后的尺寸
        self.pad = max(0, (w_rec_init.shape[-1] - 2) // 2)
        # 不使用偏置，保持重构滤波的线性特性

    def forward(self, coeffs, original_size):
        """
        根据提供的各级小波系数列表 coeffs 重构图像。
        coeffs 格式为：[LL_n, (LH_n, HL_n, HH_n), ..., (LH_1, HL_1, HH_1)]。original_size 为目标重构图像尺寸 (H, W)。
        """
        current = coeffs[0]  # 起始使用最低频 LL_n 子带
        idx = 1
        # 逐层进行逆变换，从低频逐步融合各级高频细节
        for lvl in range(self.levels):
            # 取出当前层级的高频子带 (LH, HL, HH)
            LH, HL, HH = coeffs[idx]
            idx += 1
            # 对每个子带在边缘进行填充，使尺寸对齐
            LL_pad = F.pad(current, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            LH_pad = F.pad(LH, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            HL_pad = F.pad(HL, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            HH_pad = F.pad(HH, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            # 将 LL 和各高频子带沿通道拼接：(B, 4, H_l, W_l)
            combined = torch.cat([LL_pad, LH_pad, HL_pad, HH_pad], dim=1)
            # 转置卷积上采样，将4个子带融合重建上一层分辨率的图像
            up = F.conv_transpose2d(combined, self.rec_conv_weight, stride=2, padding=self.pad)
            current = up  # current 现在是上一层重建后的 LL (B,1,H_{l-1},W_{l-1})
        # 若最终尺寸与原尺寸不符，则进行双线性插值调整（主要处理不能整除的尺寸情况）
        if current.size(2) != original_size[0] or current.size(3) != original_size[1]:
            current = F.interpolate(current, size=original_size, mode='bilinear', align_corners=True)
        return current


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道压缩：全局平均池化
        y = x.view(B, C, -1).mean(dim=2)
        # 激励：全连接层引入非线性和通道权重
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        # 按通道加权原特征
        return x * y


class CBAMBlock(nn.Module):
    """CBAM 通道+空间注意力模块。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # 通道注意力 (使用 SEBlock 实现)
        self.channel_att = SEBlock(channels, reduction=reduction)
        # 空间注意力：使用一个卷积从平均池化和最大池化的结果生成空间权重
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        x_ca = self.channel_att(x)
        # 空间注意力：计算每像素的平均和最大通道响应
        avg_pool = x_ca.mean(dim=1, keepdim=True)
        max_pool, _ = x_ca.max(dim=1, keepdim=True)
        # 拼接后通过卷积生成空间注意力图
        spatial_map = self.sigmoid(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
        # 应用空间注意力
        return x_ca * spatial_map


class WLAMBlock(nn.Module):
    """
    小波增强线性注意力模块 (WLAM)。
    对输入特征进行一级 Haar 小波分解，利用低频部分进行全局(通道)注意力，利用高频部分进行局部(空间)注意力。
    """

    def __init__(self, channels):
        super(WLAMBlock, self).__init__()
        # 定义固定的 Haar 小波滤波器（低通: [0.5,0.5], 高通: [-0.5,0.5]）
        low_filter = torch.tensor([0.5, 0.5], dtype=torch.float32)
        high_filter = torch.tensor([-0.5, 0.5], dtype=torch.float32)
        ll = torch.outer(low_filter, low_filter)  # LL 滤波核
        lh = torch.outer(low_filter, high_filter)  # LH 滤波核
        hl = torch.outer(high_filter, low_filter)  # HL 滤波核
        hh = torch.outer(high_filter, high_filter)  # HH 滤波核
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # 形状 (4,1,2,2)
        self.register_buffer('haar_filters', filt)  # 将 Haar 滤波器注册为常量缓冲
        # 通道注意力：对每通道的 LL 子带特征应用 SE-like 注意力
        self.channel_fc1 = nn.Linear(channels, max(channels // 16, 1))
        self.channel_fc2 = nn.Linear(max(channels // 16, 1), channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # 空间注意力：对高频能量图应用卷积产生空间权重
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 对每个通道的特征图进行一级 Haar 小波变换 (使用 group convolution 实现并行处理)
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')  # 填充边界以适应 2x2 滤波
        # 对每个输入通道应用 4 个 Haar 滤波器 (groups=C)，输出形状 (B, 4*C, H/2, W/2)
        out = F.conv2d(x_pad, self.haar_filters.expand(4 * C, -1, -1, -1), stride=2, groups=C)
        # 拆分子带：每组 C 个通道对应 LL, LH, HL, HH
        LL = out[:, :C]  # 低频部分 (B, C, H/2, W/2)
        LH = out[:, C:2 * C]  # 水平高频 (B, C, H/2, W/2)
        HL = out[:, 2 * C:3 * C]  # 垂直高频 (B, C, H/2, W/2)
        HH = out[:, 3 * C:4 * C]  # 对角高频 (B, C, H/2, W/2)
        # 2. 通道注意力：基于每通道的低频系数 LL 计算全局权重
        ll_mean = LL.view(B, C, -1).mean(dim=2)  # 每通道低频平均值 (B, C)
        ch_weights = self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(ll_mean)))).view(B, C, 1, 1)
        x_channel_att = x * ch_weights  # 按通道权重调整原特征
        # 3. 空间注意力：利用高频能量的平均值生成空间注意力图
        hf_energy = (LH ** 2 + HL ** 2 + HH ** 2).mean(dim=1, keepdim=True)  # 高频能量平均 (B, 1, H/2, W/2)
        spatial_map = self.sigmoid(self.spatial_conv(hf_energy))  # (B, 1, H/2, W/2)
        # 将空间注意力图上采样回原始尺寸，再施加于通道注意后的特征
        spatial_map_up = F.interpolate(spatial_map, size=(H, W), mode='bilinear', align_corners=True)
        x_spatial_att = x_channel_att * spatial_map_up
        return x_spatial_att


class AttentionModule(nn.Module):
    """
    注意力模块包装器：根据类型选择使用 WLAM、CBAM、SE 或不使用注意力。
    """

    def __init__(self, channels, att_type='WLAM'):
        super(AttentionModule, self).__init__()
        att_type = att_type.upper() if att_type else 'NONE'
        if att_type == 'WLAM':
            self.att = WLAMBlock(channels)
        elif att_type == 'CBAM':
            self.att = CBAMBlock(channels)
        elif att_type == 'SE':
            self.att = SEBlock(channels)
        else:
            self.att = None

    def forward(self, x):
        if self.att is None:
            return x
        else:
            return self.att(x)


class SeparableConv2d(nn.Module):
    """深度可分离卷积：先按通道分组卷积，再 1x1 点卷积。"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    """包含深度可分离卷积的残差块。"""

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
        # 残差连接
        out = out + residual
        out = self.relu(out)
        return out


class ResidualBranch(nn.Module):
    """用于高频子带的残差分支网络。输入1通道，输出扩展至指定通道数。"""

    def __init__(self, out_channels):
        super(ResidualBranch, self).__init__()
        # 两个残差块处理1通道输入
        self.res1 = ResidualBlock(1)
        self.res2 = ResidualBlock(1)
        # 1x1卷积扩展通道至 out_channels
        self.expand_conv = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.expand_conv(out)
        return out


class DenseBranch(nn.Module):
    """用于高频子带的稠密分支网络。输入1通道，使用小型 Dense Block 提取特征后扩展通道。"""

    def __init__(self, out_channels, growth_rate=8, num_layers=4):
        super(DenseBranch, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        # 定义稠密块中的卷积层和BN层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = 1 + i * growth_rate  # 输入通道 = 原始输入1 + 已产生的增长通道
            out_ch = growth_rate
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_ch))
        # 稠密层连接后的总通道 = 1 + num_layers*growth_rate
        total_ch = 1 + num_layers * growth_rate
        self.compress_conv = nn.Conv2d(total_ch, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = x  # 初始输入 (B,1,H,W)
        concat_feats = [features]  # 用列表收集稠密连接的特征
        for i in range(self.num_layers):
            # 将当前所有特征在通道维拼接后通过卷积
            out = self.convs[i](torch.cat(concat_feats, dim=1))
            out = self.bns[i](out)
            out = self.relu(out)
            concat_feats.append(out)  # 将新产生的特征加入列表
        # 将所有层的特征拼接后，通过1x1卷积压缩至目标通道数
        all_feats = torch.cat(concat_feats, dim=1)
        out = self.compress_conv(all_feats)
        out = self.relu(out)
        return out


class DWTNet(nn.Module):
    """
    改进后的 DWTNet 模型。
    使用 DWT 和 IDWT 代替传统池化和上采样，包含低频全局分支和高频细节分支，并通过 WLAM 注意力融合特征。
    最终输出去噪后的图像以及对应的 RFI 掩码。
    """

    def __init__(self, wavelet_name='db4', levels=3, base_channels=16,
                 subband_enhance_type='dense', use_subband_enhance=True, att_type='WLAM'):
        super(DWTNet, self).__init__()
        self.levels = levels
        self.base_channels = base_channels
        # 小波分解和重构模块
        self.dwt = LearnableDWT2D(wavelet_name=wavelet_name, levels=levels)
        self.idwt = LearnableIDWT2D(wavelet_name=wavelet_name, levels=levels)
        # 展平后子带系数的数量：1 个低频 LL_n + 3*levels 个高频
        self.num_flat_coeffs = 1 + 3 * levels
        # 是否对每个子带使用子网络增强
        self.use_subband_enhance = use_subband_enhance
        if use_subband_enhance:
            # 为每个子带系数准备一个增强分支（Dense 或 Residual）
            self.branches = nn.ModuleList()
            for _ in range(self.num_flat_coeffs):
                if subband_enhance_type.lower() == 'dense':
                    # 稠密分支提取特征后输出 base_channels 通道
                    self.branches.append(DenseBranch(out_channels=base_channels,
                                                     growth_rate=max(4, base_channels // 2),
                                                     num_layers=4))
                else:
                    # 残差分支提取特征后输出 base_channels 通道
                    self.branches.append(ResidualBranch(out_channels=base_channels))
        # 融合后特征通道数
        fusion_channels = (self.num_flat_coeffs * base_channels) if use_subband_enhance else self.num_flat_coeffs
        # 频带融合后的注意力模块（WLAM 实现全局+局部特征融合）
        self.att_band = AttentionModule(channels=fusion_channels, att_type=att_type)
        # 输出卷积层：对融合特征预测每个子带的系数图，以及 RFI 掩码
        self.output_convs = nn.ModuleList([nn.Conv2d(fusion_channels, 1, kernel_size=1)
                                           for _ in range(self.num_flat_coeffs)])
        self.rfi_mask_conv = nn.Conv2d(fusion_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_size = (x.size(2), x.size(3))
        # 1. 对输入进行多层小波分解，获取低频和各级高频系数
        coeffs = self.dwt(x)  # 列表: [LL_n, (LH_n,HL_n,HH_n), ..., (LH_1,HL_1,HH_1)]
        # 2. 将系数结构展开为扁平列表（LL_n，以及每层的 LH, HL, HH）
        flat_coeffs = []
        for c in coeffs:
            if isinstance(c, tuple):
                flat_coeffs.extend(list(c))
            else:
                flat_coeffs.append(c)
        # 3. 并行处理低频与高频子带特征
        enhanced_maps = []
        if self.use_subband_enhance:
            # 对每个子带系数通过对应子网络提取特征
            for coeff, branch in zip(flat_coeffs, self.branches):
                # 确保输入形状为 (B,1,h,w)
                coeff_in = coeff if coeff.dim() == 4 else coeff.unsqueeze(1)
                feat = branch(coeff_in)  # 提取特征，形状 (B, base_channels, h, w)
                # 将每个子带特征上采样到原始尺寸，以便后续融合
                feat_up = F.interpolate(feat, size=original_size, mode='bilinear', align_corners=True)
                enhanced_maps.append(feat_up)
        else:
            # 若不使用子带增强，则仅将系数上采样到原尺寸用于融合
            for coeff in flat_coeffs:
                coeff_map = coeff if coeff.dim() == 4 else coeff.unsqueeze(1)
                coeff_up = F.interpolate(coeff_map, size=original_size, mode='bilinear', align_corners=True)
                enhanced_maps.append(coeff_up)
        # 4. 将所有子带特征在通道维度拼接，并通过注意力模块融合全局+局部信息
        fused = torch.cat(enhanced_maps, dim=1)  # 形状 (B, fusion_channels, H, W)
        fused = self.att_band(fused)  # WLAM 注意力融合：低频特征作为 Key，高频特征作为 Value
        # 5. 根据融合特征预测每个小波子带的输出系数图
        output_coeffs = []
        for i, conv in enumerate(self.output_convs):
            coeff_pred = conv(fused)  # (B,1,h_i,w_i)
            # 确定该系数图对应的小波层级，并调整尺寸匹配
            if i == 0:
                lvl = self.levels  # 第0个为最高层 LL_n
            else:
                # 之后的系数依次对应各级高频，每3个属于同一层级
                lvl = self.levels - ((i - 1) // 3)
            target_h = original_size[0] // (2 ** lvl)
            target_w = original_size[1] // (2 ** lvl)
            if coeff_pred.size(2) != target_h or coeff_pred.size(3) != target_w:
                coeff_pred = F.interpolate(coeff_pred, size=(target_h, target_w),
                                           mode='bilinear', align_corners=True)
            output_coeffs.append(coeff_pred)
        # 6. 将预测的各子带系数重新组织为小波系数结构，逐层进行 IDWT 重建图像
        idwt_coeffs = [output_coeffs[0]]  # LL_n
        idx = 1
        for lvl in range(self.levels, 0, -1):
            # 每级依次取出3个高频系数 (LH, HL, HH)
            idwt_coeffs.append((output_coeffs[idx], output_coeffs[idx + 1], output_coeffs[idx + 2]))
            idx += 3
        # 通过可学习 IDWT 模块重建去噪后的图像
        reconstructed = self.idwt(idwt_coeffs, original_size)
        # 7. 预测 RFI 掩码 (sigmoid 输出0~1范围)
        mask = self.sigmoid(self.rfi_mask_conv(fused))
        # 确保掩码尺寸与输入一致
        if mask.size(2) != original_size[0] or mask.size(3) != original_size[1]:
            mask = F.interpolate(mask, size=original_size, mode='bilinear', align_corners=True)
        # 返回去噪后的重建图像以及对应的 RFI 掩码
        return reconstructed, mask


if __name__ == '__main__':
    model = DWTNet(wavelet_name='db4',
                   levels=3,
                   base_channels=64,
                   subband_enhance_type='dense',  # 'dense' or 'res'
                   use_subband_enhance=True,
                   att_type='WLAM'  # 'WLAM' or 'SE' or 'CBAM'
                   )
    sample = torch.randn(16, 1, 224, 224)
    denoised, rfi_mask = model(sample)
    summary(model, input_size=(16, 1, 224, 224))
    print("Denoised shape:", denoised.shape)
    print("RFI Mask shape:", rfi_mask.shape)
