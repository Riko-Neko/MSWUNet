from math import sqrt
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        backbone_coord_atts = False
        coord_att_reduction = 32
        strides = [(2, 2), (3, 2), (2, 2), (3, 2)]
        f_reduction = [s[1] for s in strides]
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

        self.freq_dim = self._get_F_dim(self.fchans, f_reduction)  # compute F'
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

    def _get_F_dim(self, fchans: int, f_list: list) -> int:
        f = fchans
        for s in f_list:
            f = (f + s - 1) // s
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


class FreqBasicBlock(nn.Module):
    """
    Simple residual 3x3 block (ResNet-like).
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class FreqHRModule(nn.Module):
    """
    High-Resolution style module specialized for frequency axis.

    - Multiple branches: different frequency resolutions (F, F/2, F/4, ...)
    - Only downsample along frequency (W axis), keep time (H) resolution.

    Inputs:
        x: list[Tensor], len = num_branches
           x[i]: (B, C_i, H, F / 2^i)

    Outputs:
        list[Tensor] with the same spatial layout.
    """

    def __init__(self, num_branches: int, block, num_blocks: List[int], num_inchannels: List[int],
                 num_channels: List[int], multi_scale_output: bool = True, norm_layer=nn.BatchNorm2d, ):
        super().__init__()
        assert num_branches == len(num_blocks)
        assert num_branches == len(num_inchannels) == len(num_channels)

        self.num_branches = num_branches
        self.block = block
        self.num_inchannels = list(num_inchannels)
        self.multi_scale_output = multi_scale_output
        self.norm_layer = norm_layer

        self.branches = self._build_branches(
            num_branches, num_blocks, num_channels, norm_layer
        )
        self.fuse_layers = self._build_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index: int, num_blocks: int, num_channels: int, norm_layer,
                         stride: int = 1, ) -> nn.Sequential:
        downsample = None
        in_ch = self.num_inchannels[branch_index]
        out_ch = num_channels * self.block.expansion

        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                                       norm_layer(out_ch), )

        layers = []
        layers.append(self.block(inplanes=in_ch, planes=num_channels, stride=stride, downsample=downsample,
                                 norm_layer=norm_layer, ))
        in_ch = out_ch
        for _ in range(1, num_blocks):
            layers.append(self.block(inplanes=in_ch, planes=num_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _build_branches(self, num_branches: int, num_blocks: List[int], num_channels: List[int],
                        norm_layer) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(branch_index=i, num_blocks=num_blocks[i], num_channels=num_channels[i],
                                      norm_layer=norm_layer))
            self.num_inchannels[i] = num_channels[i] * self.block.expansion
        return nn.ModuleList(branches)

    def _build_fuse_layers(self) -> nn.ModuleList:
        """
        For each target branch i, fuse from all source branches j:
        - If j == i: identity
        - If j < i : downsample along freq (W) with (1,2) stride repeated
        - If j > i : 1x1 conv for channel align + bilinear upsample along freq
        """
        if self.num_branches == 1:
            return nn.ModuleList([])

        fuse_layers = []
        for target in range(self.num_branches if self.multi_scale_output else 1):
            layer_ops = nn.ModuleList()
            for source in range(self.num_branches):
                if source == target:
                    layer_ops.append(nn.Identity())
                    continue

                in_ch = self.num_inchannels[source]
                out_ch = self.num_inchannels[target]

                if source > target:
                    # source has lower freq resolution, will be upsampled later
                    ops = [nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False), self.norm_layer(out_ch), ]
                    layer_ops.append(nn.Sequential(*ops))
                else:
                    # source has higher freq resolution, need to downsample
                    downs = []
                    for k in range(target - source):
                        s = (1, 2)
                        in_c = in_ch if k == 0 else out_ch
                        downs.append(nn.Sequential(
                            nn.Conv2d(in_c, out_ch, kernel_size=(1, 3), stride=s, padding=(0, 1), bias=False),
                            self.norm_layer(out_ch)))
                    layer_ops.append(nn.Sequential(*downs))
            fuse_layers.append(layer_ops)
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(xs) == self.num_branches

        # Branch-specific blocks
        xs = [branch(x) for branch, x in zip(self.branches, xs)]

        if self.num_branches == 1:
            return xs

        x_fused: List[torch.Tensor] = []
        num_outputs = self.num_branches if self.multi_scale_output else 1

        for i in range(num_outputs):
            y = None
            for j in range(self.num_branches):
                xj = xs[j]
                op = self.fuse_layers[i][j]
                if not isinstance(op, nn.Identity):
                    xj = op(xj)

                # if source has lower freq resolution (j > i), upsample along freq
                if j > i:
                    H_i = xs[i].shape[-2]
                    W_i = xs[i].shape[-1]
                    xj = F.interpolate(xj, size=(H_i, W_i), mode="bilinear", align_corners=False, )

                if y is None:
                    y = xj
                else:
                    y = y + xj

            x_fused.append(self.relu(y))

        return x_fused


class HRFreqRegressionDetector(nn.Module):
    """
    High-resolution + attention-based regressive head for 1D-frequency object detection.

    Key ideas:
    - Keep a high-resolution frequency branch (no explicit downsampling along F on branch 0).
    - Use multiple lower-resolution branches (F/2, F/4, ...) only to provide context.
    - Fuse them HRNet-style along the frequency axis.
    - Then:
        * Pool only along time (T) -> keep all frequency bins.
        * Treat frequency bins as a sequence of tokens (length = F).
        * Use N learnable queries with multi-head attention to read out N object embeddings.
        * Each embedding -> MLP -> [f_start, f_end, class_logits, confidence].

    Compared to the previous head:
        - No AdaptiveAvgPool2d over frequency.
        - Frequency resolution is preserved up to the attention module.
    """

    def __init__(self, fchans: int = 1024, in_channels: int = 1, N: int = 5, num_classes: int = 2,
                 base_channels: int = 64, num_branches: int = 3, num_stages: int = 2, bottleneck_channels: int = 128,
                 coord_att_reduction: int = 32, neck_dim_T: int = 2, dropout: float = 0.0, num_heads: int = 4):
        super().__init__()
        assert 1 <= num_branches <= 4, "1-4 branches supported."
        assert bottleneck_channels % num_heads == 0, "bottleneck_channels must be divisible by num_heads."

        self.N = N
        self.num_classes = num_classes
        self.fchans = fchans
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.num_stages = num_stages
        self.bottleneck = bottleneck_channels
        self.neck_dim_T = neck_dim_T

        norm_layer = nn.BatchNorm2d

        # ---- Stem: light conv stack to unify input channels ----
        stem_layers = [nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                       norm_layer(base_channels), nn.ReLU(inplace=True)]
        if dropout > 0:
            stem_layers.append(nn.Dropout2d(p=dropout))
        stem_layers.extend(
            [nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
             norm_layer(base_channels), nn.ReLU(inplace=True)])
        self.stem = nn.Sequential(*stem_layers)

        # ---- multi-resolution branches ----
        branch_channels = [base_channels * (2 ** i) for i in range(num_branches)]
        self.branch_channels = branch_channels
        self.initial_branches = self._make_initial_branches(base_channels, branch_channels, norm_layer, dropout)

        # ---- HR-style stages along frequency axis ----
        from typing import List as _List  # for type clarity

        self.hr_stages = nn.ModuleList()
        num_inchannels: _List[int] = list(branch_channels)
        for _ in range(num_stages):
            num_blocks = [2] * num_branches
            stage = FreqHRModule(num_branches=num_branches, block=FreqBasicBlock, num_blocks=num_blocks,
                                 num_inchannels=num_inchannels, num_channels=branch_channels, multi_scale_output=True,
                                 norm_layer=norm_layer)
            self.hr_stages.append(stage)
            num_inchannels = stage.get_num_inchannels()

        # ---- Bottleneck + CoordAtt ----
        final_in_channels = sum(num_inchannels)
        self.bottleneck_conv = nn.Conv2d(final_in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bottleneck_bn = norm_layer(bottleneck_channels)
        self.bottleneck_relu = nn.ReLU(inplace=True)
        self.coord_att = CoordAtt(bottleneck_channels, bottleneck_channels, coord_att_reduction)
        self.time_freq_pool = nn.AdaptiveAvgPool2d((neck_dim_T, fchans))

        # ---- DETR-style attention head (along frequency axis) ----
        # Frequency tokens: (F, B, C) with C = bottleneck_channels
        # Queries: N learnable embeddings
        self.num_heads = num_heads
        self.embed_dim = bottleneck_channels
        self.query_embed = nn.Embedding(N, self.embed_dim)
        self.freq_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=False)

        # Small MLP from object embedding -> output vector
        out_dim = 2 + num_classes + 1
        hidden_dim = max(self.embed_dim * 2, 128)
        self.obj_mlp = nn.Sequential(nn.Linear(self.embed_dim, hidden_dim), nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, out_dim))

        self._init_weights()

    def _make_initial_branches(self, in_ch: int, branch_channels: List[int], norm_layer,
                               dropout: float) -> nn.ModuleList:
        """
        Build initial multi-resolution branches from the stem output.

        Branch 0: high-res, no freq downsampling.
        Branch i>0: downsample along freq i times with stride (1, 2).
        """
        branches = nn.ModuleList()
        num_branches = len(branch_channels)

        for i in range(num_branches):
            ops = []
            curr_in = in_ch

            if i == 0:
                # Just align channels if needed
                if branch_channels[0] != in_ch:
                    ops.append(nn.Conv2d(curr_in, branch_channels[0], kernel_size=1, stride=1, bias=False))
                    ops.append(norm_layer(branch_channels[0]))
                    ops.append(nn.ReLU(inplace=True))
                    if dropout > 0:
                        ops.append(nn.Dropout2d(p=dropout))
                # no spatial downsampling
            else:
                # Branch i: freq downsampling i times
                for k in range(i):
                    out_c = branch_channels[i]
                    ops.append(nn.Conv2d(curr_in, out_c, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False))
                    ops.append(norm_layer(out_c))
                    ops.append(nn.ReLU(inplace=True))
                    if dropout > 0:
                        ops.append(nn.Dropout2d(p=dropout))
                    curr_in = out_c

            if len(ops) == 0:
                branches.append(nn.Identity())
            else:
                branches.append(nn.Sequential(*ops))

        return branches

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
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, T, F)

        Returns:
            dict with raw logits:
                - f_start: (B, N)
                - f_end: (B, N)
                - class_logits: (B, N, num_classes)
                - confidence: (B, N)
        """
        B = x.size(0)
        x = self.stem(x)  # (B, base_channels, T, F)

        branch_feats = [branch(x) for branch in self.initial_branches]
        # HR-style stages along frequency axis
        for stage in self.hr_stages:
            branch_feats = stage(branch_feats)

        # Upsample all branches to the highest frequency resolution (branch 0) and concat
        high_T, high_F = branch_feats[0].shape[-2:]
        assert high_F == self.fchans, f"Expected freq {self.fchans}, got {high_F}"

        up_feats = []
        for i, feat in enumerate(branch_feats):
            if i == 0:
                up_feats.append(feat)
            else:
                up_feats.append(F.interpolate(feat, size=(high_T, high_F), mode="bilinear", align_corners=False, ))
        feat = torch.cat(up_feats, dim=1)  # (B, sum(C_i), T, F)

        # Bottleneck + CoordAtt
        feat = self.bottleneck_relu(self.bottleneck_bn(self.bottleneck_conv(feat)))
        feat = self.coord_att(feat)  # (B, bottleneck, T, F)

        # (B, C, T, F) -> (B, C, T', F), T' = neck_dim_T
        feat_tf = self.time_freq_pool(feat)
        # Treat each (time, freq) pair as a token:
        feat_tf = feat_tf.permute(0, 2, 3, 1).contiguous()  # (B, C, T', F) -> (B, T', F, C)
        feat_seq = feat_tf.flatten(start_dim=1, end_dim=2)  # (B, T'*F, C)
        feat_seq = feat_seq.permute(1, 0, 2).contiguous()  # MultiheadAttention: (T'*F, B, C)
        # Queries: (N, C) -> (N, B, C)
        query_embed = self.query_embed.weight  # (N, C)
        query = query_embed.unsqueeze(1).expand(-1, B, -1)  # (N, B, C)
        # Multi-head attention over frequency tokens
        # query: (N, B, C), key/value: (T'*F, B, C), output: (N, B, C)
        hs, _ = self.freq_attn(query=query, key=feat_seq, value=feat_seq)
        hs = hs.permute(1, 0, 2).contiguous()  # (N, B, C) -> (B, N, C)

        # Each object embedding -> output vector
        out_all = self.obj_mlp(hs)  # (B, N, 2 + num_classes + 1)

        # Split fields
        f_start = out_all[..., 0]  # (B, N)
        f_end = out_all[..., 1]  # (B, N)
        class_logits = out_all[..., 2:2 + self.num_classes]  # (B, N, num_classes)
        confidence = out_all[..., -1]  # (B, N)

        return {"f_start": f_start, "f_end": f_end, "class_logits": class_logits, "confidence": confidence}
