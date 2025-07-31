import torch
import torch.nn as nn
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class RFI_DRUnet(nn.Module):
    def __init__(self):
        super(RFI_DRUnet, self).__init__()

        # 头部
        self.head = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # 下采样模块1
        self.down1_res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.down1_strided = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)

        # 下采样模块2
        self.down2_res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.down2_strided = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)

        # 下采样模块3
        self.down3_res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.down3_strided = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)

        # 瓶颈
        self.bottleneck_res_blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # 上采样模块3
        self.up3_transposed = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3_res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # 上采样模块2
        self.up2_transposed = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # 上采样模块1
        self.up1_transposed = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # 尾部
        self.tail = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 头部
        x = self.head(x)

        # 下采样模块1
        down1 = self.down1_res_blocks(x)  # 保存用于跳跃连接
        x = self.down1_strided(down1)

        # 下采样模块2
        down2 = self.down2_res_blocks(x)  # 保存用于跳跃连接
        x = self.down2_strided(down2)

        # 下采样模块3
        down3 = self.down3_res_blocks(x)  # 保存用于跳跃连接
        x = self.down3_strided(down3)

        # 瓶颈
        x = self.bottleneck_res_blocks(x)

        # 上采样模块3
        x = self.up3_transposed(x)
        x = x + down3  # 通过加法实现跳跃连接
        x = self.up3_res_blocks(x)

        # 上采样模块2
        x = self.up2_transposed(x)
        x = x + down2  # 通过加法实现跳跃连接
        x = self.up2_res_blocks(x)

        # 上采样模块1
        x = self.up1_transposed(x)
        x = x + down1  # 通过加法实现跳跃连接
        x = self.up1_res_blocks(x)

        # 尾部
        x = self.tail(x)

        return x


# 示例用法
if __name__ == "__main__":
    model = RFI_DRUnet()
    input_tensor = torch.randn(1, 1, 64, 64)  # 示例输入：batch_size=1, 通道=1, 高=64, 宽=64
    summary(model, input_size=(1, 1, 64, 64))  # 打印模型结构
    output = model(input_tensor)
    print(output.shape)  # 预期输出形状：torch.Size([1, 1, 64, 64])