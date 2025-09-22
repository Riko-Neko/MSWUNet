import numpy as np
import pywt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import interpolate


# 创建 64x64 图像，波纹背景，正弦曲线从 (16,0) 到 (47,63)
def create_sine_curve_ripple_image(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    period = 10
    for y in range(size):
        for x in range(size):
            img[y, x] = 100 * (np.sin(2 * np.pi * (x + y) / period) + 1) / 2
    for x in range(16, 48):
        t = (x - 16) / 31
        y = int(63 * (1 - np.cos(np.pi * t)) / 2)
        if 0 <= y < size:
            img[y, x] = 255
    return img


# 双线性插值
def bilinear_interpolation(image, new_height, new_width):
    img_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    interp_img = F.interpolate(img_torch, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return interp_img.squeeze().numpy()


# 单模式子图
def method_subplot(image, method, ax, title):
    coeffs = pywt.dwt2(image, wavelet='haar', mode='symmetric')
    _, (_, _, cD) = coeffs
    ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title(title)
    ax[0].axis('off')
    ax[1].imshow(cD, cmap='gray')
    ax[1].set_title(f'{title} HH Subband')
    ax[1].axis('off')


# 可视化比较
def visualize_dimension_alignment():
    size = 64
    # 创建原始图像并裁剪到奇数尺寸 63x65
    original_img = create_sine_curve_ripple_image(size)[:63, :65]

    # 方法列表
    methods = [
        ('Bilinear Interp', bilinear_interpolation(original_img, 64, 64)),
        ('Symmetric + Interp', bilinear_interpolation(
            pywt.pad(original_img, ((0, 1), (0, 1)), mode='symmetric'), 64, 64)),
        ('Symmetric Pad', pywt.pad(original_img, ((0, 1), (0, 1)), mode='symmetric')),
        ('Zero Pad', pywt.pad(original_img, ((0, 1), (0, 1)), mode='constant')),
        ('Crop', original_img[:62, :64])
    ]

    # 设置图形
    fig, axes = plt.subplots(3, len(methods), figsize=(2.5 * len(methods), 7))

    # 显示原始图像
    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original (63x65)')
    axes[0, 0].axis('off')
    for i in range(1, len(methods)):
        axes[0, i].axis('off')

    # 绘制各方法子图
    for idx, (title, img) in enumerate(methods, 0):
        method_subplot(img, title, [axes[1, idx], axes[2, idx]], f'{title} ({img.shape[0]}x{img.shape[1]})')

    plt.tight_layout()
    plt.show()


# 运行可视化
if __name__ == "__main__":
    visualize_dimension_alignment()