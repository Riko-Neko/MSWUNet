import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import interpolate

# 创建一个 64x64 图像，波纹背景，包含从 (16,0) 到 (47,63) 的正弦曲线
def create_sine_curve_ripple_image(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    # 生成波纹背景
    period = 10
    for y in range(size):
        for x in range(size):
            img[y, x] = 100 * (np.sin(2 * np.pi * (x + y) / period) + 1) / 2  # [0, 100]
    # 绘制曲线 y = 63 * (1 - cos(pi * (x-16) / 31)) / 2
    for x in range(16, 48):
        t = (x - 16) / 31
        y = int(63 * (1 - np.cos(np.pi * t)) / 2)
        if 0 <= y < size:
            img[y, x] = 255  # 曲线值为 255
    return img

# 双线性插值到新尺寸
def bilinear_interpolation(image, new_height, new_width):
    y = np.linspace(0, image.shape[0]-1, new_height)
    x = np.linspace(0, image.shape[1]-1, new_width)
    x_grid, y_grid = np.meshgrid(x, y)
    interp_func = interpolate.RegularGridInterpolator(
        (np.arange(image.shape[0]), np.arange(image.shape[1])),
        image,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    return interp_func((y_grid, x_grid))

# 单模式填充和小波变换子图
def boundary_mode_subplot(image, mode, ax, symw=False):
    pad_width = 8
    padded = pywt.pad(image, pad_widths=((pad_width, pad_width), (pad_width, pad_width)), mode=mode)
    coeffs = pywt.dwt2(image, wavelet='haar', mode=mode)
    _, (_, _, cD) = coeffs
    ax[0].imshow(padded, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title(f'{mode} Padded{" (symw)" if symw else ""}')
    ax[0].axis('off')
    ax[1].imshow(cD, cmap='gray')
    ax[1].set_title(f'{mode} HH Subband')
    ax[1].axis('off')

# 可视化填充和插值
def visualize_padding_vs_interpolation():
    size = 64
    modes = ['zero', 'constant', 'symmetric', 'reflect', 'periodic',
             'antisymmetric', 'antireflect', 'periodization', 'smooth']
    symw_flags = [False, False, False, True, False, False, True, False, False]

    # 创建原始图像
    original_img = create_sine_curve_ripple_image(size)

    # 双线性插值到 65x66
    interp_img = bilinear_interpolation(original_img, 65, 66)
    interp_coeffs = pywt.dwt2(interp_img, wavelet='haar', mode='symmetric')  # 使用 symmetric 避免额外边界效应
    _, (_, _, interp_cD) = interp_coeffs

    # 设置图形
    fig, axes = plt.subplots(3, len(modes) + 1, figsize=(2.5 * (len(modes) + 1), 7))

    # 显示原始图像
    axes[0, 0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示插值图像
    axes[1, 0].imshow(interp_img, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Bilinear Interp (65x66)')
    axes[1, 0].axis('off')
    axes[2, 0].imshow(interp_cD, cmap='gray')
    axes[2, 0].set_title('Bilinear HH Subband')
    axes[2, 0].axis('off')

    # 填充空位
    for i in range(1, len(modes) + 1):
        axes[0, i].axis('off')

    # 对每种填充模式绘制子图
    for idx, (mode, symw) in enumerate(zip(modes, symw_flags), 1):
        boundary_mode_subplot(original_img, mode, [axes[1, idx], axes[2, idx]], symw=symw)

    plt.tight_layout()
    plt.show()

# 运行可视化
if __name__ == "__main__":
    visualize_padding_vs_interpolation()