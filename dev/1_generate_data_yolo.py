import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import cv2

from gen.SETIgen import sim_dynamic_spec_seti

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def generate_spectrogram_with_lines(sample_idx, num_lines=None, fchans=256, tchans=64):
    """
    生成包含多条线的频谱图样本
    
    返回:
        image: 图像数据 (tchans, fchans)
        yolo_labels: YOLO格式标注列表 [(class_id, x_center, y_center, width, height), ...]
        metadata: 元数据字典
    """
    # 参数配置
    df = 7.5
    dt = 1.0
    fch1 = 1.42e9
    ascending = True
    
    # 随机确定线的数量（0-3条）
    if num_lines is None:
        num_lines = random.choices([0, 1, 2, 3], weights=[0.05, 0.35, 0.4, 0.2])[0]
    
    signals = []
    boxes_info = []
    
    # 为了避免信号重叠，将频率空间分成若干区域
    if num_lines > 0:
        freq_regions = []
        region_size = fchans // (num_lines + 1)
        for i in range(num_lines):
            region_start = i * region_size + region_size // 4
            region_end = (i + 1) * region_size - region_size // 4
            freq_regions.append((region_start, region_end))
    
    for i in range(num_lines):
        # 随机选择类型
        signal_type = random.choice(['straight', 'curved'])
        
        if signal_type == 'straight':
            path_type = 'constant'
            class_id = 1  # 直线
            drift_rate = random.uniform(-2.5, 2.5)
            if abs(drift_rate) < 0.8:
                drift_rate = 0.8 if drift_rate >= 0 else -0.8
        else:
            path_type = random.choice(['sine', 'squared'])
            class_id = 0  # 曲线
            drift_rate = random.uniform(-1.5, 1.5)
        
        # 从分配的频率区域中选择起始位置
        region_start, region_end = freq_regions[i]
        f_index = random.randint(region_start, region_end)
        
        # 信号参数 - 使用更高的SNR和宽度
        snr = random.uniform(40.0, 80.0)
        width = random.uniform(10.0, 20.0)
        
        sig = {
            'f_index': f_index,
            'drift_rate': drift_rate,
            'snr': snr,
            'width': width,
            'path': path_type,
            't_profile': 'constant',
            'f_profile': 'gaussian'
        }
        
        # 添加路径特定参数
        if path_type == 'sine':
            total_time = tchans * dt
            sig['period'] = random.uniform(0.4 * total_time, 1.5 * total_time)
            sig['amplitude'] = random.uniform(60, 180)
        elif path_type == 'squared':
            sig['squared_drift'] = drift_rate * random.uniform(4.e-4, 10.e-4)
        
        signals.append(sig)
        boxes_info.append({
            'class_id': class_id,
            'signal_type': signal_type,
            'path_type': path_type,
            'f_index': f_index,
            'drift_rate': drift_rate
        })
    
    # 减少噪声
    noise_std = random.uniform(0.005, 0.015)
    noise_mean = random.uniform(0.0, 0.005)
    
    # 减少RFI
    rfi_params = {
        'NBC': np.random.randint(0, 1),
        'NBC_amp': np.random.uniform(0.3, 1.0),
        'NBT': np.random.randint(0, 2),
        'NBT_amp': np.random.uniform(0.3, 1.0),
        'BBT': 0,
        'BBT_amp': 0,
        'LowDrift': 0,
        'LowDrift_amp_factor': 0,
        'LowDrift_width': 0
    }
    
    # 生成频谱
    signal_spec, clean_spec, noisy_spec, rfi_mask, freq_info = sim_dynamic_spec_seti(
        fchans=fchans,
        tchans=tchans,
        df=df,
        dt=dt,
        fch1=fch1,
        ascending=ascending,
        signals=signals if num_lines > 0 else None,
        noise_x_mean=noise_mean,
        noise_x_std=noise_std,
        mode='test',
        noise_type='normal',
        rfi_params=rfi_params,
        seed=None,
        plot=False
    )
    
    # 计算YOLO格式标注
    yolo_labels = []
    if num_lines > 0:
        N, classes, f_starts, f_stops = freq_info
        
        # 确保是列表
        if not isinstance(f_starts, list):
            f_starts = [f_starts]
        if not isinstance(f_stops, list):
            f_stops = [f_stops]
        if not isinstance(classes, list):
            classes = [classes]
        
        for j in range(N):
            f_start = int(np.clip(f_starts[j], 0, fchans - 1))
            f_stop = int(np.clip(f_stops[j], 0, fchans - 1))
            
            # 确保 f_start < f_stop
            if f_start > f_stop:
                f_start, f_stop = f_stop, f_start
            
            class_id = int(classes[j]) if hasattr(classes[j], 'item') else int(classes[j])
            
            # YOLO格式: class_id x_center y_center width height (归一化到0-1)
            x_center = (f_start + f_stop) / 2.0 / fchans
            y_center = 0.5  # 时间维度中心
            box_width = (f_stop - f_start) / fchans
            box_height = 1.0  # 整个时间范围
            
            yolo_labels.append({
                'class_id': class_id,
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(box_width),
                'height': float(box_height),
                'x_min': f_start,
                'x_max': f_stop
            })
    
    # 元数据
    metadata = {
        'sample_idx': int(sample_idx),
        'num_lines': int(num_lines),
        'yolo_labels': yolo_labels,
        'signals_info': boxes_info,
        'image_shape': {'tchans': tchans, 'fchans': fchans}
    }
    
    return noisy_spec, yolo_labels, metadata


def save_yolo_format(image, labels, image_path, label_path):
    """
    保存YOLO格式的数据
    
    参数:
        image: 图像数据
        labels: YOLO标注列表
        image_path: 图像保存路径
        label_path: 标注保存路径
    """
    # 归一化图像到0-255
    image_norm = image.copy()
    image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min() + 1e-8)
    image_norm = (image_norm * 255).astype(np.uint8)
    
    # 转换为3通道图像（YOLO需要）
    image_rgb = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2RGB)
    
    # 保存图像
    cv2.imwrite(str(image_path), image_rgb)
    
    # 保存YOLO标注
    with open(label_path, 'w') as f:
        for label in labels:
            # YOLO格式: class_id x_center y_center width height
            f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} "
                   f"{label['width']:.6f} {label['height']:.6f}\n")


def main():
    """主函数：生成训练和测试数据"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'yolo_data/dataset_{timestamp}')
    
    # 创建YOLO数据集目录结构
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("生成YOLO格式的天文频谱图数据...")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    # 数据集参数
    fchans = 256
    tchans = 64
    
    # 生成训练集
    num_train = 2000
    print(f"\n[1/2] 生成训练集: {num_train} 个样本")
    
    train_metadata = []
    for i in range(num_train):
        image, labels, metadata = generate_spectrogram_with_lines(
            sample_idx=i,
            num_lines=None,
            fchans=fchans,
            tchans=tchans
        )
        
        # 保存YOLO格式
        image_path = output_dir / 'train' / 'images' / f'{i:06d}.jpg'
        label_path = output_dir / 'train' / 'labels' / f'{i:06d}.txt'
        save_yolo_format(image, labels, image_path, label_path)
        
        train_metadata.append(metadata)
        
        if (i + 1) % 200 == 0:
            print(f"  已生成 {i + 1}/{num_train} 训练样本")
    
    # 生成验证集
    num_val = 300
    print(f"\n[2/2] 生成验证集: {num_val} 个样本")
    
    val_metadata = []
    for i in range(num_val):
        image, labels, metadata = generate_spectrogram_with_lines(
            sample_idx=i,
            num_lines=None,
            fchans=fchans,
            tchans=tchans
        )
        
        # 保存YOLO格式 - 使用不同的编号避免与训练集重复
        val_idx = num_train + i  # 从训练集结束的编号继续
        image_path = output_dir / 'val' / 'images' / f'{val_idx:06d}.jpg'
        label_path = output_dir / 'val' / 'labels' / f'{val_idx:06d}.txt'
        save_yolo_format(image, labels, image_path, label_path)
        
        val_metadata.append(metadata)
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{num_val} 验证样本")
    
    # 保存元数据
    with open(output_dir / 'train_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(train_metadata, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'val_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(val_metadata, f, indent=2, ensure_ascii=False)
    
    # 创建YOLO配置文件
    yaml_content = f"""# 天文频谱图数据集配置
path: {output_dir.absolute()}  # 数据集根目录
train: train/images  # 训练集图像目录
val: val/images  # 验证集图像目录

# 类别
nc: 2  # 类别数
names: ['曲线', '直线']  # 类别名称
"""
    
    with open(output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # 统计信息
    train_line_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    train_class_dist = {'curved': 0, 'straight': 0}
    for metadata in train_metadata:
        num_lines = metadata['num_lines']
        train_line_dist[num_lines] += 1
        for label in metadata['yolo_labels']:
            if label['class_id'] == 0:
                train_class_dist['curved'] += 1
            else:
                train_class_dist['straight'] += 1
    
    val_line_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    val_class_dist = {'curved': 0, 'straight': 0}
    for metadata in val_metadata:
        num_lines = metadata['num_lines']
        val_line_dist[num_lines] += 1
        for label in metadata['yolo_labels']:
            if label['class_id'] == 0:
                val_class_dist['curved'] += 1
            else:
                val_class_dist['straight'] += 1
    
    print(f"\n数据集统计:")
    print(f"  训练集: {num_train} 样本")
    print(f"    - 0条线: {train_line_dist[0]}")
    print(f"    - 1条线: {train_line_dist[1]}")
    print(f"    - 2条线: {train_line_dist[2]}")
    print(f"    - 3条线: {train_line_dist[3]}")
    print(f"    - 曲线总数: {train_class_dist['curved']}")
    print(f"    - 直线总数: {train_class_dist['straight']}")
    
    print(f"\n  验证集: {num_val} 样本")
    print(f"    - 0条线: {val_line_dist[0]}")
    print(f"    - 1条线: {val_line_dist[1]}")
    print(f"    - 2条线: {val_line_dist[2]}")
    print(f"    - 3条线: {val_line_dist[3]}")
    print(f"    - 曲线总数: {val_class_dist['curved']}")
    print(f"    - 直线总数: {val_class_dist['straight']}")
    
    print(f"\n" + "=" * 70)
    print(f"数据生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"YOLO配置文件: {output_dir / 'data.yaml'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

