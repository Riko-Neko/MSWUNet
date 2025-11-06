import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import cv2
import json

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_predictions(image_path, boxes, output_path, title=None):
    """
    可视化预测结果 - 使用竖线标记线条位置
    
    参数:
        image_path: 原始图像路径
        boxes: 检测到的线条 [(class_id, x_min, x_max, confidence), ...]
        output_path: 输出路径
        title: 图像标题
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为灰度用于显示
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    height, width = img_gray.shape
    
    # 显示频谱图
    im = ax.imshow(img_gray, aspect='auto', origin='lower', 
                  cmap='viridis', extent=[0, width, 0, height])
    
    # 获取y轴范围
    y_min_plot, y_max_plot = 0, height
    
    # 绘制检测到的线条 - 使用竖线标记起始和结束位置
    for box in boxes:
        class_id, x_min, x_max, confidence = box
        class_name = '曲线' if class_id == 0 else '直线'
        
        # 选择颜色: 曲线=红色, 直线=绿色
        color = 'red' if class_id == 0 else 'green'
        
        # 绘制起始竖线（连通上下边界）
        ax.plot([x_min, x_min], [y_min_plot, y_max_plot], 
               color=color, linestyle='--', linewidth=2.5, alpha=0.9)
        
        # 绘制结束竖线（连通上下边界）
        ax.plot([x_max, x_max], [y_min_plot, y_max_plot], 
               color=color, linestyle='--', linewidth=2.5, alpha=0.9)
        
        # 在顶部添加标签
        label_text = f'{class_name} ({confidence:.2f})'
        # 将标签放在两条线的中间
        label_x = (x_min + x_max) / 2
        ax.text(label_x, y_max_plot + height * 0.02, 
               label_text, color=color, fontsize=10, 
               fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 标题
    if title is None:
        title = f'检测结果 - 共检测到 {len(boxes)} 条线'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('频率 (像素)', fontsize=12)
    ax.set_ylabel('时间', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='强度')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  预测结果已保存: {output_path}")


def predict_on_dataset(model, data_dir, output_dir, num_samples=30, conf_threshold=0.25):
    """
    在数据集上进行预测
    
    参数:
        model: YOLO模型
        data_dir: 数据目录
        output_dir: 输出目录
        num_samples: 预测样本数
        conf_threshold: 置信度阈值
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n在数据集上进行预测...")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  置信度阈值: {conf_threshold}")
    
    # 获取验证集图像
    val_images_dir = data_dir / 'val' / 'images'
    image_files = sorted(list(val_images_dir.glob('*.jpg')))[:num_samples]
    
    print(f"  验证集大小: {len(image_files)} 个样本")
    
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        # 预测
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=0.45,
            verbose=False,
            save=False
        )
        
        # 解析结果
        boxes_detected = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标 (xyxy格式)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # 图像尺寸
                img_height, img_width = result.orig_shape
                
                # 转换为频率坐标 (x坐标就是频率)
                x_min = int(x1)
                x_max = int(x2)
                
                boxes_detected.append((cls, x_min, x_max, conf))
        
        # 保存结果
        results_summary.append({
            'image': str(image_path.name),
            'num_detected': len(boxes_detected),
            'boxes': [{'class_id': b[0], 'x_min': b[1], 'x_max': b[2], 'confidence': b[3]} 
                     for b in boxes_detected]
        })
        
        # 可视化
        title = f'样本 #{i} - 检测到 {len(boxes_detected)} 条线'
        output_path = output_dir / f'prediction_{i:04d}.png'
        visualize_predictions(image_path, boxes_detected, output_path, title=title)
        
        if (i + 1) % 10 == 0:
            print(f"    已预测 {i + 1}/{len(image_files)} 个样本")
    
    # 保存预测结果
    with open(output_dir / 'predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  预测完成！结果已保存到: {output_dir}")
    
    # 统计
    total_detected = sum(r['num_detected'] for r in results_summary)
    print(f"\n  统计信息:")
    print(f"    总检测线条数: {total_detected}")
    print(f"    平均每图检测: {total_detected / len(results_summary):.2f} 条")


def predict_on_custom_image(model, image_path, output_path, conf_threshold=0.25):
    """
    对自定义图像进行预测
    
    参数:
        model: YOLO模型
        image_path: 输入图像路径
        output_path: 输出路径
        conf_threshold: 置信度阈值
    """
    print(f"\n预测自定义图像: {image_path}")
    print(f"  置信度阈值: {conf_threshold}")
    
    # 预测
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=0.45,
        verbose=True,
        save=False
    )
    
    # 解析结果
    boxes_detected = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            x_min = int(x1)
            x_max = int(x2)
            
            boxes_detected.append((cls, x_min, x_max, conf))
    
    print(f"  检测到 {len(boxes_detected)} 条线:")
    for i, (cls, x_min, x_max, conf) in enumerate(boxes_detected):
        class_name = '曲线' if cls == 0 else '直线'
        print(f"    {i+1}. {class_name} - 频率范围: [{x_min}, {x_max}] - 置信度: {conf:.3f}")
    
    # 可视化
    title = f'检测结果 - 共检测到 {len(boxes_detected)} 条线'
    visualize_predictions(image_path, boxes_detected, output_path, title=title)
    
    print(f"  结果已保存: {output_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("YOLOv8频谱图线条检测与分类")
    print("=" * 70)
    
    # 查找模型
    model_path = input("\n请输入模型路径 (默认: yolo_models/spectrogram_detector/weights/best.pt): ").strip()
    if not model_path:
        model_path = 'yolo_models/spectrogram_detector/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        
        # 查找可用模型
        model_dirs = list(Path('yolo_models').glob('*/weights/best.pt'))
        if model_dirs:
            print(f"\n可用的模型:")
            for i, model_dir in enumerate(model_dirs, 1):
                print(f"  {i}. {model_dir}")
        return
    
    print(f"\n加载模型...")
    model = YOLO(model_path)
    print(f"模型已加载: {model_path}")
    
    # 设置置信度阈值
    conf_threshold = input("\n请输入置信度阈值 (默认: 0.25): ").strip()
    conf_threshold = float(conf_threshold) if conf_threshold else 0.25
    
    # 选择预测模式
    print("\n请选择预测模式:")
    print("  1. 在验证集上预测")
    print("  2. 预测自定义图像")
    
    mode = input("请输入选项 (1 或 2): ").strip()
    
    if mode == '1':
        # 验证集预测
        data_dirs = sorted(Path('yolo_data').glob('dataset_*'))
        if not data_dirs:
            print("错误: 未找到数据集")
            return
        
        data_dir = data_dirs[-1]
        print(f"\n使用数据集: {data_dir}")
        
        output_dir = Path('predictions_yolo') / data_dir.name
        
        num_samples = input("\n预测多少个样本? (默认: 30): ").strip()
        num_samples = int(num_samples) if num_samples else 30
        
        predict_on_dataset(model, data_dir, output_dir, num_samples=num_samples, 
                          conf_threshold=conf_threshold)
        
    elif mode == '2':
        # 自定义图像预测
        image_path = input("\n请输入图像路径: ").strip()
        if not Path(image_path).exists():
            print(f"错误: 图像文件不存在: {image_path}")
            return
        
        output_path = input("请输入输出路径 (默认: prediction_yolo_result.png): ").strip()
        if not output_path:
            output_path = 'prediction_yolo_result.png'
        
        predict_on_custom_image(model, image_path, output_path, conf_threshold=conf_threshold)
    
    else:
        print("无效的选项")
        return
    
    print("\n" + "=" * 70)
    print("预测完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

