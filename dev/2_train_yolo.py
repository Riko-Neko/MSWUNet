from ultralytics import YOLO
from pathlib import Path
import argparse


def main():
    """主函数"""
    # 查找最新的数据集
    data_dirs = sorted(Path('yolo_data').glob('dataset_*'))
    if not data_dirs:
        print("错误: 未找到数据集")
        return
    
    data_dir = data_dirs[-1]
    yaml_file = data_dir / 'data.yaml'
    
    print(f"\n数据集: {data_dir}")
    print(f"配置文件: {yaml_file}")
    
    # 选择YOLO模型
    print("\n请选择YOLO模型:")
    print("  1. YOLOv8n (nano - 最快，精度较低)")
    print("  2. YOLOv8s (small - 平衡)")
    print("  3. YOLOv8m (medium - 推荐) ⭐")
    print("  4. YOLOv8l (large - 精度高，较慢)")
    print("  5. YOLOv8x (extra large - 最高精度，最慢)")
    
    choice = input("\n请输入选项 (1-5，默认3): ").strip()
    if not choice:
        choice = '3'
    
    model_map = {
        '1': 'yolov8n.pt',
        '2': 'yolov8s.pt',
        '3': 'yolov8m.pt',
        '4': 'yolov8l.pt',
        '5': 'yolov8x.pt'
    }
    
    model_name = model_map.get(choice, 'yolov8m.pt')
    print(f"\n使用模型: {model_name}")
    
    # 加载预训练模型
    print(f"\n加载预训练模型...")
    model = YOLO(model_name)
    
    # 训练参数
    print(f"\n开始训练...")
    print(f"  Epochs: 100")
    print(f"  Batch size: 16")
    print(f"  Image size: 640")
    
    # 训练模型
    results = model.train(
        data=str(yaml_file),
        epochs=50,
        batch=16,
        imgsz=640,
        patience=20,  # Early stopping
        save=True,
        device=0,  # 使用GPU，如果没有GPU会自动切换到CPU
        workers=4,
        project='yolo_models',
        name='spectrogram_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=True,
        verbose=True
    )
    
    print(f"\n" + "=" * 70)
    print(f"训练完成！")
    print(f"模型保存在: yolo_models/spectrogram_detector")
    print(f"最佳模型: yolo_models/spectrogram_detector/weights/best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()

