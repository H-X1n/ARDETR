from ultralytics import RTDETR

if __name__ == '__main__':
    # 加载模型配置文件
    # 可选模型（位于 cfg/models/rt-detr/ 目录）：
    # - rtdetr-l.yaml, rtdetr-x.yaml          # 标准 RT-DETR 模型
    # - rtdetr-resnet50.yaml, rtdetr-resnet101.yaml  # 基于 ResNet 的模型
    # - ARrtdetr-l.yaml, ARrtdetr-x.yaml      # 自定义 ARrtdetr 模型
    # - ARrtdetr8.yaml, ARrtdetr10.yaml, ARrtdetr12.yaml  # 不同层数的 ARrtdetr
    # - BARrtdetr-l.yaml, BARrtdetr-x.yaml    # 自定义 BARrtdetr 模型
    # - BARrtdetr-l2.yaml, BARrtdetr-l3.yaml  # 不同配置的 BARrtdetr
    
    model = RTDETR("cfg/models/rt-detr/ARrtdetr-l.yaml")
    
    # 显示模型信息
    model.info()
    
    # 开始训练
    results = model.train(
        data="data.yaml",           # 数据集配置文件路径
        epochs=300,                 # 训练轮数
        imgsz=640,                  # 输入图像尺寸
        amp=False,                  # 是否使用自动混合精度
        batch=6,                    # 批次大小（根据显存调整）
        optimizer="AdamW",          # 优化器
        lr0=1e-4,                   # 初始学习率
        weight_decay=0.01,          # 权重衰减
        warmup_epochs=2.0,          # 预热轮数
        mosaic=0.0,                 # Mosaic 数据增强（0.0 表示关闭）
        mixup=0.0,                  # Mixup 数据增强（0.0 表示关闭）
        deterministic=False,        # 是否使用确定性算法
        device="0",                 # 使用的 GPU 设备（"0" 表示单卡，"0,1" 表示多卡）
        cache="ram"                 # 数据缓存位置（"ram" 或 "disk"）
    )
