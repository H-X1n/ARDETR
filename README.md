./nn/modules/head.py内包含添加的AttnResRTDETRDecoder和BlockAttnResRTDETRDecoder模块
./cfg/models/rt-detr/目录下为新建的模型列表.yaml

在本地ultralytics目录下需做以下更改
1.修改./nn/modules/head.py增加新模块
2../nn/modules/__init__.py里注册新增加的模块
3../nn/task.py里import已注册的模块
4../cfg/models/rt-detr/目录下增加新建的模型.yaml

训练代码
```
from ultralytics import RTDETR
if __name__ == '__main__':
    # model = RTDETR("模型名称.yaml")
    model.info()
    results = model.train(
        data="data.yaml",  
        epochs=300,
        imgsz=640,
        amp=False, 
        batch=6,
        optimizer="AdamW",  
        lr0=1e-4,
        weight_decay=0.01,  
        warmup_epochs=2.0,  
        mosaic=0.0,
        mixup=0.0,
        deterministic=False,
        device="0,1",
        cache="ram"
    )
```
