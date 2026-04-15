./nn/modules/head.py内包含添加的AttnResRTDETRDecoder和BlockAttnResRTDETRDecoder模块
./cfg/models/rt-detr/目录下为新建的模型列表.yaml

在本地ultralytics目录下需做以下更改

- 1.修改./nn/modules/head.py增加新模块
- 2../nn/modules/__init__.py里注册新增加的模块
- 3../nn/task.py里import已注册的模块
- 4../cfg/models/rt-detr/目录下增加新建的模型.yaml

数据集visdrone2019：[https://github.com/VisDrone/VisDrone-Dataset]

结果：
| Model      | Params/M | GFLOPS | P/%   | R/%   | mAP50/% | mAP50-95/% | Latency/ms |
|:-----------|---------:|-------:|------:|------:|--------:|-----------:|-----------:|
| yolo12m    |          |        |       |       |         |            |            |
| yolo12l    | 26.3     | 88.6   | 53.9  | 42.2  | 43.3    | 26.4       | 18.8       |
| yolo26l    | 24.7     | 86.1   | 54.2  | 43.1  | 44.4    | 27.0       | 14.8       |
| yolo26m    | 20.4     | 67.9   | 55.9  | 41.9  | 43.7    | 26.4       | 11.0       |
| ARDETR2    | 27.5     | 96.8   | 57.4  | 44.0  | 44.9    | 26.8       | 22.5       |
| **ARDETR3**| **28.6** | **98.5**| **60.7**| **46.2**| **47.9**| **29.6**   | 23.5       |
| ARDETR4    | 29.7     | 100.2  | 60.5  | 45.3  | 46.9    | 28.9       | 24.1       |
| ARDETR5    | 30.9     | 101.8  | 59.7  | 44.8  | 46.3    | 28.2       | 26.2       |
| ARDETR6    | 32.0     | 103.5  | 57.8  | 40.4  | 41.8    | 25.5       | 27.2       |
| ARDETR8    | 34.3     | 106.8  | 58.5  | 40.5  | 41.4    | 25.0       | 29.5       |
| ARDETR10   | 36.5     | 110.1  | 56.3  | 39.1  | 39.4    | 23.9       | 33.8       |
| ARDETR12   | 38.8     | 113.4  | 57.1  | 40.3  | 41.4    | 24.9       | 37.9       |
| RTDETR2    | 27.5     | 96.8   | 58.9  | 44.1  | 45.9    | 28.1       | 20.4       |
| RTDETR3    | 28.6     | 98.5   | 59.9  | 46.0  | 47.1    | 29.2       | 21.3       |
| RTDETR4    | 29.7     | 100.2  | 61.3  | 46.1  | 47.6    | 29.5       | 23.0       |
| RTDETR5    | 30.9     | 101.8  | 61.9  | 46.0  | 47.6    | 29.3       | 23.8       |
| RTDETR6    | 32.0     | 103.5  | 54.1  | 37.5  | 38.5    | 23.0       | 23.6       |
| RTDETR8    | 34.3     | 106.8  | 53.8  | 39.1  | 39.1    | 23.5       | 24.9       |
| RTDETR10   | 36.5     | 110.1  | 54.4  | 37.7  | 39.2    | 23.5       | 27.9       |
| RTDETR12   | 38.8     | 113.4  | 52.5  | 38.1  | 38.8    | 23.0       | 28.0       |
| BARDETR63  | 32.0     | 103.5  | 52.7  | 38.7  | 39.1    | 23.5       | 31.9       |
| BARDETR63* | 32.0     | 103.5  | 54.3  | 39.4  | 40.3    | 24.4       | 28.3       |
| BARDETR66  | 32.0     | 103.5  | 44.9  | 39.0  | 40.2    | 24.3       | 31.4       |
| DINOv3     |          |        |       |       |         |            |            |

Decoder层数对RTDETR和ARDETR的mAP50的影响
| Model  | Det2  | Det3  | Det4  | Det5  | Det6  | Det8  | Det10 | Det12 |
|:-------|------:|------:|------:|------:|------:|------:|------:|------:|
| RTDETR | 45.9  | 47.1  | **47.6**  | 47.5  | 38.5  | 39.1  | 39.2  | 38.8  |
| ARDETR | 44.9  | **47.9**  | 46.9  | 46.3  | 41.8  | 41.4  | 39.4  | 41.4  |


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
