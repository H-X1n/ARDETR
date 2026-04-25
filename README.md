The `./nn/modules/head.py` file includes the implemented **AttnResRTDETRDecoder** and **BlockAttnResRTDETRDecoder** modules.
Newly created model YAML files are placed in the `./cfg/models/rt-detr/` directory.

Several modifications should be made to the local Ultralytics directory, as listed below:
1. Add novel modules in `./nn/modules/head.py`.
2. Register newly defined modules in `./nn/modules/__init__.py`.
3. Import registered modules in `./nn/task.py`.
4. Supplement customized model YAML configurations under `./cfg/models/rt-detr/`.

**Dataset**: VisDrone2019
Official Link: [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

**The latency presented in the table denotes the average inference speed of the model on the NVIDIA GeForce RTX 3060 Laptop GPU.**
result：
| Model     | Params/M    | GFLOPS      | P/%         | R/%         | mAP50/%     | mAP50-95/%  | Latency/ms |
|-----------|-------------|-------------|-------------|-------------|-------------|-------------|------------|
| yolo12l   | 26.3        | 88.6        | 53.9        | 42.2        | 43.3        | 26.4        | 18.8       |
| yolo12m   | 20.1        | 67.2        | 53.7        | 40.5        | 41.6        | 24.9        | 18.3       |
| yolo26l   | 24.7        | 86.1        | 54.2        | 43.1        | 44.4        | 27          | 14.8       |
| yolo26m   | 20.4        | 67.9        | 55.9        | 41.9        | 43.7        | 26.4        | 11         |
| ARRTDETR2   | 27.5        | 96.8        | 57.4        | 44          | 44.9        | 26.8        | 22.5       |
| ARRTDETR3   | 28.6(-3.4)  | 98.5(-5)    | 60.7(+3.9)  | 46.2(+6.7)  | 47.9(+7.2)  | 29.6(+5.2)  | 23.5(-1.3) |
| ARRTDETR4   | 29.7        | 100.2       | 60.5        | 45.3        | 46.9        | 28.9        | 24.1       |
| ARRTDETR5   | 30.9        | 101.8       | 59.7        | 44.8        | 46.3        | 28.2        | 26.2       |
| ARRTDETR6   | 32          | 103.5       | 57.8        | 40.4        | 41.8        | 25.5        | 27.2       |
| ARRTDETR8   | 34.3        | 106.8       | 58.5        | 40.5        | 41.4        | 25          | 29.5       |
| ARRTDETR10  | 36.5        | 110.1       | 56.3        | 39.1        | 39.4        | 23.9        | 33.8       |
| ARRTDETR12  | 38.8        | 113.4       | 57.1        | 40.3        | 41.4        | 24.9        | 37.9       |
| RTDETR2   | 27.5        | 96.8        | 58.9        | 44.1        | 45.9        | 28.1        | 20.4       |
| RTDETR3   | 28.6        | 98.5        | 59.9        | 46          | 47.1        | 29.2        | 21.3       |
| RTDETR4   | 29.7        | 100.2       | 61.3        | 46.1        | 47.6        | 29.5        | 23         |
| RTDETR5   | 30.9        | 101.8       | 58.9        | 45.5        | 45.8        | 27.4        | 23.8       |
| RTDETR6   | 32          | 103.5       | 56.8        | 39.5        | 40.7        | 24.4        | 24.8       |
| RTDETR8   | 34.3        | 106.8       | 53.8        | 39.1        | 39.1        | 23.5        | 24.9       |
| RTDETR10  | 36.5        | 110.1       | 54.4        | 37.7        | 39.2        | 23.5        | 27.9       |
| RTDETR12  | 38.8        | 113.4       | 52.5        | 38.1        | 38.8        | 23          | 28         |
| BARRTDETR62 | 32          | 103.5       | 57.3        | 40.6        | 41.4        | 24.9        | 23.9       |
| BARRTDETR63 | 32          | 103.5       | 55.6        | 40.7        | 41.3        | 25          | 24.2       |
| BARRTDETR66 | 32          | 103.5       | 554         | 40.1        | 40.8        | 24.6        | 25.1       |
| DINOv3 | 36 | 108.5 | 57.6 | 44.2 | 44.7 | 26.8 | 28.3 |

The influence of decoder layer quantity on the mAP50 performance of RTDETR and ARRTDETR models
| Model  | Det2  | Det3  | Det4  | Det5  | Det6  | Det8  | Det10 | Det12 |
|:-------|------:|------:|------:|------:|------:|------:|------:|------:|
| RTDETR | 45.9  | 47.1  | **47.6**  | 45.8  | 40.7  | 39.1  | 39.2  | 38.8  |
| ARRTDETR | 44.9  | **47.9**  | 46.9  | 46.3  | 41.8  | 41.4  | 39.4  | 41.4  |



Finally, **ARRTDETR-DET3** is selected as the final adopted model.
| Model        | P/%   | R/%   | mAP50/% | GFLOPS | Params/M |
|:-------------|------:|------:|--------:|-------:|---------:|
| DINOv3 ViT-S+       |  57.6     |  44.2     |   44.7      |   108.5     | 28.68 (backbone); 36.07 (total)    |
| YOLOv12-L    | 53.9  | 42.2  | 43.3    | 88.6   | 26.3     |
| YOLOv12-M    | 53.7  | 40.5  | 41.6    | 67.2   | 20.1     |
| YOLOv26-L    | 54.2  | 43.1  | 44.4    | 86.1   | 24.7     |
| YOLOv26-M    | 55.9  | 41.9  | 43.7    | 67.9   | 20.4     |
| RTDETR-L     | 56.8  | 39.5  | 40.7    | 103.5  | 32.0     |
| **ARRTDETR-Det3** | **60.7** | **46.2** | **47.9** | 98.5   | 28.6     |

sample code for training
```
from ultralytics import RTDETR
if __name__ == '__main__':
    # model = RTDETR("model_name.yaml")
    model.info()
    results = model.train(
        data="data.yaml",  
        epochs=200,
        imgsz=640,
        amp=False, 
        batch=6,
        optimizer="AdamW",  
        lr0=1e-3,
        weight_decay=0.01,  
        warmup_epochs=2.0,  
        mosaic=0.0,
        mixup=0.0,
        deterministic=False
)
```
