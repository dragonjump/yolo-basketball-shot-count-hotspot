
# Setup
1. Use label studio to do training
2. Once you done export and dowload the folders like `labelstudio`
3. Now begin steps below. You can use google collab or your machine

Pretraining:>>
``` 
 
 python  1_convert_labelstudio_yolo_train 
 python  2_generate_yaml_for_yolo_train.py
 python  3_generate_yolo_train.py

```

Do training:>> 
```
  yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640
```
Test training:>> 
```
yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True

```