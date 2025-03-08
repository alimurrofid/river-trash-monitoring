from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="dataset2/data.yaml",
    imgsz=640,
    batch=8,
    epochs=10,
    workers=0,
    device=0,
    project="runs/d2_detect",
    name="y11n_batch8_epochs10",
)
