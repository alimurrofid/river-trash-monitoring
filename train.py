from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="datasets/dataset_clean_flip/data.yaml",
    imgsz=640,
    batch=16,
    epochs=100,
    device=0,
    project="runs/dataset_clean_flip_train",
    name="y11n_batch8_epochs10",
)