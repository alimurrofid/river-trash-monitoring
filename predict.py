from ultralytics import YOLO

model = YOLO("runs/d2_detect/y11n_batch8_epochs10/weights/best.pt")
model.predict(
    source=r"dataset2/test/images",
    project="runs/d2_predict",
    name="predict_y11n_batch8_epochs10",
    show=True,
    save=True,
    save_crop=False,
    save_txt=False,
    show_labels=True,
    show_conf=True,
)
