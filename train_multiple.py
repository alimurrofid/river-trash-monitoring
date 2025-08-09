import time
import torch
from ultralytics import YOLO


def train_model(model_path, data_yaml, project_name, run_name):
    model = YOLO(model_path)

    start_time = time.time()
    print(
        f"\nTraining dimulai pada: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device yang digunakan: {device}")

    # Training model
    model.train(
        data=data_yaml,
        batch=16,
        epochs=100,
        workers=0,
        device=0,
        project=project_name,
        name=run_name,
        cache=True,
    )

    end_time = time.time()
    print(
        f"Training selesai pada: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )
    print(
        f"Total waktu training: {int((end_time - start_time) // 3600)} jam "
        f"{int((end_time - start_time) % 3600 // 60)} menit "
        f"{int((end_time - start_time) % 60)} detik"
    )


if __name__ == "__main__":
    # Daftar dataset dan konfigurasi
    configs = [
        {
            "data_yaml": "datasets/datalabelstudio_75_15_10_clean_flip/data.yaml",
            "project_name": "runs/datalabelstudio_75_15_10_clean_flip_train",
            "run_name": "y8n_batch16_epochs100",
        }
    ]

    # Jalankan training satu per satu
    for config in configs:
        train_model(
            model_path="yolov8n.pt",
            data_yaml=config["data_yaml"],
            project_name=config["project_name"],
            run_name=config["run_name"],
        )