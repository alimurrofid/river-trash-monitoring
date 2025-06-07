import cv2
import time
from ultralytics import YOLO

# Load model YOLO
model = YOLO("runs/dataset_clean_flip_retrain/y11n_batch16_epochs100/weights/best.pt")

# Buka video input
cap = cv2.VideoCapture("datasets/actioncam/AE2X00017.mp4")
cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)

# Ukuran asli frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
line_y = int(frame_height * 0.6)

# Inisialisasi video writer (jika diperlukan)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_path = "result/AE2X00017.mp4.mp4"
# out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

# Tracking dan counting
object_counter = {}
track_history = {}
counted_objects = set()

# FPS
prev_time = 0
fps_list = []

# Inference tiap N frame
batch_interval = 2
frame_count = 0
last_results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    fps_list.append(fps)

    # Gunakan frame asli (tanpa resize)
    input_frame = frame.copy()

    # Inference setiap batch_interval
    if frame_count % batch_interval == 0:
        results = model.track(input_frame, persist=True)
        last_results = results
    else:
        results = last_results

    # Gambar garis horizontal untuk counting
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            center_y = int((y1 + y2) / 2)

            if track_id is not None:
                if track_id in track_history:
                    prev_y = track_history[track_id]
                    if prev_y < line_y <= center_y and track_id not in counted_objects:
                        object_counter[class_name] = object_counter.get(class_name, 0) + 1
                        counted_objects.add(track_id)
                        print(f"{class_name} bertambah: {object_counter[class_name]}")
                track_history[track_id] = center_y

            # Gambar bounding box dan label
            label = f"{class_name} ({track_id}) Conf: {confidence*100:.2f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Tampilkan FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Tampilkan hasil counting
    y_offset = 70
    for i, (cls, count) in enumerate(object_counter.items()):
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # out.write(frame)  # Simpan jika diperlukan
    cv2.imshow("Object Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
# out.release()
cv2.destroyAllWindows()

# Print hasil akhir
print("Total objek yang dihitung:", object_counter)

# Hitung dan tampilkan rata-rata FPS
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Rata-rata FPS selama proses: {avg_fps:.2f}")
