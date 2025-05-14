import cv2
import time
from ultralytics import YOLO

# Load model YOLO
model = YOLO("runs/dataset_clean_flip_retrain/y11n_batch16_epochs100/weights/best.pt")

# Buka video
cap = cv2.VideoCapture("actioncam/1meter1080p30fps.mp4")
# Aktifkan mode jendela bisa di-resize
cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)

# Posisi garis imaginer (70% dari tinggi frame) - horizontal
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
line_y = int(frame_height * 0.7)

# Tracking ID untuk counting
object_counter = {}       # Menyimpan jumlah per kelas
track_history = {}        # History posisi objek berdasarkan ID
counted_objects = set()   # Hindari counting ganda

# Inisialisasi FPS
prev_time = 0

# Loop video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Deteksi objek dengan YOLO
    results = model.track(frame, persist=True)

    # Gambar garis imaginer horizontal
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)  # Hijau

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            # Dapatkan pusat objek (X dan Y)
            center_y = (y1 + y2) // 2

            if track_id is not None:
                if track_id in track_history:
                    prev_y = track_history[track_id]

                    # Cek jika objek melewati garis horizontal dari atas ke bawah
                    if prev_y < line_y <= center_y and track_id not in counted_objects:
                        object_counter[class_name] = object_counter.get(class_name, 0) + 1
                        counted_objects.add(track_id)
                        print(f"{class_name} bertambah: {object_counter[class_name]}")

                # Update posisi Y terbaru
                track_history[track_id] = center_y

            # Gambar bounding box dan label
            label = f"{class_name} ({track_id}) Conf: {confidence*100:.2f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Kuning
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Merah

    # Tampilkan frame
    cv2.imshow("Object Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup video
cap.release()
cv2.destroyAllWindows()

# Print hasil akhir
print("Total objek yang dihitung:", object_counter)
