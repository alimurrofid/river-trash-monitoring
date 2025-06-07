import cv2
import time
from ultralytics import YOLO

# ====== KALIBRASI UKURAN ======
ukuran_objek_cm = 2.5
ukuran_objek_px = 50
jarak_kalibrasi_cm = 100
k = ukuran_objek_cm / (ukuran_objek_px * jarak_kalibrasi_cm)
jarak_kamera_cm = 170  # ganti sesuai kondisi nyata
cm_per_pixel = k * jarak_kamera_cm

# ===========================
# WARNA (ubah sesuai kebutuhan)
# ===========================
COLOR_BOX = (0, 215, 255)      # Emas: bounding box
COLOR_FPS = (0, 255, 255)      # Kuning: FPS
COLOR_LINE = (0, 255, 0)       # Hijau: garis counting
COLOR_TEXT = (255, 0, 0)       # Biru: teks jumlah

# ====== LOAD MODEL ======
model = YOLO("runs/dataset_clean_flip_retrain/y11n_batch16_epochs100/weights/best.pt")

# ====== BUKA VIDEO ======
cap = cv2.VideoCapture("datasets/actioncam/AE2X00017.mp4")
cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)

# Info frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
line_y = int(frame_height * 0.6)

# ====== TRACKING & COUNTING ======
object_counter = {}
track_history = {}
counted_objects = set()

# FPS
prev_time = 0
fps_list = []

# Batasi inference tiap N frame
batch_interval = 2
frame_count = 0
last_results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    fps_list.append(fps)

    input_frame = frame.copy()

    # Inference tiap beberapa frame
    if frame_count % batch_interval == 0:
        results = model.track(input_frame, persist=True)
        last_results = results
    else:
        results = last_results

    # Garis horizontal untuk counting
    cv2.line(frame, (0, line_y), (frame_width, line_y), COLOR_LINE, 2)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            # Ukuran bounding box
            w = x2 - x1
            h = y2 - y1
            width_cm = w * cm_per_pixel
            height_cm = h * cm_per_pixel
            panjang_cm = max(width_cm, height_cm)

            # Klasifikasi makro / meso
            if 0.5 <= panjang_cm < 2.5:
                kategori = "meso"
            elif 2.5 <= panjang_cm < 100:
                kategori = "makro"
            else:
                kategori = "lain"

            center_y = int((y1 + y2) / 2)

            if track_id is not None:
                if track_id in track_history:
                    prev_y = track_history[track_id]
                    if prev_y < line_y <= center_y and track_id not in counted_objects:
                        counted_objects.add(track_id)
                        if class_name not in object_counter:
                            object_counter[class_name] = {"total": 0, "meso": 0, "makro": 0}
                        object_counter[class_name]["total"] += 1
                        if kategori in ["meso", "makro"]:
                            object_counter[class_name][kategori] += 1
                        print(f"{class_name} ({kategori}) bertambah: {object_counter[class_name][kategori]}")
                track_history[track_id] = center_y

            # Gambar bounding box dan label
            label = f"{class_name} ({track_id}) Conf: {confidence*100:.1f}% {width_cm:.1f}x{height_cm:.1f} cm"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 2)

    # Tampilkan FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_FPS, 2)

    # Tampilkan hasil counting
    y_offset = 70
    for i, (cls, counts) in enumerate(object_counter.items()):
        text = f"{cls}: {counts['total']} (Makro: {counts['makro']}, Meso: {counts['meso']})"
        cv2.putText(frame, text, (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

    cv2.imshow("Object Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Ringkasan
print("\n=== Hasil Counting Akhir ===")
for cls, counts in object_counter.items():
    print(f"{cls}: Total = {counts['total']} | Makro = {counts['makro']} | Meso = {counts['meso']}")

if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Rata-rata FPS: {avg_fps:.2f}")
