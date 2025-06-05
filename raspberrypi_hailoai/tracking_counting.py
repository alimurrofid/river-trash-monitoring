import cv2
import supervision as sv
from collections import defaultdict

# Path ke video input
video_path = 'video/AE2X00017.mp4'

# Inisialisasi VideoCapture
cap = cv2.VideoCapture(video_path)

# Inisialisasi LineZone untuk counting
line = sv.LineZone(start=(100, 200), end=(500, 200))
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=1)
bounding_box_annotator = sv.BoxAnnotator()

# Inisialisasi tracker
tracker = sv.ByteTrack()
counter = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dapatkan deteksi dari frame
    detections = sv.Detections.from_hailo(frame)  # Sesuaikan dengan metode yang Anda gunakan

    # Update tracker dengan deteksi
    tracked_objects = tracker.update_with_detections(detections)

    # Trigger line zone
    line.trigger(detections=tracked_objects)

    # Hitung objek yang melewati garis
    for det in tracked_objects:
        if det.crossed_line:
            label = det.data["class_name"]
            counter[label] += 1

    # Anotasi frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=tracked_objects)
    annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line)

    # Tampilkan frame
    cv2.imshow("River Trash Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Total counts:", dict(counter))
cap.release()
cv2.destroyAllWindows()
