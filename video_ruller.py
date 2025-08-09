"""
Adds a virtual ruler overlay to a video for manual measurement reference, 
marking every 0.5 meter up to 10 meters.

Steps:
- Open the input video and retrieve its resolution and FPS
- Calculate pixel-to-centimeter ratio based on assumed width = 1000 cm
- Draw a horizontal ruler line at 70% of video height
- Add vertical tick marks every 0.5 meter with labels every 1 meter
- Write the modified frames to a new output video

Output:
- 'testing_perhitungan_manual_ruller.mp4' in the 'result' directory
"""

import cv2
import numpy as np

cap = cv2.VideoCapture("datasets/actioncam/testing_perhitungan_manual.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # 1280
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 720
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("result/testing_perhitungan_manual_ruller.mp4", fourcc, fps, (width, height))

pixels_per_cm = width / 1000  # 1.28 px/cm
step_50cm = int(round(pixels_per_cm * 50))  # ≈ 64 px
y_position = int(0.7 * height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Garis utama penggaris (tipis)
    cv2.line(frame, (0, y_position), (width, y_position), (0, 255, 0), 1)

    for i in range(21):  # dari 0m sampai 10m, tiap 0.5m
        x = i * step_50cm
        if x > width:
            break

        # Penanda tiap 0.5 meter
        if i % 2 == 0:
            tick_length = 12
            color = (0, 0, 255)
            label = f"{i//2}m"
            cv2.putText(frame, label, (x - 10, y_position - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            tick_length = 8
            color = (100, 100, 255)

        # Garis batas (tick mark) juga tipis
        cv2.line(frame, (x, y_position - tick_length),
                 (x, y_position + tick_length), color, 1)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
