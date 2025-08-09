"""
Real-time object size measurement system using camera calibration
and distance-based scaling for accurate dimensional analysis.

Features:
   - Camera distortion correction using intrinsic parameters
   - Distance-based size calibration for accurate measurements
   - Real-time object detection using contour analysis
   - Live size measurement display in pixels and centimeters
   - Automatic undistortion mapping for improved accuracy

Calibration System:
   - Reference object: 20cm at 250cm distance = 32px
   - Working distance: 300cm (adjustable)
   - Camera intrinsics loaded from camera_intrinsics.txt
   - Automatic undistortion map generation

Processing Pipeline:
   1. Load camera intrinsic parameters from file
   2. Generate undistortion maps for real-time correction
   3. Apply distortion correction to each frame
   4. Detect objects using binary thresholding and contours
   5. Calculate real-world dimensions using distance calibration
   6. Display measurements in both pixels and centimeters

Configuration:
   - Camera resolution: 1280x720 @ 30fps
   - Minimum contour area: 500 pixels
   - Threshold value: 100 (binary inverse)
   - Display resolution: 1280x720 (resized if needed)

Dependencies:
   - opencv-python
   - numpy

Controls:
   - 'q': Quit application

Output:
   - Real-time video feed with object measurements
   - Bounding boxes around detected objects
   - Size labels showing pixel and centimeter dimensions
   - Calibration status and measurement accuracy information
"""
import cv2
import numpy as np
import os

# ====== LOAD KALIBRASI KAMERA ======
def load_camera_intrinsics(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} tidak ditemukan!")
        return None, None

    camera_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros((5,))

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('fx:'):
                camera_matrix[0, 0] = float(line.split(':')[1])
            elif line.startswith('fy:'):
                camera_matrix[1, 1] = float(line.split(':')[1])
            elif line.startswith('px:'):
                camera_matrix[0, 2] = float(line.split(':')[1])
            elif line.startswith('py:'):
                camera_matrix[1, 2] = float(line.split(':')[1])
            elif line.startswith('dist:'):
                dist_values = line.split(':')[1].split(',')
                for i, val in enumerate(dist_values):
                    if i < 5:
                        dist_coeffs[i] = float(val)

        camera_matrix[2, 2] = 1.0
        print("Parameter kamera berhasil dimuat.")
        return camera_matrix, dist_coeffs

    except Exception as e:
        print(f"Gagal membaca file kalibrasi: {e}")
        return None, None


# ====== UKURAN REFERENSI OBJEK & JARAK ======
ukuran_objek_cm = 20
ukuran_objek_px = 32
jarak_kalibrasi_cm = 250
k = ukuran_objek_cm / (ukuran_objek_px * jarak_kalibrasi_cm)
jarak_kamera_cm = 300 
cm_per_pixel = k * jarak_kamera_cm  # akan dipakai setelah frame dikoreksi

# ====== INISIALISASI KAMERA ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ====== LOAD KALIBRASI ======
intrinsics_file = "camera_intrinsics.txt"
camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_file)

use_calibration = camera_matrix is not None
if use_calibration:
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_16SC2
    )
    x_roi, y_roi, w_roi, h_roi = roi
    print("Undistortion map dibuat.")
else:
    print("Kalibrasi tidak tersedia. Menggunakan video asli.")

print(f"Resolusi: {width}x{height}")
print(f"FPS: {fps}")
print(f"Rasio cm/pixel: {cm_per_pixel:.5f} cm/pixel")

# ====== LOOP UTAMA ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if use_calibration:
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        width_cm = w * cm_per_pixel
        height_cm = h * cm_per_pixel

        color = (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{w}px/{width_cm:.2f}cm x {h}px/{height_cm:.2f}cm"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Size Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
