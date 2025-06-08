import cv2
import numpy as np
import os
from datetime import datetime

# Ukuran checkerboard (jumlah sudut dalam arah baris dan kolom)
CHECKERBOARD = (10, 7)

# Direktori penyimpanan gambar
SAVE_DIR = "calib_capture2"
os.makedirs(SAVE_DIR, exist_ok=True)

# Inisialisasi kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("Tekan 's' untuk menyimpan gambar polos, 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame.")
        break

    # Salin frame asli untuk disimpan
    frame_clean = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi sudut checkerboard
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # Jika ditemukan, gambar sudutnya di frame tampilan (bukan frame yang disimpan)
    if ret_corners:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_corners)
        cv2.putText(frame, "Checkerboard Terdeteksi ✓", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Checkerboard Tidak Ditemukan", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Tampilkan frame dengan anotasi
    cv2.imshow("Live Calibration Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Simpan frame asli (tanpa coretan/tulisan)
    if key == ord('s') and ret_corners:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame_clean)
        print(f"✅ Gambar disimpan (tanpa garis/tulisan): {filepath}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
