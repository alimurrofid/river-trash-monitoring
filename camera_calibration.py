import cv2
import numpy as np
import glob
import os

# === PARAMETER YANG MUDAH DIUBAH ===
chessboard_size = (10, 7)
square_size = 25.0  # dalam mm
output_dir = "webcam_checkerboard_calibration"
image_pattern = 'webcam_checkerboard/*.jpg'
show_delay = 300  # Delay tampilan per gambar dalam milidetik (0 = tunggu user, >0 = otomatis lanjut)
# ===================================

# Siapkan titik 3D checkerboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Penampung
objpoints, imgpoints, gagal = [], [], []

# Folder output
berhasil_dir = os.path.join(output_dir, "berhasil")
gagal_dir = os.path.join(output_dir, "gagal")
os.makedirs(berhasil_dir, exist_ok=True)
os.makedirs(gagal_dir, exist_ok=True)

# Ambil semua gambar
images = glob.glob(image_pattern)

def show_resized(window_name, image, scale=0.3):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)
    cv2.waitKey(show_delay)  # Lanjut otomatis

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    filename = os.path.basename(fname)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        img_drawn = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners, ret)
        show_resized('Corners', img_drawn, scale=0.3)
        cv2.imwrite(os.path.join(berhasil_dir, filename), img_drawn)
    else:
        gagal.append(fname)
        cv2.imwrite(os.path.join(gagal_dir, filename), img)

cv2.destroyAllWindows()

# Kalibrasi kamera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez(os.path.join(output_dir, "calibration_data.npz"), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Matriks Kamera (Intrinsic Matrix):\n", mtx)
print("Koefisien Distorsi:\n", dist)

# Uji koreksi distorsi
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

if roi != (0, 0, 0, 0):
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

cv2.imwrite(os.path.join(output_dir, "hasil_koreksi.jpg"), undistorted)
show_resized('Asli', test_img, scale=0.3)
show_resized('Hasil Koreksi', undistorted, scale=0.3)
cv2.waitKey(0)
cv2.destroyAllWindows()

if gagal:
    print("\nGambar yang GAGAL deteksi checkerboard:")
    for g in gagal:
        print("-", g)
else:
    print("\nSemua gambar berhasil diproses.")
