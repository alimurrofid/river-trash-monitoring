import cv2
import numpy as np

# Load hasil kalibrasi kamera
calib_data = np.load("camera_calibrations/calib_checkerboard_calibration/fisheye_calibration_data.npz")
camera_matrix = calib_data['mtx']
dist_coeffs = calib_data['dist']

# Buka kamera
cap = cv2.VideoCapture(0)

# Set ke 720p dan 30 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Ambil info aktual
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

print(f"Resolusi: {width}x{height}")
print(f"FPS: {fps}")
print(f"FOURCC Codec: {fourcc}")

# Buat map untuk koreksi distorsi
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (width, height), alpha=0, newImgSize=(width, height)
)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_16SC2
)

# Simpan ROI (Region of Interest) untuk crop
x, y, w_roi, h_roi = roi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Koreksi distorsi lensa
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Crop otomatis untuk hilangkan bagian hitam (jika ROI valid)
    if roi != (0, 0, 0, 0):
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]

    # Proses deteksi objek pada frame yang sudah dikoreksi
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
        cv2.rectangle(undistorted, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)
        cv2.putText(undistorted, f"W:{w_box}px H:{h_box}px", (x_box, y_box - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize tampilan agar muat di layar (opsional)
    display_frame = cv2.resize(undistorted, (1280, 720))
    cv2.imshow("Object Size Detection (Undistorted & Cropped)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
