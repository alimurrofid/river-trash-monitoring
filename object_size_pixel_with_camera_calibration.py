import cv2
import numpy as np

# Parameter kalibrasi kamera
fx = 783.194256191
fy = 784.715400914
px = 619.188277481
py = 378.673680815

# Matrix kamera
camera_matrix = np.array([[fx, 0, px],
                         [0, fy, py],
                         [0, 0, 1]], dtype=np.float32)

# Koefisien distorsi
dist_coeffs = np.array([-0.344281113, 0.162875800, -0.000835462, -0.000229546, -0.043684517], dtype=np.float32)

cap = cv2.VideoCapture(0)

# Set ke 2K dan 30 FPS
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

# Hitung optimal camera matrix untuk undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

print(f"ROI setelah undistortion: {roi}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort frame menggunakan parameter kalibrasi
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop ROI jika diperlukan (opsional)
    x, y, w, h = roi
    if w > 0 and h > 0:
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Proses deteksi objek pada frame yang sudah dikoreksi
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(undistorted_frame, f"W:{w}px H:{h}px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize tampilan agar muat di layar
    display_frame = cv2.resize(undistorted_frame, (1280, 720))
    cv2.imshow("Object Size Detection (Calibrated)", display_frame)

    # Tampilkan perbandingan (opsional)
    # cv2.imshow("Original", cv2.resize(frame, (640, 360)))
    # cv2.imshow("Undistorted", cv2.resize(undistorted_frame, (640, 360)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()