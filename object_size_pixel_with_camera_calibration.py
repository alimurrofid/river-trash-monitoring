"""
Camera calibration-aware object detection system with optional distortion correction
for accurate real-time object measurement and analysis.

Features:
   - Camera intrinsic parameter loading from configuration file
   - Optional lens distortion correction using calibration data
   - Real-time object detection with contour analysis
   - Automatic ROI handling after undistortion
   - Graceful fallback to uncalibrated mode

Calibration System:
   - Reads camera matrix (fx, fy, px, py) from camera_intrinsics.txt
   - Loads distortion coefficients for lens correction
   - Generates optimal camera matrix and undistortion maps
   - Applies real-time distortion correction with ROI cropping
   - Maintains full functionality without calibration file

Processing Pipeline:
   1. Load camera intrinsic parameters (optional)
   2. Initialize camera with HD resolution and frame rate
   3. Generate undistortion maps if calibration available
   4. For each frame:
      - Apply lens distortion correction (if calibrated)
      - Crop to valid region of interest
      - Convert to grayscale and apply binary threshold
      - Find and filter contours by minimum area
      - Draw bounding boxes and pixel measurements

Configuration:
   - Camera resolution: 1280x720 @ 30fps
   - Minimum contour area: 500 pixels
   - Binary threshold: 100 (inverse threshold)
   - Display resolution: 1280x720 (resized to fit)
   - Detection color: Green (0, 255, 0)

File Format (camera_intrinsics.txt):
   fx: [focal_length_x]
   fy: [focal_length_y]
   px: [principal_point_x]
   py: [principal_point_y]
   dist: [k1,k2,p1,p2,k3]

Dependencies:
   - opencv-python
   - numpy

Controls:
   - 'q': Quit application

Output:
   - Real-time video with optional distortion correction
   - Green bounding boxes around detected objects
   - Pixel dimensions displayed for each object
   - Calibration status and camera properties on startup
"""
import cv2
import numpy as np
import os

def load_camera_intrinsics(file_path):
    """
    Membaca parameter intrinsik kamera dari file
    """
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
                    if i < 5:  # Maksimal 5 koefisien distorsi
                        dist_coeffs[i] = float(val)
        
        camera_matrix[2, 2] = 1.0  # Set elemen (2,2) = 1
        
        print("Parameter kamera berhasil dimuat:")
        print(f"fx: {camera_matrix[0, 0]:.2f}")
        print(f"fy: {camera_matrix[1, 1]:.2f}")
        print(f"px: {camera_matrix[0, 2]:.2f}")
        print(f"py: {camera_matrix[1, 2]:.2f}")
        print(f"Distorsi: {dist_coeffs}")
        
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"Error membaca file kalibrasi: {e}")
        return None, None

def main():
    # Load parameter kalibrasi kamera
    intrinsics_file = "camera_intrinsics.txt"  # Sesuaikan path file
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_file)
    
    if camera_matrix is None:
        print("Menggunakan kamera tanpa kalibrasi...")
        use_calibration = False
    else:
        use_calibration = True
    
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    # Set resolusi dan FPS
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
    print(f"Kalibrasi aktif: {use_calibration}")
    
    # Hitung undistortion map jika menggunakan kalibrasi
    if use_calibration:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_16SC2)
        x, y, w, h = roi  # koordinat ROI untuk cropping
        print("Undistortion map berhasil dibuat")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Terapkan koreksi distorsi jika tersedia
        if use_calibration:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            frame = frame[y:y+h, x:x+w]  # Crop bagian hitam
        
        # Deteksi objek
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (0, 255, 0), 2)
            cv2.putText(frame, f"W:{w_obj}px H:{h_obj}px", (x_obj, y_obj - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tampilkan frame
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Object Size Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
