"""
Real-time object detection and size measurement system using YOLO model
with camera calibration and distance-based size calculation.

Features:
   - YOLO object detection with tracking
   - Distance-based size calibration for accurate measurements
   - Camera distortion correction using calibration parameters
   - Object counting with size categorization (meso/makro)
   - Real-time measurement logging and display

Size Categories:
   - Meso: 0.5-2.5 cm objects
   - Makro: 2.5-100 cm objects
   
Calibration Setup:
   - Reference: 20cm object at 100cm distance = 79px
   - Working distance: 100cm (adjustable)
   - Camera intrinsics loaded from camera_intrinsics.txt

Controls:
   - Press 'q' to quit and show final counting results

Output:
   - Real-time object detection with size measurements
   - Counting statistics by category
   - Measurement log with detailed size data
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import sys
import os

# ====== KALIBRASI UKURAN DENGAN JARAK ======
ukuran_objek_cm = 20               # Ukuran real objek kalibrasi (cm)
ukuran_objek_px = 79               # Ukuran objek di kamera saat kalibrasi (pixel)
jarak_kalibrasi_cm = 100           # Jarak kamera ke objek saat kalibrasi (cm)
jarak_kerja_cm = 100             # Jarak kamera saat penggunaan (cm) - bisa diubah

# Hitung konstanta kalibrasi
k = ukuran_objek_cm / (ukuran_objek_px * jarak_kalibrasi_cm)
cm_per_pixel = k * jarak_kerja_cm

print(f"=== KALIBRASI JARAK ===")
print(f"Ukuran objek kalibrasi: {ukuran_objek_cm} cm")
print(f"Ukuran di kamera (kalibrasi): {ukuran_objek_px} pixel")
print(f"Jarak kalibrasi: {jarak_kalibrasi_cm} cm")
print(f"Jarak kerja saat ini: {jarak_kerja_cm} cm")
print(f"Konstanta k: {k:.8f}")
print(f"cm_per_pixel saat ini: {cm_per_pixel:.5f}")
print()

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
        
        print("=== PARAMETER KAMERA BERHASIL DIMUAT ===")
        print(f"fx: {camera_matrix[0, 0]:.2f}")
        print(f"fy: {camera_matrix[1, 1]:.2f}")
        print(f"px: {camera_matrix[0, 2]:.2f}")
        print(f"py: {camera_matrix[1, 2]:.2f}")
        print(f"Distorsi: {dist_coeffs}")
        print()
        
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"Error membaca file kalibrasi: {e}")
        return None, None

def setup_camera_capture(camera_index=0):
    """Setup camera capture with error handling"""
    print(f"📹 Setting up camera capture from camera index: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_index}!")
        return None
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera setup successful!")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Test frame reading
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Cannot read frames from camera!")
        cap.release()
        return None
    
    print(f"✅ Frame reading test successful!")
    return cap

def update_distance_calibration(new_distance_cm):
    """Update cm_per_pixel berdasarkan jarak baru"""
    global cm_per_pixel, jarak_kerja_cm
    jarak_kerja_cm = new_distance_cm
    cm_per_pixel = k * jarak_kerja_cm
    print(f"Distance updated to {jarak_kerja_cm}cm, new cm_per_pixel: {cm_per_pixel:.5f}")
    return cm_per_pixel

def get_accurate_measurement(frame, yolo_bbox):
    """
    Menggunakan contour detection dalam YOLO ROI untuk pengukuran akurat
    """
    x1, y1, x2, y2 = yolo_bbox
    
    # Pastikan koordinat valid
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return x1, y1, x2, y2, x2-x1, y2-y1
    
    # Crop ROI dari YOLO bbox
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return x1, y1, x2, y2, x2-x1, y2-y1
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Multiple threshold methods
    methods = [
        cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)[1],
        cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)[1],
        cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ]
    
    best_contour = None
    best_area = 0
    
    for thresh in methods:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 100 and area > best_area:
                best_area = area
                best_contour = largest
    
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        accurate_x1 = x1 + x
        accurate_y1 = y1 + y
        accurate_x2 = accurate_x1 + w
        accurate_y2 = accurate_y1 + h
        return accurate_x1, accurate_y1, accurate_x2, accurate_y2, w, h
    
    return x1, y1, x2, y2, x2-x1, y2-y1

# ====== MAIN PROGRAM START ======
print("🚀 Starting Object Counting Program with Camera Calibration...")

# Load parameter kalibrasi kamera
print("🔧 Loading camera calibration...")
intrinsics_file = "camera_intrinsics.txt"  # Sesuaikan path file
camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_file)

if camera_matrix is None:
    print("⚠️  Menggunakan kamera tanpa kalibrasi kamera...")
    use_calibration = False
else:
    use_calibration = True
    print("✅ Kalibrasi kamera berhasil dimuat!")

# Setup camera capture
cap = setup_camera_capture(0)  # 0 for default camera, change if needed
if cap is None:
    print("❌ Camera setup failed!")
    print("\nPossible solutions:")
    print("1. Check if camera is connected")
    print("2. Check camera permissions")
    print("3. Try different camera index (1, 2, etc.)")
    print("4. Check if camera is being used by another application")
    sys.exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = cap.get(cv2.CAP_PROP_FPS)

# Setup undistortion map jika menggunakan kalibrasi
if use_calibration:
    print("🔧 Creating undistortion maps...")
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (frame_width, frame_height), 1, (frame_width, frame_height))
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (frame_width, frame_height), cv2.CV_16SC2)
    x_roi, y_roi, w_roi, h_roi = roi  # koordinat ROI untuk cropping
    print("✅ Undistortion maps created successfully!")
    print(f"ROI after undistortion: {x_roi},{y_roi} size {w_roi}x{h_roi}")
    
    # Update frame dimensions setelah undistortion
    effective_width = w_roi
    effective_height = h_roi
else:
    effective_width = frame_width
    effective_height = frame_height

# Resize tampilan
display_width = min(1280, effective_width)
display_height = min(720, effective_height)

print(f"Display size: {display_width}x{display_height}")

# ====== LOAD MODEL ======
print("🤖 Loading YOLO model...")
try:
    model = YOLO("runs/dataset_clean_flip_retrain/y11n_batch16_epochs100/weights/best.pt")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure the model path is correct!")
    cap.release()
    sys.exit()

# ====== WARNA UI ======
COLOR_YOLO = (0, 215, 255)    # Orange untuk YOLO bbox asli
COLOR_ACCURATE = (0, 255, 0)  # Hijau untuk bbox yang sudah akurat
COLOR_FPS = (0, 255, 255)
COLOR_LINE = (0, 0, 255)      # MERAH untuk garis counting
COLOR_TEXT = (255, 0, 0)
COLOR_DISTANCE = (255, 255, 0) # Cyan
COLOR_CALIBRATION = (255, 0, 255) # Magenta untuk info kalibrasi

# ====== TRACKING & COUNTING ======
line_y = int(effective_height * 0.7)
object_counter = {}
track_history = {}
counted_objects = set()

# FPS info dan video control
prev_time = 0
fps_list = []
batch_interval = 2
frame_count = 0
last_results = None

# Measurement log
measurement_log = []

print("✅ Program ready!")
print("\nControls:")
print("- 'q': Quit")
print("\nPress any key in the camera window to start...")

# Create window
cv2.namedWindow("Distance-Calibrated Object Counting with Camera Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Distance-Calibrated Object Counting with Camera Calibration", display_width, display_height)

# ====== MAIN LOOP ======
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera!")
            break

        # Terapkan koreksi distorsi jika tersedia
        if use_calibration:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]  # Crop bagian hitam

        frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        fps_list.append(fps)

        input_frame = frame.copy()

        # Run YOLO detection with batch processing
        if frame_count % batch_interval == 0:
            try:
                results = model.track(input_frame, persist=True)
                last_results = results
            except Exception as e:
                print(f"YOLO error: {e}")
                results = last_results
        else:
            results = last_results

        # Draw counting line
        cv2.line(frame, (0, line_y), (effective_width, line_y), COLOR_LINE, 2)

        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = int(box.id.item()) if box.id is not None else None
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    confidence = box.conf.item()

                    if confidence < 0.5:
                        continue

                    # Get accurate measurement
                    acc_x1, acc_y1, acc_x2, acc_y2, w, h = get_accurate_measurement(frame, (x1, y1, x2, y2))

                    # Calculate size in cm
                    width_cm = w * cm_per_pixel
                    height_cm = h * cm_per_pixel
                    panjang_cm = max(width_cm, height_cm)

                    # Categorize
                    if 0.5 <= panjang_cm <= 2.5:
                        kategori = "meso"
                        color = (255, 255, 0)  # Cyan
                    elif 2.5 < panjang_cm <= 100:
                        kategori = "makro"
                        color = (0, 0, 255)    # Red
                    else:
                        kategori = "lain"
                        color = (128, 128, 128)  # Gray

                    # Tracking
                    center_y = int((acc_y1 + acc_y2) / 2)

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
                                
                                log_entry = f"{class_name} ({kategori}): {width_cm:.1f}x{height_cm:.1f}cm"
                                measurement_log.append(log_entry)
                                print(f"COUNTED: {log_entry}")
                                
                        track_history[track_id] = center_y

                    # Draw bounding boxes
                    cv2.rectangle(frame, (acc_x1, acc_y1), (acc_x2, acc_y2), color, 2)
                    
                    label = f"{class_name}({track_id}) {confidence*100:.0f}% | {width_cm:.1f}x{height_cm:.1f}cm [{kategori}]"
                    cv2.putText(frame, label, (acc_x1, acc_y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue

        # Display FPS and info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
        
        distance_info = f"Distance: {jarak_kerja_cm}cm | cm/px: {cm_per_pixel:.5f}"
        cv2.putText(frame, distance_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DISTANCE, 2)
        
        # Camera calibration status
        calib_status = "Camera: CALIBRATED" if use_calibration else "Camera: NO CALIBRATION"
        cv2.putText(frame, calib_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CALIBRATION, 2)

        # Show counting results
        y_offset = 120
        cv2.putText(frame, "=== COUNTING RESULTS ===", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        y_offset += 30
        for i, (cls, counts) in enumerate(object_counter.items()):
            text = f"{cls}: Total={counts['total']} | Makro={counts['makro']} | Meso={counts['meso']}"
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        # Display frame
        try:
            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("Distance-Calibrated Object Counting with Camera Calibration", display_frame)
        except Exception as e:
            print(f"Display error: {e}")

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\n⏹️  Program stopped by user")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*50)
    print("=== HASIL COUNTING AKHIR ===")
    print("="*50)

    total_objects = 0
    for cls, counts in object_counter.items():
        print(f"{cls}:")
        print(f"  Total: {counts['total']}")
        print(f"  Meso (0.5-2.5 cm): {counts['meso']}")
        print(f"  Makro (>2.5-100 cm): {counts['makro']}")
        print(f"  Rasio Makro:Meso = {counts['makro']}:{counts['meso']}")
        total_objects += counts['total']
        print()

    print(f"TOTAL SEMUA OBJEK: {total_objects}")

    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"Rata-rata FPS: {avg_fps:.2f}")

    print("\n=== LOG PENGUKURAN ===")
    for i, log in enumerate(measurement_log, 1):
        print(f"{i}. {log}")

    print(f"\nKalibrasi yang digunakan:")
    print(f"  - Konstanta k: {k:.8f}")
    print(f"  - Jarak kerja: {jarak_kerja_cm} cm")
    print(f"  - cm_per_pixel: {cm_per_pixel:.5f}")
    print(f"  - Kalibrasi kamera: {'AKTIF' if use_calibration else 'TIDAK AKTIF'}")
    print("Metode: YOLO Detection + Contour Refinement + Distance Calibration + Camera Calibration")