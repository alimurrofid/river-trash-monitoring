import cv2
import time
from ultralytics import YOLO
import sys

# ====== KALIBRASI UKURAN DENGAN JARAK ======
ukuran_objek_cm = 4.1           # Ukuran real objek kalibrasi (cm)
ukuran_objek_px = 69            # Ukuran objek di kamera saat kalibrasi (pixel)
jarak_kalibrasi_cm = 80         # Jarak kamera ke objek saat kalibrasi (cm)
jarak_kerja_cm = 80             # Jarak kamera saat penggunaan (cm) - bisa diubah

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

def find_working_camera():
    """Find a working camera"""
    print("🔍 Mencari kamera yang tersedia...")
    
    for i in range(5):  # Test camera 0-4
        print(f"Testing camera {i}...", end=" ")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Found working camera!")
                cap.release()
                return i
            else:
                print("❌ Can't read")
        else:
            print("❌ Can't open")
        
        cap.release()
    
    return None

def setup_camera(camera_id):
    """Setup camera with error handling"""
    print(f"📹 Setting up camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return None
    
    # Try to set high resolution first
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual resolution
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
print("🚀 Starting Object Counting Program...")

# Find and setup camera
camera_id = find_working_camera()
if camera_id is None:
    print("❌ No working camera found!")
    print("\nPossible solutions:")
    print("1. Check camera connection")
    print("2. Close other apps using camera")
    print("3. Check camera permissions")
    print("4. Try different USB port")
    sys.exit()

cap = setup_camera(camera_id)
if cap is None:
    print("❌ Camera setup failed!")
    sys.exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize tampilan
display_width = min(1280, frame_width)
display_height = min(720, frame_height)

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

# ====== TRACKING & COUNTING ======
line_y = int(frame_height * 0.6)
object_counter = {}
track_history = {}
counted_objects = set()

# FPS info
prev_time = 0
fps_list = []
batch_interval = 2
frame_count = 0
last_results = None

# Debug info
show_debug = True
measurement_log = []
distance_calibration_mode = False

print("✅ Program ready!")
print("\nControls:")
print("- 'd': Toggle debug info")
print("- 'r': Reset counter")
print("- 'c': Distance calibration mode")
print("- '+': Increase distance (closer)")
print("- '-': Decrease distance (farther)")
print("- 'q': Quit")
print("\nPress any key in the camera window to start...")

# Create window
cv2.namedWindow("Distance-Calibrated Object Counting", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Distance-Calibrated Object Counting", display_width, display_height)

# ====== MAIN LOOP ======
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame!")
            break

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
        cv2.line(frame, (0, line_y), (frame_width, line_y), COLOR_LINE, 3)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LINE, 2)

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
                    if show_debug:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YOLO, 1)
                        cv2.putText(frame, "YOLO", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YOLO, 1)
                    
                    cv2.rectangle(frame, (acc_x1, acc_y1), (acc_x2, acc_y2), color, 2)
                    
                    if show_debug:
                        yolo_w, yolo_h = x2-x1, y2-y1
                        yolo_w_cm, yolo_h_cm = yolo_w * cm_per_pixel, yolo_h * cm_per_pixel
                        label_debug = f"YOLO: {yolo_w_cm:.1f}x{yolo_h_cm:.1f}cm | Accurate: {width_cm:.1f}x{height_cm:.1f}cm"
                        cv2.putText(frame, label_debug, (acc_x1, acc_y1 - 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    label = f"{class_name}({track_id}) {confidence*100:.0f}% | {width_cm:.1f}x{height_cm:.1f}cm [{kategori}]"
                    cv2.putText(frame, label, (acc_x1, acc_y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue

        # Display info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
        
        distance_info = f"Distance: {jarak_kerja_cm}cm | cm/px: {cm_per_pixel:.5f}"
        cv2.putText(frame, distance_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DISTANCE, 2)
        
        if distance_calibration_mode:
            cv2.putText(frame, "DISTANCE CALIBRATION MODE", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Use +/- to adjust distance, 'c' to exit", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show counting results
        y_offset = 150 if distance_calibration_mode else 120
        cv2.putText(frame, "=== COUNTING RESULTS ===", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        y_offset += 30
        for i, (cls, counts) in enumerate(object_counter.items()):
            text = f"{cls}: Total={counts['total']} | Makro={counts['makro']} | Meso={counts['meso']}"
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        if show_debug:
            controls = ["Controls: 'd'=debug, 'r'=reset, 'c'=distance, '+/-'=adjust, 'q'=quit"]
            cv2.putText(frame, controls[0], (10, frame_height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display frame
        try:
            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("Distance-Calibrated Object Counting", display_frame)
        except Exception as e:
            print(f"Display error: {e}")

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
        elif key == ord("r"):
            object_counter.clear()
            counted_objects.clear()
            track_history.clear()
            measurement_log.clear()
            print("Counter reset!")
        elif key == ord("c"):
            distance_calibration_mode = not distance_calibration_mode
            if distance_calibration_mode:
                print("Distance calibration mode ON - Use +/- to adjust distance")
            else:
                print("Distance calibration mode OFF")
        elif key == ord("+") or key == ord("="):
            new_distance = jarak_kerja_cm + 5
            update_distance_calibration(new_distance)
        elif key == ord("-"):
            new_distance = max(10, jarak_kerja_cm - 5)
            update_distance_calibration(new_distance)

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
    print("Metode: YOLO Detection + Contour Refinement + Distance Calibration")