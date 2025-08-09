"""
object detection, measurement and counting with network integration and streaming capabilities.

Features:
   - Video file processing with YOLO object detection and tracking
   - Distance-based size calibration with camera distortion correction
   - Real-time MQTT data publishing for remote monitoring
   - RTMP live streaming of original video content
   - Single video playthrough with complete analysis
   - Comprehensive measurement logging and categorization

Processing Mode:
   - Single video playthrough at normal speed (33ms frame delay)
   - Terminates when video reaches end
   - Complete analysis of entire video content

Size Categories:
   - Meso: 0.5-2.5 cm objects
   - Makro: 2.5-100 cm objects

Configuration:
   - Reference: 20cm object at 300cm distance = 26px
   - Working distance: 568.5cm (calibrated for video perspective)
   - Camera calibration from camera_intrinsics.txt
   - MQTT/RTMP settings from .env file

Input:
   - Video file: datasets/actioncam/test.mp4
   - Single complete analysis pass

Network Integration:
   - MQTT: Publishes detection counts (plastic/nonplastic by size)
   - RTMP: Streams clean video feed without detection overlays
   - Environment configuration via .env file

Dependencies:
   - ultralytics (YOLO)
   - opencv-python
   - paho-mqtt
   - python-dotenv
   - FFmpeg (for RTMP streaming)

Controls:
   - 'q': Quit before video completion

Output:
   - Video display with detection overlays and counting information
   - MQTT data publishing for remote monitoring systems
   - RTMP streaming of original video content
   - Final comprehensive report with complete video analysis results
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import sys
import os
import json
import paho.mqtt.client as mqtt
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====== KALIBRASI UKURAN DENGAN JARAK ======
ukuran_objek_cm = 20               # Ukuran real objek kalibrasi (cm)
ukuran_objek_px = 26               # Ukuran objek di kamera saat kalibrasi (pixel)
jarak_kalibrasi_cm = 300           # Jarak kamera ke objek saat kalibrasi (cm)
jarak_kerja_cm = 568.5             # Jarak kamera saat penggunaan (cm) - bisa diubah

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

# ====== MQTT CONFIGURATION ======
mqtt_broker = os.getenv('MQTT_BROKER', '127.0.0.1')
mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
mqtt_topic = os.getenv('MQTT_TOPIC', 'waste/detections')
mqtt_client = None
mqtt_connected = False
last_mqtt_publish = 0
mqtt_publish_interval = 1.0  # Publish every 1 second

print(f"📋 MQTT Configuration loaded from .env:")
print(f"   - Broker: {mqtt_broker}")
print(f"   - Port: {mqtt_port}")
print(f"   - Topic: {mqtt_topic}")

# ====== RTMP STREAMING CONFIGURATION ======
rtmp_url = os.getenv('RTMP_URL', 'rtmp://192.168.137.1:1945/hls/test')
ffmpeg_process = None
ffmpeg_cmd = None
streaming_enabled = True  # Enable streaming by default
stream_width = 854
stream_height = 480
stream_fps = 15
frames_streamed = 0
restart_attempts = 0
max_restart_attempts = 3

print(f"📡 RTMP Configuration:")
print(f"   - URL: {rtmp_url}")
print(f"   - Resolution: {stream_width}x{stream_height}")
print(f"   - FPS: {stream_fps}")

def create_default_env():
    """Create a default .env file with MQTT and RTMP configuration"""
    default_env_content = """# MQTT Configuration
MQTT_BROKER=127.0.0.1
MQTT_PORT=1883
MQTT_TOPIC=waste/detections

# RTMP Streaming Configuration
RTMP_URL=rtmp://192.168.137.1:1945/hls/test
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(default_env_content)
        print("📝 Created .env file with default settings:")
        print("   - MQTT_BROKER=127.0.0.1")
        print("   - MQTT_PORT=1883") 
        print("   - MQTT_TOPIC=waste/detections")
        print("   - RTMP_URL=rtmp://192.168.137.1:1945/hls/test")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def setup_mqtt():
    """Initialize MQTT client and connection"""
    global mqtt_client, mqtt_connected
    try:
        mqtt_client = mqtt.Client()

        # Set up callbacks
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_disconnect = on_mqtt_disconnect
        mqtt_client.on_publish = on_mqtt_publish

        # Connect to broker
        print(f"🔗 Connecting to MQTT broker: {mqtt_broker}:{mqtt_port}")
        
        # Set timeout for connection attempt
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
        
        # Wait for connection to establish or fail
        timeout = 10  # 10 seconds timeout
        start_time = time.time()
        
        while not mqtt_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not mqtt_connected:
            raise Exception(f"Failed to connect to MQTT broker {mqtt_broker}:{mqtt_port} within {timeout} seconds")

    except Exception as e:
        print(f"❌ MQTT Error: {e}")
        print(f"❌ Cannot connect to MQTT broker at {mqtt_broker}:{mqtt_port}")
        print("❌ Please ensure MQTT broker is running and accessible")
        if mqtt_client:
            mqtt_client.loop_stop()
        raise SystemExit(f"MQTT Connection Failed: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    """Callback when MQTT client connects"""
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        print(f"✅ Connected to MQTT broker successfully!")
        print(f"📡 Publishing to topic: {mqtt_topic}")
    else:
        mqtt_connected = False
        print(f"❌ Failed to connect to MQTT broker. Code: {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    """Callback when MQTT client disconnects"""
    global mqtt_connected
    mqtt_connected = False
    print(f"⚠️  Disconnected from MQTT broker. Code: {rc}")

def on_mqtt_publish(client, userdata, mid):
    """Callback when message is published"""
    # Optional: can be used for debugging publish success
    pass

def publish_mqtt_data(object_counter):
    """Publish simplified counting data to MQTT"""
    global last_mqtt_publish, mqtt_connected
    if not mqtt_client or not mqtt_connected:
        return

    current_time = time.time()

    # Check if enough time has passed since last publish
    if current_time - last_mqtt_publish < mqtt_publish_interval:
        return

    try:
        # Get current counts
        plastic_counts = object_counter.get("plastic", {"total": 0, "makro": 0, "meso": 0})
        nonplastic_counts = object_counter.get("nonplastic", {"total": 0, "makro": 0, "meso": 0})

        # Prepare simplified data payload - only the 4 specific counts
        data = {
            "plastic_makro": plastic_counts["makro"],
            "plastic_meso": plastic_counts["meso"],
            "nonplastic_makro": nonplastic_counts["makro"],
            "nonplastic_meso": nonplastic_counts["meso"]
        }

        # Publish the simplified data
        json_data = json.dumps(data)
        result = mqtt_client.publish(mqtt_topic, json_data)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            last_mqtt_publish = current_time
            # Data published successfully (silent mode)
        else:
            print(f"❌ Failed to publish MQTT data. Error code: {result.rc}")

    except Exception as e:
        print(f"❌ Error publishing MQTT data: {e}")
        # If publishing fails, try to reconnect
        mqtt_connected = False

def cleanup_mqtt():
    """Clean up MQTT connection"""
    if mqtt_client:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("🔌 MQTT connection closed")
        except Exception as e:
            print(f"Error closing MQTT connection: {e}")

def setup_rtmp_streaming():
    """Initialize RTMP streaming with FFmpeg"""
    global ffmpeg_process, streaming_enabled, ffmpeg_cmd
    try:
        # Test FFmpeg availability
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{stream_width}x{stream_height}",
            "-r", str(stream_fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline",
            "-level", "3.0", "-b:v", "400k", "-maxrate", "400k", "-bufsize", "800k",
            "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "15",
            "-sc_threshold", "0", "-x264-params", "nal-hrd=cbr:force-cfr=1",
            "-f", "flv", rtmp_url
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        print(f"✅ RTMP streaming initialized successfully")

    except subprocess.CalledProcessError:
        print(f"❌ FFmpeg not found. RTMP streaming disabled.")
        streaming_enabled = False
    except Exception as e:
        print(f"❌ RTMP streaming setup failed: {e}")
        streaming_enabled = False

def stream_frame(frame):
    """Stream frame to RTMP server"""
    global ffmpeg_process, frames_streamed
    if not streaming_enabled or not ffmpeg_process:
        return

    try:
        # Check if ffmpeg process is still running
        if ffmpeg_process.poll() is not None:
            restart_rtmp_streaming()
            return

        # Resize frame to streaming resolution
        stream_frame = cv2.resize(frame, (stream_width, stream_height))
        frame_bytes = stream_frame.tobytes()

        # Write frame to ffmpeg stdin
        ffmpeg_process.stdin.write(frame_bytes)
        ffmpeg_process.stdin.flush()
        frames_streamed += 1

    except (BrokenPipeError, OSError) as e:
        print(f"⚠️ Streaming pipe error, attempting restart...")
        restart_rtmp_streaming()
    except Exception as e:
        # Only print errors periodically to avoid spam
        if frames_streamed % 100 == 0:
            print(f"❌ Streaming error: {e}")

def restart_rtmp_streaming():
    """Restart RTMP streaming if it fails"""
    global restart_attempts, ffmpeg_process, streaming_enabled, ffmpeg_cmd
    restart_attempts += 1
    if restart_attempts > max_restart_attempts:
        print(f"❌ Max restart attempts ({max_restart_attempts}) reached. Disabling RTMP streaming.")
        streaming_enabled = False
        return

    try:
        print(f"🔄 Restarting RTMP streaming (attempt {restart_attempts}/{max_restart_attempts})")
        
        # Terminate existing process
        if ffmpeg_process:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()

        # Wait before restart
        time.sleep(2)
        
        # Start new process
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{stream_width}x{stream_height}",
            "-r", str(stream_fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline",
            "-level", "3.0", "-b:v", "400k", "-maxrate", "400k", "-bufsize", "800k",
            "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "15",
            "-sc_threshold", "0", "-x264-params", "nal-hrd=cbr:force-cfr=1",
            "-f", "flv", rtmp_url
        ]
        
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        print(f"✅ RTMP streaming restarted successfully")

    except Exception as e:
        print(f"❌ Failed to restart RTMP streaming: {e}")
        if restart_attempts >= max_restart_attempts:
            streaming_enabled = False

def cleanup_rtmp_streaming():
    """Clean up RTMP streaming"""
    if ffmpeg_process:
        try:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=5)
            print("🔌 RTMP streaming stopped")
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
            print("🔌 RTMP streaming force stopped")
        except Exception as e:
            print(f"Error stopping RTMP streaming: {e}")

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

def setup_video_capture(video_path):
    """Setup video capture with error handling"""
    print(f"📹 Setting up video capture from: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Failed to open video file!")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"✅ Video setup successful!")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Test frame reading
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Cannot read frames from video!")
        cap.release()
        return None
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
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

# Check if .env file exists
if not os.path.exists('.env'):
    print("⚠️  .env file not found. Creating default .env file...")
    create_default_env()
    print("✅ Default .env file created. Please edit it if needed and restart the application.")
    sys.exit()

# Initialize MQTT and RTMP
setup_mqtt()
if streaming_enabled:
    setup_rtmp_streaming()

# Load parameter kalibrasi kamera
print("🔧 Loading camera calibration...")
intrinsics_file = "camera_intrinsics.txt"  # Sesuaikan path file
camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics_file)

if camera_matrix is None:
    print("⚠️  Menggunakan video tanpa kalibrasi kamera...")
    use_calibration = False
else:
    use_calibration = True
    print("✅ Kalibrasi kamera berhasil dimuat!")

# Setup video path
video_path = "datasets/actioncam/test.mp4"

# Setup video capture
cap = setup_video_capture(video_path)
if cap is None:
    print("❌ Video setup failed!")
    print("\nPossible solutions:")
    print("1. Check if video file exists")
    print("2. Check video file format (MP4, AVI, MOV supported)")
    print("3. Check file permissions")
    print("4. Try different video codec")
    cleanup_mqtt()
    cleanup_rtmp_streaming()
    sys.exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    cleanup_mqtt()
    cleanup_rtmp_streaming()
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
current_frame_idx = 0

# Measurement log
measurement_log = []

print("✅ Program ready!")
print("\nControls:")
print("- 'q': Quit")
print("\nPress any key in the video window to start...")

# Create window
cv2.namedWindow("Distance-Calibrated Object Counting with Camera Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Distance-Calibrated Object Counting with Camera Calibration", display_width, display_height)

# ====== MAIN LOOP ======
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("📹 End of video reached!")
            break

        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

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

        # Keep original frame for streaming (before overlays)
        original_frame_for_streaming = frame.copy() if streaming_enabled else None

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

                    # Draw bounding boxes - removed debug mode, only show accurate bbox
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
        
        # MQTT status
        mqtt_status = f"MQTT: {'CONNECTED' if mqtt_connected else 'DISCONNECTED'}"
        mqtt_color = (0, 255, 0) if mqtt_connected else (0, 0, 255)
        cv2.putText(frame, mqtt_status, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mqtt_color, 2)

        # RTMP status
        rtmp_status = f"RTMP: {'CONNECTED' if streaming_enabled else 'DISCONNECTED'}"
        rtmp_color = (0, 255, 0) if streaming_enabled else (0, 0, 255)
        cv2.putText(frame, rtmp_status, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rtmp_color, 2)

        # Show counting results
        y_offset = 180
        cv2.putText(frame, "=== COUNTING RESULTS ===", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        y_offset += 30
        for i, (cls, counts) in enumerate(object_counter.items()):
            text = f"{cls}: Total={counts['total']} | Makro={counts['makro']} | Meso={counts['meso']}"
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

        # Stream the original frame (without overlays) to RTMP
        if streaming_enabled and original_frame_for_streaming is not None:
            stream_frame(original_frame_for_streaming)

        # Publish MQTT data periodically
        publish_mqtt_data(object_counter)

        # Display frame
        try:
            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("Distance-Calibrated Object Counting with Camera Calibration", display_frame)
        except Exception as e:
            print(f"Display error: {e}")

        # Handle keys
        wait_time = 33  # Normal video playback speed
        key = cv2.waitKey(wait_time) & 0xFF
        
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
    cleanup_mqtt()
    cleanup_rtmp_streaming()
    
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
    print(f"  - MQTT Publishing: {'ENABLED' if mqtt_connected else 'DISABLED'}")
    if mqtt_connected:
        print(f"  - MQTT Broker: {mqtt_broker}:{mqtt_port}")
        print(f"  - MQTT Topic: {mqtt_topic}")
    print(f"  - RTMP Streaming: {'ENABLED' if streaming_enabled else 'DISABLED'}")
    if streaming_enabled:
        print(f"  - RTMP URL: {rtmp_url}")
        print(f"  - Frames streamed: {frames_streamed}")
    print("Metode: YOLO Detection + Contour Refinement + Distance Calibration + Camera Calibration + MQTT + RTMP")