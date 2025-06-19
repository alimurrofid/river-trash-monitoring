#counting_makro_meso_with_mqtt.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import supervision as sv
import json
import paho.mqtt.client as mqtt
import subprocess
import select
import sys
from typing import Dict, Set, Tuple, Optional, List
from dotenv import load_dotenv

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp


class HailoObjectCounterMacroMeso(app_callback_class):
    """Object counter with macro/meso classification, MQTT, and RTMP streaming"""

    def __init__(self, calibration_path="camera_intrinsics.txt"):
        super().__init__()

        # Load environment variables
        load_dotenv()

        # Configuration
        self.detection_threshold = 0.1
        self.line_y_ratio = 0.7
        self.line_y = None

        # ====== RTMP STREAMING CONFIGURATION ======
        self.rtmp_url = os.getenv('RTMP_URL', 'rtmp://192.168.137.1:1945/hls/test')
        self.ffmpeg_process = None
        self.streaming_enabled = True  # Enable streaming by default
        self.stream_width = 854
        self.stream_height = 480
        self.stream_fps = 15
        self.frames_streamed = 0
        self.restart_attempts = 0
        self.max_restart_attempts = 3

        print(f"📡 RTMP Configuration:")
        print(f"   - URL: {self.rtmp_url}")
        print(f"   - Resolution: {self.stream_width}x{self.stream_height}")
        print(f"   - FPS: {self.stream_fps}")

        # ====== MQTT CONFIGURATION FROM .ENV ======
        self.mqtt_broker = os.getenv('MQTT_BROKER', '127.0.0.1')
        self.mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
        self.mqtt_topic = os.getenv('MQTT_TOPIC', 'waste/detections')
        self.mqtt_client = None
        self.mqtt_connected = False
        self.last_mqtt_publish = 0
        self.mqtt_publish_interval = 1.0  # Publish every 1 second

        print(f"📋 MQTT Configuration loaded from .env:")
        print(f"   - Broker: {self.mqtt_broker}")
        print(f"   - Port: {self.mqtt_port}")
        print(f"   - Topic: {self.mqtt_topic}")

        # Initialize MQTT and RTMP
        self.setup_mqtt()
        if self.streaming_enabled:
            self.setup_rtmp_streaming()

        # ====== CAMERA CALIBRATION ======
        self.calibration_path = calibration_path
        self.camera_matrix = None
        self.dist_coeffs = None
        self.use_calibration = False
        self.map1 = None
        self.map2 = None
        self.roi_x = 0
        self.roi_y = 0
        self.roi_w = 0
        self.roi_h = 0

        # Load camera calibration
        self.load_camera_calibration()

        # ====== DISTANCE CALIBRATION SETTINGS ======
        # Kalibrasi ukuran dengan jarak (SAMA seperti OpenCV version)
        self.ukuran_objek_cm = 20               # Ukuran real objek kalibrasi (cm)
        self.ukuran_objek_px = 41               # Ukuran objek di kamera saat kalibrasi (pixel)
        self.jarak_kalibrasi_cm = 200           # Jarak kamera ke objek saat kalibrasi (cm)
        self.jarak_kamera_sekarang_cm = 568.5   # Jarak kamera saat penggunaan (cm)

        # Hitung konstanta kalibrasi
        self.k = self.ukuran_objek_cm / (self.ukuran_objek_px * self.jarak_kalibrasi_cm)
        
        # CORRECTION FACTOR: Adjust for difference between Hailo and laptop measurements
        # Based on measurement: Hailo measures ~1.6x larger pixels than laptop
        self.hailo_correction_factor = 0.625  # 1/1.6 = 0.625
        
        self.cm_per_pixel = self.k * self.jarak_kamera_sekarang_cm * self.hailo_correction_factor

        print(f"=== KALIBRASI JARAK HAILO ===")
        print(f"Ukuran objek kalibrasi: {self.ukuran_objek_cm} cm")
        print(f"Ukuran di kamera (kalibrasi): {self.ukuran_objek_px} pixel")
        print(f"Jarak kalibrasi: {self.jarak_kalibrasi_cm} cm")
        print(f"Jarak kamera sekarang: {self.jarak_kamera_sekarang_cm} cm")
        print(f"Konstanta k: {self.k:.8f}")
        print(f"Hailo correction factor: {self.hailo_correction_factor}")
        print(f"cm_per_pixel saat ini: {self.cm_per_pixel:.5f}")

        # Label colors
        self.label_colors = {
            "plastic": (0, 255, 255),      # Yellow
            "nonplastic": (0, 0, 255),     # Red
        }

        # Size category colors
        self.size_colors = {
            "meso": (255, 255, 0),    # Cyan
            "makro": (0, 0, 255),     # Red
            "lain": (128, 128, 128)   # Gray
        }

        # Label mapping
        self.class_names = {
            0: "unlabeled",
            1: "nonplastic",
            2: "plastic"
        }

        # Object counting variables - now with macro/meso breakdown
        self.object_counter = {}
        self.track_history = {}
        self.counted_objects = set()
        self.measurement_log = []

        # FPS tracking
        self.prev_time = 0
        self.fps_list = []

        # Frame processing
        self.frame_count = 0

        # FALLBACK TRACKING SYSTEM for plastic objects without valid Track ID
        self.next_fallback_id = 10000
        self.plastic_fallback_tracker = {}
        self.bbox_similarity_threshold = 50

        # ====== MEMORY MANAGEMENT ======
        self.max_track_history_size = 1000
        self.max_counted_objects_size = 1000
        self.max_fallback_tracker_size = 100
        self.cleanup_interval = 100  # frames
        self.inactive_track_timeout = 30  # frames
        self.track_last_seen = {}  # Track when each ID was last seen
        self.fallback_last_seen = {}  # Track when each fallback ID was last seen

        # Size measurement settings
        self.show_size_debug = True

        # ====== RESOLUTION TRACKING ======
        self.camera_width = None
        self.camera_height = None
        self.resolution_displayed = False  # Flag to show resolution only once
        self.calibration_maps_ready = False

        # Initialize counter structure
        self._initialize_counters()

    def setup_rtmp_streaming(self):
        """Initialize RTMP streaming with FFmpeg"""
        try:
            # Test FFmpeg availability
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

            self.ffmpeg_cmd = [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{self.stream_width}x{self.stream_height}",
                "-r", str(self.stream_fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline",
                "-level", "3.0", "-b:v", "400k", "-maxrate", "400k", "-bufsize", "800k",
                "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "15",
                "-sc_threshold", "0", "-x264-params", "nal-hrd=cbr:force-cfr=1",
                "-f", "flv", self.rtmp_url
            ]

            self.ffmpeg_process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            print(f"✅ RTMP streaming initialized successfully")

        except subprocess.CalledProcessError:
            print(f"❌ FFmpeg not found. RTMP streaming disabled.")
            self.streaming_enabled = False
        except Exception as e:
            print(f"❌ RTMP streaming setup failed: {e}")
            self.streaming_enabled = False

    def stream_frame(self, frame):
        """Stream frame to RTMP server"""
        if not self.streaming_enabled or not self.ffmpeg_process:
            return

        try:
            # Check if ffmpeg process is still running
            if self.ffmpeg_process.poll() is not None:
                self.restart_rtmp_streaming()
                return

            # Resize frame to streaming resolution
            stream_frame = cv2.resize(frame, (self.stream_width, self.stream_height))
            stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_RGB2BGR)
            frame_bytes = stream_frame.tobytes()

            # Write frame to ffmpeg stdin
            self.ffmpeg_process.stdin.write(frame_bytes)
            self.ffmpeg_process.stdin.flush()
            self.frames_streamed += 1

        except (BrokenPipeError, OSError) as e:
            print(f"⚠️ Streaming pipe error, attempting restart...")
            self.restart_rtmp_streaming()
        except Exception as e:
            # Only print errors periodically to avoid spam
            if self.frames_streamed % 100 == 0:
                print(f"❌ Streaming error: {e}")

    def restart_rtmp_streaming(self):
        """Restart RTMP streaming if it fails"""
        self.restart_attempts += 1
        if self.restart_attempts > self.max_restart_attempts:
            print(f"❌ Max restart attempts ({self.max_restart_attempts}) reached. Disabling RTMP streaming.")
            self.streaming_enabled = False
            return

        try:
            print(f"🔄 Restarting RTMP streaming (attempt {self.restart_attempts}/{self.max_restart_attempts})")
            
            # Terminate existing process
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()

            # Wait before restart
            time.sleep(2)
            
            # Start new process
            self.ffmpeg_process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            print(f"✅ RTMP streaming restarted successfully")

        except Exception as e:
            print(f"❌ Failed to restart RTMP streaming: {e}")
            if self.restart_attempts >= self.max_restart_attempts:
                self.streaming_enabled = False

    def cleanup_rtmp_streaming(self):
        """Clean up RTMP streaming"""
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
                print("🔌 RTMP streaming stopped")
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
                print("🔌 RTMP streaming force stopped")
            except Exception as e:
                print(f"Error stopping RTMP streaming: {e}")

    def setup_mqtt(self):
        """Initialize MQTT client and connection"""
        try:
            self.mqtt_client = mqtt.Client()

            # Set up callbacks
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            self.mqtt_client.on_publish = self.on_mqtt_publish

            # Connect to broker
            print(f"🔗 Connecting to MQTT broker: {self.mqtt_broker}:{self.mqtt_port}")
            
            # Set timeout for connection attempt
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
            # Wait for connection to establish or fail
            import time
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            
            while not self.mqtt_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.mqtt_connected:
                raise Exception(f"Failed to connect to MQTT broker {self.mqtt_broker}:{self.mqtt_port} within {timeout} seconds")

        except Exception as e:
            print(f"❌ MQTT Error: {e}")
            print(f"❌ Cannot connect to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
            print("❌ Please ensure MQTT broker is running and accessible")
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
            raise SystemExit(f"MQTT Connection Failed: {e}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback when MQTT client connects"""
        if rc == 0:
            self.mqtt_connected = True
            print(f"✅ Connected to MQTT broker successfully!")
            print(f"📡 Publishing to topic: {self.mqtt_topic}")
        else:
            self.mqtt_connected = False
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        """Callback when MQTT client disconnects"""
        self.mqtt_connected = False
        print(f"⚠️  Disconnected from MQTT broker. Code: {rc}")

    def on_mqtt_publish(self, client, userdata, mid):
        """Callback when message is published"""
        # Optional: can be used for debugging publish success
        pass

    def publish_mqtt_data(self):
        """Publish simplified counting data to MQTT"""
        if not self.mqtt_client or not self.mqtt_connected:
            print("⚠️  MQTT not connected - cannot publish data")
            return

        current_time = time.time()

        # Check if enough time has passed since last publish
        if current_time - self.last_mqtt_publish < self.mqtt_publish_interval:
            return

        try:
            # Get current counts
            plastic_counts = self.object_counter.get("plastic", {"total": 0, "makro": 0, "meso": 0})
            nonplastic_counts = self.object_counter.get("nonplastic", {"total": 0, "makro": 0, "meso": 0})

            # Prepare simplified data payload - only the 4 specific counts
            data = {
                "plastic_makro": plastic_counts["makro"],
                "plastic_meso": plastic_counts["meso"],
                "nonplastic_makro": nonplastic_counts["makro"],
                "nonplastic_meso": nonplastic_counts["meso"]
            }

            # Publish the simplified data
            json_data = json.dumps(data)
            result = self.mqtt_client.publish(self.mqtt_topic, json_data)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.last_mqtt_publish = current_time
                # Data published successfully (silent mode)
            else:
                print(f"❌ Failed to publish MQTT data. Error code: {result.rc}")

        except Exception as e:
            print(f"❌ Error publishing MQTT data: {e}")
            # If publishing fails, try to reconnect
            self.mqtt_connected = False

    def cleanup_mqtt(self):
        """Clean up MQTT connection"""
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("🔌 MQTT connection closed")
            except Exception as e:
                print(f"Error closing MQTT connection: {e}")

    def load_camera_calibration(self):
        """Load camera intrinsics from file"""
        if not os.path.exists(self.calibration_path):
            print(f"❌ Camera calibration file not found: {self.calibration_path}")
            print("⚠️  Proceeding without camera calibration...")
            return

        try:
            calibration_data = {}
            with open(self.calibration_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        if key == 'dist':
                            # Parse distortion coefficients
                            dist_values = [float(x.strip()) for x in value.split(',')]
                            calibration_data[key] = np.array(dist_values, dtype=np.float32)
                        else:
                            calibration_data[key] = float(value)

            # Create camera matrix
            self.camera_matrix = np.array([
                [calibration_data['fx'], 0, calibration_data['px']],
                [0, calibration_data['fy'], calibration_data['py']],
                [0, 0, 1]
            ], dtype=np.float32)

            self.dist_coeffs = calibration_data['dist']
            self.use_calibration = True

            print("✅ Camera calibration loaded successfully!")
            print(f"Camera Matrix:\n{self.camera_matrix}")
            print(f"Distortion Coefficients: {self.dist_coeffs}")

        except Exception as e:
            print(f"❌ Error loading camera calibration: {e}")
            print("⚠️  Proceeding without camera calibration...")
            self.use_calibration = False

    def setup_undistortion_maps(self, width: int, height: int):
        """Setup undistortion maps when frame dimensions are known"""
        if not self.use_calibration or self.calibration_maps_ready:
            return

        try:
            print("🔧 Computing undistortion maps...")
            # Use alpha=0 to remove black areas completely by cropping to valid pixels only
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (width, height), alpha=0,
                newImgSize=(width, height)
            )

            # Get the ROI to crop out black areas
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi
            print(f"Valid region after undistortion: x={self.roi_x}, y={self.roi_y}, w={self.roi_w}, h={self.roi_h}")

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix,
                (width, height), cv2.CV_16SC2
            )

            self.calibration_maps_ready = True
            print("✅ Undistortion maps ready!")
            print(f"Black areas will be cropped out. New effective resolution: {self.roi_w}x{self.roi_h}")

        except Exception as e:
            print(f"❌ Error setting up undistortion maps: {e}")
            self.use_calibration = False

    def apply_camera_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply camera calibration and remove black areas"""
        if not self.use_calibration or not self.calibration_maps_ready:
            return frame

        try:
            # Undistort the frame
            undistorted_frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

            # Crop to remove black areas - only keep the valid region
            if self.roi_w > 0 and self.roi_h > 0:
                cropped_frame = undistorted_frame[self.roi_y:self.roi_y+self.roi_h,
                                                 self.roi_x:self.roi_x+self.roi_w]
                return cropped_frame
            else:
                return undistorted_frame

        except Exception as e:
            print(f"❌ Error applying camera calibration: {e}")
            return frame

    def undistort_detection_coordinates(self, bbox_coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Convert detection coordinates from distorted to undistorted frame coordinates
        This is the key fix: we transform the detection coordinates instead of the frame
        """
        if not self.use_calibration or not self.calibration_maps_ready:
            return bbox_coords
        
        try:
            x1, y1, x2, y2 = bbox_coords
            
            # Create array of corner points
            distorted_points = np.array([
                [[x1, y1]], [[x2, y1]], [[x1, y2]], [[x2, y2]]
            ], dtype=np.float32)
            
            # Undistort the points
            undistorted_points = cv2.undistortPoints(
                distorted_points, 
                self.camera_matrix, 
                self.dist_coeffs, 
                P=self.new_camera_matrix
            )
            
            # Get undistorted coordinates
            undist_coords = undistorted_points.reshape(-1, 2)
            
            # Find min/max to create new bbox
            x_coords = undist_coords[:, 0]
            y_coords = undist_coords[:, 1]
            
            new_x1 = max(0, int(np.min(x_coords)))
            new_y1 = max(0, int(np.min(y_coords)))
            new_x2 = min(self.roi_w, int(np.max(x_coords)))
            new_y2 = min(self.roi_h, int(np.max(y_coords)))
            
            # Adjust for ROI cropping
            final_x1 = max(0, new_x1 - self.roi_x)
            final_y1 = max(0, new_y1 - self.roi_y)
            final_x2 = max(0, new_x2 - self.roi_x)
            final_y2 = max(0, new_y2 - self.roi_y)
            
            return (final_x1, final_y1, final_x2, final_y2)
            
        except Exception as e:
            print(f"Error in coordinate undistortion: {e}")
            return bbox_coords

    def initialize_video_writer(self, width: int, height: int):
        """Initialize video writer for output"""
        if self.output_video_path and not self.video_initialized:
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(self.output_video_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory: {output_dir}")

                # Use effective dimensions after calibration
                effective_width = self.roi_w if self.use_calibration and self.calibration_maps_ready else width
                effective_height = self.roi_h if self.use_calibration and self.calibration_maps_ready else height

                # Define codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    30.0,  # Fixed FPS at 30
                    (effective_width, effective_height)
                )
                self.video_initialized = True
                print(f"Video writer initialized: {self.output_video_path}")
                print(f"Output video resolution: {effective_width}x{effective_height} @ 30 FPS")
                if self.use_calibration and self.calibration_maps_ready:
                    print("Video will be saved with undistorted and cropped frames")
            except Exception as e:
                print(f"Error initializing video writer: {e}")
                self.video_writer = None

    def write_frame_to_video(self, frame: np.ndarray):
        """Write frame to output video"""
        if self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"Error writing frame to video: {e}")

    def release_video_writer(self):
        """Release video writer"""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                print(f"Video saved successfully: {self.output_video_path}")
            except Exception as e:
                print(f"Error releasing video writer: {e}")

    def _initialize_counters(self):
        """Initialize counter structure for all classes"""
        for class_name in ["plastic", "nonplastic"]:
            if class_name not in self.object_counter:
                self.object_counter[class_name] = {
                    "total": 0,
                    "meso": 0,
                    "makro": 0
                }

    def cleanup_memory(self):
        """Clean up memory by removing inactive tracks and old data"""
        current_frame = self.frame_count
        
        # Clean up track_history
        inactive_tracks = []
        for track_id, last_seen in self.track_last_seen.items():
            if current_frame - last_seen > self.inactive_track_timeout:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_last_seen:
                del self.track_last_seen[track_id]
        
        # Clean up fallback tracker
        inactive_fallback_tracks = []
        for bbox, track_id in list(self.plastic_fallback_tracker.items()):
            if track_id in self.fallback_last_seen:
                if current_frame - self.fallback_last_seen[track_id] > self.inactive_track_timeout:
                    inactive_fallback_tracks.append((bbox, track_id))
            else:
                # If no last_seen record, remove it
                inactive_fallback_tracks.append((bbox, track_id))
        
        for bbox, track_id in inactive_fallback_tracks:
            if bbox in self.plastic_fallback_tracker:
                del self.plastic_fallback_tracker[bbox]
            if track_id in self.fallback_last_seen:
                del self.fallback_last_seen[track_id]
        
        # Limit sizes to prevent unbounded growth
        if len(self.track_history) > self.max_track_history_size:
            # Remove oldest entries
            sorted_tracks = sorted(self.track_last_seen.items(), key=lambda x: x[1])
            tracks_to_remove = sorted_tracks[:len(sorted_tracks) - self.max_track_history_size + 100]
            for track_id, _ in tracks_to_remove:
                if track_id in self.track_history:
                    del self.track_history[track_id]
                if track_id in self.track_last_seen:
                    del self.track_last_seen[track_id]
        
        if len(self.counted_objects) > self.max_counted_objects_size:
            # Convert to list, keep most recent half
            counted_list = list(self.counted_objects)
            self.counted_objects = set(counted_list[-self.max_counted_objects_size//2:])
        
        if len(self.plastic_fallback_tracker) > self.max_fallback_tracker_size:
            # Remove oldest fallback tracks
            sorted_fallback = sorted(self.fallback_last_seen.items(), key=lambda x: x[1])
            fallback_to_remove = sorted_fallback[:len(sorted_fallback) - self.max_fallback_tracker_size + 20]
            for track_id, _ in fallback_to_remove:
                # Find and remove corresponding bbox
                bbox_to_remove = None
                for bbox, fid in self.plastic_fallback_tracker.items():
                    if fid == track_id:
                        bbox_to_remove = bbox
                        break
                if bbox_to_remove:
                    del self.plastic_fallback_tracker[bbox_to_remove]
                if track_id in self.fallback_last_seen:
                    del self.fallback_last_seen[track_id]

    def get_memory_usage_info(self) -> Dict:
        """Get current memory usage information"""
        return {
            "track_history_size": len(self.track_history),
            "counted_objects_size": len(self.counted_objects),
            "fallback_tracker_size": len(self.plastic_fallback_tracker),
            "track_last_seen_size": len(self.track_last_seen),
            "fallback_last_seen_size": len(self.fallback_last_seen),
            "measurement_log_size": len(self.measurement_log)
        }

    def set_camera_resolution(self, width: int, height: int):
        """Set and display camera resolution (only once)"""
        if not self.resolution_displayed:
            self.camera_width = width
            self.camera_height = height

            # Setup undistortion maps now that we know the dimensions
            if self.use_calibration:
                self.setup_undistortion_maps(width, height)

            # Display effective resolution after calibration
            effective_width = self.roi_w if self.use_calibration and self.calibration_maps_ready else width
            effective_height = self.roi_h if self.use_calibration and self.calibration_maps_ready else height

            print(f"\n=== INFORMASI KAMERA ===")
            print(f"Resolusi Kamera Original: {width} x {height} pixels")
            if self.use_calibration and self.calibration_maps_ready:
                print(f"Resolusi Efektif (setelah kalibrasi): {effective_width} x {effective_height} pixels")
                print(f"Area yang dipotong: x={self.roi_x}, y={self.roi_y}")
            print(f"Aspect Ratio: {effective_width/effective_height:.2f}:1")
            print(f"Camera Calibration: {'ENABLED' if self.use_calibration else 'DISABLED'}")

            # Calculate total pixels
            total_pixels = effective_width * effective_height
            if total_pixels >= 1920*1080:
                quality = "Full HD (1080p)"
            elif total_pixels >= 1280*720:
                quality = "HD (720p)"
            elif total_pixels >= 640*480:
                quality = "VGA"
            else:
                quality = "Low Resolution"

            print(f"Total Effective Pixels: {total_pixels:,} ({quality})")
            print(f"Field of View (estimasi): {effective_width * self.cm_per_pixel:.1f}cm x {effective_height * self.cm_per_pixel:.1f}cm")
            print("="*30)

            self.resolution_displayed = True

    def update_distance_calibration(self, new_distance_cm):
        """Update cm_per_pixel based on new distance"""
        self.jarak_kamera_sekarang_cm = new_distance_cm
        self.cm_per_pixel = self.k * self.jarak_kamera_sekarang_cm * self.hailo_correction_factor
        print(f"Distance updated to {self.jarak_kamera_sekarang_cm}cm, new cm_per_pixel: {self.cm_per_pixel:.5f}")

        # Update field of view if resolution is known
        if self.camera_width and self.camera_height:
            effective_width = self.roi_w if self.use_calibration and self.calibration_maps_ready else self.camera_width
            effective_height = self.roi_h if self.use_calibration and self.calibration_maps_ready else self.camera_height
            print(f"New Field of View: {effective_width * self.cm_per_pixel:.1f}cm x {effective_height * self.cm_per_pixel:.1f}cm")

        return self.cm_per_pixel

    def get_accurate_measurement(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple:
        """
        Get accurate measurement using contour detection within bbox ROI
        Returns: (accurate_bbox, width_pixels, height_pixels)
        """
        x1, y1, x2, y2 = bbox

        # Ensure valid coordinates
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return (x1, y1, x2, y2), x2-x1, y2-y1

        # Crop ROI from bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return (x1, y1, x2, y2), x2-x1, y2-y1

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
            return (accurate_x1, accurate_y1, accurate_x2, accurate_y2), w, h

        return (x1, y1, x2, y2), x2-x1, y2-y1

    def categorize_size(self, width_cm: float, height_cm: float) -> str:
        """Categorize object based on size"""
        panjang_cm = max(width_cm, height_cm)

        if 0.5 <= panjang_cm < 2.5:
            return "meso"
        elif 2.5 <= panjang_cm < 100:
            return "makro"
        else:
            return "lain"

    def set_line_position(self, height: int):
        if self.line_y is None:
            # Use effective height after calibration
            effective_height = self.roi_h if self.use_calibration and self.calibration_maps_ready else height
            self.line_y = int(effective_height * self.line_y_ratio)

    def calculate_fps(self) -> float:
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        self.fps_list.append(fps)
        return fps

    def get_label_color(self, label: str) -> Tuple[int, int, int]:
        return self.label_colors.get(label, (255, 255, 255))

    def get_size_color(self, size_category: str) -> Tuple[int, int, int]:
        return self.size_colors.get(size_category, (255, 255, 255))

    def calculate_bbox_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate distance between two bounding box centers"""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2

        return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

    def get_fallback_track_id(self, bbox: Tuple, class_name: str) -> int:
        """Get or create fallback track ID for objects without valid tracking"""
        if class_name != "plastic":
            return 0

        # Check if we can match with existing fallback tracker
        for tracked_bbox, track_id in list(self.plastic_fallback_tracker.items()):
            distance = self.calculate_bbox_distance(bbox, tracked_bbox)
            if distance < self.bbox_similarity_threshold:
                # Update the bbox position and last seen time
                self.plastic_fallback_tracker[bbox] = track_id
                self.fallback_last_seen[track_id] = self.frame_count
                # Remove old bbox
                if tracked_bbox != bbox:
                    del self.plastic_fallback_tracker[tracked_bbox]
                return track_id

        # Create new fallback track ID
        new_id = self.next_fallback_id
        self.next_fallback_id += 1
        self.plastic_fallback_tracker[bbox] = new_id
        self.fallback_last_seen[new_id] = self.frame_count
        return new_id

    def update_object_count(self, track_id: int, center_y: int, class_name: str,
                          width_cm: float, height_cm: float, size_category: str):
        """Update object count with size information"""
        if track_id is not None and track_id > 0:
            # Update last seen time for this track
            self.track_last_seen[track_id] = self.frame_count
            
            if track_id in self.track_history:
                prev_y = self.track_history[track_id]

                # Check if object crossed the line
                if prev_y < self.line_y <= center_y and track_id not in self.counted_objects:
                    # Initialize counter if needed
                    if class_name not in self.object_counter:
                        self.object_counter[class_name] = {"total": 0, "meso": 0, "makro": 0}

                    # Update counters
                    self.object_counter[class_name]["total"] += 1
                    if size_category in ["meso", "makro"]:
                        self.object_counter[class_name][size_category] += 1

                    self.counted_objects.add(track_id)

                    # Log measurement
                    log_entry = f"{class_name} ({size_category}): {width_cm:.1f}x{height_cm:.1f}cm"
                    self.measurement_log.append(log_entry)
                    print(f"COUNTED: {log_entry}")

            # Update track history
            self.track_history[track_id] = center_y

    def process_hailo_detections(self, hailo_detections: List, width: int, height: int, frame: np.ndarray = None):
        """Process detections with size measurement and fallback tracking"""
        n = len(hailo_detections)
        if n == 0:
            return None, []

        valid_detections = []

        for i, detection in enumerate(hailo_detections):
            conf = detection.get_confidence()
            cls_id = detection.get_class_id()
            class_name = self.class_names.get(cls_id, "unknown")

            if conf <= self.detection_threshold:
                continue

            # Get bounding box - coordinates from Hailo are relative to original distorted frame
            bbox = detection.get_bbox()
            x1 = bbox.xmin() * width
            y1 = bbox.ymin() * height
            x2 = bbox.xmax() * width
            y2 = bbox.ymax() * height
            original_bbox = (int(x1), int(y1), int(x2), int(y2))

            # Transform coordinates if using calibration
            if self.use_calibration and self.calibration_maps_ready:
                bbox_coords = self.undistort_detection_coordinates(original_bbox)
                # Skip detections that are outside valid region after undistortion
                if (bbox_coords[2] <= bbox_coords[0] or bbox_coords[3] <= bbox_coords[1] or
                    bbox_coords[0] >= self.roi_w or bbox_coords[1] >= self.roi_h):
                    continue
            else:
                bbox_coords = original_bbox

            # Get original track ID from Hailo
            original_track_id = 0
            try:
                track_objects = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                original_track_id = track_objects[0].get_id() if len(track_objects) > 0 else 0
            except:
                original_track_id = 0

            # Determine final track ID
            if original_track_id > 0:
                final_track_id = original_track_id
            else:
                if class_name == "plastic":
                    final_track_id = self.get_fallback_track_id(bbox_coords, class_name)
                else:
                    final_track_id = 0

            x1, y1, x2, y2 = bbox_coords
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            # Get accurate measurements if frame is available
            accurate_bbox = bbox_coords
            width_px = int(x2 - x1)
            height_px = int(y2 - y1)

            if frame is not None:
                try:
                    accurate_bbox, width_px, height_px = self.get_accurate_measurement(frame, bbox_coords)
                except Exception as e:
                    print(f"Error in accurate measurement: {e}")
                    # Fall back to original bbox
                    pass

            # Calculate size in cm
            width_cm = width_px * self.cm_per_pixel
            height_cm = height_px * self.cm_per_pixel
            size_category = self.categorize_size(width_cm, height_cm)

            detection_info = {
                'bbox': bbox_coords,
                'accurate_bbox': accurate_bbox,
                'track_id': final_track_id,
                'original_track_id': original_track_id,
                'label': class_name,
                'confidence': conf,
                'center_y': center_y,
                'center_x': center_x,
                'width_cm': width_cm,
                'height_cm': height_cm,
                'size_category': size_category,
                'width_px': width_px,
                'height_px': height_px
            }
            valid_detections.append(detection_info)

            # Update counting
            self.update_object_count(final_track_id, center_y, class_name,
                                   width_cm, height_cm, size_category)

        # Perform memory cleanup periodically
        if self.frame_count % self.cleanup_interval == 0:
            self.cleanup_memory()

        return None, valid_detections

    def draw_frame_overlay(self, frame: np.ndarray, width: int, height: int, fps: float, detections: List):
        """Draw visualization overlay with size information and memory status"""

        # Draw counting line
        if self.line_y is not None:
            cv2.line(frame, (0, self.line_y), (width, self.line_y), (0, 255, 0), 4)

        # Draw FPS, resolution, and distance info - ADJUSTED FONT SIZES
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show resolution on frame
        resolution_text = f"Resolution: {width}x{height}"
        if self.use_calibration and self.calibration_maps_ready:
            resolution_text += f" (calibrated)"
        cv2.putText(frame, resolution_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        distance_info = f"Distance: {self.jarak_kamera_sekarang_cm}cm | cm/px: {self.cm_per_pixel:.5f}"
        cv2.putText(frame, distance_info, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show calibration status
        calib_status = f"Camera Calibration: {'ON' if self.use_calibration else 'OFF'}"
        cv2.putText(frame, calib_status, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.use_calibration else (0, 0, 255), 2)

        # Show MQTT status
        mqtt_status = f"MQTT: {'CONNECTED' if self.mqtt_connected else 'DISCONNECTED'}"
        mqtt_color = (0, 255, 0) if self.mqtt_connected else (0, 0, 255)
        cv2.putText(frame, mqtt_status, (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mqtt_color, 2)

        # Show RTMP status
        rtmp_status = f"RTMP: {'CONNECTED' if self.streaming_enabled else 'DISCONNECTED'}"
        rtmp_color = (0, 255, 0) if self.streaming_enabled else (0, 0, 255)
        cv2.putText(frame, rtmp_status, (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, rtmp_color, 2)

        # Draw object counts with macro/meso breakdown
        y_offset = 210
        cv2.putText(frame, "=== COUNTING RESULTS ===", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        y_offset += 30
        for i, cls in enumerate(["plastic", "nonplastic"]):
            counts = self.object_counter.get(cls, {"total": 0, "makro": 0, "meso": 0})
            text = f"{cls}: Total={counts['total']} | Makro={counts['makro']} | Meso={counts['meso']}"
            color = self.get_label_color(cls)
            cv2.putText(frame, text, (10, y_offset + (i * 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw bounding boxes with size information
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            acc_x1, acc_y1, acc_x2, acc_y2 = det['accurate_bbox']
            track_id = det['track_id']
            label = det['label']
            confidence = det['confidence']
            center_y = det['center_y']
            center_x = det['center_x']
            width_cm = det['width_cm']
            height_cm = det['height_cm']
            size_category = det['size_category']

            # Get colors
            label_color = self.get_label_color(label)
            size_color = self.get_size_color(size_category)

            # Draw original HAILO bbox (thin line)
            if self.show_size_debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 1)
                cv2.putText(frame, "HAILO", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

            # Draw accurate bbox (thick line with size category color)
            cv2.rectangle(frame, (acc_x1, acc_y1), (acc_x2, acc_y2), size_color, 2)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 3, size_color, -1)

            # Draw size information - ADJUSTED FONT SIZES
            if track_id > 0:
                label_text = f"{label}({track_id}) {confidence*100:.0f}%"
            else:
                label_text = f"{label}(X) {confidence*100:.0f}%"

            size_text = f"{width_cm:.1f}x{height_cm:.1f}cm [{size_category}]"

            cv2.putText(frame, label_text, (acc_x1, acc_y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            cv2.putText(frame, size_text, (acc_x1, acc_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, size_color, 2)

            # Debug info for size comparison
            if self.show_size_debug:
                yolo_w, yolo_h = x2-x1, y2-y1
                yolo_w_cm, yolo_h_cm = yolo_w * self.cm_per_pixel, yolo_h * self.cm_per_pixel
                acc_w, acc_h = det['width_px'], det['height_px']
                debug_text = f"HAILO: {yolo_w_cm:.1f}x{yolo_h_cm:.1f}cm | Accurate: {width_cm:.1f}x{height_cm:.1f}cm | px:{acc_w}x{acc_h}"
                cv2.putText(frame, debug_text, (acc_x1, acc_y1 - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Additional debug: show cm_per_pixel
                debug_text2 = f"cm/px: {self.cm_per_pixel:.5f} | k: {self.k:.8f}"
                cv2.putText(frame, debug_text2, (acc_x1, acc_y1 - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def print_final_statistics(self):
        """Print final results with macro/meso breakdown and memory usage"""
        print("\n" + "="*50)
        print("=== HASIL COUNTING AKHIR HAILO ===")
        print("="*50)

        # Display resolution info in final stats
        if self.camera_width and self.camera_height:
            print(f"Resolusi Kamera Original: {self.camera_width} x {self.camera_height} pixels")
            if self.use_calibration and self.calibration_maps_ready:
                print(f"Resolusi Efektif (setelah kalibrasi): {self.roi_w} x {self.roi_h} pixels")
                print(f"Area yang dipotong: x={self.roi_x}, y={self.roi_y}")
            effective_width = self.roi_w if self.use_calibration and self.calibration_maps_ready else self.camera_width
            effective_height = self.roi_h if self.use_calibration and self.calibration_maps_ready else self.camera_height
            print(f"Field of View: {effective_width * self.cm_per_pixel:.1f}cm x {effective_height * self.cm_per_pixel:.1f}cm")
            print("-" * 50)

        total_objects = 0
        for cls, counts in self.object_counter.items():
            print(f"{cls}:")
            print(f"  Total: {counts['total']}")
            print(f"  Makro (≥2.5cm): {counts['makro']}")
            print(f"  Meso (0.5-2.5cm): {counts['meso']}")
            if counts['meso'] > 0:
                print(f"  Rasio Makro:Meso = {counts['makro']}:{counts['meso']}")
            total_objects += counts['total']
            print()

        print(f"TOTAL SEMUA OBJEK: {total_objects}")

        if self.fps_list:
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            print(f"Rata-rata FPS: {avg_fps:.2f}")

        print("\n=== LOG PENGUKURAN ===")
        for i, log in enumerate(self.measurement_log, 1):
            print(f"{i}. {log}")

        print(f"\nKalibrasi yang digunakan:")
        print(f"  - Konstanta k: {self.k:.8f}")
        print(f"  - Jarak kamera sekarang: {self.jarak_kamera_sekarang_cm} cm")
        print(f"  - cm_per_pixel: {self.cm_per_pixel:.5f}")
        print(f"  - Camera undistortion: {'ENABLED' if self.use_calibration else 'DISABLED'}")
        if self.use_calibration and self.calibration_maps_ready:
            print(f"  - Black area removal: ENABLED (cropped to {self.roi_w}x{self.roi_h})")
        print(f"  - MQTT Publishing: {'ENABLED' if self.mqtt_connected else 'DISABLED'}")
        if self.mqtt_connected:
            print(f"  - MQTT Broker: {self.mqtt_broker}:{self.mqtt_port}")
            print(f"  - MQTT Topic: {self.mqtt_topic}")
        print(f"  - RTMP Streaming: {'ENABLED' if self.streaming_enabled else 'DISABLED'}")
        if self.streaming_enabled:
            print(f"  - RTMP URL: {self.rtmp_url}")
            print(f"  - Frames streamed: {self.frames_streamed}")
        
        print("Metode: Hailo Detection + Contour Refinement + Distance Calibration + Camera Calibration + Memory Management + MQTT + RTMP")


def app_callback(pad, info, user_data: HailoObjectCounterMacroMeso):
    """Main callback with size measurement and resolution display"""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.frame_count += 1

    format, width, height = get_caps_from_pad(pad)

    # ====== DISPLAY CAMERA RESOLUTION ======
    if width is not None and height is not None:
        user_data.set_camera_resolution(width, height)

    if height is not None:
        user_data.set_line_position(height)

    current_fps = user_data.calculate_fps()

    frame = None
    force_frame_processing = user_data.streaming_enabled

    if (user_data.use_frame or force_frame_processing) and all([format, width, height]):
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Apply camera calibration and remove black areas
        if user_data.use_calibration and user_data.calibration_maps_ready:
            frame = user_data.apply_camera_calibration(frame)
            # Update dimensions after calibration
            if frame is not None and frame.size > 0:
                display_height, display_width = frame.shape[:2]
            else:
                display_width = width
                display_height = height
        else:
            display_width = width
            display_height = height

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Process detections with original frame dimensions
    # The coordinate transformation will be handled inside process_hailo_detections
    sv_detections, detection_list = user_data.process_hailo_detections(
        hailo_detections, width, height, frame
    )

    if (user_data.use_frame or force_frame_processing) and frame is not None and frame.size > 0:
        # Keep original frame for streaming before adding overlays
        original_frame_for_streaming = frame.copy() if user_data.streaming_enabled else None

        if user_data.use_frame:
            frame_height, frame_width = frame.shape[:2]
            user_data.draw_frame_overlay(frame, frame_width, frame_height, current_fps, detection_list)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(frame)

        # Stream the original frame (without overlays) to RTMP
        if user_data.streaming_enabled and original_frame_for_streaming is not None:
            user_data.stream_frame(original_frame_for_streaming)

    # Publish MQTT data periodically
    user_data.publish_mqtt_data()

    return Gst.PadProbeReturn.OK


def main():
    try:
        # Check if .env file exists
        if not os.path.exists('.env'):
            print("⚠️  .env file not found. Creating default .env file...")
            create_default_env()
            print("✅ Default .env file created. Please edit it if needed and restart the application.")
            return

        # Always use default calibration file path
        calibration_path = "camera_intrinsics.txt"

        user_data = HailoObjectCounterMacroMeso(
            calibration_path=calibration_path
        )
        user_data.use_frame = True

        app = GStreamerDetectionApp(app_callback, user_data)

        print("Starting Object Counter with Macro/Meso Classification, Memory Management, MQTT and RTMP Streaming")
        print("Camera resolution will be displayed once stream starts...")
        print("Memory management enabled with automatic cleanup")
        print("Press Ctrl+C to stop")

        app.run()

    except KeyboardInterrupt:
        print("\nApplication stopped")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'user_data' in locals():
            user_data.cleanup_mqtt()
            user_data.cleanup_rtmp_streaming()
            user_data.print_final_statistics()


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


if __name__ == "__main__":
    main()