import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import supervision as sv
from typing import Dict, Set, Tuple, Optional, List

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp


class HailoObjectCounterMacroMeso(app_callback_class):
    """Object counter with macro/meso classification based on size measurement"""

    def __init__(self):
        super().__init__()

        # Configuration
        self.detection_threshold = 0.1
        self.line_y_ratio = 0.7
        self.line_y = None

        # ====== DISTANCE CALIBRATION SETTINGS ======
        # Kalibrasi ukuran dengan jarak (sama seperti OpenCV version)
        self.ukuran_objek_cm = 4.1           # Ukuran real objek kalibrasi (cm)
        self.ukuran_objek_px = 69            # Ukuran objek di kamera saat kalibrasi (pixel)
        self.jarak_kalibrasi_cm = 80         # Jarak kamera ke objek saat kalibrasi (cm)
        self.jarak_kerja_cm = 80             # Jarak kamera saat penggunaan (cm)

        # Hitung konstanta kalibrasi
        self.k = self.ukuran_objek_cm / (self.ukuran_objek_px * self.jarak_kalibrasi_cm)
        self.cm_per_pixel = self.k * self.jarak_kerja_cm

        print(f"=== KALIBRASI JARAK HAILO ===")
        print(f"Ukuran objek kalibrasi: {self.ukuran_objek_cm} cm")
        print(f"Ukuran di kamera (kalibrasi): {self.ukuran_objek_px} pixel")
        print(f"Jarak kalibrasi: {self.jarak_kalibrasi_cm} cm")
        print(f"Jarak kerja saat ini: {self.jarak_kerja_cm} cm")
        print(f"Konstanta k: {self.k:.8f}")
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

        # Size measurement settings
        self.show_size_debug = True

        # ====== RESOLUTION TRACKING ======
        self.camera_width = None
        self.camera_height = None
        self.resolution_displayed = False  # Flag to show resolution only once

        # Initialize counter structure
        self._initialize_counters()

    def _initialize_counters(self):
        """Initialize counter structure for all classes"""
        for class_name in ["plastic", "nonplastic"]:
            if class_name not in self.object_counter:
                self.object_counter[class_name] = {
                    "total": 0,
                    "meso": 0,
                    "makro": 0
                }

    def set_camera_resolution(self, width: int, height: int):
        """Set and display camera resolution (only once)"""
        if not self.resolution_displayed:
            self.camera_width = width
            self.camera_height = height
            print(f"\n=== INFORMASI KAMERA ===")
            print(f"Resolusi Kamera: {width} x {height} pixels")
            print(f"Aspect Ratio: {width/height:.2f}:1")

            # Calculate total pixels
            total_pixels = width * height
            if total_pixels >= 1920*1080:
                quality = "Full HD (1080p)"
            elif total_pixels >= 1280*720:
                quality = "HD (720p)"
            elif total_pixels >= 640*480:
                quality = "VGA"
            else:
                quality = "Low Resolution"

            print(f"Total Pixels: {total_pixels:,} ({quality})")
            print(f"Field of View (estimasi): {width * self.cm_per_pixel:.1f}cm x {height * self.cm_per_pixel:.1f}cm")
            print("="*30)

            self.resolution_displayed = True

    def update_distance_calibration(self, new_distance_cm):
        """Update cm_per_pixel based on new distance"""
        self.jarak_kerja_cm = new_distance_cm
        self.cm_per_pixel = self.k * self.jarak_kerja_cm
        print(f"Distance updated to {self.jarak_kerja_cm}cm, new cm_per_pixel: {self.cm_per_pixel:.5f}")

        # Update field of view if resolution is known
        if self.camera_width and self.camera_height:
            print(f"New Field of View: {self.camera_width * self.cm_per_pixel:.1f}cm x {self.camera_height * self.cm_per_pixel:.1f}cm")

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
            self.line_y = int(height * self.line_y_ratio)

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
                # Update the bbox position
                self.plastic_fallback_tracker[bbox] = track_id
                # Remove old bbox
                if tracked_bbox != bbox:
                    del self.plastic_fallback_tracker[tracked_bbox]
                return track_id

        # Create new fallback track ID
        new_id = self.next_fallback_id
        self.next_fallback_id += 1
        self.plastic_fallback_tracker[bbox] = new_id
        return new_id

    def update_object_count(self, track_id: int, center_y: int, class_name: str,
                          width_cm: float, height_cm: float, size_category: str):
        """Update object count with size information"""
        if track_id is not None and track_id > 0:
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

            # Get bounding box
            bbox = detection.get_bbox()
            x1 = bbox.xmin() * width
            y1 = bbox.ymin() * height
            x2 = bbox.xmax() * width
            y2 = bbox.ymax() * height
            bbox_coords = (int(x1), int(y1), int(x2), int(y2))

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

        return None, valid_detections

    def draw_frame_overlay(self, frame: np.ndarray, width: int, height: int, fps: float, detections: List):
        """Draw visualization overlay with size information"""

        # Draw counting line
        if self.line_y is not None:
            cv2.line(frame, (0, self.line_y), (width, self.line_y), (0, 255, 0), 4)

        # Draw FPS, resolution, and distance info
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show resolution on frame
        resolution_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, resolution_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        distance_info = f"Distance: {self.jarak_kerja_cm}cm | cm/px: {self.cm_per_pixel:.5f}"
        cv2.putText(frame, distance_info, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw object counts with macro/meso breakdown
        y_offset = 120
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

            # Draw original YOLO bbox (thin line)
            if self.show_size_debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 1)
                cv2.putText(frame, "YOLO", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

            # Draw accurate bbox (thick line with size category color)
            cv2.rectangle(frame, (acc_x1, acc_y1), (acc_x2, acc_y2), size_color, 2)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 3, size_color, -1)

            # Draw size information
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
                debug_text = f"YOLO: {yolo_w_cm:.1f}x{yolo_h_cm:.1f}cm | Accurate: {width_cm:.1f}x{height_cm:.1f}cm"
                cv2.putText(frame, debug_text, (acc_x1, acc_y1 - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def print_final_statistics(self):
        """Print final results with macro/meso breakdown"""
        print("\n" + "="*50)
        print("=== HASIL COUNTING AKHIR HAILO ===")
        print("="*50)

        # Display resolution info in final stats
        if self.camera_width and self.camera_height:
            print(f"Resolusi Kamera: {self.camera_width} x {self.camera_height} pixels")
            print(f"Field of View: {self.camera_width * self.cm_per_pixel:.1f}cm x {self.camera_height * self.cm_per_pixel:.1f}cm")
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
        print(f"  - Jarak kerja: {self.jarak_kerja_cm} cm")
        print(f"  - cm_per_pixel: {self.cm_per_pixel:.5f}")
        print("Metode: Hailo Detection + Contour Refinement + Distance Calibration")


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
    if user_data.use_frame and all([format, width, height]):
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Pass frame to process_hailo_detections for accurate measurement
    sv_detections, detection_list = user_data.process_hailo_detections(
        hailo_detections, width, height, frame
    )

    if user_data.use_frame and frame is not None:
        user_data.draw_frame_overlay(frame, width, height, current_fps, detection_list)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


def main():
    try:
        user_data = HailoObjectCounterMacroMeso()
        user_data.use_frame = True

        app = GStreamerDetectionApp(app_callback, user_data)

        print("Starting Object Counter with Macro/Meso Classification")
        print("Camera resolution will be displayed once stream starts...")
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
            user_data.print_final_statistics()


if __name__ == "__main__":
    main()