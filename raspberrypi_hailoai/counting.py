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


class HailoObjectCounterFixed(app_callback_class):
    """Fixed version - Handle invalid Track IDs for plastic objects"""

    def __init__(self, output_video_path=None):
        super().__init__()

        # Configuration
        self.detection_threshold = 0.1
        self.line_y_ratio = 0.7
        self.line_y = None

        # Video output configuration
        self.output_video_path = output_video_path
        self.video_writer = None
        self.video_initialized = False

        # Label colors
        self.label_colors = {
            "plastic": (0, 255, 255),      # Yellow
            "nonplastic": (0, 0, 255),     # Red
            "unlabeled": (255, 255, 0)     # Cyan
        }

        # Label mapping
        self.class_names = {
            0: "unlabeled",
            1: "nonplastic",
            2: "plastic"
        }

        # Object counting variables
        self.object_counter = {}
        self.track_history = {}
        self.counted_objects = set()

        # FPS tracking
        self.prev_time = 0
        self.fps_list = []

        # Frame processing
        self.frame_count = 0

        # FALLBACK TRACKING SYSTEM for plastic objects without valid Track ID
        self.next_fallback_id = 10000  # Start fallback IDs from 10000
        self.plastic_fallback_tracker = {}  # bbox -> fallback_track_id
        self.bbox_similarity_threshold = 50  # pixels

    def initialize_video_writer(self, width: int, height: int):
        """Initialize video writer for output"""
        if self.output_video_path and not self.video_initialized:
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(self.output_video_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory: {output_dir}")
                
                # Define codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path, 
                    fourcc, 
                    30.0,  # Fixed FPS at 30
                    (width, height)
                )
                self.video_initialized = True
                print(f"Video writer initialized: {self.output_video_path}")
                print(f"Output video resolution: {width}x{height} @ 30 FPS")
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
            return 0  # Only use fallback for plastic

        # Check if we can match with existing fallback tracker
        for tracked_bbox, track_id in self.plastic_fallback_tracker.items():
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

    def update_object_count(self, track_id: int, center_y: int, class_name: str):
        """Update object count"""
        if track_id is not None and track_id > 0:
            if track_id in self.track_history:
                prev_y = self.track_history[track_id]

                # Check if object crossed the line
                if prev_y < self.line_y <= center_y and track_id not in self.counted_objects:
                    self.object_counter[class_name] = self.object_counter.get(class_name, 0) + 1
                    self.counted_objects.add(track_id)

            # Update track history
            self.track_history[track_id] = center_y

    def process_hailo_detections(self, hailo_detections: List, width: int, height: int):
        """Process detections with fallback tracking for plastic objects"""
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
                # Valid track ID from Hailo
                final_track_id = original_track_id
            else:
                # Invalid track ID - use fallback for plastic
                if class_name == "plastic":
                    final_track_id = self.get_fallback_track_id(bbox_coords, class_name)
                else:
                    final_track_id = 0  # Don't track non-plastic without valid ID

            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            detection_info = {
                'bbox': bbox_coords,
                'track_id': final_track_id,
                'original_track_id': original_track_id,
                'label': class_name,
                'confidence': conf,
                'center_y': center_y,
                'center_x': center_x
            }
            valid_detections.append(detection_info)

            # Update counting
            self.update_object_count(final_track_id, center_y, class_name)

        return None, valid_detections

    def draw_frame_overlay(self, frame: np.ndarray, width: int, height: int, fps: float, detections: List):
        """Draw visualization overlay"""

        # Draw counting line
        if self.line_y is not None:
            cv2.line(frame, (0, self.line_y), (width, self.line_y), (0, 255, 0), 4)

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Draw object counts
        y_offset = 70
        all_classes = ["plastic", "nonplastic"]
        for i, cls in enumerate(all_classes):
            count = self.object_counter.get(cls, 0)
            text = f"{cls}: {count}"
            color = self.get_label_color(cls)
            cv2.putText(frame, text, (10, y_offset + (i * 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            label = det['label']
            confidence = det['confidence']
            center_y = det['center_y']
            center_x = det['center_x']

            color = self.get_label_color(label)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

            # Draw label
            if track_id > 0:
                label_text = f"{label} ({track_id}) {confidence*100:.1f}%"
            else:
                label_text = f"{label} (X) {confidence*100:.1f}%"

            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def print_final_statistics(self):
        """Print final results"""
        print("\n" + "="*40)
        print("FINAL RESULTS")
        print("="*40)
        print("Counted Objects:", self.object_counter)
        print(f"Total frames: {self.frame_count}")
        print(f"Objects that crossed line: {len(self.counted_objects)}")
        if self.output_video_path:
            print(f"Output video: {self.output_video_path}")
        print("="*40)


def app_callback(pad, info, user_data: HailoObjectCounterFixed):
    """Main callback"""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    user_data.frame_count += 1

    format, width, height = get_caps_from_pad(pad)

    if height is not None:
        user_data.set_line_position(height)

    current_fps = user_data.calculate_fps()

    frame = None
    if user_data.use_frame and all([format, width, height]):
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Initialize video writer if needed
        if not user_data.video_initialized and user_data.output_video_path:
            user_data.initialize_video_writer(width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    sv_detections, detection_list = user_data.process_hailo_detections(
        hailo_detections, width, height
    )

    if user_data.use_frame and frame is not None:
        user_data.draw_frame_overlay(frame, width, height, current_fps, detection_list)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to video if output is enabled
        user_data.write_frame_to_video(frame)
        
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


def main():
    try:
        # Check if output video argument is provided
        output_video_path = None
        if len(os.sys.argv) > 1:
            for i, arg in enumerate(os.sys.argv):
                if arg == "--output-video" and i + 1 < len(os.sys.argv):
                    output_video_path = os.sys.argv[i + 1]
                    # Remove the output video arguments from sys.argv to avoid conflicts
                    os.sys.argv.remove(arg)
                    os.sys.argv.remove(output_video_path)
                    break

        user_data = HailoObjectCounterFixed(output_video_path=output_video_path)
        user_data.use_frame = True

        app = GStreamerDetectionApp(app_callback, user_data)

        print("Starting Object Counter")
        if output_video_path:
            print(f"Output video will be saved to: {output_video_path}")
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
            user_data.release_video_writer()
            user_data.print_final_statistics()


if __name__ == "__main__":
    main()