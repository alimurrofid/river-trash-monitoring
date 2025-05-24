import cv2
import time
import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, 
                           ConfigureParams, InferVStreams, InputVStreamParams, 
                           OutputVStreamParams)

class HailoObjectCounter:
    def __init__(self, hef_path, video_path):
        self.hef_path = hef_path
        self.video_path = video_path
        
        # Initialize Hailo device
        self.device = Device()
        self.hef = HEF(hef_path)
        self.network_group = self.device.configure(self.hef)[0]
        
        # Get input/output stream info
        self.input_vstreams = self.network_group.input_vstreams
        self.output_vstreams = self.network_group.output_vstreams
        
        # Get model input shape
        self.input_shape = self.input_vstreams[0].shape
        self.model_height, self.model_width = self.input_shape[0], self.input_shape[1]
        
        # Initialize tracking variables
        self.object_counter = {}
        self.track_history = {}
        self.counted_objects = set()
        self.next_track_id = 1
        
        # Class names for your plastic detection model
        self.class_names = {
            0: 'nonplastic',
            1: 'plastic'
        }
        
    def preprocess_frame(self, frame):
        """Preprocess frame for Hailo model input"""
        # Resize frame to model input size
        resized = cv2.resize(frame, (self.model_width, self.model_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] and convert to float32
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def postprocess_output(self, outputs, original_shape, conf_threshold=0.5):
        """Process model outputs to get bounding boxes"""
        detections = []
        
        # This is a generic postprocessing - you may need to adjust based on your model's output format
        # Assuming outputs contain [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        
        for output in outputs:
            if len(output.shape) == 3:  # [batch, detections, features]
                for detection in output[0]:  # Remove batch dimension
                    if len(detection) >= 6:
                        x1, y1, x2, y2, conf, cls = detection[:6]
                        
                        if conf > conf_threshold:
                            # Scale coordinates back to original frame size
                            orig_h, orig_w = original_shape[:2]
                            x1 = int(x1 * orig_w / self.model_width)
                            y1 = int(y1 * orig_h / self.model_height)
                            x2 = int(x2 * orig_w / self.model_width)
                            y2 = int(y2 * orig_h / self.model_height)
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': int(cls),
                                'class_name': self.class_names.get(int(cls), f'class_{int(cls)}')
                            })
        
        return detections
    
    def simple_tracking(self, detections, frame_shape):
        """Simple centroid-based tracking"""
        tracked_objects = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Find closest existing track
            min_distance = float('inf')
            matched_track_id = None
            
            for track_id, (prev_x, prev_y) in self.track_history.items():
                distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                if distance < min_distance and distance < 100:  # Distance threshold
                    min_distance = distance
                    matched_track_id = track_id
            
            if matched_track_id is None:
                # Create new track
                matched_track_id = self.next_track_id
                self.next_track_id += 1
            
            # Update track history
            self.track_history[matched_track_id] = (center_x, center_y)
            
            detection['track_id'] = matched_track_id
            tracked_objects.append(detection)
        
        return tracked_objects
    
    def run_inference(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(self.video_path)
        cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        line_y = int(frame_height * 0.6)
        
        # FPS tracking
        prev_time = 0
        fps_list = []
        
        # Batch processing
        batch_interval = 2
        frame_count = 0
        last_detections = []
        
        # Start inference
        input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type='FLOAT32')
        output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type='FLOAT32')
        
        with InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                fps_list.append(fps)
                
                # Run inference every batch_interval frames
                if frame_count % batch_interval == 0:
                    # Preprocess frame
                    input_data = self.preprocess_frame(frame)
                    
                    # Run inference
                    outputs = infer_pipeline.infer({self.input_vstreams[0].name: input_data})
                    
                    # Postprocess outputs
                    detections = self.postprocess_output(list(outputs.values()), frame.shape)
                    
                    # Apply tracking
                    last_detections = self.simple_tracking(detections, frame.shape)
                else:
                    detections = last_detections
                
                # Draw counting line
                cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)
                
                # Process detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    track_id = detection['track_id']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    center_y = (y1 + y2) // 2
                    
                    # Count objects crossing the line
                    if track_id in self.track_history:
                        prev_center_y = self.track_history[track_id][1]
                        if prev_center_y < line_y <= center_y and track_id not in self.counted_objects:
                            self.object_counter[class_name] = self.object_counter.get(class_name, 0) + 1
                            self.counted_objects.add(track_id)
                            print(f"{class_name} bertambah: {self.object_counter[class_name]}")
                    
                    # Draw bounding box and label
                    label = f"{class_name} ({track_id}) Conf: {confidence*100:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Display counting results
                y_offset = 70
                for i, (cls, count) in enumerate(self.object_counter.items()):
                    text = f"{cls}: {count}"
                    cv2.putText(frame, text, (10, y_offset + (i * 30)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow("Object Counting", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print("Total objek yang dihitung:", self.object_counter)
        
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"Rata-rata FPS selama proses: {avg_fps:.2f}")

# Usage
if __name__ == "__main__":
    hef_model_path = "raspberrypi_hailoai/model/1865_y8.hef"
    video_path = "video/AE2X00017.mp4"
    
    counter = HailoObjectCounter(hef_model_path, video_path)
    counter.run_inference()