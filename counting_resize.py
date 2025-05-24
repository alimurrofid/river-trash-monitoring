import cv2
import time
import psutil
import subprocess
import json
import os
from datetime import datetime
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

class PerformanceMonitor:
    def __init__(self):
        self.timestamps = []
        self.fps_data = []
        self.cpu_data = []
        self.ram_data = []
        self.temp_data = []
        self.object_counts = {}
        self.detection_history = []
        self.start_time = time.time()
        
    def log_performance(self, fps, cpu_percent, ram_percent, temperature, detections):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        self.timestamps.append(elapsed_time)
        self.fps_data.append(fps)
        self.cpu_data.append(cpu_percent)
        self.ram_data.append(ram_percent)
        self.temp_data.append(temperature if temperature else 0)
        
        # Log detections
        detection_snapshot = {}
        for class_name, count in detections.items():
            detection_snapshot[class_name] = count
        self.detection_history.append(detection_snapshot)
    
    def generate_html_report(self, output_path="object_detection_report.html"):
        """Generate an interactive HTML report with performance metrics and detection results"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('FPS Over Time', 'CPU Usage (%)', 
                          'RAM Usage (%)', 'Temperature (°C)',
                          'Object Detection Counts', 'Detection Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # FPS Plot
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=self.fps_data, 
                      name='FPS', line=dict(color='blue')),
            row=1, col=1
        )
        
        # CPU Usage Plot
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=self.cpu_data, 
                      name='CPU %', line=dict(color='red')),
            row=1, col=2
        )
        
        # RAM Usage Plot
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=self.ram_data, 
                      name='RAM %', line=dict(color='green')),
            row=2, col=1
        )
        
        # Temperature Plot
        fig.add_trace(
            go.Scatter(x=self.timestamps, y=self.temp_data, 
                      name='Temperature', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Object Detection Counts (Bar Chart)
        if self.object_counts:
            classes = list(self.object_counts.keys())
            counts = list(self.object_counts.values())
            fig.add_trace(
                go.Bar(x=classes, y=counts, name='Object Counts',
                      marker=dict(color='purple')),
                row=3, col=1
            )
        
        # Detection Timeline (Stacked Area Chart)
        if self.detection_history:
            all_classes = set()
            for detection in self.detection_history:
                all_classes.update(detection.keys())
            
            colors = ['rgba(255,0,0,0.5)', 'rgba(0,255,0,0.5)', 'rgba(0,0,255,0.5)', 
                     'rgba(255,255,0,0.5)', 'rgba(255,0,255,0.5)', 'rgba(0,255,255,0.5)']
            
            for i, class_name in enumerate(all_classes):
                class_timeline = []
                for detection in self.detection_history:
                    class_timeline.append(detection.get(class_name, 0))
                
                fig.add_trace(
                    go.Scatter(x=self.timestamps[:len(class_timeline)], 
                             y=class_timeline,
                             mode='lines',
                             name=f'{class_name} Timeline',
                             fill='tonexty' if i > 0 else 'tozeroy',
                             line=dict(color=colors[i % len(colors)])),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Object Detection Performance Report",
            showlegend=True,
            height=1000,
            template="plotly_white"
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)")
        
        # Create summary statistics
        avg_fps = sum(self.fps_data) / len(self.fps_data) if self.fps_data else 0
        avg_cpu = sum(self.cpu_data) / len(self.cpu_data) if self.cpu_data else 0
        avg_ram = sum(self.ram_data) / len(self.ram_data) if self.ram_data else 0
        avg_temp = sum([t for t in self.temp_data if t > 0]) / len([t for t in self.temp_data if t > 0]) if any(t > 0 for t in self.temp_data) else 0
        
        # Create the complete HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Object Detection Performance Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px;
                    padding: 15px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Object Detection Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Runtime: {self.timestamps[-1]:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Performance Summary</h2>
                <div class="metric">
                    <div class="metric-value">{avg_fps:.2f}</div>
                    <div class="metric-label">Average FPS</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_cpu:.1f}%</div>
                    <div class="metric-label">Average CPU</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_ram:.1f}%</div>
                    <div class="metric-label">Average RAM</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_temp:.1f}°C</div>
                    <div class="metric-label">Average Temp</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="performance-charts"></div>
            </div>
            
            <script>
                var plotData = {fig.to_json()};
                Plotly.newPlot('performance-charts', plotData.data, plotData.layout);
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")
        return output_path

# Fungsi ambil suhu Raspberry Pi
def get_temperature():
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp_str = output.strip().split("=")[1].split("'")[0]
        return float(temp_str)
    except:
        return None

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Load model YOLO
model = YOLO("runs/dataset_clean_flip_retrain/y11n_batch16_epochs100/weights/best.pt")

# Buka video input
cap = cv2.VideoCapture("actioncam/AE2X00017.mp4")
cv2.namedWindow("Object Counting", cv2.WINDOW_NORMAL)

# Ukuran asli frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps_input = cap.get(cv2.CAP_PROP_FPS)
line_y = int(frame_height * 0.6)

# Tracking dan counting
object_counter = {}
track_history = {}
counted_objects = set()

# FPS
prev_time = 0
fps_list = []

# Light inference settings
batch_interval = 2
frame_count = 0
last_results = None
inference_size = (640, 360)

print("Starting object detection... Press 'q' to quit and generate report.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    fps_list.append(fps)

    # Logging performa
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    temperature = get_temperature()

    # Log to performance monitor
    performance_monitor.log_performance(fps, cpu_percent, ram_percent, temperature, object_counter)

    print(f"FPS: {fps:.2f} | CPU: {cpu_percent:.2f}% | RAM: {ram_percent:.2f}% | Suhu: {temperature if temperature else 'N/A'} C")

    # Resize frame untuk inference
    input_frame = cv2.resize(frame, inference_size)
    scale_x = frame.shape[1] / input_frame.shape[1]
    scale_y = frame.shape[0] / input_frame.shape[0]

    # Inference setiap N frame
    if frame_count % batch_interval == 0:
        results = model.track(input_frame, persist=True)
        last_results = results
    else:
        results = last_results

    # Gambar garis horizontal
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 2)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1, y1, x2, y2 = [int(coord * scale_x if i % 2 == 0 else coord * scale_y) for i, coord in enumerate([x1, y1, x2, y2])]

            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            center_y = int((y1 + y2) / 2)

            if track_id is not None:
                if track_id in track_history:
                    prev_y = track_history[track_id]
                    if prev_y < line_y <= center_y and track_id not in counted_objects:
                        object_counter[class_name] = object_counter.get(class_name, 0) + 1
                        counted_objects.add(track_id)
                        print(f"{class_name} bertambah: {object_counter[class_name]}")
                track_history[track_id] = center_y

            # Gambar bounding box dan label
            label = f"{class_name} ({track_id}) Conf: {confidence*100:.2f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Tampilkan FPS di frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Tampilkan hasil counting
    y_offset = 70
    for i, (cls, count) in enumerate(object_counter.items()):
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Object Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Update final object counts in performance monitor
performance_monitor.object_counts = object_counter

# Generate HTML report
report_path = performance_monitor.generate_html_report("object_detection_report.html")

# Print hasil akhir
print("Total objek yang dihitung:", object_counter)

# Rata-rata FPS
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Rata-rata FPS: {avg_fps:.2f}")

print(f"\nHTML report generated at: {os.path.abspath(report_path)}")
print("Open the HTML file in your browser to view the interactive report!")