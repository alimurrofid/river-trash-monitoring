"""
Simple real-time object detection and measurement system
using basic computer vision techniques with webcam input.

Features:
   - Live webcam feed with configurable resolution
   - Real-time object detection using contour analysis
   - Pixel-based size measurement display
   - Camera properties monitoring (resolution, FPS, codec)
   - Basic binary thresholding for object isolation

Processing Pipeline:
   1. Capture video frame from default camera (index 0)
   2. Convert frame to grayscale for processing
   3. Apply binary inverse threshold to isolate objects
   4. Find external contours of detected objects
   5. Filter contours by minimum area (500 pixels)
   6. Draw bounding rectangles around valid objects
   7. Display pixel dimensions for each detected object

Configuration:
   - Target resolution: 1280x720
   - Target frame rate: 30 FPS
   - Minimum contour area: 500 pixels
   - Binary threshold value: 100 (inverse)
   - Bounding box color: Green (0, 255, 0)
   - Display window size: 1280x720 (resized if needed)

Camera Setup:
   - Uses default camera (VideoCapture index 0)
   - Attempts to set HD resolution and 30 FPS
   - Reports actual achieved camera properties
   - Displays FOURCC codec information

Dependencies:
   - opencv-python

Controls:
   - 'q': Quit application

Output:
   - Real-time video feed with object detection
   - Green bounding boxes around detected objects
   - Width and height measurements in pixels
   - Camera configuration information on startup
"""
import cv2

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"W:{w}px H:{h}px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize tampilan agar muat di layar
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Size Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
