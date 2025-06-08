
# 🧠 River Trash Monitoring with Hailo on Raspberry Pi 5

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-RaspberryPi5-blue)]()
[![Hailo](https://img.shields.io/badge/Hailo-AI--Accelerated-success)]()

Real-time object detection, tracking, counting, and size classification for river waste using YOLO on Hailo AI accelerator and Raspberry Pi 5. This system processes live video streams or pre-recorded footage to detect and count river waste categorized into `plastic` and `nonplastic`.

---

## 📁 Project Structure

```
river-trash-monitoring/
├── raspberrypi_hailoai/
│   ├── counting.py
│   ├── counting_makro_meso.py
│   ├── counting_debug.py
│   ├── model/
│   │   └── 1865_y11.hef
│   ├── resources/
│   │   └── river-trash.json
│   ├── video/
│   │   └── test.mp4
│   ├── output/
│   ├── install.sh
│   ├── requirements.txt
│   └── setup_env.sh
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/alimurrofid/river-trash-monitoring
cd river-trash-monitoring/raspberrypi_hailoai
```

### 2. Run the Installer

Make the install script executable and run it:

```bash
chmod +x install.sh
./install.sh
```

### 3. Setup the Environment

Before running any scripts, activate the environment:

```bash
source setup_env.sh
```

You must repeat this step in every new terminal session.

---

## ▶️ Running the Counting Application

### From Video File

```bash
python3 counting.py \
  --labels-json resources/river-trash.json \
  --hef-path model/1865_y11.hef \
  --input video/test.mp4 \
  --disable-sync \
  --show-fps \
  --use-frame
```

Press `Ctrl+C` to stop.

### Save Output to File

Save the detection result to `output/result.mp4`:

```bash
python3 counting.py \
  --labels-json resources/river-trash.json \
  --hef-path model/1865_y11.hef \
  --input video/test.mp4 \
  --disable-sync \
  --show-fps \
  --use-frame \
  --output-video output/result.mp4
```

---

## 📷 Running with USB Camera

### Option 1: Auto-detect

```bash
python3 counting.py \
  --labels-json resources/river-trash.json \
  --hef-path model/1865_y11.hef \
  --disable-sync \
  --show-fps \
  --use-frame \
  --input usb
```

### Option 2: Manually Check USB Cameras

Use this command to list connected USB cameras:

```bash
get-usb-camera
```

---

## 🧰 Additional Options

Display all available arguments:

```bash
python3 counting.py --help
```

---

## 🔁 Disable Video Loop (Optional)

To stop video files from looping at the end:

1. Locate the `gstreamer_app.py` file:
   ```bash
   find / -type f -name "gstreamer_app.py" 2>/dev/null
   ```

2. Edit the `on_eos()` function to:
   ```python
   def on_eos(self):
       self.shutdown()
       # Uncomment below to enable loop
       # if self.source_type == "file":
       #     success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
       #     if success:
       #         print("Video rewound successfully. Restarting playback...")
       #     else:
       #         print("Error rewinding the video.", file=sys.stderr)
       # else:
       #     self.shutdown()
   ```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Muhammad 'Ali Murrofid**  
GitHub: [@alimurrofid](https://github.com/alimurrofid)  
LinkedIn: [Muhammad 'Ali Murrofid](https://www.linkedin.com/in/muhammad-ali-murrofid-320a2b254/)

---

## ⭐️ Support the Project

If you find this project useful, feel free to give it a ⭐ on GitHub to show your support!
