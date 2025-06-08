# Hailo RPi5 Setup

## Installation

### Clone the Repository
```bash
git clone https://github.com/alimurrofid/river-trash-monitoring
```
Navigate to the repository directory:
```bash
cd raspberrypi_hailoai
```

### Installation
Run the following script to automate the installation process:
```bash
./install.sh
```

### Setup Environment
When opening a new terminal session, ensure you have sourced the environment setup script:
```bash
source setup_env.sh
```

#### Run the counting application:
```bash
python3 counting.py --labels-json resources/river-trash.json --hef-path model/1865_y11.hef --input video/test.mp4 --disable-sync --show-fps --use-frame
```
To close the application, press `Ctrl+C`.

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
python3 counting.py --labels-json resources/river-trash.json --hef-path model/1865_y11.hef --input usb --disable-sync --show-fps --use-frame
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```

For additional options, execute:
```bash
python counting.py --help
```

### Disable looping video
Find file `gstreamer_app.py`:
```bash
find / -type f -name "gstreamer_app.py" 2>/dev/null
```
Edit the file and change the function:
```python
    def on_eos(self):
        self.shutdown()
        # if self.source_type == "file":
        #      # Seek to the start (position 0) in nanoseconds
        #     success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
        #     if success:
        #         print("Video rewound successfully. Restarting playback...")
        #     else:
        #         print("Error rewinding the video.", file=sys.stderr)
        # else:
        #     self.shutdown()
```

