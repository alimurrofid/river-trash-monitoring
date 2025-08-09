"""
Simple camera mode selector with preset resolution options
and live preview functionality.

Features:
   - 4 preset camera modes (4K, 2K, 1080p variations)
   - Interactive menu selection
   - Live camera preview
   - Basic camera information display

Modes:
   1. 4K @ 30fps
   2. 2K @ 30fps  
   3. 1080p @ 60fps
   4. 1080p @ 30fps

Usage:
   - Run script and select mode from menu
   - Press 'q' to quit preview

Dependencies:
   - opencv-python
"""
import cv2

def set_camera_mode(cap, mode="1080p30"):
    presets = {
        "4k30":   (3840, 2160, 30),
        "2k30":   (2560, 1440, 30),
        "1080p60": (1920, 1080, 60),
        "1080p30": (1920, 1080, 30),
    }

    if mode not in presets:
        print(f"[WARNING] Mode '{mode}' tidak dikenal, menggunakan default 1080p30.")
        mode = "1080p30"

    width, height, fps = presets[mode]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return width, height, fps

def choose_mode():
    print("Pilih mode resolusi dan FPS kamera:")
    print("1. 4K @ 30fps (3840x2160)")
    print("2. 2K @ 30fps (2560x1440)")
    print("3. 1080p @ 60fps (1920x1080)")
    print("4. 1080p @ 30fps (1920x1080)")
    pilihan = input("Masukkan nomor pilihan (1-4): ")

    mapping = {
        "1": "4k30",
        "2": "2k30",
        "3": "1080p60",
        "4": "1080p30",
    }
    return mapping.get(pilihan, "1080p30")

def start_camera(device_index=0):
    mode = choose_mode()
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka kamera.")
        return

    width, height, fps = set_camera_mode(cap, mode)

    actual_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in reversed(range(4))])

    print("\n=== Informasi Kamera (OpenCV) ===")
    print(f"Mode yang dipilih: {mode}")
    print(f"Resolusi set     : {width}x{height}")
    print(f"FPS set          : {fps}")
    print(f"Resolusi aktual  : {int(actual_width)}x{int(actual_height)}")
    print(f"FPS aktual       : {actual_fps:.2f}")
    print(f"FOURCC Codec     : {codec if codec.isalnum() else 'Tidak tersedia'}")
    print("\n[INFO] Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame.")
            break

        cv2.imshow("Kamera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
