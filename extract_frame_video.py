import cv2
import os

# Folder input dan output
video_folder = 'camera_calibrations/calib_checkerboard'
output_base_folder = 'camera_calibrations/calib_checkerboard_frames'

# Buat folder output utama jika belum ada
os.makedirs(output_base_folder, exist_ok=True)

# Ekstensi file video yang didukung
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# Ambil semua file video dalam folder
video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(video_extensions)]

if not video_files:
    print("❌ Tidak ada file video ditemukan di folder:", video_folder)
else:
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # tanpa ekstensi
        output_folder = os.path.join(output_base_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        print(f"\n🔍 Membuka video: {video_file}")
        print(f"📍 Path: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Gagal membuka video: {video_file}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"🎞️ Total frame (dari metadata): {frame_count}, FPS: {fps}")

        success_count = 0
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"⛔ Gagal membaca frame ke-{frame_num} (kemungkinan video berakhir atau error).")
                break

            frame_filename = os.path.join(output_folder, f'{video_name}_{frame_num:04d}.jpg')
            success = cv2.imwrite(frame_filename, frame)

            if success:
                success_count += 1
            else:
                print(f"⚠️ Gagal menyimpan frame ke-{frame_num} sebagai gambar.")

            frame_num += 1

        cap.release()

        if success_count == 0:
            print(f"❌ Tidak ada frame yang berhasil diekstrak dari: {video_file}")
        else:
            print(f"✅ Berhasil ekstrak {success_count} frame dari: {video_file}")

print("\n🎉 Proses ekstraksi selesai untuk semua video.")
