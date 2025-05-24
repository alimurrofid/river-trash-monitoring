import os
import glob

# Folder output frame
output_base_folder = 'actioncam_checkerboard_frames/1080p60fps'

# Ambil semua subfolder (satu subfolder per video)
video_folders = [f for f in os.listdir(output_base_folder) if os.path.isdir(os.path.join(output_base_folder, f))]

if not video_folders:
    print("❌ Tidak ada subfolder ditemukan di:", output_base_folder)
else:
    print("📁 Memverifikasi jumlah frame di setiap folder video:\n")
    total_files = 0
    for folder in sorted(video_folders):
        folder_path = os.path.join(output_base_folder, folder)
        frame_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        num_frames = len(frame_files)
        total_files += num_frames
        print(f"📂 {folder:<30} -> {num_frames:>4} frame")

    print("\n✅ Total semua frame:", total_files)
