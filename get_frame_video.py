import os
import shutil

# Path ke folder hasil ekstraksi frame
input_base_folder = 'camera_calibrations/calib_checkerboard_frames'
# Folder tujuan untuk menyimpan frame ke-100
output_folder = 'camera_calibrations/calib_checkerboard_frames100'

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Ambil semua subfolder (nama video) di dalam input_base_folder
video_folders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]

for folder_name in video_folders:
    folder_path = os.path.join(input_base_folder, folder_name)
    frame_100_filename = f'{folder_name}_0099.jpg'  # karena indexing dari 0
    frame_100_path = os.path.join(folder_path, frame_100_filename)

    if os.path.exists(frame_100_path):
        # Copy frame ke-100 ke folder output
        shutil.copy(frame_100_path, os.path.join(output_folder, frame_100_filename))
        print(f'Berhasil mengambil frame ke-100 dari: {folder_name}')
    else:
        print(f'⚠️ Frame ke-100 tidak ditemukan di: {folder_name}')

print("Selesai mengambil semua frame ke-100.")
