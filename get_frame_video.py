import os
import shutil

# Path ke folder hasil ekstraksi frame
input_base_folder = 'actioncam_checkerboard_frames/4k30fps'
# Folder tujuan untuk menyimpan frame ke-150
output_folder = 'actioncam_checkerboard_frames150/4k30fps'

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Ambil semua subfolder (nama video) di dalam input_base_folder
video_folders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]

for folder_name in video_folders:
    folder_path = os.path.join(input_base_folder, folder_name)
    frame_150_filename = f'{folder_name}_0149.jpg'  # karena indexing dari 0
    frame_150_path = os.path.join(folder_path, frame_150_filename)

    if os.path.exists(frame_150_path):
        # Copy frame ke-150 ke folder output
        shutil.copy(frame_150_path, os.path.join(output_folder, frame_150_filename))
        print(f'Berhasil mengambil frame ke-150 dari: {folder_name}')
    else:
        print(f'⚠️ Frame ke-150 tidak ditemukan di: {folder_name}')

print("Selesai mengambil semua frame ke-150.")
