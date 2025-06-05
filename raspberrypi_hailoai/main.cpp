#include <hailo/hailort.hpp>
#include <supervision/supervision.hpp>
#include <gst/gst.h>
#include <iostream>

int main(int argc, char *argv[])
{
    // Inisialisasi GStreamer
    gst_init(&argc, &argv);

    // Load config label river_trash.json dan model 1865_y11.hef (lokasi sesuai project kamu)
    std::string label_config_path = "/home/rivermonitor/Public/river-trash-monitoring/raspberrypi_hailoai/config/river_trash.json";
    std::string model_path = "/home/rivermonitor/Public/river-trash-monitoring/raspberrypi_hailoai/model/1865_y11.hef";

    // Buat pipeline GStreamer yang baca video input kamu
    std::string video_path = "/home/rivermonitor/Public/river-trash-monitoring/raspberrypi_hailoai/video/AE2X00017.mp4";

    // Setup Hailo device dan pipeline (pseudo-code):
    // hailo::Device device = hailo::Device::create_any();
    // hailo::Network network = device.create_network(model_path);
    // dan lain-lain...

    // Setup Supervision tracking + counting (lihat contoh hailo-rpi5-examples community projects)
    // Load postprocess library hasil compile kamu:
    // supervision::PostprocessLibrary postprocess_lib("path/to/libriver_trash_post.so");

    // Tracking dan counting untuk 2 label: nonplastic dan plastic
    // Buat garis hitung arah satu, hitung objek melewati garis berdasarkan label

    // (Kode lengkapnya bisa lebih panjang, ini contoh kerangka)

    std::cout << "River trash monitoring started...\n";

    // GStreamer loop, capture frame, jalankan inference, postprocess, tracking, counting...

    return 0;
}
