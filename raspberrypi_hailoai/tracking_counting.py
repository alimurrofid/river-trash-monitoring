import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import numpy as np
import setproctitle
import hailo
import supervision as sv
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    get_caps_from_pad,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerApp

# --- Callback untuk setiap frame pipeline ---
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    n = len(hailo_detections)
    _, w, h = get_caps_from_pad(pad)

    boxes = np.zeros((n, 4))
    confidence = np.zeros(n)
    class_id = np.zeros(n)
    tracker_id = np.empty(n)

    for i, detection in enumerate(hailo_detections):
        class_id[i] = detection.get_class_id()
        confidence[i] = detection.get_confidence()
        tracker_id[i] = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)[0].get_id()
        bbox = detection.get_bbox()
        boxes[i] = [bbox.xmin() * w, bbox.ymin() * h, bbox.xmax() * w, bbox.ymax() * h]

    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id
    )

    # Trigger line zone hitung objek yang melewati garis dari bawah ke atas
    line_zone.trigger(detections)

    # Update teks overlay jumlah objek sesuai label
    textoverlay = app.pipeline.get_by_name("hailo_text")
    count_nonplastic = line_zone.in_count.get(0, 0)   # class_id=0 -> nonplastic
    count_plastic = line_zone.in_count.get(1, 0)      # class_id=1 -> plastic

    textoverlay.set_property('text', f'Plastic: {count_plastic} | Nonplastic: {count_nonplastic}')
    textoverlay.set_property('font-desc', 'Sans 36')

    return Gst.PadProbeReturn.OK


# --- Class aplikasi utama ---
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)

        # Set param model Hailo
        self.batch_size = 1
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"

        # Threshold untuk NMS dan confidence
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Path postprocess .so
        postprocess_path = os.path.join(os.getcwd(), 'resources/libriver_trash_post.so')
        self.default_postprocess_so = postprocess_path

        # HEF path
        if args.hef_path is not None:
            self.hef_path = args.hef_path
        else:
            self.hef_path = os.path.join(os.getcwd(), 'model/1865_y11.hef')

        # Load JSON label config
        if args.labels_json is not None:
            self.labels_config = f' config-path={args.labels_json} '
        else:
            self.labels_config = ''

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle("Hailo Plastic Detection App")
        self.create_pipeline()

    def get_pipeline_string(self):
        if self.source_type == "usb":
            source_element = (
                f"v4l2src device={self.video_source} name=src_0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
            )
        elif self.source_type == "file":
            source_element = (
                f"filesrc location={self.video_source} name=src_0 ! "
                "qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
                "video/x-raw, format=I420 ! "
            )
        else:
            source_element = (
                "libcamerasrc name=src_0 auto-focus-mode=2 ! "
                f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
                "videoscale ! "
                f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=60/1 ! "
            )

        source_element += "videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_str = (
            "hailomuxer name=hmux "
            + source_element
            + "tee name=t ! "
            + "hmux.sink_0 "
            + "t. ! "
            + "videoconvert n-threads=3 ! "
            + f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} {self.thresholds_str} force-writable=true ! "
            + f"hailofilter so-path={self.default_postprocess_so} {self.labels_config} qos=false ! "
            + "hailotracker keep-tracked-frames=3 keep-new-frames=3 keep-lost-frames=3 ! "
            + "hmux.sink_1 "
            + "hmux. ! "
            + "identity name=identity_callback ! "
            + "hailooverlay ! "
            + "videoconvert n-threads=3 qos=false ! "
            + "textoverlay name=hailo_text text='' valignment=top halignment=center ! "
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        )

        print(pipeline_str)
        return pipeline_str


if __name__ == "__main__":
    # Inisialisasi garis hitung (dari bawah ke atas)
    START = sv.Point(0, 460)
    END = sv.Point(640, 460)

    line_zone = sv.LineZone(start=START, end=END, triggering_anchors=(sv.Position.BOTTOM_LEFT, sv.Position.BOTTOM_RIGHT))
    # Note: line_zone.in_count akan bertambah setiap objek melewati garis dari bawah ke atas

    parser = get_default_parser()
    parser.add_argument("--hef-path", default=None, help="Path to HEF file")
    parser.add_argument("--labels-json", default=None, help="Path to label JSON file")
    args = parser.parse_args()

    user_data = app_callback

    app = GStreamerDetectionApp(args, user_data)
    app.run()
