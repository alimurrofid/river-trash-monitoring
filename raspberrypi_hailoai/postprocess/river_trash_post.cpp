#include "hailo_objects.hpp"

std::vector<HailoROIPtr> crop_detections(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    for (auto &det : detections)
    {
        if (det->get_label() == "nonplastic" || det->get_label() == "plastic")
        {
            crop_rois.push_back(det);
        }
    }

    return crop_rois;
}
