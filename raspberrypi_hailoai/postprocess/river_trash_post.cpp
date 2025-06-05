/**
 * river_trash_post.cpp
 * Postprocess for river trash detection: nonplastic & plastic
 */

#include "hailo_objects.hpp"
#include <vector>
#include <string>

#define NONPLASTIC_LABEL "nonplastic"
#define PLASTIC_LABEL "plastic"

std::vector<HailoROIPtr> crop_detections(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    std::vector<HailoROIPtr> crop_rois;
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);

    for (auto &detection : detections_ptrs)
    {
        std::string label = detection->get_label();

        // Filter hanya label nonplastic dan plastic
        if ((label == NONPLASTIC_LABEL) || (label == PLASTIC_LABEL))
        {
            crop_rois.emplace_back(detection);
        }
    }
    return crop_rois;
}
