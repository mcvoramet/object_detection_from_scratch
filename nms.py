import torch
from IoU import intersection_over_union


def nms(
        bboxes,
        iou_threshold,
        threshold,
        box_format="corners"
):

    """
    Compute Non-max Suppression (NMS)

    Parameters:
        bboxes (list): [[class, prob(bbx), x1, y1, x2, y2]]
        iou_threshold (float): IoU threshold value between 0.0 - 1.0
        threshold (float): Probability threshold value between 0.0 - 1.0
        box_format (str): "midpoint" or "corners", if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        list: final chosen bounding box after computed non-max suppression

    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # if they are not in the same class we don't want to compare them
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
                )
                < iou_threshold  # if the IoU less than some threshold we then want to keep that box
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

