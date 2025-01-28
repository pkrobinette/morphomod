"""
Object Detection (objdet) metrics for stegomarking removal.
"""
import torch
import torch.nn.functional as F


def _get_iou_bb(box1, box2):
    """Calculate the IoU (Intersection over Union) of two bounding boxes.

    Args:
        box1 (tuple): Coordinates of the first box in the format (x1, y1, x2, y2),
                      where (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.
        box2 (tuple): Coordinates of the second box in the same format as box1.

    Returns:
        float: The IoU between the two bounding boxes.
    """
    # unpack the coordinates of the boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # get box of intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # calc area of this union
    inter_area = inter_width * inter_height

    # calculate union
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area_box1 + area_box2 - inter_area # subtract union area

    iou = inter_area / union_area if union_area > 0 else 0
    
    return torch.tensor(iou)
    

def calc_iou_bb(box1, box2):
    """Calculate the IoU (Intersection over Union) of two batches of bounding boxes.

    Args:
        box1 (list of tuples): Coordinates of the first batch of boxes in the format (x1, y1, x2, y2).
        box2 (list of tuples): Coordinates of the second batch of boxes in the same format as box1.

    Returns:
        torch.Tensor: A tensor of IoU values for each pair of bounding boxes.
    """
    if type(box1[0]) != list:
        box1 = [box1]
    if type(box2[0]) != list:
        box2 = [box2]
    iou_bbs = [ _get_iou_bb(bx1, bx2) for bx1, bx2 in zip(box1, box2) ]
    return torch.stack(iou_bbs)