import numpy as np


def mask_merge2(all_masks, all_scores, iou_thresh=0.7):
    # Expects a uint 8 scored matrix

    boxes = extract_bboxes((all_masks > 0.5).astype(int) * 255)

    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    ret_mask = []
    ret_scores = []
    counter = np.array(range(boxes.shape[0]))
    n = 0
    maxi = boxes.shape[0]
    while n < maxi:
        if counter[n] == -1:
            n += 1
            continue
        else:
            i = counter[n]
            iou = compute_iou(boxes[i], boxes[1:], area[i], area[1:])
            idxs = np.where(iou > iou_thresh)[0] + 1
            if len(idxs) > 2:
                inst_mask = all_masks[idxs]
                inst_mask = np.mean(inst_mask, axis=0)
                ret_mask.append(inst_mask)
                inst_scores = all_scores[idxs]
                inst_scores = np.mean(inst_scores, axis=0)
                ret_scores.append(inst_scores)

            counter[np.isin(counter, idxs)] = -1

        n += 1

    ret_mask = np.array(ret_mask)
    ret_scores = np.array(ret_scores)

    return ret_mask, ret_scores


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            boxes[i] = np.array([y1, x1, y2, x2])
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
    return boxes.astype(np.int32)
