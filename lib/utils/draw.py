import numpy as np
import cv2


def draw_boxes(image, boxes, color=(0,255,0), linewidth=1):
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    for box in boxes.astype(np.int32):
        pt1 = tuple(box[:2].tolist())
        pt2 = tuple(box[2:].tolist())
        image = cv2.rectangle(image, pt1, pt2, color, thickness=linewidth)

    return image