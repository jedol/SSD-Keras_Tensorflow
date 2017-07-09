import numpy as np
import tensorflow as tf


def constraint_check(boxes, constraint_box):
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    x_center = boxes[:,::2].mean(axis=1)
    y_center = boxes[:,1::2].mean(axis=1)
    return (x_center >= constraint_box[0]) &\
           (x_center <= constraint_box[2]) &\
           (y_center >= constraint_box[1]) &\
           (y_center <= constraint_box[3])


def clip_box(boxes, boundary_rect=[0,0,1,1]):
    ## Input
    ##  boxes: 1d or 2d array = n*(x1,y1,x2,y2)
    ##  rect: tuple = (x1,y1,x2,y2)
    do_squeeze = False
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis,:]
        do_squeeze = True

    clipped_boxes = np.zeros(boxes.shape, boxes.dtype)
    clipped_boxes[:,::2] = np.clip(boxes[:,::2], boundary_rect[0], boundary_rect[2])
    clipped_boxes[:,1::2] = np.clip(boxes[:,1::2], boundary_rect[1], boundary_rect[3])

    if do_squeeze:
        clipped_boxes = np.squeeze(clipped_boxes)
    return clipped_boxes


def jaccard_overlap(boxes1, boxes2):
    """
    compute jaccard overlap(IOU)
    boxes should be normalized [0,1)
    """
    ## Input
    ##  boxes1: 1d or 2d array = n*(x1,y1,x2,y2)
    ##  boxes2: 1d or 2d array = n*(x1,y1,x2,y2)
    do_squeeze = False
    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis]
        do_squeeze = True
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis]
        do_squeeze = True

    boxes1_w = np.maximum(0., boxes1[:, 2] - boxes1[:, 0])
    boxes1_h = np.maximum(0., boxes1[:, 3] - boxes1[:, 1])
    boxes1_area = boxes1_w * boxes1_h

    boxes2_w = np.maximum(0., boxes2[:, 2] - boxes2[:, 0])
    boxes2_h = np.maximum(0., boxes2[:, 3] - boxes2[:, 1])
    boxes2_area = boxes2_w * boxes2_h

    inter_tl = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_br = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_w = np.maximum(0., inter_br[:, :, 0] - inter_tl[:, :, 0])
    inter_h = np.maximum(0., inter_br[:, :, 1] - inter_tl[:, :, 1])
    inter_area = inter_w * inter_h

    iou = inter_area / (boxes1_area[:, None] + boxes2_area[None, :] - inter_area)
    if do_squeeze:
        iou = np.squeeze(iou)
    return iou


def nms(boxes, confs, max_output, thres=0.3):
    ## sanity check
    if boxes.ndim == 1:
        return [0]
    if len(boxes) == 1:
        return [0]

    ## sort by confidence
    box_inds = np.argsort(confs)

    ## suppress non-maximum
    survived_box_inds = list()
    while len(box_inds) > 0 and len(survived_box_inds) < max_output:
        ## Among the remaning boxes, a box with the highest confidence shold be survived
        survived_box_ind = box_inds[-1]
        survived_box_inds.append(survived_box_ind)
        box_inds = box_inds[:-1]

        ## retrieve boxes
        survived_box = boxes[survived_box_ind]
        other_boxes = boxes[box_inds]

        ## compute overlap
        overlaps = jaccard_overlap(other_boxes, survived_box)

        ## preserve under the threshold
        box_inds = box_inds[np.where(overlaps <= thres)[0]]

    return survived_box_inds


def nms_tf(boxes, confs, max_output, thres=0.3):
    g = tf.Graph()
    with g.as_default():
        boxes_T = tf.placeholder(tf.float32, (None, 4))
        confs_T = tf.placeholder(tf.float32, (None,))
        inds_T = tf.image.non_max_suppression(boxes=boxes_T, scores=confs_T,
                                              max_output_size=max_output,
                                              iou_threshold=thres)

    sess = tf.Session(graph=g)
    inds = sess.run(inds_T, {boxes_T:boxes, confs_T:confs})
    return inds













