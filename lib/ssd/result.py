import numpy as np
from ssd.prior_box import box_transform_inv
from utils.box import clip_box, nms
from tfd.loop import predict
import timeit as ti
from utils.timer import Timer


def detect_images(net, sess, images, params, image_sizes=None, num_gpus=1, batch_size=32):
    ## Inputs
    ##  params: cfg['inference']
    ##      'nms_thres'
    ##      'nms_top_k'
    ##      'confidence_thres'
    ##      'keep_top_k'
    ##      'background_label_id'
    ##      'loc_variance'

    if images.ndim == 3:
        images = np.expand_dims(images, 0)

    assert images.ndim == 4 and images.shape[-1] == 3
    num_images = len(images)

    ## get deploy tensors
    images_T, mbox_conf_T, mbox_loc_T = net.get_deploy_tensors(num_gpus)

    ## inference
    start = ti.default_timer()
    mbox_conf, mbox_loc = predict(sess=sess,
                                  arrays=images,
                                  inputs=images_T,
                                  outputs=[mbox_conf_T, mbox_loc_T],
                                  batch_size=batch_size)
    end = ti.default_timer()
    print '  {:.4f} sec elapsed'.format(end-start)

    ## get prior boxes
    prior_boxes = net.get_prior_boxes()

    ## compute detections on validation samples
    sw = Timer(0, num_images)
    results = list()
    for i in xrange(num_images):
        sw.tic()
        if image_sizes is None:
            image_size = images[i].shape[1:3][::-1]
        else:
            image_size = image_sizes[i]
        detections = detections_from_mbox_scores(
            mbox_conf[i], mbox_loc[i], prior_boxes, image_size=image_size, **params)
        results.append({
            'objects': detections,
        })
        sw.toc()

    return results


def detections_from_mbox_scores(conf_score, loc_score, prior_boxes,
                                nms_thres=0.45, nms_top_k=400, confidence_thres=0.01, keep_top_k=200,
                                background_label_id=0, loc_variance=None, image_size=None):
    ## Inputs
    ##  conf_score: N x C array
    ##      N = number of prior boxes
    ##      C = number of classes(include background)
    ##  loc_score: N x 4 array
    ##  prior_boxes: N x 4 array
    C = conf_score.shape[1]

    ## compute bboxes through inverse box transform
    bboxes = box_transform_inv(prior_boxes, loc_score, loc_variance)
    bboxes = clip_box(bboxes, [0,0,1,1])

    mask = np.where(np.all(bboxes[:,:2] < bboxes[:,2:], axis=1))[0]
    bboxes = bboxes[mask]
    conf_score = conf_score[mask]

    ## non-maximum suppression on each class
    bg_label_offset = 0
    detections = list()
    for label in xrange(C):
        ## ignore background
        if label == background_label_id:
            bg_label_offset += 1
            continue

        ## discard scores are under the confidence_thres
        class_confs = conf_score[:,label]
        keep_mask = class_confs > confidence_thres
        class_confs = class_confs[keep_mask]
        class_bboxes = bboxes[keep_mask]

        if len(class_bboxes):
            selected_inds = nms(class_bboxes, class_confs, nms_top_k, nms_thres)
            for ind in selected_inds:
                ## unnormalize bbox, if needed
                bbox = class_bboxes[ind]
                if image_size is not None:
                    w,h = image_size
                    bbox *= [w,h,w,h]

                ## store detected object
                detections.append({
                    'bbox': bbox.tolist(),
                    'conf': float(class_confs[ind]),
                    'label': int(label-bg_label_offset),
                })

    ## sampling
    if 0 < keep_top_k < len(detections):
        detections = sorted(detections, key=lambda obj:obj['conf'], reverse=True)
        detections = detections[:keep_top_k]

    return detections