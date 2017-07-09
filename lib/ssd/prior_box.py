import numpy as np


def get_num_priors_list(min_sizes_list, max_sizes_list=[], aspect_ratios_list=[], flip=None):
    num_priors_list = list()
    for i in xrange(len(min_sizes_list)):
        num_min_sizes = len(min_sizes_list[i])
        num_default_priors = 1
        if len(max_sizes_list):
            num_max_sizes = len(max_sizes_list[i])
            assert num_min_sizes == num_max_sizes
            num_default_priors += 1
        num_aspect_ratios = len(aspect_ratios_list[i])
        if flip:
            num_aspect_ratios *= 2
        num_priors = num_min_sizes*(num_default_priors+num_aspect_ratios)
        num_priors_list.append(num_priors)
    return num_priors_list


def count_mbox_priors_grid(feature_map_size_list, num_priors_list):
    count = 0
    for i,fmap_size in enumerate(feature_map_size_list):
        num_priors = num_priors_list[i]
        count += fmap_size[0]*fmap_size[1]*num_priors
    return count


def make_prior_box_params(image_size, feature_map_size_list, min_sizes_list,
                          max_sizes_list=[], aspect_ratios_list=[], step_size_list=[],
                          offset=None, flip=None, clip=None):
    if not (isinstance(image_size, list) or isinstance(image_size, tuple)):
        image_size = (image_size, image_size)
    assert len(feature_map_size_list) == len(min_sizes_list)
    if len(max_sizes_list):
        assert len(max_sizes_list) == len(min_sizes_list)
    if len(aspect_ratios_list):
        assert len(aspect_ratios_list) == len(min_sizes_list)
    if len(step_size_list):
        assert len(step_size_list) == len(min_sizes_list)

    prior_box_params = list()
    for i in xrange(len(feature_map_size_list)):
        params = {
            'image_size': image_size,
            'feature_map_size': feature_map_size_list[i],
            'min_sizes': min_sizes_list[i]
        }
        if len(max_sizes_list):
            params['max_sizes'] = max_sizes_list[i]
        if len(aspect_ratios_list):
            params['aspect_ratios'] = aspect_ratios_list[i]
        if len(step_size_list):
            params['step_size'] = step_size_list[i]
        if offset is not None:
            params['offset'] = offset
        if flip is not None:
            params['flip'] = flip
        if clip is not None:
            params['clip'] = clip
        prior_box_params.append(params)

    return prior_box_params


def gen_prior_boxes(min_sizes, max_sizes, aspect_ratios):
    prior_sizes = list()
    for i in xrange(len(min_sizes)):
        ## first prior: aspect_ratio = 1, size = min_size
        min_size = min_sizes[i]
        prior_sizes.append((min_size, min_size))

        if len(max_sizes) > 0:
            ## second prior: aspect_ratio = 1, size = min_size
            max_size = max_sizes[i]
            prior_size = np.sqrt(min_size * max_size)
            prior_sizes.append((prior_size, prior_size))

        ## rest of priors
        for ar in aspect_ratios:
            prior_width = min_size * np.sqrt(ar)
            prior_height = min_size / np.sqrt(ar)
            prior_sizes.append((prior_width, prior_height))

    ## generate prior boxes
    prior_boxes = list()
    for w, h in prior_sizes:
        prior_boxes.append([-w/2.0, -h/2.0, w/2.0, h/2.0])
    prior_boxes = np.array(prior_boxes, np.float32)

    return prior_boxes


def gen_prior_boxes_grid(image_size, feature_map_size, min_sizes,
                         max_sizes=[], aspect_ratios=[],
                         step_size=None, offset=0.5, flip=False, clip=False):
    if len(min_sizes):
        if not (isinstance(image_size, list) or isinstance(image_size, tuple)):
            image_size = (image_size, image_size)
        img_w, img_h = image_size

        if not (isinstance(feature_map_size, list) or isinstance(feature_map_size, tuple)):
            feature_map_size = (feature_map_size, feature_map_size)
        fmap_w, fmap_h = feature_map_size

        if step_size is None:
            step_size = (float(img_w)/fmap_w, float(img_h)/fmap_h)
        else:
            if not (isinstance(step_size, list) or isinstance(step_size, tuple)):
                step_size = (step_size, step_size)
        step_w, step_h = step_size

        if len(max_sizes):
            assert len(max_sizes) == len(min_sizes)

        ## attach flipped aspect_ratios
        if flip and len(aspect_ratios):
            aspect_ratios = aspect_ratios+[1./ar for ar in aspect_ratios]

        ## generate prior boxes
        prior_boxes = gen_prior_boxes(min_sizes, max_sizes, aspect_ratios)

        ## replicate prior boxes
        dx, dy = np.meshgrid((np.arange(fmap_w)+offset)*step_w,
                             (np.arange(fmap_h)+offset)*step_h)
        delta = np.concatenate((dx[...,None], dy[...,None],
                                dx[...,None], dy[...,None]), axis=2)
        prior_boxes_grid = prior_boxes + np.expand_dims(delta, axis=2)

        ## normalize
        prior_boxes_grid /= [img_w, img_h, img_w, img_h]

        ## clip the prior's coordinate such that it is within [0, 1]
        if clip:
            prior_boxes_grid = np.clip(prior_boxes_grid, 0, 1.0)

        ## to list
        prior_boxes_grid = prior_boxes_grid.reshape(-1, 4).tolist()
    else:
        prior_boxes_grid = list()

    return prior_boxes_grid


def box_transform(prior_boxes, gt_boxes, variance=None):
    ## Input
    ##  prior_boxes: 1d or 2d array = n*(x1,y1,x2,y2)
    ##  gt_boxes: 1d or 2d array = n*(x1,y1,x2,y2)
    do_squeeze = False
    if prior_boxes.ndim == 1:
        prior_boxes = prior_boxes[np.newaxis,:]
        do_squeeze = True
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes[np.newaxis,:]
        do_squeeze = True

    prior_widths = prior_boxes[:, 2] - prior_boxes[:, 0]
    prior_heights = prior_boxes[:, 3] - prior_boxes[:, 1]
    prior_ctr_x = prior_boxes[:, 0] + 0.5 * prior_widths
    prior_ctr_y = prior_boxes[:, 1] + 0.5 * prior_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - prior_ctr_x) / prior_widths
    dy = (gt_ctr_y - prior_ctr_y) / prior_heights
    dw = np.log(gt_widths / prior_widths)
    dh = np.log(gt_heights / prior_heights)

    deltas = np.vstack((dx, dy, dw, dh)).transpose()

    if variance is not None:
        deltas /= variance

    if do_squeeze:
        deltas = np.squeeze(deltas)

    return deltas


def box_transform_inv(prior_boxes, deltas, variance=None):
    ## Input
    ##  prior_boxes: 1d or 2d array = n*(x1,y1,x2,y2)
    ##  deltas: 1d or 2d array = n*(dx,dy,dw,dh)
    do_squeeze = False
    if prior_boxes.ndim == 1:
        prior_boxes = prior_boxes[np.newaxis,:]
        do_squeeze = True
    if deltas.ndim == 1:
        deltas = deltas[np.newaxis,:]
        do_squeeze = True

    if variance is not None:
        deltas *= variance

    widths = prior_boxes[:, 2] - prior_boxes[:, 0]
    heights = prior_boxes[:, 3] - prior_boxes[:, 1]
    ctr_x = prior_boxes[:, 0] + 0.5 * widths
    ctr_y = prior_boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = np.vstack((pred_x1, pred_y1, pred_x2, pred_y2)).transpose()
    if do_squeeze:
        pred_boxes = np.squeeze(pred_boxes)
    return pred_boxes

