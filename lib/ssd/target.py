import numpy as np
from utils.box import jaccard_overlap
from ssd.prior_box import box_transform


def ssd_target_from_sample(prior_boxes_grid, labels, nbboxes, params={}):
    ## sanity check
    C = params.get('num_object_classes', 0)
    assert C > 0

    pos_thres = params.get('pos_overlap_threshold', 0.5)
    neg_thres = params.get('neg_overlap_threshold', pos_thres)
    assert pos_thres >= neg_thres

    bg_label_id = int(params.get('background_label_id', 0))
    assert bg_label_id >= 0

    ## insert background label
    if bg_label_id:
        labels[labels >= bg_label_id] += 1
    else:
        labels += 1
    C += 1

    ## check number of ground truth objects
    N = len(prior_boxes_grid) # number of priors
    M = len(nbboxes)          # number of ground truth objects
    if M:
        ## compute overlap
        overlap = jaccard_overlap(prior_boxes_grid, nbboxes) # N x M
        matched_prior_mask = np.zeros(N, np.bool)
        matched_gt_inds = np.full(N, -1, np.int32)

        ## per prediction matching
        max_gt_inds = overlap.argmax(axis=1)
        max_gt_overlaps = overlap[np.arange(N), max_gt_inds]
        matched_prior_mask |= max_gt_overlaps >= pos_thres
        matched_gt_inds[matched_prior_mask] = max_gt_inds[matched_prior_mask]

        ## bipartite matching
        all_inds_c = overlap.ravel().argsort()[::-1] # N*M
        i = 0
        bm_prior_inds = list()
        bm_gt_inds = list()
        while len(bm_gt_inds) < M:
            n,m = np.unravel_index(all_inds_c[i], (N,M))
            if m not in bm_gt_inds:
                bm_prior_inds.append(n)
                bm_gt_inds.append(m)
            i += 1
        matched_prior_mask[bm_prior_inds] = True
        matched_gt_inds[bm_prior_inds] = bm_gt_inds

        ## negative(background) sampling
        do_negative_sampling = abs(pos_thres-neg_thres) > 1e-3
        if do_negative_sampling:
            neg_prior_mask = max_gt_overlaps < neg_thres
            neg_prior_mask &= ~matched_prior_mask
        else:
            neg_prior_mask = ~matched_prior_mask

        ## assign classification target
        conf_target = np.full(N, -1, np.float32)
        conf_target[matched_prior_mask] = labels[matched_gt_inds[matched_prior_mask]]
        conf_target[neg_prior_mask] = bg_label_id

        ## assign localization target
        loc_target = np.zeros((N, 4), np.float32)
        loc_variance = params.get('loc_variance', [0.1, 0.1, 0.2, 0.2])
        delta = box_transform(prior_boxes_grid[matched_prior_mask],
                              nbboxes[matched_gt_inds[matched_prior_mask]],
                              loc_variance)
        loc_target[matched_prior_mask] = delta
    else:
        conf_target = np.full(N, bg_label_id, np.float32)
        loc_target = np.zeros((N, 4), np.float32)

    # return np.concatenate((loc_target,conf_target[...,None]), axis=1)
    return conf_target, loc_target
