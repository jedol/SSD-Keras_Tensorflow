import numpy as np
import cv2

from sample.image_transform import *
from utils.box import *


def random_distort_image(image, params={}):
    ## distort image
    if np.random.uniform() < 0.5:
        ## random brightness
        if params.has_key('brightness_prob'):
            image = random_brightness(image,
                                      params['brightness_prob'],
                                      params.get('brightness_delta', 32))

        ## random contrast
        if params.has_key('contrast_prob'):
            image = random_contrast(image,
                                    params['contrast_prob'],
                                    params.get('contrast_lower', 0.5),
                                    params.get('contrast_upper', 1.5))

        ## random saturation
        if params.has_key('saturation_prob'):
            image = random_saturation(image,
                                      params['saturation_prob'],
                                      params.get('saturation_lower', 0.5),
                                      params.get('saturation_upper', 1.5))

        ## random hue
        if params.has_key('hue_prob'):
            image = random_hue(image,
                               params['hue_prob'],
                               params.get('hue_delta', 36))
    else:
        ## random brightness
        if params.has_key('brightness_prob'):
            image = random_brightness(image,
                                      params['brightness_prob'],
                                      params.get('brightness_delta', 32))

        ## random saturation
        if params.has_key('saturation_prob'):
            image = random_saturation(image,
                                      params['saturation_prob'],
                                      params.get('saturation_lower', 0.5),
                                      params.get('saturation_upper', 1.5))

        ## random hue
        if params.has_key('hue_prob'):
            image = random_hue(image,
                               params['hue_prob'],
                               params.get('hue_delta', 36))

        ## random contrast
        if params.has_key('contrast_prob'):
            image = random_contrast(image,
                                    params['contrast_prob'],
                                    params.get('contrast_lower', 0.5),
                                    params.get('contrast_upper', 1.5))

    return image


def random_expand_image(image, fill_value=0, params={}):
    if params.has_key('expand_prob') and params.has_key('max_expand_ratio'):
        if np.random.uniform() < params['expand_prob'] and not abs(params['max_expand_ratio']-1.0) < 1e-2:
            expand_ratio = np.random.uniform(1.0, params['max_expand_ratio'])
            image, _ = expand_image(image, expand_ratio, fill_value)

    return image


def random_expand_image_with_nbboxes(image, nbboxes, fill_value=0, params={}):
    if params.has_key('expand_prob') and params.has_key('max_expand_ratio'):
        if np.random.uniform() < params['expand_prob'] and not abs(params['max_expand_ratio']-1.0) < 1e-2:
            h,w = image.shape[:2]
            expand_ratio = np.random.uniform(1.0, params['max_expand_ratio'])
            image, (x_off, y_off) = expand_image(image, expand_ratio, fill_value)
            h2,w2 = image.shape[:2]
            nbboxes = (nbboxes*[w,h,w,h]+[x_off,y_off,x_off,y_off])/[w2,h2,w2,h2]

    return image, nbboxes


def get_sampled_nbboxes(nbboxes, params={}):
    max_sample = params.get('max_sample', 1)
    max_trials = params.get('max_trials', 50)
    min_scale = params.get('min_scale', 1.0)
    max_scale = params.get('max_scale', 1.0)
    min_aspect_ratio = params.get('min_aspect_ratio', 1.0)
    max_aspect_ratio = params.get('max_aspect_ratio', 1.0)

    scale = np.random.uniform(min_scale, max_scale, max_trials)
    aspect_ratio = np.random.uniform(min_aspect_ratio, max_aspect_ratio, max_trials)

    bbox_width = scale*np.sqrt(aspect_ratio)
    bbox_height = scale/np.sqrt(aspect_ratio)

    w_off = np.random.uniform(0, 1-bbox_width)
    h_off = np.random.uniform(0, 1-bbox_height)

    sampled_nbboxes = np.vstack((
        w_off, h_off, w_off+bbox_width, h_off+bbox_height
    )).transpose()

    mask = np.ones(len(sampled_nbboxes), np.bool)
    if params.has_key('min_jaccard_overlap') or params.has_key('max_jaccard_overlap'):
        overlaps = jaccard_overlap(sampled_nbboxes, nbboxes)
        if params.has_key('min_jaccard_overlap'):
            mask = np.logical_and(mask, np.any(overlaps[mask] >= params['min_jaccard_overlap'], axis=1))
        if params.has_key('max_jaccard_overlap'):
            mask = np.logical_and(mask, np.any(overlaps[mask] <= params['max_jaccard_overlap'], axis=1))

    sampled_nbboxes = sampled_nbboxes[mask]
    if len(sampled_nbboxes):
        rand_inds = np.random.permutation(len(sampled_nbboxes))[:max_sample]
        return sampled_nbboxes[rand_inds].tolist()
    else:
        return []


def random_sampling_with_nbboxes_and_labels(image, nbboxes, labels, params={}):
    if params.has_key('cases') and len(params.get('cases', [])):
        sampled_bboxes = list()
        for case in params['cases']:
            sampled_bboxes.extend(get_sampled_nbboxes(nbboxes, case))
        sampled_bboxes = np.array(sampled_bboxes)

        if len(sampled_bboxes):
            ## choose one sampled bbox
            sampled_bbox = sampled_bboxes[np.random.randint(0,len(sampled_bboxes))]

            ## constraint check for object bboxes
            mask = constraint_check(nbboxes, sampled_bbox)
            nbboxes = nbboxes[mask]
            labels = labels[mask]

            ## crop image
            h, w = image.shape[:2]
            sampled_bbox = np.int32(clip_box(sampled_bbox)*[w,h,w,h])
            image = image[sampled_bbox[1]:sampled_bbox[3],sampled_bbox[0]:sampled_bbox[2]]
            x_off,y_off = sampled_bbox[:2]
            h2,w2 = image.shape[:2]
            nbboxes = (nbboxes*[w,h,w,h]-[x_off,y_off,x_off,y_off])/[w2,h2,w2,h2]
            nbboxes = clip_box(nbboxes)

    return image, nbboxes, labels


def resize_image(image, params={}):
    if params.has_key('height') and params.has_key('width'):
        if np.random.uniform() < params.get('resize_prob', 1.0):
            interpolation_map = {
                'LINEAR': cv2.INTER_LINEAR,
                'AREA': cv2.INTER_AREA,
                'NEAREST': cv2.INTER_NEAREST,
                'CUBIC': cv2.INTER_CUBIC,
                'LANCZOS4': cv2.INTER_LANCZOS4,
            }
            interp = interpolation_map[np.random.choice(params.get('interpolation', ['LINEAR']))]
            image,_ = resize_image_by_warp(image, params['width'], params['height'], interp)

    return image


def random_flip_image(image, params={}):
    if params.has_key('flip_prob'):
        if np.random.uniform() < params['flip_prob']:
            image = image[:,::-1]

    return image


def random_flip_image_with_nbboxes(image, nbboxes, params={}):
    if params.has_key('flip_prob'):
        if np.random.uniform() < params['flip_prob']:
            image = image[:,::-1]
            nbboxes[:,::2] = 1-nbboxes[:,2::-2]

    return image, nbboxes


def transform_for_train(image, labels, nbboxes, params={}):
    ## Input
    ##  image: array = (h,w,c)
    ##  labels: array = (n,)
    ##  nbboxes: array = (n,4)
    ##  params: dict
    ##      ...

    ## distort image
    if params.has_key('distort'):
        image = random_distort_image(
            image,
            params=params['distort']
        )

    ## expand image
    if params.has_key('expand'):
        image, nbboxes = random_expand_image_with_nbboxes(
            image,
            nbboxes,
            fill_value=params.get('per_channel_mean', 0),
            params=params['expand']
        )

    ## random sampling
    if params.has_key('sample'):
        image, nbboxes, labels = random_sampling_with_nbboxes_and_labels(
            image,
            nbboxes,
            labels,
            params=params['sample']
        )

    ## resize image
    if params.has_key('resize'):
        image = resize_image(
            image,
            params=params['resize']
        )

    ## flip image
    if params.has_key('flip'):
        image, nbboxes = random_flip_image_with_nbboxes(
            image,
            nbboxes,
            params=params['flip']
        )

    ## substract mean
    if params.has_key('per_channel_mean'):
        image = image.astype(np.float32)-np.array(params['per_channel_mean'], np.float32)

    return image, labels, nbboxes


def transform_for_test(image, params={}):
    ## Input
    ##  image: array = (h,w,c)
    ##  params: dict
    ##      ...

    ## distort image
    if params.has_key('distort'):
        image = random_distort_image(
            image,
            params=params['distort']
        )

    ## expand image
    if params.has_key('expand'):
        image = random_expand_image(
            image,
            fill_value=params.get('per_channel_mean', 0),
            params=params['expand']
        )

    ## TODO: random_crop_image

    ## resize image
    if params.has_key('resize'):
        image = resize_image(
            image,
            params=params['resize']
        )

    ## flip image
    if params.has_key('flip'):
        image, nbboxes = random_flip_image(
            image,
            params=params['flip']
        )

    ## substract mean
    if params.has_key('per_channel_mean'):
        image = image.astype(np.float32)-np.array(params['per_channel_mean'], np.float32)

    return image

