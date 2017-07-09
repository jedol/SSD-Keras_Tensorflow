from _init_path import *
from config import cfg
import os
import numpy as np
import cv2
import timeit as ti

from sample.batch_manager import TrainBatchManager
from ssd.prior_box import box_transform_inv
from utils.draw import draw_boxes
from utils.mics import grid_image


#*************************************************************************************#
## all arguments are loaded from config.py.
## edit config.py first!
#*************************************************************************************#

## we show only 4 samples at a time
cfg['train_generator']['batch_size'] = 4
cfg['train_generator']['shuffle'] = False

## create network to get prior boxes
print 'Constructing model: {}'.format(cfg['model']['model_name'])
if cfg['model']['model_name'] == 'VGG16':
    from ssd.models.VGG import VGG16_VOC as net_creator
elif cfg['model']['model_name'] == 'ResNet50':
    from ssd.models.ResNet import ResNet50_VOC as net_creator
else:
    raise LookupError
net = net_creator(cfg['model']['image_size'],
                  cfg['model']['num_object_classes'],
                  cfg['model']['background_label_id'],
                  cfg['optimize']['weight_decay'])
prior_boxes = net.get_prior_boxes()

## mean substraction
if net.get_summary().has_key('per_channel_mean'):
    per_channel_mean = net.get_summary()['per_channel_mean']

if 'per_channel_mean' in locals():
    cfg['train_generator']['transform']['per_channel_mean'] = per_channel_mean

## train sample generator
tbm = TrainBatchManager(prior_boxes=prior_boxes, **cfg['train_generator'])
train_generator = tbm.get_generator()

## insert background class name
ind_to_class = cfg['dataset']['ind_to_class']
background_label_id = cfg['train_generator']['target']['background_label_id']
ind_to_class.insert(background_label_id, 'background')

## show samples
cv2.namedWindow('1')
while True:
    images, conf_targets, loc_targets = train_generator.next()
    if 'per_channel_mean' in locals():
        images += per_channel_mean
    images = images.astype(np.uint8)

    for i in range(len(images)):
        image = images[i].copy()
        h,w = image.shape[:2]
        conf_target = conf_targets[i]
        loc_target = loc_targets[i]
        matched_prior_mask = (conf_target != -1) & (conf_target != background_label_id)
        matched_prior_boxes = prior_boxes[matched_prior_mask]
        matched_prior_boxes = np.int32(matched_prior_boxes*[w,h,w,h])
        image = draw_boxes(image, matched_prior_boxes, (0,0,255))
        nbboxes = box_transform_inv(prior_boxes[matched_prior_mask],
                                    loc_target[matched_prior_mask],
                                    cfg['train_generator']['target'].get('loc_variance', [0.1, 0.1, 0.2, 0.2]))
        bboxes = np.int32(nbboxes*[w,h,w,h])
        image = draw_boxes(image, bboxes, (0,255,0))
        images[i] = image
        labels = conf_target[matched_prior_mask].astype(np.int32)
        classes = np.array([ind_to_class[label] for label in labels])
        print classes

    if len(images) == 1:
        image = images[0]
    else:
        image = grid_image(images)

    cv2.imshow('1', image)
    key = cv2.waitKey() & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()

## check speed
start = ti.default_timer()
for _ in range(100):
    _ = train_generator.next()
end = ti.default_timer()
print '{:.4f} sec/batch'.format((end-start)/100)


