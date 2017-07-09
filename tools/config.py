from _init_path import *
import os

from utils.mics import load_json


#*************************************************************************************#
## select one dataset used for training
dataset_name = 'VOC07'
# dataset_name = 'VOC07+12'
#*************************************************************************************#


## load dataset configuration
dataset_cfg_file = os.path.join(DATA_PATH, dataset_name, 'cfg.json')
if os.path.exists(dataset_cfg_file):
    cfg = load_json(dataset_cfg_file)
else:
    raise ValueError('Dataset must be dumped to LMDB first.')


#*************************************************************************************#
## select one base model
##  (note that model should be defined first in 'root/lib/ssd/models')
model_name = 'VGG16'
# model_name = 'ResNet50'

## common arguments
image_size = (300,300)
num_object_classes = 20
background_label_id = 0
batch_size = 32
base_lr = 0.001

steps_per_epoch = 10000
epochs = 12
lr_decay_epochs = [8,10]

loc_variance = [0.1, 0.1, 0.2, 0.2]
random_seed = 2
#*************************************************************************************#

## configurations for model
cfg['model'] = {
    'model_name': model_name,
    'image_size': image_size,
    'num_object_classes': num_object_classes,
    'background_label_id': background_label_id,
}

## configurations for optimization
cfg['optimize'] = {
    'base_lr': base_lr,
    'gamma': 0.1,
    'lr_decay_epochs': lr_decay_epochs,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'steps_per_epoch': steps_per_epoch,
    'epochs': epochs,
    'random_seed': random_seed,
}

## configurations for inference
cfg['inference'] = {
    'nms_thres': 0.45,
    'nms_top_k': 400,
    'confidence_thres': 0.01,
    'keep_top_k': 200,
    'background_label_id': background_label_id,
    'loc_variance': loc_variance,
}

## configurations for batch generator
cfg['train_generator'] = {
    'source': os.path.join(DATA_PATH, cfg['dataset']['name'], cfg['dataset']['source_train']),
    'batch_size': batch_size,
    'shuffle': True,
    'transform': {
        'distort':{
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
        },
        'expand':{
            'expand_prob': 0.5,
            'max_expand_ratio': 4.0,
        },
        'sample':{
            'cases': [
                {
                    'max_sample': 1,
                    'max_trials': 1,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'min_jaccard_overlap': 0.1,
                    'max_sample': 1,
                    'max_trials': 50,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'min_jaccard_overlap': 0.3,
                    'max_sample': 1,
                    'max_trials': 50,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'min_jaccard_overlap': 0.5,
                    'max_sample': 1,
                    'max_trials': 50,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'min_jaccard_overlap': 0.7,
                    'max_sample': 1,
                    'max_trials': 50,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'min_jaccard_overlap': 0.9,
                    'max_sample': 1,
                    'max_trials': 50,
                },
                {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    'max_jaccard_overlap': 1.0,
                    'max_sample': 1,
                    'max_trials': 50,
                },
            ]
        },
        'resize':{
            'resize_prob': 1.0,
            'height': image_size[1],
            'width': image_size[0],
            'interpolation': ['LINEAR', 'AREA', 'NEAREST', 'CUBIC', 'LANCZOS4']
        },
        'flip':{
            'flip_prob': 0.5,
        },
    },
    'target': {
        'num_object_classes': num_object_classes,
        'background_label_id': background_label_id,
        'pos_overlap_threshold': 0.5,
        'neg_overlap_threshold': 0.5,
        'loc_variance': loc_variance
    },
}

do_validation = cfg['dataset'].has_key('source_valid')
if do_validation:
    cfg['valid_generator'] = {
        'source': os.path.join(DATA_PATH, cfg['dataset']['name'], cfg['dataset']['source_valid']),
        'batch_size': batch_size,
        'shuffle': False,
        'transform': {
            'resize':{
                'resize_prob': 1.0,
                'height': image_size[1],
                'width': image_size[0],
                'interpolation': ['LINEAR']
            },
        },
    }

