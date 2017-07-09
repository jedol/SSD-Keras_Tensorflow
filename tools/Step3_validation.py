from _init_path import *
import os
import numpy as np
import timeit as ti
import tensorflow as tf
import keras.backend as K

from sample.batch_manager import ValidBatchManager
from ssd.result import detect_images
from dataset.PASCAL_VOC import eval_detection
from utils.mics import load_json


#*************************************************************************************#
## number of gpus for validation
num_gpus = 2

## dataset name to test
dataset_name = 'VOC07'

## type of trained model
model_name = 'VGG16'
# model_name = 'ResNet50'

## path to trained weights
trained_weights_path = os.path.join(EXP_PATH, 'SSD300_VGG16_VOC07', 'final.h5')
# trained_weights_path = os.path.join(EXP_PATH, 'SSD300_VGG16_VOC07+12', 'final.h5')
# trained_weights_path = os.path.join(EXP_PATH, 'SSD300_ResNet50_VOC07+12', 'final.h5')

## model arguments
image_size = (300,300)
num_object_classes = 20
background_label_id = 0

## inference arguments
nms_thres = 0.45
nms_top_k = 400
confidence_thres = 0.01
keep_top_k = 200
loc_variance = [0.1, 0.1, 0.2, 0.2]
#*************************************************************************************#


## load dataset configuration
dataset_cfg_file = os.path.join(DATA_PATH, dataset_name, 'cfg.json')
if os.path.exists(dataset_cfg_file):
    cfg = load_json(dataset_cfg_file)
else:
    raise ValueError('Dataset must be dumped to LMDB first.')

assert cfg['dataset'].has_key('source_valid'), 'DB has no validation dataset.'

## configurations for model
cfg['model'] = {
    'image_size': image_size,
    'num_object_classes': num_object_classes,
    'background_label_id': background_label_id,
}

## configurations for inference
cfg['inference'] = {
    'nms_thres': nms_thres,
    'nms_top_k': nms_top_k,
    'confidence_thres': confidence_thres,
    'keep_top_k': keep_top_k,
    'background_label_id': background_label_id,
    'loc_variance': loc_variance,
}

## configurations for transform
cfg['valid_generator'] = {
    'source': os.path.join(DATA_PATH, cfg['dataset']['name'], cfg['dataset']['source_valid']),
    'batch_size': 1,
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

## create session
cp = tf.ConfigProto()
cp.allow_soft_placement = True
cp.gpu_options.allow_growth = True
sess = tf.Session(config=cp)
K.set_session(sess)

## construct model
print 'Constructing model: {}'.format(model_name)
if model_name == 'VGG16':
    from ssd.models.VGG import VGG16_VOC as net_creator
elif model_name == 'ResNet50':
    from ssd.models.ResNet import ResNet50_VOC as net_creator
else:
    raise LookupError

## create network
net = net_creator(**cfg['model'])

## initialize variables
init = tf.global_variables_initializer()
sess.run(init)

## get model
model = net.get_model()

## load trained model
print 'Loading weights from trained weights...'
model.load_weights(trained_weights_path, by_name=True)

## mean substraction
if net.get_summary().has_key('per_channel_mean'):
    per_channel_mean = net.get_summary()['per_channel_mean']
    cfg['valid_generator']['transform']['per_channel_mean'] = per_channel_mean

## get validation samples
print 'Collecting and transforming validation samples...'
vm = ValidBatchManager(use_prefetch=False, **cfg['valid_generator'])
dataset = vm.get_all()
images = np.array([data['image'] for data in dataset])
image_sizes = np.array([[data['width'], data['height']] for data in dataset])

## inference and convert scores(mbox conf, mbox loc) to detection
print 'Inference and Detection...'
results = detect_images(net, sess, images, cfg['inference'],
                        image_sizes=image_sizes, num_gpus=num_gpus, batch_size=32)
print '  number of detected objects: {}'.format(np.sum([len(r['objects']) for r in results]))

## evaluate detections
print 'Evaluation...'
start = ti.default_timer()
evals = eval_detection(dataset, results, cfg['model']['num_object_classes'],
                       overlap_thres=0.5, use_difficult=False,
                       use_07_metric=True, ignore_id=True)
end = ti.default_timer()
print '  {:.4f} sec elapsed'.format(end-start)
print '  mAP: {}'.format(np.mean([ev['AP'] for ev in evals]))


