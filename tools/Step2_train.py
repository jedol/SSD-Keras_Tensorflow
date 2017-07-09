from _init_path import *
from config import cfg
import os
import numpy as np
import shutil
import timeit as ti

## set random seed
np.random.seed(cfg['optimize']['random_seed'])

import tensorflow as tf
import keras.backend as K

from sample.batch_manager import TrainBatchManager, ValidBatchManager
from ssd.result import detect_images
from dataset.PASCAL_VOC import eval_detection
from utils.mics import save_json
from tfd.loop import fit_generator
from tfd.scheduler import multi_step_learning_rate_decay


#*************************************************************************************#
## all arguments are loaded from config.py.
## edit config.py first!

## number of gpus for training and validation
num_gpus = 2

## folder name for saving weights, configurations, etc...
exp_name = 'SSD{}_{}_{}'.format(int(np.mean(cfg['model']['image_size'])),
                                cfg['model']['model_name'],
                                cfg['dataset']['name'])
#*************************************************************************************#

## create session
cp = tf.ConfigProto()
cp.allow_soft_placement = True
cp.gpu_options.allow_growth = True
sess = tf.Session(config=cp)
K.set_session(sess)

## construct model
print 'Constructing model: {}'.format(cfg['model']['model_name'])
if cfg['model']['model_name'] == 'VGG16':
    from ssd.models.VGG import VGG16_VOC as net_creator
    model_def_path = os.path.join(LIB_PATH, 'ssd', 'models', 'VGG.py')
    pretrained_weights_path = os.path.join(MODEL_PATH, 'VGG16_fc_reduced', 'weights_from_caffe.h5')
elif cfg['model']['model_name'] == 'ResNet50':
    from ssd.models.ResNet import ResNet50_VOC as net_creator
    model_def_path = os.path.join(LIB_PATH, 'ssd', 'models', 'ResNet.py')
    pretrained_weights_path = os.path.join(MODEL_PATH, 'ResNet50', 'ResNet50_from_keras.h5')
else:
    raise LookupError

## create network
net = net_creator(cfg['model']['image_size'],
                  cfg['model']['num_object_classes'],
                  cfg['model']['background_label_id'],
                  cfg['optimize']['weight_decay'])

## optimizer
lr_T = tf.placeholder(tf.float32, None, name='lr')
optimizer = tf.train.MomentumOptimizer(lr_T, cfg['optimize']['momentum'])

## get train tensors
train_images_T, mbox_conf_target_T, mbox_loc_target_T, loss_T, optimize_O = net.get_train_tensors(optimizer, num_gpus)

## initialize variables
init = tf.global_variables_initializer()
sess.run(init)

## get model
model = net.get_model()

## load pretrained model
if 'pretrained_weights_path' in locals():
    print 'Loading weights from pretrained model...'
    model.load_weights(pretrained_weights_path, by_name=True)

## whether do validation
do_validation = cfg.has_key('valid_generator')

## update model configurations
cfg['model'].update(net.get_summary())

## mean substraction
if cfg['model'].has_key('per_channel_mean'):
    per_channel_mean = cfg['model']['per_channel_mean']
    cfg['train_generator']['transform']['per_channel_mean'] = per_channel_mean
    if do_validation:
        cfg['valid_generator']['transform']['per_channel_mean'] = per_channel_mean

## train sample generator
tbm = TrainBatchManager(prior_boxes=net.get_prior_boxes(), **cfg['train_generator'])
train_generator = tbm.get_generator()

## get valid samples
if do_validation:
    print 'Collecting and transforming validation samples...'
    vm = ValidBatchManager(use_prefetch=False, **cfg['valid_generator'])
    dataset = vm.get_all()
    images = np.array([data['image'] for data in dataset])
    image_sizes = np.array([[data['width'], data['height']] for data in dataset])
    mAP_list = list()

## record all logs(model definition, snapshots, configurations...) on experiment folder
expset_path = os.path.join(EXP_PATH, exp_name)
weights_path = os.path.join(expset_path, 'weights')
if os.path.isdir(expset_path):
    cmd = raw_input('Existing folder will be deleted. Continue? (y/n)\n >> ')
    assert cmd == 'y', 'Job canceled...'
    shutil.rmtree(expset_path)

## make experiments directory
os.makedirs(expset_path)

## make model(snapshot) directory
os.makedirs(weights_path)

## copy model definition file
shutil.copy(model_def_path, os.path.join(expset_path, 'model.py'))

## dump configurations
save_json(os.path.join(expset_path, 'cfg.json'), cfg, indent=4)

print 'Training start...'
for epoch in xrange(cfg['optimize']['epochs']):
    lr = multi_step_learning_rate_decay(epoch,
                                        cfg['optimize']['base_lr'],
                                        cfg['optimize']['gamma'],
                                        cfg['optimize']['lr_decay_epochs'])
    print 'epoch {}/{} learning rate: {:.5f}'.format(epoch, cfg['optimize']['epochs'], lr)
    fit_generator(sess=sess,
                  num_iter=cfg['optimize']['steps_per_epoch'],
                  operations=optimize_O,
                  batch_generator=train_generator,
                  inputs=[train_images_T, mbox_conf_target_T, mbox_loc_target_T],
                  outputs=loss_T,
                  static_inputs={lr_T: lr},
                  print_interval=10)

    if do_validation:
        print 'Inference and detection...'
        results = detect_images(net, sess, images, cfg['inference'],
                                image_sizes=image_sizes, num_gpus=num_gpus, batch_size=64)
        print '  number of detected objects: {}'.format(np.sum([len(r['objects']) for r in results]))

        print 'Evaluation...'
        start = ti.default_timer()
        evals = eval_detection(dataset, results, cfg['model']['num_object_classes'],
                               overlap_thres=0.5, use_difficult=False,
                               use_07_metric=True, ignore_id=True)
        end = ti.default_timer()
        print '  {:.4f} sec elapsed'.format(end-start)
        mAP_list.append(np.mean([ev['AP'] for ev in evals]))
        for i,mAP in enumerate(mAP_list):
            print 'epoch: {:2d} mAP: {}'.format(i, mAP)

    ## save weights
    print 'Saving weights...'
    save_path = os.path.join(weights_path, 'epoch_{}.h5'.format(epoch))
    model.save_weights(save_path)

## copy final model
shutil.copy(save_path, os.path.join(expset_path, 'final.h5'))

