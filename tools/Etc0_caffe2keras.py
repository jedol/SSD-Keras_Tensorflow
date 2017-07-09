from _init_path import *
import os
import numpy as np
import pdb

import caffe.proto.caffe_pb2 as cpb
from keras.models import Model

from ssd.models.VGG import VGG16_VOC


def load_caffe_weights(layer_names, model_path):
    net_param = cpb.NetParameter()
    net_param.MergeFromString(open(model_path, 'rb').read())
    layers = net_param.layer
    find_weights = dict()
    for layer in layers:
        if str(layer.name) in layer_names:
            weights = list()
            for blob in layer.blobs:
                weights.append(np.array(blob.data).reshape(blob.shape.dim))
            find_weights[str(layer.name)] = weights
    return find_weights


model_path = os.path.join(MODEL_PATH, 'VGG16_fc_reduced')
caffe_model_path = os.path.join(model_path, 'final.caffemodel')
keras_weights_save_path = os.path.join(model_path, 'weights_from_caffe.h5')

caffe_to_keras = {
    'conv1_1': 'conv1_1',
    'conv1_2': 'conv1_2',
    'conv2_1': 'conv2_1',
    'conv2_2': 'conv2_2',
    'conv3_1': 'conv3_1',
    'conv3_2': 'conv3_2',
    'conv3_3': 'conv3_3',
    'conv4_1': 'conv4_1',
    'conv4_2': 'conv4_2',
    'conv4_3': 'conv4_3',
    'conv5_1': 'conv5_1',
    'conv5_2': 'conv5_2',
    'conv5_3': 'conv5_3',
    'fc6': 'fc6',
    'fc7': 'fc7',
    'conv6_1': 'conv6_1',
    'conv6_2': 'conv6_2',
    'conv7_1': 'conv7_1',
    'conv7_2': 'conv7_2',
    'conv8_1': 'conv8_1',
    'conv8_2': 'conv8_2',
    'conv9_1': 'conv9_1',
    'conv4_3_norm': 'conv4_3_norm',
    'conv9_2': 'conv9_2',
    'conv4_3_norm_mbox_conf': 'conv4_3_norm_mbox_logit',
    'fc7_mbox_conf': 'fc7_mbox_logit',
    'conv6_2_mbox_conf': 'conv6_2_mbox_logit',
    'conv7_2_mbox_conf': 'conv7_2_mbox_logit',
    'conv8_2_mbox_conf': 'conv8_2_mbox_logit',
    'conv9_2_mbox_conf': 'conv9_2_mbox_logit',
    'conv4_3_norm_mbox_loc': 'conv4_3_norm_mbox_loc',
    'fc7_mbox_loc': 'fc7_mbox_loc',
    'conv6_2_mbox_loc': 'conv6_2_mbox_loc',
    'conv7_2_mbox_loc': 'conv7_2_mbox_loc',
    'conv8_2_mbox_loc': 'conv8_2_mbox_loc',
    'conv9_2_mbox_loc': 'conv9_2_mbox_loc',
}

caffe_weights_list = load_caffe_weights(caffe_to_keras.keys(), caffe_model_path)

## construct keras model
vgg16_voc = VGG16_VOC((300,300))
keras_model = vgg16_voc.get_model()

## dim order of weight in convolution layer
##  caffe: (out_c, in_c, k_h, k_w)
##  keras: (k_h, k_w, in_c, out_c)

for caffe_name, keras_name in caffe_to_keras.iteritems():
    caffe_weights = caffe_weights_list[caffe_name]
    keras_layer = keras_model.get_layer(keras_name)
    keras_weights = keras_layer.get_weights()
    assert len(caffe_weights) == len(keras_weights)
    weights = list()
    for caffe_weight, keras_weight in zip(caffe_weights, keras_weights):
        assert caffe_weight.size == keras_weight.size
        if caffe_weight.ndim == 4:
            ## case weights in convolution
            # weight = np.rot90(caffe_weight, 2, (2,3))
            weight = caffe_weight.transpose(2,3,1,0)
            weights.append(weight)
        else:
            ## case bias in convolution
            weights.append(caffe_weight)
    keras_layer.set_weights(weights)

keras_model.save_weights(keras_weights_save_path)