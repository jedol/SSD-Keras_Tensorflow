import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras import backend as K

from ssd.layers import Normalize2D, multibox_loss
from ssd.prior_box import get_num_priors_list, make_prior_box_params, gen_prior_boxes_grid

from tfd.parallel import split_batch, concat_batch, sum_gradients


class ResNet50_VOC(object):
    def __init__(self, image_size, num_object_classes=20, background_label_id=0, weight_decay=0.0005):
        self.image_size = image_size
        self.num_object_classes = num_object_classes
        self.background_label_id = background_label_id
        self.weight_decay = weight_decay

        ## parameters for generating priors.
        self.feature_map_name_list = ['res3d', 'res5c', 'res6a', 'res7a', 'res8a', 'pool8']
        self.step_size_list = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios_list = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.flip = True
        self.clip = False
        self.min_ratio = 20  # in percent %
        self.max_ratio = 90  # in percent %
        self.offset = 0.5

        ## parameters for mbox loss
        self.mbox_loss_params = {
            'background_label_id': self.background_label_id,
            'loc_weight': 1.0,
            'neg_pos_ratio': 3.0,
            'sigma': 1.0,
        }

        ## summary of model
        self.summary = {
            'image_size': self.image_size,
            'num_object_classes': self.num_object_classes,
            'background_label_id': self.background_label_id,
            'mbox_loss_params': self.mbox_loss_params,
        }

        ## construct model
        self._construct_model()

    def get_deploy_tensors(self, num_gpus=1):
        K.set_learning_phase(False)

        images_T = tf.placeholder(tf.float32, (None, self.image_size[1], self.image_size[0], 3))
        inputs_list = split_batch(images_T, num_gpus)

        output_list = list()
        for i,inputs in enumerate(inputs_list):
            with tf.device('/gpu:{}'.format(i)):
                mbox_logit, mbox_loc = self.model(inputs)
                output_list.append([
                    tf.nn.softmax(mbox_logit, dim=2),
                    mbox_loc
                ])

        mbox_conf_T, mbox_loc_T = concat_batch(output_list)

        return images_T, mbox_conf_T, mbox_loc_T

    def get_train_tensors(self, optimizer, num_gpus=1):
        K.set_learning_phase(True)

        regularization_loss_list = list()
        for l in self.model.layers:
            if hasattr(l, 'losses'):
                regularization_loss_list.extend(l.losses)
        regularization_loss = tf.add_n(regularization_loss_list)

        images_T = tf.placeholder(tf.float32, (None, self.image_size[1], self.image_size[0], 3))
        mbox_conf_target_T = tf.placeholder(tf.float32, (None, len(self.prior_boxes)))
        mbox_loc_target_T = tf.placeholder(tf.float32, (None, len(self.prior_boxes), 4))

        inputs_list = split_batch([images_T, mbox_conf_target_T, mbox_loc_target_T], num_gpus)

        loss_list = list()
        grads_list = list()
        operation_list = list()
        for i,inputs in enumerate(inputs_list):
            with tf.device('/gpu:{}'.format(i)):
                mbox_logit, mbox_loc = self.model(inputs[0])
                mbox_loss = multibox_loss([mbox_logit, mbox_loc, inputs[1], inputs[2]],
                                          **self.mbox_loss_params)
                loss = (tf.reduce_mean(mbox_loss)+regularization_loss)/num_gpus
                grads = optimizer.compute_gradients(loss)
                update_ops = self.model.get_updates_for(inputs[0])

                loss_list.append(loss)
                grads_list.append(grads)
                operation_list.extend(update_ops)

        with tf.device('/cpu:0'):
            operation_list.append(optimizer.apply_gradients(sum_gradients(grads_list)))
            loss_T = tf.add_n(loss_list)
            optimize_O = tf.group(*operation_list)

        return images_T, mbox_conf_target_T, mbox_loc_target_T, loss_T, optimize_O

    def _construct_model(self):
        kr = l2(self.weight_decay)
        score_args = {
            'padding': 'same',
            'kernel_regularizer': kr,
        }

        ## specify layers
        T = dict() # tensor storage
        T['data'] = Input(shape=(self.image_size[1],self.image_size[0],3), name='data')

        ## Block 1
        x = ZeroPadding2D((3, 3))(T['data'])
        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x) # 150 x 150
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        T['conv1'] = Activation('relu')(x)
        T['pool1'] = MaxPooling2D(3, strides=(2, 2), padding='same')(T['conv1']) # 75 x 75

        ## Block 2
        T['res2a'] = conv_block(T['pool1'], 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), kernel_regularizer=kr)
        T['res2b'] = identity_block(T['res2a'], 3, [64, 64, 256], stage=2, block='b', kernel_regularizer=kr)
        T['res2c'] = identity_block(T['res2b'], 3, [64, 64, 256], stage=2, block='c', kernel_regularizer=kr)

        ## Block 3
        T['res3a'] = conv_block(T['res2c'], 3, [128, 128, 512], stage=3, block='a', kernel_regularizer=kr) # 38 x 38
        T['res3b'] = identity_block(T['res3a'], 3, [128, 128, 512], stage=3, block='b', kernel_regularizer=kr)
        T['res3c'] = identity_block(T['res3b'], 3, [128, 128, 512], stage=3, block='c', kernel_regularizer=kr)
        T['res3d'] = identity_block(T['res3c'], 3, [128, 128, 512], stage=3, block='d', kernel_regularizer=kr)

        ## Block 4
        T['res4a'] = conv_block(T['res3d'], 3, [256, 256, 1024], stage=4, block='a', kernel_regularizer=kr) # 19 x 19
        T['res4b'] = identity_block(T['res4a'], 3, [256, 256, 1024], stage=4, block='b', kernel_regularizer=kr)
        T['res4c'] = identity_block(T['res4b'], 3, [256, 256, 1024], stage=4, block='c', kernel_regularizer=kr)
        T['res4d'] = identity_block(T['res4c'], 3, [256, 256, 1024], stage=4, block='d', kernel_regularizer=kr)
        T['res4e'] = identity_block(T['res4d'], 3, [256, 256, 1024], stage=4, block='e', kernel_regularizer=kr)
        T['res4f'] = identity_block(T['res4e'], 3, [256, 256, 1024], stage=4, block='f', kernel_regularizer=kr)

        T['res5a'] = conv_block(T['res4f'], 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), kernel_regularizer=kr)
        T['res5b'] = identity_block(T['res5a'], 3, [512, 512, 2048], stage=5, block='b', dilation=(2, 2))
        T['res5c'] = identity_block(T['res5b'], 3, [512, 512, 2048], stage=5, block='c', dilation=(2, 2))

        ## Block 5
        T['res6a'] = conv_block(T['res5c'], 3, [128, 128, 512], stage=6, block='a', kernel_regularizer=kr) # 10 x 10

        ## Block 6
        T['res7a'] = conv_block(T['res6a'], 3, [128, 128, 512], stage=7, block='a', kernel_regularizer=kr) # 5 x 5

        ## Block 7
        T['res8a'] = conv_block(T['res7a'], 3, [128, 128, 512], stage=8, block='a', kernel_regularizer=kr) # 3 x 3

        ## Block 8
        T['pool8'] = AveragePooling2D(3)(T['res8a']) # 1 x 1

        ## spatial size of end-point feature map
        feature_map_size_list = [T[fmap_name].get_shape().as_list()[1:3]
                                 for i, fmap_name in enumerate(self.feature_map_name_list)]

        ## attach multibox layers
        prior_boxes, num_unit_priors_list = self.generate_prior_boxes(feature_map_size_list)
        for i, fmap_name in enumerate(self.feature_map_name_list):
            num_unit_priors = num_unit_priors_list[i]
            mbox_logit_channel = num_unit_priors*(self.num_object_classes+1)
            mbox_loc_channel = num_unit_priors*4

            T[fmap_name+'_mbox_logit'] = Conv2D(mbox_logit_channel, 3,
                                                name=fmap_name+'_mbox_logit', **score_args)(T[fmap_name])
            T[fmap_name+'_mbox_logit_reshape'] = Reshape((-1, self.num_object_classes+1),
                                                         name=fmap_name+'_mbox_logit_reshape')(T[fmap_name+'_mbox_logit'])

            T[fmap_name+'_mbox_loc'] = Conv2D(mbox_loc_channel, 3,
                                                name=fmap_name+'_mbox_loc', **score_args)(T[fmap_name])
            T[fmap_name+'_mbox_loc_reshape'] = Reshape((-1, 4),
                                                         name=fmap_name+'_mbox_loc_reshape')(T[fmap_name+'_mbox_loc'])

        ## merge all multi box scores
        T['mbox_logit'] = Concatenate(axis=1, name='mbox_logit')([T[fmap_name+'_mbox_logit_reshape'] for fmap_name in self.feature_map_name_list])
        T['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([T[fmap_name+'_mbox_loc_reshape'] for fmap_name in self.feature_map_name_list])

        self.model = Model(inputs=T['data'], outputs=[T['mbox_logit'], T['mbox_loc']])
        self.T = T
        self.prior_boxes = prior_boxes

    def generate_prior_boxes(self, feature_map_size_list):
        ## compute list of min sizes and max sizes
        step = int(np.floor((self.max_ratio - self.min_ratio) / (len(self.feature_map_name_list) - 2)))
        min_dim = min(self.image_size)
        min_sizes_list = [[min_dim * 10 / 100.]]
        max_sizes_list = [[min_dim * 20 / 100.]]
        for ratio in xrange(self.min_ratio, self.max_ratio + 1, step):
            min_sizes_list.append([min_dim * ratio / 100.])
            max_sizes_list.append([min_dim * (ratio + step) / 100.])

        ## number of unit prior boxes in each feature map
        num_unit_priors_list = get_num_priors_list(min_sizes_list=min_sizes_list,
                                                   max_sizes_list=max_sizes_list,
                                                   aspect_ratios_list=self.aspect_ratios_list,
                                                   flip=self.flip)

        ## prior_boxes_param_list
        prior_boxes_param_list = make_prior_box_params(image_size=self.image_size,
                                                       feature_map_size_list=feature_map_size_list,
                                                       min_sizes_list=min_sizes_list,
                                                       max_sizes_list=max_sizes_list,
                                                       aspect_ratios_list=self.aspect_ratios_list,
                                                       step_size_list=self.step_size_list,
                                                       offset=self.offset,
                                                       flip=self.flip,
                                                       clip=self.clip)

        ## generate prior boxes
        prior_boxes = np.concatenate([gen_prior_boxes_grid(**prior_boxes_param)
                                      for prior_boxes_param in prior_boxes_param_list], axis=0)

        ## add to summary
        self.summary.update({
            'prior_boxes_param_list': prior_boxes_param_list,
        })

        return prior_boxes, num_unit_priors_list

    def get_summary(self):
        return self.summary

    def get_model(self):
        return self.model

    def get_prior_boxes(self):
        return self.prior_boxes

    def get_endpoints(self):
        return self.T


def ConvBnRelu(inputs, c, k, s, name, kernel_regularizer=None):
    x = Conv2D(c, k, strides=s, padding='same', use_bias=False, kernel_regularizer=kernel_regularizer,
               name=name+'_conv')(inputs)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=(1, 1), kernel_regularizer=None):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation, use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=(1, 1), kernel_regularizer=None, padding='same'):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding=padding, dilation_rate=dilation, use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, kernel_regularizer=kernel_regularizer,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x
