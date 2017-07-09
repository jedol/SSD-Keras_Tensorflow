import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Concatenate
from keras.regularizers import l2

from ssd.layers import Normalize2D, multibox_loss
from ssd.prior_box import get_num_priors_list, make_prior_box_params, gen_prior_boxes_grid

from tfd.parallel import split_batch, concat_batch, sum_gradients


class VGG16_VOC(object):
    def __init__(self, image_size, num_object_classes=20, background_label_id=0, weight_decay=0.0005):
        self.image_size = image_size
        self.num_object_classes = num_object_classes
        self.background_label_id = background_label_id
        self.per_channel_mean = [103.939, 116.779, 123.68] # BGR order
        self.weight_decay = weight_decay

        ## parameters for generating priors.
        self.feature_map_name_list = ['conv4_3_norm', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
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
            'per_channel_mean': self.per_channel_mean, # BGR order
            'num_object_classes': self.num_object_classes,
            'background_label_id': self.background_label_id,
            'mbox_loss_params': self.mbox_loss_params,
        }

        ## construct model
        self._construct_model()

    def get_deploy_tensors(self, num_gpus=1):
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
        regularization_loss_list = list()
        for l in self.model.layers:
            if hasattr(l, 'losses'):
                regularization_loss_list.extend(l.losses)
        regularization_loss = tf.add_n(regularization_loss_list)

        images_T = tf.placeholder(tf.float32, (None, self.image_size[1], self.image_size[0], 3))
        mbox_conf_target_T = tf.placeholder(tf.float32, (None, len(self.prior_boxes)))
        mbox_loc_target_T = tf.placeholder(tf.float32, (None, len(self.prior_boxes), 4))

        inputs_list = split_batch([images_T, mbox_conf_target_T, mbox_loc_target_T], num_gpus)

        mbox_loss_list = list()
        num_pos_list = list()
        for i,inputs in enumerate(inputs_list):
            with tf.device('/gpu:{}'.format(i)):
                mbox_logit, mbox_loc = self.model(inputs[0])
                mbox_loss, num_pos = multibox_loss([mbox_logit, mbox_loc, inputs[1], inputs[2]],
                                                   **self.mbox_loss_params)
                mbox_loss_list.append(mbox_loss)
                num_pos_list.append(num_pos)

        with tf.device('/cpu:0'):
            total_num_pos = tf.add_n(num_pos_list)

        loss_list = list()
        grads_list = list()
        for i,mbox_loss in enumerate(mbox_loss_list):
            with tf.device('/gpu:{}'.format(i)):
                loss = mbox_loss/total_num_pos+regularization_loss/num_gpus
                grads = optimizer.compute_gradients(loss)

                loss_list.append(loss)
                grads_list.append(grads)

        with tf.device('/cpu:0'):
            loss_T = tf.add_n(loss_list)
            optimize_op_T = optimizer.apply_gradients(sum_gradients(grads_list))

        return images_T, mbox_conf_target_T, mbox_loc_target_T, loss_T, optimize_op_T

    def _construct_model(self):
        same_args = {
            'padding': 'same',
            'activation': 'relu',
            'kernel_regularizer': l2(self.weight_decay),
        }
        valid_args = {
            'padding': 'valid',
            'activation': 'relu',
            'kernel_regularizer': l2(self.weight_decay),
        }
        score_args = {
            'padding': 'same',
            'kernel_regularizer': l2(self.weight_decay),
        }

        ## specify layers
        T = dict() # tensor storage
        T['data'] = Input(shape=(self.image_size[1],self.image_size[0],3), name='data')

        ## Block 1
        T['conv1_1'] = Conv2D(64, 3, name='conv1_1', **same_args)(T['data'])
        T['conv1_2'] = Conv2D(64, 3, name='conv1_2', **same_args)(T['conv1_1'])
        T['pool1'] = MaxPooling2D(2, 2, padding='same', name='pool1')(T['conv1_2']) # 150 x 150

        ## Block 2
        T['conv2_1'] = Conv2D(128, 3, name='conv2_1', **same_args)(T['pool1'])
        T['conv2_2'] = Conv2D(128, 3, name='conv2_2', **same_args)(T['conv2_1'])
        T['pool2'] = MaxPooling2D(2, 2, padding='same', name='pool2')(T['conv2_2']) # 75 x 75

        ## Block 3
        T['conv3_1'] = Conv2D(256, 3, name='conv3_1', **same_args)(T['pool2'])
        T['conv3_2'] = Conv2D(256, 3, name='conv3_2', **same_args)(T['conv3_1'])
        T['conv3_3'] = Conv2D(256, 3, name='conv3_3', **same_args)(T['conv3_2'])
        T['pool3'] = MaxPooling2D(2, 2, padding='same', name='pool3')(T['conv3_3']) # 38 x 38

        ## Block 4
        T['conv4_1'] = Conv2D(512, 3, name='conv4_1', **same_args)(T['pool3'])
        T['conv4_2'] = Conv2D(512, 3, name='conv4_2', **same_args)(T['conv4_1'])
        T['conv4_3'] = Conv2D(512, 3, name='conv4_3', **same_args)(T['conv4_2'])
        T['conv4_3_norm'] = Normalize2D(20, name='conv4_3_norm')(T['conv4_3'])
        T['pool4'] = MaxPooling2D(2, 2, padding='same', name='pool4')(T['conv4_3']) # 19 x 19

        ## Block 5
        T['conv5_1'] = Conv2D(512, 3, name='conv5_1', **same_args)(T['pool4'])
        T['conv5_2'] = Conv2D(512, 3, name='conv5_2', **same_args)(T['conv5_1'])
        T['conv5_3'] = Conv2D(512, 3, name='conv5_3', **same_args)(T['conv5_2'])
        T['pool5'] = MaxPooling2D(3, 1, padding='same', name='pool5')(T['conv5_3'])
        T['fc6'] = Conv2D(1024, 3, dilation_rate=6, name='fc6', **same_args)(T['pool5'])
        T['fc7'] = Conv2D(1024, 1, name='fc7', **same_args)(T['fc6'])

        ## Block 6
        T['conv6_1'] = Conv2D(256, 1, name='conv6_1', **same_args)(T['fc7'])
        T['conv6_2'] = Conv2D(512, 3, strides=2, name='conv6_2', **same_args)(T['conv6_1']) # 10 x 10

        ## Block 7
        T['conv7_1'] = Conv2D(128, 1, name='conv7_1', **same_args)(T['conv6_2'])
        T['conv7_2'] = Conv2D(256, 3, strides=2, name='conv7_2', **same_args)(T['conv7_1']) # 5 x 5

        ## Block 8
        T['conv8_1'] = Conv2D(128, 1, name='conv8_1', **same_args)(T['conv7_2'])
        T['conv8_2'] = Conv2D(256, 3, name='conv8_2', **valid_args)(T['conv8_1']) # 3 x 3

        ## Block 9
        T['conv9_1'] = Conv2D(128, 1, name='conv9_1', **same_args)(T['conv8_2'])
        T['conv9_2'] = Conv2D(256, 3, name='conv9_2', **valid_args)(T['conv9_1']) # 1 x 1

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

