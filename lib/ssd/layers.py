import tensorflow as tf

from keras.engine.topology import Layer
from keras.initializers import Constant


class Normalize2D(Layer):
    def __init__(self, scale=20.0, scale_regularizer=None, **kwargs):
        self.scale_initializer = Constant(scale)
        self.scale_regularizer = scale_regularizer
        super(Normalize2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[3],),
                                     initializer=self.scale_initializer,
                                     name='scale',
                                     regularizer=self.scale_regularizer)
        super(Normalize2D, self).build(input_shape)

    def call(self, inputs):
        output = tf.nn.l2_normalize(inputs, dim=3)
        output *= self.scale
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


def smooth_l1(labels, scores, sigma=1.0):
    diff = scores-labels
    abs_diff = tf.abs(diff)
    return tf.where(tf.less(abs_diff, 1/(sigma**2)), 0.5*(sigma*diff)**2, abs_diff-1/(2*sigma**2))


def multibox_loss(inputs, background_label_id=0, loc_weight=1.0, neg_pos_ratio=3.0, sigma=1.0):
    ## Inputs
    ##  inputs[0]: mbox_conf_scores = (None, N, C)
    ##  inputs[1]: mbox_loc_scores = (None, N, 4)
    ##  inputs[2]: mbox_conf_targets = (None, N)
    ##  inputs[3]: mbox_loc_targets = (None, N, 4)
    ##      N = number of prior boxes
    ##      C = number of classes
    ## Ouputs
    ##  mbox_loss: (None)

    mbox_conf_scores = inputs[0]
    mbox_loc_scores = inputs[1]
    mbox_conf_targets = inputs[2]
    mbox_loc_targets = inputs[3]

    ## we denote batch size as 'B'
    batch_size = tf.shape(mbox_conf_scores)[0]

    ## label '-1' indicates discard samples (samples that neither positive nor negative)
    valid_mask = tf.not_equal(mbox_conf_targets, -1)

    ## background label indicates negative sample
    neg_mask = tf.equal(mbox_conf_targets, background_label_id)

    ## valid and not negative sample is positive
    pos_mask = tf.logical_and(valid_mask, tf.logical_not(neg_mask))

    ## compute confidence(classification) loss
    ##  since '-1' is not allowed as an input of cross-entropy loss function,
    ##  we multiply zero to convert '-1' to '0'. (we aready know which samples are negative)
    mbox_conf_targets = tf.multiply(mbox_conf_targets, tf.to_float(valid_mask))
    mbox_conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.to_int32(mbox_conf_targets),
        logits=mbox_conf_scores
    )

    ## number of positive sample
    num_pos = tf.reduce_sum(tf.to_float(pos_mask), axis=1)

    ## number of negative sample
    num_neg = tf.reduce_sum(tf.to_float(neg_mask), axis=1)
    num_neg = tf.minimum(num_neg, num_pos*neg_pos_ratio)

    ## pick up confidence loss of negative samples
    neg_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(neg_mask))

    ## get top k indices of negative confidence loss
    ##  to vectorization, 'k' should be N+1
    ##  where 'N' is maximum number of negavtive samples among the batch images
    ##  and '+1' is preventing zero sampling
    ##  as a result, shape of sorted_inds is (B,N+1)
    _,sorted_inds = tf.nn.top_k(neg_mbox_conf_loss, k=tf.to_int32(tf.reduce_max(num_neg))+1)

    ##  only 'num_neg' samples should be used to train
    ##  so we suppress negative conf loss less than 'num_neg'th loss
    ##  as a result, 'thres' indicates 'num_neg'th loss and shape is (B,)
    last_neg_inds = tf.concat([tf.expand_dims(tf.range(0, batch_size), 1),
                               tf.expand_dims(tf.to_int32(num_neg), 1)], axis=1) # (B,2)
    thres_ind = tf.gather_nd(sorted_inds, last_neg_inds) # (B,)
    thres_ind = tf.concat([tf.expand_dims(tf.range(0, batch_size), 1),
                           tf.expand_dims(thres_ind, 1)], axis=1) # (B,2)
    thres = tf.gather_nd(neg_mbox_conf_loss, thres_ind) # (B,)

    ## negative conf loss larger than 'thres' is 'hard negative'
    hard_neg_mask = tf.greater(neg_mbox_conf_loss, tf.expand_dims(thres, 1))

    ## compute positive and negative conf loss and sum it all
    pos_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(pos_mask))
    neg_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(hard_neg_mask))
    mbox_conf_loss = tf.reduce_sum(pos_mbox_conf_loss+neg_mbox_conf_loss) # (1,)

    ## compute localization(regression) loss
    mbox_loc_loss = tf.reduce_sum(smooth_l1(mbox_loc_targets, mbox_loc_scores, sigma), axis=2)
    mbox_loc_loss = tf.reduce_sum(tf.multiply(mbox_loc_loss, tf.to_float(pos_mask))) # (1,)

    ## normalize loss by number of positive(matched) sample
    ## when multiple gpus are used on training, normalization should be occurred after the sum of all gpus loss
    # mbox_conf_loss = tf.divide(mbox_conf_loss, tf.reduce_sum(num_pos)+1e-8)
    # mbox_loc_loss = tf.divide(mbox_loc_loss, tf.reduce_sum(num_pos)+1e-8)

    ## total multibox loss
    mbox_loss = mbox_conf_loss+loc_weight*mbox_loc_loss

    return mbox_loss, tf.reduce_sum(num_pos)


# import tensorflow as tf
# import keras.backend as K
# import numpy as np
# import pdb
#
# batch_size = 8
# num_priors = 1000
#
# mbox_conf_scores = np.random.uniform(size=(batch_size,num_priors,21))
# mbox_conf_scores /= mbox_conf_scores.sum(axis=2)[...,None]
# mbox_loc_scores = np.random.uniform(size=(batch_size,num_priors,4))-0.5
#
# mbox_conf_targets = np.random.randint(-1, 1, size=(batch_size,num_priors))
# for i in xrange(1, batch_size):
#     num_pos = np.random.randint(1,num_priors/50)
#     inds = np.random.permutation(num_priors)[:num_pos]
#     for ind in inds:
#         mbox_conf_targets[i,ind] = np.random.randint(1, 21)
# mbox_loc_targets = np.random.uniform(size=(batch_size,num_priors,4))-0.5
# mbox_loc_targets[batch_size/2:] = (np.random.uniform(size=(batch_size/2,num_priors,4))-0.5)*2
#
# background_label_id = 0
# neg_pos_ratio = 3.0
# loc_weight = 1.0
# sigma = 1.0
#
# mbox_conf_scores = tf.constant(mbox_conf_scores, tf.float32)
# mbox_loc_scores = tf.constant(mbox_loc_scores, tf.float32)
# mbox_conf_targets = tf.constant(mbox_conf_targets, tf.float32)
# mbox_loc_targets = tf.constant(mbox_loc_targets, tf.float32)
#
#
# ## we denote batch size as 'B'
# batch_size = tf.shape(mbox_conf_scores)[0]
#
# ## valid sample is both positive and negative
# valid_mask = tf.not_equal(mbox_conf_targets, -1)
#
# ## negative is background
# neg_mask = tf.equal(mbox_conf_targets, background_label_id)
#
# ## valid and not negative sample is positive
# pos_mask = tf.logical_and(valid_mask, tf.logical_not(neg_mask))
#
# ## compute confidence(classification) loss
# ##  '-1' is not allowed as an input of cross-entropy loss function,
# ##  so we multiply zero to convert '-1' to '0'.
# mbox_conf_targets = tf.multiply(mbox_conf_targets, tf.to_float(valid_mask))
# mbox_conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#     labels=tf.to_int32(mbox_conf_targets),
#     logits=mbox_conf_scores
# )
#
# ## number of positive sample
# num_pos = tf.reduce_sum(tf.to_float(pos_mask), axis=1)
#
# ## number of negative sample
# num_neg = tf.reduce_sum(tf.to_float(neg_mask), axis=1)
# num_neg = tf.minimum(num_neg, num_pos*neg_pos_ratio)
#
# ## retrieve conf loss of negative samples
# neg_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(neg_mask))
#
# ## get top k indices of negative conf loss
# ##  for vectorization, 'k' should be N+1
# ##  where 'N' is maximum number of negavtive samples among the batch images
# ##  and '+1' is preventing zero sampling
# ##  as a result, shape of sorted_inds is (B,N+1)
# _,sorted_inds = tf.nn.top_k(neg_mbox_conf_loss, k=tf.to_int32(tf.reduce_max(num_neg))+1)
#
# ##  only 'num_neg' samples should be used for training
# ##  so we suppress negative conf loss less than 'num_neg'th loss
# ##  as a result, 'thres' indicates 'num_neg'th loss and shape is (B,)
# last_neg_inds = tf.concat([tf.expand_dims(tf.range(0, batch_size), 1),
#                            tf.expand_dims(tf.to_int32(num_neg), 1)], axis=1) # (B,2)
# thres_ind = tf.gather_nd(sorted_inds, last_neg_inds) # (B,)
# thres_ind = tf.concat([tf.expand_dims(tf.range(0, batch_size), 1),
#                        tf.expand_dims(thres_ind, 1)], axis=1) # (B,2)
# thres = tf.gather_nd(neg_mbox_conf_loss, thres_ind) # (B,)
#
# ## negative conf loss larger than 'thres' is 'hard negative'
# hard_neg_mask = tf.greater(neg_mbox_conf_loss, tf.expand_dims(thres, 1))
#
# ## compute positive and negative conf loss and sum it all
# pos_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(pos_mask))
# neg_mbox_conf_loss = tf.multiply(mbox_conf_loss, tf.to_float(hard_neg_mask))
# mbox_conf_loss = tf.reduce_sum(pos_mbox_conf_loss+neg_mbox_conf_loss) # (1,)
#
# ## normalize conf loss by number of positive(matched) sample
# mbox_conf_loss = tf.divide(mbox_conf_loss, tf.reduce_sum(num_pos)+1e-8)
#
# ## compute localization(regression) loss
# mbox_loc_loss = tf.reduce_sum(smooth_l1(mbox_loc_targets, mbox_loc_scores, sigma), axis=2)
# mbox_loc_loss = tf.reduce_sum(tf.multiply(mbox_loc_loss, tf.to_float(pos_mask))) # (1,)
#
# ## normalize loc loss by number of positive(matched) sample
# mbox_loc_loss = tf.divide(mbox_loc_loss, tf.reduce_sum(num_pos)+1e-8)
#
# ## total multibox loss
# mbox_loss = loc_weight*mbox_loc_loss+mbox_conf_loss
#
# sess = tf.InteractiveSession()
#
# print sess.run(mbox_loss)