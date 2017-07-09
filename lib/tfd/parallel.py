import numpy as np
import tensorflow as tf


def split_batch(inputs=[], num=1):
    ## Inputs
    ##  inputs: list of tensor = [tensor0, tensor1, ...]
    ## Outputs
    ##  outputs: list of split tensors = [[tensor0_0, tensor1_0, ...], [tensor0_1, tensor1_1, ...], ...]

    assert num > 0
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        assert tf.is_numeric_tensor(inputs)
        inputs = [inputs]
    if num == 1:
        if len(inputs) == 1:
            return inputs
        else:
            return [inputs]
    else:
        with tf.device('/cpu:0'):
            if len(inputs) == 1:
                return tf.split(inputs[0], num, axis=0)
            else:
                return np.array([tf.split(i, num, axis=0) for i in inputs], np.object).transpose().tolist()


def concat_batch(inputs=[]):
    ## Inputs
    ##  inputs: list of split tensors = [[tensor0_0, tensor1_0, ...], [tensor0_1, tensor1_1, ...], ...]
    ## Ouputs
    ##  outputs: list of concatenated tensor = [tensor0, tensor1, ...]

    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        return inputs
    inputs = np.array(inputs, np.object)
    with tf.device('/cpu:0'):
        if inputs.ndim == 1:
            return tf.concat(inputs.tolist(), axis=0)
        else:
            return [tf.concat(inputs[:,i].tolist(), axis=0) for i in xrange(len(inputs[0]))]


def sum_gradients(grads_list):
    """
    Inputs
      grads_list: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    outputs
      sum_grads: List of pairs of (gradient, variable) where the gradient has been summed
        across all towers.
    """
    sum_grads = list()
    for grad_and_vars in zip(*grads_list):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [tf.expand_dims(g_T, 0) for g_T,_ in grad_and_vars if g_T is not None]
        if len(grads):
            grad = tf.reduce_sum(tf.concat(grads, 0), 0)
            sum_grads.append((grad, grad_and_vars[0][1]))

    return sum_grads

