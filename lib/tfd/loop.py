import numpy as np
import tensorflow as tf
import timeit as ti
from batch_manager import ArrayBatchManager


def fit(sess, num_iter, operations=[], batch_size=32, arrays=[], inputs=[], outputs={},
        static_inputs={}, shuffle=True, print_interval=100, capacity=32):
    if not isinstance(arrays, list) and not isinstance(arrays, tuple):
        assert isinstance(arrays, np.ndarray)
        arrays = [arrays]
    assert len(arrays) == len(inputs)
    batch_generator = None
    if len(arrays):
        assert len(np.unique([len(array) for array in arrays])) == 1
        bm = ArrayBatchManager(arrays, batch_size, shuffle, True, capacity)
        batch_generator = bm.get_generator()
    fit_generator(sess, num_iter, operations, batch_generator, inputs, outputs,
                  static_inputs, print_interval)
    if len(arrays):
        bm.close()


def fit_generator(sess, num_iter, operations=[], batch_generator=None, inputs=[], outputs={},
                  static_inputs={}, print_interval=100):
    if not isinstance(operations, list):
        if isinstance(operations, tuple):
            operations = list(operations)
        assert isinstance(operations, tf.Operation)
        operations = [operations]
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        assert isinstance(inputs, tf.Tensor)
        inputs = [inputs]
    if not isinstance(outputs, dict):
        assert isinstance(outputs, tf.Tensor)
        outputs = {'loss':outputs}

    output_names = outputs.keys()
    output_tensors = outputs.values()

    tic = ti.default_timer()
    for step in xrange(1,num_iter+1):
        feed_dict = dict()
        if batch_generator is not None:
            feed_dict.update(dict(zip(inputs,batch_generator.next())))
        feed_dict.update(static_inputs)

        if step % print_interval == 0:
            outputs = sess.run(operations+output_tensors, feed_dict=feed_dict)[len(operations):]

            toc = ti.default_timer()
            eta = (toc-tic)/step*(num_iter-step)
            log = '[Step: {}/{} ETA: {:.0f}s]'.format(step, num_iter, eta)

            for output_name, output in zip(output_names, outputs):
                log += ' {}: {:.4f}'.format(output_name, output)

            print log
        else:
            _ = sess.run(operations, feed_dict=feed_dict)
    toc = ti.default_timer()
    log = '[Step: {}/{} ETA: {:.0f}s]'.format(step, num_iter, toc-tic)
    print log


def predict(sess, arrays=[], inputs=[], outputs=[], batch_size=32,
            num_iter=None, static_inputs={}):
    if not isinstance(arrays, list) and not isinstance(arrays, tuple):
        assert isinstance(arrays, np.ndarray)
        arrays = [arrays]
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        assert isinstance(inputs, tf.Tensor)
        inputs = [inputs]
    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        assert isinstance(outputs, tf.Tensor)
        outputs = [outputs]

    assert len(outputs)
    assert len(arrays) == len(inputs)

    if len(arrays) == 0:
        if num_iter is None:
            raise ValueError
    else:
        assert len(np.unique([len(array) for array in arrays])) == 1

    if len(arrays):
        num_data = len(arrays[0])
        num_iter = num_data/batch_size
        if num_data % batch_size:
            num_iter += 1

    batch_outputs = list()
    for i in xrange(num_iter):
        feed_dict = dict()
        if len(arrays):
            p1 = i*batch_size
            p2 = (i+1)*batch_size
            p2 = num_data if p2 > num_data else p2
            batch = [array[p1:p2] for array in arrays]
            feed_dict.update(dict(zip(inputs,batch)))
        feed_dict.update(static_inputs)
        batch_outputs.append(sess.run(outputs, feed_dict=feed_dict))

    output_arrays = list()
    for i in xrange(len(outputs)):
        output_arrays.append(
            np.concatenate([batch_output[i] for batch_output in batch_outputs], axis=0)
        )

    if len(output_arrays) == 1:
        return output_arrays[0]
    else:
        return output_arrays


def predict_generator(sess, num_iter, batch_generator=None, inputs=[], outputs=[],
                      static_inputs={}):
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        assert isinstance(inputs, tf.Tensor)
        inputs = [inputs]
    if not isinstance(outputs, list) and not isinstance(outputs, tuple):
        assert isinstance(outputs, tf.Tensor)
        outputs = [outputs]

    assert len(outputs)

    batch_outputs = list()
    for i in xrange(num_iter):
        feed_dict = dict()
        if batch_generator is not None:
            feed_dict.update(dict(zip(inputs,batch_generator.next())))
        feed_dict.update(static_inputs)
        batch_outputs.append(sess.run(outputs, feed_dict=feed_dict))

    output_arrays = list()
    for i in xrange(len(outputs)):
        output_arrays.append(
            np.concatenate([batch_output[i] for batch_output in batch_outputs], axis=0)
        )

    if len(output_arrays) == 1:
        return output_arrays[0]
    else:
        return output_arrays

