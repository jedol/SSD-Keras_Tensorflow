import os
import numpy as np
import tensorflow as tf
import pdb

os.chdir('../tools')

f = np.load('a.npz')
boxes = f.items()[1][1]
confs = f.items()[0][1]

# g = tf.Graph()
# with g.as_default():
#     boxes_T = tf.placeholder(tf.float32, (None, 4))
#     confs_T = tf.placeholder(tf.float32, (None,))
#     inds_T = tf.image.non_max_suppression(boxes=boxes_T, scores=confs_T,
#                                           max_output_size=10,
#                                           iou_threshold=0.3)
#
# sess = tf.Session(graph=g)
# print sess.run(inds_T, {boxes_T:boxes, confs_T:confs})

from utils.box import nms, nms_tf
inds = nms_tf(boxes, confs, 10, 0.3)
print inds
inds2 = nms(boxes, confs, 10, 0.3)
print inds2