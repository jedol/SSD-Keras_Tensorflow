from _init_path import *
import os
import numpy as np
import cv2
import tensorflow as tf
import keras.backend as K

from sample.preprocessing import transform_for_test
from ssd.result import detect_images
from utils.draw import draw_boxes


#*************************************************************************************#
## path to test images
image_path = os.path.join(DATA_PATH, 'images')

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
confidence_thres = 0.6
keep_top_k = 200
loc_variance = [0.1, 0.1, 0.2, 0.2]

## VOC class list (optional)
ind_to_class = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#*************************************************************************************#


cfg = dict()

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
cfg['transform'] = {
    'resize':{
        'resize_prob': 1.0,
        'height': image_size[1],
        'width': image_size[0],
        'interpolation': ['LINEAR']
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
    cfg['transform']['per_channel_mean'] = per_channel_mean

## read images
image_names = os.listdir(image_path)
num_images = len(image_names)
images = [cv2.imread(os.path.join(image_path, image_name)) for image_name in image_names]
image_sizes = [image.shape[:2][::-1] for image in images]

## transform image
transformed_images = np.array([transform_for_test(image, cfg['transform']) for image in images])

print 'Inference and Detection...'
results = detect_images(net, sess, transformed_images, cfg['inference'],
                        image_sizes=image_sizes, num_gpus=1, batch_size=32)
print '  number of detected objects: {}'.format(np.sum([len(r['objects']) for r in results]))

## show results
cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
i = 0
while True:
    image = images[i].copy()
    result = results[i]
    bboxes = np.array([obj['bbox'] for obj in result['objects']])
    image = draw_boxes(image, bboxes)

    print image_names[i]
    for obj in result['objects']:
        if 'ind_to_class' in locals():
            c = ind_to_class[obj['label']]
        else:
            c = str(obj['label'])
        bbox = np.array(obj['bbox'], np.int)
        print '  class: {}  bbox: {}  confidence: {:.4}'.format(c, str(bbox.tolist()), obj['conf'])

    cv2.imshow('result', image)
    key = cv2.waitKey(0) & 0xFF

    if key == 81:   ## left
        i = 0 if i < 0 else i-1
    if key == 83:   ## right
        i = num_images-1 if i >= num_images-1 else i+1
    if key == ord('q'):
        break

