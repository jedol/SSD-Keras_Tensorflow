# SSD(Single Shot MultiBox Detector) with Keras and Tensorflow
This project is re-implementation version of original [Caffe project](https://github.com/weiliu89/caffe/tree/ssd).

SSD is CNN(Convolutional Neural Network) based object detection framework. It combines predictions from multiple feature maps with different resolutions to handle objects of various sizes. Additionally, it can be run in real time because the network does not resample pixels or features. Please refer to [this research](https://arxiv.org/abs/1512.02325) for more details.

The author's implementation of SSD is based on Caffe and it is hard to study because most sources are written in C++(except some scripts). So to make SSD easier to understand, we write the preparation steps(image transform, sampling, generate prior boxes, compute training targets) in numpy with brief comments. The network model(VGG16, ResNet50) is written in Keras for easy reading and writing of weights(this makes finetuning easy), and the loss function and training loops are written in Tensorflow.

## Prerequisites
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [OpenCV](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [LMDB](https://lmdb.readthedocs.io/en/release/)

## Results
| Input size | Base model | Train data | Test data | mAP |
|:-:|:-:|:-:|:-:|:-:|
|[300x300](https://drive.google.com/file/d/13sv5gJ3ysNu8_miNk_vFxke5cOFAaVRL/view?usp=sharing) | VGG16 | VOC07 | VOC07 | 0.685 |
|[300x300](https://drive.google.com/file/d/1gw7OhFgoCYQ6BEXFeYbwqoGtlZNLDikL/view?usp=sharing) | VGG16 | VOC07+12 | VOC07 | 0.773 |
|[300x300](https://drive.google.com/file/d/1Vbj_MsdlPucSwQWF6hCviLXqiCEV4yyh/view?usp=sharing) | ResNet50 | VOC07+12 | VOC07 | 0.756 |

Click the **input size** to download the trained model.

## Test image
To test image with trained model,first download the trained model from the link above.
After downloading the model, edit some arguments in `tools/Step4_image_test.py` and run scrpit.
```python
## path to test images
image_path = 'path_to_test_images'

## type of trained model
model_name = 'VGG16'

## path to trained weights
trained_weights_path = 'path_to_trained_model/final.h5'

## model arguments
image_size = (300,300)
...
```

## Dataset
To train or evaluate the SSD model, dataset should be downloaded first. (currently only [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) datasets are supported) Once the dataset is downloaded, edit **dataset path** in `Step0_dataset_to_LMDB.py` and run the scrpit. The dataset is automatically parsed in json format, and dumped into LMDB.
```python
## select one dataset to dump to LMDB
dataset_name = 'VOC07'

## path to VOCdevkit
voc_path = 'path_to_VOC/VOCdevkit'
```

## Evaluation
If the trained model and the dataset have already been downloaded, edit some arguments in `tools/Step3_validation.py` and run the scrpit.
```python
...
## dataset name to test
dataset_name = 'VOC07'

## type of trained model
model_name = 'VGG16'

## path to trained weights
trained_weights_path = 'path_to_trained_model/final.h5'

## model arguments
image_size = (300,300)
...
```

## Training
1. To train a new SSD model, finetuning the existing model trained on ImageNet.
* [VGG16_fc_reduced](https://drive.google.com/file/d/1LJoU80WpzaYHj3t0ofZrvMsu6iNcmwx_/view?usp=sharing): keras converted version of [caffe weights](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6)
* [ResNet50](https://drive.google.com/file/d/1tAio0kN7uB9Dd76u0p4tTQhYvvz3oz8J/view?usp=sharing): zero bias removed version of [keras weights](https://github.com/fchollet/deep-learning-models/releases)

2. `tools/config.py` contains all the parameters used to train SSD. In general, you only need to edit the frequently modified parameters at the top of the script.
```python
## select one dataset used for training
dataset_name = 'VOC07'

## select one base model
model_name = 'VGG16'

## common arguments
image_size = (300,300)
batch_size = 32
base_lr = 0.001
...
```

3. run `tools/Step2_train.py`. All trained weights, configurations, model definition will be saved in `experiments/exp_name`.
```python
## number of gpus for training and validation
num_gpus = 2

## folder name for saving weights, configurations, etc...
exp_name = 'SSD300_VGG16_VOC07+12'
```

