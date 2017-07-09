## SSD(Single Shot MultiBox Detector) with Keras and Tensorflow
This project is re-implementation version of original [Caffe project](https://github.com/weiliu89/caffe/tree/ssd).

SSD is CNN(Convolutional Neural Network) based object detection framework. It combines predictions from multiple feature maps with different resolutions to handle objects of various sizes. Additionally, it can be run in real time because the network does not resample pixels or features. Please refer to [this research](https://arxiv.org/abs/1512.02325) for more details.

The author's implementation of SSD is based on Caffe and it is hard to study because most sources are written in C++(except some scripts). So to make SSD easier to understand, we write the preparation steps(image transform, sampling, generate prior boxes, compute training targets) in numpy with brief comments. The network model(VGG16, ResNet50) is written in Keras for easy reading and writing of weights(this makes finetuning easy), and the loss function and training loops are written in Tensorflow.

### Prerequisites
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [OpenCV](http://opencv.org/) (read and show image)
* [Numpy](http://www.numpy.org/) (python)
* [LMDB](https://lmdb.readthedocs.io/en/release/) (python)

### Working...
