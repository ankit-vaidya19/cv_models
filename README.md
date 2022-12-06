# cv_models

This repository contains the implementation of basic CV models using TensorFlow and Keras.

All the models are trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasest. The dataset consists of 60000 32x32 rgb images belonging to 10 classes - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. There are 6000 images per class.

The implemented models are:


| Model     | Reference                                                                                                                                 |
| :-------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| [AlexNet](https://github.com/ankit-vaidya19/cv_models/blob/master/models/AlexNet.py)   |  [Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E. (2017-05-24). "ImageNet classification with deep convolutional neural networks"](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)                                                                 |
| [VGG16](https://github.com/ankit-vaidya19/cv_models/blob/master/models/VGG16.py)     |  [Simonyan, K. and Zisserman, A. (2015) Very Deep Convolutional Networks for Large-Scale Image Recognition. The 3rd International Conference on Learning Representations (ICLR2015)](https://arxiv.org/abs/1409.1556)                                                                                            |
| [ResNet](https://github.com/ankit-vaidya19/cv_models/blob/master/models/ResNet.py)    | [K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)](https://arxiv.org/abs/1512.03385)                                                                                                               |

