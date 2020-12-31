# Multilabel (Atmospheric Conditions and Land Cover) Classification of Amazon Rainforest Satellite Images

## Project Idea
To help governments and local stakeholders understand the location of deforestation and human encroachment on forests, I will analyze small-scale deforestation and human activity influences from Amazon Rainforest satellite images. The goal of this analysis is to correctly label images with atmospheric conditions, common land cover / land use phenomena, and rare land cover / land use phenomena. My algorithm will use a convolutional network to output a set of predicted labels. In this project, I will explore and analyze how different architectures of convolutional neural network models perform in terms of training time, f2 score and generalization ability.

## Problem Under Investigation
In this project, we take on the Kaggle challenge “Planet: Understanding the Amazon from Space”. Our goal is to accurately label satellite images with atmospheric conditions, land use and land cover. The input to our algorithms is a satellite image of the Amazon basin. The image is represented as a 256x256 grid of pixels and 3 or 4 channels, described further in the data section. We then use variations of Convolutional Neural Network to output one or more predicted labels, belonging to a set of 17 possible labels, describing atmospheric conditions, land cover and land use in the input image. The primary performance metric is the average F2 score on the validation or test dataset.


## Algorithms / Models
In this project different architectures of Convolutional Neural Network is used for this problem. I have also tuned hyperparameters to achieve the best performance with the given dataset. Other than these, I have also experimented with several activation functions like sigmoid, softmax, tanh, etc to analyze which one works best for the given problem and why. Following architecutres of CNN were implemented and used for this project:

- [4 Layer CNN](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/4L-CNN.py)
- [6 Layer CNN](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/6L-CNN.py)
- [8 Layer CNN](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/8L-CNN.py)
- [10 Layer CNN](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/10L-CNN.py)
- [ResNet](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/resnet.py)
- [VGG-16](https://github.com/muneeb706/multilabel-classification/blob/master/cnn-models/vgg16.py)

## Dataset Details
The dataset is provided by Kaggle which contains 40479 labeled satellite images and there are 17 classes. These classes address different aspects of the image content, for example, atmospheric conditions and land cover / user. In the training dataset, the labels or classes are not evenly distributed. There are two types of images, JPG and TIF. Both JPG and TIF images are 256x256 pixels. The JPG images have 3 channels - Red, Green, and Blue. The TIF images have 4 channels - Red, Green, Blue, and IR. The labels have significant correlations. For example, every image has exactly one atmospheric condition label from among clear, haze, partly cloudy and cloudy. Labels like “habitation” tend to occur with other markers of human activity. “Cultivation” and “agriculture” don’t co-occur in images.

Dataset can be downloaded from 
[planet-understanding-the-amazon-from-space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

## Further Details
Further details related to the project can be found from [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/overview). Results, conclusion and other reports have been recorded in the form of an HTML project which can be found in the [project-report](https://github.com/muneeb706/multilabel-classification/tree/master/project-report) directory.

## Software
**IDE:** PyCharm, 
**Deep Learning Framework:** TensorFlow, 
**Programming Language:** Python

## Related Research Articles
- [Amazon Rainforest Satellite Image Labelling Challenge](http://cs231n.stanford.edu/reports/2017/pdfs/902.pdf)
- [Labeling Satellite Imagery with Atmospheric Conditions and Land Cover](http://cs231n.stanford.edu/reports/2017/pdfs/9.pdf)
- [Classification of natural landmarks and human footprints of Amazon using satellite data](http://cs231n.stanford.edu/reports/2017/pdfs/907.pdf)

