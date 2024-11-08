# Face-Emotion-Detection-using FER 2013
This repository contains code for training a Convolutional Neural Network (CNN) model on the FER2013 dataset for emotion detection. The model is implemented using PyTorch, and the goal is to classify facial expressions into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Table of contents
- Project Overview
- Dataset
- Model Performance
- Model Architecture
- Accuracy Results
- Acknowledgments

## Project Overview
Facial expression recognition is a challenging problem that aims to classify human emotions based on facial features. This project uses the FER2013 dataset, which contains 48x48 grayscale images of faces, to train a CNN model to detect emotions. The images are resized to 224x224 for training, and the model is built using PyTorch.

## Dataset
The FER2013 dataset consists of:

- 28,709 training images
- 3,589 test images
( Used sample images only)
The images are labeled into one of the seven categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral
   
You can download the dataset from the [Kaggle FER2013 page](https://www.kaggle.com/datasets/damnithurts/fer2013-dataset-images)

## Model Performance
In this project, we compare the performance of a custom Convolutional Neural Network (CNN) with some pre-trained models including ResNet18, GoogleNet on the FER2013 dataset based on accuracy.

## Model Architechture
The custom Convolutional Neural Network (CNN) model is built with the following key components:

- Multiple convolutional layers with ReLU activation.
- Pooling layers for downsampling.
- Fully connected layers for final emotion classification.
- The images are resized to 128x128 to allow for deeper CNN architectures.
  
We have used batch normalization and dropout in this model.

## Accuracy Results:
- Custom CNN Model:45%
- ResNet18: 40%
- GoogleNet: 38%
  
The custom CNN model outperforms all pre-trained models in terms of accuracy. This indicates that the custom model is better suited for the task of facial emotion recognition on the FER2013 dataset.

## Acknowledgments
- The dataset is from the [FER2013 Kaggle Competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- The project uses PyTorch as the deep learning framework.


