#Let's start with importing the necessary libraries and the dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

#Let's have a quick look at one of the samples of the images from the dataset

#imgIndex = 9
#image = xtrain[imgIndex]
#print("Image Label :",ytrain[imgIndex])
#plt.imshow(image)
#plt.show()

#Let's have a look at the shape of both the training and the test data

print(xtrain.shape)
print(xtest.shape)


#NEURAL NETWORK ARCHITECTURE

#I will build a neural network architecture with 2 hidden layers.

#Sequential: linear stack of layers that can be used to create neural networks. It allows you to construct
#models layer by layer in a step by step fashion.

#Flatten: Function that converts a multi-dimensional tensor into a one-dimentional tensor. This is often used when you're
#transitioning between convolutional layers and dense layers in a neural network 

#Dense: this refers to a fully connected layers, where each neuron in the layer
#is connected to every neuron in the previous layer. The first argument is the number of neurons in that layer. 
#The activation parameter is used to apply an activation function which introduces non-linearity into the output of a neuron. This helps
#the model to learn from the complex patterns in the data. For example, "relu" stands for Rectified Linear Unit, one of the most commonly used 
#activation function in deep learning models. The "softmax" function is often used in the final layer of a neural network model for multi-class
#classification problems, is returns the probabilities of each class, with all probabilities summing to 1. 


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

#Before training the model, I will split the training data into training and validation sets: 

xvalid, xtrain = xtrain[:5000] / 255.0, xtrain[5000:] / 255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

#TRAINING A CLASSIFICATION MODEL WITH NEURAL NETWORKS

#sparse_categorical_crossentropy: loss function suitable for classification problems with multiple classes. It is used when the classes are
#mutually exclusive (each entry belongs to precisely one class). the "sparse" part means that the labels are represented as integers.

#optimizer: algorithm that the model uses to minimize the loss function. "sgd" stands for Stochastic Gradient Descent , a common optimizer. Other popular optimizers include Adam and RMSprop.

#metrics: measures of quality that the model should track during training and testing. "accuracy" is a common metric for classification problems. It calculates
#the proportion of the correct predictions over total predictions. Other metrics could include precision, recall, F1 score etc., depending on the problem. 

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
history = model.fit(xtrain, ytrain, epochs=30,validation_data=(xvalid,yvalid))

#Let's have a look at the predictions

new = xtest[:5]
predictions = model.predict(new)
print(predictions)

#This is how we can look at the predicted classes:

classes= np.argmax(predictions, axis=1)
print(classes)