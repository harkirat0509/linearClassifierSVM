#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:01 2019

@author: harkirat
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
dataset = loadmat('dataset_linear.mat')
x_data = dataset['X']
y_data = dataset['y'].reshape(51,)
m = y_data.shape[0]


plt.scatter(x=x_data[:,0],y=x_data[:,1],c=y_data)
plt.show()

C = 1

weights = np.array([0,0])
bias = 0

def cost_1(inputs):
    return ((inputs<1))*(slope*inputs+1)

def cost_0(inputs):
    return ((inputs>-1)*(-slope*inputs+1))

def loss(weights, bias, x_data, y_data):
    predictions = np.dot(x_data,weights) + bias
    loss = C*(1/m)*np.sum(y_data*cost_1(predictions) + (1-y_data)*cost_0(predictions)) + (1/(2*m))*np.sum(np.power(weights,2))
    return loss

def gradient_1(inputs):
    return ((inputs<1))*(slope)

def gradient_0(inputs):
    return ((inputs>-1))*(-slope)

def learn(weights, bias, y_data, x_data, learning_rate=0.0005, num_iterations = 10000,C=1):
    losses = []
    for i in range(num_iterations):
        predictions = np.dot(x_data,weights) + bias
        db = C*(y_data*gradient_1(predictions) + (1-y_data)*gradient_0(predictions))
        dw = np.dot(x_data.T,db)  + weights
        db = np.sum(db)
        bias = bias - learning_rate*db
        weights = weights - learning_rate*dw
        losses.append(loss(weights,bias,x_data,y_data))
    plt.clf()
    plt.plot(range(num_iterations),losses)
    plt.show()
    return {'weights':weights,
            'bias':bias}

results = learn(weights,bias,y_data,x_data,C=1)

plt.clf()
x = np.linspace(0,4,50)
y = -1*(results['weights'][0]*x+results['bias'])/(results['weights'][1])
plt.plot(x,y)
plt.scatter(x=x_data[:,0],y=x_data[:,1],c=y_data)
plt.show()

results = learn(weights,bias,y_data,x_data,C=100)

plt.clf()
x = np.linspace(0,4,50)
y = -1*(results['weights'][0]*x+results['bias'])/(results['weights'][1])
plt.plot(x,y)
plt.scatter(x=x_data[:,0],y=x_data[:,1],c=y_data)
plt.show()





    
