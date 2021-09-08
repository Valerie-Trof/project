# -*- coding: utf-8 -*-
"""
@author: Trofimova Valeriya
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import random
d=pd.read_csv('ypred.csv')#delete Date column
data=pd.DataFrame(d, columns=['Country Code','Txn Code','Qtr','Year', 'Volume'])
data['Year'] = data['Year']/max(data['Year'])
data['Volume'] = data['Volume']/max(data['Volume'])
X = data.drop('Volume', axis = 1)
y = pd.DataFrame(data['Volume']).to_numpy()
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 96)
xtrain=xtrain.to_numpy()
xtest=xtest.to_numpy()
#Initialize weights  
def init(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
#Neuron activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation
#Neuron transfer
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))

def forward_prop(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
#s(1-s)
def transfer_deriv(output):
    return output * (1.0 - output)
def backprop_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_deriv(neuron['output']) 
#weight = weight + learning_rate * error * input
#update weights            
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']         
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        c = 0
        for row in train:
            outputs = forward_prop(network, row)
            expected = ytrain[c]
            sum_error += np.sum([(expected - outputs)**2])
            backprop_error(network, expected)
            update_weights(network, row, l_rate)
            c+=1
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))            
dataset = xtrain
network = init(4,1,1)# 4 input layers, 1 hidden, 1 output
train_network(network, dataset, 0.1, 1000, 1)        
def predict(network, row):
    outputs = forward_prop(network, row)
    return outputs
c = 0
for row in xtest:
    prediction = predict(network, row)
    print("Predicted: ", prediction)
    print("True: ", ytest[c])
    c+=1        
  