import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

"""Lets calculate the weight gradients for the hidden layer and the output layer. Considering a neural network with 2 input neurons, 3 hidden neurons and 2 output neurons."""
error_output = np.array([[-0.04700074],[ 0.04434257]])
error_hidden = np.array([[-0.00173541],[-0.00129399],[ 0.00220962]])
activation_hidden = sigmoid(np.array([[0.9],[0.5],[0.8]]))
activation_input =(np.array([[0.1],[0.9]]))
grad_weight_output = np.dot(error_output, activation_hidden.T)
grad_weight_hidden = np.dot(error_hidden, activation_input.T)
print("Gradient of weight at output layer:", grad_weight_output)
print("Gradient of weight at hidden layer:", grad_weight_hidden)