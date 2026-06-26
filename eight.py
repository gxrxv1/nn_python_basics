import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

"""Lets calculate the bias gradients for the hidden layer and the output layer."""
error_output = np.array([[-0.04700074],[ 0.04434257]])
error_hidden = np.array([[-0.00173541],[-0.00129399],[ 0.00220962]])

grad_bias_output = error_output
grad_bias_hidden = error_hidden
print("Gradient of bias at output layer:", grad_bias_output) 
print("Gradient of bias at hidden layer:", grad_bias_hidden) 