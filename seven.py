import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

"""Lets backpropagate the error from the output layer to the hidden layer."""
error_output = np.array([[-0.04700074],[ 0.04434257]])
weight_output = np.array([[0.5,0.3,0.1],[0.7, 0.2, 0.4]])
weighted_sum_hidden = np.array([[0.9],[0.5],[0.8]])
error_hidden = np.dot(weight_output.T, error_output) * sigmoid_prime(weighted_sum_hidden)
print("Error at hidden layer:", error_hidden)