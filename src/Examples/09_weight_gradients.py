import numpy as np

"""Lets update the weights and biases of the neural network using the backpropagation algorithm. The backpropagation algorithm is a supervised learning algorithm that is used to train artificial neural networks."""

weight_output = np.array([[0.5,0.3,0.1],[0.7, 0.2, 0.4]])
weight_hidden = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
bias_output = np.array([[0.1],[0.2]])
bias_hidden = np.array([[0.1],[0.2],[0.3]])
grad_bias_output = np.array([[-0.04700074], [ 0.04434257]])
grad_bias_hidden = np.array([[-0.00173541], [-0.00129399], [ 0.00220962]])
grad_weight_output = np.array([[-0.03341515, -0.02925605, -0.03242931],[ 0.03152533, 0.02760145, 0.03059524]])
grad_weight_hidden = np.array([[-0.00017354, -0.00156187],[-0.0001294, -0.00116459],[0.00022096, 0.00198866]])
learning_rate = 0.1
weight_output -= learning_rate * grad_weight_output
weight_hidden -= learning_rate * grad_weight_hidden
bias_output -= learning_rate * grad_bias_output
bias_hidden -= learning_rate * grad_bias_hidden
print("Updated weights at output layer:", weight_output)
print("Updated weights at hidden layer:", weight_hidden)
print("Updated biases at output layer:", bias_output)
print("Updated biases at hidden layer:", bias_hidden)