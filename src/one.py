import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


weights=[0.7,0.3,-0.5]
inputs=[3,2,4]
bias=2

activation = sigmoid(np.dot(inputs, weights) + bias)

print(activation)
