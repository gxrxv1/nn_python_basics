import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


weights=np.random.randn(3)
inputs=[3,2,4]
bias=np.random.randn(1)

activation = sigmoid(np.dot(inputs, weights) + bias)

print(activation)