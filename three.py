import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))

class basics:
    def __init__(self, o_neurons, i_neurons):
        self.weights = np.random.randn(o_neurons, i_neurons)
        self.biases = np.random.randn(o_neurons,1)

o_neurons = 2
i_neurons = 3
y=np.array([[3], [2], [4]])

network = basics(o_neurons, i_neurons)
print("Weights:", network.weights)
print("Biases:", network.biases)
activation = sigmoid(np.dot(network.weights,y) + network.biases)
print("activation:", activation)