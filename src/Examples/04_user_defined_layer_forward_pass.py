import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))

class basics:
    def __init__(self, o_neurons, i_neurons):
        self.weights = np.random.randn(o_neurons, i_neurons)
        self.biases = np.random.randn(o_neurons, 1)

i_neurons = int(input("Enter the number of input neurons: "))
o_neurons = int(input("Enter the number of output neurons: "))
inputs = []
for i in range(i_neurons):
    value =np.random.randn()
    inputs.append([value])
network = basics(o_neurons, i_neurons)
print("Weights:", network.weights)
print("Biases:", network.biases)
print("Inputs:", inputs)
activation = sigmoid(np.dot(network.weights, inputs) + network.biases)
print("Activation:", activation)