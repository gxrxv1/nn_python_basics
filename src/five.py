import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))

class basics:
    def __init__(self, o_neurons, i_neurons):
        self.weights = np.random.randn(o_neurons, i_neurons)
        self.biases = np.random.randn(o_neurons, 1)

i_neurons = int(input("Enter the number of input neurons: "))
o_neurons = int(input("Enter the number of output neurons: "))
n_samples = int(input("Enter the number of samples: "))
sample=[]
for j in range(n_samples):
    inputs = []
    for i in range(i_neurons):
        value = float(input(f"Enter the value for input neuron {i+1} in sample {j+1}: "))
        inputs.append(value)
    sample.append(inputs)

network = basics(o_neurons, i_neurons)
print("Weights:", network.weights)
print("Biases:", network.biases)
activation = sigmoid(np.dot(network.weights, np.array(sample).T) + network.biases)
print("Activation:", activation)