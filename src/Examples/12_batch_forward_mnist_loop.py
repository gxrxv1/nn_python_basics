import mnist_test_load
import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class basics:
    def __init__(self, o_neurons, i_neurons):
        self.weights = np.random.randn(o_neurons, i_neurons)
        self.biases = np.random.randn(o_neurons,)
    
o_neurons = 10
i_neurons = 784  # MNIST images are 28x28 pixels
n_samples = 50000  # Number of samples to process
sample = mnist_test_load.t_set[0]

network = basics(o_neurons, i_neurons)
print("Weights:", network.weights.shape)
print("Biases:", network.biases.shape)
print("Sample:", np.array(sample).shape)
for j in range(n_samples):
    activation = sigmoid(np.dot(network.weights, np.array(sample[j])) + network.biases)
    print("Activation for sample", j, ":", activation.shape)