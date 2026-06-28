import mnist_test_load
import numpy as np

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class basics:
    def __init__(self, o_neurons, i_neurons):
        self.weights = np.random.randn(o_neurons, i_neurons)
        self.biases = np.random.randn(o_neurons)
    
o_neurons = 10
i_neurons = 784  # MNIST images are 28x28 pixels
sample = mnist_test_load.t_set[0]
n_samples = len(sample)  # Number of samples to process
network = basics(o_neurons, i_neurons)
print("Weights:", network.weights.shape)
print("Biases:", network.biases.shape)
print("Sample:", sample.shape)
activation = sigmoid(np.dot(sample, network.weights.T) + network.biases)
print("Activation for sample:", activation.shape)
prediction = np.argmax(activation[0])
print("Predicted digit:", prediction)
print("Actual digit:", mnist_test_load.t_set[1][0])