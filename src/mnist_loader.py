"""
Program to load MNIST dataset.
"""

import pickle
import gzip
import numpy as np


def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    training_set, validation_set, test_set = load_data()
    training_inputs = [np.reshape(x, (784,1)) for x in training_set[0]]
    training_results = [vectorized_result(x) for x in training_set[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784,1)) for x in validation_set[0]]
    validation_results = validation_set[1]
    validation_data = list(zip(validation_inputs, validation_results))
    test_inputs = [np.reshape(x, (784,1)) for x in test_set[0]]
    test_results = test_set[1]
    test_data = list(zip(test_inputs, test_results))
    return training_data, validation_data, test_data