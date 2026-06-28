"""
Program to load and visualize the MNIST dataset.
This program print a random image from the training set, its corresponding label and its vectorized representation.
"""

import pickle
import gzip
import numpy as np
import matplotlib   #I USE ARCH BTW, Skip this line if you are using windows or macOS.
matplotlib.use('QtAgg')  #I USE ARCH BTW, Skip this line if you are using windows or macOS.
import matplotlib.pyplot as plt 



def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

t_set, v_set, test_set = load_data()
n=np.random.randint(0, len(t_set[0]))
image = t_set[0][n].reshape(28, 28)
label = t_set[1][n]
label_vector = vectorized_result(label)