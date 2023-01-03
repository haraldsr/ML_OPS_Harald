import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    train = zip(*np.load('corruptmnist/train_0.npz'))
    test = zip(*np.load('corruptmnist/test.npz'))
    return train, test
