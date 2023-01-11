from tests import _PATH_DATA
from data import MnistDataset
import numpy as np

dataset_train = MnistDataset(dataset_dir=_PATH_DATA, train=True)
dataset_test = MnistDataset(dataset_dir=_PATH_DATA, train=False)
assert len(dataset_train) == 25000 
assert len(dataset_test) == 5000
for i in dataset_train:
    assert np.shape(i[0]) == (28,28)

for i in np.unique(dataset_train.labels):
    if i not in np.arange(0,10):
        assert i not in np.arange(0,10)