import numpy as np
import os
import glob
from torch.utils.data import Dataset

def get_mnist_data(folderpath, train: bool = True):
    filepaths = glob.glob(os.path.join(folderpath, "*.npz")) 

    mnist_data = {}
    for file in filepaths:
        data = np.load(file)
        images = data['images']
        labels = data['labels']

        if 'train' in file and train:
            if len(mnist_data) == 0:
                mnist_data['images'] = images
                mnist_data['labels'] = labels
            else:
                mnist_data['images'] = np.concatenate((mnist_data['images'], images), axis=0)
                mnist_data['labels'] = np.concatenate((mnist_data['labels'], labels), axis=0)
        elif not train:
            mnist_data['images'] = images
            mnist_data['labels'] = labels

    return mnist_data['images'], mnist_data['labels']

class MnistDataset(Dataset):
    def __init__(self, dataset_dir, train):
        self.dataset_dir = dataset_dir
        self.images, self.labels = get_mnist_data(dataset_dir, train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
