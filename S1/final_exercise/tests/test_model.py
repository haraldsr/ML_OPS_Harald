from model import MyAwesomeModel
from data import MnistDataset
import torch
from tests import _PATH_DATA

model = MyAwesomeModel(784, 10, [256, 128, 64], drop_p=0.5)
test_set = MnistDataset(dataset_dir=_PATH_DATA, train=False)
trainloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
for images, labels in trainloader:            
    log_ps = model(images)
    assert log_ps.size() == torch.Size([64,10]) 