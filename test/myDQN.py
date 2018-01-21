import torch.nn.functional as F
from torch import nn


class MyDQN(nn.Module):

    def __init__(self, params):
        super(MyDQN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, 3)  # Layer 1 (Convolutional)
        self.conv2 = nn.Conv2d(16, 32, 3)  # Layer 2 (Convolutional)
        self.fc1 = nn.Linear(params['width'] * params['height'] * 32, 256)  # Layer 3 (Fully connected)
        self.fc2 = nn.Linear(256, 4)  # Layer 4 (Fully connected)

        self.params = params

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.params['width'] * self.params['height'] * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
