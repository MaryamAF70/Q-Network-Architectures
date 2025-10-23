import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN_QNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=3):
        super(CNN_QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)  # خروجی: 32×20×20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)             # خروجی: 64×9×9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)             # خروجی: 64×7×7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, num_actions)
    def forward(self, x):
        x = F.relu(self.conv1(x))     # Feature Maps: Conv1
        x = F.relu(self.conv2(x))     # Feature Maps: Conv2
        x = F.relu(self.conv3(x))     # Feature Maps: Conv3
        x = x.view(x.size(0), -1)     # Flatten
        x = F.relu(self.fc1(x))
        return self.out(x)