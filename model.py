"""
@author: ShaktiWadekar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 5, stride=(2, 2))        
        self.conv2 = nn.Conv2d(24, 36, 5, stride=(2, 2))
        self.conv3 = nn.Conv2d(36, 48, 5, stride=(2, 2))
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 3 * 13, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))           
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.drop(x)
        
        x = x.view(-1, 64 * 3 * 13)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

class DriveNet(nn.Module):
    def __init__(self):
        super(DriveNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 5, stride=(2, 2))        
        self.conv2 = nn.Conv2d(24, 36, 5, stride=(2, 2))
        self.conv3 = nn.Conv2d(36, 48, 5, stride=(2, 2))
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.5)
        
        self.fc1_steering = nn.Linear(64 * 3 * 13, 100)
        self.fc2_steering = nn.Linear(100, 50)
        self.fc3_steering = nn.Linear(50, 10)
        self.fc4_steering = nn.Linear(10, 1)
        
        self.fc1_throttle = nn.Linear(64 * 3 * 13, 100)
        self.fc2_throttle = nn.Linear(100, 50)
        self.fc3_throttle = nn.Linear(50, 10)
        self.fc4_throttle = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))           
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.drop(x)
        x = x.view(-1, 64 * 3 * 13)
        
        x_steering = F.elu(self.fc1_steering(x))
        x_steering = F.elu(self.fc2_steering(x_steering))
        x_steering = F.elu(self.fc3_steering(x_steering))
        x_steering = self.fc4_steering(x_steering)
        
        x_throttle = F.elu(self.fc1_throttle(x))
        x_throttle = F.elu(self.fc2_throttle(x_throttle))
        x_throttle = F.elu(self.fc3_throttle(x_throttle))
        x_throttle = self.fc4_throttle(x_throttle)
        
        
        return x_steering,x_throttle