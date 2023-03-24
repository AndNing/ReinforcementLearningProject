import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        kernel_size = 13
        stride = 1
        padding = 'same'

        self.rconv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.rbatch1 = nn.BatchNorm2d(16)
        self.rconv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.rbatch2 = nn.BatchNorm2d(32)
        self.rconv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.rbatch3 = nn.BatchNorm2d(32)

        # self.rconv4 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.rbatch4 = nn.BatchNorm2d(64)

        self.cconv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cbatch1 = nn.BatchNorm2d(16)
        self.cconv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cbatch2 = nn.BatchNorm2d(32)
        self.cconv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cbatch3 = nn.BatchNorm2d(32)

        self.dconv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dbatch1 = nn.BatchNorm2d(16)
        self.dconv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dbatch2 = nn.BatchNorm2d(32)
        self.dconv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dbatch3 = nn.BatchNorm2d(32)

        self.flat = nn.Flatten()
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.LazyLinear(4)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, r, c, d):
        # r = F.relu(self.fc1(r))
        # r = F.relu(self.fc2(r))
        # r = self.fc3(r)

        r = r.unsqueeze(1)
        
        r = self.rconv1(r)
        r = self.rbatch1(r)
        r = F.relu(r)
        r = self.rconv2(r)
        r = self.rbatch2(r)
        r = F.relu(r)
        r = self.rconv3(r)
        r = self.rbatch3(r)
        r = F.relu(r)

        # r = self.rconv4(r)
        # r = self.rbatch4(r)
        # r = F.relu(r)

        c = c.unsqueeze(1)

        c = self.cconv1(c)
        c = self.cbatch1(c)
        c = F.relu(c)
        c = self.cconv2(c)
        c = self.cbatch2(c)
        c = F.relu(c)
        c = self.cconv3(c)
        c = self.cbatch3(c)
        c = F.relu(c)
        
        # d = d.unsqueeze(1)
        #
        # d = self.dconv1(d)
        # d = self.dbatch1(d)
        # d = F.relu(d)
        # d = self.dconv2(d)
        # d = self.dbatch2(d)
        # d = F.relu(d)
        # d = self.dconv3(d)
        # d = self.dbatch3(d)
        # d = F.relu(d)
        #
        x = torch.add(r, c)
        # x = torch.add(x, d)
        x = self.flat(r)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x