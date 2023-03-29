import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()

        #kernel_size = 13
        #stride = 1
        #padding = 0

        self.feature_layer = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        self.value_layer = nn.Sequential(
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Linear(128, 1)
        )

        #self.fc1 = nn.Linear(n_obs, 512)
        # self.fcdropout1 = nn.Dropout(0.5)
        #self.fc5 = nn.Linear(512, 256)
        #self.fcdropout5 = nn.Dropout(0.5)
        #self.fc2 = nn.Linear(256, 64)
        #self.fcdropout2 = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(128, 64)
        #self.fcdropout3 = nn.Dropout(0.5)
        #self.fc4 = nn.Linear(64, n_actions)

    def forward(self, x):

        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fcdropout1(x)
        #x = self.fc5(x)
        #x = F.relu(x)
        #x = self.fcdropout5(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fcdropout2(x)
        #x = self.fc3(x)
        #x = F.relu(x)
        #x = self.fcdropout3(x)
        #x = self.fc4(x)

        return q