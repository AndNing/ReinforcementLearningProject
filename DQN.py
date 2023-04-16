import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()


        self.feature_layer = nn.Sequential(
            nn.Linear(n_obs, 128),
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


    def forward(self, x):

        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)


        return q

