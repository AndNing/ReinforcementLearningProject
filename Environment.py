import numpy as np
import torch
from matlabenv import *
from RainbowDQNAgent import *
from DuelingAgent import *
from PEDuelingAgent import *
from VanillaDQNAgent import *
from RainbowwithoutPurpleAgent import *
import random

# environment
gridx = 13
gridy = 13
num_of_grid = 4
action_space = 4
stopTime = 200

env = matlabenv(4, action_space, stopTime, gridx=gridx, gridy=gridy)

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 50_000
memory_size = 10000
batch_size = 128
target_update = 150
epsilon_decay = 3000
lr = 1e-4

#train
#agent = RainbowDQNAgent(env, memory_size, batch_size, target_update, lr)
agent = RainbowwithoutPurpleAgent(env, memory_size, batch_size, target_update, lr)
#agent = PEDuelingAgent(env, memory_size, batch_size, target_update, epsilon_decay, lr, n_step=3)
#agent = PEDuelingAgent(env, memory_size, batch_size, target_update, epsilon_decay, lr)
#agent = DuelingAgent(env, memory_size, batch_size, target_update, epsilon_decay, lr)
#agent = VanillaDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, lr)

losses, scores, discounted_total = agent.train(num_frames)
#np.savetxt('Rainbowlosses3.csv', losses, delimiter=',')
#np.savetxt('Rainbowscores3.csv', scores, delimiter=',')
#np.savetxt('Rainbowdiscounted3.csv', discounted_total, delimiter=',')
np.savetxt('RainbowwithoutPurpleAgent.csv', losses, delimiter=',')
np.savetxt('RainbowwithoutPurpleAgent.csv', scores, delimiter=',')
np.savetxt('RainbowwithoutPurpleAgent.csv', discounted_total, delimiter=',')
#np.savetxt('StepPEDuelinglosses.csv', losses, delimiter=',')
#np.savetxt('StepPEDuelingscores.csv', scores, delimiter=',')
#np.savetxt('StepPEDuelingdiscounted.csv', discounted_total, delimiter=',')
#np.savetxt('PEDuelinglosses.csv', losses, delimiter=',')
#np.savetxt('PEDuelingscores.csv', scores, delimiter=',')
#np.savetxt('PEDuelingdiscounted.csv', discounted_total, delimiter=',')
#np.savetxt('Duelinglosses.csv', losses, delimiter=',')
#np.savetxt('Duelingscores.csv', scores, delimiter=',')
#np.savetxt('Duelingdiscounted.csv', discounted_total, delimiter=',')
#np.savetxt('Vanillalosses.csv', losses, delimiter=',')
#np.savetxt('Vanillascores.csv', scores, delimiter=',')
#np.savetxt('Vanialladiscounted.csv', discounted_total, delimiter=',')
print('training is complete!')
#torch.save(agent.dqn.state_dict(), 'model_weights_Vanilla.pth')
#torch.save(agent.dqn.state_dict(), 'model_weights_stepPEDueling.pth')
#torch.save(agent.dqn.state_dict(), 'model_weights_PEDueling.pth')
#torch.save(agent.dqn.state_dict(), 'model_weights_Dueling.pth')
torch.save(agent.dqn.state_dict(), 'model_weights_RainbowwithoutPurple.pth')
#torch.save(agent.dqn.state_dict(), 'model_weights_Rainbow3.pth')

