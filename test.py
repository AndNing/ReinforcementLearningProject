import numpy as np
import torch
import random
import torch.optim as optim
from PrioritizedReplayBuffer import *
from ReplayBuffer import *
from NoisyLinear import *
from Network import *
import matlab.engine
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from matlabenv import *
from DQN import *
from VanillaDQN import *
from Network import *
from NoisyDQN import *
import math
import torch.nn.functional as F

gridx = 13  ## width of grid
gridy = 13  ## length of grid
num_of_grid = 4  ## number of grid
action_space = 4  ## size of action space
stopTime = 1000  ## maximum steps per episode
gamma = 0.99  ## discount rate
epsilon_threshold = 0.05  ## epsilon threshold for the epsilon-greedy exploration

## parameters for the rainbow DQN
atom_size = 11
v_min = -10.0
v_max = 10.0



env = matlabenv(4, action_space, stopTime, gridx=gridx, gridy=gridy, seednum=777)

seed = 251

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)


device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
print(device)

support = torch.linspace(v_min, v_max, atom_size).to(device)



#dqn = DQN(env.observation_space, env.action_space).to(device)
#dqn.load_state_dict(torch.load('models/model_weights_Dueling.pth'))
#dqn = VanillaDQN(env.observation_space, env.action_space).to(device)
#dqn.load_state_dict(torch.load('models/model_weights_Vanilla.pth'))

dqn =Network(env.observation_space, env.action_space, atom_size, support).to(device)
dqn.load_state_dict(torch.load('models/model_weights_Rainbow3.pth'))
dqn.eval()
state = env.reset()
done = False
score = 0
step = 0

## action selection for the double and dueling agents
def select_action(state: np.ndarray) -> np.ndarray:
    """Select an action from the input state."""
    # epsilon greedy policy
    if epsilon_threshold > np.random.random():
        selected_action = env.action_sample()
    else:
        selected_action = dqn(
            torch.FloatTensor(state).to(device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()

    return selected_action

## action selection for the rainbow agent
def select_action_Rainbow(state: np.ndarray) -> np.ndarray:
    """Select an action from the input state."""
    # NoisyNet: no epsilon greedy action selection
    selected_action = dqn(
        torch.FloatTensor(state).to(device)
    ).argmax()
    selected_action = selected_action.detach().cpu().numpy()


    return selected_action

scores = []  ## dicounted cumulative rewards per episode
encounters = []  ## number of traffic count per episode
off_roads = []  ## total number of being off-road per episode
num_termination = 0  ## number of incomplete episode
steps = []  ## number of steps per episode
for i in range(100):
    state = env.reset()
    done = False
    score = 0
    step = 0
    encounter = 0
    off_road = 0
    while not done and not env.termination:
        with torch.no_grad():
            action = select_action_Rainbow(state)
        next_state, reward, done = env.step(action)
        encounter += np.sum(np.multiply(env.countGrid, env.roadGridPosition))
        off_road += np.sum(np.multiply(env.roadGridRoads - 1, -env.roadGridPosition))
        state = next_state
        score += reward*(gamma**step)
        step += 1
    scores.append(score)
    encounters.append(encounter)
    off_roads.append(off_road)
    if env.termination:
        num_termination += 1
    steps.append(step)

socre_mean = sum(scores) / len(scores)
print("score_mean: ", socre_mean)
scores = np.array(scores)
score_std = scores.std()
print("score_std: ", score_std)
encounter_mean = sum(encounters) / len(encounters)
off_road_mean = sum(off_roads) / len(off_roads)
print("encounter_mean: ", encounter_mean)
print("off_road_mean: ", off_road_mean)
print("num_termination: ", num_termination)
env.close()
