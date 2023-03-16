import matlab.engine
import torch.optim as optim
import numpy as np
import torch
from DQN import *
from ReplayMemory import *

def choose_action(roadGrid, countGrid):
    with torch.no_grad():
        action = policy_net(roadGrid, countGrid).max(1)[1].view(1, 1)
    return action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch['state']).gather(1, action_batch)



NUM_EPISODES = 1
BATCH_SIZE = 16
NUM_STEPS = 100

eng = matlab.engine.start_matlab()
eng.cd('./simulation2', nargout=0)

policy_net = DQN((13, 13), 4).to('cpu')
target_net = DQN((13, 13), 4).to('cpu')
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)

eng.simulationsetup(nargout=0)

roadGrid = eng.workspace['roadGrid']
scenario = eng.workspace['scenario']
running = True

for eps in range(NUM_EPISODES):
    rewards = []
    for i in range(NUM_STEPS):
        action = 1 #choose_action(roadGrid, countGrid)
        (scenario, running, reward, nextRoadGrid, nextCountGrid) = eng.simulate(scenario, roadGrid, eng.workspace['rewardValues'], eng.workspace['gridsize'], eng.workspace['goalGridPosition'], action, eng.workspace['egoVehicleSpeed'], nargout=5)
        nextRoadGrid = torch.tensor(np.asarray(nextRoadGrid))
        nextCountGrid = torch.tensor(np.asarray(nextCountGrid))

        memory.push((roadGrid, countGrid), action, (nextRoadGrid, nextCountGrid), reward)

        optimize_model()

        roadGrid = nextRoadGrid.numpy()
        countGrid = nextCountGrid.numpy()

