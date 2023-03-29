import random

import matlab.engine
import torch.optim as optim
import numpy as np
import torch
from DQN import *
from ReplayMemory import *
#from PReplayMemory import *
from torchinfo import summary
import math
import matplotlib
import matplotlib.pyplot as plt
from Network import *
from torch.nn.utils import clip_grad_norm_



DEVICE = 'cuda'
NUM_EPISODES = 1000
BATCH_SIZE = 128
NUM_STEPS = 600
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10_000
GAMMA = 0.99
TAU = 0.005
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 3000
INPUT_SIZE = (13,13)
NUM_ACTIONS = 4
n_obs = 13*13*4
sync_freq = 500



policy_net = DQN(n_obs, NUM_ACTIONS).to(DEVICE)
target_net = DQN(n_obs, NUM_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer1 = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
#optimizer2 = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
#memory = PReplayMemory(MEMORY_SIZE)
memory = ReplayMemory(MEMORY_SIZE)
steps_done = 0

def choose_action(state):

    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1), eps_threshold
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=DEVICE, dtype=torch.long), eps_threshold


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #non_final_next_states_road_position = torch.cat([s[0] for s in batch.next_state if s[0] is not None])
    #non_final_next_states_road_target = torch.cat([s[1] for s in batch.next_state if s[1] is not None])
    #non_final_next_states_road_roads = torch.cat([s[2] for s in batch.next_state if s[2] is not None])

    # non_final_next_states_count = torch.stack([s[1] for s in batch.next_state if s[1] is not None])

    state_batch = torch.cat([s for s in batch.state])
    #state_batch_road_position = torch.cat([s[0] for s in batch.state])
    #state_batch_road_target = torch.cat([s[1] for s in batch.state])
    #state_batch_road_roads = torch.cat([s[2] for s in batch.state])
    # state_batch_count = torch.stack([s[1] for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # state_action_values = policy_net(state_batch_road, state_batch_count).gather(1, action_batch)
    state_action_values1 = policy_net(state_batch).gather(1, action_batch)
    #state_action_values2 = target_net(state_batch_road_position, state_batch_road_target, state_batch_road_roads).gather(1, action_batch)


    #next_state_values1 = torch.zeros(BATCH_SIZE, device=DEVICE)
    #next_state_values2 = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        #next_state_values1[non_final_mask] = policy_net(non_final_next_states_road_position, non_final_next_states_road_target, non_final_next_states_road_roads).max(1)[0]
        #next_state_values2[non_final_mask] = target_net(non_final_next_states_road_position, non_final_next_states_road_target, non_final_next_states_road_roads).max(1)[0]
        #next_state_values = torch.min(next_state_values1, next_state_values2)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss1 = criterion(state_action_values1, expected_state_action_values.unsqueeze(1))
    #loss2 = criterion(state_action_values2, expected_state_action_values.unsqueeze(1))
    #loss1 = torch.mean((state_action_values1 - expected_state_action_values.unsqueeze(1))**2 * weights)
    # Optimize the model
    optimizer1.zero_grad()
    loss1.backward()
    # In-place gradient clipping
    clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer1.step()

    #optimizer2.zero_grad()
    #loss2.backward()
    #optimizer2.step()

    #memory.update_priorities(tree_idxs, td_error.numpy())

    return loss1.item()


# model = DQN((13,13),4)
# batch_size = 50

eng = matlab.engine.start_matlab()
eng.cd('./simulation2', nargout=0)
episode_totalreward = []
episode_loss = []

for eps in range(NUM_EPISODES):
    eng.workspace['seednum'] = eps*2+3
    eng.setseed(nargout=0)
    eng.simulationsetup(nargout=0)
    staticRoadGrid = np.asarray(eng.workspace['staticRoadGrid'])
    roadGridRoads = np.where(staticRoadGrid == 30, 10, staticRoadGrid)
    roadGridRoads = np.where(roadGridRoads == 10, 1, 0)
    #roadGridRoads = torch.tensor(roadGridRoads, device=DEVICE, dtype=torch.float32)
    roadGridTarget = np.where(staticRoadGrid == 30, 1, 0)
    #roadGridTarget = torch.tensor(roadGridTarget, device=DEVICE, dtype=torch.float32)
    roadGrid = np.asarray(eng.workspace['updatedRoadGrid'])
    roadGridPosition = np.where(roadGrid == 20, 1, 0)
    #roadGridPosition = torch.tensor(roadGridPosition, device=DEVICE, dtype=torch.float32)
    countGrid = np.asarray(eng.workspace['countGrid'])
    #countGrid = torch.tensor(countGrid, device=DEVICE, dtype=torch.float32)
    scenario = eng.workspace['scenario']
    state = np.concatenate((roadGridRoads.flatten(), roadGridTarget.flatten(), roadGridPosition.flatten(), countGrid.flatten()))
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    totalreward = 0
    loss1 = 0
    loss2 = 0
    steps = 0
    running = True
    while running and steps < NUM_STEPS:
        steps += 1
        (action, epsilon) = choose_action(state)

        (scenario, running, reward, nextRoadGrid, nextCountGrid) = eng.simulate(scenario, eng.workspace['rewardValues'], eng.workspace['gridsize'], eng.workspace['goalGridPosition'], action.item()+1, eng.workspace['egoVehicleSpeed'], eng.workspace['staticRoadGrid'], nargout=5)

        nextRoadGrid = np.asarray(nextRoadGrid)
        nextRoadGridPosition = np.where(nextRoadGrid == 20, 1, 0)
        #nextRoadGridPosition = torch.tensor(np.where(nextRoadGrid == 20, 1, 0), device=DEVICE, dtype=torch.float32)
        #nextRoadGridTarget = roadGridTarget
        #nextRoadGridRoads = roadGridRoads
        nextCountGrid = np.asarray(nextCountGrid)
        next_state = np.concatenate((roadGridRoads.flatten(), roadGridTarget.flatten(), nextRoadGridPosition.flatten(), nextCountGrid.flatten()))
        next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        reward = torch.tensor([reward], device=DEVICE)
        totalreward = totalreward + reward.item()*(GAMMA**(steps-1))

        if running == False:
            next_state = None
            memory.push(state, action, next_state, reward)
            #nextRoadGrid = None
            #nextRoadGridPosition = None
            #nextRoadGridRoads = None
            #nextRoadGridTarget = None
            #nextCountGrid = None
            #memory.push((roadGridPosition.flatten().unsqueeze(0),
            #             roadGridTarget.flatten().unsqueeze(0),
            #             roadGridRoads.flatten().unsqueeze(0)),
            #            action,
            #            (nextRoadGridPosition,
            #             nextRoadGridTarget,
            #             nextRoadGridRoads),
            #            torch.tensor([reward], device=DEVICE))

        else:
            memory.push(state, action, next_state, reward)
            #memory.push((roadGridPosition.flatten().unsqueeze(0),
            #             roadGridTarget.flatten().unsqueeze(0),
            #            roadGridRoads.flatten().unsqueeze(0)),
            #            action,
            #            (nextRoadGridPosition.flatten().unsqueeze(0),
            #             nextRoadGridTarget.flatten().unsqueeze(0),
            #             nextRoadGridRoads.flatten().unsqueeze(0)),
            #            torch.tensor([reward], device=DEVICE))

        #memory.add()

        losses = optimize_model()
        if losses is not None:
            episode_loss.append(losses)

        state = next_state

        if steps % sync_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        """
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        """

        if steps_done % 100 == 0:
            print(totalreward, losses, steps_done, epsilon)

    print('Episode: ', eps, 'Steps: ', steps, 'Loss: ', losses, 'Total Reward: ', totalreward)
    episode_totalreward.append(totalreward)

print('Complete')
plt.plot(episode_totalreward)
plt.ylabel('total reward per episode')
plt.xlabel('episode number')
plt.show()

plt.plot(episode_loss)
plt.ylabel('loss per episode')
plt.xlabel('episode number')
plt.show()
