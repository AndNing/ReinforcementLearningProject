import random

import matlab.engine
import torch.optim as optim
import numpy as np
import torch
from DQN import *
from ReplayMemory import *
from torchinfo import summary
import math

def choose_action(roadGridPosition, roadGridTarget, roadGridRoads,steps):
    with torch.no_grad():
        epsilon = epsilon_by_frame(steps)
        if random.random() < epsilon:
            action = torch.tensor([[random.randrange(NUM_ACTIONS)]], device=DEVICE, dtype=torch.int)
        else:
            # action = policy_net(roadGrid.unsqueeze(0), countGrid.unsqueeze(0)).max(1)[1].view(1, 1)
            action = policy_net(roadGridPosition.unsqueeze(0),roadGridTarget.unsqueeze(0), roadGridRoads.unsqueeze(0)).max(1)[1].view(1, 1)

    return action, epsilon

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s[0] is not None, batch.next_state)), dtype=torch.bool)

    # non_final_next_states_road = torch.stack([s[0] for s in batch.next_state if s[0] is not None])
    non_final_next_states_road_position = torch.stack([s[0] for s in batch.next_state if s[0] is not None])
    non_final_next_states_road_target = torch.stack([s[1] for s in batch.next_state if s[1] is not None])
    non_final_next_states_road_roads = torch.stack([s[2] for s in batch.next_state if s[2] is not None])

    # non_final_next_states_count = torch.stack([s[1] for s in batch.next_state if s[1] is not None])

    # state_batch_road = torch.stack([s[0] for s in batch.state])
    state_batch_road_position = torch.stack([s[0] for s in batch.state])
    state_batch_road_target = torch.stack([s[1] for s in batch.state])
    state_batch_road_roads = torch.stack([s[2] for s in batch.state])
    # state_batch_count = torch.stack([s[1] for s in batch.state])
    action_batch = torch.stack([s for s in batch.action])
    reward_batch = torch.stack([s for s in batch.reward])

    # print(state_batch_count.shape)
    # print(state_batch_road.shape)

    # state_action_values = policy_net(state_batch_road, state_batch_count).gather(1, action_batch)
    state_action_values = policy_net(state_batch_road_position, state_batch_road_target, state_batch_road_roads).gather(1, action_batch)


    # print(state_action_values.shape)
    # print(state_action_values.shape)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        # print(non_final_next_states_road)
        # print(non_final_next_states_count)
        # print(non_final_mask)
        # print(target_net(non_final_next_states_road, non_final_next_states_count))
        # print(target_net(non_final_next_states_road, non_final_next_states_count).max(1))
        # print(target_net(non_final_next_states_road, non_final_next_states_count).max(1)[0])
        # next_state_values[non_final_mask] = target_net(non_final_next_states_road, non_final_next_states_count).max(1)[0]
        next_state_values[non_final_mask] = target_net(non_final_next_states_road_position,non_final_next_states_road_target, non_final_next_states_road_roads).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA).unsqueeze(1) + reward_batch
    # print((next_state_values * GAMMA).unsqueeze(1).shape,reward_batch.shape)
    # print(expected_state_action_values)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print(state_action_values)
    # print(expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


# model = DQN((13,13),4)
# batch_size = 50
# print(summary(model, input_size=(batch_size, 13, 13)))

DEVICE = 'cpu'

NUM_EPISODES = 1000
BATCH_SIZE = 25
NUM_STEPS = 1000
LEARNING_RATE = 0.00001
MEMORY_SIZE = 100000
GAMMA = 0.99
TAU = 0.005
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 2000

epsilon_by_frame = lambda frame_idx: EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1. * frame_idx / EPSILON_DECAY)

eng = matlab.engine.start_matlab()
eng.cd('./simulation2', nargout=0)

INPUT_SIZE = (13,13)
NUM_ACTIONS = 4

policy_net = DQN(INPUT_SIZE, NUM_ACTIONS).to('cpu')
policy_net(torch.ones(INPUT_SIZE).unsqueeze(0),torch.ones(INPUT_SIZE).unsqueeze(0),torch.ones(INPUT_SIZE).unsqueeze(0))
target_net = DQN(INPUT_SIZE, NUM_ACTIONS).to('cpu')
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

eng.simulationsetup(nargout=0)

running = True
steps = 0
totalsteps = 0

for eps in range(NUM_EPISODES):
    roadGridStatic = eng.workspace['roadGrid']
    roadGrid = np.asarray(eng.workspace['roadGrid']).astype(int)
    roadGridPosition = torch.tensor(np.where(roadGrid == 20, roadGrid, 0), dtype=torch.float)
    roadGridTarget = torch.tensor(np.where(roadGrid == 30, roadGrid, 0), dtype=torch.float)
    roadGridRoads = torch.tensor(np.where(roadGrid == 10, roadGrid, 0), dtype=torch.float)
    countGrid = torch.tensor(np.asarray(eng.workspace['countGrid']), dtype=torch.float)
    scenario = eng.workspace['scenario']

    totalreward = 0

    loss = 0
    steps = 0
    running = True
    while running and steps < NUM_STEPS:
        steps += 1
        totalsteps += 1
        # action = choose_action(roadGrid, countGrid, eps+1).item()
        action,epsilon = choose_action(roadGridPosition, roadGridTarget, roadGridTarget, totalsteps)
        action = action.item()

        # action = steps % 4
        # action = 1
        (scenario, running, reward, nextRoadGrid, nextCountGrid, distance) = eng.simulate(scenario, eng.workspace['rewardValues'], eng.workspace['gridsize'], eng.workspace['goalGridPosition'], action+1, eng.workspace['egoVehicleSpeed'], eng.workspace['staticRoadGrid'], nargout=6)

        nextRoadGrid = np.asarray(nextRoadGrid).astype(int)
        nextRoadGridPosition = torch.tensor(np.where(nextRoadGrid == 20, 1, 0),dtype=torch.float)
        nextRoadGridTarget = torch.tensor(np.where(nextRoadGrid == 30, 1, 0), dtype=torch.float)
        nextRoadGridRoads = torch.tensor(np.where(nextRoadGrid == 10, 1, 0),dtype=torch.float)

        # print(nextRoadGrid)
        # print(nextRoadGridPosition)
        # print(nextRoadGridTarget)
        # print(nextRoadGridRoads)

        # nextRoadGrid = torch.tensor(np.asarray(nextRoadGrid),dtype=torch.float)
        nextCountGrid = torch.tensor(np.asarray(nextCountGrid),dtype=torch.float)

        totalreward = totalreward + reward

        if running == False:
            nextRoadGrid = None
            nextRoadGridPosition = None
            nextRoadGridRoads = None
            nextRoadGridTarget = None
            nextCountGrid = None

        # memory.push((roadGrid, countGrid), torch.tensor([action]), (nextRoadGrid, nextCountGrid), torch.tensor([reward]))
        memory.push((roadGridPosition, roadGridTarget, roadGridRoads), torch.tensor([action]), (nextRoadGridPosition, nextRoadGridTarget, nextRoadGridRoads), torch.tensor([reward]))
        loss = optimize_model()


        roadGridPosition = nextRoadGridPosition
        roadGridTarget = nextRoadGridTarget
        roadGridRoads = nextRoadGridRoads
        countGrid = nextCountGrid

        if steps % 100 == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(policy_net_state_dict)
            print(totalreward, loss, totalsteps, epsilon)

    print('Episode: ', eps, 'Steps: ', steps, 'Loss: ', loss, 'Total Reward: ', totalreward)
    eng.simulationsetup(nargout=0)
