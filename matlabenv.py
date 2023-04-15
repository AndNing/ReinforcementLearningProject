import matlab.engine
import numpy as np
import random

class matlabenv:
    def __init__(self, num_of_grid, action_space, stopTime, gridx=13, gridy=13, seednum=1, gamma=0.99):
        """Initialization."""

        self.seednum = seednum
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('./simulation3', nargout=0)
        self.eng.workspace['seednum'] = self.seednum
        self.eng.setseed(nargout=0)
        self.stopTime = stopTime
        self.gamma = gamma
        self.roadGridRoads = np.zeros((gridx, gridy))
        self.roadGridTarget = np.zeros((gridx, gridy))
        self.roadGridPosition = np.zeros((gridx, gridy))
        self.scenario = None
        self.countGrid = np.zeros((gridx, gridy))
        self.done = False
        self.reward = 0
        self.state = None
        self.num_of_grid = num_of_grid
        self.observation_space = num_of_grid * gridx * gridy
        self.action_space = action_space
        self.step_count = 0
        self.termination = False
        self.eng.workspace['stopTime'] = self.stopTime
        self.eng.workspace['gamma'] = self.gamma


    def reset(self):
        self.eng.simulationsetup(nargout=0)
        self.seednum += 1
        self.eng.workspace['seednum'] = self.seednum
        self.eng.setseed(nargout=0)
        self.roadGridRoads = np.asarray(self.eng.workspace['roadGrid'])
        self.roadGridPosition = np.asarray(self.eng.workspace['positionGrid'])
        self.roadGridTarget = np.asarray(self.eng.workspace['goalGrid'])
        self.countGrid = np.asarray(self.eng.workspace['countGrid'])
        self.scenario = self.eng.workspace['scenario']
        self.state = np.concatenate((self.roadGridPosition.flatten(),
                                    self.roadGridTarget.flatten(),
                                    self.roadGridRoads.flatten(),
                                    self.countGrid.flatten()))
        self.termination = False
        self.step_count = 0
        self.done = False
        return self.state

    def step(self, action):
       (scenario, nextCountGrid, nextpositionGrid, done, reward,) = \
            self.eng.simulate(self.scenario,
                              action + 1,
                              self.eng.workspace['egoVehicleSpeed'],
                              self.eng.workspace['gridlength'],
                              self.eng.workspace['gridsize'],
                              self.eng.workspace['goalGridPosition'],
                              self.eng.workspace['staticRoadGrid'],
                              self.eng.workspace['rewardValues'],
                              self.eng.workspace['gamma'],
                              nargout=5)
       self.scenario = scenario
       self.reward = reward
       nextRoadGridPosition = np.asarray(nextpositionGrid)
       nextCountGrid = np.asarray(nextCountGrid)
       self.countGrid = nextCountGrid
       self.roadGridPosition = nextRoadGridPosition
       self.done = done

       next_state = np.concatenate((self.roadGridPosition.flatten(),
                                    self.roadGridTarget.flatten(),
                                    self.roadGridRoads.flatten(),
                                    self.countGrid.flatten()))
       self.step_count += 1
       if self.step_count == self.stopTime:
           self.termination = True

       return next_state, reward, self.done

    def close(self):
        self.eng.quit()
        print('matlab closed')


    def action_sample(self):
        return random.randrange(self.action_space)

