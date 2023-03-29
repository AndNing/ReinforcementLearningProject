import matlab.engine
import numpy as np


class matlabenv:
    def __init__(self, num_of_grid, action_space, gridx=13, gridy=13, seednum=1):
        """Initialization."""

        self.seednum = seednum
        self.eng = matlab.engine.start_matlab()
        self.eng.cd('./simulation2', nargout=0)
        self.eng.workspace['seednum'] = seednum
        self.eng.setseed(nargout=0)
        self.roadGridRoads = np.zeros((gridx, gridy))
        self.roadGridTarget = np.zeros((gridx, gridy))
        self.roadGridPosition = np.zeros((gridx, gridy))
        self.scenario = None
        self.countGrid = np.zeros((gridx, gridy))
        self.roadGrid = np.zeros((gridx, gridy))
        self.running = True
        self.done = False
        self.reward = 0
        self.state = None
        self.num_of_grid = num_of_grid
        self.observation_space = num_of_grid * gridx * gridy
        self.action_space = action_space


    def reset(self):
        self.eng.simulationsetup(nargout=0)
        staticRoadGrid = np.asarray(self.eng.workspace['staticRoadGrid'])
        staticRoadGrid = np.where(staticRoadGrid == 30, 10, staticRoadGrid)
        self.roadGridRoads = np.where(staticRoadGrid == 10, 1, 0)
        self.roadGrid = np.asarray(self.eng.workspace['updatedRoadGrid']).astype(int)
        self.roadGridPosition = np.where(self.roadGrid == 20, 1, 0)
        self.roadGridTarget = np.where(self.roadGrid == 30, 1, 0)
        self.countGrid = np.asarray(self.eng.workspace['countGrid'])
        self.scenario = self.eng.workspace['scenario']
        self.state = np.concatenate((self.roadGridPosition.flatten(),
                                    self.roadGridTarget.flatten(),
                                    self.roadGridRoads.flatten()))
        return self.state

    def step(self, action):
        (scenario, running, reward, nextRoadGrid, nextCountGrid) = \
            self.eng.simulate(self.scenario,
                              self.eng.workspace['rewardValues'],
                              self.eng.workspace['gridsize'],
                              self.eng.workspace['goalGridPosition'],
                              action + 1,
                              self.eng.workspace['egoVehicleSpeed'],
                              self.eng.workspace['staticRoadGrid'],
                              nargout=5)
        self.scenario = scenario
        self.running = running
        self.reward = reward
        nextRoadGrid = np.asarray(nextRoadGrid).astype(int)
        nextRoadGridPosition = np.where(nextRoadGrid == 20, 1, 0)
        nextRoadGridTarget = np.where(nextRoadGrid == 30, 1, 0)
        nextCountGrid = np.asarray(nextCountGrid)
        self.countGrid = nextCountGrid
        self.roadGrid = nextRoadGrid
        self.roadGridTarget = nextRoadGridTarget
        self.roadGridPosition = nextRoadGridPosition
        self.done = not running
        if self.done == True:
            next_state = None
        else:
            next_state = np.concatenate((self.roadGridPosition.flatten(),
                                        self.roadGridTarget.flatten(),
                                        self.roadGridRoads.flatten()))

        return next_state, reward, self.done

    def close(self):
        self.eng.quit()
        print('matlab closed')

