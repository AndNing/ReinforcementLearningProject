import matlab.engine
import numpy as np
import random

"""
This class creates environment using Autonomous Driving Toolbox in MATLAB
"""

class matlabenv:
    def __init__(self, num_of_grid, action_space, stopTime, gridx=13, gridy=13, seednum=1, gamma=0.99):
        """Initialization."""

        self.seednum = seednum  #seed number
        self.eng = matlab.engine.start_matlab()  ##start MATLAB engine
        self.eng.cd('./simulation', nargout=0)  ##Path for the MATLAB codes
        self.eng.workspace['seednum'] = self.seednum  
        self.eng.setseed(nargout=0)   ##Set seed number in MATLAB
        self.stopTime = stopTime  ## Maximum time step per episode
        self.gamma = gamma  ## discount rate
        self.roadGridRoads = np.zeros((gridx, gridy))  ## grid for road topology road=1 none-road=0
        self.roadGridTarget = np.zeros((gridx, gridy))  ## grid for the goal position goal=1 others=0
        self.roadGridPosition = np.zeros((gridx, gridy))  ## grid for the current position of agent. current position =1 others=0
        self.scenario = None
        self.countGrid = np.zeros((gridx, gridy))  ## groid for the number of other vehicles
        self.done = False
        self.reward = 0
        self.state = None
        self.num_of_grid = num_of_grid
        self.observation_space = num_of_grid * gridx * gridy  ## size of state space
        self.action_space = action_space  ## size of action space
        self.step_count = 0
        self.termination = False
        self.eng.workspace['stopTime'] = self.stopTime
        self.eng.workspace['gamma'] = self.gamma

    ## reset function for intializing the first state
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
                                    self.countGrid.flatten()))    ## state is represented as concatenation of vectorized grids
        self.termination = False
        self.step_count = 0
        self.done = False
        return self.state
    
    ## step function gives the next state, reward, and the status of episode (done or not)
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
    
    ## close the MATLAB engine
    def close(self):
        self.eng.quit()
        print('matlab closed')

    ## sampling action for the epsilon-greedy exploration
    def action_sample(self):
        return random.randrange(self.action_space)

