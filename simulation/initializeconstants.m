% Simulation parameters
sampleTime = 1;
numSteps = 150;

% Size of grid cells (m)
gridlength = 10;

% Internal identifiers for state
roadGridIdentifier = 10;
goalGridIdentifier = 30;
egoVehicleGridIdentifier = 20;

% Agent vehicle parameters
egoVehicleSpeed = 10;
egoVehicleRandomDistance = 10;                  % Distance between start and end position
egoVehicleStartROI = [0 130; 10 0];             % Start region
egoVehicleEndROI = [130 130; 110 0];            % End region

% Non-agent vehicle parameters
actorVehicleMinSpeed = 1;
actorVehicleMaxSpeed = 5;
numActorVehicles = 10;
actorvehicleStartingDistributionDeviation = 15; % Standard deviation of normal distribution for trajectory creation
actorGenerationFrequency = 10;                  % Frequency (in steps) of non-agent vehicle generation
numActorsInitial = 20;                          % Initial number of non-agent vehicles to start with

% Reward values
rewardValues.offroad = -1;
rewardValues.time = -0.05;
rewardValues.vehicle = -0.1;
rewardValues.finish = 10;
rewardValues.boundary = -1;

% Plotting
doPlots = false;