% seednum = 1;

sampleTime = 1;
stopTime = 100000;
numSteps = 600;

gridlength = 10;

roadGridIdentifier = 10;
goalGridIdentifier = 30;
egoVehicleGridIdentifier = 20;

egoVehicleSpeed = 10;
egoVehicleRandomDistance = 10;
egoVehicleStartROI = [0 130; 10 0];
egoVehicleEndROI = [130 130; 110 0];

actorVehicleMinSpeed = 1;
actorVehicleMaxSpeed = 5;
numActorVehicles = 40;
actorvehicleStartingDistributionDeviation = 15;
actorGenerationFrequency = 5;
numActorsInitial = 20;

rewardValues.offroad = -10;
rewardValues.time = -0.1;
rewardValues.vehicle = -0.05;
rewardValues.finish = 10;
rewardValues.boundary = -10;

doPlots = true;