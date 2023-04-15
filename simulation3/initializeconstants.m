sampleTime = 1;
numSteps = 150;

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
numActorVehicles = 10;
actorvehicleStartingDistributionDeviation = 15;
actorGenerationFrequency = 10;
numActorsInitial = 20;

rewardValues.offroad = -1;
rewardValues.time = -0.05;
rewardValues.vehicle = -0.1;
rewardValues.finish = 10;
rewardValues.boundary = -1;

doPlots = false;