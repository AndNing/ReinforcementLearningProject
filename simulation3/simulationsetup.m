close all

% rng shuffle
initializeconstants

[scenario, roads, gridsize] = scenariosetup(sampleTime, stopTime);

pdEgoVehicleTopBot = makedist('Uniform','lower',0,'upper',gridsize(2)*gridlength);
pdEgoVehicleLeftRight = makedist('Uniform','lower',0,'upper',gridsize(1)*gridlength);

% Left
if randi(4) == 1
    egoX = random(pdEgoVehicleLeftRight);
    egoVehicleStartPosition = [egoX,125,0];

    egoX = random(pdEgoVehicleLeftRight);
    egoVehicleEndPosition = [egoX,5,0];
% Right
elseif randi(4) == 2
    egoX = random(pdEgoVehicleLeftRight);
    egoVehicleStartPosition = [egoX,5,0];

    egoX = random(pdEgoVehicleLeftRight);
    egoVehicleEndPosition = [egoX,125,0];
% Top
elseif randi(4) == 3
    egoY = random(pdEgoVehicleTopBot);
    egoVehicleStartPosition = [125,egoY,0];

    egoY = random(pdEgoVehicleTopBot);
    egoVehicleEndPosition = [5,egoY,0];
% Bot
else
    egoY = random(pdEgoVehicleTopBot);
    egoVehicleStartPosition = [5,egoY,0];

    egoY = random(pdEgoVehicleTopBot);
    egoVehicleEndPosition = [125,egoY,0];
end

disp('Ego Vehicle Starting Position')
disp(egoVehicleStartPosition)
disp('Ego Vehicle Goal Position')
disp(egoVehicleEndPosition)

egoVehicle = vehicle(scenario, ...
'ClassID', 1, ...
'Position', egoVehicleStartPosition, ...
'Name', 'Ego Vehicle');

if doPlots
    figScene = figure(Name="AutomaticScenarioGeneration");
    set(figScene,Position=[0,0,900,500]);
    hPanel1 = uipanel(figScene,Position=[0 0 0.5 1]);
    hPlot1 = axes(hPanel1);
    plot(scenario,Parent=hPlot1,Meshes="on");
    title("Selecting Ego Vehicle Start Positions")
    hold on
    plot(egoVehicleStartPosition(:,1),egoVehicleStartPosition(:,2),"bo",MarkerSize=5,MarkerFaceColor="b");
    plot(egoVehicleEndPosition(:,1),egoVehicleEndPosition(:,2),"bo",MarkerSize=5,MarkerFaceColor="g");
    hold off
end


actorVehicleStartingDistributionMeanTopBot = gridsize(1)*gridlength/2;
pdTopBot = makedist('Normal','mu',actorVehicleStartingDistributionMeanTopBot,'sigma',actorvehicleStartingDistributionDeviation);
pdtTopBot = truncate(pdTopBot,0,gridsize(1)*gridlength);

actorVehicleStartingDistributionMeanLeftRight = gridsize(2)*gridlength/2;
pdLeftRight = makedist('Normal','mu',actorVehicleStartingDistributionMeanLeftRight,'sigma',actorvehicleStartingDistributionDeviation);
pdtLeftRight = truncate(pdLeftRight,0,gridsize(2)*gridlength);

figure
x = linspace(-20,140,1000);
plot(x,pdf(pdtTopBot,x))
hold on
plot(x,pdf(pdtLeftRight,x))

startposs = [];
endposs = [];

steps = 0;
numerrs = 0;

for t = 1:numActorsInitial
    rng shuffle
    steps = steps + 1;
    startDirection = mod(steps,4)+1;

    % Left -> right
    if startDirection == 1
        startpos = [random(pdtLeftRight),125,0];
        endpos = [random(pdtLeftRight),5,0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];

    % Right -> left
    elseif startDirection == 2
        startpos = [random(pdtLeftRight),5,0];
        endpos = [random(pdtLeftRight),125,0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    % Top -> bot
    elseif startDirection == 3
        startpos = [125, random(pdtTopBot),0];
        endpos = [5, random(pdtTopBot),0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    % Bot -> top
    else
        startpos = [5, random(pdtTopBot),0];
        endpos = [125, random(pdtTopBot),0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    end

    try
        info = helperGenerateWaypoints(scenario,startpos,endpos);
        actorVehicle = actor(scenario,'ClassID',1, ...
        'Position',startpos,'EntryTime',t,'ExitTime',t+100);
        speed = randi([actorVehicleMinSpeed,actorVehicleMaxSpeed],1,1);
        waypts = info(1).waypoints;
        trajectory(actorVehicle,waypts,speed);
    catch err
        numerrs = numerrs + 1;
    end
end

steps = 0;
numerrs = 0;

for t = 1:actorGenerationFrequency:numSteps
    rng shuffle
    steps = steps + 1;
    startDirection = mod(steps,4)+1;

    % Left -> right
    if startDirection == 1
        startpos = [random(pdtLeftRight),125,0];
        endpos = [random(pdtLeftRight),5,0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];

    % Right -> left
    elseif startDirection == 2
        startpos = [random(pdtLeftRight),5,0];
        endpos = [random(pdtLeftRight),125,0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    % Top -> bot
    elseif startDirection == 3
        startpos = [125, random(pdtTopBot),0];
        endpos = [5, random(pdtTopBot),0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    % Bot -> top
    else
        startpos = [5, random(pdtTopBot),0];
        endpos = [125, random(pdtTopBot),0];
        startposs = [startposs; startpos];
        endposs = [endposs; endpos];
    end

    try
        info = helperGenerateWaypoints(scenario,startpos,endpos);
        actorVehicle = actor(scenario,'ClassID',1, ...
        'Position',startpos,'EntryTime',t,'ExitTime',t+100);
        speed = randi([actorVehicleMinSpeed,actorVehicleMaxSpeed],1,1);
        waypts = info(1).waypoints;
        trajectory(actorVehicle,waypts,speed);
    catch err
        numerrs = numerrs + 1;
    end
end

disp(['Number of failed trajectory generation tries: ',num2str(numerrs)])

if doPlots
    figScene = figure(Name="Actor Vehicle Generation");
    set(figScene,Position=[0,0,900,500]);
    hPanel1 = uipanel(figScene,Position=[0 0 0.5 1]);
    hPlot1 = axes(hPanel1);
    plot(scenario,Parent=hPlot1,Meshes="on");
    title("Simulation")
    hold on
    plot(startposs(:,1),startposs(:,2),"bo",MarkerSize=5,MarkerFaceColor="b");
    plot(endposs(:,1),endposs(:,2),"go",MarkerSize=5,MarkerFaceColor="g");
    hold off
end

goalGridPosition = ceil(egoVehicleEndPosition/gridlength);
goalGridPosition = goalGridPosition(1:2);

staticRoadGrid = zeros(gridsize(1),gridsize(2));
for i=1:size(roads,1)/2
    roadStart = roads(2*i-1,1:2);
    roadEnd = roads(2*i,1:2);

    if (roadStart(2) - roadEnd(2)) == 0
        roadColumn = ceil((roadStart(2) - 5)/gridlength) + 1;
        staticRoadGrid(:,roadColumn) = 1;
    else
        roadRow = ceil((roadStart(1) - 5)/gridlength) + 1;
        staticRoadGrid(roadRow,:) = 1;
    end
end

% For initial state
staticRoadGridInitial = flip(flip(staticRoadGrid,2));

% disp(staticRoadGrid)

staticGoalGrid = zeros(gridsize(1),gridsize(2));
staticGoalGrid(goalGridPosition(1),goalGridPosition(2)) = 1;
staticGoalGridInitial = flip(flip(staticGoalGrid),2);

positionGridInitial = zeros(gridsize(1),gridsize(2));
egoVehicleInitialGridPosition = ceil(egoVehicleStartPosition/gridlength);
egoVehicleInitialGridPosition = egoVehicleInitialGridPosition(1:2);
positionGridInitial(egoVehicleInitialGridPosition(1),egoVehicleInitialGridPosition(2)) = 1;
positionGridInitial = flip(flip(positionGridInitial),2);

countGridInitial = calculatecountgrid(scenario,gridsize,gridlength);
countGridInitial = flip(flip(countGridInitial),2);

% Comment this stuff out unless testing
action = 1;
while true
    [scenario, countGrid, positionGrid, roadGrid, goalGrid, termination, reward] = simulate(scenario, action, egoVehicleSpeed, gridlength, gridsize, goalGridPosition, staticGoalGrid, staticRoadGrid, rewardValues);
    pause(0.1);
end