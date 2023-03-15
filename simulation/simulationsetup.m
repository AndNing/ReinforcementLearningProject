[scenario, egoVehicle] = createDrivingScenario;

lidars = createSensors(scenario);

% Set up sensor configurations for each lidar
sensorConfigs = cell(numel(lidars),1);

% Fill in sensor configurations
for i = 1:numel(sensorConfigs)
    sensorConfigs{i} = helperGetLidarConfig(lidars{i},egoVehicle);
end

% Set up tracker
tracker = trackerGridRFS('SensorConfigurations',sensorConfigs,...
    'HasSensorConfigurationsInput',true,...
    'GridLength',120,...
    'GridWidth',120,...
    'GridResolution',2,...
    'GridOriginInLocal',[-60 -60],...
    'NumParticles',1e5,...
    'NumBirthParticles',2e4,...
    'VelocityLimits',[-15 15;-15 15],...
    'BirthProbability',0.025,...
    'ProcessNoise',5*eye(2),...
    'DeathRate',1e-3,...
    'FreeSpaceDiscountFactor',1e-2,...
    'AssignmentThreshold',8,...
    'MinNumCellsPerCluster',4,...
    'ClusteringThreshold',4,...
    'ConfirmationThreshold',[3 4],...
    'DeletionThreshold',[4 4]);

waypoints = [-40 21.4; 40 21.4];

refPath = referencePathFrenet(waypoints);
fig = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
ax = axes(fig);
hold(ax,'on');
plot(scenario,'Parent',ax);
show(refPath,'Parent',ax);
xlim(ax,[-120 80]);
ylim(ax,[-160 40]);
snapnow;

connector = trajectoryGeneratorFrenet(refPath,'TimeResolution',0.1);

% Visualize path regions for sampling strategy visualization
pathPoints = closestPoint(refPath, refPath.Waypoints(:,1:2));
roadS = pathPoints(:,end);
intersectionS = roadS(2,end);
intersectionBuffer = 20;
pathGreen = [interpolate(refPath,linspace(0,intersectionS-intersectionBuffer,20));...
            nan(1,6);...
            interpolate(refPath,linspace(intersectionS,roadS(end),100))];
pathBlue = interpolate(refPath,linspace(intersectionS-intersectionBuffer,roadS(2,end),20));
hold(ax,'on');
plot(ax,pathGreen(:,1),pathGreen(:,2),'Color',[0 1 0],'LineWidth',5);
plot(ax,pathBlue(:,1),pathBlue(:,2),'Color',[0 0 1],'LineWidth',5);
snapnow;

speedLimit = 3;
laneWidth = 2.975;

accMax = 3;

vehDims = vehicleDimensions(egoVehicle.Length,egoVehicle.Width);
collisionValidator = HelperDynamicMapValidator('MaxTimeHorizon',2, ... % Maximum horizon for validation
    'TimeResolution',connector.TimeResolution, ... % Time steps between trajectory samples
    'Tracker',tracker, ... % Provide tracker for prediction
    'ValidPredictionSpan',5, ... % Prediction valid for 5 steps
    'VehicleDimensions',vehDims); % Provide dimensions of ego


display = helperGridBasedPlanningDisplay;

% Initial ego state
currentEgoState = [-55 21.4 0 0 2 0];
helperMoveEgoVehicleToState(egoVehicle, currentEgoState);

% Initialize pointCloud outputs from each sensor
ptClouds = cell(numel(lidars),1);
sensorConfigs = cell(numel(lidars),1);


% while true
%    simulate(scenario, accMax, connector, display, intersectionBuffer, intersectionS, laneWidth, lidars, ptClouds, currentEgoState, tracker, collisionValidator, refPath, speedLimit, egoVehicle);
% end

count = 0;

while advance(scenario)
    % Current simulation time
    time = scenario.SimulationTime;

    count = count + 1;
    if count > 20
        waypoints = [40 -50; 40 -60;];
        refPath = referencePathFrenet(waypoints);
        connector = trajectoryGeneratorFrenet(refPath,'TimeResolution',0.1);
    end

    
    % Poses of objects with respect to ego vehicle
    tgtPoses = targetPoses(egoVehicle);
    
    % Simulate point cloud from each sensor
    for i = 1:numel(lidars)
        [ptClouds{i}, isValidTime] = step(lidars{i},tgtPoses,time);
        sensorConfigs{i} = helperGetLidarConfig(lidars{i},egoVehicle);
    end
    
    % Pack point clouds as sensor data format required by the tracker
    sensorData = packAsSensorData(ptClouds,sensorConfigs,time);
    
    % Call the tracker
    [tracks, ~, ~, map] = tracker(sensorData,sensorConfigs,time);
    
    % Update validator's future predictions using current estimate
    step(collisionValidator, currentEgoState, map, time);
    
    % Sample trajectories using current ego state and some kinematic
    % parameters
    [frenetTrajectories, globalTrajectories] = helperGenerateTrajectory(connector, refPath, currentEgoState, speedLimit, laneWidth, intersectionS, intersectionBuffer);
    
    % Calculate kinematic feasibility of generated trajectories
    isKinematicsFeasible = helperKinematicFeasibility(frenetTrajectories,speedLimit,accMax);
    
    % Calculate collision validity of feasible trajectories
    feasibleGlobalTrajectories = globalTrajectories(isKinematicsFeasible);
    feasibleFrenetTrajectories = frenetTrajectories(isKinematicsFeasible);
    [isCollisionFree, collisionProb] = isTrajectoryValid(collisionValidator, feasibleGlobalTrajectories);
    
    % Calculate costs and final optimal trajectory
    nonCollidingGlobalTrajectories = feasibleGlobalTrajectories(isCollisionFree);
    nonCollidingFrenetTrajectories = feasibleFrenetTrajectories(isCollisionFree);
    nonCollodingCollisionProb = collisionProb(isCollisionFree);
    costs = helperCalculateTrajectoryCosts(nonCollidingFrenetTrajectories, nonCollodingCollisionProb, speedLimit);
    
    % Find optimal trajectory
    [~,idx] = min(costs);
    optimalTrajectory = nonCollidingGlobalTrajectories(idx);
    
    % Assemble for plotting
    trajectories = helperAssembleTrajectoryForPlotting(globalTrajectories, ...
        isKinematicsFeasible, isCollisionFree, idx);
    
    % Update display
    display(scenario, egoVehicle, lidars, ptClouds, tracker, tracks, trajectories, collisionValidator);
    
    % Move ego with optimal trajectory
    if ~isempty(optimalTrajectory)
        currentEgoState = optimalTrajectory.Trajectory(2,:);
        helperMoveEgoVehicleToState(egoVehicle, currentEgoState);
    else
        % All trajectories either violated kinematic feasibility
        % constraints or resulted in a collision. More behaviors on
        % trajectory sampling may be needed.
        error('Unable to compute optimal trajectory');
    end
end