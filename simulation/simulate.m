function simulate(scenario, accMax, connector, display, intersectionBuffer, intersectionS, laneWidth, lidars, ptClouds, currentEgoState, tracker, collisionValidator, refPath, speedLimit, egoVehicle)
    temp = advance(scenario);

%     x = scenario.Actors.Position(1);
%     y = scenario.Actors.Position(2);


    time = scenario.SimulationTime;
%     disp(time);
    
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
