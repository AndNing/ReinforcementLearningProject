function [scenario, running, reward, updatedRoadGrid, countGrid] = simulate(scenario, rewardValues, gridsize, goalGridPosition, action, egoVehicleSpeed, staticRoadGrid)
    egoVehiclePosition = scenario.Actors(1).Position;


    % gridtarget = calculategridtarget(egoVehicleGridPosition, action, gridsize);
    newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, scenario.SampleTime, egoVehicleSpeed, action);
    scenario.Actors(1).Position = newEgoVehiclePosition;

    egoVehicleGridPosition = ceil(newEgoVehiclePosition/10);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);

    [goalGridPositionX, goalGridPositionY] = find(staticRoadGrid==30);
    goalGridPosition_flipped = [goalGridPositionX, goalGridPositionY];
    temp = advance(scenario);

    [countGrid,updatedRoadGrid] = calculategrid(scenario, staticRoadGrid, gridsize, egoVehicleGridPosition);

    running = checktermination(egoVehicleGridPosition,goalGridPosition);
    reward = calculatereward(newEgoVehiclePosition,staticRoadGrid, countGrid, egoVehicleGridPosition, rewardValues, goalGridPosition_flipped, running, egoVehiclePosition);






