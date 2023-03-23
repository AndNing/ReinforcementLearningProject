function [scenario, running, reward, updatedRoadGrid, countGrid, distance] = simulate(scenario, rewardValues, gridsize, goalGridPosition, action, egoVehicleSpeed, staticRoadGrid)
    egoVehiclePosition = scenario.Actors(1).Position;

    % gridtarget = calculategridtarget(egoVehicleGridPosition, action, gridsize);
    newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, scenario.SampleTime, egoVehicleSpeed, action);
    scenario.Actors(1).Position = newEgoVehiclePosition;

    egoVehicleGridPosition = ceil(newEgoVehiclePosition/10);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);
%     disp(egoVehicleGridPosition)

    temp = advance(scenario);

    [countGrid,updatedRoadGrid] = calculategrid(scenario, staticRoadGrid, gridsize, egoVehicleGridPosition);
    running = checktermination(egoVehicleGridPosition,goalGridPosition);
    [reward,distance] = calculatereward(newEgoVehiclePosition,staticRoadGrid, countGrid, egoVehicleGridPosition, rewardValues, goalGridPosition, running);





