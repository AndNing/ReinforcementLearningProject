function [scenario, running, reward, roadGrid, countGrid] = simulate(scenario, roadGrid, rewardValues, gridsize, goalGridPosition, action, egoVehicleSpeed)
    egoVehiclePosition = scenario.Actors(1).Position;
    egoVehicleGridPosition = ceil(egoVehiclePosition/10);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);

    % gridtarget = calculategridtarget(egoVehicleGridPosition, action, gridsize);
    newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, scenario.SampleTime, egoVehicleSpeed, action);
    scenario.Actors(1).Position = newEgoVehiclePosition;

    egoVehicleGridPosition = ceil(newEgoVehiclePosition/10);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);

    temp = advance(scenario);

    [countGrid,roadGrid] = calculategrid(scenario, roadGrid, gridsize, egoVehicleGridPosition);
    running = checktermination(egoVehicleGridPosition,goalGridPosition);
    reward = calculatereward(roadGrid, countGrid, egoVehicleGridPosition, rewardValues);





