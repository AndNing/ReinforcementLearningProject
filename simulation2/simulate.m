function [running, reward, roadGrid] = simulate(scenario, roadGrid, rewardValues, gridsize, goalGridPosition)
    temp = advance(scenario);
    [countGrid,roadGrid, egoVehicleGridPosition] = calculategrid(scenario, roadGrid, gridsize);
    running = checktermination(egoVehicleGridPosition,goalGridPosition);
    reward = calculatereward(roadGrid, countGrid, egoVehicleGridPosition, rewardValues);





