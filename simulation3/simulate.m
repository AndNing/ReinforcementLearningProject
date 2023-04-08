function [scenario, countGrid, positionGrid, roadGrid, goalGrid, termination, reward] = simulate(scenario, action, egoVehicleSpeed, gridlength, gridsize, goalGridPosition, staticGoalGrid, staticRoadGrid, rewardValues)
    egoVehiclePosition = scenario.Actors(1).Position;

    newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, scenario.SampleTime, egoVehicleSpeed, action, gridlength, gridsize);
    scenario.Actors(1).Position = newEgoVehiclePosition;
    egoVehicleGridPosition = ceil(newEgoVehiclePosition/gridlength);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);

    temp = advance(scenario);

    positionGrid = calculatepositiongrid(egoVehicleGridPosition, gridsize);
    countGrid = calculatecountgrid(scenario, gridsize, gridlength);

    termination = checktermination(egoVehicleGridPosition,goalGridPosition);
    reward = calculatereward(egoVehicleGridPosition, egoVehiclePosition, newEgoVehiclePosition, goalGridPosition,gridsize, rewardValues, staticRoadGrid, countGrid,termination);

    % Flip grids at end
    goalGrid = flip(flip(staticGoalGrid),2);
    roadGrid = flip(flip(staticRoadGrid),2);
    countGrid = flip(flip(countGrid),2);
    positionGrid = flip(flip(positionGrid),2);