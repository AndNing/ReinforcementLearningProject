function [scenario, countGrid, positionGrid, done, reward] = simulate(scenario, action, egoVehicleSpeed, gridlength, gridsize, goalGridPosition, staticRoadGrid, rewardValues, gamma)
    egoVehiclePosition = scenario.Actors(1).Position;
    pastegoVehicleGridPosition = ceil(egoVehiclePosition/gridlength);
    pastegoVehicleGridPosition = pastegoVehicleGridPosition(1:2);


    newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, scenario.SampleTime, egoVehicleSpeed, action, gridlength, gridsize);
    scenario.Actors(1).Position = newEgoVehiclePosition;
    egoVehicleGridPosition = ceil(newEgoVehiclePosition/gridlength);
    egoVehicleGridPosition = egoVehicleGridPosition(1:2);

    temp = advance(scenario);

    positionGrid = calculatepositiongrid(egoVehicleGridPosition, gridsize);
    countGrid = calculatecountgrid(scenario, gridsize, gridlength);

    done = checkdone(egoVehicleGridPosition,goalGridPosition);
    reward = calculatereward(pastegoVehicleGridPosition, egoVehiclePosition, egoVehicleGridPosition, newEgoVehiclePosition, goalGridPosition,gridsize, rewardValues, staticRoadGrid, countGrid,gamma,done);

    % Flip grids at end
    countGrid = flip(flip(countGrid),2);
    positionGrid = flip(flip(positionGrid),2);
end


