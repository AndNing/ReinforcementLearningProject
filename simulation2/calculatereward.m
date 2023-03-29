function [reward] = calculatereward(newEgoVehiclePosition, roadGrid, countGrid, egoVehicleGridPosition, rewardValues, goalGridPosition_flipped, running, egoVehiclePosition)
%   egoVehicleGridPosition is not flipped
    temproadGrid = flip(flip(roadGrid),2);
    temproadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) = 20;
    temproadGrid = flip(flip(temproadGrid),2);

    [egoVehicleGridPositionX, egoVehicleGridPositionY] = find(temproadGrid==20);
    egoVehicleGridPosition_flipped = [egoVehicleGridPositionX, egoVehicleGridPositionY];

    distance = sum(abs(egoVehicleGridPosition_flipped - goalGridPosition_flipped));
    %reward = 300 - distance^2;
    gridsize = size(temproadGrid);
    distance =  distance / (gridsize(1) + gridsize(2) - 2);
    reward = -distance + rewardValues.time;


    if roadGrid(egoVehicleGridPositionX,egoVehicleGridPositionY) == 0
        reward = rewardValues.offroad + reward;
    end
    if  isequal(egoVehiclePosition, newEgoVehiclePosition)
          reward = reward +  rewardValues.boundary;

    end

    if countGrid(egoVehicleGridPositionX,egoVehicleGridPositionY) > 0
        reward = reward + rewardValues.vehicle * countGrid(egoVehicleGridPositionX,egoVehicleGridPositionY);
    end
    if running == false
        reward = rewardValues.finish;
    end

