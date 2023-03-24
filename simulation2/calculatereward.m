function [reward,distance] = calculatereward(newEgoVehiclePosition, roadGrid, countGrid, egoVehicleGridPosition, rewardValues, goalGridPosition, running)
    temproadGrid = flip(flip(roadGrid),2);
    temproadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) = 2;
    temproadGrid = flip(flip(temproadGrid),2);
%     disp(temproadGrid)
    [egoVehicleGridPositionX, egoVehicleGridPositionY] = find(temproadGrid==2);
%     disp(egoVehicleGridPositionX)
%     disp(egoVehicleGridPositionY)


    distance = norm(newEgoVehiclePosition(1:2) - (goalGridPosition*10 - 5));
    reward = -distance;
    reward = reward + rewardValues.time;
%     disp(roadGrid)
%     disp(egoVehicleGridPosition)
%     disp(countGrid)
    if roadGrid(egoVehicleGridPositionX,egoVehicleGridPositionY) == 0
        reward = reward + rewardValues.offroad;
    end
    if egoVehicleGridPositionX > 13 || egoVehicleGridPositionX <= 0
        reward = reward + rewardValues.boundary;
    end
    if egoVehicleGridPositionY > 13 || egoVehicleGridPositionY <= 0
        reward = reward + rewardValues.boundary;
    end
    if countGrid(egoVehicleGridPositionX,egoVehicleGridPositionY) > 0
        reward = reward + rewardValues.vehicle * countGrid(egoVehicleGridPositionX,egoVehicleGridPositionY);
    end
    if running == false
        reward = rewardValues.finish;
    end

