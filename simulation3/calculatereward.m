function reward = calculatereward(egoVehicleGridPosition, egoVehiclePosition, newEgoVehiclePosition, goalGridPosition,gridsize, rewardValues, roadGrid, countGrid,termination)
    distance = sum(abs(egoVehicleGridPosition - goalGridPosition));
    distance = distance / (gridsize(1) + gridsize(2) - 2);
    reward = -distance + rewardValues.time;

    if roadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) == 0
        reward = rewardValues.offroad + reward;
    end
    if  isequal(egoVehiclePosition, newEgoVehiclePosition)
          reward = reward +  rewardValues.boundary;
    end

    if countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) > 0
        reward = reward + rewardValues.vehicle * countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2));
    end
    if termination == true
        reward = rewardValues.finish;
    end
end
