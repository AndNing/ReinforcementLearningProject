function reward = calculatereward(roadGrid, countGrid, egoVehicleGridPosition, rewardValues)
    if roadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) == 0
        reward = rewardValues.offroad;
    elseif countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) > 0
        reward = rewardValues.vehicle * countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2));
    else
        reward = rewardValues.time;
    end