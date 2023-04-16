function done = checkdone(egoVehicleGridPosition,goalGridPosition)
    if egoVehicleGridPosition == goalGridPosition
        done = true;
    else
        done = false;
    end