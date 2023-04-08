function terminated = checktermination(egoVehicleGridPosition,goalGridPosition)
    if egoVehicleGridPosition == goalGridPosition
        terminated = true;
    else
        terminated = false;
    end