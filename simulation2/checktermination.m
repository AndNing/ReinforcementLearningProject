function termination = checktermination(egoVehicleGridPosition,goalGridPosition)
    if egoVehicleGridPosition == goalGridPosition
        termination = true;
    else
        termination = false;
    end
