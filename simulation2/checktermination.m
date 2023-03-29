function running = checktermination(egoVehicleGridPosition,goalGridPosition)
    if egoVehicleGridPosition == goalGridPosition
        running = false;
    else
        running = true;
    end
