function running = checktermination(egoVehicleGridPosition,goalGridPosition)
    if egoVehicleGridPosition == goalGridPosition
        running = false;
        disp(running);
    else
        running = true;
    end
