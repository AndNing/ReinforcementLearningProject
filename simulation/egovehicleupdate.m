% Calculate new position for agent vehicle based on action
function newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, sampleTime, egoVehicleSpeed, action, gridlength, gridsize)
    distance = sampleTime * egoVehicleSpeed;

    newEgoVehiclePosition = egoVehiclePosition;

    if action == 1 %up
        newEgoVehiclePosition(1) = egoVehiclePosition(1) + distance;
    elseif action == 2 %left
        newEgoVehiclePosition(2) = egoVehiclePosition(2) + distance;
    elseif action == 3 %down
        newEgoVehiclePosition(1) = egoVehiclePosition(1) - distance;
    else %right
        newEgoVehiclePosition(2) = egoVehiclePosition(2) - distance;
    end

    if newEgoVehiclePosition(1) <= 0 || newEgoVehiclePosition(1) > (gridsize(1)*gridlength)
        newEgoVehiclePosition(1) = egoVehiclePosition(1);
    end

    if newEgoVehiclePosition(2) <= 0 || newEgoVehiclePosition(2) > (gridsize(2)*gridlength)
        newEgoVehiclePosition(2) = egoVehiclePosition(2);
    end

    end
    




    
    
    