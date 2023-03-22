function newEgoVehiclePosition = egovehicleupdate(egoVehiclePosition, sampleTime, egoVehicleSpeed, action)
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
    
    %added
    if newEgoVehiclePosition(1) < 0 || newEgoVehiclePosition(1) > 130
        newEgoVehiclePosition = egoVehiclePosition;
    end

    if newEgoVehiclePosition(2) < 0 || newEgoVehiclePosition(2) > 130
        newEgoVehiclePosition = egoVehiclePosition;
    end

    end
    




    
    
    