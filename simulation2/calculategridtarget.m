function gridtarget = calculategridtarget(egoVehicleGridPosition,action, gridsize)
    nextVehicleGridPosition = zeros(2);
    if action == 1 %up
        nextVehicleGridPosition(1) = egoVehicleGridPosition(1) - 1;
        nextVehicleGridPosition(2) = egoVehicleGridPosition(2);
    elseif action == 2 %left
        nextVehicleGridPosition(1) = egoVehicleGridPosition(1);
        nextVehicleGridPosition(2) = egoVehicleGridPosition(2) - 1;
    elseif action == 3 %down
        nextVehicleGridPosition(1) = egoVehicleGridPosition(1) + 1;
        nextVehicleGridPosition(2) = egoVehicleGridPosition(2);
    else %right
        nextVehicleGridPosition(1) = egoVehicleGridPosition(1);
        nextVehicleGridPosition(2) = egoVehicleGridPosition(2) + 1;
    end


    flipgrid = zeros(gridsize(1),gridsize(2));


 
    flipgrid(nextVehicleGridPosition(1),nextVehicleGridPosition(2)) = 1;
 

    flippedgrid = flip(flip(flipgrid),2);

    [~,column] = max(max(flippedgrid,1));
    [~,row] = max(max(flippedgrid,2));

    trajectoryx = row * 10 - 5;
                trajectoryy = column * 10 - 5;

                gridtarget = [trajectoryx, trajectoryy];


