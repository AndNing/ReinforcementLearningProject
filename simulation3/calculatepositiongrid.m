function positionGrid = calculatepositiongrid(egoVehicleGridPosition, gridsize)
    positionGrid = zeros(gridsize(1),gridsize(2));
    positionGrid(egoVehicleGridPosition(1), egoVehicleGridPosition(2)) = 1;
end