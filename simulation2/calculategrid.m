function [countGrid,roadGrid] = calculategrid(scenario, roadGrid, gridsize, egoVehicleGridPosition)

    roadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) = 2;


    numActors = size(scenario.Actors,2)-1;
%     actorGridPositions = zeros(gridsize(1), gridsize(2),2);
%     for i=2:numActors+1
%         actorPosition = scenario.Actors(i).Position;
%         actorGridPosition = ceil(actorPosition/10);
%         actorGridPositions(actorGridPosition,1) = actorGridPositions(actorGridPosition,1) + 1;
%     end

    countGrid = zeros(gridsize(1),gridsize(2));
    for i=2:numActors+1
        actorPosition = scenario.Actors(i).Position;
        actorGridPosition = ceil(actorPosition/10);
        countGrid(actorGridPosition(1),actorGridPosition(2)) = countGrid(actorGridPosition(1),actorGridPosition(2)) + 1;
    end
    countGrid = flip(flip(countGrid),2);
end