function [countGrid,roadGrid] = calculategrid(scenario, roadGrid, gridsize, egoVehicleGridPosition)
    roadGrid = flip(flip(roadGrid),2);
    roadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) = 20;
    roadGrid = flip(flip(roadGrid),2);


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
        if actorGridPosition(1) <= gridsize(1) && actorGridPosition(2) <= gridsize(2) %%added
        countGrid(actorGridPosition(1),actorGridPosition(2)) = countGrid(actorGridPosition(1),actorGridPosition(2)) + 1;
        end
    end
    countGrid = flip(flip(countGrid),2);
end