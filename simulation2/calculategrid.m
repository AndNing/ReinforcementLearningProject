function [roadGrid, countGrid] = calculategrid(scenario, roads)
    rmax = max(roads);
    rmin = min(roads);
    rdist = rmax - rmin;
    gridsize = rdist / 10;

    roadGrid = zeros(gridsize(1),gridsize(2));

    for i=1:size(roads,1)/2
        roadStart = roads(2*i-1,1:2);
        roadEnd = roads(2*i,1:2);

        if (roadStart(2) - roadEnd(2)) == 0
            roadColumn = ((roadStart(2) - 5)/10) + 1;
            roadGrid(:,roadColumn) = 1;
        else
            roadRow = ((roadStart(1) - 5)/10) + 1;
%             disp(roadRow);
            roadGrid(roadRow,:) = 1;
        end
    end
    roadGrid = flip(flip(roadGrid),2);

    egoVehiclePosition = scenario.Actors(1).Position;
    egoVehicleGridPosition = ceil(egoVehiclePosition/10);
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