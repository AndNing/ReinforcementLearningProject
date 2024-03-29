% Grid creation for non-agent vehicles
function countGrid = calculatecountgrid(scenario, gridsize, gridlength)
    countGrid = zeros(gridsize(1),gridsize(2));

    numActors = size(scenario.Actors,2);

    for i=2:numActors
        actorPosition = scenario.Actors(i).Position;
        actorGridPosition = ceil(actorPosition/gridlength);

        if actorGridPosition(1) < 1 || actorGridPosition(1) > gridsize(1)
            continue
        end

        if actorGridPosition(2) < 1 || actorGridPosition(2) > gridsize(2)
           continue
        end

        countGrid(actorGridPosition(1),actorGridPosition(2)) = countGrid(actorGridPosition(1),actorGridPosition(2)) + 1;

    end
end