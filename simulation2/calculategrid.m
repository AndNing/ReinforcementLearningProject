function calculategrid(scenario, roads)
    rmax = max(roads);
    rmin = min(roads);
    rdist = rmax - rmin;
    gridsize = rdist / 10;

    grid = zeros(gridsize(1),gridsize(2));

    for i=1:size(roads,1)/2
        roadStart = roads(2*i-1,1:2);
        roadEnd = roads(2*i,1:2);

        if (roadStart(2) - roadEnd(2)) == 0
            roadColumn = ((roadStart(2) - 5)/10) + 1;
            disp(roadColumn);
            grid(:,roadColumn) = 1;
        else
            roadRow = ((roadStart(1) - 5)/10) + 1.5;
            disp(roadRow);
            grid(roadRow,:) = 1;
        end
        disp(grid);

    end


end