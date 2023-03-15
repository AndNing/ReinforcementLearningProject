function [ds, tout] = helpergetCFSMScenario(out, currScenario)

[vehicles, tout] = createVehicleStruct(out);
sampleTime = currScenario.SampleTime;
ds = drivingScenario;
ds.SampleTime = 0.1;
ds.StopTime = currScenario.StopTime;
ds.VerticalAxis = 'Y';
copyMap(ds,currScenario)
for i = 1 : numel(vehicles)
    oldActor = currScenario.Actors(i);
    waypoints = vehicles(i).Position;
    
    n = size(waypoints,1);
    
    % find the course angles at each waypoint
    course = NaN(n,1);
    course = matlabshared.tracking.internal.scenario.clothoidG2fitMissingCourse(waypoints,course);
    
    % obtain the (horizontal) initial positions
    hip = complex(waypoints(:,1), waypoints(:,2));
    
    % obtain the starting curvature, final curvature, and length of each segment.
    [~, ~, hl] = matlabshared.tracking.internal.scenario.clothoidG1fit2(hip(1:n-1),course(1:n-1),hip(2:n),course(2:n));
    
    % report cumulative horizontal distance traveled from initial point.
    hcd = [0; cumsum(hl)];
    
    % Fetch the Scaled velocity used to avoid collisions
    Velocity = vehicles(i).Speed';
    
    % To be removed
    %Velocity(isnan(Velocity))=0.01;
    
    % Find positions that were repeated because the actor was stopped
    [~,idx] = unique(hcd,'rows','stable');
    
    % Remove those waypoints and velocity from trajectory and add waitTime
    waypoints = waypoints(idx,:);
    Velocity = Velocity(idx);
    d = diff(idx);
    waitTime = [ds.SampleTime*(d-ones(size(d)));0];
    
    newActor =  vehicle(ds, 'Position', waypoints(1,:), 'Length', ...
                        oldActor.Length, 'ClassID', oldActor.ClassID,...
                        'Width', oldActor.Width, 'Height', ...
                        oldActor.Height);
    
    trajectory(newActor, waypoints, Velocity, waitTime);
end
end

function [vehicles, tout] = createVehicleStruct(out)
    numActors =  numel(out.logsout.get('Actor').Values.ID.Data(:,:,end));
    tout = out.logsout.get('Actor').Values.ID.Time;
    for i = 1 : numActors
        Position = out.logsout.get('Actor').Values.Position.Data(i,:,:);
        Pos = reshape(Position, 3, [])';
        vehicles(i).Position = Pos;

        Speed = out.logsout.get('Actor').Values.Speed.Data(i,:,:);
        Speed = reshape(Speed, 1, [])';
        vehicles(i).Speed = Speed;
    end
end

function copyMap(newScenario,currScenario)

    newScenario.RoadHistory = currScenario.RoadHistory;
    newScenario.RoadSegments = currScenario.RoadSegments;
    newScenario.RoadTiles = currScenario.RoadTiles;
    newScenario.RoadCenters = currScenario.RoadCenters;
    newScenario.RoadTileCentroids = currScenario.RoadTileCentroids;
    newScenario.RoadTileMaxRadii = currScenario.RoadTileMaxRadii;
    newScenario.RoadID = currScenario.RoadID;
    newScenario.CachedLaneMarkingVertices = currScenario.CachedLaneMarkingVertices;
    newScenario.CachedLaneMarkingFaces = currScenario.CachedLaneMarkingFaces;
    newScenario.ShowRoadBorders = currScenario.ShowRoadBorders;
    newScenario.IsOpenDRIVERoad = currScenario.IsOpenDRIVERoad;
    newScenario.EndRoadIDsRoadCenters = currScenario.EndRoadIDsRoadCenters;
    newScenario.SampleTime = currScenario.SampleTime;
    newScenario.StopTime = currScenario.StopTime;
end