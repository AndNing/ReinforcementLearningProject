function [scenario, roads, gridsize] = scenariosetup(sampleTime, stopTime)
    scenario = drivingScenario('SampleTime',sampleTime, 'StopTime', stopTime);

    roads = [];

    roadCenters = [0 5 0;
    130 5 0];
    laneSpecification = lanespec([1,1],'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [0 65 0;
        130 65 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [0 100 0;
        130 100 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];

    roadCenters = [5 0 0;
        5 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [55 0 0;
        55 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [105 0 0;
        105 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];

    roadCenters = [125 0 0;
    125 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];

    roadCenters = [0 125 0;
    125 125 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];

    rmax = max(roads);
    rmin = min(roads);
    rdist = rmax - rmin;
    gridsize = rdist / 10;
        