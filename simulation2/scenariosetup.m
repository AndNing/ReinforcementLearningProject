function [scenario, egoVehicle, roads] = scenariosetup(sampleTime, stopTime, egoVehiclePosition)
    scenario = drivingScenario('SampleTime',sampleTime, 'StopTime', stopTime);

    roads = [];

    roadCenters = [0 15 0;
    130 15 0];
    laneSpecification = lanespec([1,1],'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [0 65 0;
        130 65 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [0 115 0;
        130 115 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];

    roadCenters = [0 0 0;
        0 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [50 0 0;
        50 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];
    
    roadCenters = [100 0 0;
        100 130 0];
    laneSpecification = lanespec([1,1], 'Width', 5);
    road(scenario, roadCenters, 'Name', 'Road', 'Lanes', laneSpecification);
    roads =[roads;roadCenters];


    egoVehicle = vehicle(scenario, ...
    'ClassID', 1, ...
    'Position', egoVehiclePosition, ...
    'Name', 'Car');
    