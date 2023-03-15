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


    egoVehicle = vehicle(scenario, ...
    'ClassID', 1, ...
    'Position', egoVehiclePosition, ...
    'Name', 'Car');
    