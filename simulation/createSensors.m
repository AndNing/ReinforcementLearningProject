function [sensors, numSensors] = createSensors(scenario)

profiles = actorProfiles(scenario);
sensors{1} = lidarPointCloudGenerator('SensorIndex', 1, ...
    'SensorLocation', [3.7 0], ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian', ...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits', [-20 20], ...
    'ActorProfiles', profiles);

sensors{2} = lidarPointCloudGenerator('SensorIndex', 2, ...
    'SensorLocation', [2.8 0.9], ...
    'Yaw', 75, ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian', ...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits',[-20 20], ...
    'ActorProfiles', profiles);

sensors{3} = lidarPointCloudGenerator('SensorIndex', 3, ...
    'SensorLocation', [2.8 0.9], ...
    'Yaw', 135, ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian',...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits',[-20 20],...
    'ActorProfiles', profiles);

sensors{4} = lidarPointCloudGenerator('SensorIndex', 4, ...
    'SensorLocation', [-1 0], ...
    'Yaw', 180, ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian', ...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits',[-20 20],...
    'ActorProfiles', profiles);

sensors{5} = lidarPointCloudGenerator('SensorIndex', 5, ...
    'SensorLocation', [2.8 -0.9], ...
    'Yaw', -75, ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian', ...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits',[-20 20], ...
    'ActorProfiles', profiles);

sensors{6} = lidarPointCloudGenerator('SensorIndex', 6, ...
    'SensorLocation', [2.8 -0.9], ...
    'Yaw', -135, ...
    'MaxRange', 100, ...
    'DetectionCoordinates', 'Sensor Cartesian', ...
    'HasOrganizedOutput', false, ...
    'HasEgoVehicle', false, ...
    'HasRoadsInputPort', false, ...
    'AzimuthLimits', [-45 45], ...
    'ElevationLimits',[-20 20],...
    'ActorProfiles', profiles);

numSensors = 6;

for i = 1:numSensors
sensors{i}.AzimuthResolution = 0.16;
sensors{i}.Height = 0.4;
sensors{i}.HasRoadsInputPort = false;
sensors{i}.HasOrganizedOutput = false;
end