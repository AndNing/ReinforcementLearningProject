clear all
close all

clear 

rng('shuffle');
% seednum = randi(100000);

sampleTime = 1;
stopTime = 100000;
% s = setseed(seednum);  % need eng.workspace['seednum'] = some number before this script run

egoVehiclePositionOffset = 1;
startpositions = [5 12.5; 5 62.5; 5 112.5];

% egoVehiclePosition = [startpositions(1,:),0];
egoVehiclePosition = [startpositions(randi([1 3],1,1),:),0];
egoVehicleSpeed = 10;

numActorVehicles = 5;
actorVehicleMinSpeed = 1;
actorVehicleMaxSpeed = 5;

rewardValues.offroad = 0;
rewardValues.time = -1;
rewardValues.vehicle = 0;
rewardValues.finish = 1000;
rewardValues.boundary = -10000;


endpositions = [13 12; 13 2; 13 7; 13 13; 13 1; 13 3; 13 4; 13 5; 13 6; 13 8; 13 9; 13 10; 13 11];
% goalGridPosition = [endpositions(1,:)];
goalGridPosition = endpositions(randi([1 13],1,1),:);

[scenario, egoVehicle, roads] = scenariosetup(sampleTime, stopTime, egoVehiclePosition);

%xCor = 0;
%yCor = 0;
%radius = 50;
%theta = 0: pi/10: 2*pi;
%roiCircular(:,1) = xCor+radius*cos(theta);
%roiCircular(:,2) = yCor+radius*sin(theta);

roiPolygon = [10 120; 120 120; 120 10; 10 10;];


[startSet2,yaw2] = helperSamplePositions(scenario,numActorVehicles,'ROI',roiPolygon);
for idx = 1 : numActorVehicles
    vehicle(scenario,'Position',startSet2(idx,:),'Yaw',yaw2(idx),'ClassID',2);
end

% figScene = figure('Name','AutomaticScenarioGeneration');
% set(figScene,'Position',[0,0,900,500]);
% 
% hPanel1 = uipanel(figScene,'Position',[0 0 0.5 1]);
% hPlot1 = axes(hPanel1);
% plot(scenario,'Parent',hPlot1);
% title('Points for Selecting Start Positions')
% hold on
% %plot(roiCircular(:,1),roiCircular(:,2),'LineWidth',1.2,'Color','k')
% plot(startSet2(:,1),startSet2(:,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
% 
% xlim([-50 190])
% ylim([-85 330])
% hold off

% hPanel2 = uipanel(figScene,'Position',[0.5 0 0.5 1]);
% hPlot2 = axes(hPanel2);
% plot(scenario,'Parent',hPlot2);
% title('Start Positions and Vehicle Placement')
% hold on
% plot(startSet2(:,1),startSet2(:,2),'ks','MarkerSize',15,'LineWidth',1.2);
% xlim([-50 190])
% ylim([-85 330])
% hold off

roiPolygon = [0 130; 130 130; 130 0; 0 0; 0 130];
%roiPolygon = [10 120; 120 120; 120 10; 10 10; 10 120];
%numPoints1 = 1;
%goalSet1 = helperSamplePositions(scenario,numPoints1,'ROI',roiPolygon);

numPoints2 = numActorVehicles;
goalSet2 = helperSamplePositions(scenario,numPoints2,'Lanes',1);

% figure
% plot(scenario); 
% title('Goal Positions')
% hold on
% plot(roiPolygon(:,1), roiPolygon(:,2),'LineWidth',1.2,'Color','r')
% %plot(goalSet1(:,1), goalSet1(:,2),'ro','MarkerSize',5,'MarkerFaceColor','r')
% plot(goalSet2(:,1),goalSet2(:,2),'bo','MarkerSize',5,'MarkerFaceColor','b')
% xlim([-50 190])
% ylim([-85 310])
% hold off

startPositions =startSet2;
goalPositions = goalSet2;

info = helperGenerateWaypoints(scenario,startPositions,goalPositions);
for indx = 1:length(startPositions)
    vehicleData = scenario.Actors(indx+1);
    speed = randi([actorVehicleMinSpeed,actorVehicleMaxSpeed],1,1);
    waypts = info(indx).waypoints;
    trajectory(vehicleData,waypts,speed);
end


figScene = figure;
set(figScene,'Position',[0,0,600,600]);
movegui(figScene,'center');
hPanel = uipanel(figScene,'Position',[0 0 1 1]);
hPlot = axes(hPanel);
plot(scenario,'Parent',hPlot);
title('Generated Scenario')


% Open the Simulink system block
% open_system('CollisionFreeSpeedManipulator');
% 
% % Pass the scenario object as input
% set_param('CollisionFreeSpeedManipulator/VelocityUpdate',...
%           'ScenarioName','scenario')
% 
% % Run the simulation and log the output
% out = sim('CollisionFreeSpeedManipulator','StopTime','20');
% 
% % Run the simulation
% newScenario = helpergetCFSMScenario(out,scenario);
% close all;
% figScene = figure;
% set(figScene,'Position',[0,0,600,600]);
% movegui(figScene,'center');
% hPanel = uipanel(figScene,'Position',[0 0 1 1]);
% hPlot = axes(hPanel);
% plot(newScenario,'Parent',hPlot);
% title('Updated Scenario')
% hold on
% h1 = plot(goalPositions(:,1),goalPositions(:,2),'gs','MarkerSize',15,'LineWidth',1.2);
% h2 = plot(startPositions(:,1),startPositions(:,2),'rs','MarkerSize',15,'LineWidth',1.2);
% legend([h2 h1],{'Start Positions';'Goal Positions'},'Location','southoutside','Orientation','horizontal')
% hold off
% Run the simulation

rmax = max(roads);
rmin = min(roads);
rdist = rmax - rmin;
gridsize = rdist / 10;
% 
roadGrid = zeros(gridsize(1),gridsize(2));

for i=1:size(roads,1)/2
    roadStart = roads(2*i-1,1:2);
    roadEnd = roads(2*i,1:2);

    if (roadStart(2) - roadEnd(2)) == 0
        roadColumn = ((roadStart(2) - 5)/10) + 1;
        roadGrid(:,roadColumn) = 10;
    else
        roadRow = ((roadStart(1) - 5)/10) + 1;
%             disp(roadRow);
        roadGrid(roadRow,:) = 10;
    end
end
roadGrid(goalGridPosition(1),goalGridPosition(2)) = 30;
staticRoadGrid = flip(flip(roadGrid),2);
% disp(staticRoadGrid)

egoVehicleGridPosition = ceil(egoVehiclePosition/10);
egoVehicleGridPosition = egoVehicleGridPosition(1:2);
% egoVehicleGridPosition = [9 5];

[countGrid,roadGrid] = calculategrid(scenario, staticRoadGrid, gridsize, egoVehicleGridPosition);

 
running = true;
%while running
%     [scenario, running, reward, roadGrid, countGrid] = simulate(scenario, roadGrid, rewardValues, gridsize, goalGridPosition, 1, egoVehicleSpeed);
% end
 %simulate(scenario, roads, rewardValues);
