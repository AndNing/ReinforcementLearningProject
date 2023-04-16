% Reward calculation based on current environment and action taken by agent
function reward = calculatereward(pastegoVehicleGridPosition, egoVehiclePosition, egoVehicleGridPosition, newEgoVehiclePosition, goalGridPosition,gridsize, rewardValues, staticRoadGrid, countGrid, gamma, done)
    
      % Distance reward
      distance_old = sum(abs(pastegoVehicleGridPosition - goalGridPosition));
      distance_new = sum(abs(egoVehicleGridPosition - goalGridPosition));
      distance_old = 1 - distance_old / (gridsize(1) + gridsize(2) - 2);
      distance_new = 1 - distance_new / (gridsize(1) + gridsize(2) - 2);
      delta = gamma*distance_new - distance_old;
      reward = delta + rewardValues.time;

      % Off-road penalty
      if staticRoadGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) == 0
        reward = rewardValues.offroad + reward;
      end

      % Boundary penalty
      if  isequal(egoVehiclePosition, newEgoVehiclePosition)
          reward = reward +  rewardValues.boundary;
      end

      % Traffic penalty (amount of cars in same grid position)
      if countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2)) > 0
        reward = reward + rewardValues.vehicle * countGrid(egoVehicleGridPosition(1),egoVehicleGridPosition(2));
      end
      if done == true
        reward = rewardValues.finish;
      end
end
