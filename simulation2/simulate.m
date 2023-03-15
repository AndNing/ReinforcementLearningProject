function running = simulate(scenario, roads)
    running = advance(scenario);
    calculategrid(scenario, roads);
