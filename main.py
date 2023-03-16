import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('./simulation2', nargout=0)

eng.simulationsetup(nargout=0)

roadGrid = eng.workspace['roadGrid']
running = True
while running:
    (running, reward, roadGrid) = eng.simulate(eng.workspace['scenario'], roadGrid, eng.workspace['rewardValues'], eng.workspace['gridsize'], eng.workspace['goalGridPosition'], nargout=3)