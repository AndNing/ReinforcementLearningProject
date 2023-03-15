import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('./simulation', nargout=0)

eng.simulationsetup(nargout=0)

while True:
    (x,y) = eng.simulate(eng.workspace['scenario'], nargout=2)
    eng.