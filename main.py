import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('./simulation2', nargout=0)

eng.simulationsetup(nargout=0)