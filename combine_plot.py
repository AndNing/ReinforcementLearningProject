import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Rainbowscores = pd.read_csv('Rainbowdiscounted3.csv').to_numpy().flatten()
Duelingscores = pd.read_csv('Duelingdiscounted.csv').to_numpy().flatten()
Vanillascores = pd.read_csv('Vanialladiscounted.csv').to_numpy().flatten()

xmin = min(len(Rainbowscores), len(Duelingscores), len(Vanillascores))


x = range(1, xmin+1)
Rainbowscores = Rainbowscores[:xmin]
Duelingscores = Duelingscores[:xmin]
Vanillascores = Vanillascores[:xmin]

plt.plot(x, Rainbowscores, label ='Rainbow', alpha = 0.7)
plt.plot(x, Duelingscores, label ='Dueling',  alpha = 0.7)
plt.plot(x, Vanillascores, label ='Double',  alpha = 0.7)

plt.xlabel("episode")
plt.ylabel("return")
plt.legend()
plt.title('Comparing Returns')
plt.show()

print(np.std(Rainbowscores[-100:]))
print(np.std(Duelingscores[-100:]))
print(np.std(Vanillascores[-100:]))
