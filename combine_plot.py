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
'''
Rainbowlosses = pd.read_csv('Rainbowlosses.csv').to_numpy().flatten()
Duelinglosses = pd.read_csv('Duelinglosses.csv').to_numpy().flatten()
Vanillalosses = pd.read_csv('Vanillalosses.csv').to_numpy().flatten()

plt.plot(Rainbowlosses, label ='Rainbow')
plt.plot(Duelinglosses, '-.', label ='Dueling')
plt.plot(Vanillalosses, '--', label ='Double')

plt.xlabel("episode")
plt.ylabel("loss")
plt.legend()
plt.title('Comparing Loss')
plt.show()
'''

Rainbowsteps = pd.read_csv('num_steps_Rainbow.csv').to_numpy().flatten()
Duelingsteps = pd.read_csv('num_steps_Dueling.csv').to_numpy().flatten()
Vanillastpes = pd.read_csv('num_steps_Vanilla.csv').to_numpy().flatten()
idx1 = np.where(Duelingsteps == 1000)
idx2 = np.where(Vanillastpes == 1000)
idx3 = np.where(Rainbowsteps == 1000)
Duelingsteps = np.delete(Duelingsteps, idx1)
Vanillastpes = np.delete(Vanillastpes, idx2)
Rainbowsteps = np.delete(Rainbowsteps, idx3)
print(np.mean(Duelingsteps))
print(np.mean(Vanillastpes))
print(np.mean(Rainbowsteps))
xmin = min( len(Duelingsteps), len(Vanillastpes))


x = range(1, xmin+1)
#Rainbowscores = Rainbowscores[:xmin]
Duelingsteps = Duelingsteps[:xmin]
Vanillastpes = Vanillastpes[:xmin]

#plt.plot(x, Rainbowsteps, label ='Rainbow', alpha = 0.7)
plt.plot(x, Duelingsteps, label ='Dueling',  alpha = 0.7)
plt.plot(x, Vanillastpes, label ='DoubleDQN',  alpha = 0.7)

plt.xlabel("episode")
plt.ylabel("number of steps")
plt.legend()
plt.title('Comparing the number of steps per episode')
plt.show()



scores = pd.read_csv('Rainbowscores3.csv').to_numpy().flatten()
losses = pd.read_csv('Rainbowlosses3.csv').to_numpy().flatten()
discounted_total = pd.read_csv('Rainbowdiscounted3.csv').to_numpy().flatten()
plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.title('score: %s' % np.mean(scores[-10:]))
plt.xlabel('number of episode')
plt.ylabel('score')
plt.plot(scores)
plt.subplot(132)
plt.title('loss')
plt.xlabel('number of steps')
plt.ylabel('loss')
plt.plot(losses)
plt.subplot(133)
plt.title('discounted cumulative rewards: %s' % np.mean(discounted_total[-10:]))
plt.xlabel('number of episode')
plt.ylabel('discounted cumulative rewards')
plt.plot(discounted_total)
plt.show()