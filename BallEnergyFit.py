import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

data = np.genfromtxt('REnergyBall.txt')
n ,energy = np.hsplit(data, 2)

n = n.flatten()
energy = energy.flatten()


napprox = np.linspace(2, 70, 200)

elower = 0.9*n*(n**(2/3.0) - 1)

eupper = 0.9*n*(n - 1)*(2/3.0)

ratio = energy/elower



fig = plt.figure()





ax2 = fig.add_subplot(111)
ax2.plot(n[18:], ratio[18:],  color = 'blue', zorder = 15)

ax2.set_xlabel('N')
ax2.set_ylabel('E(N)/L(N)')
"""
ax = fig.add_subplot(111)
ax.scatter(n, energy, color= "red", marker='x',s = 15 ,label= 'E(N)', zorder = 20)
ax.plot(n, elower, color = 'blue', zorder = 1)

ax.legend()
ax.set_xlabel('n')
ax.set_ylabel('Energy')
"""

plt.savefig('BallBoundRatio.png', dpi = 300, format = 'png')

plt.show()
