import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


data = np.genfromtxt('FINALDiscEnergies.txt')   #import mimimum energy values for n=2 to 30
n, energy = np.hsplit(data, 2)
n = n.flatten()                      #to use optimization.curve_fit function, need n and energy to be a 1d array, this is currently 1 x n (need n x 1)
energy = energy.flatten()

n= n[1:]
energy = energy[1:]

def func(n,a,b,c):
    return np.pi*0.25*n**2.0 + b*n**1.5 + c*n


guess = np.array([0.0,0.0,0.0])

opt = optimization.curve_fit(func, n, energy,p0 = guess)[0]

aa = opt[0]
#AA = opt[1]
bb = opt[1]
#BB = opt[3]
cc = opt[2]
#CC = opt[5]

fit = func(n, aa, bb, cc)
resid = energy - fit
zeroline = np.zeros(len(n))

fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(n, energy, color= "red", marker='x',s = 15 ,label= 'E(N)', zorder = 200)
ax.plot(n, fit, color="black", label = 'W(N)', zorder = 100)

ax.legend()
ax.set_xlabel('N')
ax.set_ylabel('Energy')

"""
ax2 = fig.add_subplot(111)
#ax2.plot(n, resid, color = "black")
ax2.scatter(n, resid, color = "blue",marker = 'x', s=20)
ax2.plot(n, zeroline, color = "red")
ax2.set_xlabel('N')
ax2.set_ylabel('E(N) - W(N)')
fig.tight_layout()
"""
plt.savefig('DiscFitEnergy.png', dpi = 300, format = 'png')
#plt.savefig('DiscFitResid.png', dpi = 300, format = 'png')

plt.show()
print("a: {0:.8f} \n".format(aa))
#print("A: {0:.8f} \n".format(AA))
print("b: {0:.8f} \n".format(bb))
#print("B: {0:.8f} \n".format(BB))
print("c: {0:.8f} \n".format(cc))
#print("C: {0:.8f} \n".format(CC))
