import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


data = np.genfromtxt('FINALDiscEnergies.txt')   #import mimimum energy values for n=2 to 30
ALLn, ALLenergy = np.hsplit(data, 2)
ALLn = ALLn.flatten()                      #to use optimization.curve_fit function, need n and energy to be a 1d array, this is currently 1 x n (need n x 1)
ALLenergy = ALLenergy.flatten()

n = ALLn[1:]
energy = ALLenergy[1:]

guess = np.array([0.0, 0.0, 0.0])         #initial guess for a and b


def func(n,a,n, c):
    return 0.5*n**2 + a*n**1.5 + + b*n**0.5 + c*n   #function we want to optimize to fit our data values
    
opt = optimization.curve_fit(func, n, energy, p0 = guess)[0]   #optimizes func to fit our energy values
aa = opt[0]  
bb = opt[1]
cc=opt[2]

fity = func(n, aa, bb, cc)                   #find our estimates for energy using our function with fitted a and b

resid =(energy - fity)                    #calculating residuals
zeroline = np.zeros(len(n))              #plot line at y=o
fig = plt.figure()

"""
ax = fig.add_subplot(111)
ax.scatter(n, energy, color= "red", marker='x',s = 15 ,label= 'E(N)', zorder = 20)
ax.plot(n, fity, color="black", label = 'W(N)', zorder = 100)

ax.legend()
ax.set_xlabel('n')
ax.set_ylabel('Energy')
"""


ax2 = fig.add_subplot(111)
ax2.scatter(n, resid, color = "blue",marker = 'x' ,s=20, zorder = 15)
ax2.plot(n, resid, color = "blue", zorder = 15)
ax2.axhline(linestyle = '--', color = 'red', zorder = 1)
ax2.set_xlabel('N')
ax2.set_ylabel('E(N) - W(N)')
ax2.legend()


fig.tight_layout()

plt.show()
print("a: {0:.8f} \n".format(aa))
print("b: {0:.8f} \n".format(bb))
print("c: {0:.8f} \n".format(cc))
