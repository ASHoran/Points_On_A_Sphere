#
# python code to compute points on a sphere
# uses a simple Monte Carlo method 
# draws the points on the sphere and plots the energy against iteration number
#

import numpy as np
import matplotlib.pyplot as plt
import random
import time

#from mpl_toolkits.mplot3d import Axes3D        we dont need this as it is already in enthought

# project one or all points to the sphere
def proj(x,j):                           #x is a list, j is an integer
    if(j==n):
        for i in range(0,n):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm                #makes all points unit vector
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x
    
# set the number of points
n = 12         

#assign random start points on the sphere
randstart = 2.0*np.random.random((n,3))-1.0
x = proj(randstart, n)


def calcenergy(x):
    energy=0.0
    for i in range(0,x.shape[0]):
        for j in range(i+1, x.shape[0]):
            distance = np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    return energy

def move(x, amplitude):
    i = random.randint(0, n-1)
    old = x.copy()
    direction = 2.0*np.random.random(3) - 1
    x[i] = x[i] + amplitude * direction
    x = proj(x, n)
    return x, old

    
def montecarlo(x,loops, amplitude):
    start = time.time()
    initial = x.copy()
    
    for loop in range(0, loops+1):
        oldenergy = calcenergy(x)
        x , old = move(x, amplitude)
        energy = calcenergy(x)
        difference = 0.0
        difference = energy - oldenergy

        if (difference > 0.0):
            x = old
            energy = oldenergy.copy()
            
        if (loop % 1000 == 0):
            print loop, energy
            amplitude = amplitude/2
        
        if (amplitude < 1e-8):
            return  energy, x
    return  energy,x
    
energy, x = montecarlo(x, 10000, 0.05)


"""
#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]   
#j is a complex number, it interprets this as how many intervals to divide the grid into soin this case 100 intervals
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)



#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])
   

#Render
fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
ax.scatter(x1,x2,x3,color="black",s=80)
plt.axis('off')
ax.view_init(elev=0.,azim=0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")


ax2 = fig.add_subplot(111)
ax2.plot(looplist, energylist)

ax2.set_xlabel("iteration number")
ax2.set_ylabel("energy")
#ax2.set_xscale('log')
plt.show() 
"""