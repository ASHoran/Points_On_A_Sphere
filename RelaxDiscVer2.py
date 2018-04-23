import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

start = datetime.now()

n = 10

#If any point moves off the disc, projects it back onto the disc boundary
def bounded(x):
    i= 0
    while (i < x.shape[0]):
        if (np.sqrt(sum(x[i]**2)) > 1.0):
            norm = np.sqrt(sum(x[i]**2))
            x[i] = x[i]/norm
            i = i + 1
        else:
            i =i + 1
    return x

#Find a random starting configuration of n points on a unit disc    

def randstart(n):
    x = np.zeros((n,2))
    r = np.random.uniform(low = 0, high = 1, size = n)
    theta = np.random.uniform(low = 0, high = 2*np.pi, size = n)
    for i in range(0,n):
        x[i,0] = r[i]*np.cos(theta[i])
        x[i,1] = r[i]*np.sin(theta[i])
    return x   


#Non-random starting config of n points on unit disc using the distribution given in paper


def distribute(n):
    x = np.zeros((n, 2))
    u = np.random.uniform(low=0, high=1, size=n)
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)
    r = np.zeros((n,1))
    for i in range (0,n):
        r[i] = (np.sin((2*np.pi*u[i]/4)))
        x[i,0] = r[i]*np.cos(theta[i])
        x[i,1] = r[i]*np.sin(theta[i])
    return x
#x  = distribute(n)


#Distribution of starting points in a disc of radius r:
def rstart(n, r):
    x = np.zeros((n,2))
    radius = np.random.uniform(low = 0, high = r, size = n)
    theta = np.random.uniform(low = 0, high = 2*np.pi, size = n)
    for i in range(0,n):
        x[i,0] = radius[i]*np.cos(theta[i])
        x[i,1] = radius[i]*np.sin(theta[i])


"""Given an n function returns an n x 2 array of points of unit length.
Randlist creates a list of n vectors of unit length and reject ones that are not.
This list is added to an empty n x 2 array.
"""

def bestdist(n):
    x = np.zeros((n, 2))
    u = np.random.uniform(low=0, high=1, size=n)
    print u
    theta = np.random.uniform(low=0, high=2*np.pi, size=n)
    r = np.zeros((n,1))
    for i in range (0,n):
        r[i] = (u[i]*(2 - u[i]))**0.5
        x[i,0] = r[i]*np.cos(theta[i])
        x[i,1] = r[i]*np.sin(theta[i])
    return x
x = bestdist(n)
 
"""
Hopefully the correct (final) distribution for the starting points on the disc 
"""


#Calcuates the energy of the system
def calcenergy(x):
    energy=0.0
    for i in range(0,x.shape[0]):
        for j in range(i+1, x.shape[0]):
            distance = np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    return energy
    
def moveforces(x, amplitude):                             
    resultforceslist=[]
    old = np.zeros((x.shape[0],2))
    for i in range(x.shape[0]):        
        forces=np.zeros((x.shape[0],2))
        for j in range(0,x.shape[0]):
            if j!=i:
                Fij=(x[i]-x[j])/(np.sqrt(sum((x[i]-x[j])**2)))**3    
                forces[j]=Fij                                        
        
        resultforce=np.sum(forces,axis=0)                                 
        resultforceslist.append(resultforce)                          
        old[i] = x[i].copy()
        x[i]=x[i]+amplitude*resultforceslist[i]
        x = bounded(x)
      
    return old, x
    
def relax(x,loops,amplitude):
    for loop in range (0,loops+1):
        oldenergy = calcenergy(x)
        old, x = moveforces(x, amplitude)
        energy = calcenergy(x)
        difference=0.0
        difference=energy-oldenergy
        #if(loop%20==0):
            #print("{0} {1:.6f}".format(loop,energy))
        if(difference>0.0):
            x=old
            energy=oldenergy.copy()
            amplitude=amplitude/2.0                                  
        if (amplitude < 1e-6):
            return x, energy
    return x, energy

#x, energy = relax(randstart(n), 10000, 0.05)       
x, energy = relax(bestdist(n), 1000, 0.05)
print n, energy
#print("Run time = {0}\n".format((datetime.now()-start).total_seconds()))
 
 
#Cacluate Moment of Inertia Tensor
def moit(x):
    xsquare=x**2
    #Find components of MoI Tensor
    Ixx = sum(xsquare[:,1])
    Iyy = sum(xsquare[:,0])
    Ixy = -sum(x[:,0]*x[:,1])
    #Write as a matrix
    I = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    #Find the eigenvalues & eigenvectors
    Ieigvals, Ieigvecs = np.linalg.eig(I)
    return Ieigvals, Ieigvecs

#Find intial symmetry classification (i.e find if any eigenvalues are equal)
def eigcheck(x):
    Ieigvals, Ieigvecs = moit(x)
    if (abs(Ieigvals[0] - Ieigvals[1]) < 1e-5):
        return "At least C3 symmetry"    
    else:
        return "At most C2 symmetry"


#print eigcheck(x)

def compare(x,y):
    d = 0.0
    for i in range(0, n):
        dmin=100.0
        for j in range(0,n):
            dij = ((x[i,0]-y[j,0])**2 + (x[i,1]-y[j,1])**2)**0.5    
            if (dmin > dij):
                dmin = dij
        d = d + dmin
    return d

"""#Find cyclic symmetries
def cyclic(x):
    savex = np.zeros((x.shape[0], 2))
    savex = x.copy()
    xydiff = 10.0
    j = x.shape[0]
    while (xydiff > 1e-3):
        while True:
            print j
            q = 2*np.pi/j
            rotation = np.array([[np.cos(q),np.sin(q)],[np.sin(q), np.cos(q)]])
            x = np.dot(rotation, savex.T).T
            xydiff = compare(x, savex)
            if (xydiff < 1e-3):
                return 'C{} symmetry'.format(j)
            elif (j<3):
                break
            else:
                j = j-1
print cyclic(x)        
"""

#Plot the disc boundary
thetas = np.linspace(0,2*np.pi,1000)
xdisc = np.cos(thetas)
ydisc = np.sin(thetas)

#convert data
x1=[]
x2=[]
for i in range(0,x.shape[0]):
    x1.append(x[i,0])
    x2.append(x[i,1])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xdisc, ydisc, color = "orange" , linewidth = 5.0, zorder = 1)
ax.scatter(x1, x2, color = "teal", s = 100, zorder = 10)
ax.set_aspect('equal') 
plt.axis('off')

#ax2=fig.add_subplot(212)
#ax2.plot(looplist,energylist)
#ax2.set_xlabel("iteration number")
#ax2.set_ylabel("energy")

#plt.savefig('16points3.png', dpi = 300, format = 'png')

plt.show()