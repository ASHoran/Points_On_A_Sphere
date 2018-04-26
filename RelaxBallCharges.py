import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time

start = time.time()
n = 38

#Random distribution of points in the range [-1,1] for each component x,y,z.
randstart = (2.0*np.random.random((n,3))-1.0)

#Create vector of size of charges of each points
q = np.ones((n,1))

"""Use this to give a point a charge greater than one"""
"""
charge =  np.random.randint(0,n)
q[charge] = q[charge] + 4
print charge
"""

"""Calculates the energy of a system x"""

def calcenergy(x):
    ballenergy = 0.0
    pointsenergy = 0.0
    for i in range (0, x.shape[0]):
        ballenergy = ballenergy + 0.5*q[i]*np.dot(x[i],x[i])
        for j in range (i+1, x.shape[0]):
            pointsenergy = pointsenergy + (q[i]*q[j])/(np.linalg.norm(x[i]-x[j]))
    
    energytotal = ballenergy + pointsenergy
    return energytotal 

"""
Finds the total energy of the ball by first summing through the attraction
energy from the ball and then the energies due to particles interacting
with each other
"""
def moveforces(x, amplitude):                             
    resultforceslist=[]
    old = np.zeros((x.shape[0],3))
    for i in range(x.shape[0]):        
        forces=np.zeros((x.shape[0],3))
        for j in range(0,x.shape[0]):
            if j!=i:
                Fij=(q[i]*q[j]*(x[i]-x[j]))/(np.sqrt(sum((x[i]-x[j])**2)))**3    
                forces[j]=Fij                                        
        
        resultforce=np.sum(forces,axis=0)
        resultforce = resultforce - q[i]*x[i]                                 
        #resultforce=resultforce-np.dot(resultforce,x[i])*x[i]                    
        resultforceslist.append(resultforce)                          
        old[i] = x[i].copy()
        x[i]=x[i]+amplitude*resultforceslist[i]
      
    return old, x    

"""
Given the set of points x finds the force on each particle and moves each 
point in the directiong of this force.
Function first sums through the forces acting on point i due to all other 
points j. Total force acting on

"""        


def relax(x,loops,amplitude):
    for loop in range (0,loops+1):
        oldenergy = calcenergy(x)
        old, x = moveforces(x, amplitude)
        energy = calcenergy(x)
        difference=0.0
        difference=energy-oldenergy
        if (loop%100 == 0):
            print loop, energy
        if(difference>0.0):
            x=old
            energy=oldenergy.copy()
            amplitude=amplitude/2.0                                  
        
        if (abs(difference) < 1e-8):
            return x, energy
    return x, energy
        
x, energy = relax(randstart, 1000, 0.05)    

print energy
end = time.time()

print ("Run time = {0}\n".format(end-start))


#Find components of MoI Tensor
print x

def diameter(x):
    d = 0.0
    dist = []
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            dist.append(np.linalg.norm(x[i]-x[j]))
        a = max(dist)
        if (a > d):
            d = a
    return d
        
radius = 0.5*diameter(x)

#Plotting onto a sphere
points=open('points.out','w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(x[i,j]))
    points.write('\n')              
points.close()

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = radius*np.sin(theta)*np.cos(phi)
ys = radius*np.sin(theta)*np.sin(phi)
zs = radius*np.cos(theta)

def radiuscalc(x):
    dist = []
    for i in range(0, x.shape[0]):
        dist.append(np.linalg.norm(x[i]))
    a = max(dist)
    return a

def rings(x):
    rad = radiuscalc(x)
    ring1 = []
    ring2 = []
    colourlist = []
    for k in range(0, x.shape[0]):
        if k != charge:
            if (abs(np.linalg.norm(x[k]) - rad) < 0.5):
                ring1.append(x[k])
                colourlist.append(0.0)
            elif (abs(np.linalg.norm(x[k]) - rad) > 1.0):
                ring2.append(x[k])
                colourlist.append(1.0)
    return np.array(ring1), np.array(ring2), colourlist

ring1, ring2, colors = rings(x)        

#convert data
x1charge=[]
x2charge=[]
x3charge=[]

x1charge.append(x[charge,0])
x2charge.append(x[charge,1])
x3charge.append(x[charge,2])


#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,ring1.shape[0]):
    x1.append(ring1[i,0])
    x2.append(ring1[i,1])
    x3.append(ring1[i,2])

y1=[]
y2=[]
y3=[]
for i in range(0,ring2.shape[0]):
    y1.append(ring2[i,0])
    y2.append(ring2[i,1])
    y3.append(ring2[i,2])

#Render
fig = plt.figure()
ax = plt.axes( projection = '3d')
#ax = plt.axes(projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.2, linewidth=0)
ax.scatter(x1,x2,x3,color="blue",s=170)
ax.scatter(y1, y2, y3, color="red",s=170)
ax.scatter(x1charge, x2charge, x3charge, color = 'green', s = 300)
plt.axis('off')
ax.view_init(elev=90.0,azim=90.0)


ax.set_xlim([-radius,radius])
ax.set_ylim([-radius,radius])
ax.set_zlim([-radius,radius])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")

"""ax2=fig.add_subplot(212)
ax2.plot(looplist,energylist)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("energy")"""

#plt.savefig('4points.png', dpi = 300, format = 'png')

plt.show()
