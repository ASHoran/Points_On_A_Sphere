import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time


a = open('GFenergy.txt' , 'w+')
b = open('GFtime.txt', 'w+')
start = time.time()
n = 9

#Projects a set of points onto the surface of a sphere
def proj(x,j):                           
    if(j==x.shape[0]):
        for i in range(0,x.shape[0]):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm                
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm 
    return x

#For random starting positions of n points on the sphere
randstart = proj((2.0*np.random.random((n,3))-1.0),n)

#Non-random starting points for MC comparison
"""
x = np.array([[ 0.53623838, -0.59670876, -0.54269971],
       [-0.23008107, -0.72882284, -0.10879696],
       [-0.62314275, -0.7021917 ,  0.63019829],
       [-0.32977346,  0.29151228, -0.75407049],
       [ 0.11805897, -0.89684512,  0.43434246],
       [ 0.23341929, -0.37254452,  0.80520988],
       [-0.76662625,  0.89828664, -0.83026658],
       [ 0.75187254, -0.88144381, -0.74593487],
       [-0.55027591, -0.99959556,  0.48707672],
       [-0.98140752,  0.1356138 , -0.07206939],
       [-0.20140858,  0.03881581,  0.96934777],
       [-0.02093414,  0.82771005,  0.42439192],
       [ 0.12213835, -0.58824262,  0.0757515 ],
       [-0.00838308, -0.60457987, -0.10052152],
       [ 0.81656151, -0.73951168,  0.49634432],
       [-0.15519182, -0.31543117, -0.491001  ],
       [ 0.59992363,  0.38301432,  0.87542512],
       [ 0.37046539, -0.71883606, -0.93520648]])
"""
 
#n = x.shape[0]
#x = proj(x, n)


"""Calculates the energy of a system x"""
def calcenergy(x):
    energy=0.0
    for i in range(0,x.shape[0]):
        for j in range(i+1, x.shape[0]):
            distance = np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    return energy


"""Finds the forces on each particle in x and moves each point in the direction 
of this force by a size specified by amplitude, then renormalises the points 
onto the surface of the sphere, returns the new position of the points as x and
saves the old position of the points as old"""

def moveforces(x, amplitude):                             
    resultforceslist=[]
    old = np.zeros((x.shape[0],3))
    for i in range(x.shape[0]):        
        forces=np.zeros((x.shape[0],3))
        for j in range(0,x.shape[0]):
            if j!=i:
                Fij=(x[i]-x[j])/(np.sqrt(sum((x[i]-x[j])**2)))**3    
                forces[j]=Fij                                        
        
        resultforce=np.sum(forces,axis=0)
        #print  resultforce                                 
        resultforce=resultforce-np.dot(resultforce,x[i])*x[i]                    
        resultforceslist.append(resultforce)                          
        old[i] = x[i].copy()
        x[i]=x[i]+amplitude*resultforceslist[i]
    x = proj(x, x.shape[0])  
    return old, x

"""For a given set of points x, minimises the potential energy of the system by
repeatedly moving the points in the direction of the forces acting on them. 
Loops through this until the specified  amplitude < 1e-6 or for a specified 
number of loops, returns the mimimum energy"""
def relax(x,loops,amplitude):
    energylist = []
    looplist = []
    for loop in range (0,loops+1):
        oldenergy = calcenergy(x)
        old, x = moveforces(x, amplitude)
        energy = calcenergy(x)
        difference=0.0
        difference=energy-oldenergy
        
        if(difference>0.0):
            x=old
            energy=oldenergy.copy()
            amplitude=amplitude/2.0 
        
        if (loop % 100 == 0):
            print loop, energy
               
        
        a = open('GFenergy.txt' , 'a')
        b = open('GFtime.txt', 'a')
        a.write('%s \n' % energy)
        b.write('%s \n' % time.time())
        a.close()
        b.close()                                      
        
        if (abs(difference) < 1e-8):
            return x, energy, looplist, energylist
    return x, energy, looplist, energylist
        
x, energy, looplist, energylist = relax(randstart, 10000, 0.05)    

#print energy
end = time.time()
print looplist
print energylist

print ("GF run time = {0}\n".format(end-start))
print x

#Find components of MoI Tensor
def MoIT(x):
    xsquare = x**2
    Ixx = sum(xsquare[:,1]) + sum(xsquare[:,2])
    Iyy = sum(xsquare[:,0]) + sum(xsquare[:,2])
    Izz = sum(xsquare[:,0]) + sum(xsquare[:,1])
    Ixy = -sum(x[:,0]*x[:,1])
    Ixz = -sum(x[:,0]*x[:,2])
    Iyz = -sum(x[:,1]*x[:,2])
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    Ieigvals, Ieigvecs = np.linalg.eig(I)
    return Ieigvals, Ieigvecs

Ieigvals, Ieigvecs = MoIT(x) 

def scount(x):
    s = np.array([0,0,0])
    if (abs(Ieigvals[0] - Ieigvals[1]) < 1e-5):
        s = s+np.array([1,0,0])
        
    if (abs(Ieigvals[0] - Ieigvals[2]) < 1e-5):
        s = s+np.array([0,1,0])
        
    if (abs(Ieigvals[1] -Ieigvals[2]) < 1e-5):
        s=s+np.array([0,0,1])
    return s
s = scount(x)

#Rotate so that points are in terms of principal axis
rotx= np.zeros((x.shape[0],3))
for i in range (0, x.shape[0]):
    x[i] = np.dot(Ieigvecs.T, x[i])
    
rotx = x.copy()
      
if (np.count_nonzero(s)==1):
    if (np.array_equal(s, [0,0,1]) == True ):
        temp = np.copy(x[:,2])
        x[:,2] = np.copy(x[:,0])
        x[:,0] = temp
        
    if (np.array_equal(s, [0,1,0]) == True ):
       temp = np.copy(x[:,2])
       x[:,2] = np.copy(x[:,1])
       x[:,1] = temp    

#print x

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

"""
#Octahedron plot:
for i in range(x.shape[0]):
    all_distances = []
    for j in range(x.shape[0]):
        distances = []
        if j!=i:
            distances.append(np.linalg.norm(x[i] - x[j]))
    all_distances.append(distances)
    
def smallest(points, numberclosest, dists):
    l = 0.0
    closestpoints = []
    while l != numberclosest:
        if (len(distances) == 0):
            return closestpoints
            
        minimum = np.min(distances)
        k = np.argmin(distances)
        diff = 0.01

        if (abs(minimum - diff)> 1e-6):
                closestpoints.append(x[k])
                np.delete(points, distances[k])
                diff = minimum
                l = l + 1
            
        else:
            np.delete(points, distances[k])
    return closestpoints

for i in range (0, x.shape[0]):
    all_closest = []
    closestpoints = smallest(x, 4 ,all_distances[i])
    all_closest.append(closestpoints)

"""
#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,x.shape[0]):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])
    

#Render
fig = plt.figure()

ax = plt.axes( projection = '3d')
#ax = plt.axes(projectiox.shape[0]='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)

"""
for i in range(0, x.shape[0]):
   closestpoints = all_closest[i]
   closestpoints1 = []
   closestpoints2 = []
   closestpoints3 = []
   for j in range(0, x.shape[0]):
       closestpoints1.append(closestpoints[j, 0])
       closestpoints2.append(closestpoints[j, 1])
       closestpoints3.append(closestpoints[j, 2])
   for k in range(len(closestpoints1)):
       ax.plot([x1[i],closestpoints1[k]],[x2[i],closestpoints2[k]],[x3[i],closestpoints3[k]],'r-')


"""
#Tetrahedron plot:
"""
for i in range(len(x1)):
    for j in range(len(x1)):
        ax.plot([x1[i],x1[j]],[x2[i],x2[j]],[x3[i],x3[j]],'r-')
"""

#ax.plot([0,0],[0,0],[1,-1], 'r--')
#ax.plot([1,-1],[0,0,],[0,0], 'b--')
xx = np.zeros((10,10))
yy = np.linspace(-1,1,10)
zz = np.linspace(-1,1,10)
YY,ZZ = np.meshgrid(yy,zz)
ax.plot_surface(YY, xx, ZZ, alpha = 0.4, color = 'red')

ax.scatter(x1,x2,x3,color="black",s=170)
#ax.plot([x1[2],0, x1[1]], [x2[2],0, x2[1]], [x3[2],0, x3[1]], 'b--')
plt.axis('off')
ax.view_init(elev = 0.0,azim=10.0)

#ax.plot(x1,x2,x3, 'r-')
ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(x.shape[0])+"points on a sphere")
"""
ax2=fig.add_subplot(111)
ax2.plot(looplist,energylist)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("energy")
#ax2.semilogx()
#ax2.set_xlim([1,1000])
"""


plt.savefig('Sphere_%shoriz2.png'% (n), dpi = 300, format = 'png')

plt.show()