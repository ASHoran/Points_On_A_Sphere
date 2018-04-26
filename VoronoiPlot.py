import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.colors
#import pickle


"""For best images use 300 points into mesh grid plotter and plot points of 
size 1
"""

"""Import points """
n = 12
x = np.genfromtxt('GFpoints_%s.txt'% (n))



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


def find(x,y):
    """X is the points on sphere, want to find which x any random y is closest to"""
    xpoints = []
    for i in range(0, y.shape[0]):
        dmin=np.zeros(x.shape[0])
        
        for j in range(0, x.shape[0]):
            dij = np.linalg.norm(x[j]-y[i])    
            dmin[j] = dij
        xpoints.append(dmin.argmin())
    return np.array(xpoints)
    
"""Voronoi plot for points on a sphere"""    
s = 300     
phi, theta = np.linspace(0,2*np.pi,s),np.linspace(0,np.pi,s)
PHI,THETA = np.meshgrid(phi,theta)
xs = radius*np.sin(THETA)*np.cos(PHI)
ys = radius*np.sin(THETA)*np.sin(PHI)
zs = radius*np.cos(THETA)
P = np.zeros((PHI.size,3))
P[:,0] = xs.flatten()
P[:,1] = ys.flatten()
P[:,2] = zs.flatten()

"""Voronoi plot for points in a disc"""
"""
s = 300
phi, theta = np.linspace(0,2*np.pi,s), np.linspace(0, np.pi,s)
PHI, THETA = np.meshgrid(phi, theta)
xs = np.sin(THETA)*np.cos(PHI)
ys = np.sin(THETA)*np.sin(PHI)

P = np.zeros((PHI.size,2))

P[:,0] = xs.flatten()
P[:,1] = ys.flatten()
"""


colors = find(x, P)

color_dimension = colors 
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

"""Plotting the sphere"""
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
ax = plt.axes(projection='3d')
ax.scatter(P[:,0],P[:,1],P[:,2], c=colors,cmap='rainbow', s = 5 )
plt.axis('off')
ax.view_init(elev=15.0,azim =25.0)

b = 2.6
ax.set_xlim([-b,b])
ax.set_ylim([-b,b])
ax.set_zlim([-b,b])
ax.set_aspect("equal")
ax.set_title("{0} ".format(x.shape[0])+"points on a sphere")

plt.savefig('BallVP_%s.png' % (n),dpi=300,format='png')
#pickle.dump(fig,file('VP12plot.pickle', 'wb'))

plt.show()

"""For the disc"""
"""
x1=[]
x2=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    
    

#Render
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(P[:,0],P[:,1], c=colors,cmap='jet', s=1)
plt.axis('off')
ax.set_aspect('equal')


ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])

ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")
plt.show()
"""
