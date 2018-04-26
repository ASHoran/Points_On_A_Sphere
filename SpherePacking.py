import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

x = np.genfromtxt('38pointswith5charge.txt')
n = x.shape[0]


#x = np.genfromtxt('25PointsAndPointOfCharge2.txt')
n = x.shape[0]
charge = 21    #Position in array of point with charge = 2

#charge = 600
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

def radiuscalc(x):
    r = 0.0
    dist = []
    for i in range(0, x.shape[0]): 
        dist.append(np.linalg.norm(x[i]))
    a = max(dist)
    return a

radius = 0.5*diameter(x)
print radius
def distmin(x):
    k = 1000.0
    dist = []
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            if j!=i:
                dist.append(np.linalg.norm(x[i]-x[j]))
        a = min(dist)
        if (a < k):
            k = a
    return k

def rings(x):
    rad = radiuscalc(x)
    ring1 = []
    ring2 = []
    colourlist = []
    for k in range(0, x.shape[0]):
        if k!= charge:
            if (abs(np.linalg.norm(x[k]) - rad) < 0.5):
                ring1.append(x[k])
                colourlist.append(0.0)
                
            elif (abs(np.linalg.norm(x[k]) - rad) > 1.0):
                #print abs(np.linalg.norm(x[k]) - rad)
                ring2.append(x[k])
                colourlist.append(1.0)
    return np.array(ring1), np.array(ring2), colourlist

ring1, ring2, colors = rings(x)
print ring1, ring2
      
    
kmin = distmin(x)   
sphereradii = 0.5*kmin 
#print kmin          
#Create spheres of radius of smalled dist between two points aboout every point

u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, np.pi, 20)
xx = np.outer(np.cos(u), np.sin(v))
yy = np.outer(np.sin(u), np.sin(v))
zz = np.outer(np.ones(np.size(u)), np.cos(v))


"""
#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = radius*np.sin(theta)*np.cos(phi)
ys = radius*np.sin(theta)*np.sin(phi)
zs = radius*np.cos(theta)
"""
#print len(xs)
#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,x.shape[0]):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
#ax = plt.axes(projection='3d')
#ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
#ax.plot(x0,y0,z0, color = 'red' )

for k in range(0, ring1.shape[0]):
    r = sphereradii
    ax.plot_surface(r*xx +ring1[k,0], r*yy + ring1[k,1], r*zz + ring1[k,2], color = 'blue', alpha = 1.0, linewidth = 0)
for k in range(0, ring2.shape[0]):
    r = sphereradii
    ax.plot_surface(r*xx +ring2[k,0], r*yy + ring2[k,1], r*zz + ring2[k,2], color = 'yellow', alpha = 1.0, linewidth = 0)    

rcharge = 2**(1/3.0)
ax.plot_surface(rcharge*xx + x[charge,0], rcharge*yy + x[charge,1], rcharge*zz + x[charge,2], color = 'green', alpha = 1.0, linewidth = 0)    

#ax.scatter(x1, x2, x3, color="black",s=170)
plt.axis('off')
ax.view_init(elev=6.0,azim=126.0)
k = 0.2
ax.set_xlim([-radius-k, radius+k])
ax.set_ylim([-radius-k, radius+k])
ax.set_zlim([-radius-k, radius+k])
ax.set_aspect("equal")


picname = '0spherepack_%s.png' % x.shape[0]

plt.savefig(picname, dpi = 300, format = 'png')
plt.show()