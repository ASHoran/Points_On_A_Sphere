import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.colors
start=datetime.now()

# set the number of points
n=32

# set the number of loops
loops=10000

# set how often to output the energy 
often=100

def proj(x,j):                           
    if(j==n):
        for i in range(0,x.shape[0]):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm                
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm 
    return x
    
 
    
# set the maximum amplitude of the change
amplitude=0.05
"""Store the energy each time then if energy goes up change the step size to smaller, e.g half the
step size when the energy goes up"""
 
# open file to output energy during minimization 
out=open('out','w')
out.close()

looplist=[]
energylist=[]

#find random start positions for the n points on the sphere
random.seed()
x=proj((2.0*np.random.random((n,3))-1.0),n)

# calculate the initial energy
energy=0.0
for i in range(0,n):
    for j in range(i+1,n):
        distance=np.sqrt(sum((x[i]-x[j])**2))
        energy=energy+1.0/distance


#the main loop to reduce the energy  
for loop in range(0,loops+1):
    resultforceslist=[]                                              #creates an empty list that the resultant forces for each point i will be added to
    old=np.zeros((n,3))
    oldenergy=energy.copy()                                          #store the old energy of the system

    for i in range(0,n):
        forces=np.zeros((n,3))                                       #makes an n x 3 matrix that the forces on i due to each j will be added to.
        for j in range(0,n):
            if j!=i:
                Fij=(x[i]-x[j])/(np.sqrt(sum((x[i]-x[j])**2)))**3    #Fij is the force on i due to j
                forces[j]=Fij                                        #adds Fij (the force on i due to j) to the matrix of forces on i
        
        resultforce=np.sum(forces,axis=0)                            #finds the resultant force on i by summing the columns of the matrix forces, axis=0 tells it to sum columns     
        resultforce=resultforce-np.dot(resultforce,x[i])*x[i]        #Remove force moving away from the sphere            
        resultforceslist.append(resultforce)                         #adds the resultant force on i to the list 
        old[i]=x[i].copy()                                           # store the old coordinates of this point              
        x[i]=x[i]+amplitude*resultforceslist[i]                      #move every point in the direction of its resultant force and renormalize
        #print(x[i],old[i])

    x=proj(x,n) 
    
    
    #Calculate new energy
    energy=0.0
    for i in range(0,n):
        for j in range(i+1,n):
            distance=np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
            
    #Calculate the difference in energy
    difference=0.0
    difference=energy-oldenergy
    if(difference>0.0):
        x=old
        energy=oldenergy.copy()
        amplitude=amplitude/2.0                                  #if the energy increases halve the step size
    if (amplitude < 1e-6):                         #When the diapole moment is near to 0 stop the loop
        break
    
 
    
    # output energy to screen and a file
    
    if(loop%often==0):
        print("{0} {1:.6f}".format(loop,energy))
        out = open('out','a')
        out.write("{0} {1:.6f} \n".format(loop,energy))
        out.close()
        looplist.append(loop)
        energylist.append(energy)


 #output final energy to the screen and points to a file
print("Final energy = {0:.6f} \n".format(energy))


def find(x,y):
    """X is the points on sphere, want to find which x any random y is closest to"""
    xpoints = []
    for i in range(0, y.shape[0]):
        dmin=np.zeros(n)
        
        for j in range(0, n):
            dij = ((x[j,0]-y[i,0])**2 + (x[j,1]-y[i,1])**2 + (x[j,2]-y[i,2])**2)**0.5    
            dmin[j] = dij
        xpoints.append(dmin.argmin())
    return np.array(xpoints)
    
    
s = 300
phi, theta = np.linspace(0,2*np.pi,s),np.linspace(0,np.pi,s)
PHI,THETA = np.meshgrid(phi,theta)
xs = np.sin(THETA)*np.cos(PHI)
ys = np.sin(THETA)*np.sin(PHI)
zs = np.cos(THETA)
P = np.zeros((PHI.size,3))
P[:,0] = xs.flatten()
P[:,1] = ys.flatten()
P[:,2] = zs.flatten()

colors = find(x, P)

color_dimension = colors # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
m.set_array([])
fcolors = m.to_rgba(color_dimension)


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
ax = plt.axes(projection='3d')
ax.scatter(P[:,0],P[:,1],P[:,2], c=colors,cmap='jet', s=1)
plt.axis('off')
ax.view_init(elev=90.0,azim=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")

plt.savefig('Voronoi32.png',dpi=300,format='png')

plt.show()