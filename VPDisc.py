import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

x = np.genfromtxt('ADisc_58.txt')


def find(x,y):
    #X is the points on sphere, want to find which x any random y is closest to
    xpoints = []
    for i in range(0, y.shape[0]):
        dmin=np.zeros(x.shape[0])
        
        for j in range(0, x.shape[0]):
            dij = ((x[j,0]-y[i,0])**2 + (x[j,1]-y[i,1])**2)**0.5    
            dmin[j] = dij
        xpoints.append(dmin.argmin())
    return np.array(xpoints)



#Plot the disc boundary
s = 300
phi, theta = np.linspace(0,2*np.pi,s), np.linspace(0, np.pi,s)
PHI, THETA = np.meshgrid(phi, theta)
xs = np.sin(THETA)*np.cos(PHI)
ys = np.sin(THETA)*np.sin(PHI)

P = np.zeros((PHI.size,2))

P[:,0] = xs.flatten()
P[:,1] = ys.flatten()

colors = find(x, P)

color_dimension = colors # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='Set1')
m.set_array([])
fcolors = m.to_rgba(color_dimension)


#convert data
x1=[]
x2=[]
for i in range(0,x.shape[0]):
    x1.append(x[i,0])
    x2.append(x[i,1])
    
    

#Render
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(P[:,0],P[:,1], c=colors,cmap='gist_ncar', s=5)
plt.axis('off')
ax.set_aspect('equal')


ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])

ax.set_aspect("equal")
ax.set_title("{0} ".format(x.shape[0])+"points on a sphere")
plt.savefig('VoronoiDisc60.png', dpi = 300, format = 'png')

plt.show()
