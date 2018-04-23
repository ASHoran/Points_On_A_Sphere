import numpy as np
import random
import time
import sys

start = time.time()
#n = 13

#Random distribution of points in the range [-1,1] for each component x,y,z.
#randstart = (2.0*np.random.random((n,3))-1.0)


"""Calculates the energy of a system x"""

def calcenergy(x):
    ballenergy = 0.0
    pointsenergy = 0.0
    for i in range (0, x.shape[0]):
        ballenergy = ballenergy + 0.5*np.dot(x[i],x[i])
        for j in range (i+1, x.shape[0]):
            pointsenergy = pointsenergy + 1/(np.linalg.norm(x[i]-x[j]))
    
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
                Fij=(x[i]-x[j])/(np.sqrt(sum((x[i]-x[j])**2)))**3    
                forces[j]=Fij                                        
        
        resultforce=np.sum(forces,axis=0)
        resultforce = resultforce - x[i]                                 
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
        #if (loop%100 == 0):
            #print loop, energy
        if(difference>0.0):
            x=old
            energy=oldenergy.copy()
            amplitude=amplitude/2.0                                  
        
        if (abs(difference) < 1e-8):
            return x, energy
    return x, energy
        
#x, energy = relax(randstart, 100, 0.005)    

def rotation(d):
    u = np.concatenate(np.random.rand(1,3))
    u = u/np.dot(u,u)
    v = np.concatenate(np.random.rand(1,3))
    v = v/np.dot(v,v)
    utilda = u - np.dot(v,u)*v/np.dot(v,v)
    w = np.cross(v, utilda)
    rotmatrix = np.array([[utilda[0], v[0], w[0]], [utilda[1], v[1], w[1]], [utilda[2], v[2], w[2]]])
    d = np.dot(rotmatrix, d.T).T
    return d
        


                                
def half(d):
    while True:
        #d = rotation(d)
        top = []
        bottom = []
        for i in range(0, d.shape[0]):
            if d[i,2] > 0.0:
                top.append(d[i,:])
            else:
                bottom.append(d[i,:])
        if len(top) > 0.0 and len(bottom) > 0.0:
            break
        else:
            pass
    return top, bottom 
    

def pair(p, c):                         
    l = 0.0
    childrenlist = []
    while (l in range(0,c)):
        while True:
            i = random.randint(0,len(p)-1)
            itop, ibottom = half(p[i])
            j  = random.randrange(0,len(p)-1)
            jtop, jbottom = half(p[j])
            if (len(itop) + len(jbottom)) == len(p[i]):
                break
            else:
                pass
        
        ijpair = np.concatenate((itop, jbottom), axis=0)
        
        childrenlist.append(ijpair)
        l = l+1
    return childrenlist


def relaxall(c):
    relaxed = []
    relaxedenergy = []
    for i in range(0,len(c)):
        newc, newenergy = relax(c[i], 10000,0.05)
        relaxed.append(newc)
        relaxedenergy.append(newenergy)
    return relaxed, relaxedenergy


def smallest(p,e,numbermin):
    l = 0.0
    minenergy = []
    minpoints = []
    #energies = e
    energies = []
    for j in range(0, len(e)):
        energies.append(e[j])
    
    points = p
    diff = 0.01
    
    while l != numbermin:
        if (len(energies) == 0):
            return points[np.argmin(e)] , np.min(e)
        
        minimum = np.min(energies)
        i  = np.argmin(energies)
        #print i

        if (abs(minimum - diff)> 1e-6):
            minenergy.append(minimum)
            minpoints.append(points[i])
            np.delete(points, points[i])
            energies.remove(energies[i])
            diff = minimum
            l = l + 1
        
        else:
            np.delete(points, points[i])
            energies.remove(energies[i])
        
        
    return minpoints, minenergy

def genetic(n,p,generations):
    #Create p random starting points
    k = 0.0
    parents = []
    parentenergy = []
    while k < p :
        rand = (2.0*np.random.random((n,3))-1.0)
        relaxedrand, energyrelaxedrand = relax(rand,10000,0.05)
        parents.append(relaxedrand)
        parentenergy.append(energyrelaxedrand)
        k = k + 1
    
    for loop in range (0, generations+1):
        children = pair(parents,16)
        children, childrenenergy = relaxall(children)
        
        firstgen = np.concatenate((parents, children), axis = 0)
        firstgenenergy = np.concatenate((parentenergy, childrenenergy), axis = 0)
        
        secondgen, secondgenenergy = smallest(firstgen, firstgenenergy, 4)       # need to change number of new parents depending on how many minima there are for n
        print firstgenenergy
        print secondgen
        print secondgenenergy
         
        if (np.size(secondgenenergy) == 1 ):
            return secondgen, secondgenenergy        
            
        elif (np.min(firstgenenergy) == np.min(secondgenenergy)):
            m = np.argmin(secondgenenergy)
            return secondgen[m], np.min(secondgenenergy)              
                   
        elif (loop == generations):
            m = np.argmin(secondgenenergy)
            return secondgen[m], np.min(secondgenenergy)          
        
        else:
            parents = secondgen
            parentenergy = secondgenenergy
  
#print genetic(13,4,5)    

def calcdiameter(x):
    d = 0.0
    dist = []
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            dist.append(np.linalg.norm(x[i]-x[j]))
        a = max(dist)
        if (a > d):
            d = a
    return d

n = int(sys.argv[1])

e = open('BallPoints_%s.txt' % (n), 'a') 
f = open('BallEnergy_%s.txt'%(n), 'a')
g = open('BallDiameters_%s.txt'%(n), 'a')
points, minenergy = genetic(n, 4, 5)
diameter = calcdiameter(points)
    
e.write('%s \n' % n)
for i in range(0,n): e.write('%s %s %s \n' % (points[i,0],points[i,1],points[i,2]))
e.close()


f.write('%s %s \n' % (n, minenergy))
f.close()


g.write('%s %s \n' % (n, diameter))
g.close()