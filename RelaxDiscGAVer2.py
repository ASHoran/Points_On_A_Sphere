import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

start = datetime.now()

#If any point moves off the disc, projects it back onto the disc boundary
def bounded(x):
    i= 0
    while (i < x.shape[0]):
        if (np.sqrt(sum(x[i]**2)) > 1.0):
            norm = np.sqrt(sum(x[i]**2))
            x[i] = x[i]/norm
            i = i + 1
        else:
            x[i]=x[i]
            i =i + 1
    return x

#Find a random starting configuration of n points on a unit disc    
def randstart(n):
    x = np.zeros((n, 2))
    randlist = []
    l = 0.0
    while  (l<n):
        rand = (2.0*np.random.random((1,2))-1.0)
        if ((rand[0,0]**2 + rand[0,1]**2)**0.5 < 1.0):
            randlist.append(rand)
            rand = (2.0*np.random.random((1,2))-1.0)
            l = l+1
    for i in range(0, n):
        x[i] = x[i] + randlist[i]        
    return x

"""Given an n function returns an n x 2 array of points of unit length.
Randlist creates a list of n vectors of unit length and reject ones that are not.
This list is added to an empty n x 2 array.
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
        if(difference>0.0):
            x=old
            energy=oldenergy.copy()
            amplitude=amplitude/2.0                                  
        if (amplitude < 1e-8):
            return x, energy
    return x, energy

def rotation(d):
    u = np.concatenate(np.random.rand(1,2))
    u = u/np.dot(u,u)
    v = np.concatenate(np.random.rand(1,2))
    v = v/np.dot(v,v)
    utilda = u - np.dot(v,u)*v/np.dot(v,v)
    rotmatrix = np.array([[utilda[0], v[0]], [utilda[1], v[1]]])
    d = np.dot(rotmatrix, d.T).T
    d = bounded(d)
    return d

"""Finds a random rotation matrix by forming u, v, and utilda. v and utilda are
orthogonal so we use these for the rotation matrix
"""

def half(d):
    while True:
        d = rotation(d)
        top = []
        bottom = []
        for i in range(0, d.shape[0]):
            if d[i,1] > 0.0:
                top.append(d[i,:])
            else:
                bottom.append(d[i,:])
        if len(top) > 0.0 and len(bottom) > 0.0:
            break
        else:
            pass
    return top, bottom 

"""Given an array d halves by taking points with y>0 and making the list top 
from these points. All other points ( those with y<=0 are added to the list bottom
"""
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
  
"""p is the list of parent arrays, c is the number of new arrays we want 
  to make. We take two random arrrays from the list p and pair one top with the 
  other bottom. If we have the right number of points:
      add the new array to childrenlist
If not:
    try again
End when we have c arrays in childrenlist
  """   
def relaxall(c):
    relaxed = []
    relaxedenergy = []
    for i in range(0,len(c)):
        newc, newenergy = relax(c[i], 10000,0.05)
        relaxed.append(newc)
        relaxedenergy.append(newenergy)
    return relaxed, relaxedenergy

"""Given a list of arrays, c, takes each array in c and relax using the relaxation
method above. Create a new list of the relaxed arrays and a list of the minimum 
energy of each relaxed array
"""

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
        rand = randstart(n)
        rand, energyrand = relax(rand,10000,0.05)
        parents.append(rand)
        parentenergy.append(energyrand)
        k = k + 1
    
    for loop in range (0, generations+1):
        children = pair(parents,16)
        children, childrenenergy = relaxall(children)
        
        firstgen = np.concatenate((parents, children), axis = 0)
        firstgenenergy = np.concatenate((parentenergy, childrenenergy), axis = 0)
        
        secondgen, secondgenenergy = smallest(firstgen, firstgenenergy, 4)                       # need to change number of new parents depending on how many minima there are for n
        #return firstgenenergy
        #return secondgenenergy
        print "firstgen", firstgen, firstgenenergy
        print 'secondgen', secondgen, secondgenenergy
        
        print ("Run time = {0}\n".format((datetime.now()-start).total_seconds()))
        
        if (np.size(secondgenenergy) == 1 ):
            return  secondgen, secondgenenergy
            
        """elif (np.min(firstgenenergy) == np.min(secondgenenergy)):
            return  np.min(secondgenenergy)"""
             
        if (loop == generations):
            m = np.argmin(secondgenenergy)
            return secondgen[m], np.min(secondgenenergy)
            
        else:
            parents = secondgen
            parentenergy = secondgenenergy

e = open('Discpoints.txt', 'w+') 
f = open('Discenergy.txt', 'w+')
                        
for n in range(12, 31):
    e = open('Discpoints.txt', 'a')
    print n
    print genetic(n, 4, 5)
    points, minenergy = genetic(n, 4, 5)
    
    e.write('%s \n' % n)
    for i in range(0,n): e.write('%s %s \n' % (points[i,0],points[i,1]))
    e.close()
    
    f = open('Discenergy.txt', 'a')
    f.write('%s %s \n' % (n, minenergy))
    f.close()