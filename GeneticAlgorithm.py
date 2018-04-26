import numpy as np
import random
from datetime import datetime

start=datetime.now()


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
        if (amplitude < 1e-6):
            return x, energy
    return x, energy


def rotation(d):
    u = np.concatenate(np.random.rand(1,3))
    u = u/np.dot(u,u)
    v = np.concatenate(np.random.rand(1,3))
    v = v/np.dot(v,v)
    utilda = u - np.dot(v,u)*v/np.dot(v,v)
    w = np.cross(v, utilda)
    rotmatrix = np.array([[utilda[0], v[0], w[0]], [utilda[1], v[1], w[1]], [utilda[2], v[2], w[2]]])
    d = np.dot(rotmatrix, d.T).T
    d = proj(d, d.shape[0])
    return d
        
                
def half(d):
    while True:
        d = rotation(d)
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
    
def pair(p, c):                         #p is the list of parent arrays, c is the number of new arrays we want to make
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
    # n = number of points on sphere
    # p = number of random parent structures to start with
    # generations = number of generations to run the algorithm for
    
    k = 0.0
    parents = []
    parentenergy = []
    while k < p :
        rand = proj((2.0*np.random.random((n,3))-1.0),n)
        rand, energyrand = relax(rand,10000,0.05)
        parents.append(rand)
        parentenergy.append(energyrand)
        k = k + 1
    
    for loop in range (0, generations+1):
        children = pair(parents,16)
        children, childrenenergy = relaxall(children)
        
        firstgen = np.concatenate((parents, children), axis = 0)
        firstgenenergy = np.concatenate((parentenergy, childrenenergy), axis = 0)
        
        secondgen, secondgenenergy = smallest(firstgen, firstgenenergy, 4)                   
        print ("Run time = {0}\n".format((datetime.now()-start).total_seconds()))
        
        if type(secondgenenergy)==float:
            return secondgen, secondgenenergy
        
                   
        elif (loop == generations):
            m = np.argmin(secondgenenergy)
            return secondgen[m], np.min(secondgenenergy)
                 
        
        else:
            parents = secondgen
            parentenergy = secondgenenergy
        
e = open('GA5points.txt', 'a') 
f = open('GA5energy.txt', 'a')
                        
for n in range(46, 101):
    e = open('GA5points.txt', 'a')
    print n
    print genetic(n, 4, 5)
    points, minenergy = genetic(n, 4, 5)
    
    e.write('%s \n' % n)
    for i in range(0,n): e.write('%s %s %s \n' % (points[i,0],points[i,1],points[i,2]))
    e.close()
    
    f = open('GA5energy.txt', 'a')
    f.write('%s %s \n' % (n, minenergy))
    f.close()

    
    
    
                
            
    
        
    
    
    
    
    
    
print("Run time = {0}\n".format((datetime.now()-start).total_seconds()))
