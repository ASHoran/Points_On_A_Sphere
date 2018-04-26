import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time

n = 7

#Projects the points onto the surface of the sphere
def proj(x,j):                           
    if(j==x.shape[0]):
        for i in range(0,x.shape[0]):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm                
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm 
    return x

#Calculates the energy of a system of points
def calcenergy(x):
    energy=0.0
    for i in range(0,x.shape[0]):
        for j in range(i+1, x.shape[0]):
            distance = np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    return energy

#Calcluates the resultant forces on each point and moves them in this direction by an amount, amplitude
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

#Relaxes the energy of a configuration x, for a given number of iterations or until amplitude is below a given tolerence
def relax(x,loops,amplitude):
    start = time.time()      
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
        
        if (loop % 100 == 0):
            print loop, energy    
                   
    return x, energy
    
    
#A random starting configuration of n points in a 3x3 cube about the origin, then projected onto the sphere
randstart = proj((2.0*np.random.random((n,3))-1.0),n)

x , energy = relax(randstart,20000, 0.05)

print x, energy
  
                        
                                                
