import numpy as np
import random
import sys

n = 31


x = np.genfromtxt('GFpoints_%s.txt' % (n))
print n


"""Calculates the energy of the array x"""
def calcenergy(x):
    energy=0.0
    for i in range(0,x.shape[0]):
        for j in range(i+1, x.shape[0]):
            distance = np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    return energy

#print calcenergy(x)

"""Compares whether two arrays of points x and y are the same (the rows can be in a different order)"""
def compare(x,y):
    d = 0.0
    for i in range(0, n):
        dmin=100.0
        for j in range(0,n):
            dij = ((x[i,0]-y[j,0])**2 + (x[i,1]-y[j,1])**2 + (x[i,2]-y[j,2])**2)**0.5    
            if (dmin > dij):
                dmin = dij
        d = d + dmin
    return d

"""Calculate the dipole moment of a configuration x"""
def dipole(x):
    dipole = np.sum(x, axis=0)
    d = (dipole[0]**2 + dipole[1]**2 + dipole[2]**2)**0.5
    return d

"""Calculate the moment of inertia tensor for a confguration x, then return it's eigenvalues and eigenvectors"""
def MOIT(x):
    xsquare=x**2
    Ixx = sum(xsquare[:,1]) + sum(xsquare[:,2])
    Iyy = sum(xsquare[:,0]) + sum(xsquare[:,2])
    Izz = sum(xsquare[:,0]) + sum(xsquare[:,1])
    Ixy = -sum(x[:,0]*x[:,1])
    Ixz = -sum(x[:,0]*x[:,2])
    Iyz = -sum(x[:,1]*x[:,2])
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    Ieigvals, Ieigvecs = np.linalg.eig(I)
    return Ieigvals, Ieigvecs

"""Find which of the eigenvalues of the moment of inertia tensor of x are equal,
stores the result as an array so we know where the axis of rotation lies"""
def scount(x):
    Ieigvals, Ieigvecs = MOIT(x) 
    s = np.array([0,0,0])
    if (abs(Ieigvals[0] - Ieigvals[1]) < 1e-5):
        s = s+np.array([1,0,0])
    
    if (abs(Ieigvals[0] - Ieigvals[2]) < 1e-5):
        s = s+np.array([0,1,0])
    
    if (abs(Ieigvals[1] -Ieigvals[2]) < 1e-5):
        s=s+np.array([0,0,1])
    return s
        
s = scount(x)

Ieigvals, Ieigvecs = MOIT(x)
#print Ieigvals
 
rotx= np.zeros((n,3))
for i in range (0, n):
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


"""Function to rotate the points so that they are in terms of the principal axis
of the moment of inertia tensor, then change the points so that the z-axis is the 
rotation axis"""
def principal(x):
    Ieigvals, Ieigvecs = MOIT(x)
    newx = np.zeros((n,3))
    for i in range (0, n):
        newx[i] = newx[i] + np.dot(Ieigvecs.T, x[i])
    s = scount(x)
    if (np.count_nonzero(s)==1):
        if (np.array_equal(s, [0,0,1]) == True ):
            temp = np.copy(newx[:,2])
            newx[:,2] = np.copy(newx[:,0])
            newx[:,0] = temp
            
        if (np.array_equal(s, [0,1,0]) == True ):
            temp = np.copy(newx[:,2])
            newx[:,2] = np.copy(newx[:,1])
            newx[:,1] = temp 
    return newx

"""Function to find horizontal reflective symmetry plane for x"""
def horiz(x):
    x  = principal(x)
    savex = np.zeros((n,3))
    savex = x.copy()
    j = 0.0
    reflection =  np.array([[1,0,0],[0,1,0],[0,0,-1]])
    newx = np.dot(savex, reflection)
    xydiff = compare(newx, savex)
    if (xydiff < 1e-4):
        return 'horizontal symmetry'
    else:
        return 'X' 


"""Function to find vertical reflective symmetry plane for x"""
def vert(x):
    x = principal(x)
    savex = np.zeros((n,3))
    savex = x.copy()
    j = 0.0
    while j in range(0,1):
        while True:
            q = j*np.pi
            rotation = np.array([[-np.cos(q),np.sin(q), 0], [np.sin(q),np.cos(q), 0], [0,0,1] ])
            x = np.dot(rotation, savex.T).T
            a = x[:,0]
            b = x[:,1]
            c = x[:,2]
            #b = -1*b
            newx = np.column_stack((a,-1*b,c))
            
            xydiff = compare(newx, savex)
            #print j
            if (xydiff < 1e-4) and (j < 1.0):
                return 'vertical symmetry'
            elif (j >= 1):
                break
            else:
                j = j + 0.001


"""Finds if a configuration x, has dihedral symmetry"""
def dihedral(x):
    x = principal(x)
    savex = np.zeros((n,3))
    savex = x.copy()
    j = 0.0
    while (j in range(0,1)):
        while True:       
            #print j
            q = 2*j*np.pi
            rotation = np.array([[1,0,0],[0,np.cos(q),-np.sin(q)],[0, np.sin(q), np.cos(q)]])
            x2 = np.dot(rotation, savex.T).T
            a = x2[:,0]
            b = x2[:,1]
            c = x2[:,2]
            b = -b
            c = -c
            newx = np.column_stack((a,b,c))
            
            xydiff = compare(newx, savex)
            #print xydiff
            
            if (xydiff < 1e-4):
                return 'Diehedral symmetry'
            elif (j>=1):
                break
            else:
                j = j + 0.0005


"""Program exits if it finds platonic symmetry where three eigenvalues are equal"""
sval = np.sum(s, axis = 0)
if sval == 3:
    print 'Platonic'
    #sys.exit()


"""Searches for cyclic symmetry if there are exactly two equal eigenvalues"""
if sval == 1:
    print 'At least C3 symmetry'
    savex = np.zeros((n,3))
    savex = x.copy()
    xydiff = 10.0
    j = 13
    while (xydiff > 1e-4) and (j>=2) and (np.count_nonzero(s)==1):
        j=j-1
        q = 2*np.pi/j
        rotation = np.array([[np.cos(q),-np.sin(q),0],[np.sin(q),np.cos(q),0],[0,0,1]])
        x = np.dot(savex, rotation)
        xydiff = compare(x, savex)
        
    print 'C%s symmetry' % j
    print horiz(x)
    if (horiz(x) == 'horizontal symmetry') == True:
        sys.exit()
    
    print vert(x)
     
    #cannot have dihedral symmetry if the dipole moment is non-zero       
    d = dipole(x)
    if abs(d) > 1e-6:
        sys.exit()
    print dihedral(x)
    
if sval == 0:
    print 'At most C2 or D2 symmetry'
    print cyclic(x, 2)
    print vert(x)
    print horiz(x)
    d = dipole(x)
    if abs(d) > 1e-6:
        sys.exit()     
    print dihedral(x)
    #sys.exit()            
                           
                                
                                    
                                        
                                                
            