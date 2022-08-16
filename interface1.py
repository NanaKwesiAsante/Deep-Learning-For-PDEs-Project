# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:50:56 2021

@author: Nana Kwesi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import math
import deepxde as dde
from deepxde.backend import tf


#for domain 1 circle
def main():
    def pde1(x, y):
        beta1=1
        dy_xx1 = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy1 = dde.grad.hessian(y, x, i=1, j=1)
        fct=(3*(2*np.power(x[:,0],2)+np.power(x[:,1],2))/np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2)))
        return -(dy_xx1 + dy_yy1 )*beta1+fct

    def boundary(_, on_boundary):
        return on_boundary
    
    #for domain 2 rectangle
    def pde2(x, y):
        beta2=1
        dy_xx2 = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy2 = dde.grad.hessian(y, x, i=1, j=1)
        fct1=(3*(np.power(x[:,0],2)+2*np.power(x[:,1],2))/np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2)))
        return -(dy_xx2 + dy_yy2 )*beta2+fct1

#domain 1 geom1
    def boundary(_, on_boundary):
        return on_boundary
    geom1 = dde.geometry.Disk([0,0], 0.5)
    
    #domain 2 geom
    geom22 = dde.geometry.Rectangle([-1,-1], [1,1])
    geom2= dde.geometry.CSGDifference(geom22, geom1)
    
    #Gj
    def func1(x):
        beta2=1
        beta1=1
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        ro=0.4
        return (beta2*(r**3-ro**3)+beta1(ro**3-r**3))/(beta1*beta2)
    
     #normalx
    def nx(x):
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        return (-x[:,1]/r)
    #normaly
    def ny(x):
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        return (x[:,0]/r)
    
    #Gf
    def func2(x):
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        return ((3*np.array(x[:,0])*r)*np.array([nx,ny]))-((3*np.array(x[:,1])*r)*np.array([nx,ny]))
    
    #Gd domain 2
    def func3d2(x):
        beta2=1
        beta1=1
        r=np.sqrt(np.array(x[:,0])**2+np.array(x[:,1])**2)
        ro=0.4
        return ((r**3)/beta2)+((1/beta1)-(1/beta2))*ro**3
    
    bc1n = dde.DirichletBC(geom1, lambda x: func1, boundary)
    bc1nn = dde.NeumannBC(geom1, lambda x: func2, boundary)
    
    bc2d = dde.DirichletBC(geom2, lambda x:func3d2, boundary)
    
    #domain 1
    data1 = dde.data.PDE(geom1, pde1, [bc1n,bc1nn], num_domain=50, num_boundary=10, num_test=100)
    net1 = dde.maps.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")
    #model1 = dde.Model(data1, net1)
    
    #domain 2
    data2 = dde.data.PDE(geom2, pde2, bc2d, num_domain=50, num_boundary=10, num_test=100)
    net2 = dde.maps.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model([data1,data2], [net1,net2])
    model.compile("adam", lr=0.001)
    model.train(epochs=50)
    losshistory, train_state = model.train()
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()