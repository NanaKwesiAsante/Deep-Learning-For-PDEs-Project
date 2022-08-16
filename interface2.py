# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:30:57 2021

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


def main():
    def pde(x, y):
    # Here y1 = y[:, 0:1], y2 = y[1:2]
        y1 = y[:, 0:1], y2 = y[:,1:2]
    # compute the two residuals for y1 and y2, similar as your code pde1 and pde2
        beta1=1
        dy_xx1 = dde.grad.hessian(y, x,components=0, i=0, j=0)
        dy_yy1 = dde.grad.hessian(y, x, components=0,i=1, j=1)
        fct1=(3*(2*np.power(x[:,0],2)+np.power(x[:,1],2))/np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2)))
        error1 = -(dy_xx1 + dy_yy1 )*beta1+fct1
        
        beta2=1
        dy_xx2 = dde.grad.hessian(y, x,components=1, i=0, j=0)
        dy_yy2 = dde.grad.hessian(y, x, components=1,i=1, j=1)
        fct2=(3*(np.power(x[:,0],2)+2*np.power(x[:,1],2))/np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2)))
        error2 = -(dy_xx2 + dy_yy2 )*beta2+fct2
    # indicate if x is inside the disk
        r2 = tf.sum(tf.square(x), axis=1, keep_dim=True) # r**2 for each x
        is_inside = tf.math.less(r2, 0.5**2)  # True --> inside, False --> outside
    # For x inside, use error1; for x outside, use error2
        return error1 * is_inside, error2 * (1 - is_inside)
    
    geom = dde.geometry.Rectangle([-1,-1], [1,1])    
    
    def boundary(x, on_boundary):
        return on_boundary
    #on the boundary rectangle
    def func1(x):
        beta2=1
        beta1=1
        r=np.sqrt(np.array(x[:,0])**2+np.array(x[:,1])**2)
        ro=0.4
        return ((r**3)/beta2)+((1/beta1)-(1/beta2))*ro**3
    
    # the 1st function on the interface which is the circle
    def on_circle(x, _):
        r2 = x[0] ** 2 + x[1] **2
        return np.isclose(r2, 0.5**2)

    def func2(x, y, X):
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]
        ro=0.4
        beta2=1 
        beta1=1
        r=np.sqrt(np.array(X[:,0])**2+np.array(X[:,1])**2)
        return y2 - y1 - (beta2*(r**3-ro**3)+beta1(ro**3-r**3))/(beta1*beta2)
    
    
    
    #normals
     #normalx
    def nx(x):
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        return (-x[:,1]/r)
    #normaly
    def ny(x):
        r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
        return (x[:,0]/r)
    
    
   # the 2nd function on the interface which is the circle(flux)
    def func3(x, y, X):
        #y1 = y[:, 0:1]
        #y2 = y[:, 1:2]
        #ro=0.4
        du1 = dde.grad.jacobian(y, x, i=0,j=1)
        du2=dde.grad.jacobian(y, x, i=0,j=1)
        beta2=1 
        beta1=1
        r=np.sqrt(np.array(X[:,0])**2+np.array(X[:,1])**2)
        return ((beta1*du1*np.array([nx,ny]))-(beta2*du2*np.array([nx,ny])))-(((beta1*3*np.array(x[:,0])*r)*np.array([nx,ny]))-((beta2*3*np.array(x[:,1])*r)*np.array([nx,ny])))
    bc1 = dde.DirichletBC(geom,  func1, boundary,component=1)
    bc2 = dde.bc.OperatorBC(geom, func2, on_circle)
    bc3 = dde.bc.OperatorBC(geom, func3, on_circle)
    
    # Generate manually some points on the circle; a Numpy array of size N x 2
    N=50
    r=0.5
    theta = np.linspace(0, 2*np.pi, N) 
    ox, oy = r * np.cos(theta), r * np.sin(theta)
    X=np.transpose([ox,oy])
    #dde.data.PDE(..., anchors=X)  
    data = dde.data.PDE(geom, pde, [bc1,bc2,bc3], num_domain=50, num_boundary=10, num_test=100,anchors=X)
    net=dde.maps.FNN([2] + [64] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=50000)
    #model.compile("L-BFGS-B") 
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
  
    
    
if __name__ == "__main__":
    main()    