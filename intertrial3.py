# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:23:49 2021

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
        
        dy_xx1 = dde.grad.hessian(y, x,component=0, i=0, j=0)
        dy_yy1 = dde.grad.hessian(y, x, component=0,i=1, j=1)
        dy_xx2 = dde.grad.hessian(y, x,component=1, i=0, j=0)
        dy_yy2 = dde.grad.hessian(y, x, component=1,i=1, j=1)
        
        #u1
        beta1=1
        fct1=(3*(2*x[:,0:1]**2+x[:,1:2]**2)/tf.sqrt(x[:,0:1]**2+x[:,1:2]**2))
        error1 = -(dy_xx1 + dy_yy1 )*beta1+fct1
        
        #u2
        beta2=1
        fct2=(3*(x[:,0:1]**2+x[:,1:2]**2)/tf.sqrt(x[:,0:1]**2+x[:,1:2]**2))
        error2 = -(dy_xx2 + dy_yy2 )*beta2+fct2
        
        # indicate if x is inside the disk
        #r2 = tf.sum(tf.square(x), axis=1, keep_dim=True) # r**2 for each x
        r2=tf.math.reduce_sum (tf.square(x), axis=1, keepdims=True)
        
        #r2=x[:,0:1]**2+x[:,1:2]**2
        is_inside = tf.math.less(r2, 0.5**2)  # True --> inside, False --> outside
        
        #tf.cast(is_inside, 'int32')
        #tf.cast(is_inside, tf.int32)
    # For x inside, use error1; for x outside, use error2
        #return error1 * is_inside, error2 * (1 - is_inside)
        return error1 * tf.cast(is_inside, dtype = tf.float32), error2 * (1 - tf.cast(is_inside, dtype = tf.float32))
       #print(error1 * tf.cast(is_inside, dtype = tf.float32))
       #return error1 , error2   
    geom = dde.geometry.Rectangle([-1,-1], [1,1])    
    
    
    def boundary(x, on_boundary):
         return on_boundary
    #on the boundary rectangle
    def func1(x):
        beta2=1
        beta1=1
        r=np.sqrt(x[:,0:1]**2+x[:,1:2]**2)
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
        r=np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
        return y2 - y1 - (beta2*(r**3-ro**3)+beta1*(ro**3-r**3))/(beta1*beta2)
    
    
    
    #  #normals
    #   #normalx
    # def nx(x):
    #      #r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
    #      r=np.sqrt(x[:,0:1]**2+x[:,1:2]**2)
    #      return (-x[:,1:2]/r)
    #  #normaly
    # def ny(x):
    #     # r=np.sqrt(np.power(x[:,0],2)+np.power(x[:,1],2))
    #     r=np.sqrt(x[:,0:1]**2+x[:,1:2]**2)
    #     return (x[:,0:1]/r)
    
    
    # the 2nd function on the interface which is the circle(flux)
    def func3(x, y, X):
         #y1 = y[:, 0:1]
         #y2 = y[:, 1:2]
         #ro=0.4
         r=np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
         #normal for x
         nx=X[:,0:1]/r
         #normal for y
         ny=X[:,1:2]/r
         du1x = dde.grad.jacobian(y, x, i=0,j=0)
         du1y= dde.grad.jacobian(y, x, i=0,j=1)
         du1=[du1x,du1y]
         du2x=dde.grad.jacobian(y, x, i=1,j=0)
         du2y=dde.grad.jacobian(y, x, i=1,j=1)
         du2=[du2x,du2y]
         beta2=1 
         beta1=1
         n1=np.transpose([nx,ny])
         #normal u2
         n2=-n1
         #return beta1*du1*np.transpose([nx,ny])
         #return ((beta1*du1*nx)-(beta2*du2*nx))-(((beta1*3*X[:,0:1]*r)*nx)-((beta2*3*X[:,1:2]*r)*ny))
         #return ((beta1*[nx,ny])-(beta2*[nx,ny]))-(((beta1*3*X[:,0:1]*r)*[nx,ny])-((beta2*3*X[:,1:2]*r)*[nx,ny]))
         
         #return ((beta1*np.dot(du1, n1))-(beta2*np.dot(du2, n2)))-((beta1*3*np.dot(np.dot(X[:,0:1],r),n1))-(beta2*3*np.dot(np.dot(X[:,1:2],r),n2)))
         return ((beta1*(du1*n1))-(beta2*du2*n2))-(((beta1*3*X[:,0:1]*r)*n1)-((beta2*3*X[:,1:2]*r)*n2))
    bc1 = dde.DirichletBC(geom,  func1, boundary,component=0)
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
    
    u=model.predict(X, operator=func3)
    print(u)
    
    
if __name__ == "__main__":
    main()    

