# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:41:20 2021

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
    r0 = 0.5
    rg=0.4
    B1=1
    B2=1
    innerdisk = dde.geometry.Disk([0, 0], rg)
    geom = dde.geometry.Rectangle([-1,-1], [1,1]) 
    def pde(x, y):
        
        u1, u2 = y[:, 0:1], y[:, 1:2]
        
        #radius of circles of points inside
        i0 = tf.reduce_sum(x**2, axis=1, keepdims=True) < r0**2
        inside = tf.cast(i0, tf.float32)
        outside=1-inside
        
        f1=(3*(2*x[:,0:1]**2+x[:,1:2]**2)/tf.sqrt(x[:,0:1]**2+x[:,1:2]**2))
        f2=(3*(x[:,0:1]**2+x[:,1:2]**2)/tf.sqrt(x[:,0:1]**2+x[:,1:2]**2))
        
        #u1
        du1_x = tf.gradients(u1, x)[0][:,0:1]
        du1_y = tf.gradients(u1, x)[0][:,1:]
        du1_xx = tf.gradients(du1_x, x)[0][:, 0:1]
        du1_yy = tf.gradients(du1_y, x)[0][:, 1:]
        Laplacian_u1 = -(du1_xx + du1_yy )*B1+f1
        
        #u2
        du2_x = tf.gradients(u2, x)[0][:,0:1]
        du2_y = tf.gradients(u2, x)[0][:,1:]
        du2_xx = tf.gradients(du2_x, x)[0][:, 0:1]
        du2_yy = tf.gradients(du2_y, x)[0][:, 1:]
        Laplacian_u2 = -(du2_xx + du2_yy)*B2+f2
        
        f1=Laplacian_u1*inside
        f2=Laplacian_u2*outside
        return f1, f2
    
    # circle boundary
    def boundary(x, _):
        return innerdisk.on_boundary(x)
    
    #on the interface circle
    def func1(inputs, outputs, X):
        u1, u2 = outputs[:, 0:1], outputs[:, 1:2]
        r=np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
        gj=(B2*(r**3-rg**3)+B1*(rg**3-r**3))/(B1*B2)
        return (u1 - u2)-gj
    
     #on the interface but for the flux
    def func2(inputs, outputs, X):
        u1, u2 = outputs[:, 0:1], outputs[:, 1:2]
        r=np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
        du1dx = tf.gradients(u1, inputs)[0]
        du2dx = tf.gradients(u2, inputs)[0]
        n = np.array(list(map(innerdisk.boundary_normal, X)))
        gf=(((B1*3*X[:,0:1]*r)*n)-((B2*3*X[:,1:2]*r)*n))
        f1 = tf.reduce_sum(du1dx * n * B1, axis=1, keepdims=True)
        f2 = tf.reduce_sum(du2dx * n * B2, axis=1, keepdims=True)
        return  (f1 - f2)-gf
    
    #rectangle boundary
    def boundary2(x, _):
        return geom.on_boundary(x)
    
    #on the rectangle
    def func3(inputs, outputs, X):
        u2 = outputs[:, 1:2]
        r=np.sqrt(X[:,0:1]**2+X[:,1:2]**2)
        gd=((r**3)/B2)+((1/B1)-(1/B2))*rg**3
        return u2-gd
    
    
    
    bc1 = dde.OperatorBC(geom, func1, boundary)
    bc2 = dde.OperatorBC(geom, func2, boundary)
    bc3 = dde.OperatorBC(geom, func3, boundary2)
    #bc3 = dde.DirichletBC(geom,func3, boundary2,component=1)
    
    data = dde.data.PDE(geom,pde, [bc1, bc2, bc3], num_domain=100,num_boundary=0, anchors=np.vstack((innerdisk.random_boundary_points(100),geom.random_boundary_points(100))))
    net = dde.maps.FNN([2] + [128] * 4 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    #loss_weights = [1,1,100,10, 1]
    model.train(epochs=10000)
    #model.compile("L-BFGS-B") 
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
     
    
    
    
    
if __name__ == "__main__":
    main()
