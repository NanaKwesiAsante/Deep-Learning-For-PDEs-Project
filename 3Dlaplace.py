# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:33:45 2021

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
        du_xx = dde.grad.hessian(y, x, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, i=1, j=1)
        du_zz=dde.grad.hessian(y, x, i=2, j=2)
        return du_xx + du_yy+ du_zz
        # dy_xx = dde.grad.hessian(y, x)
        # return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

    def boundary(x, on_boundary):
        return on_boundary

    def func(x):
        a=1
        b=1
        return 0.0001*(np.exp(-np.sqrt(a**2+b**2)*np.pi*x[:,0:1])*np.sin(a*np.pi * x[:,1:2])*np.sin(b*np.pi * x[:,2:3]))

    # def func(x):
    #     return 10*(x[:,0:1]*x[:,1:2]*x[:,2:])
     
    geom1 = dde.geometry.geometry_3d.Sphere([0,0,0], 2)
    geom2 = dde.geometry.geometry_3d.Sphere([0,0,0], 1)
    geom=dde.geometry.CSGDifference(geom1, geom2)
    bc = dde.DirichletBC(geom,func, boundary)
    data = dde.data.PDE(geom, pde, bc,num_domain=1000, num_boundary=100, solution=func, num_test=12000)

    layer_size = [3] + [50] * 4 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)
    #net.apply_output_transform(lambda x, y:100)
    #net.apply_output_transform(
    #    lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + tf.sin(np.pi * x[:, 0:1])
    #)
    def output_transform(x, y):
        a,b=1,1
        r1=2
        r2 = tf.math.reduce_sum(tf.square(x), axis=1, keepdims=True)
        return (r2 - r1**2)(r2 - r2**2) * y + (r2 - r1**2) / (r2**2 - r1**2) * (b - a) + a
    
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    
    model.compile("adam", lr=0.001, loss_weights=[1,1])
    #model.train(epochs=50000)
    
    losshistory, train_state = model.train(epochs=10000)
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()
    

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#     # # Plot PDE residual
#     # model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
#     # x = geom.uniform_points(1000, True)
#     # y = model.predict(x, operator=pde)
#     # plt.figure()
#     # plt.plot(x, y)
#     # plt.xlabel("x")
#     # plt.ylabel("PDE residual")
#     # plt.show()


if __name__ == "__main__":
    main()
