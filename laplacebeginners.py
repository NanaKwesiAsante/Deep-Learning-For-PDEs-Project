# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 20:11:20 2021

@author: Nana Kwesi
"""
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D 
#matplotlib inline

nx = 10
ny = 10

dx = 1. / nx
dy = 1. / ny

# creating a rectangle domain  between 0 and 1
x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)
#plt.scatter(x_space,y_space) 

#The analytic solution of the labplace equation
def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
    		np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))

#findingb the functional values using the analyt solution and ploting the surface
surface = np.zeros((ny, nx))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');

#the right hand side of the laplace eqn f(x)=0
def f(x):
    return 0.

#the activation functions to be used (Sigmoid and tanh)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# def tanh(x):
#     return (2. / (1. + np.exp(-2.*x)))-1

# nn using sigmoid activation function
def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

#nn using tanh activation function
# def neural_network(W, x):
#     a1 = tanh(np.dot(x, W[0]))
#     return np.dot(a1, W[1])


# def neural_network_x(x):
#     a1 = tanh(np.dot(x, W[0]))
#     return np.dot(a1, W[1])



#the boundary condition
def A(x):
    return x[1] * np.sin(np.pi * x[0])

# the trial solution for the laplace equation. This trial soln is then optimized by optimizing the parameters in it to be close to 0 ,i.e almost equal to the analyt soln
def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out

#Optimizing the trial solution using the loss function d2u/dx2 + d2u/dy2=0
def loss_function(W, x, y):
    loss_sum = 0.
    
    for xi in x:
        for yi in y:
            #input points from the rectangle
            input_point = np.array([xi, yi])
            #getting the output (u(x)) from using nn
            net_out = neural_network(W, input_point)[0]
            #now we find the 2nd derivative of the output(u(x)) but we find 1st derivative first 
            net_out_jacobian = jacobian(neural_network_x)(input_point)
            #now we find the 2nd derivative
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)
            
            #we do find the 2nd derivative for the trial solution too
            psy_t = psy_trial(input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)
            
            #we store d2u/dx2 
            gradient_of_trial_d2x = psy_t_hessian[0][0]
            #we store d2u/dy2
            gradient_of_trial_d2y = psy_t_hessian[1][1]
            
            # right part function
            func = f(input_point) 
            
            #we find the error square=((d2u/dx2 + d2u/dy2)-0)^2
            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum += err_sqr
        
    return loss_sum


W = [npr.randn(2, 10), npr.randn(10, 1)]
lmb = 0.001

#print(W)
#print (neural_network(W, np.array([1, 1])))
 #updating the weights using gradient descents
for i in range(100):
    loss_grad =  grad(loss_function)(W, x_space, y_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    
#print(W)    
#print (loss_function(W, x_space, y_space))

#creating empty holders for the functional values for analytic and trial soln by using updated weights
surface2 = np.zeros((ny, nx))
surface = np.zeros((ny, nx))

#functional values for analyt soln
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        


#functional values for trial soln
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
      net_outt = neural_network(W, [x, y])[0]
      surface2[i][j]= psy_trial([x, y], net_outt)
 
# MSEerror = np.zeros((ny, nx))
# #MSEerror=0
# for i,x in enumerate(x_space):
#     for j,y in enumerate(y_space):
#         #error= (surface[i][j]-surface2[i][j])^2/nx
#         MSEerror[i][j]= (surface[i][j]-surface2[i][j])^2/nx
#         #MSEerror+=error
       
MSEerror1=np.square(np.subtract(surface,surface2))
MSEerror2=np.sum(MSEerror1)
MSEerror3=np.divide(MSEerror2, ny*nx)
MSEerror=np.sqrt(MSEerror3)
print(surface)
print(surface2)
print(MSEerror)        
#print(net_outt)   
# print (surface[2])
# print (surface2[2])
        
        
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');    


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');

#contour plotting
# feature_x = np.linspace(-5.0, 3.0, 70)
# feature_y = np.linspace(-5.0, 3.0, 70)
  
# # Creating 2-D grid of features
# [X, Y] = np.meshgrid(feature_x, feature_y)
  
# fig, ax = plt.subplots(1, 1)
  
# Z = X ** 2 + Y ** 2
  
# # plots filled contour plot
# ax.contourf(X, Y, Z)
  
# ax.set_title('Filled Contour Plot')
# ax.set_xlabel('feature_x')
# ax.set_ylabel('feature_y')
  
# plt.show()