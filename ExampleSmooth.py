# =============================================
# IMPROVEMENTS
# =============================================

# Add the possibility to sample according a specific probability distribution, and quadrature wrt a specific measure.
# Define a way to compute Linf norm: empirical maximum difference on training/test set ? Use an optimization routine ?

# =============================================
# USER GUIDE
# =============================================

# Define an instance of the Kernel class, using the built-in kernel types (Gaussian, Sobolev, ...)
# Define the instances of the PDE's linear differential operators and gather them in a tuple, in the following order:
#   - In the domain differential operator
#   - Border differential operators (for the moment, only one available)
# Define the hyper-parameters of the adapted kernel ridge regression:
#   - w: weight on the border empirical loss (If more border operators are available, it becomes a tuple)
#   - lambda: Regularization parameter
# Provide the data: tuples of sampling points and associated differential operator values.


import jax.numpy as jnp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

import differential_operators
import kernels
import estimator

from scipy.stats.qmc import Halton

##########################################
# Useful sampling functions (in [0,1]^2) #
##########################################

def random_border_sampling(m):
    # Choisir aléatoirement un des 4 côtés pour chaque point
    sides = np.random.randint(0, 4, m)
    
    # Générer des coordonnées aléatoires pour chaque point
    random_coords = np.random.rand(m)
    
    # Initialiser les tableaux x et y pour les points
    x = np.zeros(m)
    y = np.zeros(m)
    
    # Assigner les coordonnées en fonction du côté choisi
    x = np.where(sides == 0, random_coords, x)  # Bas (x in [0,1], y=0)
    y = np.where(sides == 0, 0, y)
    
    x = np.where(sides == 1, 1, x)  # Droite (x=1, y in [0,1])
    y = np.where(sides == 1, random_coords, y)
    
    x = np.where(sides == 2, random_coords, x)  # Haut (x in [0,1], y=1)
    y = np.where(sides == 2, 1, y)
    
    x = np.where(sides == 3, 0, x)  # Gauche (x=0, y in [0,1])
    y = np.where(sides == 3, random_coords, y)
    
    points = np.column_stack((x, y))
    return points

def random_indomain_sampling(n):
    x_coords = np.random.uniform(0, 1, n)
    y_coords = np.random.uniform(0, 1, n)
    
    return np.vstack((x_coords, y_coords)).T

def regular_indomain_sampling(n):# nb de points par ligne

    points = (np.arange(n)+1)/(n+1)
    x, y = np.meshgrid(points, points)
    grid_points = np.column_stack([x.ravel(), y.ravel()])

    return grid_points

def regular_border_sampling(n):# nb d epoints par ligne
    points = (np.arange(n))/n
    line_1 = points[:,np.newaxis]*np.array([0,1])
    line_2 = np.tile(np.array([0,1]), (n,1)) + points[:,np.newaxis]*np.array([1,0])
    line_3 = np.tile(np.array([1,1]), (n,1)) - points[:,np.newaxis]*np.array([0,1])
    line_4 = np.tile(np.array([1,0]), (n,1)) - points[:,np.newaxis]*np.array([1,0])
    return np.concatenate((line_1, line_2, line_3, line_4))

def Halton_sampling_domain(n, dim):
    generator = Halton(dim)
    return generator.random(n)

def Halton_sampling_border(n):
    # Initialize the Halton sequence generator for 1D
    halton_1d = Halton(d=1, scramble=False)
    
    # Generate 4n points using Halton sequence
    xb_seq = halton_1d.random(n).flatten()
    xt_seq = halton_1d.random(n).flatten()
    yl_seq = halton_1d.random(n).flatten()
    yr_seq = halton_1d.random(n).flatten()
    
    # Generate points on the border
    bottom = np.column_stack((xb_seq, np.zeros(n)))  # Bottom side (y = 0)
    top = np.column_stack((xt_seq, np.ones(n)))      # Top side (y = 1)
    left = np.column_stack((np.zeros(n), yl_seq))    # Left side (x = 0)
    right = np.column_stack((np.ones(n), yr_seq))    # Right side (x = 1)
    
    # Combine all border points
    border_points = np.vstack((bottom, top, left, right))
    
    return border_points

#######################################################
# Useful display functions (specific for the 2D case) #
#######################################################

def plot_color_map(f_hat, L_operator, target, nb_points, title="untitled plot"):

    x = (np.arange(nb_points)+0.5)/nb_points
    y = (np.arange(nb_points)+0.5)/nb_points

    x, y = np.meshgrid(x, y)

    input = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis]), axis=2)
    input = input.reshape(-1,2)
    #tt = time.time()
    
    Z_raw_1 = f_hat.predict(input, L_operator) # Mettre un argument optionnel dans la fct predict, qui est store the time ?
    Z_raw_2 = target(input)
    #print("Predicting for the plot: " + str(round(time.time()-tt, 3))+"s")
    z = Z_raw_1.reshape(nb_points,nb_points) - Z_raw_2.reshape(nb_points,nb_points)

    #print(f_hat.predict(np.array([np.array([1/8, 1/8])]), I_operator))

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    u_surf=ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    fig.colorbar(u_surf, shrink= 0.5, aspect = 5)
    plt.show()

def plot_points(in_points, border_points):
    x, y = in_points[:, 0], in_points[:, 1]
    xbis, ybis = border_points[:, 0], border_points[:, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, label="In-domain data points")
    plt.scatter(xbis, ybis, label='Border data points')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

##############
# Parameters #
##############

n, m = 15**2, 4*16
N = n+m
sigma = 0.2

lambd = 10**(-10)
w = 1/2 # 1- 10**(-10)

# (lambda, w) définissent (eta, G_ratio)
# G_ratio, eta = w**2/(m*(1-w)), lambd*m/w

eta = 10**(-10)
#G_om_om, G_bom_bom = 8/(sigma**4), 1
#G_ratio = G_om_om / G_bom_bom

# (lambda, w) pour former eta* I_{n+m}:
# lambd, w = (eta/(n+m), m/(m+n))

# (eta, G_ratio) définissent (lambda, w) ((Stuart setting)):
# w = m*G_ratio*(-1+np.sqrt(1+4/(m*G_ratio)))/2
# lambd = w*eta/m


#########################################
# Defining solution and its derivatives #
#########################################

def target_function(x): # Domain: [0, 1]^2
    return np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) + 4*np.sin(4*np.pi*x[:,0]) * np.sin(4*np.pi*x[:,1])

def d_target(x):
    return np.pi * np.cos(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) + 4*4*np.pi*np.cos(4*np.pi*x[:,0]) * np.sin(4*np.pi*x[:,1])

def dx_target(x):
    return d_target(x)

def dy_target(x):
    return d_target(x[:,[1,0]])

def dxx_target(x):
    return - np.pi**2 * np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) - 4 * (4*np.pi)**2 *np.sin(4*np.pi*x[:,0]) * np.sin(4*np.pi*x[:,1])

def dxy_target(x):
    return np.pi**2 * np.cos(np.pi*x[:,0]) * np.cos(np.pi*x[:,1]) + 4 * (4*np.pi)**2 *np.cos(4*np.pi*x[:,0]) * np.cos(4*np.pi*x[:,1])

def target_laplacian(x):
    return - 2*np.pi**2 * np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) - 8*16*np.pi**2*np.sin(4*np.pi*x[:,0]) * np.sin(4*np.pi*x[:,1])

def null_function(x):
    return np.zeros((x.shape[0]))

##########################
# Defining PDE operators #
##########################

dim=2
coeff_list_Laplacian = [(2*np.eye(1, dim, i).reshape(-1), 1) for i in range(dim)]
Laplacian_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients(coeff_list_Laplacian)

coeff_list_1st_derivative = [(np.eye(1, dim, 0).reshape(-1), 1)]
fst_derivative_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients(coeff_list_1st_derivative)

coeff_list_border = [(np.zeros((dim)),1)]
Border_dirichlet = differential_operators.LinearDifferentialOperatorConstantCoefficients(coeff_list_border)

#####################
# Initializing data #
#####################

X, Z = Halton_sampling_domain(n, 2), Halton_sampling_border(m//4) # np.random.uniform(size=(n,dim)), sample_border(m)
# X, Z = random_indomain_sampling(n), random_border_sampling(m)
P, B = target_laplacian(X), target_function(Z)

kernel_gauss = kernels.Gaussian_Kernel(sigma)
kernel_gauss_jax = kernels.JaxKernel("gaussian", sigma, dim)

input_metadata = {"PDE_desc":"Poisson 2D equation, same example as Stuart&al.", "n":n, "m":m, "sampling_desc":"Uniform random", "w":w, "lambda":lambd, "kernel_desc":"Gaussian hand-written kernel", "kernel_param":sigma}

f_hat = estimator.Estimator((X,P,Z,B), (w, lambd, kernel_gauss), (Laplacian_operator, Border_dirichlet))
#f_hat = estimator.Estimator((X,P,Z,B), (w, lambd, kernel_gauss_jax), (Laplacian_operator, Border_dirichlet))

######################################
# Computing the fitting coefficients #
######################################

f_hat.fit()

#######################
# Display the results #
#######################

print(f_hat.chronometer)

I_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([( np.zeros((dim)),1)])
null_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.zeros((dim)),0)])
dx_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.array([1,0]),1)])
dy_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.array([0,1]),1)])
dxy_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.array([1,1]),1)])
dxx_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.array([2,0]),1)])

plot_points(X,Z)
#plot_color_map(f_hat, dxx_operator, dxx_target, 50)
#plot_color_map(f_hat, dx_operator, dx_target, 50)

#plot_color_map(f_hat, dxy_operator, dxy_target, 50)

plot_color_map(f_hat, Laplacian_operator, target_laplacian, 50, 'Plot of the error on the laplacian')

plot_color_map(f_hat, I_operator, null_function, 50, 'Plot of the estimation')
plot_color_map(f_hat, I_operator, target_function, 50, 'Plot of the error')
#plot_color_map(f_hat, null_operator, target_function, 50, 'Plot of the target')
#plot_color_map(f_hat, I_operator, target_function, 50, 'Plot of the target')

#f_hat.plot_along_line(np.array([0,0]), np.array([1,1]), 100, Laplacian_operator, target_laplacian)

#f_hat.plot_along_line(np.array([0,0]), np.array([1,1]), 100, I_operator, target_function)

#f_hat.plot_along_line(np.array([0,0]), np.array([0,1]), 100, I_operator, target_function)


f_hat.extract_data(input_metadata, {"L2_error":target_function, "H1_error":[dx_target, dy_target], "H2_error":[[dxx_target, dxy_target],[dxy_target,dxx_target]]}, f_hat.X) # , "H2_error":[[dxx_target, dxy_target],[dxy_target,dxx_target]]

print(f_hat.estimator_data)

print()
print("Trace dans le domaine: " + str(np.trace(f_hat.gram_matrix[:n, :n])))
print("Trace au bord: " + str(np.trace(f_hat.gram_matrix[n:, n:])))

#f_hat.estimator_data.store_data("generated_data/ici.csv")