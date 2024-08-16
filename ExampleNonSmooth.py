import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from scipy.stats.qmc import Halton

import jax.numpy as jnp
import pandas

import differential_operators
import kernels
import estimator

#######################################
# Useful sampling functions in B(0,1) #
#######################################

def random_border_sampling(m):
    angles = np.random.uniform(0, 2*np.pi, size=(m))
    X_sampled = np.concatenate((np.cos(angles)[:,np.newaxis], np.sin(angles)[:,np.newaxis]), axis=1)
    return X_sampled

def random_indomain_sampling(n):
    X_sampled = []
    while len(X_sampled) < n:
        X_candidates = np.random.uniform(-1, 1, (n,2))
        for x in X_candidates:
            if np.linalg.norm(x) <= 1 and len(X_sampled) < n: X_sampled.append(x)
    
    return np.array(X_sampled)

def eqdisk(Nr):
    dR, x, y, k = 1/Nr, [0], [0], 1
    for r in np.arange(dR, 1 + dR, dR):
        n = round(np.pi / np.arcsin(1 / (2 * k)))
        theta = np.linspace(0, 2 * np.pi, n + 1)
        x.extend(r * np.cos(theta[:-1]))
        y.extend(r * np.sin(theta[:-1]))
        k += 1
    Nb = n
    Np = len(x)
    return np.concatenate((np.array(x)[:,np.newaxis], np.array(y)[:,np.newaxis]), axis=1), Nb, Np

def regular_sampling(Nr, withdraw_0): # http://www.holoborodko.com/pavel/2015/07/23/generating-equidistant-points-on-unit-disk/
    X, Nb, Np = eqdisk(Nr)
    X_boundary = []
    X_indomain = []
    if withdraw_0: X = X[1:]
    for x in X:
        if np.linalg.norm(x) >= 0.99999: X_boundary.append(x)
        else: X_indomain.append(x)
    return np.array(X_indomain), np.array(X_boundary)

def custom_sampling(Nr, withdraw_0): # http://www.holoborodko.com/pavel/2015/07/23/generating-equidistant-points-on-unit-disk/
    X, Nb, Np = eqdisk(Nr)
    X_boundary = []
    X_indomain = []
    for x in X:
        if np.linalg.norm(x) >= 0.99999: X_boundary.append(x)
        else: X_indomain.append(x)
    # Centrifuge transformation
    #   X_indomain = np.array(X_indomain[withdraw_0:])
    #   R = np.linalg.norm(X_indomain, axis=1)
    #   shift = np.concatenate((np.array([1]), np.sqrt(R)/R))
    #   X_indomain = shift[:,np.newaxis]*X_indomain

    # Centripetal transformation (without taking 0)
    X_indomain = np.array(X_indomain[withdraw_0:])
    R = np.linalg.norm(X_indomain, axis=1)
    shift = R
    X_indomain = shift[:,np.newaxis]*X_indomain    

    return X_indomain, np.array(X_boundary)

def Halton_sampling(n, dim):
    generator = Halton(dim)
    X_sampled = []
    while len(X_sampled) < n:
        X_candidates = 2*generator.random(n) - np.ones((n,dim))
        for x in X_candidates:
            if np.linalg.norm(x) <= 1 and len(X_sampled) < n: X_sampled.append(x)
    
    return np.array(X_sampled)

def Halton_border_sampling(m):
    generator = Halton(1)
    Theta = 2*np.pi*generator.random(m).reshape(-1)
    return np.array([np.array([np.cos(theta), np.sin(theta)]) for theta in Theta])

def regular_border_sampling(m):
        theta = np.arange(m)*2*np.pi/m
        x, y = np.cos(theta), np.sin(theta)
        return np.concatenate((x[:,np.newaxis],y[:,np.newaxis]), axis=1)

#######################################################
# Useful display functions (specific for the 2D case) #
#######################################################

def plot_color_map(f_hat, L_operator, target, nb_points, title="untitled plot", points=None): # points expected format: X, Z

    x = np.arange(nb_points)/(nb_points-1)*2 - 1
    y = np.arange(nb_points)/(nb_points-1)*2 - 1

    x, y = np.meshgrid(x, y)

    input = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis]), axis=2)
    input = input.reshape(-1,2)
    #tt = time.time()
    
    Z_raw_1 = f_hat.predict(input, L_operator) # Mettre un argument optionnel dans la fct predict, qui est store the time ?
    Z_raw_2 = target(input)
    #print("Predicting for the plot: " + str(round(time.time()-tt, 3))+"s")
    z = Z_raw_1.reshape(nb_points,nb_points) - Z_raw_2.reshape(nb_points,nb_points)

    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]**2 + y[i][j]**2 > 1: z[i][j] = 0

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    u_surf=ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, alpha=0.5)

    if points is not None:
        X, Z = points[0], points[1]
        ax.scatter(X[:,0], X[:,1], Z, color='r', s=20)

        Z_surf = f_hat.predict(X, L_operator)

        ax.scatter(X[:,0], X[:,1], Z_surf, color='b', s=20, marker='x')
        
        for i in range(len(X)):
            ax.plot([X[i][0], X[i][0]], [X[i][1], X[i][1]], [Z_surf[i], Z[i]])


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    fig.colorbar(u_surf, shrink= 0.5, aspect = 5)
    plt.show()

def plot_2D_function(f, nb_points, title="untitled plot"):

    x = np.arange(nb_points)/(nb_points-1)*2 - 1
    y = np.arange(nb_points)/(nb_points-1)*2 - 1

    x, y = np.meshgrid(x, y)

    input = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis]), axis=2)
    input = input.reshape(-1,2)
    
    Z_raw = f(input)
    z = Z_raw.reshape(nb_points,nb_points)

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
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

def scatter_points(f_hat, L_operator, target, title='untitled plot'):

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    X = f_hat.X
    Z_1 = f_hat.predict(X, L_operator)
    Z_2 = target(X)

    x, y, z_1, z_2 = [], [], [], []
    for i in range(len(X)):
        if abs(Z_1[i] - Z_2[i]) >=0.05:
            x.append(X[i,0])
            y.append(X[i,1])
            z_1.append(Z_1[i])
            z_2.append(Z_2[i])
            ax.plot([x[-1], x[-1]], [y[-1], y[-1]], [z_1[-1], z_2[-1]], color='g')
    ax.scatter(x, y, z_1, color='r', s=20, label='Predicted points')
    ax.scatter(x, y, z_2, color='b', s=20, marker='x', label='Target values')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

##############
# Parameters #
##############
kkk = 11
n, m = round(np.pi*kkk**2), round(2*np.pi*(kkk+1))
print("n=" + str(n) + "    ;    m=" + str(m))
N = n+m
sigma = 0.2

lambd = 10**(-6)
w = 0.5

#########################################
# Defining solution and its derivatives #
#########################################

X_support = np.array([ np.array([0.41928549, 0.34873639]),np.array([ 0.39781167, 0.19032019]), np.array([-0.21181035, 0.32632958]), np.array([-0.05156656, 0.19253878]), np.array([ 0.22515999, 0.85092658]), np.array([-0.1326113,  0.83001498]), np.array([ 0.23187336, 0.40295021]), np.array([ 0.83114687, -0.40403148]) ])

n_support = len(X_support)
scale=1

def target_3over2(X):
    D = euclidean_distances(X, X_support)
    R = np.linalg.norm(X, axis=1)
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    return np.sum(np.exp(-D/scale)*(np.ones(D.shape) + D/scale), axis=1)*(1-R**2)*in_domain

def target_laplacian_3over2(X):
    n, d = X.shape
    D = euclidean_distances(X, X_support) # shape n, n_support
    S = linear_kernel(X, X_support) # shape n, n_support
    R = np.linalg.norm(X, axis=1) # shape n
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    term_1 = -2*d*np.sum( np.exp(-D/scale)*(np.ones(D.shape) + D/scale) , axis=1)
    term_2 = (np.ones(R.shape)-R**2)/scale**2 * np.sum( np.exp(-D/scale)*(D/scale - d*np.ones(D.shape)) , axis=1)
    term_12 = 4/scale**2 * np.sum( np.exp(-D/scale) * (np.tile(R**2, (n_support,1)).T - S) , axis=1) # R shape must be changed to n n_support
    return ( term_1 + term_2 + term_12)*in_domain

def target_5over2(X):
    D = euclidean_distances(X, X_support)
    R = np.linalg.norm(X, axis=1)
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    return np.sum(np.exp(-D/scale)*(np.ones(D.shape) + D/scale + D**2/(3*scale**2)), axis=1)*(1-R**2)*in_domain

def target_laplacian_5over2(X):
    n, d = X.shape
    D = euclidean_distances(X, X_support) # shape n, n_support
    S = linear_kernel(X, X_support) # shape n, n_support
    R = np.linalg.norm(X, axis=1) # shape n
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    term_1 = -2*d*np.sum(np.exp(-D/scale)*(np.ones(D.shape) + D/scale + D**2/(3*scale**2)), axis=1)
    term_2 = (1-R**2)*np.sum( np.exp(-D/scale) * (D**2/scale**2 - d*(D/scale + np.ones(D.shape)))  , axis=1)/(3*scale**2)
    term_12 = 4 * np.sum( np.exp(-D/scale)*(np.ones(D.shape) + D/scale)*(np.tile(R**2, (n_support,1)).T - S) , axis=1)/(3*scale**2)
    return ( term_1 + term_2 + term_12)*in_domain

def target_toy(X):
    R = np.linalg.norm(X, axis=1)
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    return ( np.exp(-R/scale) - np.exp(-1/scale)) * in_domain

def target_laplacian_toy(X):
    n, d = X.shape
    R = np.linalg.norm(X, axis=1)
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    return np.exp(-R/scale)*(np.ones(R.shape) + scale*(1-d)/R) * in_domain/scale**2

def target_1over2(X):
    D = euclidean_distances(X, X_support)
    R = np.linalg.norm(X, axis=1)
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    return np.sum(np.exp(-D/scale), axis=1)*(1-R**2)*in_domain

def target_laplacian_1over2(X):
    n, d = X.shape
    D = euclidean_distances(X, X_support) # shape n, n_support
    S = linear_kernel(X, X_support) # shape n, n_support
    R = np.linalg.norm(X, axis=1) # shape n
    in_domain = (np.sign(np.ones(R.shape)-R)+1)/2
    term_1 = -2*d*np.sum(np.exp(-D/scale), axis=1)
    term_2 = (np.ones(R.shape)-R**2)/scale**2 * np.sum( np.exp(-D/scale)*(np.ones(D.shape) + scale*(1-d)/D) , axis=1)
    term_12 = 4/(scale) * np.sum( np.exp(-D/scale) * (np.tile(R**2, (n_support,1)).T - S)/D , axis=1) # R shape must be changed to n n_support
    return ( term_1 + term_2 + term_12)*in_domain

def null_function(x):
    return np.zeros((x.shape[0]))

def target_step(x):
    return null_function(x)

def target_laplacian_step(x):
    n, d = x.shape
    is_in = np.ones((n))
    for k in range(d):
        is_in = is_in * (np.sign(1/2 - x[:,k]) + 1)/2 * (np.sign(x[:,k] + 1/2) + 1)//2
    return is_in

def target_circle(x):
    return null_function(x)

def target_laplacian_circle(x):
    R = np.linalg.norm(x, axis=1)
    return (np.sign(1 - (2*R)**2) + 1)/2

############################
# Defining all derivatives #
############################

def target_1_toy(x, k):
    R = np.linalg.norm(x, axis=1)
    return - x[:,k]*np.exp(-R/scale)/(scale*R)

def target_2_toy(x, k, i):
    R = np.linalg.norm(x, axis=1)
    delta_ik = i==k
    return np.exp(-R/scale)*(-delta_ik*np.ones(R.shape)*scale + x[:,k]*x[:,i]*(np.ones(R.shape) + scale/R)/R)/(R*scale**2)


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

work_target, work_laplacian = target_toy, target_laplacian_toy
work_target_1 = target_1_toy
work_target_2 = target_2_toy

X, Z = Halton_sampling(n, dim), Halton_border_sampling(m) #regular_sampling(24, True)#random_indomain_sampling(n), random_border_sampling(m) # custom_sampling(14, True) # np.random.uniform(size=(n,dim)), sample_border(m) 
n, m = len(X), len(Z) # Because depending on the sampling method used, they can change

# X, Z = random_indomain_sampling(n), random_border_sampling(m)
P, B = work_laplacian(X), work_target(Z)


kernel_gauss = kernels.Gaussian_Kernel(sigma)

input_metadata = {"PDE_desc":"Poisson 2D Dirchlet with null border conditions, solution taken in H^2", "n":n, "m":m, "sampling_desc":"Uniform random", "w":w, "lambda":lambd, "kernel_desc":"Gaussian hand-written kernel", "kernel_param":sigma}

f_hat = estimator.Estimator((X,P,Z,B), (w, lambd, kernel_gauss), (Laplacian_operator, Border_dirichlet))

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
###########
plot_2D_function(work_target, 200, 'target')

plot_color_map(f_hat, I_operator, null_function, 100, 'Estimation', (Z, B))

plot_color_map(f_hat, I_operator, work_target, 100, 'Error map')

plot_2D_function(work_laplacian, 200, 'target laplacian')

plot_color_map(f_hat, Laplacian_operator, null_function, 50, 'Laplacian of the estimation', (X,P))

#plot_color_map(f_hat, Laplacian_operator, work_laplacian, 50, 'Laplacian error map')

#plot_2D_function(lambda x: work_target_1(x,0), 200, 'target fst deriv')

#plot_color_map(f_hat, fst_derivative_operator, null_function, 200, 'estimate fst deriv')

#plot_color_map(f_hat, fst_derivative_operator, lambda x: work_target_1(x, 0), 200, 'fst deriv error map')

scatter_points(f_hat, fst_derivative_operator, lambda x: work_target_1(x, 0), 'quadrature error')
###########
#plot_color_map(f_hat, null_operator, target_function, 50, 'Plot of the target')
#plot_color_map(f_hat, I_operator, target_function, 50, 'Plot of the target')

#f_hat.plot_along_line(np.array([0,0]), np.array([1,1]), 100, Laplacian_operator, target_laplacian)

#f_hat.plot_along_line(np.array([0,0]), np.array([1,1]), 100, I_operator, target_function)

#f_hat.plot_along_line(np.array([0,0]), np.array([0,1]), 100, I_operator, target_function)


f_hat.extract_data(input_metadata, {"L2_error":work_target, "H1_error":[lambda x: work_target_1(x, i) for i in range(dim)], "H2_error":[[lambda x: work_target_2(x, i, k) for i in range(dim)] for k in range(dim)]}, Halton_sampling(1500,2)) # , "H2_error":[[dxx_target, dxy_target],[dxy_target,dxx_target]]

print(f_hat.estimator_data)

#f_hat.estimator_data.store_data("generated_data/ici.csv")



