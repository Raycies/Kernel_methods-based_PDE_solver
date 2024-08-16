import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg.lapack import dposv
import matplotlib.pyplot as plt

import differential_operators
import kernels
import chronometer
import error_estimates

def check_order_compatibility(PDE_operators, kernel):
    for operator in PDE_operators:
        if operator.degree > kernel.max_degree: raise Exception("The PDE order is too high for this choice of kernel: it has not been implemented yet.")


class Estimator: # Estimates the solution of a linear PDE based on the samplings of the PDE operators, in a defined RKHS.

    solving_routine_choice = 0

    def __init__(self, data:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], hyper_parameters:tuple[float, float, kernels.Kernel], PDE_operators:tuple[differential_operators.LinearDifferentialOperator, differential_operators.LinearDifferentialOperator]) -> None:
    
        self.X, self.P, self.Z, self.B = data # X in Omega^n (sampling points in the domain), P in R^n the associated PDE operator values, Z in (border of Omega)^m, B in R^m the associated PDE operator values
        self.d, self.n, self.m = len(self.X[0]), len(self.X), len(self.Z)

        self.w, self.lambd, self.kernel = hyper_parameters # w: weight on boundary loss contribution, nu: coefficient associated with the function norm penalty, kernel: kernel used for the regression
        self.PDE_operators = PDE_operators
        self.chronometer = chronometer.Chronometer()

        check_order_compatibility(self.PDE_operators, self.kernel)
        #self.kernel.method = self.PDE_operators.method

        

    def fit(self): # Wrapper method for the computation of the coefficients
        self.chronometer.start()
        self.gram_matrix = self.kernel.build_gram_matrix(self.X, self.Z, self.PDE_operators) # Raw scalar products matrix
        self.chronometer.stop('Building Gram matrix')

        self.chronometer.start()
        D_inv = np.diag( np.concatenate( ((self.n)*np.ones((self.n))/(1-self.w), self.m*np.ones((self.m))/self.w) ) )

        system_solution = self.solve_linear_system((self.gram_matrix + self.lambd * D_inv), np.concatenate((self.P, self.B)))
        
        self.coefficients = system_solution
        self.chronometer.stop('Solving linear system')

        self.compute_estimator_characteristics()

    def solve_linear_system(self, A, b):# Here, possible to have more details on the lstsq resolution process
        X = 0
        if self.solving_routine_choice==0:
            b.reshape((-1,1))
            # Solve the system Ax = b using dposv LAPACK routine. R is the Cholesky decomposition of A, when only looking at the upper triangular part. X is the solution. When info=0, no error
            R, X, info = dposv(A, b)
            if info==0: print("Linear system solved without any issue")
            else: print("Linear system solving issue occurred, info for lapack dposv: " + str(info))
        if self.solving_routine_choice==1:
            lstsq_result = np.linalg.lstsq(A, b, rcond=-1)
            X = lstsq_result[0]
        return X
    
    def compute_estimator_characteristics(self):
        self.compute_loss()

    def compute_loss(self): # Add the expected Z values
        self.chronometer.start()
        empirical_loss_terms = self.gram_matrix @ self.coefficients

        sq_norm_RKHS = self.coefficients.T @ empirical_loss_terms
        self.loss_regularization = self.lambd * sq_norm_RKHS

        self.norm_RKHS = np.sqrt(sq_norm_RKHS)

        self.loss_border = np.sum(np.linalg.norm(empirical_loss_terms[self.n:])**2)/self.m

        self.weighted_loss_border = self.w * self.loss_border
        self.loss_indomain = np.sum(np.linalg.norm(empirical_loss_terms[:self.n])**2)/self.n

        self.weighted_loss_indomain = (1-self.w)* self.loss_indomain
        self.chronometer.stop("Computing the empirical loss")

    def predict(self, X_test, L_operator):
        predict_matrix = self.kernel.build_predict_matrix(self.X, self.Z, X_test, L_operator, self.PDE_operators)
        return predict_matrix @ self.coefficients
    
    def compute_L2_error(self, L_operator, target_function, X, chronometer_message="Computing the L2 error"): # By default, this function computes the empirical L2 error between the given function and the function L(estimator) on the given set of points X.
        self.chronometer.start()
        estimate = self.predict(X, L_operator)
        L2_error = np.sqrt(np.sum((estimate - target_function(X))**2)/len(X))
        self.chronometer.stop(chronometer_message)
        return L2_error
    
    def plot_along_line(self, x_low, x_high, nb_points, L_operator, target=None):
        alpha = np.arange(nb_points)/nb_points
        X = x_low[np.newaxis,:] + alpha[:,np.newaxis]*(x_high[np.newaxis,:]-x_low[np.newaxis,:])

        Y = self.predict(X, L_operator)
        plt.plot(alpha, Y, label='Prediction')
        if target!=None:
            if target==np.ndarray:
                plt.plot(alpha, target, label='Exact solution')
            else:
                Y2 = target(X)
                plt.plot(alpha, Y2, label='Exact solution')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def extract_data(self, input_metadata, error_estimates_targets={}, X_test=None, X_test_desc=""): # This method is used after fitting the parameters. input_metadata is a dictionary which must respect strict conventions, error_estimates_targets contains labels of the desired norm, X_test are the points used for quadrature (all weights equal to 1).
        self.estimator_data = error_estimates.EstimatorData(self, input_metadata, error_estimates_targets, X_test, X_test_desc)