from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import jax.numpy as jnp
import jax


def init_gaussian_kernel(kernel_parameter, d):
    def func(*x):
        return jnp.exp(-jnp.sum(jnp.square(jnp.array(x)))/(2*kernel_parameter**2))
    return func

# =============================================
# MOTHER KERNEL CLASS
# =============================================

class Kernel(ABC): # Mother class: which gives the structure of the specific kernel classes

    def build_gram_matrix(self, X, Z, PDE_operators) -> np.ndarray:
        P_operator, B_operator = PDE_operators # In-domain linear differential operator, Border linear differential operator
        if self.method==1:
            P_coeffs = P_operator.give_coeff_dict()
            B_coeffs = B_operator.give_coeff_dict()

            TL = self.build_gram_block(P_coeffs, P_coeffs, X, X) # Top left
            TR = self.build_gram_block(P_coeffs, B_coeffs, X, Z) # Top right
            BR = self.build_gram_block(B_coeffs, B_coeffs, Z, Z) # Bottom right

            final_gram_matrix = np.block([[TL, TR], [TR.T, BR]])

            return final_gram_matrix
        
    def build_gram_block(self, coeffs_1, coeffs_2, X_1, X_2): # When coeffs_1=coeffs_2 and X_1=X_2: We are doing twice more calculus than necessary (symmetry).
        # Creates the block: (<fstOp^1(K)({X_1}_i, .),  sndOp^2(K)({X_2}_j, .)>)_{i,j}
        n_1, n_2 = len(X_1), len(X_2)
        block = np.zeros((n_1, n_2))

        for order_1, coeff_1 in coeffs_1: # pb de listes vides
            for order_2, coeff_2 in coeffs_2:
                raw_block = self.route_gram_order(order_1, order_2, X_1, X_2)
                block += (-1)**(int(sum(order_2)))*coeff_1 * coeff_2 * raw_block # The power of (-1): because kernel functions are translation-invariant
        return block
    
    def build_predict_matrix(self, X, Z, X_test, L_operator, PDE_operators):
        P_operator, B_operator = PDE_operators
        if self.method==1:
            L_coeffs = L_operator.give_coeff_dict()
            P_coeffs = P_operator.give_coeff_dict()
            B_coeffs = B_operator.give_coeff_dict()

            
            L = self.build_gram_block(L_coeffs, P_coeffs, X_test, X) # Left INVERSION DES ARGUMENTS P ET L, B ET L
            R = self.build_gram_block(L_coeffs, B_coeffs, X_test, Z) # Right

            return np.block([[L, R]])
        
    @abstractmethod
    def route_gram_order(self, order_1, order_2, X_1, X_2):
        pass
        


# =============================================
# SPECIFIC KERNEL CLASSES
# =============================================

# GAUSSIAN KERNEL -----------------------------


class Gaussian_Kernel(Kernel):

    max_degree = 2 # Maximum derivatives implemented
    method = 1

    def __init__(self, sigma) -> None:
        self.sigma = sigma # Gaussian Kernel parameter
        
    
    def route_gram_order(self, order_1, order_2, X_1, X_2): # Here, this function does not belong to the abstract class, because we use the fact that one can write K(x,y) = k(x-y).
        order = []
        for index, value in enumerate(order_1 + order_2):
            order = order + [index]*int(value)
        global_order = len(order)
        if global_order==0: return self.gram_order_0(X_1/self.sigma, X_2/self.sigma, order)
        elif global_order==1: return  self.gram_order_1(X_1/self.sigma, X_2/self.sigma, order)/self.sigma
        elif global_order==2: return  self.gram_order_2(X_1/self.sigma, X_2/self.sigma, order)/self.sigma**2
        elif global_order==3: return  self.gram_order_3(X_1/self.sigma, X_2/self.sigma, order)/self.sigma**3
        elif global_order==4: return  self.gram_order_4(X_1/self.sigma, X_2/self.sigma, order)/self.sigma**4

    # On veut que dans ce qui est renvoyé, len(X1) soit le nb de lignes

    def gram_order_0(self, X_1, X_2, orders): # Creates the matrix < K_{(x_1)_i} , K_{(x_2)_j} > = K({(x_1)_i}, {(x_2)_j}) = k({(x_1)_i} - {(x_2)_j})
        D = euclidean_distances(X_1, X_2, squared=True) # squared distances matrix
        return np.exp(-D/2)
    
    def gram_order_1(self, X_1, X_2, orders): # Creates the matrix (< d^{i|} K_{(x_1)_l} , K_{(x_2)_c} >) = d^{i|} K((x_1)_l, (x_2)_c) = d^i k((x_1)_l - (x_2)_c) where l is a row index and c a column index
        i = orders[0]
        D = euclidean_distances(X_1, X_2, squared=True) # squared distances matrix
        Z_i = np.tile(np.reshape(X_1[:,i], (-1,1)), len(X_2)) - np.tile(np.reshape(X_2[:,i], (-1,1)), len(X_1)).T
        return - Z_i * np.exp(-D/2)
    
    def gram_order_2(self, X_1, X_2, orders): # Creates the matrix (< d^{i,j|} K_{(x_1)_l} , K_{(x_2)_c} >) = d^{i,j}k ((x_1)_l - (x_2)_c)  where l is a row index and c a column index.
        i, j = orders[0], orders[1]
        n_1, n_2 = len(X_1), len(X_2)

        K_gram = np.exp(-euclidean_distances(X_1, X_2, squared=True)/2)

        delta_ij = (i==j)

        Z_i = np.tile(np.reshape(X_1[:,i], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,i], (-1,1)), n_1).T
        Z_j = np.tile(np.reshape(X_1[:,j], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,j], (-1,1)), n_1).T

        return K_gram * (Z_i*Z_j - delta_ij*np.ones((n_1, n_2)))
    
    def gram_order_3(self, X_1, X_2, orders): # Creates the matrix (- < d^{i1, i2|} K_((x_1)_l} , d^{j|} K_{(x_2)_c} >) = d^{i1,i2,j}k ((x_1)_l - (x_2)_c)  where l is a row index and c a column index.
        i, j, l = orders[0], orders[1], orders[2] 
        n_1, n_2 = len(X_1), len(X_2)

        Z_i = np.tile(np.reshape(X_1[:,i], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,i], (-1,1)), n_1).T
        Z_j = np.tile(np.reshape(X_1[:,j], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,j], (-1,1)), n_1).T
        Z_l = np.tile(np.reshape(X_1[:,l], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,l], (-1,1)), n_1).T

        K_gram = np.exp(-euclidean_distances(X_1, X_2, squared=True)/2)

        delta_i_j = (i==j)
        delta_i_l = (i==l)
        delta_j_l = (j==l)

        ones = np.ones((n_1, n_2))

        return K_gram * ( Z_l * (delta_i_j*ones - Z_i*Z_j) + delta_i_l * Z_j + delta_j_l * Z_i) # last last member: inversion Z_i Z_j and Z_j Z_i
    
    def gram_order_4(self, X_1, X_2, orders): # Creates the matrix (< d^{i1, i2|} K_{(x_1)_l} , d^{j1, j2|} K_{(x_2)_c} >) = d^{i1,i2,j1,j2}k ((x_1)_l - (x_2)_c) where l is a row index and c a column index.
        i, j, l, c = orders[0], orders[1], orders[2], orders[3]
        n_1, n_2 = len(X_1), len(X_2)

        Z_i = np.tile(np.reshape(X_1[:,i], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,i], (-1,1)), n_1).T
        Z_j = np.tile(np.reshape(X_1[:,j], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,j], (-1,1)), n_1).T
        Z_l = np.tile(np.reshape(X_1[:,l], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,l], (-1,1)), n_1).T
        Z_c = np.tile(np.reshape(X_1[:,c], (-1,1)), n_2) - np.tile(np.reshape(X_2[:,c], (-1,1)), n_1).T

        K_gram = np.exp(-euclidean_distances(X_1, X_2, squared=True)/2)

        delta_i_j = (i==j)
        delta_l_c = (l==c)
        delta_i_l = (i==l)
        delta_j_l = (j==l)
        delta_i_l = (i==l)
        delta_j_c = (j==c)
        delta_i_c = (i==c)

        ones = np.ones((n_1, n_2))

        fst_term = (delta_i_j*ones - Z_i * Z_j)*(delta_l_c*ones - Z_c*Z_l)
        snd_term = - Z_i*(delta_j_l * Z_c + delta_j_c*Z_l) - Z_j*(delta_i_l *Z_c + delta_i_c*Z_l)
        thd_term = (delta_i_l*delta_j_c + delta_j_l*delta_i_c)*ones

        # fst_term = (delta_l_c*ones - Z_c*Z_l)*(delta_i_j*ones - Z_i*Z_j + 2*delta_i_j_l*ones)
        # snd_term = (1-delta_i_j_l)*((delta_i_l_c + delta_j_l_c)*ones - (delta_i_l*Z_i +delta_j_l*Z_j)*Z_c )
        # thd_term = - Z_l*(delta_i_c * Z_j + delta_j_c * Z_i)

        return K_gram * (fst_term + snd_term + thd_term)


class JaxKernel(Kernel):

    init_map = {"gaussian": init_gaussian_kernel}
    max_degree = 9
    method=1

    def __init__(self, kernel_type, kernel_parameter, d) -> None:
        self.kernel_type, self.kernel_parameter = kernel_type, kernel_parameter
        self.kernel = self.init_map[kernel_type](kernel_parameter, d)

    def route_gram_order(self, order_1, order_2, X_1, X_2): 
        # Construire la dérivée d'ordre order_1 + order_2, puis l'évaluer pour former la matrice
        order = order_1 + order_2
        successive_derivatives = self.kernel
        for i in range(len(order)):
            for j in range(int(order[i])):
                successive_derivatives = jax.grad(successive_derivatives, i)

        return self.gram_all_orders(successive_derivatives, X_1, X_2)

    def gram_all_orders(self, successive_derivatives, X_1, X_2):

        n_1, n_2 = X_1.shape, X_2.shape

        Z = np.tile(np.reshape(X_1, (n_1[0], 1, n_2[1])), (n_2[0], 1)) - np.transpose(np.tile(np.reshape(X_2, (n_2[0], 1, n_1[1])), (n_1[0], 1)), (1,0,2))
        Z_dim_flat = []
        for i in range(n_1[1]): # On parcourt les dimensions
            Z_dim_flat.append(Z[:,:,i].flatten())
        
        val = jax.vmap(successive_derivatives)(*Z_dim_flat)
        gram_block = val.reshape((n_1[0], n_2[0]))

        return gram_block


