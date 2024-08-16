from abc import ABC, abstractmethod

class LinearDifferentialOperator(ABC):
    pass

class LinearDifferentialOperatorConstantCoefficients(LinearDifferentialOperator):

    def __init__(self, coefficients):
        input_type = type(coefficients)

        if input_type==list: # New input example: [ (ndarray[0,0,1],6.7), (ndarray[0,0,0],3.14), (ndarray[2,1, 0],0.25)] -> = 6.7 dz f(x,y,z) + 3.14 f(x,y,z) + 0.25 dxxy f(x,y,z)
            # Input example: in = [ ([0,0,1],6.7), ([0,0,0],3.14), ([0,0],0.25), ([],0.1) ] -> = 6.7 dxxy f(x,y) + 3.14 dxxx f(x,y) + 0.25 dxx f(x,y) + 0.1 f(x,y)
            self.init_dict(coefficients) # When few derivatives are present in the operator
            self.method = 1

        elif input_type==tuple:
            self.init_list(coefficients)
            self.method = 0

        else: raise Exception("The given input is not supported to initialize this class.")
    
    def init_dict(self, coefficients):
        self.degree = 0
        for coefficient in coefficients: self.degree = max(self.degree, len(coefficient))
        self.coeff_dict = coefficients

    def init_tuple(self, coefficients):
        self.degree = len(coefficients) - 1 # Operator degree
        self.coeff_tuple = coefficients

    def give_coeff_dict(self):
        return self.coeff_dict