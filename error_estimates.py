import differential_operators
import pandas as pd
import os
import numpy as np
from numpy import sqrt
# Add a method to extract time data

def null_function(x):
    return np.zeros((x.shape[0]))

class EstimatorData(): # Associated with one Estimator instance
    metadata_labels = ["PDE_desc", "n", "m", "sampling_desc", "w", "lambda", "kernel_desc", "kernel_param"]
    loss_labels = ["unweighted_border_loss", "unweighted_indomain_loss", "norm_RKHS"]
    error_labels = ["L2_error", "H1_error", "H2_error", "L2_target", "H1_target", "H2_target"]
    
    def __init__(self, estimator, input_metadata, error_estimates_targets, X_test, X_test_desc): # error_estimates_targets of the format: {"H1_error":[1st_partial_derivative_target, ..., d-th_partial_derivative_target]} (H2 would be a matrix dxd)
        self.data = input_metadata
        self.data["unweighted_border_loss"], self.data["unweighted_indomain_loss"], self.data["RKHS_norm"] = estimator.loss_border , estimator.loss_indomain, estimator.norm_RKHS

        self.compute_data(estimator, error_estimates_targets, X_test, X_test_desc)
    


    def compute_data(self, estimator, error_estimates_targets, X_test, X_test_desc):
        labels = error_estimates_targets.keys()

        if X_test_desc != "":
            self.data["error_quadrature_desc"] = X_test_desc

        if "H2_error" in labels:
            s = 0
            s_target = 0
            for i in range(estimator.d):
                for j in range(estimator.d):
                    double_partial_derivative_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.eye(1, estimator.d, i).reshape(-1) + np.eye(1, estimator.d, j).reshape(-1), 1)])
                    s += estimator.compute_L2_error(double_partial_derivative_operator, error_estimates_targets["H2_error"][i][j], X_test)**2
                    s_target += estimator.compute_L2_error(double_partial_derivative_operator, null_function, X_test)**2
            self.data["H2_error"] = sqrt(s)
            self.data["H2_target"] = sqrt(s_target)
            
        if "H1_error" in labels:
            s = 0
            s_target = 0
            for i in range(estimator.d):
                partial_derivative_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.eye(1, estimator.d, i).reshape(-1), 1)])
                s += estimator.compute_L2_error(partial_derivative_operator, error_estimates_targets["H1_error"][i], X_test)**2
                s_target += estimator.compute_L2_error(partial_derivative_operator, null_function, X_test)**2
            self.data["H1_error"] = sqrt(s)
            self.data["H1_target"] = sqrt(s_target)

        if "L2_error" in labels:
            I_operator = differential_operators.LinearDifferentialOperatorConstantCoefficients([(np.zeros((estimator.d)),1)])
            self.data["L2_error"] = estimator.compute_L2_error(I_operator, error_estimates_targets["L2_error"], X_test)
            self.data["L2_target"] = estimator.compute_L2_error(I_operator, null_function, X_test)

    def store_data(self, storage_path=""):

        number_data = {x: self.data[x] for x in self.data.keys() if x not in {"PDE_desc", "sampling_desc", "kernel_desc", "error_quadrature_desc"}}
        text_data = {"PDE_desc": self.data["PDE_desc"], "sampling_desc": self.data["sampling_desc"], "kernel_desc": self.data["kernel_desc"]}

        if storage_path=="":
            storage_path="generated_data/qqchose.csv"
        storage_df = pd.DataFrame.from_dict(number_data, orient='index')
        if not os.path.exists(storage_path):
            storage_df.to_csv(path_or_buf=storage_path)
            return storage_path
        else: return self.store_data(storage_path[:-4] + "1.csv")

    
    def __str__(self):
        return str(self.data)
    