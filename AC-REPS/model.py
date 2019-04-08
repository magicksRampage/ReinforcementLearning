import numpy as np
import time
import itertools
import scipy.stats as stats
import scipy.special as special

POLYNOMIAL_LINEAR = 'linear'
POLYNOMIAL_QUADRATIC = 'quadratic'
POLYNOMIAL_CUBIC = 'cubic'
RANDOM_RBFS = 'random_gaussians'


class Model:

    def __init__(self, model_name, len_in, len_out=1, number_of_basis_functions=None):
        self.model_name = model_name
        self.len_in = len_in
        self.len_out = len_out
        self.number_of_parameters = None
        self.number_of_inner_parameters = None
        self.number_of_basis_functions = number_of_basis_functions

        if self.model_name == POLYNOMIAL_LINEAR:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.number_of_parameters = len_in
            self.number_of_basis_functions = self.number_of_parameters

        elif self.model_name == POLYNOMIAL_QUADRATIC:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.number_of_parameters = len_in + np.power(len_in, 2)
            self.number_of_basis_functions = self.number_of_parameters

        elif self.model_name == POLYNOMIAL_CUBIC:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.number_of_parameters = len_in + np.power(len_in, 2) + np.power(len_in, 3)
            self.number_of_basis_functions = self.number_of_parameters

        elif self.model_name == RANDOM_RBFS:
            if self.number_of_basis_functions is None:
                print("No number of basis_functions has been chosen for RBFS.")
                print("The model will try to evenly tile the input-space.")
                print("If your input-space is high-dimensional this will produce many(!) features.")
                # Produce a feature for every "corner" of the space and one for the center
                self.number_of_parameters = np.power(2, self.len_in) + 1
            else:
                self.number_of_parameters = self.number_of_basis_functions
            # Inner parameters contain multivariate means and one general multivariate variance
            self.number_of_inner_parameters = len_in * self.number_of_parameters + len_in

        else:
            print("Model name is not one of the designated model names!")
        self.parameters = None
        self.inner_parameters = None
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Let every basis function influence the outcome at first
        if self.model_name == RANDOM_RBFS:
            self.inner_parameters = np.ones((self.number_of_inner_parameters, 1))
            if self.number_of_basis_functions is None:
                # Place Gaussians on the corners of the space
                for i in itertools.product([0, 1], repeat=self.len_in):
                    means_processed = 0.0
                    for j in range(0, self.len_in):
                        means_processed += i[j] * np.power(2, self.len_in - (j+1))
                    for j in range(0, self.len_in):
                        self.inner_parameters[int(means_processed * self.len_in + j)] = i[j]
                # Place one gaussian over the center of the space
                for i in range(0, self.len_in):
                    self.inner_parameters[-(i+1)] = 0.5
            else:
                for i in range(0, self.number_of_parameters):
                    for j in range(0, self.len_in):
                        self.inner_parameters[i * self.len_in + j] = np.random.random()
        # Let every basis function influence the outcome at first
        self.parameters = np.ones((self.number_of_parameters, 1))

    def evaluate(self, args):
        evaluated_basis_functions = np.zeros(np.shape(self.parameters))
        if self.model_name == POLYNOMIAL_LINEAR:
            for i in range(0, self.len_in):
                evaluated_basis_functions[i] = args[i]

        elif self.model_name == POLYNOMIAL_QUADRATIC:
            for i in range(0, self.len_in):
                for j in range(0, self.len_in):
                    evaluated_basis_functions[i] = args[i] * args[j]

        elif self.model_name == POLYNOMIAL_CUBIC:
            for i in range(0, self.len_in):
                for j in range(0, self.len_in):
                    for k in range(0, self.len_in):
                        evaluated_basis_functions[i] = args[i] * args[j] * args[k]

        elif self.model_name == RANDOM_RBFS:
            # Calculate variance so that RBFS should cover most of the space
            # Assume a gaussian is mostly relevant inside of one standard-deviation
            # Look at the size equidistributed cubes would need to have to fill the action
            dim = self.len_in
            if self.number_of_basis_functions is None:
                n = self.number_of_parameters - 1
            else:
                n = self.number_of_basis_functions
            # Interpret the space under one sigma as the relevant support for a gaussian
            # Now adjust the support that each gaussian "covers one corner of your space"
            in_var = self.inner_parameters[-(dim+1):-1]
            for i in range(0, n):
                mean = self.inner_parameters[i*dim:(i+1)*dim]
                # Using the pseudo-multi-variate gaussian improves speed drastically
                evaluated_basis_functions[i] = self._evaluate_pseudo_mv_gaussian(args, mean, in_var)

        result = np.dot(self.parameters, np.array(evaluated_basis_functions))
        return result

    def _evaluate_pseudo_mv_gaussian(self, args, mean, inverted_var):
        # Ignore any constant parts of a multivariate gaussian as the parameters adopt anyway
        diff_vec = np.reshape(args, (-1,1)) - np.reshape(mean, (-1,1))
         # Handle the variances as if the covariance-matrix had already been inverted. Optimization will adapt
        inv_covar = np.diag(np.reshape(inverted_var,(-1,)))
        result = np.matmul(np.matmul(np.transpose(diff_vec), inv_covar), diff_vec) [0]
        return result

