import numpy as np
import itertools
import scipy.stats as stats
import scipy.special as special

POLYNOMIAL_LINEAR = 'linear'
POLYNOMIAL_QUADRATIC = 'quadratic'
POLYNOMIAL_CUBIC = 'cubic'
RANDOM_RBFS = 'random_gaussians'


class Model:

    def __init__(self, model_name, len_in, len_out=1, number_of_basis_functions=None, min_in=None, max_in=None):
        self.model_name = model_name
        self.len_in = len_in
        self.len_out = len_out
        self.number_of_parameters = None
        self.number_of_inner_parameters = None
        self.number_of_basis_functions = number_of_basis_functions
        # TODO: This gets handed max_actions atm
        # TODO: Remove hard-coding
        self.min_in = -1
        self.max_in = 1

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
            if (self.min_in is None) | (self.max_in is None):
                print("RANDOM_RBFS need a specified range of possible outputs")
            # Inner parameters contain multivariate mean
            self.number_of_inner_parameters = len_in * self.number_of_parameters

        else:
            print("Model name is not one of the designated model names!")
        self.parameters = None
        self.inner_parameters = None
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Let every basis function influence the outcome at first
        self.inner_parameters = np.zeros((self.number_of_inner_parameters, 1))
        if self.model_name == RANDOM_RBFS:
            input_range = self.max_in - self.min_in
            if self.number_of_basis_functions is None:
                for i in itertools.product([0, 1], repeat=self.len_in):
                    means_processed = 0.0
                    for j in range(0, self.len_in):
                        means_processed += i[j] * np.power(2, self.len_in - (j+1))
                    for j in range(0, self.len_in):
                        self.inner_parameters[int(means_processed * self.len_in + j)] = i[j] * input_range + self.min_in
            else:
                for i in range(0, self.number_of_parameters):
                    for j in range(0, self.len_in):
                        self.inner_parameters[i * self.len_in + j] = np.random.random() * input_range + self.min_in
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
            # Look at the size equidistributed cubes would need to have to fill the action space
            input_range = self.max_in - self.min_in
            dim = self.len_in
            if self.number_of_basis_functions is None:
                n = self.number_of_parameters - 1
            else:
                n = self.number_of_basis_functions
            # var = np.log(np.power(input_range, dim) / n) / np.log(dim)
            # TODO: Remove hard-coding
            var = 1
            # Respect the n-dimensional-cube-sphere-ratio
            volume_ratio = special.gamma((dim / 2) + 1) * np.power(2, dim) / np.power(np.pi, dim / 2)
            var = var * np.log(volume_ratio)/np.log(dim)
            for i in range(0, n):
                # for the guassian in the middle increase variance
                if (self.number_of_basis_functions is None) & (i == np.power(2, dim)):
                    var = np.sqrt(dim) * (input_range / 2)
                myu = self.inner_parameters[i*dim:(i+1)*dim]
                evaluated_basis_functions[i] = stats.multivariate_normal(mean=np.reshape(myu, (-1,)), cov=var).pdf(args)

        if self.model_name == RANDOM_RBFS:
            result = np.dot(self.parameters[:self.number_of_basis_functions], np.array(evaluated_basis_functions))
        else:
            result = np.dot(self.parameters, np.array(evaluated_basis_functions))
        return result


