import numpy as np
import scipy.stats as stats

POLYNOMIAL_LINEAR = 'linear'
POLYNOMIAL_QUADRATIC = 'quadratic'
POLYNOMIAL_CUBIC = 'cubic'
RBF = 'gaussian'


class Model:

    def __init__(self, model_name, len_in, len_out=1, number_of_basis_functions=None):
        self.model_name = model_name
        self.len_in = len_in
        self.len_out = len_out
        self.len_parameters = None
        self.number_of_inner_parameters = None
        self.number_of_basis_functions = number_of_basis_functions

        if self.model_name == POLYNOMIAL_LINEAR:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.len_parameters = len_in
            self.number_of_basis_functions = self.len_parameters

        elif self.model_name == POLYNOMIAL_QUADRATIC:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.len_parameters = len_in + np.power(len_in, 2)
            self.number_of_basis_functions = self.len_parameters

        elif self.model_name == POLYNOMIAL_CUBIC:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.len_parameters = len_in + np.power(len_in, 2) + np.power(len_in, 3)
            self.number_of_basis_functions = self.len_parameters

        elif self.model_name == RBF:
            if self.number_of_basis_functions is None:
                print("A number of basis functions must be specified for RBFs")
            self.len_parameters = 3 * self.number_of_basis_functions
            # Inner parameters contain mean and a scalar variance
            self.number_of_inner_parameters = 2

        else:
            print("Model name is not one of the designated model names!")
        self.parameters = None
        self._initialize_parameters()

    def _initialize_parameters(self):
        # Let every basis function influence the outcome at first
        self.parameters = np.ones((self.len_parameters, 1))
        if self.model_name == RBF:
            # The outer parameters start normally
            for i in range(0, self.len_in):
                self.parameters[i] = 1.0
                # Scatter the mean a little bit
                self.parameters[i + self.number_of_basis_functions] = np.random.random() - 0.5
                # Variance starts at one
                self.parameters[i + 2 * self.number_of_basis_functions] = 1.0
        # Let every basis function influence the outcome at first
        self.parameters = np.ones((self.len_parameters, 1))

    def evaluate(self, args):
        if self.model_name:
            evaluated_basis_functions = np.zeros((self.number_of_basis_functions, 1))
        else:
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

        elif self.model_name == RBF:
            for i in range(0, self.number_of_basis_functions):
                myu = np.ones((self.len_in, )) * self.parameters[i + self.number_of_basis_functions]
                evaluated_basis_functions[i] = stats.multivariate_normal(mean=myu,
                                                                         cov=self.parameters[i + 2*self.number_of_basis_functions],
                                                                         ).pdf(args)

        if self.model_name == RBF:
            result = np.dot(self.parameters[:self.number_of_basis_functions], np.array(evaluated_basis_functions))
        else:
            result = np.dot(self.parameters, np.array(evaluated_basis_functions))
        return result


