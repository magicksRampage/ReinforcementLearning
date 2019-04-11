import numpy as np
import time
import itertools
import scipy.stats as stats
import scipy.special as special

POLYNOMIAL_LINEAR = 'linear'
POLYNOMIAL_QUADRATIC = 'quadratic'
POLYNOMIAL_CUBIC = 'cubic'
RBFS = 'gaussians'


class Model:
    """
    The abstraction layer for any linear model used in the programm.
    At the moment there are 5 possible Models:
        - a polynomial linear model
        - a polynomial quadratic model
        - a polynomial cubic model
        - a model of random RBFS (if you specify a number of basis functions)
        - a model of soft-tiling RBFS (if you don't specify a number of basis functions)

    To this point the program can only handle a number of approx. 10 or fewer parameters,
    otherwise the computation time become overbearing.
    Thus for state-dimensions < 3 only the polynomial linear model is useful.
    (So far the polynomial linear model doesn't seem expressive enough for the environment 'Qube')

    Attributes:
        model_name (String): The constant String defining the type of the model
        len_in (int): the size (or number of dimensions) of the input
        len_out (int): the size (or number of dimensions) of the output
                 (At the moment only one-dimensional outputs are supported)
        number_of_parameters (int): the number of parameters of a general linear model
        number_of_inner_parameters (int): the number of additional parameters (e.g. mean and variance of RBFS)
        number_of_basis_functions (int): number of basis_functions if random tiling is chosen
    """

    def __init__(self, model_name, len_in, len_out=1, number_of_basis_functions=None):
        """
        :param model_name (String): The constant String defining the type of the model
        :param len_in (int): the size (or number of dimensions) of the input
        :param len_out (int): the size (or number of dimensions) of the output
        :param number_of_basis_functions (int): amount of basis functions to be chosen
        """
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
            self.number_of_parameters = np.power(len_in, 2) + len_in
            self.number_of_basis_functions = self.number_of_parameters

        elif self.model_name == POLYNOMIAL_CUBIC:
            if self.number_of_basis_functions is not None:
                print("The number of basis functions for polynomial models are fixed")
            self.number_of_parameters = np.power(len_in, 3) + np.power(len_in, 2) + len_in
            self.number_of_basis_functions = self.number_of_parameters

        elif self.model_name == RBFS:
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
        """
        Initializes the outer and inner parameters

        Updates:
            inner_parameters
            parameters

        Calls:
            itertools.product
            np.random.random

        :return: None
        """
        if self.model_name == RBFS:
            self.inner_parameters = np.ones((self.number_of_inner_parameters, 1))
            if self.number_of_basis_functions is None:
                # Place Gaussians on the corners of the space
                for i in itertools.product([0, 1], repeat=self.len_in):
                    means_processed = 0.0
                    for j in range(0, self.len_in):
                        means_processed += i[j] * np.power(2, self.len_in - (j+1))
                    for j in range(0, self.len_in):
                        self.inner_parameters[int(means_processed * self.len_in + j)] = i[j] * 2 - 1
                # Place one gaussian over the center of the space
                for i in range(0, self.len_in):
                    self.inner_parameters[-(i+1)] = 0
            else:
                for i in range(0, self.number_of_parameters):
                    for j in range(0, self.len_in):
                        self.inner_parameters[i * self.len_in + j] = np.random.random() * 2 - 1
        self.parameters = np.zeros((self.number_of_parameters, 1))

    def evaluate(self, args):
        """
        Evaluates the model for a single input

        :param args (n x 1): contains the input values

        Calls:
            _evaluate_pseudo_mv_gaussian

        :return: the output of the model
        """
        evaluated_basis_functions = np.zeros(np.shape(self.parameters))
        if (self.model_name == POLYNOMIAL_LINEAR) \
           | (self.model_name == POLYNOMIAL_QUADRATIC) \
           | (self.model_name == POLYNOMIAL_CUBIC):
            for i in range(0, self.len_in):
                evaluated_basis_functions[i] = args[i]

        if (self.model_name == POLYNOMIAL_QUADRATIC) \
           | (self.model_name == POLYNOMIAL_CUBIC):
            for i in range(0, self.len_in):
                for j in range(0, self.len_in):
                    evaluated_basis_functions[(i+1)*self.len_in + j] = args[i] * args[j]

        if self.model_name == POLYNOMIAL_CUBIC:
            for i in range(0, self.len_in):
                for j in range(0, self.len_in):
                    for k in range(0, self.len_in):
                        evaluated_basis_functions[(i+1) * (self.len_in ** 2) + j*self.len_in + k] = args[i] * args[j] * args[k]

        elif self.model_name == RBFS:
            dim = self.len_in
            if self.number_of_basis_functions is None:
                n = self.number_of_parameters - 1
            else:
                n = self.number_of_basis_functions
            in_var = self.inner_parameters[-(dim+1):-1]
            for i in range(0, n):
                mean = self.inner_parameters[i*dim:(i+1)*dim]
                # Using the pseudo-multi-variate gaussian improves speed drastically
                evaluated_basis_functions[i] = self._evaluate_pseudo_mv_gaussian(args, mean, in_var)

        result = np.dot(self.parameters, np.array(evaluated_basis_functions))
        return result

    def _evaluate_pseudo_mv_gaussian(self, args, mean, inverted_var):
        """
        Evaluates a multivariate gaussian at for a given argument vector

        :param args (n x 1): vector containing the values for each dimension
        :param mean (n x 1): the mean of the multivariate
        :param inverted_var (double): a value to be broadcasted over a diagonal matrix which represents the inverted
                                      covariance matrix. Optimizing directly on the inverted covariance matrix should be
                                      just as viable, more efficient and should prevent numeric issues

        :return: the value of the multivariate gaussian at the point specified by "args"
        """
        # Ignore any constant parts of a multivariate gaussian as the parameters adopt anyway
        diff_vec = np.reshape(args, (-1, 1)) - np.reshape(mean, (-1, 1))
        # Handle the variances as if the covariance-matrix had already been inverted. Optimization will adapt
        inv_covar = np.diag(np.reshape(inverted_var, (-1,)))
        result = np.matmul(np.matmul(np.transpose(diff_vec), inv_covar), diff_vec)[0]
        return result

