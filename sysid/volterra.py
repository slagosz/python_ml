import itertools
import numpy as np
from scipy.special import eval_legendre, eval_hermitenorm


def volterra_function(indices, x, t=-1):
    output = 1
    for i in indices:
        output *= x[t - i]
    return output


def hermite_function(indices, x, t=-1):
    output = 1
    unique_indices = set(indices)
    for i in unique_indices:
        order = indices.count(i)
        output *= eval_hermitenorm(order, x[t - i])
    return output


def legendre_function(indices, x, t=-1):
    output = 1
    unique_indices = set(indices)
    for i in unique_indices:
        order = indices.count(i)
        output *= eval_legendre(order, x[t - i])
    return output


class VolterraModel:
    def __init__(self, order, memory_length, dict_type='standard'):
        self.order = order
        self.memory_length = memory_length
        self.dict_type = dict_type
        self.dictionary_indices = []
        self.dictionary = []
        self.generate_dictionary()
        self.D = len(self.dictionary)
        self.parameters = np.zeros(self.D)

    def generate_indices(self, order):
        return itertools.combinations_with_replacement(range(0, self.memory_length), order)

    def generate_dictionary(self):
        func = {
            'standard': volterra_function,
            'hermite': hermite_function,
            'legendre': legendre_function
        }[self.dict_type]

        self.dictionary = [lambda x, t: 1]  # constant function
        self.dictionary_indices = [[]]
        for order in range(self.order):
            indices = self.generate_indices(order + 1)
            for ind in indices:
                # closure hack https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                f = (lambda i: lambda x, t: func(i, x, t))(ind)
                self.dictionary.append(f)
                self.dictionary_indices.append(ind)

    def evaluate_output(self, x, t=-1):
        if len(x) < self.memory_length:
            print('WARNING: the memory length is greater than the length of the input vector ({0} > {1})'.format(
                self.memory_length, len(x)
            ))
        dict_output = [f(x, t) for f in self.dictionary]
        return np.dot(self.parameters, dict_output)

    def set_parameters(self, parameters):
        assert len(parameters) == self.D
        self.parameters = parameters


class OnlineGradientDescent:
    def __init__(self, volterra_model, stepsize_function, boundary_radius=1, boundary_norm_order=2):
        self.volterra_model = volterra_model
        self.volterra_model.parameters = np.zeros(self.volterra_model.D)
        self.iteration = 0
        self.boundary_radius = boundary_radius
        self.boundary_norm_order = boundary_norm_order
        self.stepsize_function = stepsize_function

    def update(self, x, y):
        assert len(x) >= self.volterra_model.memory_length

        self.iteration += 1
        gradient = np.array(self.compute_gradient(x, y))

        parameters = np.array(self.volterra_model.parameters)
        parameters = parameters - self.stepsize_function(self.iteration) * gradient
        self.project(parameters)
        self.volterra_model.set_parameters(parameters)

    def compute_gradient(self, x, y):
        gradient = []
        for f in self.volterra_model.dictionary:
            gradient.append(
                (self.volterra_model.evaluate_output(x) - y) * f(x, -1)
            )
        return gradient

    def project(self, parameters):
        norm = np.linalg.norm(parameters, self.boundary_norm_order)
        if norm > self.boundary_radius:
            if self.boundary_norm_order == 2:
                # TODO make sure it's valid
                parameters = parameters * self.boundary_radius / norm
            else:
                raise Exception('Projection onto l_{0} ball not implemented'.format(self.boundary_norm_order))
