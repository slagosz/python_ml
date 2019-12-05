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
    def __init__(self, volterra_model, boundary_radius=1, boundary_norm_order=2):
        self.volterra_model = volterra_model
        self.volterra_model.parameters = np.zeros(self.volterra_model.D)
        self.iteration = 0
        self.boundary_radius = boundary_radius
        self.boundary_norm_order = boundary_norm_order

    def update(self, x, y):
        self.iteration += 1
        if self.iteration > self.volterra_model.memory_length:
            gradient = self.compute_gradient(x, y)

    def compute_gradient(self, x, y):
        gradient = []
        for f in self.volterra_model.dictionary:
            gradient.append(
                (self.volterra_model.evaluate_output(x) - y) * f(x, -1)
            )
        return gradient

    def compute_stepsize(self):
        D = self.boundary_radius
        if self.boundary_norm_order == 2:
            G = 0  # todo

        return 0


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # m = VolterraModel(3, 2, 'legendre')
    # xx = [1, 2, 3]
    # for ff in m.dictionary:
    #     y = ff(xx, 2)
    #     print(y)

    # Experiment: calculate the maximum of expected sqrt( ||g||^2_2 )
    num_of_experiments = 1

    N = 100
    x = np.random.uniform(-1, 1, N)

    Order = 2
    Memory = 10
    y_sys = 1  # assume constant - this become M const. in notes
    boundary_norm_ord = 1
    boundary_radius = 1
    m = VolterraModel(Order, Memory)
    alg = OnlineGradientDescent(m, boundary_radius, boundary_norm_ord)

    avg_g = []
    for exp in range(num_of_experiments):
        m.parameters = np.random.uniform(-1, 1, m.D)
        m.parameters *= boundary_radius / np.linalg.norm(m.parameters, boundary_norm_ord)

        g = []
        for i in range(Memory-1, N):
            x_vec = x[0:i+1]
            g.append(np.linalg.norm(alg.compute_gradient(x_vec, y_sys), 2) ** (1/2))

        avg_g.append(np.average(g))

    max_avg_g = np.max(avg_g)


    sup_g = m.D ** (1/2) * (boundary_radius + y_sys)
    print("sup_g={0}, max_avg_g={1}".format(sup_g, max_avg_g))