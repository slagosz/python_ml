import itertools
import numpy as np
from scipy.special import eval_legendre, eval_hermitenorm
from models import *


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


def legendre_normalized_function(indices, x, t=-1):
    output = 1
    unique_indices = set(indices)
    for i in unique_indices:
        order = indices.count(i)
        norm_factor = np.sqrt(2 * order + 1)
        output *= norm_factor * eval_legendre(order, x[t - i])
    return output


class VolterraDictionary(Dictionary):
    def __init__(self, kernels, dict_type='standard', scaling_factor=1, include_constant_function=True):
        super().__init__()
        self.kernels = kernels
        self.dict_type = dict_type
        self.dictionary_indices = []
        self.scaling_factor = scaling_factor
        self.include_constant_function = include_constant_function

        self.generate_dictionary()
        self.size = len(self.dictionary)

        if not kernels:
            self.memory_length = 1
        else:
            self.memory_length = np.max(kernels)

    @staticmethod
    def generate_indices(order, memory_length):
        return itertools.combinations_with_replacement(range(0, memory_length), order)

    def generate_dictionary(self):
        func = {
            'standard': volterra_function,
            'hermite': hermite_function,
            'legendre': legendre_function,
            'legendre_norm': legendre_normalized_function
        }[self.dict_type]

        self.dictionary = []
        self.dictionary_indices = []

        if self.include_constant_function:
            self.dictionary.append(lambda x, t: 1)  # constant function
            self.dictionary_indices.append([])

        order = 1
        for memory_length in self.kernels:
            indices = self.generate_indices(order, memory_length)
            for ind in indices:
                # closure hack https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                f = (lambda i: lambda x, t: self.scaling_factor * func(i, x, t))(ind)
                self.dictionary.append(f)
                self.dictionary_indices.append(ind)
            order += 1

        # for order in range(self.order):
        #     indices = self.generate_indices(order + 1)
        #     for ind in indices:
        #         # closure hack https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
        #         f = (lambda i: lambda x, t: self.scaling_factor * func(i, x, t))(ind)
        #         self.dictionary.append(f)
        #         self.dictionary_indices.append(ind)


class VolterraModel(DictionaryBasedModel):
    def __init__(self, order=None, memory_length=None, kernels=None, dict_type='standard', scaling_factor=1,
                 include_constant_function=True):

        if kernels is None:
            assert order is not None
            assert memory_length is not None
            kernels = [memory_length] * order

        dictionary = VolterraDictionary(kernels, dict_type, scaling_factor=scaling_factor,
                                        include_constant_function=include_constant_function)
        DictionaryBasedModel.__init__(self, dictionary)

        if not kernels:
            self.memory_length = 1
        else:
            self.memory_length = np.max(kernels)

        self.kernels = kernels
        self.dict_type = dict_type
        self.D = self.dictionary.size

    def evaluate_output(self, x, x0=None, t=None):
        if x0 is None:
            x0 = np.zeros(self.memory_length - 1)
        else:
            x0 = np.array(x0)
            assert len(x0) == (self.memory_length - 1)

        if np.isscalar(t):
            t_list = [t]
        elif t is None:
            t_list = list(range(0, len(x)))
        else:
            t_list = t

        x = np.concatenate([x0, x])
        y = np.zeros(len(t_list))

        i = 0
        for t in t_list:
            y[i] = super().evaluate_output(x, t + self.memory_length - 1)
            i += 1

        return y
