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


class VolterraDictionary(Dictionary):
    def __init__(self, order, memory_length, dict_type='standard', scaling_factor=1, include_constant_function=True):
        super().__init__()
        self.order = order
        self.memory_length = memory_length
        self.dict_type = dict_type
        self.dictionary_indices = []
        self.scaling_factor = scaling_factor
        self.include_constant_function = include_constant_function

        self.generate_dictionary()
        self.size = len(self.dictionary)

    def generate_indices(self, order):
        return itertools.combinations_with_replacement(range(0, self.memory_length), order)

    def generate_dictionary(self):
        func = {
            'standard': volterra_function,
            'hermite': hermite_function,
            'legendre': legendre_function
        }[self.dict_type]

        self.dictionary = []
        self.dictionary_indices = []

        if self.include_constant_function:
            self.dictionary.append(lambda x, t: 1)  # constant function
            self.dictionary_indices.append([])

        for order in range(self.order):
            indices = self.generate_indices(order + 1)
            for ind in indices:
                # closure hack https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                f = (lambda i: lambda x, t: self.scaling_factor * func(i, x, t))(ind)
                self.dictionary.append(f)
                self.dictionary_indices.append(ind)


class VolterraModel(DictionaryBasedModel):
    def __init__(self, order, memory_length, dict_type='standard', scaling_factor=1, include_constant_function=True):
        assert memory_length > 0

        dictionary = VolterraDictionary(order, memory_length, dict_type, scaling_factor=scaling_factor,
                                        include_constant_function=include_constant_function)
        DictionaryBasedModel.__init__(self, dictionary)

        self.order = order
        self.memory_length = memory_length
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


# this model contains redundant dictionary elements,
# as exponential weights algorithm constraint is a positive subset of l1 ball
class VolterraModelForExpWeights(VolterraModel):
    def __init__(self, order, memory_length, dict_type='standard', scaling_factor=1):
        VolterraModel.__init__(self, order, memory_length, dict_type, scaling_factor)
        self.extend_dictionary()
        self.parameters = np.zeros(self.D)

    def extend_dictionary(self):
        for f_idx in range(self.D):
            redundant_fun = (lambda i: lambda x, t: -self.dictionary.dictionary[i](x, t))(f_idx)
            self.dictionary.dictionary.append(redundant_fun)

        self.dictionary.size *= 2
