import itertools
import numpy as np
from scipy.special import eval_legendre, eval_hermitenorm
import cvxpy as cvx


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


class VolterraDictionary:
    def __init__(self, order, memory_length, dict_type='standard', scaling_factor=1, include_constant_function=True):
        self.order = order
        self.memory_length = memory_length
        self.dict_type = dict_type
        self.dictionary_indices = []
        self.dictionary = []
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


class LinearModel:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.parameters = np.zeros(self.dictionary.size)

    def evaluate_output(self, x, t=-1):
        dict_output = [f(x, t) for f in self.dictionary.dictionary]

        return np.dot(self.parameters, dict_output)

    def set_parameters(self, parameters):
        assert len(parameters) == self.dictionary.size
        self.parameters = parameters


class VolterraModel(LinearModel):
    def __init__(self, order, memory_length, dict_type='standard', scaling_factor=1, include_constant_function=True):
        assert memory_length > 0

        dictionary = VolterraDictionary(order, memory_length, dict_type, scaling_factor=scaling_factor,
                                        include_constant_function=include_constant_function)
        LinearModel.__init__(self, dictionary)

        self.order = order
        self.memory_length = memory_length
        self.dict_type = dict_type
        self.D = self.dictionary.size

    def evaluate_output(self, x, x0=0, t=None):
        # if len(x) < self.memory_length:
        #     print('WARNING: the memory length is greater than the length of the input vector ({0} > {1})'.format(
        #         self.memory_length, len(x)
        #     ))

        if x0 == 0:
            x0 = np.zeros(self.memory_length - 1)
        else:
            x0 = np.array(x0)
            assert len(x0) == (self.memory_length - 1)

        x = np.concatenate([x0, x])

        if np.isscalar(t):
            t_list = [t]
        elif t is None:
            t_list = list(range(0, len(x)))
        else:
            t_list = t

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

        self.D *= 2


class LTISystem:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response
        self.memory_length = len(impulse_response)

    # assuming initial condition = 0
    def evaluate_output(self, x, x0=0):
        y = np.zeros(len(x))

        if x0 == 0:
            x0 = np.zeros(self.memory_length - 1)
        else:
            x0 = np.array(x0)
            assert len(x0) == (self.memory_length - 1)

        x = np.concatenate([x, x0])

        for t in range(len(y)):
            for i in range(len(self.impulse_response)):
                y[t] += self.impulse_response[i] * x[t - i]

        return y


class WienerHammerstein:
    def __init__(self, lti_in_impulse_response, f, lti_out_impulse_response):
        self.lti_in = LTISystem(lti_in_impulse_response)
        self.f = f
        self.lti_out = LTISystem(lti_out_impulse_response)

    def evaluate_output(self, x):
        y = self.lti_in.evaluate_output(x)
        y = self.f(y)
        y = self.lti_out.evaluate_output(y)

        return y


# creates a Volterra model of a Wiener-Hammerstein system given impulse response coefficients of linear subsystems
# and coefficients of polynomial nonlinear characteristic
def create_Volterra_model_of_WH_system(lti_in_impulse_response, f_coeffs, lti_out_impulse_response):
    # TODO
    pass


class GradientDescent:
    def __init__(self, volterra_model, stepsize_function, averaging=0, boundary_radius=1, boundary_norm_order=2):
        self.volterra_model = volterra_model
        self.volterra_model.set_parameters(np.zeros(self.volterra_model.D))
        self.iteration = 0
        self.boundary_radius = boundary_radius
        self.boundary_norm_order = boundary_norm_order
        self.stepsize_function = stepsize_function
        self.averaging = averaging
        self.parameters_history = []
        self.stepsizes_history = []

    def update(self, x, y):
        assert len(x) >= self.volterra_model.memory_length

        self.iteration += 1
        gradient = np.array(self.compute_gradient(x, y))
        stepsize = self.stepsize_function(self.iteration)

        parameters = np.array(self.volterra_model.parameters)
        parameters = parameters - stepsize * gradient
        parameters = self.project(parameters)
        if self.averaging != 0:
            self.parameters_history.append(parameters)
            self.stepsizes_history.append(stepsize)
            parameters = self.average()

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
        if norm <= self.boundary_radius:
            return parameters
        else:
            if self.boundary_norm_order == 1:
                x = cvx.Variable(len(parameters))
                cost = cvx.sum_squares(x - parameters)
                constr = [cvx.norm(x, 1) <= self.boundary_radius]
                prob = cvx.Problem(cvx.Minimize(cost))
                prob.solve()
                return x
            elif self.boundary_norm_order == 2:
                return parameters * self.boundary_radius / norm
            else:
                raise NotImplementedError('Projection onto l_{0} ball not implemented'.format(self.boundary_norm_order))

    def average(self):
        num_of_avg_observations = int(np.ceil(self.averaging * self.iteration))
        parameters = np.dot(self.stepsizes_history[-num_of_avg_observations:],
                            self.parameters_history[-num_of_avg_observations:])

        parameters /= sum(self.stepsizes_history[-num_of_avg_observations:])

        # remove unused values - memory optimization
        # del self.stepsizes_history[0:num_of_avg_observations]
        # del self.parameters_history[0:num_of_avg_observations]

        return parameters
