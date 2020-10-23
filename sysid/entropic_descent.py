from volterra import *
import copy
from tqdm import tqdm


def compute_gradient(model, x, y, t, x0=None):
    y_sys = y[t]

    if x0 is not None:
        x = np.concatenate([x0, x])
        t += model.dictionary.memory_length - 1

    # y_mod = model.evaluate_output(x, t=t)
    # gradient = np.zeros(model.dictionary.size)
    # i = 0

    # y_diff = y_mod - y_sys
    # for f in model.dictionary.dictionary:
    #     gradient[i] = y_diff * f(x, t)
    #     i += 1

    dict_output = [f(x, t) for f in model.dictionary.dictionary]
    y_mod = np.dot(model.parameters, dict_output)

    gradient = (y_mod - y_sys) * dict_output

    return gradient


class EntropicAlgorithm:
    def __init__(self, dictionary, R, constraint):
        self.dictionary = copy.deepcopy(dictionary)

        if R is not 1:
            scale_dictionary(self.dictionary, R)

        if constraint == 'ball':
            extend_dictionary(self.dictionary)

        self.R = R
        self.D = self.dictionary.size
        self.constraint = constraint


def scale_dictionary(dictionary, R):
    for f_idx, f in enumerate(dictionary.dictionary):
        scaled_f = (lambda func: lambda x, t: R * func(x, t))(f)
        dictionary.dictionary[f_idx] = scaled_f


def extend_dictionary(dictionary):
    redundant_functions = []

    for f in dictionary.dictionary:
        negative_fun = (lambda func: lambda x, t: -func(x, t))(f)
        redundant_functions.append(negative_fun)

    for f in redundant_functions:
        dictionary.append(f)


def map_parameters_to_simplex(parameters, R):
    assert len(parameters) % 2 == 0
    D = int(len(parameters) / 2)

    transformed_parameters = np.zeros(D)

    for i in range(D):
        transformed_parameters[i] = R * (parameters[i] - parameters[i + D])

    return transformed_parameters


class EntropicDescentAlgorithm(EntropicAlgorithm):
    def __init__(self, dictionary, R=1, constraint='simplex'):
        super().__init__(dictionary, R, constraint)

    def run(self, x, y, stepsize_function):
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)

        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        T = len(x)

        theta_avg = theta_0

        for i in range(T):
            gradient = compute_gradient(model, x, y, i)
            stepsize = stepsize_function(i, gradient)

            theta_i = np.array(model.parameters) * np.exp(-stepsize * gradient)
            theta_i /= np.linalg.norm(theta_i, 1)

            theta_avg = (theta_i + theta_avg * (i + 1)) / (i + 2)

            assert bool(np.any(np.isnan(theta_avg))) is False  # check if none of numbers is NaN

            model.set_parameters(theta_i)

        if self.constraint == 'simplex':
            return self.R * theta_avg
        elif self.constraint == 'ball':
            return map_parameters_to_simplex(theta_avg, self.R)


class AdaptiveEntropicDescentAlgorithm(EntropicDescentAlgorithm):
    def __init__(self, dictionary, R=1, constraint='simplex'):
        super().__init__(dictionary, R, constraint)

    def run(self, x, y, G_sq):
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)

        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        T = len(x)

        theta_avg = theta_0

        gradient_max_sq_sum = 0

        for i in range(T):
            gradient = compute_gradient(model, x, y, i)

            gradient_max_sq = np.max(gradient) ** 2
            gradient_max_sq_sum += gradient_max_sq

            stepsize = np.sqrt(np.log(self.D) / ((T - i - 1) * G_sq + gradient_max_sq_sum))

            print("         stepsize: {0}".format(stepsize))

            theta_i = np.array(model.parameters) * np.exp(-stepsize * gradient)
            theta_i /= np.linalg.norm(theta_i, 1)

            theta_avg = (theta_i + theta_avg * (i + 1)) / (i + 2)

            assert bool(np.any(np.isnan(theta_avg))) is False  # check if none of numbers is NaN

            model.set_parameters(theta_i)

        if self.constraint == 'simplex':
            return self.R * theta_avg
        elif self.constraint == 'ball':
            return map_parameters_to_simplex(theta_avg, self.R)


class EntropicDualAveragingAlgorithm(EntropicAlgorithm):
    def __init__(self, dictionary, R=1, constraint='simplex'):
        super().__init__(dictionary, R, constraint)

    def run(self, x, y, G_sq):
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)
        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        gradient_avg = 0
        T = len(x)

        theta_avg = 0

        for i in tqdm(range(T)):
            gradient_i = compute_gradient(model, x, y, i)
            gradient_avg = (gradient_i + gradient_avg * i) / (i + 1)

            stepsize = np.sqrt(2 * np.log(self.D) / (G_sq * T))

            theta_i = np.array(model.parameters) * np.exp(-stepsize * gradient_avg)
            theta_i /= np.linalg.norm(theta_i, 1)

            model.set_parameters(theta_i)

            theta_avg = (theta_i + theta_avg * i) / (i + 1)

        if self.constraint == 'simplex':
            return self.R * theta_avg
        elif self.constraint == 'ball':
            return map_parameters_to_simplex(theta_avg, self.R)


class AdaptiveEntropicDualAveragingAlgorithm(EntropicAlgorithm):
    def __init__(self, dictionary, R=1, constraint='simplex'):
        super().__init__(dictionary, R, constraint)
        self.stepsize_seq = []

    def run(self, x, y, G_sq, x0=None, adaptive_stepsize=True):
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)
        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        gradient_sum = 0
        gradient_max_sum = 0
        T = len(x)

        theta_avg = 0

        for i in tqdm(range(T)):
            gradient_i = compute_gradient(model, x, y, i, x0=x0)

            stepsize = 0
            if adaptive_stepsize:
                stepsize = np.sqrt(np.log(self.D) / (G_sq + gradient_max_sum))
            else:
                stepsize = np.sqrt(np.log(self.D) / (G_sq * (i+1)))

            self.stepsize_seq.append(stepsize)  # for illustration

            gradient_sum += gradient_i
            gradient_max_sum += max(gradient_i)

            theta_i = np.exp(-stepsize * gradient_sum)
            theta_i /= np.linalg.norm(theta_i, 1)

            model.set_parameters(theta_i)

            theta_avg = (theta_i + theta_avg * i) / (i + 1)

        if self.constraint == 'simplex':
            return self.R * theta_avg
        elif self.constraint == 'ball':
            return map_parameters_to_simplex(theta_avg, self.R)
