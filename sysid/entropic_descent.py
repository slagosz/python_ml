from volterra import *
import copy


class EntropicDescentAlgorithm:
    def __init__(self, dictionary, R=1, constraint='simplex'):
        self.dictionary = copy.deepcopy(dictionary)

        if R is not 1:
            scale_dictionary(dictionary, R)

        if constraint == 'ball':
            extend_dictionary(self.dictionary)

        self.R = R
        self.D = dictionary.size
        self.constraint = constraint

    def compute_gradient(self, model, x, y, t):
        y_mod = model.evaluate_output(x, t=t)  # TODO check is it ok?
        gradient = np.zeros(model.dictionary.size)
        i = 0
        for f in model.dictionary.dictionary:
            gradient[i] = (y_mod - y[t]) * f(x, t)
            i += 1

        return gradient

    def run(self, x, y, stepsize_function):
        assert len(x) == len(y)

        model = DictionaryBasedModel(self.dictionary)

        theta_0 = np.ones(self.D) / self.D
        model.set_parameters(theta_0)

        T = len(x)

        theta_avg = theta_0

        for i in range(T):
            gradient = self.compute_gradient(x, y, i)
            stepsize = stepsize_function(i)

            theta_i = np.array(model.parameters) * np.exp(-stepsize * gradient)
            theta_i /= np.linalg.norm(theta_i, 1)

            theta_avg = (theta_i + theta_avg * (i + 1)) / (i + 2)

            assert bool(np.any(np.isnan(theta_avg))) is False  # check if none of numbers is NaN

            model.set_parameters(theta_i)

        if self.constraint == 'simplex':
            return theta_avg
        elif self.constraint == 'ball':
            return map_parameters_to_simplex(theta_avg, self.R)


def scale_dictionary(self, dictionary, R):
        for f_idx, f in enumerate(dictionary.dictionary):
            scaled_f = (lambda func: lambda x, t: R * func(x, t))(f)
            dictionary[f_idx] = scaled_f

def extend_dictionary(self, dictionary):
        for f in dictionary.dictionary:
            negative_fun = (lambda func: lambda x, t: -func(x, t))(f)
            dictionary.append(negative_fun)

def map_parameters_to_simplex(self, parameters, R):
        assert len(parameters) % 2 == 0
        D = int(len(parameters) / 2)

        transformed_parameters = np.zeros(D)

        for i in range(D):
            transformed_parameters[i] = ((parameters[i] - parameters[i + D]) * R)

        return transformed_parameters