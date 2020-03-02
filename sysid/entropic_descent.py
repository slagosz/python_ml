from volterra import *


class EntropicDescentAlgorithm:
    def __init__(self, model, stepsize_function):
        self.model = model
        self.stepsize_function = stepsize_function
        self.D = model.dictionary.size

        # debug
        self.max_g_sq = np.zeros(self.D)

    def compute_gradient(self, x, y, t):
        y_mod = self.model.evaluate_output(x, t=t)  # TODO check is it ok?
        gradient = np.zeros(self.D)
        i = 0
        for f in self.model.dictionary.dictionary:
            gradient[i] = (y_mod - y[t]) * f(x, t)
            i += 1

        return gradient

    def run(self, x, y):
        assert len(x) == len(y)

        theta_0 = np.ones(self.D) / self.D
        self.model.set_parameters(theta_0)

        T = len(x)

        theta_avg = theta_0

        for i in range(T):
            gradient = self.compute_gradient(x, y, i)
            stepsize = self.stepsize_function(i)

            theta_i = np.array(self.model.parameters) * np.exp(-stepsize * gradient)
            theta_i /= np.linalg.norm(theta_i, 1)

            theta_avg = (theta_i + theta_avg * (i + 1)) / (i + 2)

            assert bool(np.any(np.isnan(theta_avg))) is False  # check if none of numbers is NaN

            self.model.set_parameters(theta_i)

            # debug
            self.max_g_sq = np.maximum(self.max_g_sq, gradient)

        return theta_avg


def map_parameters(parameters, R):
    assert len(parameters) % 2 == 0
    D = int(len(parameters) / 2)

    transformed_parameters = np.zeros(D)

    for i in range(D):
        transformed_parameters[i] = ((parameters[i] - parameters[i + D]) * R)

    return transformed_parameters
