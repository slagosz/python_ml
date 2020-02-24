from volterra import *
import numba.cuda as cuda


class EntropicDescentAlgorithm:
    def __init__(self, model, stepsize_function):
        self.model = model
        self.stepsize_function = stepsize_function

    def compute_gradient(self, x, y):
        gradient = []
        y_mod = self.model.evaluate_output(x)
        for f in self.model.dictionary.dictionary:
            gradient.append(
                (y_mod - y) * f(x, -1)
            )
        return gradient

    def parallel(self):
        assert cuda.is_available()

    def run(self, x, y):
        assert len(x) == len(y)
        assert len(x) >= self.model.memory_length

        memory_offset = self.model.memory_length - 1

        T = len(x) - memory_offset

        theta_0 = np.ones(self.model.D) / self.model.D
        self.model.set_parameters(theta_0)

        for i in range(T):
            data_index = memory_offset + i
            x_sliced = x[:data_index + 1]
            gradient = np.array(self.compute_gradient(x_sliced, y[data_index]))
            stepsize = self.stepsize_function(i)

            theta_i = np.array(self.model.parameters) * np.exp(-stepsize * gradient)
            theta_i /= np.linalg.norm(theta_i, 1)

            theta_avg = (theta_i + self.model.parameters * i) / (i + 1)

            self.model.set_parameters(theta_avg)

        return self.model.parameters


def map_parameters(parameters, R):
    assert len(parameters) % 2 == 0
    D = int(len(parameters) / 2)

    transformed_parameters = np.zeros(D)

    for i in range(D):
        transformed_parameters[i] = ((parameters[i] - parameters[i + D]) * R)

    return transformed_parameters
