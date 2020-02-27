import numpy as np


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


class DictionaryBasedModel:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.parameters = np.zeros(self.dictionary.size)

    def evaluate_output(self, x, t=-1):
        dict_output = [f(x, t) for f in self.dictionary.dictionary]

        return np.dot(self.parameters, dict_output)

    def set_parameters(self, parameters):
        assert len(parameters) == self.dictionary.size
        self.parameters = parameters
