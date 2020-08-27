import unittest
from volterra import *
from entropic_descent import *


class TestVolterraFunction(unittest.TestCase):
    def test_volterra_function_basic(self):
        x = [2, 3]
        self.assertEqual(volterra_function([0], x), 3)
        self.assertEqual(volterra_function([1], x), 2)
        self.assertEqual(volterra_function([0, 0], x), 3 * 3)
        self.assertEqual(volterra_function([1, 1], x), 2 * 2)
        self.assertEqual(volterra_function([0, 1], x), 3 * 2)
        self.assertEqual(volterra_function([1, 0], x), 3 * 2)

    def test_volterra_function_time_shift(self):
        x = [1, 2, 3, 4]
        self.assertEqual(volterra_function([0], x, 0), 1)
        self.assertEqual(volterra_function([0], x, 1), 2)
        self.assertEqual(volterra_function([0], x, 2), 3)
        self.assertEqual(volterra_function([0], x, 3), 4)
        self.assertEqual(volterra_function([1], x, 1), 1)
        self.assertEqual(volterra_function([2], x, 2), 1)
        self.assertEqual(volterra_function([2], x, 3), 2)

    def test_volterra_function_too_short_input(self):
        x = [1]
        self.assertRaises(IndexError, volterra_function, [1], x)


class TestVolterraModel(unittest.TestCase):
    def test_volterra_model_const(self):
        m = VolterraModel(order=0, memory_length=0)
        self.assertEqual(m.D, 1)
        m.set_parameters([3])
        x = [0, 1, 2]
        y = m.evaluate_output(x, t=2)
        self.assertEqual(m.evaluate_output(x, t=2), 3)

    def test_volterra_model_basic(self):
        order = 1
        memory_len = 2
        m = VolterraModel(order=order, memory_length=memory_len)

        self.assertEqual(m.D, 3)

        m.set_parameters([0, 1, 2])
        x = [1, 2, 3]
        self.assertEqual(m.evaluate_output(x, t=1), 1 * 2 + 2 * 1)
        self.assertEqual(m.evaluate_output(x, t=2), 1 * 3 + 2 * 2)

    def test_volterra_model_higher_order(self):
        order = 2
        memory_len = 2
        m = VolterraModel(order=order, memory_length=memory_len)

        self.assertEqual(m.D, 6)

        m.set_parameters([0.5, 1, 2, 3, 4, -5])
        x = [1, 2, 3]
        self.assertEqual(m.evaluate_output(x, t=2), 0.5 + 1 * 3 + 2 * 2 + 3 * 3 * 3 + 4 * 3 * 2 - 5 * 2 * 2)

    def test_volterra_model_different_kernels(self):
        kernels = [2, 1]
        m = VolterraModel(kernels=kernels, include_constant_function=False)

        self.assertEqual(m.D, 3)

        m.set_parameters([3, 2, 1])
        x = [1, 2, 3]
        self.assertEqual(m.evaluate_output(x, t=2), 3 * 3 + 2 * 2 + 1 * 3 * 3)

'''
class TestOnlineGradientDescent(unittest.TestCase):
    def test_gradient_computation(self):
        order = 2
        memory_len = 2
        m = VolterraModel(order, memory_len)
        alg = GradientDescent(m)

        m.set_parameters([0.5, 1, 2, 3, 4, -5])
        x = [1, 2, 3]
        y_mod = m.evaluate_output(x)
        y_observed = 1
        grad = alg.compute_gradient(x, y_observed)

        self.assertEqual(len(grad), m.D)
        # grad[i] = (y_mod - y_observed) * f_i(x)
        i = 0
        for ind in m.dictionary.dictionary_indices:
            self.assertEqual(grad[i], (y_mod - y_observed) * volterra_function(ind, x))
            i += 1
'''


class TestOtherModels(unittest.TestCase):
    def test_lti_model(self):
        imp_resp = [3, 2, 1]
        x = [10, 5, -1]
        lti = LTISystem(imp_resp)

        y = lti.evaluate_output(x)

        self.assertEqual(len(y), 3)
        self.assertEqual(y[0], 30)
        self.assertEqual(y[1], 35)
        self.assertEqual(y[2], 17)

    def test_Wiener_Hammerstein_model(self):
        imp_resp_in = [2, 1]
        imp_resp_out = [1, 1]
        f = lambda x: x * 2

        wh = WienerHammerstein(imp_resp_in, f, imp_resp_out)

        x = [10, 1]
        y = wh.evaluate_output(x)

        self.assertEqual(len(y), 2)
        self.assertEqual(y[0], 40)
        self.assertEqual(y[1], 64)

    def test_dictionary_model(self):
        dictionary = Dictionary()
        dictionary.append(lambda x, t: x[t])

        m = DictionaryBasedModel(dictionary)
        m.set_parameters([1])

        x = [1, 2, 3]
        y = m.evaluate_output(x)

        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)
        self.assertEqual(y[2], 3)

        y = m.evaluate_output(x, t=2)
        self.assertEqual(y, 3)

    def test_scaling_and_extending_dictionary(self):
        dictionary = Dictionary()
        dictionary.append(lambda x, t: 1)
        dictionary.append(lambda x, t: x[t])

        R = 2
        scale_dictionary(dictionary, R)
        self.assertEqual(dictionary.size, 2)

        m = DictionaryBasedModel(dictionary)
        m.set_parameters([1, 1])

        x = [2]
        y = m.evaluate_output(x)

        y = m.evaluate_output(x)
        self.assertEqual(y, R * 3)

        extend_dictionary(dictionary)
        self.assertEqual(dictionary.size, 4)

        m = DictionaryBasedModel(dictionary)
        m.set_parameters([1, 1, 0, 0.5])

        y = m.evaluate_output(x)
        self.assertEqual(y, R + 2 * R - R)


class TestEntropicDescent(unittest.TestCase):
    @staticmethod
    def generate_system(sys_parameters):
        sys_dict = Dictionary()
        sys_dict.append(lambda x, t: 1)
        sys_dict.append(lambda x, t: x[t])

        true_sys = DictionaryBasedModel(sys_dict)
        true_sys.set_parameters(sys_parameters)

        return true_sys

    def test_algorithm_not_affecting_dictionary(self):
        sys_dict = Dictionary()
        sys_dict.append(lambda x, t: 1)

        alg = EntropicDescentAlgorithm(sys_dict, R=3, constraint='ball')

        self.assertEqual(sys_dict.size, 1)
        self.assertEqual(sys_dict.dictionary[0]([1], 0), 1)

    def test_ed(self):
        sys_dict = Dictionary()
        sys_dict.append(lambda x, t: 1)
        sys_dict.append(lambda x, t: x[t])

        D = sys_dict.size

        alg = EntropicDescentAlgorithm(sys_dict, R=1, constraint='simplex')

        stepsize_function = lambda i, gradient: 1

        ###
        params = alg.run([], [], stepsize_function)
        self.assertEqual(params.size, D)
        self.assertEqual(params[0], 1/D)
        self.assertEqual(params[1], 1/D)

        ###
        x = np.array([1, 0, -1])
        y = np.array([2, 0.5, 1])
        params = alg.run(x, y, stepsize_function)

        p0 = np.array([1/2, 1/2])
        p1 = np.array([1/2, 1/2])
        p2 = np.array([1/2, 1/2])
        p3 = np.array([np.e, np.e ** (-1)]) / (np.e + np.e ** (-1))
        params_expected = (p0 + p1 + p2 + p3) / 4
        self.assertEqual(params.size, D)
        self.assertSequenceEqual(params.tolist(), params_expected.tolist())


if __name__ == '__main__':
    unittest.main()
