import unittest
from volterra import *


class TestVolterraFunction(unittest.TestCase):
    def test_volterra_function_basic(self):
        x = [2, 3]
        self.assertEqual(volterra_function([0], x), 3)
        self.assertEqual(volterra_function([1], x), 2)
        self.assertEqual(volterra_function([0], x, 0), 2)
        self.assertEqual(volterra_function([0, 0], x), 3 * 3)
        self.assertEqual(volterra_function([1, 1], x), 2 * 2)
        self.assertEqual(volterra_function([0, 1], x), 3 * 2)
        self.assertEqual(volterra_function([1, 0], x), 3 * 2)

    def test_volterra_function_too_short_input(self):
        x = [1]
        self.assertRaises(IndexError, volterra_function, [1], x)


class TestVolterraModel(unittest.TestCase):
    def test_volterra_model_const(self):
        m = VolterraModel(0, 0)
        self.assertEqual(m.D, 1)
        m.set_parameters([3])
        x = []
        self.assertEqual(m.evaluate_output(x), 3)

    def test_volterra_model_basic(self):
        order = 1
        memory_len = 2
        m = VolterraModel(order, memory_len)

        self.assertEqual(m.D, 3)

        m.set_parameters([0, 1, 2])
        x = [1, 2, 3]
        self.assertEqual(m.evaluate_output(x), 1 * 3 + 2 * 2)
        self.assertEqual(m.evaluate_output(x, 1), 1 * 2 + 2 * 1)

    def test_volterra_model_higher_order(self):
        order = 2
        memory_len = 2
        m = VolterraModel(order, memory_len)

        self.assertEqual(m.D, 6)

        print(m.dictionary_indices)

        m.set_parameters([0.5, 1, 2, 3, 4, -5])
        x = [1, 2, 3]
        self.assertEqual(m.evaluate_output(x), 0.5 + 1 * 3 + 2 * 2 + 3 * 3 * 3 + 4 * 3 * 2 - 5 * 2 * 2)


class TestOnlineGradientDescent(unittest.TestCase):
    def test_gradient_computation(self):
        order = 2
        memory_len = 2
        m = VolterraModel(order, memory_len)
        alg = OnlineGradientDescent(m)

        m.set_parameters([0.5, 1, 2, 3, 4, -5])
        x = [1, 2, 3]
        y_mod = m.evaluate_output(x)
        y_observed = 1
        grad = alg.compute_gradient(x, y_observed)

        self.assertEqual(len(grad), m.D)
        # grad[i] = (y_mod - y_observed) * f_i(x)
        i = 0
        for ind in m.dictionary_indices:
            self.assertEqual(grad[i], (y_mod - y_observed) * volterra_function(ind, x))
            i += 1


if __name__ == '__main__':
    unittest.main()
