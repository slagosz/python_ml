from volterra import *
import matplotlib.pyplot as plt
import math

def experiment_max_avg_gradient():
    num_of_experiments = 10

    N = 100
    x = np.random.uniform(-1, 1, N)

    order = 2
    memory_len = 10
    y_sys = 1  # assume constant - this is M const. in notes
    boundary_norm_ord = 2
    boundary_radius = 1
    m = VolterraModel(order, memory_len)
    alg = OnlineGradientDescent(m, boundary_radius, boundary_norm_ord)

    avg_g = []
    for exp in range(num_of_experiments):
        m.parameters = np.random.uniform(-1, 1, m.D)
        m.parameters *= boundary_radius / np.linalg.norm(m.parameters, boundary_norm_ord)

        g = []
        for i in range(memory_len-1, N):
            x_vec = x[0:i+1]
            g.append(np.linalg.norm(alg.compute_gradient(x_vec, y_sys), 2) ** 2)

        avg_g.append(np.average(g) ** (1/2))

    max_avg_g = np.max(avg_g)

    sup_g = m.D ** (1/2) * (boundary_radius + y_sys)
    print("sup_g={0}, max_avg_g={1}".format(sup_g, max_avg_g))

    plt.plot(avg_g)
    plt.show()


def experiment_gradient_descent():
    N = 100
    z_sigma = 0.1

    sys_order = 2
    sys_memory_len = 10

    sys = VolterraModel(sys_order, sys_memory_len)
    m = VolterraModel(sys_order, sys_memory_len)

    boundary_norm_ord = 2
    boundary_radius = 1
    sys_parameters = np.random.uniform(-1, 1, sys.D)
    sys_parameters *= boundary_radius / np.linalg.norm(sys_parameters, boundary_norm_ord)
    sys.set_parameters(sys_parameters)

    stepsize_scaling = 1  # hyperparameter for controlling stepsize
    G = math.sqrt(m.D * (math.pow(2 * boundary_radius, 2) + math.pow(z_sigma, 2)))
    stepsize_function = lambda i: \
        stepsize_scaling * boundary_radius / math.sqrt(i) / G

    alg = OnlineGradientDescent(m, stepsize_function, boundary_radius, boundary_norm_ord)

    # create input vector with initial conditions
    x = list(np.random.uniform(-1, 1, sys_memory_len - 1))

    mean_squared_errors = []
    y_sys_vec = []
    y_mod_vec = []
    for i in range(N):
        x.append(np.random.uniform(-1, 1))
        z = z_sigma * np.random.standard_normal()
        y_sys = sys.evaluate_output(x) + z
        y_mod = m.evaluate_output(x)
        mean_squared_errors.append((y_mod - y_sys) ** 2)
        alg.update(x, y_sys)

        y_sys_vec.append(y_sys)
        y_mod_vec.append(y_mod)

    mse = np.average(mean_squared_errors)
    print('mse={0}'.format(mse))

    plt.plot(y_sys_vec[-100:])
    plt.plot(y_mod_vec[-100:])
    plt.show()


def main():
    # experiment_max_avg_gradient()
    experiment_gradient_descent()


if __name__ == "__main__":
    main()
