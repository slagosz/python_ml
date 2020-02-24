from volterra import *
from entropic_descent import *
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
    alg = GradientDescent(m, boundary_radius, boundary_norm_ord)

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


def generate_system(order, memory_len, parameters_ball_norm_ord, parameters_ball_radius, seed=42):
    np.random.seed(seed)

    # Volterra system, uniform random parameters, norm(parameters, 2) <= 1

    sys = VolterraModel(order, memory_len)
    #sys_parameters = np.random.uniform(-1, 1, sys.D)
    sys_parameters = np.random.uniform(0, 1, sys.D) # TODO TMEPORARY
    sys_parameters *= parameters_ball_radius / np.linalg.norm(sys_parameters, parameters_ball_norm_ord)
    sys.set_parameters(sys_parameters)

    return sys


def experiment_gradient_descent():
    # model parameters
    model_order = 2
    model_memory_len = 10

    # init model
    m = VolterraModel(model_order, model_memory_len)
    m_exp = VolterraModelForExpWeights(model_order, model_memory_len)

    # input signals parameters
    z_sigma = 0.1

    # experiment setup
    experiment_length = 500  # num of observations, i.e. N
    num_of_experiments = 1

    # algorithm parameters
    stepsize_scaling = 1  # hyperparameter for controlling stepsize
    boundary_norm_ord = 2
    boundary_radius = 1
    averaging = 0

    # gradient descent stepsize calculation
    G = math.sqrt(m.D * (math.pow(2 * boundary_radius, 2) + math.pow(z_sigma, 2)))
    # G = math.sqrt(m.D * (math.pow((1 + math.sqrt(m.D)) * boundary_radius, 2) + math.pow(z_sigma, 2)))
    var_stepsize_function = lambda i: \
        stepsize_scaling * boundary_radius / math.sqrt(i) / G

    const_stepsize_function = lambda i: \
        stepsize_scaling * boundary_radius / math.sqrt(experiment_length) / G

    stepsize_function = var_stepsize_function

    # exp weights stepsize calculation
    G_exp = math.sqrt((math.pow(2 * boundary_radius, 2) + math.pow(z_sigma, 2)))
    exp_weights_var_stepsize_function = lambda i: \
        stepsize_scaling * np.sqrt(2 * np.log(m_exp.D)) / math.sqrt(i) / G_exp

    exp_stepsize_function = exp_weights_var_stepsize_function

    mean_squared_errors = np.zeros(experiment_length)
    y_sys_vec = np.zeros(experiment_length)
    y_mod_vec = np.zeros(experiment_length)

    for exp in range(num_of_experiments):
        print('experiment {0}/{1}'.format(exp + 1, num_of_experiments), end='\r')
        # generate system
        sys = generate_system(0, exp)

        # init algorithm
        alg_gd = GradientDescent(m, stepsize_function, averaging, boundary_radius, boundary_norm_ord)
        alg = alg_gd

        # create input vector with initial conditions
        x = list(np.random.uniform(-1, 1, max(model_memory_len, sys.memory_length) - 1))

        for i in range(experiment_length):
            x.append(np.random.uniform(-1, 1))
            z = z_sigma * np.random.standard_normal()
            y_sys = sys.evaluate_output(x) + z
            y_mod = m_exp.evaluate_output(x)
            mean_squared_errors[i] = ((y_mod - y_sys) ** 2)
            alg.update(x, y_sys)

            if exp == num_of_experiments - 1:
                y_sys_vec[i] = y_sys
                y_mod_vec[i] = y_mod

    mean_squared_errors = mean_squared_errors / num_of_experiments
    plt.plot(mean_squared_errors)
    plt.title('mse')
    plt.show()

    plt.plot(y_sys_vec[-100:])
    plt.plot(y_mod_vec[-100:])
    plt.title('example realization')
    plt.show()


def experiment_entropic_descent():
    # model parameters
    model_order = 3
    model_memory_len = 20

    # system parameters
    system_order = model_order
    system_memory_len = model_memory_len
    parameters_ball_norm_ord = 1
    parameters_ball_radius = 1
    M = parameters_ball_radius # true for parameters_ball_norm_ord == 1

    # init model
    m = VolterraModel(model_order, model_memory_len)

    # input signals parameters
    z_sigma = 0.01

    # experiment setup
    est_batch_size = 500 + model_memory_len - 1
    val_batch_size = 1000 + model_memory_len - 1
    num_of_experiments = 1

    # algorithm parameters
    stepsize_scaling = 1  # hyperparameter for controlling stepsize
    R = parameters_ball_radius

    # stepsize calculation
    T = est_batch_size - model_memory_len + 1
    D = m.D
    G_sq = R ** 2 * ((R + M) ** 2 + z_sigma)

    stepsize_function = lambda i: \
        stepsize_scaling * np.sqrt(2 * np.log(2 * D) / (T * G_sq))

    print('step size: {0}'.format(stepsize_function(1)))

    mean_squared_errors = np.zeros(num_of_experiments)

    y_sys_vec = []
    y_mod_vec = []

    for exp in range(num_of_experiments):
        print('experiment {0}/{1}'.format(exp + 1, num_of_experiments), end='\r')
        # generate system
        sys = generate_system(system_order, system_memory_len,
                              parameters_ball_norm_ord, parameters_ball_radius, seed=exp)

        # create learning batch
        x_est = np.random.uniform(-1, 1, est_batch_size)
        z_est = z_sigma * np.random.standard_normal(est_batch_size)
        y_est = np.zeros(est_batch_size)
        for t in range(sys.memory_length - 1, est_batch_size):
            y_est[t] = sys.evaluate_output(x_est, t) + z_est[t]

        # estimate parameters
        use_extended_dict = 0

        if use_extended_dict:
            m_exp = VolterraModelForExpWeights(model_order, model_memory_len)
            alg = EntropicDescentAlgorithm(m_exp, stepsize_function)
            extended_model_parameters = alg.run(x_est, y_est)
            model_parameters = map_parameters(extended_model_parameters, R)
            m.set_parameters(model_parameters)
        else:
            alg = EntropicDescentAlgorithm(m, stepsize_function)
            model_parameters = alg.run(x_est, y_est)
            m.set_parameters(model_parameters)

        # validate model
        x_val = np.random.uniform(-1, 1, val_batch_size)
        z_val = z_sigma * np.random.standard_normal(val_batch_size)
        y_val = np.zeros(val_batch_size)
        y_mod = np.zeros(val_batch_size)
        for t in range(sys.memory_length - 1, val_batch_size):
            y_val[t] = sys.evaluate_output(x_val, t) + z_val[t]
            y_mod[t] = m.evaluate_output(x_val, t)

        mean_squared_errors[exp] = np.linalg.norm((y_val - y_mod) ** 2, 1)

        if exp == num_of_experiments - 1:
            y_sys_vec = y_val
            y_mod_vec = y_mod

    expected_MSE = 2 * np.sqrt(2 * R * ((R + M) ** 2 + z_sigma) * np.log(2*D) / (T + 1))
    MSE = np.mean(mean_squared_errors)
    print('expected MSE {0}'.format(expected_MSE))
    print('observed MSE {0}'.format(MSE))

    plt.plot(y_sys_vec[-100:])
    plt.plot(y_mod_vec[-100:])
    plt.title('example realization')
    plt.show()


def experiment_entropic_descent_WH():
    # model parameters
    model_order = 2
    model_memory_len = 6

    # system parameters
    imp_resp_in = [1, 0.5, 0.25, 0.125, 0.1, 0.05]
    imp_resp_out = imp_resp_in
    f = lambda x: x + 0.05 * x**2

    sys = WienerHammerstein(imp_resp_in, f, imp_resp_out)

    # init model
    m = VolterraModel(model_order, model_memory_len)

    # input signals parameters
    z_sigma = 0.1

    # experiment setup
    est_batch_size = 500 + model_memory_len - 1
    val_batch_size = 1000 + model_memory_len - 1

    # algorithm parameters
    stepsize_scaling = 400  # hyperparameter for controlling stepsize
    R = 100

    # stepsize calculation
    M = 40
    T = est_batch_size - model_memory_len + 1
    D = m.D
    G_sq = R ** 2 * ((R + M) ** 2 + z_sigma)

    stepsize_function = lambda i: \
        stepsize_scaling * np.sqrt(2 * np.log(2 * D) / (T * G_sq))

    print('step size: {0}'.format(stepsize_function(1)))

    input_amplitude = 10

    # create learning batch
    x_est = np.random.uniform(-input_amplitude, input_amplitude, est_batch_size)
    z_est = z_sigma * np.random.standard_normal(est_batch_size)
    y_est = sys.evaluate_output(x_est) + z_est

    x_est_scaled = x_est / input_amplitude

    # estimate parameters
    use_extended_dict = 1

    if use_extended_dict:
        m_exp = VolterraModelForExpWeights(model_order, model_memory_len, scaling_factor=R)
        alg = EntropicDescentAlgorithm(m_exp, stepsize_function)
        extended_model_parameters = alg.run(x_est_scaled, y_est)
        model_parameters = map_parameters(extended_model_parameters, R)
        m.set_parameters(model_parameters)
    else:
        alg = EntropicDescentAlgorithm(m, stepsize_function)
        model_parameters = alg.run(x_est_scaled, y_est)
        m.set_parameters(model_parameters)

    # validate model
    x_val = np.random.uniform(-input_amplitude, input_amplitude, val_batch_size)
    x_val_scaled = x_val / input_amplitude
    z_val = z_sigma * np.random.standard_normal(val_batch_size)

    y_val = sys.evaluate_output(x_val) + z_val

    y_mod = np.zeros(val_batch_size)
    for t in range(m.memory_length - 1, val_batch_size):
        y_mod[t] = m.evaluate_output(x_val_scaled, t)

    expected_MSE = 2 * np.sqrt(2 * R * ((R + M) ** 2 + z_sigma) * np.log(2*D) / (T + 1))
    MSE = np.linalg.norm((y_val - y_mod) ** 2, 1)
    print('expected MSE {0}'.format(expected_MSE))
    print('observed MSE {0}'.format(MSE))

    plt.plot(y_val[-100:])
    plt.plot(y_mod[-100:])
    plt.title('example realization')
    plt.show()


def experiment_entropic_descent_LTI():
    # model parameters
    model_order = 1
    model_memory_len = 20

    m = VolterraModel(model_order, model_memory_len, include_constant_function=False)

    # system parameters
    imp_resp = np.linspace(10, 0.5, 20)
    M = np.sum(imp_resp)  # this is the system's output upper bound, on assumption that |x| <= 1

    sys = LTISystem(imp_resp)

    # input signals parameters
    z_sigma = 0.1

    # experiment setup
    est_batch_size = 500 + model_memory_len - 1
    val_batch_size = 1000 + model_memory_len - 1

    # algorithm parameters
    stepsize_scaling = 100  # hyperparameter for controlling stepsize
    R = M

    # for stepsize calculation
    T = est_batch_size - model_memory_len + 1
    D = m.D
    G_sq = R ** 2 * ((R + M) ** 2 + z_sigma)

    input_amplitude = 1

    # create learning batch
    x_est = np.random.uniform(-input_amplitude, input_amplitude, est_batch_size)
    z_est = z_sigma * np.random.standard_normal(est_batch_size)
    y_est = sys.evaluate_output(x_est) + z_est

    # estimate parameters
    use_extended_dict = 0

    if use_extended_dict:
        stepsize_function = lambda i: \
            stepsize_scaling * np.sqrt(2 * np.log(2 * D) / (T * G_sq))

        m_exp = VolterraModelForExpWeights(model_order, model_memory_len, scaling_factor=R)
        alg = EntropicDescentAlgorithm(m_exp, stepsize_function)
        extended_model_parameters = alg.run(x_est, y_est)
        model_parameters = map_parameters(extended_model_parameters, R)

        expected_MSE = 2 * np.sqrt(2 * R * ((R + M) ** 2 + z_sigma) * np.log(2 * D) / (T + 1))

    else:
        stepsize_function = lambda i: \
            stepsize_scaling * np.sqrt(2 * np.log(D) / (T * G_sq))

        m_tmp = VolterraModel(model_order, model_memory_len, scaling_factor=R, include_constant_function=False)

        alg = EntropicDescentAlgorithm(m_tmp, stepsize_function)
        model_parameters = R * alg.run(x_est, y_est)

        expected_MSE = 2 * np.sqrt(2 * R * ((R + M) ** 2 + z_sigma) * np.log(D) / (T + 1))

    m.set_parameters(model_parameters)

    print('step size: {0}'.format(stepsize_function(1)))

    # validate model
    x_val = np.random.uniform(-input_amplitude, input_amplitude, val_batch_size)
    z_val = z_sigma * np.random.standard_normal(val_batch_size)

    y_val = sys.evaluate_output(x_val) + z_val

    y_mod = np.zeros(val_batch_size)
    for t in range(m.memory_length - 1, val_batch_size):
        y_mod[t] = m.evaluate_output(x_val, t)

    MSE = np.linalg.norm((y_val - y_mod) ** 2, 2)
    print('expected MSE {0}'.format(expected_MSE))
    print('observed MSE {0}'.format(MSE))

    plt.plot(y_val[-100:])
    plt.plot(y_mod[-100:])
    plt.title('example realization')
    plt.show()


def main():
    # experiment_max_avg_gradient()
    # experiment_gradient_descent()
    # experiment_entropic_descent()
    # experiment_entropic_descent_WH()
    experiment_entropic_descent_LTI()



if __name__ == "__main__":
    main()
