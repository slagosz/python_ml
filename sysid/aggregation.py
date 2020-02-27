from numpy.random import randn, randint
from numpy.linalg import norm
from numpy import arange, cos, kron, mat, r_ as rng, round, stack
import numpy as np

from cvxpy import Problem, Minimize, Variable
from cvxpy import norm as cvx_norm


'''
# Main function stuff                 # MATLAB's origins
def m(X, A):                          # function Y = m(X, A)
    L = arange(A.size)                #   L  = length(A);
    Φ = cos(kron(X, L))               #   Φ = cos(kron(X, 1:L));
    return Φ @ A                      #   Y  = Φ * A;

## Measurements (note N << L - the one way around...)
α = randint(-2, 3, (6, 1))            # α = randi([-2, 3], 6, 1);
ρ = norm(α, 1) * 1
N  = 128; X, Z = (randn(N, 1), randn(N, 1) * .125); Y = m(X, α) + Z
## Regressors matrix (note L >> N - the other way around...)
L  = 512; Φ = cos(kron(X, arange(L))) # Φ = cos(kron(X, 1:L));


#Python CVX                           # MATLAB CVX
                                      # cvx_begin quiet
A = Variable((L, 1))                  #  variable A(L)
o = Minimize(cvx_norm(Φ @ A - Y, 2))  #  minimize(norm(Φ * A - Y, 2))
c = [cvx_norm(A, 1) <= ρ]             #  subject to norm(A, 1) <= ρ
p = Problem(o, c); p.solve()          # cvx_end
'''


def aggregation(X, Y, R=1):
    """
    :param X: design matrix
    :param Y: system's output
    :param R: radius of l1 ball (feasible set)
    :return: vector of parameters
    """

    num_of_params = X.shape[1]
    A = Variable(num_of_params)
    o = Minimize(cvx_norm(X @ A - Y, 2))
    c = [cvx_norm(A, 1) <= R]
    p = Problem(o, c)
    p.solve()  # cvx_end

    return A.value


def create_design_matrix(dictionary, x, x0=None):
    t_list = list(range(0, len(x)))
    X = np.zeros([len(x), dictionary.size])

    if x0 is None:
        x0 = np.zeros(dictionary.memory_length - 1)
    else:
        x0 = np.array(x0)
        assert len(x0) == (dictionary.memory_length - 1)

    x = np.concatenate([x0, x])

    row_idx = 0
    for t in t_list:
        X[row_idx, :] = [f(x, t + dictionary.memory_length - 1) for f in dictionary.dictionary]
        row_idx += 1

    return X


def aggregation_for_volterra(dictionary, x, y, x0=None, R=1):
    X = create_design_matrix(dictionary, x, x0)

    return aggregation(X, y, R)
