from numpy.random import randn, randint
from numpy.linalg import norm
from numpy import arange, cos, kron, mat, r_ as rng, round, stack

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


def aggregation(X, Y, R):
    """
    :param X: design matrix
    :param Y: system's output
    :param R: radius of l1 ball (feasible set)
    :return: vector of parameters
    """

    num_of_params = X.shape[2]
    A = Variable((num_of_params, 1))
    o = Minimize(cvx_norm(X @ A - Y, 2))
    c = [cvx_norm(A, 1) <= R]
    p = Problem(o, c)
    p.solve()  # cvx_end

    return A.value


def aggregation_for_volterra(m, x, y, x0=0):
    """

    :param m:
    :param x:
    :param y:
    :param x0:
    :return:
    """
