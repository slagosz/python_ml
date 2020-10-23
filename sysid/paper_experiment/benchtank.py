# %% load data

from scipy.io import loadmat
import numpy as np

data = loadmat('dataBenchmark.mat')

u_est = np.squeeze(data['uEst'])
y_est = np.squeeze(data['yEst'])
u_val = np.squeeze(data['uVal'])
y_val = np.squeeze(data['yVal'])

# %% scale data

scale_parameter = np.abs(np.max(u_est))
u_est /= scale_parameter
u_val /= scale_parameter

# %% setup model parameters

kernels = (100, 100, 20)
R = 25

model_order = len(kernels)
model_memory_len = np.max(kernels)

# %% setup experiment parameters

N_tab = [256, 384, 512, 640, 768, 896, 1024]
# N_tab = [896]

# %% DA constants calculation

y_var = np.var(y_est)

max_sys_output = np.max(np.abs(y_est))
max_model_output = R
G_sq = (R * R * 2.1) ** 2

# %% experiment

import numpy as np
from entropic_descent import *
from aggregation import *

import time

e_da = {}
e_aggr = {}

time_da = {}
time_aggr = {}

y_mod_da = 0
y_mod_aggr = 0

for N in N_tab:
    print("Number of measurements: {0}".format(N))

    x = u_est[model_memory_len - 1: N - 1]
    x0 = u_est[0: model_memory_len - 1]
    y = y_est[model_memory_len - 1: N - 1]

    # DA
    m_da = VolterraModel(kernels=kernels)
    alg = AdaptiveEntropicDualAveragingAlgorithm(m_da.dictionary, R=R, constraint='ball')
    start = time.time()
    da_parameters = alg.run(x, y, G_sq, x0=x0)
    end = time.time()
    time_da[N] = end - start
    print('time: {0}'.format(end - start))

    # validate
    m_da.set_parameters(da_parameters)
    y_mod_da = m_da.evaluate_output(u_val)
    e_da[N] = 1 / len(u_val) * np.sum((y_mod_da - y_val) ** 2)

    # aggregation
    m_aggr = VolterraModel(kernels=kernels)
    start = time.time()
    aggr_parameters = aggregation_for_volterra(m_aggr.dictionary, x, y, x0=x0, R=R)
    end = time.time()
    time_aggr[N] = end - start
    print('time: {0}'.format(end - start))

    # validate
    m_aggr.set_parameters(aggr_parameters)
    y_mod_aggr = m_aggr.evaluate_output(u_val)
    e_aggr[N] = 1 / len(u_val) * np.sum((y_mod_aggr - y_val) ** 2)

print(e_da)
print(e_aggr)

print(time_da)
print(time_aggr)

# %%

N_tab = [256, 384, 512, 640, 768, 896, 1024]
e_da = {256: 3.430135626075662, 384: 1.6187419361820459, 896: 1.5685863558688844, 512: 1.5682159943498826,
        640: 1.6073828153297613, 768: 1.5876735356750327, 1024: 1.3667963352628698}
e_aggr = {256: 1.8146420296318522, 384: 1.1763044560015317, 512: 1.1551906257363689, 640: 1.2360096057384065,
          768: 1.3364090918593552, 896: 1.360749047310269, 1024: 1.101105958546409}

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150
plt.rcParams['text.usetex'] = True
plt.plot(N_tab, e_da.values(), '--.')
plt.plot(N_tab, e_aggr.values(), '--.')
plt.xlabel('N')
plt.ylabel('err')
plt.legend(['Dual Averaging', 'Aggregation (CVX)'])
plt.grid()
plt.show()

# %%

import matplotlib.pyplot as plt

plot_output = True

if plot_output:
    plt.rcParams['figure.dpi'] = 150
    plt.plot(y_mod_da)
    plt.plot(y_mod_aggr)
    plt.plot(y_val)
    plt.xlabel('t')
    plt.ylabel('output')
    plt.legend(['Dual Averaging', 'Aggregation (CVX)', 'True system'])
    plt.grid()
    plt.show()

# %%

import matplotlib.pyplot as plt

plot_stepsizes = True

if plot_stepsizes:
    plt.rcParams['figure.dpi'] = 150
    plt.plot(alg.stepsize_seq)
    plt.xlabel('t')
    plt.show()

# %%

import pickle
# with open('benchtank_vars', 'wb') as f:
#     pickle.dump([y_val, y_mod_da, y_mod_aggr, da_parameters, aggr_parameters], f)

# with open('benchtank_vars2', 'wb') as f:
#     pickle.dump([y_mod_da, da_parameters], f)
