#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.io import loadmat
import numpy as np

data = loadmat('dataBenchmark.mat')

u_est = np.squeeze(data['uEst'])
y_est = np.squeeze(data['yEst'])
u_val = np.squeeze(data['uVal'])
y_val = np.squeeze(data['yVal'])


# In[ ]:


scale_parameter = np.abs(np.max(u_est))
u_est /= scale_parameter
u_val /= scale_parameter


# In[ ]:


kernels = (100, 100, 20)
R = 25

model_order = len(kernels)
model_memory_len = np.max(kernels)


# In[ ]:


N = 1024
stepsize_scaling_tab = [0.6, 0.8, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
stepsize_scaling_tab = [0.1, 0.2, 0.3, 0.4, 0.5]
stepsize_scaling_tab = [0.025, 0.05]
stepsize_scaling_tab = [0.02, 0.015]
stepsize_scaling_tab = [0.6, 0.8, 1, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
stepsize_scaling_tab = [5, 6, 7, 8, 9, 10]

stepsize_scaling_tab = [0.0005, 0.0001, 0.00005, 0.00001]


# In[ ]:


from entropic_descent import *
from aggregation import *

max_sys_output = np.max(np.abs(y_est))
max_model_output = R
#G_sq = (max_model_output + max_sys_output) ** 2 * R ** 2
G_sq = (R * R * 2.1) ** 2

err = {}

for stepsize_scaling in stepsize_scaling_tab:
    print("stepsize_scaling: {0}".format(stepsize_scaling))
    
    G_sq_scaled = stepsize_scaling * G_sq
    
    x = u_est[model_memory_len - 1 : N - 1]
    x0 = u_est[0 : model_memory_len - 1]
    y = y_est[model_memory_len - 1 : N - 1]
    
    # DA
    m_da = VolterraModel(kernels=kernels)
    alg = AdaptiveEntropicDualAveragingAlgorithm(m_da.dictionary, R=R, constraint='ball')
    da_parameters = alg.run(x, y, G_sq_scaled, x0=x0, adaptive_stepsize=True)
    
    # validate
    m_da.set_parameters(da_parameters)
    y_mod_da = m_da.evaluate_output(u_val)
    err[stepsize_scaling] = 1 / len(u_val) * np.sum((y_mod_da - y_val) ** 2)


print(err)

