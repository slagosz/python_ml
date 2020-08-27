#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import numpy as np

data = loadmat('dataBenchmark.mat')

u_est = np.squeeze(data['uEst'])
y_est = np.squeeze(data['yEst'])
u_val = np.squeeze(data['uVal'])
y_val = np.squeeze(data['yVal'])


# In[2]:


scale_parameter = np.abs(np.max(u_est))
u_est /= scale_parameter
u_val /= scale_parameter


# In[3]:


kernels = (100, 100, 20)
R = 25

model_order = len(kernels)
model_memory_len = np.max(kernels)


# In[4]:


#N_tab = [128, 256, 384, 512, 640, 768, 1024]
N_tab = [1024]


# In[5]:


import numpy as np
from entropic_descent import *
from aggregation import *

max_sys_output = np.max(np.abs(y_est))
# max_input = np.max(np.abs(u_est))
max_model_output = R
G_sq = (max_model_output + max_sys_output) ** 2 * R ** 2

e_da = {}
e_aggr = {}
y_mod_da = 0
y_mod_aggr = 0

for N in N_tab:
    print("Number of measurements: {0}".format(N))
    
    x = u_est[model_memory_len - 1 : N - 1]
    x0 = u_est[0 : model_memory_len - 1]
    y = y_est[model_memory_len - 1 : N - 1]
    
    # DA
    m_da = VolterraModel(kernels=kernels)
    alg = AdaptiveLazyEntropicDescentAlgorithm(m_da.dictionary, R=R, constraint='ball')
    da_parameters = alg.run(x, y, G_sq, x0=x0)
    
    # validate
    m_da.set_parameters(da_parameters)
    y_mod_da = m_da.evaluate_output(u_val)
    e_da[N] = 1 / len(u_val) * np.sum((y_mod_da - y_val) ** 2)

    # aggregation
    m_aggr = VolterraModel(kernels=kernels)
    aggr_parameters = aggregation_for_volterra(m_aggr.dictionary, x, y, x0=x0, R=R)
    
    # validate
    m_aggr.set_parameters(aggr_parameters)
    y_mod_aggr = m_aggr.evaluate_output(u_val)
    e_aggr[N] = 1 / len(u_val) * np.sum((y_mod_aggr - y_val) ** 2)


print(e_da)
print(e_aggr)


# In[ ]:


import matplotlib.pyplot as plt
plot_output = True

if plot_output:
    plt.rcParams['figure.dpi'] = 150
    plt.plot(y_mod_da)
    plt.plot(y_mod_aggr)
    plt.plot(y_val)
    plt.xlabel('t')
    plt.legend(['da', 'aggregation', 'true system'])
    plt.show()

