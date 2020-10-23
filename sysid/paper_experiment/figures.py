#%%

import matplotlib.pyplot as plt

import pickle
with open('benchtank_vars', 'rb') as f:
    y_val, y_mod_da, y_mod_aggr, da_parameters, aggr_parameters = pickle.load(f)

with open('benchtank_vars2', 'rb') as f:
    y_mod_da, da_parameters = pickle.load(f)

#%%

# scaling G
#err = {0.6: 1.4202240962030686, 0.8: 1.3873346614533495, 2.0: 1.3465964718475618, 3: 1.375720151248246, 1: 1.3667963352628698, 3.5: 1.3956104096791615, 1.5: 1.3454473066991852, 4: 1.4167478066314734, 2.5: 1.358483711358119, 4.5: 1.4383784508450468, 5: 1.460091632639576, 6: 1.5029688296159727, 7: 1.5445670160628382, 8: 1.584701125336648, 9: 1.623394702117237, 10: 1.6607436346874787}
err = {1e-05: 1.8931846177873686, 0.0001: 1.703729277846867, 5e-05: 1.8328377129643079, 0.0005: 1.9177442176556398, 0.001: 1.887441409850314, 0.005: 1.5944321376343715, 0.5: 1.44363603533303, 0.4: 1.4737436929115868, 0.3: 1.5128052872164472, 0.05: 1.6704967684433552, 0.2: 1.5635741070469684, 0.01: 1.7124630200225677, 0.1: 1.628893488563183,
0.6: 1.4202240962030686, 0.8: 1.3873346614533495, 2.0: 1.3465964718475618, 3: 1.375720151248246, 1: 1.3667963352628698, 3.5: 1.3956104096791615, 1.5: 1.3454473066991852, 4: 1.4167478066314734, 2.5: 1.358483711358119, 4.5: 1.4383784508450468, 5: 1.460091632639576}

err_nonadaptive = {
 1e-05: 6.302814734731191,
 5e-05: 3.264218908545047,
 0.0001: 1.8332703912146315,
 0.0005: 1.2206760843359659,
 0.001: 1.2337346403616531,
 0.005: 1.3378861747767656,
 0.01: 1.4443755452204876,
 0.015: 1.5380587439621973,
 0.02: 1.6237768504577585,
 0.025: 1.7035966759337713,
 0.05: 2.0444968598046582,
 0.1: 2.5684975109090757,
 0.2: 3.350695720089412,
 0.3: 3.9672410635988395,
 0.4: 4.487711898985051,
 0.5: 4.941619033796801,
 0.6: 5.345906686317292,
 0.8: 6.046275307007886,
 1: 6.643392885934224,
 1.5: 7.853081897658935,
 2.0: 8.812188432672368,
 2.5: 9.614995820722772,
 3: 10.309115729033719,
 3.5: 10.922390571985861,
 4: 11.472718148471067,
 4.5: 11.97238830401863,
 5: 12.430260089362026,
 # 6: 13.245600334120866,
 # 7: 13.956023196544972,
 # 8: 14.585456077281997,
 # 9: 15.150286114797566,
 # 10: 15.662283712312703
}

err_lists = sorted(err.items()) # sorted by key, return a list of tuples
err_nonadaptive_lists = sorted(err_nonadaptive.items()) # sorted by key, return a list of tuples

x, y = zip(*err_lists)
x2, y2 = zip(*err_nonadaptive_lists)

plt.rcdefaults()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='large')

plt.xscale("log")
# plt.yscale("log")

plt.plot(x2, y2, '.--', color='tab:orange')
plt.plot(x, y, '.-', color='tab:blue')
#plt.semilogy(x2, y2, '.--', color='tab:orange')
#plt.semilogy(x, y, '.-', color='tab:blue')
plt.axhline(y=1.101105958546409, color='tab:red', linestyle='--')
#plt.axhline(y=1.3667963352628698, color='tab:blue', linestyle='--')
plt.xlabel('$\\alpha$')
plt.ylabel('err')
# plt.xticks(range(1, 6))
plt.legend(['nonadaptive DA', 'adaptive DA'])
plt.grid()

plt.savefig('err_scaling.pdf', dpi=1200, transparent=False, bbox_inches='tight')

#%%

# err
N_tab = [256, 384, 512, 640, 768, 896, 1024]
e_da = {256: 3.430135626075662, 384: 1.6187419361820459, 896: 1.5685863558688844, 512: 1.5682159943498826, 640: 1.6073828153297613, 768: 1.5876735356750327, 1024: 1.3667963352628698}
e_aggr = {256: 1.8146420296318522, 384: 1.1763044560015317, 512: 1.1551906257363689, 640: 1.2360096057384065, 768: 1.3364090918593552, 896: 1.360749047310269, 1024: 1.101105958546409}

import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='large')


plt.plot(N_tab, e_da.values(), '.-')
plt.plot(N_tab, e_aggr.values(), '--.')
plt.xlabel('N')
plt.ylabel('err')
plt.legend(['Dual Averaging', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('err.pdf', dpi=300, transparent=False, bbox_inches='tight')


#%% output

import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='large')

plt.plot(y_mod_da)
plt.plot(y_mod_aggr, '--')
plt.plot(y_val, '-.')
plt.xlabel('t')
plt.ylabel('output')
plt.legend(['Dual Averaging', '$\ell_{1}$ convex aggregation', 'True system'])
plt.grid()

plt.savefig('output.pdf', dpi=300, transparent=False, bbox_inches='tight')


#%% time of execution
N_tab = [256, 384, 512, 640, 768, 896, 1024]
t_da1 = {256: 48.07178616523743, 384: 86.76114678382874, 896: 249.93832421302795, 512: 128.39023351669312, 640: 168.3745939731598, 768: 209.9153971672058, 1024: 296.89918875694275}
t_aggr1 = {256: 25.635968685150146, 384: 63.60898685455322, 896: 329.07462787628174, 512: 95.65626883506775, 640: 176.94693517684937, 768: 254.19430923461914, 1024: 413.84074544906616}

t_da2 = {256: 47.51151251792908, 384: 87.29423689842224, 896: 252.58881282806396, 512: 126.59779047966003, 640: 167.24856209754944, 768: 209.51995396614075, 1024: 293.0336697101593}
t_aggr2 = {256: 25.19324803352356, 384: 61.603713035583496, 896: 320.79164385795593, 512: 86.78410053253174, 640: 171.79263377189636, 768: 246.7803087234497, 1024: 408.4618546962738}

t_da = {N: (t_da1.get(N, 0) + t_da2.get(N, 0)) / 2 for N in set(t_da1)}
t_aggr = {N: (t_aggr1.get(N, 0) + t_aggr2.get(N, 0)) / 2 for N in set(t_aggr1)}

t_da = sorted(t_da.items()) # sorted by key, return a list of tuples
t_aggr = sorted(t_aggr.items()) # sorted by key, return a list of tuples

x, y = zip(*t_da)
x2, y2 = zip(*t_aggr)

plt.clf()
plt.rcdefaults()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='large')

plt.plot(x, y, '.-')
plt.plot(x2, y2, '.--')
plt.xlabel('N')
plt.ylabel('time of execution [s]')
plt.legend(['Dual Averaging', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('time.pdf', dpi=1200, transparent=False, bbox_inches='tight')
