import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import matplotlib
import sys
sys.path.append('../')
import helper_functions as h


# Invoke ggplot:
plt.style.use("seaborn-whitegrid")

cwd = os.getcwd()

r_a_l = np.array([1])
r_b_l = r_a_l
r_cr = 1
R = 100
n_files = 2000
n_samples = 100

markers = ['*', 'd', 'X', 'o', 'h', '<', 'x', '1', 's', '+', 'P', '*', 'd', '.', 'o']
colors = ['teal', 'orangered', 'forestgreen', 'purple', 'gold', 'magenta', 'cyan', 'coral','b']


phi = np.array([0, 0.1, 0.2, 0.3, 0.4])

path_to_figures = os.path.join(cwd,
                                    '../figures/')
# path_to_rates = os.path.join(cwd,
#                                     'data/inferred_rates/n10_reaclim/')
# path_to_GP = os.path.join(cwd,
#                                     '../Bimolecular/data/BD_data/run_1_diflim/')
path_to_times_gene = os.path.join(cwd, '../Gene/data/optimisation_times/gene_smallerk0ks/')
path_to_times_enz = os.path.join(cwd, 'data/optimisation_times/grid_time_bimol/')
# fig = plt.figure(figsize=(28, 7))
fig = plt.figure(figsize=(14, 7))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=25)

WD_total = []
WD_inf_total = []
x = []

x_t = []
y_e = []
y_b = []

for r_a_num, r_a in enumerate(r_a_l):
    r_b = r_a
    times_enz = np.zeros(len(phi))
    times_gene = np.zeros(len(phi))
    for i, phi_n in enumerate(phi):
        times_enz[i], times_gene[i] = h.get_time_gene(phi_n, r_a, r_b, r_cr, R, n_files, path_to_times_gene, path_to_times_enz)

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(phi, times_enz, linestyle="None", color=colors[-1], marker=markers[r_a_num],
             markersize=20, alpha=0.6, label="data")
    plt.xlabel('$\phi$')
    plt.ylabel('BO time, s')
    plt.title("Enzyme")
    # plt.ylabel('iter')
    plt.yscale('log')
    plt.text(-0.1, 1.1, '(a)', fontweight='bold',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(phi, times_gene, linestyle="None", color=colors[-1], marker=markers[r_a_num],
             markersize=20,  alpha=0.6)
    plt.xlabel('$\phi$')
    plt.ylabel('BO time, s')
    plt.title("Gene")
    # plt.ylabel('iter')
    plt.yscale('log')
    plt.text(-0.1, 1.1, '(b)', fontweight='bold',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax2.transAxes)
    x_t.append(phi)
    y_b.append(times_gene)
    y_e.append(times_enz)


# f = 0.5
tau = 0.08

#Plot loess smoothing
# Enzyme
x_mod_e = np.array(x_t).flatten()
y_mod_e = np.array(y_e).flatten()

arr_mod_e = np.array([x_mod_e, y_mod_e])
arr_mod_sorted_e = arr_mod_e[:, arr_mod_e[0, :].argsort()]

x_mod_e = arr_mod_sorted_e[0, :]
y_mod_e = arr_mod_sorted_e[1, :]

# # Gene
x_mod_b = np.array(x_t).flatten()
y_mod_b = np.array(y_b).flatten()

arr_mod_b = np.array([x_mod_b, y_mod_b])
arr_mod_sorted_b = arr_mod_b[:, arr_mod_b[0, :].argsort()]

x_mod_b = arr_mod_sorted_b[0, :]
y_mod_b = arr_mod_sorted_b[1, :]
# print(arr_mod_sorted_b)

yest_bell_b = h.lowess_bell_shape_kern(x_mod_b, y_mod_b, tau=tau)
yest_bell_e = h.lowess_bell_shape_kern(x_mod_e, y_mod_e, tau=tau)

# print(yest_bell_b)

ax1.plot(x_mod_e, yest_bell_e, color=colors[-1], linewidth=5.0, label='Loess', alpha=0.8)
ax2.plot(x_mod_b, yest_bell_b, color=colors[-1], linewidth=5.0, alpha=0.8)

plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig(path_to_figures +'plot_optimisation_times.pdf')
plt.show()

