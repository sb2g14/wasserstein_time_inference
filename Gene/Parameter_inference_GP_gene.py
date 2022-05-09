# Script to infer rates in gene network using BO with GP
import optuna
import numpy as np
from skopt import Optimizer
from skopt import gp_minimize

import sys
sys.path.append('../')

import cme
import models
import pdist
import os
import pickle
import datetime

import helper_functions as h


print(sys.argv)
if len(sys.argv) == 7:
    i = int(sys.argv[1])
    j = int(sys.argv[2])
    k = int(sys.argv[3])
    l = int(sys.argv[4])
    ac = int(sys.argv[5])
    iter = int(sys.argv[6])
else:
    raise Warning("Incorrect number of arguments")

# Parameters used
phi_l = [0.0, 0.1, 0.2, 0.3, 0.4]
r_l = [1]
r_cr_l = [1]
n_files_l = [2000]

# Updating variables from the bash input
phi = phi_l[i]
r = r_l[j]
r_cr = r_cr_l[k]
n_files = n_files_l[l]
# Grid size
R = 100
R1 = 50
# Number of crowders
n_cr = round(phi*R*R1)
# Initial state: number of molecules G, GP, GP2, M, P, E and EP
NG0 = 1  # Initial number of molecules G0
NGP0 = 0  # Initial number of molecules GP0
NGP20 = 0  # Initial number of molecules GP20
NM0 = 0  # Initial number of molecules M0
NP0 = 0  # Initial number of molecules P0
NE0 = 100  # Initial number of molecules E0
NEP0 = 0  # Initial number of molecules EP0
initial_state = [NG0, NGP0, NGP20, NM0, NP0, NE0, NEP0]
# Number of timesteps
n_st = 5000
# Number of point in time
n = 26
# Time step
dt = 1
# CME simulation time
T = n_st * dt / (n - 1)
# Number of crowders
if phi == 0.0:
    # Make sure that n=0 and r_cr=0 when there is no crowding
    n_cr = 0
    r_cr = 0

# Define the objective function
def objective(x):
    k1 = 10**x[0]
    k2 = 10**x[1]
    k3 = 10**x[2]
    k0 = 0.05
    ks = 0.02
    km3 = 0.05
    k4 = 0.01
    kdM = 0.01
    km1 = 0.02
    km2 = 0.05
    WD_list, dist_list = model.evaluate_ss_separate(params=np.log10([k1, k2, k3, k0, k0, ks, km3, k4, kdM, km1, km2]))
    res = 0
    for dist_num in range(len(WD_list[0])):
        # Consider only second half of the trajectory
        if dist_num >= 0:
            if dist_list[dist_num].mean > 0:
                res += WD_list[0][dist_num] / dist_list[dist_num].mean
            else:
                res += WD_list[0][dist_num]
    return res[0]


if __name__ == '__main__':

    # Gene feedback reactions
    # Bimolecular
    # P + G->GP # 7
    # P + GP->GP2  # 9
    # P + E->EP  # 3
    # Unimolecular
    # G->G+M, # 1
    # GP->GP+M # 11
    # M->P+M # 2
    # EP->P+E # 4
    # EP->E # 5
    # M->0 # 6
    # GP->P+G # 8
    # GP2->P+GP, # 10
    # G - 0
    # GP - 1
    # GP2 - 2
    # M - 3
    # P - 4
    # E - 5
    # EP - 6

    # list the reactions, assign numbers to species as in the example above.
    # Assing known rates as values and unknown as None. Write down every reaction specifying
    # reactants first, and then list of products
    reactions = [cme.BiReaction(rate=None, specA=4, specB=0, products=(1,)),
                 cme.BiReaction(rate=None, specA=4, specB=1, products=(2,)),
                 cme.BiReaction(rate=None, specA=4, specB=5, products=(6,)),
                 cme.UniReaction(rate=None, spec=0, products=(0, 3)),
                 cme.UniReaction(rate=None, spec=1, products=(1, 3)),
                 cme.UniReaction(rate=None, spec=3, products=(4, 3)),
                 cme.UniReaction(rate=None, spec=6, products=(4, 5)),
                 cme.UniReaction(rate=None, spec=6, products=(5,)),
                 cme.UniReaction(rate=None, spec=3, products=()),
                 cme.UniReaction(rate=None, spec=1, products=(4, 0)),
                 cme.UniReaction(rate=None, spec=2, products=(4, 1))
                 ]
    # statistics 1D Wasserstein distance
    summ_stats = [models.WassersteinDistance([0])]

    # Import data from raw files (old way)
    cwd = os.getcwd()
    path_to_pkl = os.path.join(cwd, '../Gene/data/BD_data/gene_smallk0ks/')
    file_name = 'phi_{0}_ncr_{1}_r_{2}_rcr_{3}_R_{4}_nfiles_{5}'.format(phi, n_cr, r, r_cr, R, n_files)
    path_to_data = '/home/sbraiche/Documents/data/Gene/gene_smallk0ks/phi_{phi}_ncr_{ncr}_r_{ra}_rcr_{rcr}/'.format(phi=phi, ncr=n_cr, ra=r, rb=r, rcr=r_cr, R=R)
    print(path_to_data)
    # Array to write distributions
    dist = []
    data_bimol_res = []
    if os.path.isfile(os.path.join(path_to_pkl, file_name + '_n_{0}.pkl'.format(n))):
        with open(os.path.join(path_to_pkl, file_name + '_n_{0}.pkl'.format(n)), 'rb') as f:
            data = pickle.load(f)
        for item in data:
            dist.append(item)
    else:
        data = h.return_data_iter(path_to_data, n_files, 0)
        # Times to generate distributions
        print(data.shape)
        nt = np.shape(data)[1]
        tspan = np.linspace(0, nt, n)
        for tstep in tspan:
            if tstep > 0:
                tstep = int(round(tstep))
                for sp_num in range(0, len(data[0, 0, :])):
                    tmp = data[:, 0:tstep, sp_num].reshape(np.shape(data[:, 0:tstep, sp_num])[0] * np.shape(data[:, 0:tstep, sp_num])[1], 1)
                    dist.append(pdist.ParticleDistribution(tmp.T, weights=None, hist=False))
        file_pi = open(os.path.join(path_to_pkl, file_name + '.pkl'), 'wb')
        pickle.dump(dist, file_pi)
        file_pi.close()

    # Initialise the model, sim_kwards leave as default
    model = models.CMEModelTime(n_species=7,
                                reactions=reactions,
                                summ_stats=summ_stats,
                                initial_state=initial_state,
                                ref_dist=dist,
                                ref_bimol=data_bimol_res,
                                sim_kwargs={"t_block": T,
                                            "max_iter": n - 1,
                                            "n_samples": 50,
                                            "disable_pbar": True})

    # Set ranges of change in the parameters in log scale
    ranges = [(-6.0, -2.0), (-6.0, -2.0), (-6.0, -2.0)]

    start = datetime.datetime.now()
    # BO
    opt = Optimizer(dimensions=ranges, acq_func="gp_hedge", n_initial_points=10,
                    base_estimator='gp')

    for i in range(1000):
        suggested = opt.ask()
        y = objective(suggested)
        opt.tell(suggested, y)
        print('iteration:', i, np.power(10, suggested), y)
        elapsed_time_fl = (datetime.datetime.now() - start)
        print("Optimal rate: ", min(opt.Xi))
        print("WD: ", min(opt.yi))
        print("Elapsed time: ", elapsed_time_fl)



