# Script to perform BO with GP to infer effective reaction rates as training data
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
n_files_l = [2000, 4000]

# Updating variables from the bash input
phi = phi_l[i]
r = r_l[j]
r_cr = r_cr_l[k]
n_files = n_files_l[l]
# Grid size
R = 100
# Number of crowders
n_cr = round(phi*R**2)
# Initial state: number of molecules A, B, C and D
CS0 = 0.2  # Initial concentration of molecules S0
NS0 = round(CS0*R**2)
CE0 = 0.01  # Initial concentration of molecules E0
NE0 = round(CE0*R**2)
CC0 = 0  # Initial concentration of molecules C0 only 0 for simplicity
NC0 = round(CC0*R**2)
CP0 = 0  # Initial concentration of molecules P0 only 0 for simplicity
NP0 = round(CP0*R**2)
initial_state = [NS0, NE0, NC0, NP0]
# Number of timesteps
n_st = 1000
# Number of points in time
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
    kb0 = x[0]
    tau = x[1]
    h = x[2]
    kr = x[3]
    kc = x[4]
    WD_list, dist_list, bi_obj = model.evaluate_ss_bimol_counter(10**np.array([kb0, tau, h]), params=np.array([1, kr, kc]), time=True)
    res = bi_obj
    for dist_num in range(len(WD_list[0])):
        # Consider only second half of the trajectory
        if dist_num >= 0:
            res += WD_list[0][dist_num] / dist_list[dist_num].mean
    return res[0]


if __name__ == '__main__':

    # Enzyme reactions
    # E + S ->kb C
    # C ->kr E + S
    # C ->kc E + D
    # S - 0
    # E - 1
    # C - 2
    # P - 3

    # list the reactions, assign numbers to species as in the example above.
    # Assing known rates as values and unknown as None. Write down every reaction specifying
    # reactants first, and then list of products
    reactions = [cme.BiReaction(rate=None, specA=0, specB=1, products=(2,)),
                 cme.UniReaction(rate=None, spec=2, products=(0, 1)),
                 cme.UniReaction(rate=None, spec=2, products=(1, 3))]
    # statistics 1D Wasserstein distance
    summ_stats = [models.WassersteinDistance([0])]

    # Import data from raw files (old way)
    cwd = os.getcwd()
    path_to_pkl = os.path.join(cwd, 'data/BD_data/grid_time_bimol/')
    file_name = 'phi_{0}_ncr_{1}_ra_{2}_rb_{2}_rcr_{3}_R_{4}_nfiles_{5}'.format(phi, n_cr, r, r_cr, R, n_files)
    path_to_data = '/home/sbraiche/Documents/data/Enzyme/Grid/same_sizes_2000/phi_{phi}_ncr_{ncr}_ra_{ra}_rb_{rb}_rcr_{rcr}/'.format(phi=phi, ncr=n_cr, ra=r, rb=r, rcr=r_cr, R=R)
    print(path_to_data)
    path_to_bimol_data = '/home/sbraiche/Documents/data/Enzyme/Grid/same_sizes_2000/gam_phi_{phi}_ncr_{ncr}_ra_{ra}_rb_{rb}_rcr_{rcr}/'.format(phi=phi, ncr=n_cr, ra=r, rb=r, rcr=r_cr, R=R)
    # Array to write distributions
    dist = []
    data_bimol_res = []
    # Read data from pkl
    if os.path.isfile(os.path.join(path_to_pkl, file_name + '.pkl')):
        with open(os.path.join(path_to_pkl, file_name + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        for item in data:
            dist.append(item)
        with open(os.path.join(path_to_pkl, 'gam_' + file_name + '.pkl'), 'rb') as f_bi:
            data_bi = pickle.load(f_bi)
        for item in data_bi:
            data_bimol_res.append(item)
    # Read data from the raw data file
    else:
        data = h.return_data_iter(path_to_data, n_files, 0)
        # Times to generate distributions
        print(data.shape)
        data_bimol = h.return_data_bimol_count(path_to_bimol_data)
        data_bimol_mean = np.mean(data_bimol, 0)
        nt = np.shape(data)[1]
        tspan = np.linspace(0, nt, n)
        for tstep in tspan:
            if tstep > 0:
                tstep = int(round(tstep))
                for sp_num in range(0, len(data[0, 0, :])):
                    tmp = data[:, 0:tstep, sp_num].reshape(np.shape(data[:, 0:tstep, sp_num])[0] * np.shape(data[:, 0:tstep, sp_num])[1], 1)
                    dist.append(pdist.ParticleDistribution(tmp.T, weights=None, hist=False))
                data_bimol_res.append(data_bimol_mean[tstep - 1])
        file_pi = open(os.path.join(path_to_pkl, file_name + '.pkl'), 'wb')
        pickle.dump(dist, file_pi)
        file_pi.close()
        file_pi_bimol = open(os.path.join(path_to_pkl, 'gam_'+ file_name + '.pkl'), 'wb')
        pickle.dump(data_bimol_res, file_pi_bimol)
        file_pi_bimol.close()

    # Initialise the model, sim_kwards leave as default
    model = models.CMEModelTime(n_species=4,
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
    ranges = [(-5.0, -3.0), (1.0, np.log10(300)), (-1.0, np.log10(0.8)), (-3.0, -1.0), (-3.0, -1.0)]

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

