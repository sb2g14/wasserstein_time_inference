# Script to extract data form BO files
import numpy as np
import os
import re
import pandas as pd

# Extract rates from optuna files
def extract_rate(line, str):
    groups = line.split(str)
    ind = re.split(', |{|}|_', groups[-1])
    return float(ind[0])

def load_all_rates(data_file):
    """
    Load the data from the specified file
    """
    data = []
    with open(data_file) as f:
        content = f.readlines()
        k1 = extract_rate(content[0], "'k1':")
        k2 = extract_rate(content[0], "'k2':")
        k3 = extract_rate(content[0], "'k3':")
        WD = extract_rate(content[1], "WD:")
        data.append(k1)
        data.append(k2)
        data.append(k3)
        data.append(WD)
    return data

def get_data(regex, path_to_rates):
    """
    Process the loaded data splitting it into a dataset and a header
    """
    ratesets = []
    for file in os.listdir(path_to_rates):
        if re.match(regex, file):
            rates = load_all_rates(os.path.join(path_to_rates, file))
            ratesets.append(rates)
    return ratesets
if __name__ == '__main__':
    # Parameters
    r = 1
    r_cr = 1
    R = 100
    phi_l = [0.0, 0.1, 0.2, 0.3, 0.4]
    file_numbers = [2000]
    # Get current path
    cwd = os.getcwd()
    n_files = file_numbers[1]
    n_rates = 4
    n_sam = 5
    rates = np.zeros((len(phi_l), n_sam, n_rates))
    rates_av = np.zeros((len(phi_l), n_rates))
    rates_std = np.zeros((len(phi_l), n_rates))
    for i, phi in enumerate(phi_l):
        if phi == 0.0:
            r_cr_val = 0
            n_cr = 0
        else:
            r_cr_val = r_cr
            n_cr = int(phi*5000)
        pattern = 'phi_{0}_ncr_{1}_r_{2}_rcr_{3}_R_{4}_nfiles_{5}_'.format(phi, n_cr, r, r_cr_val, R, n_files)
        regex = r'.*' + pattern + '.*'

        print(pattern)
        cwd = os.getcwd()
        path_to_rates = os.path.join(cwd, 'data/inferred_rates/gene_smallerk0ks/')
        ratesets = np.array(get_data(regex, path_to_rates))
        print(ratesets)
        print(np.shape(ratesets))
        rates[i, :, :] = ratesets
        rates_av[i, :] = np.mean(ratesets, axis=0)
        rates_std[i, :] = np.std(ratesets, axis=0)
    print(rates)
    # Save rates kb0, kr, kc, h, tau
    np.savetxt('k1.txt', rates[:, :, 0])
    np.savetxt('k2.txt', rates[:, :, 1])
    np.savetxt('k3.txt', rates[:, :, 2])
    np.savetxt('WD.txt', rates[:, :, 3])
    # Save averages
    np.savetxt('mean.txt', rates_av)
    np.savetxt('std.txt', rates_std)