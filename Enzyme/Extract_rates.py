# Script to extract inference rates from the BO files
import numpy as np
import os
import re

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
        kb0 = extract_rate(content[0], "'kb0':")
        kr = extract_rate(content[0], "'kr':")
        kc = extract_rate(content[0], "'kc':")
        h = extract_rate(content[0], "'h':")
        tau = extract_rate(content[0], "'tau':")
        WD = extract_rate(content[1], "WD:")
        data.append(kb0)
        data.append(kr)
        data.append(kc)
        data.append(h)
        data.append(tau)
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
    # phi_l = [0.0, 0.1, 0.2, 0.3, 0.4]
    phi_l = [0.0508, 0.1114, 0.2529, 0.3259]
    file_numbers = [1000, 2000, 4000, 6000]
    # Get current path
    cwd = os.getcwd()
    n_files = file_numbers[3]
    rates = np.zeros((len(phi_l), 15, 6))
    rates_av = np.zeros((len(phi_l), 6))
    rates_std = np.zeros((len(phi_l), 6))
    for i, phi in enumerate(phi_l):
        if phi == 0.0:
            r_cr_val = 0
            n_cr = 0
        else:
            r_cr_val = r_cr
            n_cr = int(phi*1e4)
        pattern = 'phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_'.format(phi, n_cr, r, r, r_cr_val, R, n_files)
        regex = r'.*' + pattern + '.*'

        print(pattern)
        cwd = os.getcwd()
        path_to_rates = os.path.join(cwd, 'data/inferred_rates/grid_time_bimol/')
        ratesets = np.array(get_data(regex, path_to_rates))
        print(ratesets)
        rates[i, :, :] = ratesets
        rates_av[i, :] = np.mean(ratesets, axis=0)
        rates_std[i, :] = np.std(ratesets, axis=0)
    print(rates)
    # Save rates kb0, kr, kc, h, tau
    np.savetxt('test_{0}_kb0.txt'.format(file_numbers[3]), rates[:, :, 0])
    np.savetxt('test_{0}_kr.txt'.format(file_numbers[3]), rates[:, :, 1])
    np.savetxt('test_{0}_kc.txt'.format(file_numbers[3]), rates[:, :, 2])
    np.savetxt('test_{0}_h.txt'.format(file_numbers[3]), rates[:, :, 3])
    np.savetxt('test_{0}_tau.txt'.format(file_numbers[3]), rates[:, :, 4])
    # Save averages
    np.savetxt('test_{0}_mean.txt'.format(file_numbers[3]), rates_av)
    np.savetxt('test_{0}_std.txt'.format(file_numbers[3]), rates_std)