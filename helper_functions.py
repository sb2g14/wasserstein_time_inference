import numpy as np
import re
import os
import joblib
from scipy import linalg
import datetime

# Extract rates from optuna files
def extract_rate(line, str):
    groups = line.split(str)
    ind = re.split(', |{|}|_', groups[-1])
    return float(ind[0])


def load_rates(data_file):
    """
    Load the data from the specified file
    """
    data = []
    with open(data_file) as f:
        content = f.readlines()
        kb = extract_rate(content[0], "'kb':")
        WD = extract_rate(content[1], "WD:")
        data.append(kb)
        data.append(WD)
    return data

def load_all_rates(data_file):
    """
    Load the data from the specified file
    """
    data = []
    with open(data_file) as f:
        content = f.readlines()
        kb = extract_rate(content[0], "'kb':")
        kr = extract_rate(content[0], "'kr':")
        kc = extract_rate(content[0], "'kc':")
        WD = extract_rate(content[1], "WD:")
        data.append(kb)
        data.append(kr)
        data.append(kc)
        data.append(WD)
    return data

def load_all_rates_CA(data_file):
    """
    Load the data from the specified file
    """
    data = []
    with open(data_file) as f:
        content = f.readlines()
        kb0 = extract_rate(content[0], "'kb0':")
        tau = extract_rate(content[0], "'tau':")
        h = extract_rate(content[0], "'h':")
        WD = extract_rate(content[1], "WD:")
        data.append(kb0)
        data.append(tau)
        data.append(h)
        data.append(WD)
    return data

def load_rates_fixed(data_file):
    """
    Load the data from the specified file
    """
    data = []
    with open(data_file) as f:
        content = f.readlines()
        kb = extract_rate(content[0], "'kb':")
        kr = 0.01
        # kc = extract_rate(content[0], "'kc':")
        kc = 0.01
        WD = extract_rate(content[1], "WD:")
        data.append(kb)
        data.append(kr)
        data.append(kc)
        data.append(WD)
    return data

def get_data(regex, path_to_rates, unimolecular=False, fixed=False):
    """
    Process the loaded data splitting it into a dataset and a header
    """
    ratesets = []
    for file in os.listdir(path_to_rates):
        if re.match(regex, file):
            if unimolecular and not fixed:
                rates = load_all_rates(os.path.join(path_to_rates, file))
            elif unimolecular and fixed:
                rates = load_rates_fixed(os.path.join(path_to_rates, file))
            else:
                rates = load_rates(os.path.join(path_to_rates, file))
            ratesets.append(rates)
    return ratesets

def get_data_CA(regex, path_to_rates):
    """
    Process the loaded data splitting it into a dataset and a header
    """
    ratesets = []
    for file in os.listdir(path_to_rates):
        if re.match(regex, file):
            rates = load_all_rates_CA(os.path.join(path_to_rates, file))
            ratesets.append(rates)
    return ratesets

def data_export(filepath):
    """
    Function to export data from the BD simulations

    :param filepath: name of the file

    :type filepath: string

    :return: val

    :rtype: double
    """
    # Extracted values:
    # 1:(end - 3) - numbers of proteins and state of MR
    # (end - 2) - Mean number of proteins
    # (end - 1) - Standard deviation
    # (end) - Computing time
    data_file = np.genfromtxt(filepath, delimiter=',')
    data = []
    for i, row in enumerate(data_file):
            data.append(row)
    return data


def natural_sort(l):
    """
    Function to sort list in the natual order

    :param l: list to be sorted

    :type l: list

    :return: sorted list

    :rtype: list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    # # print(l)
    # for item in (l):
    #     print(item)

    # num = re.findall('ra_(\d\.\d)', key)

    s2 = sorted(l, key=alphanum_key)
    return s2

def return_data(folder):
    data = []
    ex_vol = []
    for (dirpath, dirnames, files) in os.walk(folder):
        #print(natural_sort(files))
        for filename in [f for f in natural_sort(files) if f.endswith(".txt")]:
            filepath = os.path.join(dirpath, filename)
            #print(filepath)
            res = data_export(filepath)
            # print(len(res))
            data.append(res)
            ex_vol.append(float(re.split('_', filename)[2]))
    # np.savetxt(path_to_file + file, data, delimiter=',')
    # print("Ex vol ", data[0], "\n")
    # print("Oc vol ", ex_vol, "\n")
    return data, ex_vol

def return_data_iter(folder, n, iter):
    offset = iter*n
    data = []

    for (dirpath, dirnames, files) in os.walk(folder):
        for i, filename in enumerate([f for f in natural_sort(files) if f.endswith(".txt")]):
            #print(filename)
            if i >= offset:
                filepath = os.path.join(dirpath, filename)
                print(filepath)
                data.append(data_export(filepath))
            if i == (iter + 1)*n - 1:
               break

    # Truncate data if there are unfinished files
    #print(data)
    return np.array(data)

def return_data_bimol_count(folder):
    data = []

    for (dirpath, dirnames, files) in os.walk(folder):
        for i, filename in enumerate([f for f in natural_sort(files) if f.endswith(".txt")]):
            #print(filename)
            filepath = os.path.join(dirpath, filename)
            print(filepath)
            data.append(data_export(filepath))

    # Truncate data if there are unfinished files
    #print(data)
    return np.array(data)

def return_data_bimol_iter(folder, n, iter):
    offset = iter*n
    data = []

    for (dirpath, dirnames, files) in os.walk(folder):
        for i, filename in enumerate([f for f in natural_sort(files) if f.endswith(".txt")]):
            #print(filename)
            if i >= offset:
                filepath = os.path.join(dirpath, filename)
                print(filepath)
                data.append(data_export(filepath))
            if i == (iter + 1) * n - 1:
                break

    # Truncate data if there are unfinished files
    #print(data)
    return np.array(data)

def get_pred(phi, r_a, r_b, r_cr, R, n_files, path_to_pkl, path_to_rates):

    # Load corresponding ML classifier
    gp = joblib.load(path_to_pkl + 'GP_{0}.pkl'.format(n_files))

    if phi == 0.0:
        n_cr = 0
        r_cr = 0
    else:
        n_cr = int(np.ceil((3 * phi * R ** 3) / (4 * np.pi * r_cr ** 3)))
    #n_cr = round(1/r_cr**3*(3.0*phi*R**3/(4.0*np.pi) - initial_state[0]*r_a**3 - initial_state[1]*r_b**3))
    if n_cr < 0:
        n_cr = 0

    # Extract prediction fro GP
    k_cr = gp.predict(np.array([phi, r_a]).reshape(1, -1), return_std=False)
    # Get pred at 0
    k_0 = gp.predict(np.array([0.0, r_a]).reshape(1, -1), return_std=False)

    r = k_cr/k_0

    print(r)

    # Import infered reaction rates from optuna
    pattern = 'phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_'.format(0.0, 0, r_a, r_b, 0, R, n_files)
    print(pattern)
    regex = r'.*' + pattern + '.*'

    ratesets = np.array(get_data(regex, path_to_rates))
    print(ratesets)
    # Apply rates from the best match
    min_value = min(ratesets[:, -1])
    min_index = np.where(min_value == ratesets[:, -1])
    kb_0 = ratesets[min_index[0], 0]
    kb_pred = kb_0*r

    # Import predicted reaction rates from optuna
    pattern = 'phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(pattern)
    regex = r'.*' + pattern + '.*'

    ratesets = np.array(get_data(regex, path_to_rates))
    print(ratesets)
    # Apply rates from the best match
    min_value = min(ratesets[:, -1])
    min_index = np.where(min_value == ratesets[:, -1])
    kbt = ratesets[min_index[0], 0]
    WD = ratesets[min_index[0], -1]

    return kb_pred, kbt, WD

def get_unimolecular(phi, r_a, r_b, r_cr, R, n_files, path_to_rates):

    if phi == 0.0:
        n_cr = 0
        r_cr = 0
    else:
        n_cr = int(np.ceil((3 * phi * R ** 3) / (4 * np.pi * r_cr ** 3)))
    #n_cr = round(1/r_cr**3*(3.0*phi*R**3/(4.0*np.pi) - initial_state[0]*r_a**3 - initial_state[1]*r_b**3))
    if n_cr < 0:
        n_cr = 0

    # Import infered reaction rates from optuna
    pattern = 'phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(pattern)
    regex = r'.*' + pattern + '.*'
    print(path_to_rates)
    ratesets = np.array(get_data(regex, path_to_rates, unimolecular=True))
    print(ratesets)
    # Apply rates from the best match
    min_value = min(ratesets[:, -1])
    min_index = np.where(min_value == ratesets[:, -1])
    kr = ratesets[min_index[0], 1]
    kc = ratesets[min_index[0], 2]

    return kr, kc

def get_unimolecular_1(phi, r_a, r_b, r_cr, R, n_files, path_to_rates):

    if phi == 0.0:
        n_cr = 0
        r_cr = 0
    else:
        n_cr = int(np.ceil((3 * phi * R ** 3) / (4 * np.pi * r_cr ** 3)))
    #n_cr = round(1/r_cr**3*(3.0*phi*R**3/(4.0*np.pi) - initial_state[0]*r_a**3 - initial_state[1]*r_b**3))
    if n_cr < 0:
        n_cr = 0

    # Import infered reaction rates from optuna
    pattern = 'phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(pattern)
    regex = r'.*' + pattern + '.*'
    print(path_to_rates)
    ratesets = np.array(get_data(regex, path_to_rates, unimolecular=True, fixed=True))
    print(ratesets)
    # Apply rates from the best match
    min_value = min(ratesets[:, -1])
    min_index = np.where(min_value == ratesets[:, -1])
    kr = ratesets[min_index[0], 1]
    kc = ratesets[min_index[0], 2]

    return kr, kc

# Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi, x0, tau=.005):
    return np.exp(- (xi - x0) ** 2 / (2 * tau))


def lowess_bell_shape_kern(x, y, tau=.005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve.
    """
    m = len(x)
    yest = np.zeros(m)

    # Initializing all weights from the bell shape kernel function
    w = np.array([np.exp(- (x - x[i]) ** 2 / (2 * tau)) for i in range(m)])

    # Looping through all x-points
    for i in range(m):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                      [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]

    return yest

def get_time(phi, r_a, r_b, r_cr, R, n_files, path_to_times_bi, path_to_times):

    if phi == 0.0:
        n_cr = 0
        r_cr = 0
    else:
        n_cr = int(np.ceil((3 * phi * R ** 3) / (4 * np.pi * r_cr ** 3)))
    #n_cr = round(1/r_cr**3*(3.0*phi*R**3/(4.0*np.pi) - initial_state[0]*r_a**3 - initial_state[1]*r_b**3))
    if n_cr < 0:
        n_cr = 0


    # Import inference of enzyme reaction times from optuna
    file_name = 'time_phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_0.txt'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(file_name)
    cwd = os.getcwd()
    #times = np.genfromtxt(path_to_times + file_name, delimiter='\n')
    with open(path_to_times + file_name) as f1:
        content = f1.readlines()
        begin = datetime.datetime.strptime(content[0].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.datetime.strptime(content[-1].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        times = end - begin
    print(times)
    f1.close()

    # Import bimolecular inference times from optuna
    file_name_bi = 'time_phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_0.txt'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(file_name_bi)
    cwd = os.getcwd()
    #times_bi = np.genfromtxt(path_to_times_bi + file_name_bi, delimiter='\n')
    with open(path_to_times_bi + file_name_bi) as f2:
        content = f2.readlines()
        begin = datetime.datetime.strptime(content[0].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.datetime.strptime(content[-1].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        times_bi = end - begin
    print(times_bi)
    f2.close()

    return times.seconds, times_bi.seconds

def get_time_gene(phi, r_a, r_b, r_cr, R, n_files, path_to_times_gene, path_to_times):

    if phi == 0.0:
        n_cr = 0
        r_cr = 0
    else:
        n_cr = int(phi*R**2)
    #n_cr = round(1/r_cr**3*(3.0*phi*R**3/(4.0*np.pi) - initial_state[0]*r_a**3 - initial_state[1]*r_b**3))
    if n_cr < 0:
        n_cr = 0


    # Import inference of enzyme reaction times from optuna
    file_name = 'time_phi_{0}_ncr_{1}_ra_{2}_rb_{3}_rcr_{4}_R_{5}_nfiles_{6}_0.txt'.format(phi, n_cr, r_a, r_b, r_cr, R, n_files)
    print(file_name)
    cwd = os.getcwd()
    #times = np.genfromtxt(path_to_times + file_name, delimiter='\n')
    with open(path_to_times + file_name) as f1:
        content = f1.readlines()
        begin = datetime.datetime.strptime(content[0].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.datetime.strptime(content[-1].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        times = end - begin
    print(times)
    f1.close()

    n_cr = int(n_cr/2)
    # Import gene inference times from optuna
    file_name_gene = 'time_phi_{0}_ncr_{1}_r_{2}_rcr_{3}_R_{4}_nfiles_{5}_0.txt'.format(phi, n_cr, r_a, r_cr, R, n_files)
    print(file_name_gene)
    cwd = os.getcwd()
    #times_bi = np.genfromtxt(path_to_times_bi + file_name_bi, delimiter='\n')
    with open(path_to_times_gene + file_name_gene) as f2:
        content = f2.readlines()
        begin = datetime.datetime.strptime(content[0].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        end = datetime.datetime.strptime(content[-1].rstrip(), '%Y-%m-%d %H:%M:%S.%f')
        times_gene = end - begin
    print(times_gene)
    f2.close()

    return times.seconds, times_gene.seconds


