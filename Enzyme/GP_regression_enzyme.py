# Script to perform GP regression of inferred rates in a range of phi
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import time
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from gp_extras.kernels import ManifoldKernel

import sys
sys.path.append('../')
import helper_functions as help

plt.style.use("seaborn-whitegrid")

if __name__ == '__main__':

    # Parameters
    r_a = 1
    r_b = r_a
    r_cr = 1
    R = 100
    n_phi = 5
    file_numbers = [2000, 6000]
    # Get current path
    cwd = os.getcwd()
    # Colours and markers
    markers = ['*', 'd', 'X', 'o', 'h', 's', '<', '+', 'x', 'P', '*', 'd', '.', 'o']
    colors = ['teal', 'orangered', 'forestgreen', 'purple', 'coral', 'magenta', 'cyan', 'gold', 'springgreen', 'b', 'r', 'k']
    path_to_pkl = os.path.join(cwd, 'data/BD_data/grid_time_bimol/')

    err_GP = []
    err_CA = []

    path_to_figures = os.path.join(cwd, '../figures/')
    fig = plt.figure(figsize=(21, 11))
    # use LaTeX fonts in the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=25)

    # Initialise figure
    # Arrays to store inferred rates and errors
    kb_tot = np.zeros((3, n_phi))
    kb_err = np.zeros((3, n_phi))

    # Define 2D space for regression
    oc_vol = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    # Grid size
    R = 100
    # Load reaction rates extracted from the BO
    n_files = file_numbers[0]
    kb = [5.3*10**(-4)*R**2, 5.4*10**(-4)*R**2, 5.4*10**(-4)*R**2, 6.7*1e-4*R**2, 7.4 * 1e-4*R**2]
    tau = [157.6, 114.5, 82.5, 53.0, 24.7]
    h = [0.27, 0.3, 0.35, 0.46, 0.6]
    kb_test_data = np.array([[R**2*0.0006581, 126.2, 0.3212], [R**2*0.0004627, 90.99, 0.2804], [R**2*0.0005621, 64.45, 0.3859], [R**2*0.000663, 42.77, 0.4859]])
    kb_abs_data = np.array([[R**2*0.0004974, 115.1, 0.2748], [R**2*0.0005388, 100.6, 0.3067], [R**2*0.0006674, 85.22, 0.418], [R**2*0.0006428, 45.42, 0.4789]])

    # Prepare data for GP
    for i in range(3):
        print(i)
        x = oc_vol.reshape(-1, 1)
        if i == 0:
            y = np.array(kb).ravel()
        elif i == 1:
            y = np.array(tau).ravel()
        else:
            y = np.array(h).ravel()

        # Parameters of the kernel
        n_features = 1
        n_dim_manifold = 1
        n_hidden = 5
        architecture = ((n_features, n_hidden, n_dim_manifold),)
        kernel = ConstantKernel(1.0, (1e-10, 100)) \
                 * ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 100.0)),
                                            architecture=architecture,
                                            transfer_fct="tanh", max_nn_weight=1.0)\
                 + WhiteKernel(1e-3, (1e-10, 10))

        # Fit GP for every variable separately
        # Training data
        X_train = x
        y_train = y

        # Load testing data
        phi_t = [0.0508, 0.1114, 0.2529, 0.3259]
        kb_test = []
        kb_abs = np.zeros(len(phi_t))
        X_test = []
        n_test = 1
        kb_abs = kb_abs_data[:, i]
        X_test.append(phi_t)
        kb_test = np.array(kb_test_data[:, i])
        X_test = np.array(X_test).reshape(-1, 1)
        y_test = kb_abs

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01*y_train)
        gp.fit(X_train, y_train)

        y_pred, std_pred = gp.predict(X_test, return_std=True)
        # Save model of pkl file
        joblib.dump(gp, path_to_pkl + 'GP_{0}.pkl'.format(i), compress=9)

        # Calculate training time
        for j, x in enumerate(X_test):
            start = time.time()
            gp.predict(x.reshape(1, -1), return_std=True)
            end = time.time()
            print('Training time ', end - start)
        print(y_pred)

        # Append errors
        err_GP.append(abs(y_pred - y_test)/y_test)
        err_CA.append(abs(y_test - kb_test)/y_test)

        # Predict at the testing points
        pred_size = 50
        phi_pred = np.linspace(0.0, 0.4, pred_size).reshape(-1, 1)
        kb_pred, sigma = gp.predict(phi_pred, return_std=True)

        # Plot predictions
        print(y)
        fig.add_subplot(2, 3, i+1)
        plt.plot(oc_vol, y, linestyle="None", color=colors[0], alpha=0.9, marker=markers[0],
                 markersize=20, markerfacecolor='none', markeredgewidth=2,
                 label="train")
        plt.plot(phi_pred, kb_pred, ':', linewidth=2, color=colors[0], alpha=0.9)
        plt.fill_between(phi_pred.ravel(), (kb_pred.ravel() - sigma), (kb_pred.ravel() + sigma), color=colors[0],
                         alpha=0.5, label="error")
        plt.plot(X_test, kb_abs, linestyle="None", color=colors[0], alpha=0.9, marker=markers[1],
                 markersize=20, markerfacecolor='none', markeredgewidth=2,
                 label="test")
        plt.xlabel('$\phi$')
        if i == 0:
            plt.ylabel(r'$\tilde{k}_0$')
            ax = plt.gca()
            plt.text(-0.18, 1.0, '(a)', fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
            plt.legend()
        elif i == 1:
            plt.ylabel(r'$\tau$')
            ax = plt.gca()
            plt.text(-0.18, 1.0, '(b)', fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)
        else:
            plt.ylabel(r'$h$')
            ax = plt.gca()
            plt.text(-0.18, 1.0, '(c)', fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes)

    # Plot errors
    ax3 = fig.add_subplot(2, 3, 4)
    for num, err in enumerate(err_GP):
        plt.plot(X_test, err_GP[num], color=colors[1+num], linestyle="None", marker=markers[2],
                 markersize=20, markerfacecolor='none', markeredgewidth=2,
                 label="GP")
        plt.plot(X_test, err_CA[num], color=colors[1+num], linestyle="None",
                 marker=markers[3], markersize=20, markerfacecolor='none', markeredgewidth=2,
                 label="CA")
    plt.xlabel('$\phi$')
    #ax3.set_yscale('log')
    plt.ylabel('RE')
    plt.legend()
    plt.text(-0.18, 1.0, '(d)', fontweight='bold',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax3.transAxes)

    # Plot CA time vs SSA time
    t_SSA = [0.34493064880371094, 0.5092401504516602, 0.4865267276763916, 0.41924166679382324]
    t_BD = [181.413, 179.441, 175.003, 178.102]
    ax4 = fig.add_subplot(2, 3, 5)
    plt.plot(X_test, t_SSA, color=colors[-1], linestyle="None", marker=markers[0], markersize=20,
             label="SSA")
    plt.plot(X_test, t_BD, color=colors[1], linestyle="None", marker=markers[1], markersize=20,
             label="CA")
    plt.xlabel('$\phi$')
    plt.ylabel('Time, s')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.text(-0.18, 1.0, '(e)', fontweight='bold',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax4.transAxes)
    plt.tight_layout()
    plt.savefig(path_to_figures + 'GP_regression_enzyme.pdf')
    plt.show()

