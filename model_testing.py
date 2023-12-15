# File Name: model_testing.py
# Author: Christopher Parker
# Created: Fri Mar 31, 2023 | 03:09P EDT
# Last Modified: Fri Dec 15, 2023 | 09:46P EST

"""Load saved NN and optimizer states and run network on test data to check the
results of training"""

import torch
import torchcde
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from model_overfitting_individuals import ANN
# from model_fitting_virtualpop import NN
# from galerkin_node import NDEOutputLayer, CDEFunc, NeuralCDE
from stacked_node import NDEOutputLayer
from torchdyn.core import NeuralODE
from torchdyn.nn.galerkin import Chebychev, GalLinear, Fourier
from torchdyn.nn.node_layers import DepthCat

PATIENT_GROUP = 'Control'
ITERS = 5000
LEARNING_RATE = 1e-3
OPT_RESET = 200
ATOL = 1e-9
RTOL = 1e-7
METHOD = 'dopri5'

def runModel_mean(patient_group, model_state, classifier=''):
    func = ANN()
    func.load_state_dict(model_state)
    func.double().to(device)
    TEST_ACTH_FILE = f'NelsonMeanACTH_{patient_group}.txt'
    TEST_CORT_FILE = f'NelsonMeanCortisol_{patient_group}.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    CORT_data = np.genfromtxt(TEST_CORT_FILE)

    true_y = torch.from_numpy(np.concatenate((ACTH_data, CORT_data), 1))[:,[1,3]]
    y0_tensor = torch.tensor((ACTH_data[0,1], CORT_data[0,1]))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    pred_y = odeint(
        func, y0_tensor, dense_t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
    )

    _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient Mean ACTH')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient Mean CORT')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}PatientMean{classifier}.png', dpi=300)

def runModel_ode(model_state, vpop_num=0, classifier=''):
    func = NN()
    func.load_state_dict(model_state)
    func.double().to(device)
    # data = np.genfromtxt(f'Sriram Model w Random ICs/sriram-model_original_num{vpop_num}.txt')
    data = np.genfromtxt('sriram-model_original_0-24-24linspace.txt')

    true_y = torch.from_numpy(data[:,1:])
    y0_tensor = torch.tensor(data[0,1:])
    t_tensor = torch.from_numpy(data[:,0])

    pred_y = odeint(
        func, y0_tensor, t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
    )

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Sriram Model Control Params')
    ax1.plot(t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='CRH',
        xlabel='Time (minutes)',
        ylabel='CRH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Sriram Model Control Params')
    ax2.plot(t_tensor, pred_y[:,1], '-', label='Simulated ACTH')
    ax2.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    ax3.plot(t_tensor, true_y[:,2], 'o', label=f'Sriram Model Control Params')
    ax3.plot(t_tensor, pred_y[:,2], '-', label='Simulated CORT')
    ax3.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax3.legend(fancybox=True, shadow=True,loc='upper right')

    ax4.plot(t_tensor, true_y[:,3], 'o', label=f'Sriram Model Control Params')
    ax4.plot(t_tensor, pred_y[:,3], '-', label='Simulated ACTH')
    ax4.set(
        title='GR',
        xlabel='Time (minutes)',
        ylabel='GR Concentration (pg/ml)'
    )
    ax4.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/sriram-model_control-params_trained-w-randomICs-0-24-1440linspace_num{vpop_num}.png', dpi=300)
    plt.close(fig)

def runModel_ode_vpop(model_state, classifier=''):
    func = NN()
    func.load_state_dict(model_state)
    func.double().to(device)
    # data = np.genfromtxt(f'Sriram Model w Random ICs/sriram-model_original_num{vpop_num}.txt')
    # data = np.genfromtxt('sriram-model_original_0-24-24linspace.txt')
    vpop_size = 100
    for  p in range(vpop_size):
        data = np.genfromtxt(f'Sriram Model w Noise/virtual_pop_normal-dist_sriram-model_vpop{p}.txt')
        # data_raw = torch.as_tensor(data_raw.reshape(data_raw.shape[0], data_raw.shape[1]//data_raw_size, data_raw_size))
        t_tensor = torch.linspace(0, 24, 1440)
        # true_y = data_raw[:,i*4:(i+1)*4]
        true_y = torch.from_numpy(data)

        y0_tensor = true_y[0,...]

        pred_y = odeint(
            func, y0_tensor, t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
        )

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10,10))

        ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Sriram Model Control Params')
        ax1.plot(t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
        ax1.set(
            title='CRH',
            xlabel='Time (minutes)',
            ylabel='CRH Concentration (pg/ml)'
        )
        ax1.legend(fancybox=True, shadow=True,loc='upper right')

        ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Sriram Model Control Params')
        ax2.plot(t_tensor, pred_y[:,1], '-', label='Simulated ACTH')
        ax2.set(
            title='ACTH',
            xlabel='Time (minutes)',
            ylabel='ACTH Concentration (pg/ml)'
        )
        ax2.legend(fancybox=True, shadow=True,loc='upper right')

        ax3.plot(t_tensor, true_y[:,2], 'o', label=f'Sriram Model Control Params')
        ax3.plot(t_tensor, pred_y[:,2], '-', label='Simulated CORT')
        ax3.set(
            title='Cortisol',
            xlabel='Time (minutes)',
            ylabel='Cortisol Concentration (micrograms/dL)'
        )
        ax3.legend(fancybox=True, shadow=True,loc='upper right')

        ax4.plot(t_tensor, true_y[:,3], 'o', label=f'Sriram Model Control Params')
        ax4.plot(t_tensor, pred_y[:,3], '-', label='Simulated ACTH')
        ax4.set(
            title='GR',
            xlabel='Time (minutes)',
            ylabel='GR Concentration (pg/ml)'
        )
        ax4.legend(fancybox=True, shadow=True,loc='upper right')

        plt.savefig(f'Figures/sriram-model_control-params_trained-w-noise-0-24-1440linspace_num{i}.png', dpi=300)
        plt.close(fig)

def runModel_group(patient_group, model_state, classifier=''):
    func = ANN()
    func.load_state_dict(model_state)
    func.double().to(device)
    for i in range(1, 16):
        TEST_ACTH_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{i}_ACTH.txt'
        TEST_CORT_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{i}_CORT.txt'
        ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
        CORT_data = np.genfromtxt(TEST_CORT_FILE)

        true_y = torch.from_numpy(np.concatenate((ACTH_data, CORT_data), 1))[:,[1,3]]
        y0_tensor = torch.tensor((ACTH_data[0,1], CORT_data[0,1]))
        t_tensor = torch.from_numpy(ACTH_data[:,0])
        dense_t_tensor = torch.linspace(0, 140, 10000)

        pred_y = odeint(
            func, y0_tensor, dense_t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
        )

        _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

        ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient {i} ACTH')
        ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
        ax1.set(
            title='ACTH',
            xlabel='Time (minutes)',
            ylabel='ACTH Concentration (pg/ml)'
        )
        ax1.legend(fancybox=True, shadow=True,loc='upper right')

        ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {i} CORT')
        ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
        ax2.set(
            title='Cortisol',
            xlabel='Time (minutes)',
            ylabel='Cortisol Concentration (micrograms/dL)'
        )
        ax2.legend(fancybox=True, shadow=True,loc='upper right')

        plt.show()
        plt.savefig(f'Figures/Nelson{patient_group}Patient{i}{classifier}.png', dpi=300)

def runModel_indiv(patient_group, patient_num, model_state, input_channels,
                   hidden_channels, output_channels, classifier=''):
    func = ANN()
    func.load_state_dict(model_state)
    func.double().to(device)
    TEST_ACTH_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_ACTH.txt'
    TEST_CORT_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_CORT.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    CORT_data = np.genfromtxt(TEST_CORT_FILE)

    true_y = torch.from_numpy(np.concatenate((ACTH_data, CORT_data), 1))[:,[1,3]]
    y0_tensor = torch.tensor((ACTH_data[0,1], CORT_data[0,1]))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    pred_y = odeint(
        func, y0_tensor, dense_t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
    )

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient {patient_num} ACTH')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {patient_num} CORT')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Ableson{patient_group}Patient{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)

def runModel_indiv_1feature(patient_group, feature, patient_num, model_state,
                   input_channels, hidden_channels, output_channels,
                   classifier=''):
    func = ANN(input_channels, hidden_channels, output_channels)
    func.load_state_dict(model_state)
    func.double().to(device)
    TEST_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_{feature}.txt'
    test_data = np.genfromtxt(TEST_FILE)

    true_y = torch.from_numpy(test_data)[:,1].reshape((-1,1))
    y0_tensor = true_y[0].reshape((1,))
    t_tensor = torch.from_numpy(test_data[:,0]).reshape((-1,1))
    dense_t_tensor = torch.linspace(0, 140, 10000)

    pred_y = odeint(
        func, y0_tensor, dense_t_tensor, atol=ATOL, rtol=RTOL, method='dopri5'
    )

    fig, ax = plt.subplots(nrows=1, figsize=(10,10))

    ax.plot(t_tensor, true_y, 'o', label=f'Nelson {patient_group} Patient {patient_num} {feature}')
    ax.plot(dense_t_tensor, pred_y, '-', label='Simulated {feature}')
    ax.set(
        title=f'{feature}',
        xlabel='Time (minutes)',
        ylabel=f'{feature} Concentration'
    )
    ax.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}Patient{patient_num}{classifier}_{feature}-only.png', dpi=300)
    plt.close(fig)

def runModel_indiv_torchdyn_normed(patient_group, patient_num, model_state, input_channels,
                   hidden_channels, output_channels, classifier=''):

    TEST_ACTH_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_ACTH.txt'
    TEST_CORT_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_CORT.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    ACTH_mean = np.mean(ACTH_data[:,1])
    ACTH_norm = ACTH_data[:,1]/ACTH_mean
    CORT_data = np.genfromtxt(TEST_CORT_FILE)
    CORT_mean = np.mean(CORT_data[:,1])
    CORT_norm = CORT_data[:,1]/CORT_mean

    true_y = torch.from_numpy(np.concatenate((ACTH_norm.reshape(-1,1), CORT_norm.reshape(-1,1)), 1))
    y0_tensor = torch.tensor((ACTH_norm[0], CORT_norm[0]))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    f = nn.Sequential(
        nn.Linear(input_channels, hidden_channels),
        nn.ReLU(),
        # DepthCat(1),
        # GalLinear(hidden_channels, hidden_channels, expfunc=Chebychev(15)),
        # nn.Tanh(),
        nn.Linear(hidden_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, 2)
    ).double()

    # Initialize parameters of the last linear layer to zero
    for p in f[-1].parameters():
        torch.nn.init.zeros_(p)

    # We pass the vector field f, the time steps at which we want evaluations
    #  and kwargs for the diff eq solver options
    nde = NeuralODE(
        f, dense_t_tensor, sensitivity='adjoint', solver=METHOD,
        atol=ATOL, rtol=RTOL
    ).to(device)

    # This layer does not compute anything, it simply re-orders the dimensions
    #  of the NeuralODE output to match the NelsonData format
    out_layer = NDEOutputLayer()

    func = nn.Sequential(nde, out_layer).to(device)
    func.load_state_dict(model_state)
    func.double().to(device)

    pred_y = func(y0_tensor)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient {patient_num} ACTH')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {patient_num} CORT')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}Patient{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)

def runModel_indiv_torchdyn(patient_group, patient_num, model_state, input_channels,
                   hidden_channels, output_channels, classifier=''):

    TEST_ACTH_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_ACTH.txt'
    TEST_CORT_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_CORT.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    CORT_data = np.genfromtxt(TEST_CORT_FILE)

    true_y = torch.from_numpy(np.concatenate((ACTH_data, CORT_data), 1))[:,[1,3]]
    y0_tensor = torch.tensor((ACTH_data[0,1], CORT_data[0,1]))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    f = nn.Sequential(
        nn.Linear(input_channels, hidden_channels),
        nn.Tanh(),
        DepthCat(1),
        GalLinear(hidden_channels, hidden_channels, Chebychev(15)),
        nn.Tanh(),
        nn.Linear(hidden_channels, hidden_channels),
        nn.Tanh(),
        nn.Linear(hidden_channels, 2)
    ).double()

    # Initialize parameters of the last linear layer to zero
    for p in f[-1].parameters():
        torch.nn.init.zeros_(p)

    # We pass the vector field f, the time steps at which we want evaluations
    #  and kwargs for the diff eq solver options
    nde = NeuralODE(
        f, dense_t_tensor, sensitivity='adjoint', solver=METHOD,
        atol=ATOL, rtol=RTOL
    ).to(device)

    # This layer does not compute anything, it simply re-orders the dimensions
    #  of the NeuralODE output to match the NelsonData format
    out_layer = NDEOutputLayer()

    func = nn.Sequential(nde, out_layer).to(device)
    func.load_state_dict(model_state)
    func.double().to(device)

    pred_y = func(y0_tensor)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient {patient_num} ACTH')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {patient_num} CORT')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}Patient{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)

def runModel_indiv_NCDE_normed(patient_group, patient_num, model_state, input_channels,
                   hidden_channels, output_channels, classifier=''):

    TEST_ACTH_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_ACTH.txt'
    TEST_CORT_FILE = f'Nelson TSST Individual Patient Data/{patient_group}Patient{patient_num}_CORT.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    ACTH_mean = np.mean(ACTH_data[:,1])
    ACTH_norm = ACTH_data[:,1]/ACTH_mean
    CORT_data = np.genfromtxt(TEST_CORT_FILE)
    CORT_mean = np.mean(CORT_data[:,1])
    CORT_norm = CORT_data[:,1]/CORT_mean

    true_y = torch.from_numpy(np.concatenate((ACTH_norm.reshape(-1,1), CORT_norm.reshape(-1,1)), 1))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(true_y.reshape(1,11,2), t=t_tensor)
    path = torchcde.CubicSpline(coeffs, t=t_tensor)
    path_y = path.evaluate(dense_t_tensor)
    dense_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path_y, t=dense_t_tensor)

    func = NeuralCDE(input_channels, hidden_channels, output_channels, t_interval=dense_t_tensor).double()

    func.load_state_dict(model_state)
    func.double().to(device)

    pred_y = func(dense_coeffs)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Patient {patient_num} ACTH')
    ax1.plot(dense_t_tensor, pred_y[0,:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {patient_num} CORT')
    ax2.plot(dense_t_tensor, pred_y[0,:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}Patient{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)


def runModel_mean_NCDE_normed(patient_group, model_state, device, input_channels,
                   hidden_channels, output_channels, classifier=''):

    TEST_ACTH_FILE = f'NelsonMeanACTH_{patient_group}.txt'
    TEST_CORT_FILE = f'NelsonMeanCortisol_{patient_group}.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE)
    ACTH_mean = np.mean(ACTH_data[:,1])
    ACTH_norm = ACTH_data[:,1]/ACTH_mean
    CORT_data = np.genfromtxt(TEST_CORT_FILE)
    CORT_mean = np.mean(CORT_data[:,1])
    CORT_norm = CORT_data[:,1]/CORT_mean

    true_y = torch.from_numpy(np.concatenate((ACTH_norm.reshape(-1,1), CORT_norm.reshape(-1,1)), 1))
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    dense_t_tensor = torch.linspace(0, 140, 10000)

    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(true_y.reshape(1,11,2), t=t_tensor)
    path = torchcde.CubicSpline(coeffs, t=t_tensor)
    path_y = path.evaluate(dense_t_tensor)
    dense_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path_y, t=dense_t_tensor)

    func = NeuralCDE(input_channels, hidden_channels, output_channels, t_interval=dense_t_tensor).double()

    func.load_state_dict(model_state)
    func.double().to(device)

    pred_y = func(dense_coeffs)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Mean')
    ax1.plot(dense_t_tensor, pred_y[0,:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Mean')
    ax2.plot(dense_t_tensor, pred_y[0,:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}Mean{classifier}.png', dpi=300)
    plt.close(fig)


def runModel_indiv_stackedNODEs_normed(
    patient_group, patient_num, state_file, n_chunks,
    input_channels, hidden_channels, output_channels, classifier=''
):
    TEST_ACTH_FILE = f'NelsonMeanACTH_{patient_group}.txt'
    TEST_CORT_FILE = f'NelsonMeanCortisol_{patient_group}.txt'
    ACTH_data = np.genfromtxt(TEST_ACTH_FILE).reshape(11, 2)
    # ACTH_mean = np.mean(ACTH_data[:,1])
    # ACTH_norm = ACTH_data[:,1]/ACTH_mean
    CORT_data = np.genfromtxt(TEST_CORT_FILE).reshape(11, 2)
    # CORT_mean = np.mean(CORT_data[:,1])
    # CORT_norm = CORT_data[:,1]/CORT_mean

    true_y = np.concatenate((ACTH_data, CORT_data), 1)
    true_y = torch.from_numpy(true_y)
    t_tensor = torch.from_numpy(ACTH_data[:,0])
    chunksize = int(np.floor(len(t_tensor)/n_chunks))

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    for i in range(n_chunks):
        if i == n_chunks-1:
            chunk_y = true_y[i*chunksize:, [1,3]]
            chunk_t = t_tensor[i*chunksize:]
        else:
            chunk_y = true_y[i*chunksize:(i+1)*chunksize+1, [1,3]]
            chunk_t = t_tensor[i*chunksize:(i+1)*chunksize+1]
        y0_tensor = torch.tensor((chunk_y[0, 0], chunk_y[0, 1]))
        model_state = torch.load(f'Networks/StackedNODEs/NN_state_2HL_{hidden_channels}nodes_{PATIENT_GROUP}{patient_num}_stackedNODE{i}_{state_file}')
        f = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)
        ).double()
        chunk_dense_t_tensor = torch.linspace(chunk_t[0], chunk_t[-1], 10000)


        # Initialize parameters of the last linear layer to zero
        # for p in f[-1].parameters():
        #     torch.nn.init.zeros_(p)

        # We pass the vector field f, the time steps at which we want evaluations
        #  and kwargs for the diff eq solver options
        nde = NeuralODE(
            f, chunk_dense_t_tensor, sensitivity='adjoint', solver=METHOD,
            atol=ATOL, rtol=RTOL
        ).to(device)

        # This layer does not compute anything, it simply re-orders the dimensions
        #  of the NeuralODE output to match the NelsonData format
        out_layer = NDEOutputLayer()

        func = nn.Sequential(nde, out_layer).to(device)
        func.load_state_dict(model_state)
        func.double().to(device)

        pred_y = func(y0_tensor)

        ax1.plot(chunk_dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
        ax1.set(
            title='ACTH',
            xlabel='Time (minutes)',
            ylabel='ACTH Concentration (pg/ml)'
        )
        ax1.legend(fancybox=True, shadow=True,loc='upper right')

        ax2.plot(chunk_dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
        ax2.set(
            title='Cortisol',
            xlabel='Time (minutes)',
            ylabel='Cortisol Concentration (micrograms/dL)'
        )
        ax2.legend(fancybox=True, shadow=True,loc='upper right')

    ax1.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Patient {patient_num} ACTH')
    ax2.plot(t_tensor, true_y[:,3], 'o', label=f'Nelson {patient_group} Patient {patient_num} CORT')
    plt.savefig(f'Figures/Nelson{patient_group}Patient{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device('cpu')

    with torch.no_grad():
        state = torch.load('/Users/christopher/Documents/PTSD/NODE/NODE/NN_state_2HL_11nodes_controlPatient8_5kITER_200optreset.txt')
        runModel_indiv('Control', 9, state, 2, 11, 2)
    # with torch.no_grad():
    #     runModel_indiv_stackedNODEs_normed('Control', 1, '3000ITER_500optreset.txt', 3, 2, 11, 2, '_stackedNODEs_3000ITER')

    # state = torch.load("NN_state_2HL_11nodes_100virtual-pop_sriram-model_normal-dist.txt")
    # with torch.no_grad():
    #     runModel_ode_vpop(state)

    # for i in range(100):
    #     with torch.no_grad():
    #         state = torch.load(f'Sriram Model w Random ICs/NN_state_2HL_11nodes_sriram-model_no-noise_num{i}.txt')
    #         device = torch.device('cpu')
    #         runModel_ode(state, vpop_num=i)

    # with torch.no_grad():
    #     runModel_mean('Control', state, '_2HL_11nodes_batch-trained')

    # for i in range(15):
    #     for j in range(2):
    #         state = torch.load(f'Refitting/NN_state_2HL_128nodes_NCDE_Neither{i+1}_{j+1}000ITER_normed.txt', map_location=device)
    #         with torch.no_grad():
    #             runModel_indiv_NCDE_normed(
    #                 'Neither',
    #                 i+1, state,
    #                 2, 2, 2,
    #                 f'_2HL_128nodes_NCDE_trained1indiv_{j+1}000ITER_normed'
    #             )

    # with torch.no_grad():
    #     state = torch.load(f'Refitting/NN_state_2HL_12nodes_NCDE_Atypical_batchsize3_2000ITER_normed.txt')
    #     runModel_mean_NCDE_normed('AtypicalMDD', state, device, 2, 2, 2, '_2HL_12nodes_NCDE_batchsize3_2000ITER_normed')

    # with torch.no_grad():
    #     runModel_indiv_1feature(
    #         'Atypical',
    #         'CORT',
    #         6, state,
    #         1, 11, 1,
    #         '_2HL_11nodes_trained1indiv_28kITER_200optreset'
    #     )

    for group in ['Control', 'MDD']:
        # for i in range(15 if group in ['control', 'melancholic'] else 14):
        for i in range(37 if group == 'Control' else 13):
            state = torch.load(f'/Users/christopher/Documents/PTSD/NODE/NODE/NN_state_2HL_11nodes_ablesonMDDPatient{i}_5kITER_200optreset.txt')

            with torch.no_grad():
                runModel_indiv(
                    group, i+1, state, 2, 11, 2,
                    f''
                )
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#                                 MIT License                                 #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#     Copyright (c) 2022 Christopher John Parker <parkecp@mail.uc.edu>        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

