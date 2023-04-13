# File Name: model_testing.py
# Author: Christopher Parker
# Created: Fri Mar 31, 2023 | 03:09P EDT
# Last Modified: Wed Apr 12, 2023 | 09:50P EDT

"""Load saved NN and optimizer states and run network on test data to check the
results of training"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from model_overfitting_individuals import ANN


ITERS = 2000
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

def runModel_indiv(patient_group, patient_num, model_state, classifier=''):
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

    _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

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

if __name__ == "__main__":
    state = torch.load('NN_state_2HL_11nodes_good-control-patients.txt')
    device = torch.device('cpu')
    with torch.no_grad():
        runModel_mean('Control', state, '_2HL_11nodes_batch-trained')
    # optimizer = torch.load('optimizer_state_Adam_10control-patients.txt')
    # with torch.no_grad():
    #     runModel_indiv('Control', 1, state, '_2HL_80nodes_trained1indiv_2kITER_200optreset')

    # for i in range(15):
    #     state = torch.load(f'NN_state_2HL_11nodes_controlPatient{i}_5kITER_200optreset.txt')
    #     with torch.no_grad():
    #         runModel_indiv('Control', i+1, state, '_2HL_11nodes_5kITER_200optreset_trained1indiv_smooth')
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

