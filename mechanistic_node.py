# File Name: mechanistic_node.py
# Author: Christopher Parker
# Created: Wed Jun 07, 2023 | 12:13P EDT
# Last Modified: Sat Jun 10, 2023 | 06:27P EDT

"""This script will train a model consisting of 4 ANNs and 2 mechanistic
components against individual patients from the Nelson data to see if we can
match when training/testing on the same patient."""

from galerkin_node import INPUT_CHANNELS


ITERS = 50000
LR = 1e-3
DECAY = 1e-6
OPT_RESET = 1000
ATOL = 1e-7
RTOL = 1e-5
METHOD = 'dopri5'
PATIENT_GROUP = 'Atypical'
N_POINTS = 240
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
HDIM = 20

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

class ANN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        # Set up neural networks for each element of a 3 equation HPA axis
        #  model
        self.hpa_net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_channels, output_channels, bias=True),
        )

        for m in self.hpa_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.normal_(m.bias, mean=0, std=0.5)

    def forward(self, t, y):
        "Compute the next step of the diff eq by iterating the neural network"
        # Do we need to make the NN take time as an input, also?
        # self.file.write(f"y: {y},\ntype of y: {type(y)}")
        return self.hpa_net(y)

class NelsonData(Dataset):
    def __init__(self, data_dir, patient_group):
        self.data_dir = data_dir
        self.patient_group = patient_group

    def __len__(self):
        return 15

    def __getitem__(self, idx):
        """This function will be used by the DataLoader to iterate through the
        data files of the given patient group and load the data and labels.
        Due to the nature of the problem, we actually call the time points the
        data and the concentrations the labels because given the 'data' the
        ANN should try to match the 'label'. This is slightly different than
        what would normally be used for training on an image, or something
        because the data is a time series, as is the label."""
        ACTHdata_path = os.path.join(
            self.data_dir, f'{self.patient_group}Patient{idx+1}_ACTH.txt'
        )
        CORTdata_path = os.path.join(
            self.data_dir, f'{self.patient_group}Patient{idx+1}_CORT.txt'
        )

        ACTHdata = np.genfromtxt(ACTHdata_path)
        CORTdata = np.genfromtxt(CORTdata_path)

        data = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[0,2]]
        )
        label = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[1,3]]
        )
        ACTH_mean = torch.mean(label[:,0])
        CORT_mean = torch.mean(label[:,1])
        label_norm = torch.cat(((label[:,0]/ACTH_mean).reshape(-1,1), (label[:,1]/CORT_mean).reshape(-1,1)), 1)
        return data, label_norm

class SriramData(Dataset):
    def __init__(self, n_points, data_dir=''):
        self.data_dir = data_dir
        self.n_points = n_points

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """This function will be used by the DataLoader to iterate through the
        data files of the given patient group and load the data and labels.
        Due to the nature of the problem, we actually call the time points the
        data and the concentrations the labels because given the 'data' the
        ANN should try to match the 'label'. This is slightly different than
        what would normally be used for training on an image, or something
        because the data is a time series, as is the label."""
        data_path = os.path.join(
            self.data_dir, f'sriram-model_original_0-24-{self.n_points}linspace.txt'
        )

        raw_data = np.genfromtxt(data_path)

        data = torch.from_numpy(
            raw_data[:,0]
        )
        label = torch.from_numpy(
            raw_data[:,[2,3]]
        )
        ACTH_mean = torch.mean(label[:,0])
        CORT_mean = torch.mean(label[:,1])
        label_norm = torch.cat(((label[:,0]/ACTH_mean).reshape(-1,1), (label[:,1]/CORT_mean).reshape(-1,1)), 1)
        return data, label_norm


class NNSystem(nn.Module):
    """Defines the system of equations as a combination of parameters and
    neural networks."""
    def __init__(self):
        super().__init__()

        self.acth_pos = ANN(1, 10, 1).double()
        self.acth_neg = ANN(1, 10, 1).double()

        self.cort_pos = ANN(1, 10, 1).double()
        self.cort_neg = ANN(1, 10, 1).double()


        # Define parameters to determine feedback strength
        self.K_i = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=False).double()
        self.n = torch.nn.Parameter(torch.ones(1)*5, requires_grad=False).double()


    def forward(self, t, y):
        """Defines the RHS of the vector field"""

        dy_acth = (self.K_i**self.n/(self.K_i**self.n + y[1]**(self.n)))*self.acth_pos(t, y[0]) - self.acth_neg(t, y[0])
        dy_cort = y[0]*self.cort_pos(t, y[1]) - self.cort_neg(t, y[1])
        dy = torch.cat((dy_acth, dy_cort), 0).reshape(2,1)
        return dy


class NNSystem_mech(nn.Module):
    """Defines the system of equations as a combination of parameters and
    neural networks."""
    def __init__(self):
        super().__init__()

        self.acth_initial = nn.Linear(1, 2).double()
        self.acth_pos = ANN(1, 10, 1).double()
        self.acth_neg = ANN(1, 10, 1).double()
        self.acth_readout = nn.Linear(2, 1).double()

        self.cort_initial = nn.Linear(1, 2).double()
        self.cort_pos = ANN(1, 10, 1).double()
        self.cort_neg = ANN(1, 10, 1).double()
        self.cort_readout = nn.Linear(2, 1).double()


        # for p in self.initial.parameters():
        #     nn.init.normal_(p)
        # for p in self.readout.parameters():
        #     nn.init.normal_(p)

        # Define parameters to determine feedback strength
        self.K_i = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=False).double()
        self.n = torch.nn.Parameter(torch.ones(1)*5, requires_grad=False).double()


    def forward(self, t, y):
        """Defines the RHS of the vector field"""
        dy_acth = nn.functional.softplus(self.acth_initial(y[0]).reshape(2,1))
        dy_cort = nn.functional.softplus(self.cort_initial(y[1]).reshape(2,1))

        dy_acth = torch.cat((self.acth_pos(t, dy_acth[0]),self.acth_neg(t, dy_acth[1])), 0).reshape(2,1)
        dy_acth = torch.cat(((self.K_i**self.n/(self.K_i**self.n + y[1]**(self.n)))*dy_acth[0], dy_acth[1]), 0).reshape(1,2)
        dy_acth = self.acth_readout(nn.functional.softplus(dy_acth))

        dy_cort = torch.cat((self.cort_pos(t, dy_cort[0]),self.cort_neg(t, dy_cort[1])), 0).reshape(2,1)
        dy_cort = torch.cat((y[0]*dy_cort[0], dy_cort[1]), 0).reshape(1,2)
        dy_cort = self.cort_readout(nn.functional.softplus(dy_cort))
        dy = torch.cat((dy_acth, dy_cort), 0).reshape(2,1)
        return dy


class NNSystem_1d(nn.Module):
    """Defines the system of equations as a combination of parameters and
    neural networks."""
    def __init__(self):
        super().__init__()

        self.pos = ANN(1, 10, 1).double()
        self.neg = ANN(1, 10, 1).double()

        # self.initial = nn.Linear(2, 4).double()
        # self.readout = nn.Linear(2, 2).double()

        # for p in self.initial.parameters():
        #     nn.init.normal_(p)
        # for p in self.readout.parameters():
        #     nn.init.normal_(p)

        # Define parameters to determine feedback strength
        self.K_i = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=True).double()
        # self.n = torch.nn.Parameter(torch.ones(1)*5, requires_grad=True).double()

    def forward(self, t, y):
        """Defines the RHS of the vector field"""
        # y = self.initial(y).relu().reshape(2,2)

        # out = self.readout(
            # torch.cat((
            #     (self.K_i**self.n/(self.K_i**self.n + y[1]**(self.n)))*self.acth_pos(t, y[0].reshape(2)) - self.acth_neg(t, y[0].reshape(2)),
            #     y[0]*self.cort_pos(t, y[1].reshape(2)) - self.cort_neg(t, y[1].reshape(2))
            # )).relu()
        # )
        # return out
        return self.K_i*self.pos(t, y) - self.neg(t, y)


def main():
    # Define the system of equations
    device = torch.device('cpu')
    model = NNSystem().to(device)
    # state = torch.load('Refitting/NN_state_1HL_10nodes_mechanisticFeedback_Ki1p5_n5_controlPatient1_5000ITER_Noneoptreset_normed.txt')
    # model.load_state_dict(state)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()
    loss_over_time = []

    # dataset = NelsonData('Nelson TSST Individual Patient Data', 'Control')
    dataset = NelsonData('Nelson TSST Individual Patient Data', 'Control')

    start_time = time.time()
    for i in range(1):
        data, label = dataset[i]
        t_interval = data[:,0]
        y0 = label[0,:]

        for itr in range(1, ITERS+1):
            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = odeint(
                model, y0, t_interval, rtol=RTOL, atol=ATOL, method=METHOD
            )

            # Compute the loss based on the results
            output = loss(pred_y, label)
            loss_over_time.append(output.item())

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the 
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate and momentum
            if OPT_RESET is None:
                pass
            elif itr % OPT_RESET == 0:
                optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

            if itr % 1000 == 0:
                runtime = time.time() - start_time
                print(f"Runtime: {runtime:.6f} seconds")
                torch.save(
                    model.state_dict(),
                    f'Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient{i+1}_'
                    f'{itr}ITER_{OPT_RESET}optreset_normed.txt'
                )
                with open(f'Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient{i+1}'
                          f'_{itr}ITER_{OPT_RESET}optreset_normed_setup.txt',
                          'w+') as file:
                    file.write(f'Model Setup for {PATIENT_GROUP} Patient {i+1}:\n')
                    file.write(
                        f'ITERS={itr}\nLEARNING_RATE={LR}\n'
                        f'OPT_RESET={OPT_RESET}\nATOL={ATOL}\nRTOL={RTOL}\n'
                        f'METHOD={METHOD}\n'
                        f'Input channels={INPUT_CHANNELS}\n'
                        f'Hidden channels={HDIM}\n'
                        f'Output channels={OUTPUT_CHANNELS}\n'
                        f'Runtime={runtime}\n'
                        f'Optimizer={optimizer}'
                        f'Loss over time={loss_over_time}'
                    )
    return


def main_sriram():
    # Define the system of equations
    device = torch.device('cpu')
    model = NNSystem_mech().to(device)
    # model = ANN(1, 10, 1).double().to(device)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=LR, weight_decay=DECAY)
    loss = nn.MSELoss()
    loss_over_time = []

    dataset = SriramData(N_POINTS)

    start_time = time.time()
    for i in range(1):
        data, label = dataset[i]
        # label = label.reshape(-1,1)
        t_interval = data
        # y0 = label[0].reshape(1)
        y0 = label[0,:].reshape(2,1)

        for itr in range(1, ITERS+1):
            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = odeint(
                model, y0, t_interval, rtol=RTOL, atol=ATOL, method=METHOD
            ).reshape(N_POINTS,2)

            # Compute the loss based on the results
            output = loss(pred_y, label)
            loss_over_time.append(output.item())

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the 
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate and momentum
            if OPT_RESET is None:
                pass
            elif itr % OPT_RESET == 0:
                optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

            if itr % 1000 == 0:
                runtime = time.time() - start_time
                print(f"Runtime: {runtime:.6f} seconds")
                torch.save(
                    model.state_dict(),
                    f'Refitting/NN_state_2x_1HL_10nodes_SriramModel{N_POINTS}_mechanistic_Softplus_noParamGrad_init-readout_'
                    f'{itr}ITER_{OPT_RESET}optreset_normed.txt'
                )
                with open(f'Refitting/NN_state_2x_1HL_10nodes_SriramModel{N_POINTS}_mechanistic_Softplus_noParamGrad_init-readout_'
                          f'{itr}ITER_{OPT_RESET}optreset_normed_setup.txt',
                          'w+') as file:
                    file.write(f'Model Setup for {PATIENT_GROUP} Patient {i+1}:\n')
                    file.write(
                        f'ITERS={itr}\nLEARNING_RATE={LR}\n'
                        f'OPT_RESET={OPT_RESET}\nATOL={ATOL}\nRTOL={RTOL}\n'
                        f'METHOD={METHOD}\n'
                        f'Input channels={INPUT_CHANNELS}\n'
                        f'Hidden channels={HDIM}\n'
                        f'Output channels={OUTPUT_CHANNELS}\n'
                        f'Runtime={runtime}\n'
                        f'Optimizer={optimizer}'
                        f'Loss over time={loss_over_time}'
                    )
    return


def test(state, patient_group, patient_num, classifier=''):
    device = torch.device('cpu')
    model = NNSystem().to(device)
    model.load_state_dict(state)

    dataset = NelsonData('Nelson TSST Individual Patient Data', 'Control')
    data, true_y = dataset[patient_num-1]
    y0 = true_y[0,:]
    t_tensor = data[:,0]
    dense_t_tensor = torch.linspace(0, 140, 10000)

    pred_y = odeint(model, y0, dense_t_tensor, atol=ATOL, rtol=RTOL, method=METHOD)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Nelson {patient_group} Mean')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='Simulated ACTH')
    ax1.set(
        title='ACTH',
        xlabel='Time (minutes)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Nelson {patient_group} Mean')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='Simulated CORT')
    ax2.set(
        title='Cortisol',
        xlabel='Time (minutes)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/Nelson{patient_group}{patient_num}{classifier}.png', dpi=300)
    plt.close(fig)

    return


def test_sriram(state, classifier=''):
    device = torch.device('cpu')
    model = NNSystem_mech().to(device)
    # model = ANN(1, 10, 1).double().to(device)
    model.load_state_dict(state)

    dataset = SriramData(N_POINTS)
    data, true_y = dataset[0]
    # y0 = true_y[0].reshape(1)
    y0 = true_y[0,:].reshape(2,1)
    t_tensor = data
    dense_t_tensor = torch.linspace(0, 24, 10000)

    pred_y = odeint(model, y0, dense_t_tensor, atol=ATOL, rtol=RTOL, method=METHOD)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,10))

    ax1.plot(t_tensor, true_y[:,0], 'o', label=f'Sriram Model Simulation')
    ax1.plot(dense_t_tensor, pred_y[:,0], '-', label='NODE ACTH Simulation')
    ax1.set(
        title='ACTH',
        xlabel='Time (hours)',
        ylabel='ACTH Concentration (pg/ml)'
    )
    ax1.legend(fancybox=True, shadow=True,loc='upper right')

    ax2.plot(t_tensor, true_y[:,1], 'o', label=f'Sriram Model Simulation')
    ax2.plot(dense_t_tensor, pred_y[:,1], '-', label='NODE CORT Simulation')
    ax2.set(
        title='Cortisol',
        xlabel='Time (hours)',
        ylabel='Cortisol Concentration (micrograms/dL)'
    )
    ax2.legend(fancybox=True, shadow=True,loc='upper right')

    plt.savefig(f'Figures/SriramModel_fitting{classifier}', dpi=300)
    plt.close(fig)

    return


if __name__ == "__main__":
    # main()
    # main_sriram()

    # state = torch.load('Refitting/NN_state_1HL_10nodes_mechanisticFeedback_paramOpt_Ki1p5_n5_controlPatient1_1000ITER_Noneoptreset_normed.txt')
    # with torch.no_grad()
        # test(state, 'Control', 1, '_1HL_10nodes_mechanistic_normed_params_noneg_1kITER_Ki1p5_n5')

    state = torch.load('Refitting/NN_state_2x_1HL_10nodes_SriramModel240_mechanistic_Softplus_noParamGrad_init-readout_3000ITER_1000optreset_normed.txt')
    with torch.no_grad():
        test_sriram(state, '_240pts_2x_1HL_10nodes_3000ITER_Softplus_init-readout_1000optreset')

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

