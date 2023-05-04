# File Name: model.py
# Author: Christopher Parker
# Created: Fri Mar 24, 2023 | 10:10P EDT
# Last Modified: Thu Apr 13, 2023 | 12:34P EDT

"First pass at an NODE model with PyTorch"

ITERS = 2000
LEARNING_RATE = 1e-3
OPT_RESET = 200
ATOL = 1e-9
RTOL = 1e-7
METHOD = 'dopri5'

FILENAME_ACTH = 'NelsonMeanACTH_Control.txt'
FILENAME_CORT = 'NelsonMeanCortisol_Control.txt'

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from IPython.core.debugger import set_trace

class ANN(nn.Module):
    def __init__(self, file=None):
        super().__init__()

        # Set up neural networks for each element of a 3 equation HPA axis
        #  model
        self.hpa_net = nn.Sequential(
            nn.Linear(2, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 2, bias=True)
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
        return data, label

class NelsonGoodData(Dataset):
    def __init__(self, data_dir, patient_group):
        self.data_dir = data_dir
        self.patient_group = patient_group

    def __len__(self):
        return 11

    def __getitem__(self, idx):
        """This function will be used by the DataLoader to iterate through the
        data files of the given patient group and load the data and labels.
        Due to the nature of the problem, we actually call the time points the
        data and the concentrations the labels because given the 'data' the
        ANN should try to match the 'label'. This is slightly different than
        what would normally be used for training on an image, or something
        because the data is a time series, as is the label."""
        match idx:
            case 0:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient2_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient2_CORT.txt'
                )
            case 1:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient5_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient5_CORT.txt'
                )
            case 2:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient6_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient6_CORT.txt'
                )
            case 3:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient7_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient7_CORT.txt'
                )
            case 4:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient8_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient8_CORT.txt'
                )
            case 5:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient9_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient9_CORT.txt'
                )
            case 6:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient10_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient10_CORT.txt'
                )
            case 7:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient11_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient11_CORT.txt'
                )
            case 8:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient12_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient12_CORT.txt'
                )
            case 9:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient13_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient13_CORT.txt'
                )
            case 10:
                ACTHdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient15_ACTH.txt'
                )
                CORTdata_path = os.path.join(
                    self.data_dir, f'{self.patient_group}Patient15_CORT.txt'
                )

        ACTHdata = np.genfromtxt(ACTHdata_path)
        CORTdata = np.genfromtxt(CORTdata_path)

        data = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[0,2]]
        )
        label = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[1,3]]
        )
        return data, label

def rms_norm(tensor):
    # tensor_norms = []
    # if isinstance(tensor, tuple):
    #     for tnsr in tensor:
    #         tensor_norms.append(tnsr.pow(2).mean().sqrt())
    #     return torch.mean(torch.tensor(tensor_norms))
    # else:
    return tensor.pow(2).mean().sqrt()

def make_norm(state):
    # state_size = state.numel()
    def norm(aug_state):
        _, y, adj_y, *_ = aug_state
        # y = aug_state[1:1 + state_size]
        # adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm


if __name__ == '__main__':
    # dataset = NelsonData(
    #     '/Users/christopher/Documents/PTSD/NODE Model.nosync/Nelson TSST'
    #     ' Individual Patient Data', 'Control'
    # )
    dataset = NelsonGoodData(
        '/Users/christopher/Documents/PTSD/NODE Model.nosync/Nelson TSST'
        ' Individual Patient Data', 'Control'
    )

    # Define the device to use for neural network computations
    device = torch.device('cpu')

    # We need to convert the model parameters to double precision because that
    #  is the format of the datasets and they must match
    func = ANN().double().to(device)

    # List of parameters to optimize
    opt_params = list(func.parameters())

    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
    loss = nn.MSELoss()

    # Initialize tensor to track change in loss over each iteration
    loss_over_time = torch.zeros(ITERS)

    # Each iteration, we choose a different dataset to test instead of
    #  train
    loader = DataLoader(
        dataset=dataset, batch_size=3, shuffle=True
    )

    start_time = time.time()
    # Start main optimization loop
    for itr in range(1, ITERS + 1):
        for data, label in loader:
            # Reset gradient for each training example
            optimizer.zero_grad()

            for i, d in enumerate(label):
                y0_tensor = d[0,:]

                pred_temp = odeint(
                    func,
                    y0_tensor,
                    data[0,:,0],
                    rtol=RTOL,
                    atol=ATOL,
                    method=METHOD,
                    adjoint_options=dict(norm=make_norm(y0_tensor))
                )
                if i == 0:
                    pred_y = pred_temp
                else:
                    pred_y = torch.cat(
                        (pred_y.view(-1,11,2), pred_temp.view(-1,11,2)), 0
                    ).view(-1,11,2)

            # set_trace()
            # Compute the loss for this iteration
            output = loss(pred_y, label)

            # Backpropagation to calculate the gradient from the loss
            output.backward()

            # Step the optimizer with the new gradient
            optimizer.step()

        try:
            # Save the loss value to the loss_over_time tensor
            loss_over_time[itr-1] = output.item()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}: loss = {output.item():.6f}")
        except NameError as e:
            print(f'Must not be any batches in the loader. Got error: {e}')
            continue

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate
        if itr % OPT_RESET == 0:
            optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.6f} seconds")


    torch.save(
        func.state_dict(),
        'NN_state_2HL_11nodes_good-control-patients_seminorm.txt'
    )
    # torch.save(
    #     optimizer.state_dict(),
    #     'optimizer_state_Adam_10control-patients.txt'
    # )

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
