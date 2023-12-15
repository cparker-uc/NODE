# File Name: model.py
# Author: Christopher Parker
# Created: Fri Mar 24, 2023 | 10:10P EDT
# Last Modified: Mon Dec 11, 2023 | 11:33P EST

"First pass at an NODE model with PyTorch"

ITERS = 50000
LEARNING_RATE = 1e-3
OPT_RESET = 500
ATOL = 1e-6
RTOL = 1e-4
METHOD = 'rk4'
N_NETWORKS = 100

from IPython.core.debugger import set_trace
import os
import time
import torch
from copy import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint

class Found(Exception): pass

class ANN(nn.Module):
    def __init__(self):
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
            np.concatenate((ACTHdata, CORTdata), 1)[:,[0]]
        )
        label = torch.from_numpy(
            np.concatenate((ACTHdata, CORTdata), 1)[:,[1,3]]
        )
        return data, label

class AblesonData(Dataset):
    def __init__(self, data_dir, patient_group):
        self.data_dir = data_dir
        self.patient_group = patient_group

    def __len__(self):
        return 37

    def __getitem__(self, idx):
        """This function will be used by the DataLoader to iterate through the
        data files of the given patient group and load the data and labels.
        Due to the nature of the problem, we actually call the time points the
        data and the concentrations the labels because given the 'data' the
        ANN should try to match the 'label'. This is slightly different than
        what would normally be used for training on an image, or something
        because the data is a time series, as is the label."""
        data_path = os.path.join(
            self.data_dir, f'{self.patient_group}Patient{idx+1}.txt'
        )
        raw_data = np.genfromtxt(data_path)

        data = torch.from_numpy(
            raw_data[:,0]
        )
        label = torch.from_numpy(
            raw_data[:,[1,2]]
        )
        return data, label

if __name__ == '__main__':
    # dataset = AblesonData(
    #     'Ableson TSST Individual Patient Data (Without First 30 Min)',
    #     'MDD'
    # )
    # for i in range(13):
    dataset = NelsonData(
        'Nelson TSST Individual Patient Data', 'Atypical'
    )
    for i in [1]:
        data, label = dataset[i]
        data = data.view(-1)
        # Define the device to use for neural network computations
        device = torch.device('cpu')
        data = data.double().to(device)
        label = label.double().to(device)

        # We need to convert the model parameters to double precision because
        #  that is the format of the datasets and they must match
        # func = ANN().double().to(device)
        funcs = [ANN().double().to(device) for _ in range(N_NETWORKS)]
        func = funcs[0]


        # opt_params = list(func.parameters())
        # optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

        # Create 5 copies of the network, which we will start training and only
        #  keep the most successful after ITERS/10 training iterations

        optimizers = []
        for model in funcs:
            # List of parameters to optimize
            opt_params = list(model.parameters())

            # Initialize the optimizer and the loss function
            optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
            optimizers.append(optimizer)

        # Define the number of initial iterations for running 5 networks before
        #  we select the best performer
        # dist_training_iters = int((ITERS+1)/20)

        loss = nn.MSELoss()

        # Initialize tensor to track change in loss over each iteration
        loss_over_time = torch.zeros(ITERS)

        # # Also create a tensor to track which of the 5 initial networks performs
        # #  best in the first 10% of training
        # initial_losses = torch.zeros((5, int((ITERS+1)/20)))
        losses = torch.zeros((N_NETWORKS))
        loss_cutoff = 3

        start_time = time.time()
        # Start main optimization loop
        try:
            for itr in range(1, ITERS+1):
                for opt in optimizers:
                    opt.zero_grad()

                y0_tensor = label[0,:]

                for idx, model in enumerate(funcs):
                    pred_y = odeint(
                        funcs[idx],
                        y0_tensor,
                        data,
                        rtol=RTOL,
                        atol=ATOL,
                        method=METHOD,
                        # adjoint_rtol=0.0001,
                        # adjoint_atol=0.0001,
                        # adjoint_method='rk4',
                    )
                    # Compute the loss for this iteration
                    output = loss(pred_y, label)

                    # Backpropagation to calculate the gradient from the loss
                    try:
                        output.backward()
                        losses[idx] = output.item()
                    except RuntimeError as e:
                        print(f"{e=}")

                    # If the loss of any model becomes too large, re-initialize
                    #  the model parameters
                    if losses[idx] > 1e4:
                        for p in model.parameters():
                            nn.init.normal_(p, mean=0, std=0.1)

                    # If we reach a low enough loss, we raise a custom Exception
                    #  which just breaks us out of the double for loop
                    if losses[idx] < loss_cutoff:
                        print(f"Good enough")
                        func = model
                        raise Found

                    # Step the optimizer with the new gradient
                    optimizers[idx].step()

                if itr % 10 == 0:
                    print(f"Iteration {itr}: losses = {losses}")

                # If we have made it 10k iterations without finishing, we
                #  increase the loss cutoff by 1
                if itr % 5000 == 0:
                    loss_cutoff += 3
                # In case we make it through all 50k iterations without finding
                #  a good enough network, this will set the saved network to
                #  the best network we achieved
                if itr == ITERS:
                    func = funcs[torch.argmin(losses)]
        except Found:
            pass

        runtime = time.time() - start_time
        print(f"Runtime: {runtime:.6f} seconds")


        torch.save(
            func.state_dict(),
            f'NN_state_2HL_11nodes_atypicalPatient{i}_5kITER_200optreset.txt'
        )
        # torch.save(
        #     func.state_dict(),
        #     f'NN_state_2HL_11nodes_ablesonMDDPatient{i}_5kITER_200optreset.txt'
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
