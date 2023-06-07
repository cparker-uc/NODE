# File Name: stacked_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Tue Jun 06, 2023 | 05:09P EDT

"""Implementing the torchdyn library Galerkin NODE class to allow
depth-variance among the neural network parameters"""

INPUT_CHANNELS = 2
HDIM = 32
OUTPUT_CHANNELS = 2

ITERS = 1000
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-5
RTOL = 1e-5
METHOD = 'dopri5'

PATIENT_GROUP = 'Atypical'
N_CHUNKS = 5

# from IPython.core.debugger import set_trace
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset
from torchdyn.core import NeuralODE
from torchdyn.nn.node_layers import DepthCat
from typing import Tuple


# Not certain if this is necessary, but in the quickstart docs they have
#  done a wildcard import of torchdyn base library, and this is all that does
TTuple = Tuple[torch.Tensor, torch.Tensor]

class NelsonData(Dataset):
      def __init__(self, data_dir, patient_group):
          self.data_dir = data_dir
          self.patient_group = patient_group

      def __len__(self):
          return 14

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

class NDEOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tup):
        (t_eval, sol) = tup
        # The result returned from NeuralODE is (11, 1, 2) instead of
        #  (11, 2, 1) so we swap the last two axes
        # return torch.swapaxes(sol, 1, 2)
        return sol

if __name__ == "__main__":
    # set_trace()
    device = torch.device('cpu')

    for i in range(1):
        dataset = NelsonData('Nelson TSST Individual Patient Data', PATIENT_GROUP)
        data, label = dataset[i]
        chunksize = int(np.floor(len(label)/N_CHUNKS))

        # list to store the series of networks being trained
        models = []

        for j in range(N_CHUNKS):
            if j == N_CHUNKS-1:
                datachunk = data[j*chunksize:, :]
                labelchunk = data[j*chunksize:, :]
            datachunk = data[j*chunksize:(j+1)*chunksize, :]
            labelchunk = label[j*chunksize:(j+1)*chunksize, :]
            t_eval = datachunk[:,0]  # Time points we need the solver to output
            y0 = labelchunk[0,:]     # ICs for the vector field

            f = nn.Sequential(
                nn.Linear(2, HDIM),
                nn.Tanh(),
                nn.Linear(HDIM, HDIM),
                nn.Tanh(),
                nn.Linear(HDIM, 2)
            ).double()

            # Initialize parameters of the last linear layer to zero
            for p in f[-1].parameters():
                torch.nn.init.zeros_(p)

            # We pass the vector field f, the time steps at which we want evaluations
            #  and kwargs for the diff eq solver options
            nde = NeuralODE(
                f, t_eval, sensitivity='adjoint', solver=METHOD,
                atol=ATOL, rtol=RTOL
            ).double().to(device)

            # This layer does not compute anything, it simply re-orders the dimensions
            #  of the NeuralODE output to match the NelsonData format
            out_layer = NDEOutputLayer()

            model = nn.Sequential(nde, out_layer).double().to(device)
            models.append(model)

            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
            loss = nn.MSELoss()
            loss_over_time = []

            start_time = time.time()
            for itr in range(1, ITERS+1):
                optimizer.zero_grad()

                # Compute the forward direction of the NODE
                pred_y = model(y0)

                # Compute the loss based on the results
                output = loss(pred_y, labelchunk)
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
                    optimizer = optim.AdamW(model.parameters(), lr=LR)

                if itr % 1000 == 0:
                    runtime = time.time() - start_time
                    print(f"Runtime: {runtime:.6f} seconds")
                    torch.save(
                        model.state_dict(),
                        f'Refitting/StackedNODEs/NN_state_2HL_32nodes_stackedNODE{j}_atypicalPatient{i+1}_'
                        f'{itr}ITER_{OPT_RESET}optreset.txt'
                    )
                    with open(f'Refitting/StackedNODEs/NN_state_2HL_32nodes_stackedNODE{j}_atypicalPatient{i+1}'
                              f'_{itr}ITER_{OPT_RESET}optreset_setup.txt',
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
