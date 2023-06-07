# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Tue Jun 06, 2023 | 12:27P EDT

"""Implementing the torchdyn library Galerkin NODE class to allow
depth-variance among the neural network parameters"""

INPUT_CHANNELS = 2
HDIM = 32
OUTPUT_CHANNELS = 2

BATCH_SIZE = 3
ITERS = 2000
LR = 1e-3
DECAY = 1e-6
OPT_RESET = None
ATOL = 1e-5
RTOL = 1e-5
METHOD = 'dopri5'

PATIENT_GROUP = 'Atypical'

# from IPython.core.debugger import set_trace
import os
import time
import torch
import torchcde
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchdyn.core import NeuralODE
from torchdyn.nn.node_layers import DepthCat
from torchdyn.nn.galerkin import GalLinear, Fourier, Polynomial, Chebychev
from typing import Tuple


# Not certain if this is necessary, but in the quickstart docs they have
#  done a wildcard import of torchdyn base library, and this is all that does
TTuple = Tuple[torch.Tensor, torch.Tensor]


class CDEFunc(torch.nn.Module):
    """CDEs are defined as: z_t = z_0 + \int_{0}^t f_{theta}(z_s) dX_s, where
    f_{theta} is a neural network (and X_s is a rough path controlling the
    diff eq. This class defines f_{theta}"""
    def __init__(self, input_channels, hidden_channels):
          super().__init__()
          self.input_channels = input_channels
          self.hidden_channels = hidden_channels

          # Define the layers of the NN, with 128 hidden nodes (arbitrary)
          self.linear1 = torch.nn.Linear(hidden_channels, 12)
          self.linear2 = torch.nn.Linear(12, hidden_channels*input_channels)

    def forward(self, t, z):
          """t is passed as an argument by the solver, but it is unused in most
          cases"""
          z = self.linear1(z)
          # print(f'self.linear1.parameters(): {[p for p in self.linear1.parameters()]}')
          z = z.relu()
          z = self.linear2(z)
          # print(f'self.linear2.parameters(): {[p for p in self.linear2.parameters()]}')

          # The first author of the NCDE paper (Kidger) suggests that using tanh
          #  for the final activation leads to better results
          z = z.tanh()

          # The output represents a linear map from R^input_channels to
          #  R^hidden_channels, so it takes the form of a
          #  (hidden_channels x input_channels) matrix
          z = z.view(z.size(0), self.hidden_channels, self.input_channels)
          return z


class NeuralCDE(torch.nn.Module):
    """This class packages the CDEFunc class with the torchcde NCDE solver,
    so that when we call the instance of NeuralCDE it solves the system"""
    def __init__(self, input_channels, hidden_channels, output_channels,
                 t_interval=None, interpolation='cubic'):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels)

        # self.initial represents l_{theta}^2 in the equation
        #  z_0 = l_{theta}^2 (x)
        # This is essentially augmenting the dimension with a linear map,
        #  something Massaroli et al warned against
        self.initial = torch.nn.Linear(input_channels, hidden_channels)

        # self.readout represents l_{theta}^1 in the equation
        #  y ~ l_{theta}^1 (z_T)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation
        self.t_interval = t_interval

    def forward(self, coeffs):
        """coeffs is the coefficients that describe the spline between the
        datapoints. In the case of cubic interpolation (the default), this
        is a, b, 2c and 3d because the derivative of the spline is used more
        often with cubic Hermite splines"""
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=self.t_interval)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only cubic and linear interpolation are implemented"
            )

        # The initial hidden state
        #   (a linear map from the first observation)
        X0 = X.evaluate(X.interval[0]) # evaluate the spline at its first point
        z0 = self.initial(X0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=self.t_interval
        )
        # cdeint returns the initial value and terminal value from the
        #  integration, we only need the terminal
        # z_T = z_T[:,1]

        # Convert z_T to y with the readout linear map
        pred_y = self.readout(z_T)
        return pred_y


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
          ACTH_mean = torch.mean(label[:,0])
          CORT_mean = torch.mean(label[:,1])
          label_norm = torch.cat(((label[:,0]/ACTH_mean).reshape(-1,1), (label[:,1]/CORT_mean).reshape(-1,1)), 1)
          return data, label_norm


class NDEOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tup):
        (t_eval, sol) = tup
        # The result returned from NeuralODE is (11, 1, 2) instead of
        #  (11, 2, 1) so we swap the last two axes
        # return torch.swapaxes(sol, 1, 2)
        return sol


def NODE_main():
    device = torch.device('cpu')

    for i in range(1):
        dataset = NelsonData('Nelson TSST Individual Patient Data', PATIENT_GROUP)
        data, label = dataset[i]
        data = data
        label = label
        t_eval = data[:,0]  # Time points we need the solver to output
        y0 = label[0,:]     # ICs for the vector field

        f = nn.Sequential(
            nn.Linear(2, HDIM),
            nn.ReLU(),
            # DepthCat(1),
            # GalLinear(HDIM, HDIM, expfunc=Chebychev(15)),
            # nn.Tanh(),
            nn.Linear(HDIM, HDIM),
            nn.ReLU(),
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

        # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss = nn.MSELoss()
        loss_over_time = []

        start_time = time.time()
        for itr in range(1, ITERS+1):
            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = model(y0)

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
                    # f'Refitting/NN_state_2HL_32nodes_Galerkin_AdamW_Tanh_atypicalPatient{i+1}_'
                    f'Refitting/NN_state_2HL_32nodes_Adam_ReLU_atypicalPatient{i+1}_'
                    f'{itr}ITER_{OPT_RESET}optreset_normed.txt'
                )
                # with open(f'Refitting/NN_state_2HL_32nodes_Galerkin_AdamW_Tanh_atypicalPatient{i+1}'
                with open(f'Refitting/NN_state_2HL_32nodes_Adam_ReLU_atypicalPatient{i+1}'
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


def NCDE_main():
    device = torch.device('cpu')

    dataset = NelsonData('Nelson TSST Individual Patient Data', PATIENT_GROUP)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    t_eval = dataset[0][0][:,0]  # Time points we need the solver to output

    model = NeuralCDE(2, 2, 2, t_interval=t_eval).double()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    # optimizer = optim.Adam(model.parameters(), lr=LR)

    loss = nn.MSELoss()
    loss_over_time = []

    start_time = time.time()
    for itr in range(1, ITERS+1):
        for j, (_, labels) in enumerate(dataloader):
            # labels = labels.reshape(-1,11,2)
            # y0 = label[0,:]     # ICs for the vector field

            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(labels, t=t_eval)

            optimizer.zero_grad()

            # Compute the forward direction of the NODE
            pred_y = model(coeffs)

            # Compute the loss based on the results
            output = loss(pred_y, labels)
            loss_over_time.append((j, output.item()))

            # Backpropagate through the adjoint of the NODE to compute gradients
            #  WRT each parameter
            output.backward()

            # Use the gradients calculated through backpropagation to adjust the 
            #  parameters
            optimizer.step()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d} Batch {j}: loss = {output.item():.6f}")

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
                # f'Refitting/NN_state_2HL_128nodes_NCDE_{PATIENT_GROUP}{i+1}_'
                f'Refitting/NN_state_2HL_128nodes_NCDE_{PATIENT_GROUP}_batchsize3_'
                f'{itr}ITER_normed.txt'
            )
            # with open(f'Refitting/NN_state_2HL_128nodes_NCDE_{PATIENT_GROUP}{i+1}'
            with open(f'Refitting/NN_state_2HL_128nodes_NCDE_{PATIENT_GROUP}_batchsize3'
                      f'_{itr}ITER_normed_setup.txt',
                      'w+') as file:
                file.write(f'Model Setup for {PATIENT_GROUP} Trained Network:\n')
                file.write(
                    f'BATCH_SIZE={BATCH_SIZE}'
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


if __name__ == "__main__":
    NCDE_main()

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
