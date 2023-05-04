# File Name: model.py
# Author: Christopher Parker
# Created: Fri Mar 24, 2023 | 10:10P EDT
# Last Modified: Wed May 03, 2023 | 01:43P EDT

"First pass at an NODE model with PyTorch"

ITERS = 100
LEARNING_RATE = 1e-3
OPT_RESET = 200
ATOL = 1e-9
RTOL = 1e-7
METHOD = 'dopri5'

# from IPython.core.debugger import set_trace
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hpa_net = nn.Sequential(
            nn.Linear(4, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 11, bias=True),
            nn.ReLU(),
            nn.Linear(11, 4, bias=True)
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

def main():
    # Load the virtual patients we have created by adding noise to a Sriram
    #  model solution
    vpop = np.loadtxt('virtual_pop_normal-dist_sriram-model.txt')
    vpop_size = 100
    vpop = torch.as_tensor(vpop.reshape(vpop.shape[0], vpop.shape[1]//vpop_size, vpop_size))

    # Define the device to use for neural network computations
    device = torch.device('cpu')

    # We need to convert the model parameters to double precision because
    #  that is the format of the datasets and they must match
    func = NN().double().to(device)

    # List of parameters to optimize
    opt_params = list(func.parameters())

    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
    loss = nn.MSELoss()

    # Initialize tensor to track change in loss over each iteration
    loss_over_time = torch.zeros(ITERS)

    start_time = time.time()
    # Start main optimization loop
    for itr in range(1, ITERS + 1):
        for i in range(vpop_size-1):
            label = vpop[:,1:,i]
            y0_tensor = label[0,:]

            # Reset gradient for each training example
            optimizer.zero_grad()

            pred_y = odeint(
                func,
                y0_tensor,
                torch.linspace(0,24,1440),
                rtol=RTOL,
                atol=ATOL,
                method=METHOD,
            )
            # Compute the loss for this iteration
            output = loss(pred_y, label)

            # Backpropagation to calculate the gradient from the loss
            output.backward()

            # Step the optimizer with the new gradient
            optimizer.step()

            # Save the loss value to the loss_over_time tensor
            loss_over_time[itr-1] = output.item()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}, Patient {i}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate
            if itr % OPT_RESET == 0:
                optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

        runtime = time.time() - start_time
        print(f"Iteration {itr} Runtime: {runtime:.6f} seconds")
    torch.save(
        func.state_dict(),
        f'NN_state_2HL_11nodes_sriram-model_normal-dist.txt'
    )

def main_noise():
    # Load the virtual patients we have created by adding noise to a Sriram
    #  model solution
    vpop_size = 100
    vpop = torch.zeros(size=(1440,4,vpop_size))
    for p in range(vpop_size):
        vpop_ = np.loadtxt('Sriram Model w Noise/virtual_pop_normal-dist_'
                          f'sriram-model_vpop{p}.txt')
        vpop[...,p] = torch.as_tensor(vpop_)

    # Define the device to use for neural network computations
    device = torch.device('cpu')

    # We need to convert the model parameters to double precision because
    #  that is the format of the datasets and they must match
    func = NN().to(device)

    # List of parameters to optimize
    opt_params = list(func.parameters())

    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
    loss = nn.MSELoss()

    # Initialize tensor to track change in loss over each iteration
    loss_over_time = torch.zeros(ITERS)

    start_time = time.time()
    # Start main optimization loop
    for itr in range(1, ITERS + 1):
        for i in range(vpop_size):
            label = vpop[...,i]
            y0_tensor = label[0,:]

            # Reset gradient for each training example
            optimizer.zero_grad()

            pred_y = odeint(
                func,
                y0_tensor,
                torch.linspace(0,24,1440),
                rtol=RTOL,
                atol=ATOL,
                method=METHOD,
            )
            # Compute the loss for this iteration
            output = loss(pred_y, label)

            # Backpropagation to calculate the gradient from the loss
            output.backward()

            # Step the optimizer with the new gradient
            optimizer.step()

            # Save the loss value to the loss_over_time tensor
            loss_over_time[itr-1] = output.item()

            # If this is the first iteration, or a multiple of 100, present the
            #  user with a progress report
            if (itr == 1) or (itr % 10 == 0):
                print(f"Iter {itr:04d}, Patient {i}: loss = {output.item():.6f}")

            # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
            #  reset the learning rate
            if itr % OPT_RESET == 0:
                optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

        runtime = time.time() - start_time
        print(f"Iteration {itr} Runtime: {runtime:.6f} seconds")
    torch.save(
        func.state_dict(),
        f'NN_state_2HL_11nodes_sriram-model_normal-dist.txt'
    )


def main_randomICs(num):
    # Load the virtual patients we have created by adding noise to a Sriram
    #  model solution
    vpop = np.loadtxt(
        f'Sriram Model w Random ICs/sriram-model_original_num{num}.txt'
    )
    vpop_size = 1
    vpop = torch.as_tensor(vpop.reshape(vpop.shape[0], vpop.shape[1]//vpop_size, vpop_size))

    # Define the device to use for neural network computations
    device = torch.device('cpu')

    # We need to convert the model parameters to double precision because
    #  that is the format of the datasets and they must match
    func = NN().double().to(device)

    # List of parameters to optimize
    opt_params = list(func.parameters())

    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
    loss = nn.MSELoss()

    # Initialize tensor to track change in loss over each iteration
    loss_over_time = torch.zeros(ITERS)

    start_time = time.time()
    # Start main optimization loop
    for itr in range(1, ITERS + 1):
        label = vpop[:,1:,0]
        y0_tensor = label[0,:]

        # Reset gradient for each training example
        optimizer.zero_grad()

        pred_y = odeint(
            func,
            y0_tensor,
            torch.linspace(0,24,1440),
            rtol=RTOL,
            atol=ATOL,
            method=METHOD,
        )
        # Compute the loss for this iteration
        output = loss(pred_y, label)

        # Backpropagation to calculate the gradient from the loss
        output.backward()

        # Step the optimizer with the new gradient
        optimizer.step()

        # Save the loss value to the loss_over_time tensor
        loss_over_time[itr-1] = output.item()

        # If this is the first iteration, or a multiple of 100, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d}: loss = {output.item():.6f}")

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate
        if itr % OPT_RESET == 0:
            optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

    runtime = time.time() - start_time
    print(f'Runtime: {runtime}')
    torch.save(
        func.state_dict(),
        f'Sriram Model w Random ICs/NN_state_2HL_11nodes_'
        f'sriram-model_no-noise_num{num}.txt'
    )


def main_noNoise():
    # Load the virtual patients we have created by adding noise to a Sriram
    #  model solution
    vpop = np.loadtxt('sriram-model_original_0-24-24linspace.txt')
    vpop_size = 1
    vpop = torch.as_tensor(vpop.reshape(vpop.shape[0], vpop.shape[1]//vpop_size, vpop_size))
    # set_trace()

    # Define the device to use for neural network computations
    device = torch.device('cpu')

    # We need to convert the model parameters to double precision because
    #  that is the format of the datasets and they must match
    func = NN().double().to(device)

    # List of parameters to optimize
    opt_params = list(func.parameters())

    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)
    loss = nn.MSELoss()

    # Initialize tensor to track change in loss over each iteration
    loss_over_time = torch.zeros(ITERS)

    start_time = time.time()
    # Start main optimization loop
    for itr in range(1, ITERS + 1):
        label = vpop[:,1:,0]
        y0_tensor = label[0,:]

        # Reset gradient for each training example
        optimizer.zero_grad()

        pred_y = odeint(
            func,
            y0_tensor,
            torch.linspace(0,24,24),
            rtol=RTOL,
            atol=ATOL,
            method=METHOD,
        )
        # Compute the loss for this iteration
        output = loss(pred_y, label)

        # Backpropagation to calculate the gradient from the loss
        output.backward()

        # Step the optimizer with the new gradient
        optimizer.step()

        # Save the loss value to the loss_over_time tensor
        loss_over_time[itr-1] = output.item()

        # If this is the first iteration, or a multiple of 100, present the
        #  user with a progress report
        if (itr == 1) or (itr % 10 == 0):
            print(f"Iter {itr:04d}: loss = {output.item():.6f}")

        # If itr is a multiple of OPT_RESET, re-initialize the optimizer to
        #  reset the learning rate
        if itr % OPT_RESET == 0:
            optimizer = optim.Adam(opt_params, lr=LEARNING_RATE)

    runtime = time.time() - start_time
    print(f'Runtime: {runtime}')
    torch.save(
        func.state_dict(),
        f'NN_state_2HL_24nodes_sriram-model_no-noise_0-24-24linspace.txt'
    )


if __name__ == '__main__':
    # Train the network against the control parameters from Sriram Model
    # for i in range(100):
    #     main_randomICs(i)
    main_noise()
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
