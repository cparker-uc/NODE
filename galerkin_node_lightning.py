# File Name: galerkin_node.py
# Author: Christopher Parker
# Created: Tue May 30, 2023 | 03:04P EDT
# Last Modified: Tue May 30, 2023 | 04:38P EDT

"""Implementing the torchdyn library Galerkin NODE class to allow
depth-variance among the neural network parameters"""

HDIM = 32

from IPython.core.debugger import set_trace
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchdyn.core import NeuralODE
from torchdyn.nn.node_layers import DepthCat
from torchdyn.nn.galerkin import GalLayer, GalLinear
from typing import Tuple


# Not certain if this is necessary, but in the quickstart docs they have
#  done a wildcard import of torchdyn base library, and this is all that does
TTuple = Tuple[torch.Tensor, torch.Tensor]

class Learner(pl.LightningModule):
    """This class implements the training steps for a given model"""
    def __init__(self, model:nn.Module, dataset:Dataset):
        super().__init__()
        self.model = model
        self.iter = 0 # Initialize the iteration counter to 0
        self.dataset = dataset

    def forward(self, x):
        "Calls the forward method of the associated model"
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.iter += 1
        data, label = batch
        y0 = label[:,0]
        y_pred = self.model(y0)
        loss = nn.MSELoss()(y_pred, label)
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-6)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1)

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
        return torch.swapaxes(sol, 1, 2)

if __name__ == "__main__":
    # set_trace()
    device = torch.device('cpu')

    dataset = NelsonData('Nelson TSST Individual Patient Data', 'Atypical')
    data, _ = dataset[0]

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

    nde = NeuralODE(
        f, data[:,0], sensitivity='adjoint', solver='dopri5',
        atol=1e-5, rtol=1e-5
    ).to(device)

    out_layer = NDEOutputLayer()

    model = nn.Sequential(nde, out_layer).to(device)

    learn = Learner(model, dataset)
    trainer = pl.Trainer(max_epochs=2000, devices=8, accelerator='cpu')
    trainer.fit(learn)

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

