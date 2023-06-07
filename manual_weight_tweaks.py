# File Name: manual_weight_tweaks.py
# Author: Christopher Parker
# Created: Tue Jun 06, 2023 | 02:39P EDT
# Last Modified: Tue Jun 06, 2023 | 03:02P EDT

"""This script loads a state dict for a trained network and manually tweaks
some weights to see what the weights control."""

import os
import torch
import torchcde
import numpy as np
from model_testing import runModel_mean_NCDE_normed
from galerkin_node import NeuralCDE

device = torch.device('cpu')
state = torch.load("Refitting/NN_state_2HL_12nodes_NCDE_Atypical_batchsize3_2000ITER_normed.txt")

state['func.linear1.bias'] = torch.zeros(12)

with torch.no_grad():
    runModel_mean_NCDE_normed('AtypicalMDD', state, device, 2, 2, 2, '_testing_weight_tweaks_linear1biaseq0_Atypical_batchsize3')
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

