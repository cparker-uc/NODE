# File Name: gen_datafiles_Nelson.py
# Author: Christopher Parker
# Created: Thu Mar 30, 2023 | 04:20P EDT
# Last Modified: Thu Mar 30, 2023 | 04:53P EDT

"""Generate text files containing the Nelson data for each individual patient
with time in the first column and data in the second"""

import pandas as pd
import numpy as np

ACTH_df = pd.read_excel('nelson_data_ACTH-and-CORT.xlsx', sheet_name='ACTH')
CORT_df = pd.read_excel('nelson_data_ACTH-and-CORT.xlsx', sheet_name='CORT')
times = np.array([0, 15, 30, 40, 50, 65, 80, 95, 110, 125, 140])

subtypes = np.genfromtxt('nelson-MDD-subtypes-and-PatientID-vs-Number.txt')

atypicalACTH_list = []
atypicalCORT_list = []
melancholicACTH_list = []
melancholicCORT_list = []
neitherACTH_list = []
neitherCORT_list = []
controlACTH_list = []
controlCORT_list = []

for index, subtype in enumerate(subtypes[:,1]):
    ACTH_arr = np.vstack((times, np.array(ACTH_df[f'Patient {index+1}'])))
    CORT_arr = np.vstack((times, np.array(CORT_df[f'Patient {index+1}'])))

    match subtype:
        case 1:
            atypicalACTH_list.append(ACTH_arr.T)
            atypicalCORT_list.append(CORT_arr.T)
            # np.savetxt(f'AtypicalPatient{index}_ACTH.txt', ACTH_arr)
            # np.savetxt(f'AtypicalPatient{index}_Cortisol.txt', CORT_arr)
        case 2:
            melancholicACTH_list.append(ACTH_arr.T)
            melancholicCORT_list.append(CORT_arr.T)
            # np.savetxt(f'MelancholicPatient{index}_ACTH.txt', ACTH_arr)
            # np.savetxt(f'MelancholicPatient{index}_Cortisol.txt', CORT_arr)
        case 3:
            neitherACTH_list.append(ACTH_arr.T)
            neitherCORT_list.append(CORT_arr.T)
            # np.savetxt(f'NeitherPatient{index}_ACTH.txt', ACTH_arr)
            # np.savetxt(f'NeitherPatient{index}_Cortisol.txt', CORT_arr)
        case 4:
            controlACTH_list.append(ACTH_arr.T)
            controlCORT_list.append(CORT_arr.T)
            # np.savetxt(f'ControlPatient{index}_ACTH.txt', ACTH_arr)
            # np.savetxt(f'ControlPatient{index}_Cortisol.txt', CORT_arr)

for index, data in enumerate(atypicalACTH_list):
    np.savetxt(f'AtypicalPatient{index+1}_ACTH.txt', data)

for index, data in enumerate(atypicalCORT_list):
    np.savetxt(f'AtypicalPatient{index+1}_CORT.txt', data)

for index, data in enumerate(melancholicACTH_list):
    np.savetxt(f'MelancholicPatient{index+1}_ACTH.txt', data)

for index, data in enumerate(melancholicCORT_list):
    np.savetxt(f'MelancholicPatient{index+1}_CORT.txt', data)

for index, data in enumerate(neitherACTH_list):
    np.savetxt(f'NeitherPatient{index+1}_ACTH.txt', data)

for index, data in enumerate(neitherCORT_list):
    np.savetxt(f'NeitherPatient{index+1}_CORT.txt', data)

for index, data in enumerate(controlACTH_list):
    np.savetxt(f'ControlPatient{index+1}_ACTH.txt', data)

for index, data in enumerate(controlCORT_list):
    np.savetxt(f'ControlPatient{index+1}_CORT.txt', data)
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

