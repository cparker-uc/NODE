# File Name: sriram-model-plus-noise.py
# Author: Christopher Parker
# Created: Mon Apr 24, 2023 | 01:12P EDT
# Last Modified: Wed May 03, 2023 | 01:33P EDT

"""This script will generate 1000 data sets based on the Sriram model plus
normally distributed random noise on each point"""

import numpy as np
from scipy.integrate import solve_ivp

def ode_system(t, y):
    dy = np.zeros(4)

    dy[0] = (k_stress*(k_i**n2/(k_i**n2 + y[3]**n2))
             -V_S3*(y[0]/(K_m1 + y[0]))
             -K_d1*y[0])
    dy[1] = (K_P2*y[0]*(k_i**n2/(k_i**n2 + y[3]**n2))
             -V_S4*(y[1]/(K_m2 + y[1]))
             -K_d2*y[1])
    dy[2] = (K_P3*y[1]
             -V_S5*(y[2]/(K_m3 + y[2]))
             -K_d3*y[2])
    dy[3] = (K_b*y[2]*(G_tot - y[3])
             +V_S2*(y[3]**n1/(y[3]**n1 + K1**n1))
             -K_d5*y[3])

    return dy

if __name__ == "__main__":
    G_tot = 3.28
    K1 = 0.645
    K_P2 = 8.3
    K_P3 = 0.945
    K_b = 0.0202
    K_d1 = 0.00379
    K_d2 = 0.00916
    K_d3 = 0.356
    K_d5 = 0.0854
    K_m1 = 1.74
    K_m2 = 0.112
    K_m3 = 0.0768
    V_S2 = 0.0509
    V_S3 = 3.25
    V_S4 = 0.907
    V_S5 = 0.00535
    k_i = 1.51
    k_stress = 10.1
    n1 = 5.43
    n2 = 5.1

    # Time interval with 1440 points covering 24 hours (a point each minute)
    t_interval = np.linspace(0,24,1440)

    # Unsure of proper initial conditions
    y0 = (1, 5, 5, 2)

    # Solve the system
    sol = solve_ivp(ode_system, (0, 24), y0, t_eval=t_interval)

    # Generate virtual patients by adding normally distributed noise to the
    #  solution values
    vpop = 100

    # z = sol.y + np.random.normal(scale=0.2, size=sol.y.shape)*sol.y
    # z = z.reshape(len(t_interval), len(y0), 1)
    # for p in range(vpop-1):
    #     z_ = sol.y + np.random.normal(scale=0.2, size=sol.y.shape)*sol.y
    #     z_ = z_.reshape(len(t_interval), len(y0), 1)
    #     z = np.concatenate((z,z_), axis=2)

    for p in range(vpop):
        z_ = sol.y + np.random.normal(scale=0.2, size=sol.y.shape)*sol.y
        z_ = z_.reshape(len(t_interval), len(y0), 1)
        np.savetxt(f'Sriram Model w Noise/virtual_pop_normal-dist_sriram-model_vpop{p}.txt',
                   z_.reshape(z_.shape[0], -1))
        # np.savetxt(
        #     f'Sriram Model w Random ICs/sriram-model_original_num{vpop_num}.txt',
        #     np.hstack((sol.t.reshape(len(sol.t),1), sol.y.T))
        # )
    # y0 = [1, 5, 5, 2]
    # t_interval = np.linspace(0, 24, 240)
    # sol = solve_ivp(ode_system, (0, 24), y0, t_eval=t_interval)
    # np.savetxt(
    #     f'Sriram Model w Random ICs/sriram-model_original_0-24-240linspace.txt',
    #     np.hstack((sol.t.reshape(len(sol.t),1), sol.y.T))
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

