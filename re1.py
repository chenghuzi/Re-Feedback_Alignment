# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 30/12/2017, 10:43

"""
for simple linear network fig.1
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import get_angle
import pdb

def forward(X, W_0, W_1):
    h = W_0 * X  # shape: row:30, column: 1
    output = W_1 * h
    return output, h


def main():
    NUM = 5000
    t_M = np.mat(np.random.rand(10, 30)*2 -1)
    np.savez_compressed('save/t_M', t_M)
    xs=[]
    ys=[]
    for i in range(NUM):
        tmp_X = np.mat(np.random.rand(30,1))
        tmp_Y = t_M * tmp_X
        xs.append(tmp_X.copy())
        ys.append(tmp_Y.copy())

    W_0 = np.mat(np.random.rand(20,30)*0.02 - 0.01)
    W_1 = np.mat(np.random.rand(10,20)*0.02 - 0.01)
    W_0_backup = W_0.copy()
    W_1_backup = W_1.copy()

    lr = 0.027
    min_lr = 0.0029
    B = np.mat(np.random.rand(20,10) - 0.5)
    error_hsFA = []
    error_hsX = []
    costsFA = []
    for i in range(NUM):
        y_approximate, hidden = forward(xs[i], W_0, W_1)
        error = ys[i] - y_approximate
        error_h = B * error
        error_hX = W_1.transpose() * error

        delta_W1 = error * hidden.transpose()
        delta_W0 = error_h * xs[i].transpose()
        if lr > min_lr:
            lr = lr *  np.power(0.96, i/NUM) 
        W_0 = W_0 + lr * delta_W0
        W_1 = W_1 + lr * delta_W1
        cost = 0.5 * error.transpose() * error
        print "costFA: {0}".format(cost)
        costsFA.append(cost[0,0])
        error_hsFA.append(error_h.copy())
        error_hsX.append(error_hX.copy())

    np.savez_compressed('save/W_0', W_0)
    np.savez_compressed('save/W_1', W_1)

    costs = []
    error_hs = []
    lr = 0.03
    for i in range(NUM):
        y_approximate, hidden = forward(xs[i], W_0_backup, W_1_backup)
        error = ys[i] - y_approximate
        error_h = W_1_backup.transpose() * error
        delta_W1 = error * hidden.transpose()
        delta_W0 = error_h * xs[i].transpose()

        W_0_backup = W_0_backup + lr * delta_W0
        W_1_backup = W_1_backup + lr * delta_W1
        cost = 0.5 * error.transpose() * error
        print "cost: {0}".format(cost)
        costs.append(cost[0,0])
        error_hs.append(error_h.copy())

    np.savez_compressed('save/W_0_backup', W_0_backup)
    np.savez_compressed('save/W_1_backup', W_1_backup)

    angles = []
    for i in range(NUM):
        vecBP =  error_hsX[i]
        vecFA = error_hsFA[i]
        angles.append(get_angle(vecFA, vecBP))
        angles.append(angle)
    angles = np.array(angles)
    np.savez_compressed('save/angles', angles)
    np.savez_compressed('save/error_hs', error_hs)
    np.savez_compressed('save/error_hsFA', error_hsFA)

    plt.figure(figsize=[13,5])
    plt.subplot(121)
    plt.semilogy(costs, label="BP")
    plt.semilogy(costsFA, label="FA")
    plt.xlim([0,NUM])
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    window = 10
    averaged_angles = []
    error_angles = []
    ns = []
    for i in range(angles.shape[0] - window):
        ags = angles[i:i+window]
        averaged_angle = np.sum(ags)/float(window)
        error_angle = np.std(ags)
        averaged_angles.append(averaged_angle)
        error_angles.append(error_angle)
        ns.append(i)

    plt.errorbar(ns, averaged_angles, yerr=error_angles, color="#19AD1D", ecolor="#B5E6B5")
    plt.xlim([0,NUM])
    plt.ylim([0,90])
    plt.grid(True)
    plt.ylabel("Angle between BP and FA")
    plt.savefig('figs/e1.png')
    # plt.show()

if __name__ == '__main__':
    main()