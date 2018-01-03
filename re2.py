# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 02/01/2018, 19:16
"""
Modified from https://github.com/pytorch/examples/blob/master/mnist/main.py
Used for testing raw nolinear network and feedback alignment.
"""

from __future__ import print_function

import argparse
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import get_angle

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)



def one_hot(x):
    o = np.zeros([10,1])
    o[x, 0] = 1
    return o.copy()


def softmax(x):
    xs = np.exp(x)
    return xs/np.sum(xs)


def forward(X, W_0, W_1):
    """
    X: (784, 1)
    W_0: (1000, 784)
    W_1: (10, 1000)
    output: (10, 1)
    """
    z1 = W_0 * X
    h = 1. / (1 + np.exp(-z1))
    z2 = W_1 * h
    # output = 1/(1 + np.exp(z2))
    output = softmax(z2)
    return output, h


def get_label(x):
    """x.shape: [10,1]"""
    return x.A1.tolist().index(np.max(x))


def sigmoid_derivative(x):
    return np.multiply(x, 1-x)


def cross_entropy(output, label):
    return -np.sum(np.multiply(label, np.log(output)))


def train(epoch, W_0, W_1, B, lr, correct_rates, angles):
    cal_window = 100
    cals = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(784, 1).numpy()
        target = target[0]
        output, hidden = forward(data, W_0, W_1)
        label = one_hot(target)
        cost = cross_entropy(output, label)
        error = output -label

        error_h = np.multiply(B*error, sigmoid_derivative(hidden))
        error_hX = np.multiply(W_1.transpose()*error, sigmoid_derivative(hidden))

        delta_W1 = error * hidden.transpose()
        delta_W0 = error_h * data.transpose()

        W_0 = W_0 - lr * delta_W0
        W_1 = W_1 - lr * delta_W1


        cals.append(get_label(output) == target)
        if len(cals)> cal_window:
            cr = np.sum(cals[-cal_window:])/float(cal_window)
            correct_rates.append(cr)
            angles.append(get_angle(error_h.copy(), error_hX.copy()))
            print("Train Epoch: {} Correct_rate: {:.6f} Loss: {:.6f} Num: {}".format(epoch, cr, cost, batch_idx))

    return W_0.copy(), W_1.copy()


def main():
    B = np.mat(np.random.rand(1000, 10)-0.5)
    W_0 = np.mat(np.random.rand(1000, 784)*0.2-0.1)
    W_1 = np.mat(np.random.rand(10, 1000)*0.2-0.1)

    np.savez_compressed('save/B-nolinear', B)
    correct_rates = []
    angles = []
    try:
        for epoch in range(1, args.epochs+1):
            W_0, W_1 = train(epoch, W_0, W_1, B, args.lr, correct_rates, angles)
            np.savez_compressed('save/W_0-nolinear', W_0)
            np.savez_compressed('save/W_1-nolinear', W_1)
    except KeyboardInterrupt:
        np.savez_compressed('save/W_0-nolinear', W_0)
        np.savez_compressed('save/W_1-nolinear', W_1)
        print("End Training")
        pass

    angles = np.array(angles)
    plt.figure()
    plt.figure(figsize=[13,5])
    plt.subplot(121)
    plt.ylabel("Correct Rate")
    plt.semilogy(correct_rates)
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
    plt.ylim([0,90])
    plt.ylabel("Angle between BP and FA")
    plt.grid(True)
    plt.savefig('figs/e2.png')


if __name__ == '__main__':
    main()
