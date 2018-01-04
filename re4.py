# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 02/01/2018, 19:16
"""
Modified from https://github.com/pytorch/examples/blob/master/mnist/main.py
Used for testing raw nolinear network and feedback alignment.
deeper neural network

benchmark
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
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def one_hot(x):
    o = np.zeros([10, 1])
    o[x, 0] = 1
    return o.copy()


def softmax(x):
    xs = np.exp(x)
    return xs/np.sum(xs)


def forward(X, W_0, W_1, W_2):
    """
    X: (784, 1)
    W_0: (1000, 784)
    W_1: (100, 1000)
    W_1: (10, 100)
    output: (10, 1)
    """
    z1 = W_0 * X
    h1 = 1. / (1 + np.exp(-z1))
    z2 = W_1 * h1
    h2 = 1. / (1 + np.exp(-z2))
    z3 = W_2 * h2
    output = softmax(z3)
    return output, h2, h1


def get_label(x):
    """x.shape: [10,1]"""
    return x.A1.tolist().index(np.max(x))


def sigmoid_derivative(x):
    return np.multiply(x, 1-x)


def softmax_derivative(output, label):
    return output -label


def cross_entropy(output, label):
    return -np.sum(np.multiply(label, np.log(output)))


def train(epoch, W_0, W_1, W_2, B1, B2, lr):
    cal_window = 100
    cals = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(784, 1).numpy()
        target = target[0]
        output, hidden2, hidden1 = forward(data, W_0, W_1, W_2)
        label = one_hot(target)
        cost = cross_entropy(output, label)
        error = output - label

        error_h2 = np.multiply(B2*error, sigmoid_derivative(hidden2))
        error_h2X = np.multiply(W_2.transpose()*error, sigmoid_derivative(hidden2))

        error_h1 = np.multiply(B1*error_h2, sigmoid_derivative(hidden1))
        error_h1X = np.multiply(W_1.transpose()*error_h2X, sigmoid_derivative(hidden1))
        # pdb.set_trace()
        delta_W2 = error * hidden2.transpose()
        delta_W1 = error_h2X * hidden1.transpose()
        delta_W0 = error_h1X * data.transpose()

        W_0 = W_0 - lr * delta_W0
        W_1 = W_1 - lr * delta_W1
        W_2 = W_2 - lr * delta_W2

        cals.append(get_label(output) == target)
        if len(cals)> cal_window:
            cr = np.sum(cals[-cal_window:])/float(cal_window)
            print("Train Epoch: {} Correct_rate: {:.6f} Loss: {:.6f} Num: {}".format(epoch, cr, cost, batch_idx))

    return W_0.copy(), W_1.copy()


def main():
    B1 = np.mat(np.random.rand(1000, 100)-0.5)
    B2 = np.mat(np.random.rand(100, 10)-0.5)
    W_0 = np.mat(np.random.rand(1000, 784)*0.02-0.01)
    W_1 = np.mat(np.random.rand(100, 1000)*0.02-0.01)
    W_2 = np.mat(np.random.rand(10, 100)*0.02-0.01)

    np.savez_compressed('save/B1-nolinear_deep', B1)
    np.savez_compressed('save/B2-nolinear_deep', B2)

    for epoch in range(1, args.epochs + 1):
        W_0, W_1 = train(epoch, W_0, W_1, W_2, B1, B2, args.lr)

    np.savez_compressed('save/W_0-nolinear', W_0)
    np.savez_compressed('save/W_1-nolinear', W_1)

if __name__ == '__main__':
    main()
