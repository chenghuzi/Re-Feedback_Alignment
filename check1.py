import pdb
import numpy as np
import matplotlib.pyplot as plt
from re1 import forward
np.set_printoptions(precision=3)

W_0 = np.mat(np.load('save/W_0.npz')['arr_0'])
W_1 = np.mat(np.load('save/W_1.npz')['arr_0'])
t_M = np.mat(np.load('save/t_M.npz')['arr_0'])

print W_1 * W_0 - t_M
pdb.set_trace()

for i in range(10):
    tmp_X = np.mat(np.random.rand(30,1))
    tmp_Y = t_M * tmp_X
    app_Y, h = forward(tmp_X, W_0, W_1)
    print "*"*71
    print tmp_Y.A1
    print app_Y.A1