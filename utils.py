# -*- coding: utf-8 -*-
# @Author: Huzi Cheng
# @Date: 02/01/2018, 01:59

"""
tools
"""
import numpy as np

def get_angle(vecFA, vecBP):
    x = np.power(np.sum(np.power(vecFA.transpose() * vecBP,2)), 0.5)
    y = np.power(np.sum(np.power(vecBP,2)), 0.5) * np.power(np.sum(np.power(vecFA, 2)), 0.5)
    angle = np.arccos(x/float(y))/np.pi * 180
    return angle