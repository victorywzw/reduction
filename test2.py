# -*- coding:utf-8 -*-

from model_reduction import model_reduction
from GenData_multiprocess import *
# 系统的包
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import random
import sys, os
import time
import json
import cantera as ct
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt 


x = np.array([  1.5066,  -5.4707,  -1.8087,   0.4626,  -5.3258,  -1.2961,  -3.9397,
         -5.1551,  -4.8491,  -1.8794,  -7.8700,  -2.0913,  -1.7734,  -3.0201,
          0.2048,  -3.8339,   1.4628,  -4.7023,  -1.4152,  -1.2340,  -3.3672,
          9.7082, -14.2374,   0.3927,  -4.5013,  -5.9520,  -0.9197])
x = 1 * (1 / (1 + np.exp(-100 * x)) > 0.9)
print(x.tolist())


# x = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# x = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
# a = np.array([[-1,2],[3,4]])
# print(np.abs(a))


