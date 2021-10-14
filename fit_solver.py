import numpy as np
import pandas as pd
import csv
from iminuit import Minuit
from iminuit.cost import LeastSquares
# import BK

from dis_solver import Solve as dis
import warnings
warnings.filterwarnings("ignore")

# theory

alpha = 1/137
def f2(x_, qsq2_):
    sig_t = dis.rhs(x_, qsq2_, "T")
    sig_l = dis.rhs(x_, qsq2_, "L")

    f2 = (qsq2_/(4 * np.pi * np.pi * alpha)) * (sig_t + sig_l)
    return f2

# data import
data = pd.read_csv("f2_dat.csv.csv")
data.columns = ['qsq2', 'x', 'f2', 'f2_staterr', 'f2_syserr', 'f2_experr']

qsq2 = np.array(data.qsq2)
x = np.array(data.x)
f2_dat = np.array(data.f2)
f2_err = np.array(data.f2_experr)

# fit performance
def chi_squared(qsq20, c, gamma):
    # run BK for given parameters qsq2, c, and gamma
    res = 0
    for i in range(len(f2_dat)):
        theory = f2(x[i], qsq2[i])
        exp = f2_dat[i]
        err = f2_err[i]
        res += (theory - exp) * (theory - exp) / (err * err)
    return res

chi_squared.errordef = Minuit.LEAST_SQUARES
m = Minuit(chi_squared, qsq20=10.08, c=1, gamma=1)
m.simplex()
