import numpy as np
import scipy.interpolate as interpolate
# from scipy.interpolate import PchipInterpolator as chip
# from scipy.interpolate import Akima1DInterpolator as aki
from scipy.integrate import simps
from scipy.integrate import dblquad as qq
# from numba import jit, float64
import csv
from multiprocessing import Pool
from scipy.signal import butter, filtfilt, medfilt, sosfiltfilt
# try ndimage.medfilt, and ndimage.meanfilt
# from scipy.ndimage import medfilt, meanfilt
import time
import pandas as pd
# import math
import matplotlib.pyplot as plt
# from math import factorial
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# import statsmodels.api as sm
# import cython

import warnings
warnings.filterwarnings('ignore')
class Solve():

    def __init__(self):

        # variables
        self.n = 399  # number of r points to be evaluated at each evolution step in Y
        self.r1 = 3.e-6                    # limits of r
        self.r2 = 60.e0

        self.xr1 = np.log(self.r1)
        self.xr2 = np.log(self.r2)

        self.hr = (self.xr2 - self.xr1)/self.n

        self.hy = 0.1                                 # step size in rapidity evolution
        self.ymax = 30.0                              # limit of evolution in rapidity
        self.y = np.arange(0.0, self.ymax, self.hy)  # array of rapidity values

        # parameters
        self.lamb = 0.241  # lambda_QCD
        self.x0 = 0.01  # beginning of small-x evolution
        self.qs02 = 0.4 # saturation scale, Qs0^2 GeV^2 !!!!! CHECK BC THIS DIFFERS FROM ORIGINAL BK SOLUTION (10.08) !!!!!!
        self.gamma = 1.  # fit parameter
        self.nc = 3
        self.nf = 3  # number of active flavors
        self.c = 1

        self.beta = (11 * self.nc - 2. * self.nf)/(12 * np.pi)
        self.afr = 0.7  # frozen coupling constant
        self.rfr = (2./self.lamb) * np.exp(-0.5/(self.beta * self.afr))  # IR cutoff

        # Arrays for N and r in N(r), evaluated at some rapidity Y (including next step N(r,Y) in the evolution
        self.xlr_ = [self.xr1 + i * self.hr for i in range(self.n + 1)]
        self.r_ = np.exp(self.xlr_)
        self.n_ = np.zeros(self.n + 1)

        self.index = 0
        self.r0 = 0
        self.xr0 = 0
        self.nr0 = 0

        self.method = "RK4"

# integration/filtering routines
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def dbl_simp(self, func, ain, bin, aout, bout, nin, nout):
        # integral = self.simps(lambda u: self.simps(lambda v: func(u, v), ain, bin, N=nin), aout, bout, N=nout)
        dt = (bout - aout)/nout
        dr = (bin - ain)/nin
        t = np.linspace(aout, bout, nout + 1)
        r = np.linspace(ain, bin, nin + 1)

        # reshape to use broadcasting
        zz = func(t.reshape(-1,1), r.reshape(1,-1))

        integral = simps([simps(zz_r, r) for zz_r in zz], t)
        return integral

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def find_r1(self, r, z, thet):
        r12 = (0.25 * r * r) + (z * z) - (r * z * np.cos(thet))
        return np.sqrt(r12)

    def find_r2(self, r, z, thet):
        r22 = (0.25 * r * r) + (z * z) + (r * z * np.cos(thet))
        return np.sqrt(r22)

    def mv(self, r):  # initial conditions, rapidity y = 0
        qsq = 0.168*60.0
        p1 = (qsq * r * r)/4
        p2 = np.log(1/(self.lamb * r) + np.exp(1))

        out = 1 - np.exp(-np.power(p1, self.gamma) * p2)
        return out

    # @jit()
    def alphaS(self, rsq):  # running coupling, frozen at alpha=0.7
        if np.sqrt(rsq) > self.rfr:
            return self.afr
        else:
            log = np.log((4 * self.c * self.c)/(rsq * 0.241 * 0.241))  # Lambda_QCD = 0.241 squared
            return 1/(self.beta * log)

    # 5th order polynomial interpolation over logarithmic grid
    # for r <= rmin, extrapolate as 1/r^2, and for r > rmax, freeze at afr=0.7
    def nfuncv(self, qlr):
        x = 0.0
        if qlr < self.xr1:
            # ex = 1.99
            # x = self.n_[0] * np.power(np.exp(qlr)/self.r_[0], ex)  # check check
            x = self.n_[0] * (np.exp(2 * qlr))/(self.r_[0] * self.r_[0])
        elif qlr >= self.xr2:
            x = 1.
        else:
            # f = interpolate.interp1d(self.xlr_, self.n_, kind=5)
            x = self.n_interpolated(qlr)[()]

        if x < 0.: return 0.0
        if x > 1.: return 1.0

        return x

    def nfunc(self, qlr):
        f = np.vectorize(self.nfuncv)
        return f(qlr)

    def kv(self, r, r1, r2):
        if (r1 < 1e-20) or (r2 < 1e-20):
            return 0
        else:
            rr = r * r
            r12 = r1 * r1
            r22 = r2 * r2

            t1 = rr/(r12 * r22)
            t2 = (1/r12) * (self.alphaS(r12)/self.alphaS(r22) - 1)
            t3 = (1/r22) * (self.alphaS(r22)/self.alphaS(r12) - 1)

            prefac = (self.nc * self.alphaS(rr))/(2 * np.pi * np.pi)

            return prefac * (t1 + t2 + t3)

    def k(self, r, r1, r2):
        f = np.vectorize(self.kv)
        return f(r, r1, r2)

    def f_kernel(self, theta, a):
        z = np.exp(a)
        r1 = self.find_r1(self.r0, z, theta)
        r2 = self.find_r2(self.r0, z, theta)

        return 2 * z * z * self.k(self.r0, r1, r2)

    def f_split(self, theta, a):
        z = np.exp(a)
        r1 = self.find_r1(self.r0, z, theta)
        r2 = self.find_r2(self.r0, z, theta)

        return 2 * z * z * self.k(self.r0, r1, r2) * (self.nfunc(np.log(r1)) + self.nfunc(np.log(r2)))


    def f_recomb(self, theta, a):
        z = np.exp(a)
        r1 = self.find_r1(self.r0, z, theta)
        r2 = self.find_r2(self.r0, z, theta)
        return 2 * z * z * self.k(self.r0, r1, r2) * self.nfunc(np.log(r1)) * self.nfunc(np.log(r2))

    def f_combined(self, theta, a):

        z = np.exp(a)
        r1 = self.find_r1(self.r0, z, theta)
        r2 = self.find_r2(self.r0, z, theta)

        xlr1 = np.log(r1)
        xlr2 = np.log(r2)

        nr1 = self.nfunc(xlr1)
        nr2 = self.nfunc(xlr2)

        return 2 * z * z * self.k(self.r0, r1, r2) * (nr1 + nr2 - self.nr0 - nr1 * nr2)

    def filtered_nfunc(self, xr, xn):
        # in x: numpy array
        # out: interpolatd object over (xr, xn) such that xn is monotonically increasing, and has values between [0, 1]

        sos = butter(1, 0.97, output='sos') # orignally order=2, wn=0.1, 0.125. wn=0.999 not good
        y_ = sosfiltfilt(sos, xn)
        return interpolate.CubicSpline(xr, y_)

    def evolve(self, xlr):
        h = self.hy
        self.index = self.xlr_.index(xlr)
        self.xr0 = xlr
        self.r0 = np.exp(self.xr0)
        self.nr0 = self.n_[self.index]
        Kernel = self.dbl_simp(self.f_kernel, self.xr1, self.xr2, 0, np.pi, nin=175, nout=30)  # 175, original final results
        Split = self.dbl_simp(self.f_split, self.xr1, self.xr2, 0, np.pi, nin=175, nout=30)
        Combined = self.dbl_simp(self.f_combined, self.xr1, self.xr2, 0, np.pi, nin=175, nout=30)


        # k1_ = Split - self.nr * Kernel - Recombined
        k1 = Combined
        k2 = k1 + (0.5 * h * k1 * Kernel) - (0.5 * h * k1 * Split) - (0.25 * h * h * k1 * k1 * Kernel)
        k3 = k1 + (0.5 * h * k2 * Kernel) - (0.5 * h * k2 * Split) - (0.25 * h * h * k2 * k2 * Kernel)
        k4 = k1 + (0.5 * h * k3 * Kernel) - (0.5 * h * k3 * Split) - (0.25 * h * h * k3 * k3 * Kernel)

        return (1/6) * h * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(self):
        with open("temp.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter="\t")
            writer.writerow(["y", "r", "N(r,Y)"])

            # initial condition----------------------------------------------------------
            self.n_ = [self.mv(self.r_[i]) for i in range(len(self.r_))]
            self.n_interpolated = interpolate.interp1d(self.xlr_, self.n_, kind='cubic')
            #----------------------------------------------------------------------------
            # begin evolution
            for i in range(len(self.y)):
                y0 = self.y[i]
                print("y = " + str(y0))

                # write current N(r,Y) to file-------------------------------------------
                # self.nr = self.n_(self.xlr_)

                for j in range(len(self.r_)):
                    print("r=" + str(self.r_[j]) + ", N(r)=" + str(self.n_[j]))
                    writer.writerow([y0, self.r_[j], self.n_[j]])
                #------------------------------------------------------------------------

                # calculate correction and update N(r,Y) to next step in rapidity

                xk = []
                with Pool(processes=4) as pool:
                    xk = pool.map(self.evolve, self.xlr_, chunksize=100)

                self.n_ = [self.n_[j] + xk[j] for j in range(len(self.n_))]

                self.n_ = medfilt(self.n_)
                self.n_interpolated = self.filtered_nfunc(self.xlr_, self.n_)
                self.n_ = np.vectorize(self.n_interpolated)(self.xlr_)

                for i in range(len(self.n_)):
                    if self.n_[i] < 0.:
                        self.n_[i] = np.round(0.0, 2)
                    if self.n_[i] > 0.9999:
                        self.n_[i] = np.round(1.0, 2)


if __name__ == "__main__":
    t = Solve()
    t.solve()
