# from FBT import FBT
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.integrate as intg
import time
import cmath


class N:

    def __init__(self):
        self.n_ = 400
        self.x0 = 0.02
        self.xr1 = np.log(3.e-6)
        # self.xr1 = 0.0
        self.xr2 = np.log(60.e0)  # limit of integration in fourier transform calculation
        tol = 1.e-8
        self.width = 100
        self.pointsy = 600
        self.pointsr = 600


        # read results.csv file from BK solution to pandas dataframe
        self.df = pd.read_csv("results.csv", sep="\t") # results csv has frozen alhpa = 0.7
        self.df.columns = ['y', 'vr', 'vfr']
        # self.df = self.df.drop(self.df[self.df.kuta == "kuta"].index)

        # converting dataframe element types
        self.df["y"] = (self.df["y"].astype('float32')).round(decimals=1)
        self.df["vr"] = self.df["vr"].astype('float64')
        self.df["vfr"] = self.df["vfr"].astype('float64')


        self.r = np.concatenate(np.array(self.df.loc[self.df['y'] == 0.][['vr']]))  # r values over which N(r,Y) are evaluated
        self.y = np.unique(np.array(self.df[['y']]))

        self.z = [self.df.loc[(self.df['y'] == yy) & ((self.df['vr'] < (rr + tol)) & (self.df['vr'] > (rr - tol)))][['vfr']].iloc[0]['vfr'] for rr in self.r for yy in self.y]
        self.f = interp.interp2d(self.r, self.y, self.z, kind='cubic')

        '''print(self.f)
        print(type(self.f))
        print("grid created, interpolated")'''

    # finds location of val in grid in the sense that if val is between two elements in grid, find_index will return the index of the lower element
    def find_index(self, val, grid):
        index = 0

        cont = True
        i = 0
        while cont:
            if val >= grid[i]:
                index = i
                cont = False

            i += 1

            if i >= len(grid):
                cont = False

        return index

    # given any r, Y within bounds of self.r and self.y, master returns interpolated value of N(r,Y)
    def master(self, r, y):
        n = self.f(r, y)
        return n

    def master_adj(self, r, y):
        return 2 * self.master(r, y) - np.power(self.master(r, y), 2)

    def udg_f(self, x, k):
        t1 = time.time()
        y_ = np.log(self.x0 / x)
        integrand = lambda r_: (1 - self.master(r_, y_)) * self.bessel(k * r_, 0)
        a = 2 * np.pi * intg.quad(integrand, self.r[0], self.r[len(self.r)-1], epsabs=1.e-4)[0]
        t2 = time.time()
        print("udg_f took " + str(t2-t1) + " seconds")
        return a

    def udg_a(self, x, k):
        t1 = time.time()
        y_ = np.log(self.x0 / x)
        integrand = lambda r_: (1 - self.master_adj(r_, y_)) * self.bessel(k * r_, 0)
        a = 2 * np.pi * intg.quad(integrand, self.r[0], self.r[len(self.r)-1], epsabs=1.e-4)[0]
        t2 = time.time()
        print("udg_a took " + str(t2-t1) + " seconds")
        return a

    def bessel(self, x, alpha):
        # print("entering bessel")
        f = lambda t: np.cos(alpha * t - x * np.sin(t))
        g = (1 / np.pi) * intg.quad(f, 0, np.pi, epsabs=1.e-3)[0]
        # print("exiting bessel")
        return g

# end of class


if __name__ == "__main__":
    n = N()
    print("starting test")
    print(n.udg_a(0.001161195799828, 2.0959615528))