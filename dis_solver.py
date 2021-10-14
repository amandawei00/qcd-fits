import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.special as spec
import csv
from bk_interpolate_interp2d import N

# Units chosen to be in GeV


class Solve:

    def __init__(self):
        self.sig = 1.  # constant
        self.alpha = 1./137  # FIND VALUE (EM coupling)
        self.e = 1.60217e-19  # Coulomb
        self.x0 = 0.01  # Bjorken-x, value of x at which evolution starts. (highest experimental value of x included in fit)
        self.q0 = 1.0  # GeV

        self.lamb = 0.241  # lambda_QCD (GeV)
        self.root_sNN = 319 # for data from 1993, s = 4 * Ep * Ee

        """ordering of flavors:
            1. up      -1. antiup
            2. down    -2. antidown
            3. charm   -3. anticharm
            4. strange -4. antistrange
            5. top     -5. antitop
            6. bottom  -6. antibottom  
        """

        self.f = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]  # CHECK VALUES (quark flavors, including charm with mass of 1.5 GeV)
        self.mf = [0.002, 0.0045, 1.270, 0.101, 172, 5., 0.002, 0.0045, 1.270, 0.101, 172, 5.] # * np.full(1, np.power(3.e8,2)) quark masses in GeV
        self.ef = [2/3, -1/3, 2/3, -1/3, 2/3, -1/3, -2/3, 1/3, -2/3, 1/3, -2/3, 1/3] # * np.full(1, self.e)  # CHECK VALUES quark charges

        self.n = N()  # bk interpolated

        # these variables are initialized here. to be assigned in self.rhs()
        self.x = 0.0
        self.qsq2 = 0.0
        self.y = 0.0  # rapidity

        self.r = 0.0  # upper bound on integral
        self.r0 = 0.0

    # (transverse) wave function for splitting of photon to quark-antiquark dipole
    def psi_t2(self, z, r):
        coeff = (6 * self.alpha)/(4 * np.power(np.pi, 2))
        sum = 0

        for i in range(len(self.f)):   # summing over all flavors
            eta2 = self.eta_squared(z, self.mf[i])
            eta = np.power(eta2, 0.5)
            k02 = np.power(spec.kn(0, eta * r), 2) # modified Bessel function of the second kind, 0th order (MacDonald's Function)
            k12 = np.power(spec.kn(1, eta * r), 2)  # MacDonald's Function first order

            t1 = (np.power(z, 2) + np.power(1-z, 2)) * eta2 * k12
            t2 = np.power(self.mf[i], 2) * k02

            sum += np.power(self.ef[i], 2) * (t1 + t2)

        return coeff * sum

    # (longitudinal) wave function for splitting of photon to quark-antiquark dipole
    def psi_l2(self, z, r):

        coeff = (6 * self.alpha) / (4 * np.power(np.pi, 2))
        sum = 0

        for i in range(len(self.f)):
            eta2 = self.eta_squared(z, self.mf[i])
            eta = np.power(eta2, 0.5)
            k02 = np.power(spec.kn(0, eta * r), 2)
            a = 4 * self.qsq2 * np.power(z * (1 - z), 2) * k02 * np.power(self.ef[i], 2)
            sum += a
        return coeff * sum
        
    def eta_squared(self, z, m_f):
        # print("z = " + str(z) + ", qsq2 = " + str(qsq2) + ", m_f = " + str(m_f))
        return z * (1 - z) * self.qsq2 + np.power(m_f, 2)

    def inner_t(self, z):
        m = lambda r_: self.psi_t2(z, r_) * self.n.master(r_, self.y)
        u = intg.quad(m, 0.0, 1 / self.qsq2, epsabs=1.e-5)[0]
        return u

    def inner_l(self, z):
        m = lambda r_: self.psi_l2(z, r_) * self.n.master(r_, self.y)
        u = intg.quad(m, 0.0, 1 / self.qsq2, epsabs=1.e-5)[0]
        return u

    def rhs(self, x, qsq2, tl):  # returns F_T, F_L
        self.x = x
        self.qsq2 = qsq2
        self.r = 1/self.qsq2
        self.r0 = (1/self.q0) * np.power(self.x/self.x0, self.lamb/2)

        self.y = np.log(self.x0/self.x)
        # self.y = 0.71 # for FL calculations

        # s = np.power(318,2)
        # self.y = qsq2/(s*x)

        if tl == 'T':
            integral = intg.quad(self.inner_t, 0, 1, epsabs=1.e-5)[0]
        elif tl == 'L':
            integral = intg.quad(self.inner_l, 0, 1, epsabs=1.e-5)[0]

        return 2 * np.pi * self.sig*integral  # 2*np.pi comes from theta component in the inner double integral. since there is no dependence on theta, multiply it out

if __name__ == "__main__":
    alpha = 1/137. #?? HWERE DID HTIS COME FROM
    c = 2.568  # unit conversion factor

    # create new instance of solve class
    st = Solve("T")
    sl = Solve("L")

    x = [2.09E-04, 2.37E-04, 2.68E-04, 3.28E-04, 5.00E-04, 8.00E-04, 1.30E-03, 2.00E-03, 3.20E-03, 5.00E-03, 8.00E-03, 2.00E-02]
    qsq2 = 18. # GeV^2
    s = np.power(318, 2)

    for i in range(len(x)):
        sig_t = st.rhs(x[i], qsq2)
        sig_l = sl.rhs(x[i], qsq2)

        f2 = (qsq2/(4 * np.power(np.pi, 2) * alpha)) * (sig_t + sig_l)
        # print("x = " + str(x[i]) + ", f2 = " + str(f2 * 2.568))

        fl = (qsq2/(4 * np.power(np.pi, 2) * alpha)) * sig_l
        # print("x = " + str(x[i]) + ", qsq2 = " + str(qsq2) + ", fl = " + str(fl))

        y = np.log(0.01/x[i])
        yp = 1 + np.power(1-y, 2)

        sig = f2 - (np.power(y, 2)/yp) * fl
        print(str(sig))