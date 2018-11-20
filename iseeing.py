#!/miniconda3/bin/python

import numpy as np
import math
import logging
import imageio


class ISeeing:
    def __init__(self):
        # user defined boundary and initial conditions
        self.T_start = np.float64(0.0)                      # starting temperature
        self.T_walls = np.float64(0.0)                      # temperature at the domain boundaries
        self.n = np.float64(300.0)                          # number of grid points in both x and y directions
        self.p_source = np.array([4.5, 4.5])                # x and y coords of source of p
        self.rad_source = np.float64(0.08)                  # radius of source
        self.eps_source = np.float64(0.004)                 # epsilon constant for Planck-taper window function
        # computational domain and grid
        self.X = np.float64(9.0)                            # extent of domain
        self.Y = np.float64(9.0)                            # extent of domain
        self.dx = self.X/self.n
        self.dy = self.Y / self.n
        # set-up of physical modelling
        self.maxt = np.float64(1.0)                         # maximum time
        self.dt = np.float64(2E-4)                          # time-step
        self.t = np.float64(0.0)
        self.alpha = np.float64(0.9)
        self.gamma = np.float64(10.0)
        self.T_e = np.float64(1.0)
        self.j = np.float64(6.0)
        self.theta_0 = np.float64(np.pi/2.0)
        self.eps_bar = np.float64(0.01)
        self.tau = np.float64(0.0003)
        self.a = np.float64(1.0)
        self.K = np.float64(1.8)
        self.delta = np.float64(0.02)
        # calculation variables
        self.T = np.full((int(self.n), int(self.n)), self.T_start) # initial temperature
        self.p = np.full((int(self.n), int(self.n)), 0.0)
        self.pg = np.full((2, int(self.n), int(self.n)), 0.0) # solid phase gradient
        self.pl = np.full((int(self.n), int(self.n)), 0.0)  # solid phase laplacian
        self.Tl = np.full((int(self.n), int(self.n)), 0.0) # temperature laplacian
        self.eps = np.full((int(self.n), int(self.n)), 0.0)  # epsilon
        self.eps_deps_dtheta_dp_dy = np.full((int(self.n), int(self.n)), 0.0)
        self.eps_deps_dtheta_dp_dx = np.full((int(self.n), int(self.n)), 0.0)

    def initialize_T(self):
        rows = self.T.shape[0]
        cols = self.T.shape[1]
        for idx, T_i in np.ndenumerate(self.T):
            i = idx[0]
            j = idx[1]
            if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                self.T[i,j] = self.T_walls

    def initialize_p(self):
        for idx, p_i in np.ndenumerate(self.p):
            i = idx[0]
            j = idx[1]
            x = self.X/self.n*i
            y = self.Y/self.n*j
            r = np.sqrt(pow((x - self.p_source[0]), 2.0) + pow((y - self.p_source[1]), 2.0))

            if r >= self.rad_source - 1E-8:
                self.p[idx] = 0.0
            elif r <= self.rad_source*(1-2*self.eps_source):
                self.p[idx] = 1.0
            else:
                r_1 = self.rad_source*(1-2*self.eps_source)
                z = (r_1 - self.rad_source)/(r - r_1) + (r_1 - self.rad_source)/(r - self.rad_source)
                self.p[idx] = 1/(1 + math.exp(z))

        # plot initial solid phase concentration
        imageio.imwrite('00000.png', self.p)

    def grads_update(self):
        rows = self.p.shape[0]
        cols = self.p.shape[1]
        for i in range(0, rows-1):
            for j in range(0, cols-1):
                # periodic boundary conditions
                self.pg[0, i, j] = (self.p[i, j+1] - self.p[i, j-1]) / self.dx * 0.5
                self.pg[1, i, j] = (self.p[i+1, j] - self.p[i-1, j]) / self.dy * 0.5

                self.Tl[i,j] = (2.0 * (self.T[i+1,j] + self.T[i-1,j] + self.T[i,j+1] + self.T[i,j-1])
                                + self.T[i+1,j+1] + self.T[i-1,j-1] + self.T[i-1,j+1] + self.T[i+1,j-1]
                                - 12.0 * self.T[i,j])/(3.0*self.dx*self.dy)

                self.pl[i,j] = (2.0 * (self.p[i+1,j] + self.p[i-1,j] + self.p[i,j+1] + self.p[i,j-1])
                                + self.p[i+1,j+1] + self.p[i-1,j-1] + self.p[i-1,j+1] + self.p[i+1,j-1]
                                - 12.0 * self.p[i,j])/(3.0*self.dx*self.dy)

                pg_x = self.pg[0, i, j]
                pg_y = self.pg[1, i, j]
                theta = np.arctan2(pg_x, pg_y)

                self.eps[i,j] = self.eps_bar*(1.0 + self.delta*np.cos(self.j*(-self.theta_0 + theta)))

    def solution_update(self):
        rows = self.p.shape[0]
        cols = self.p.shape[1]

        for i in range(0, rows-1):
            for j in range(0, cols-1):
                # periodic boundary conditions
                if not (i == 0 or j == 0 or i == rows-1 or j == cols-1):
                    m = self.alpha/np.pi*np.arctan(self.gamma*(self.T_e-self.T[i,j]))
                    p_term = self.p[i,j]*(1-self.p[i,j])*(self.p[i,j]-0.5+m)
                    eps_grad_term = -(self.eps_deps_dtheta_dp_dy[i,j+1]-self.eps_deps_dtheta_dp_dy[i,j-1])/2.0/self.dx\
                                    +((self.eps_deps_dtheta_dp_dx[i+1,j]-self.eps_deps_dtheta_dp_dx[i-1,j])/2.0/self.dy)
                    p_lapl_term = self.pl[i,j]*np.power(self.eps[i,j],2.0)
                    grad_eps_x = (np.power(self.eps[i,j+1],2.0)-np.power(self.eps[i,j-1],2.0))*0.5/self.dx
                    grad_eps_y = (np.power(self.eps[i,j+1],2.0)-np.power(self.eps[i,j-1],2.0))*0.5/self.dy
                    p_convective_term = grad_eps_x*self.pg[0,i,j] + grad_eps_y*self.pg[1,i,j]

                    dp_dt = 1/self.tau*(p_lapl_term + p_convective_term + eps_grad_term + p_term)
                    self.p[i, j] = self.p[i, j] + dp_dt*self.dt

                    dT_dt = np.power(self.a,2.0)*self.Tl[i,j] + self.K*dp_dt
                    self.T[i, j] = self.T[i, j] + dT_dt * self.dt

    def Tw_update(self):
        self.T_walls = min(0.0, 1.0-self.t/2.0)
        self.initialize_T()

    def run(self):
        print("Start iSeeing")
        self.initialize_p()
        self.initialize_T()
        i = 10000
        while self.t < self.maxt:
            # self.Tw_update()
            self.grads_update()
            self.solution_update()
            print("t = ", self.t)
            # save solution every 20 time-steps
            if i%20 == 0:
                imageio.imwrite(str(i) + '.png', self.p)
            self.t += self.dt
            i += 1

        print("End")


def main():
    logging.basicConfig(level=logging.DEBUG)

    icing = ISeeing()
    try:
        icing.run()
    except RuntimeError as e:
            logging.error(str(e))


if __name__ == '__main__':
    main()