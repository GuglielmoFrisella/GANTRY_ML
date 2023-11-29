"""
ENVIRONMENT
R. Ramjiawan
Oct 2020
Environment for optimiser
"""

import get_beam_size
import numpy as np
import gym
from cpymad.madx import Madx
import matplotlib.pyplot as plt
import time
import pickle


class kOptEnv(gym.Env):

    def __init__(self, solver, n_particles, _n_iter, init_dist, x, thin):
        self.rew = 10 ** 50  # Must be higher than initial cost function - there is definitely a better way to do this
        self.counter = 0
        self.solver = solver
        self.x = x
        self.x_all = []
        self.num_q = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.num_s = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.num_o = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.num_a = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.dof = self.num_q + self.num_s + self.num_o + self.num_a
        # Store actions, beam size, loss and fraction for every iteration
        self.x_best = np.zeros([1, self.dof])
        self.output_all = []
        # Vector to normalise actions
        self.norm_vect = [y['norm'] for y in x.values()]
        # Number of particles to track
        self.n_particles = n_particles
        # Max number of iterations
        self._n_iter = _n_iter

        # Spawn MAD-X process
        self.thin = thin
        self.madx = Madx(stdout=False)
        self.madx.call(file='gantry_2.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        if thin:
            print("making thin")
            self.madx.select(FLAG='makethin', THICK=True)
            self.madx.makethin(SEQUENCE='GANTRY', STYLE='teapot')
        self.madx.use(sequence='GANTRY')
        self.init_dist = init_dist
        params = {'axes.labelsize': 26,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 26,
                  'legend.fontsize': 26,  # was 10
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25,
                  'axes.linewidth': 1.5,
                  'lines.linewidth': 3,
                  'text.usetex': True,
                  'font.family': 'serif'
                  }
        plt.rcParams.update(params)

    def step(self, x_nor):
        self.counter = self.counter + 1
        print("iter = " + str(self.counter))
        # if self.counter > 100:
        #     self.n_particles = 1000

        x_unnor = self.unnorm_data(x_nor)  # normalise actions
        if np.size(self.x_all) == 0:
            self.x_all = x_nor
        else:
            self.x_all = np.vstack((self.x_all, x_nor))

        c1 = get_beam_size.getBeamSize(x_unnor, self.n_particles, self.madx, self.init_dist, self.x)
        a = (c1.get_beam_size())
        
        # If MAD-X has failed re-spawn process
        if a[4]:
            print("reset")
            self.reset(thin=self.thin)    # potential for problems
        # print("KL = " + str(round(a[7], 4)))
        dim=10**3
        print('At starting point: SIG_x =' + str(round(a[10]*dim, 4)) + ' mm' + ', SIG_y=' + str(round(a[11]*dim, 4))+ ' mm')
        print('At isocenter: SIG_x =' + str(round(a[0]*dim, 4)) + ' mm' + ', SIG_y=' + str(round(a[1]*dim, 4)) + ' mm' + ', SIG_z=' + str(round(a[2]*dim, 2)) + ' mm')
        print("LOSS = " + str(a[3]*100) + "%")
        
        MF=1 #magnification factor

        self.parameters = [(a[0]),  # beam size x matched
                           (a[1]),  # beam size y macthed
                           a[20], # beta x at isocenter
                           a[21], # beta y at isocenter
                           a[8],  # dx
                           a[9],  # dx2
                           a[5],  # alfax
                           a[6],  # alfay
                           a[3]   # losses percentage
                           ]
        self.targets = [MF*8*10**(-3),  # beam size x
                        MF*8*10**(-3),  # beam size y
                        (MF**2)*8.24397,   # betx 
                        (MF**2)*8.24397,   # bety
                        0,  # dx
                        0,  # dx2
                        0,  # alfax
                        0,  # alfay
                        0
                        ]
        self.weights = [10,  # beam size x
                        10,  # beam size y
                        100,  # kurt x
                        100,  # kurt y
                        #
                        # 0,   # x = y
                        100,  # dx
                        100,  # dx2
                        100,  # alfax
                        100,  # alfay
                        100   # Number of Losses (multiplied for 100)
                        ]
        # y_raw = np.tanh(np.multiply(np.array(self.parameters) - np.array(self.targets), self.weights)/1000)
        y_raw = np.multiply(np.array(self.parameters) - np.array(self.targets), self.weights)
        print(y_raw)
        self.madx.input("delete, table = trackone;")
        self.madx.input("delete, table = trackloss;")
        self.madx.input("delete, table = tracksumm;")

        print("ymse = " + str(self._mse(y_raw)))
        output = self._mse(y_raw)
        if output < self.rew:
            self.rew = output
            self.x_best = x_unnor
            if np.size(self.output_all) == 0:
                self.output_all = [output, a[0], a[1]]
            else:
                self.output_all = np.vstack((self.output_all, [output, a[0], a[1]]))
        else:
            if len(np.shape(self.output_all)) == 1:
                self.output_all = np.vstack((self.output_all, self.output_all))
            else:
                self.output_all = np.vstack((self.output_all, self.output_all[-1, :]))
        print("best = " + str(self.rew))
        if self.counter % 1000 == 0:
            print('reset madx')
            self.kill_reset(thin=self.thin)
        print(type(output))
        return output

    
    def _mse(self, values):
        return np.sum(values ** 2) / len(values)

    def norm_data(self, x_data):
        """
        Normalise data
        """
        print(x_data)
        x_norm = np.divide(x_data, self.norm_vect)
        return x_norm

    def unnorm_data(self, x_norm):
        """
        Unnormalise data
        """
        x_data = np.multiply(x_norm, self.norm_vect)
        return x_data

    def reset(self, thin):
        """
         If MAD-X fails, re-spawn process
         """
        self.madx = Madx(stdout=False)
        self.madx.call(file='gantry_2.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        if thin:
            self.madx.select(FLAG='makethin', THICK=True)
            self.madx.makethin(SEQUENCE='GANTRY', STYLE='teapot')
        self.madx.use(sequence='GANTRY')
        self.madx.twiss(BETX=5, ALFX=0, DX=0, DPX=0, BETY=5, ALFY=0, DY=0, dpy=0)

    def kill_reset(self, thin):
        """
         Kill madx, re-spawn process
         """
        self.madx.quit()
        del self.madx
        time.sleep(1)
        self.madx = Madx(stdout=False)
        self.madx.call(file='gantry_2.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        if thin:
            self.madx.select(FLAG='makethin', THICK=True)
            self.madx.makethin(SEQUENCE='GANTRY', STYLE='teapot')
        self.madx.use(sequence='GANTRY')
        self.madx.twiss(BETX=5, ALFX=0, DX=0, DPX=0, BETY=5, ALFY=0, DY=0, dpy=0)

    def render(self, mode='human'):
        pass

    def seed(self, seed):
        np.random.seed(seed)
