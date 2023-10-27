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

    def __init__(self, solver, n_particles, _n_iter, init_dist, foil_w, x, thin):
        self.rew = 10 ** 50  # Must be higher than initial cost function - there is definitely a better way to do this
        self.counter = 0
        self.sigma_x = 1000
        self.sigma_y = 1000
        self.solver = solver
        self.x = x
        self.x_all = []
        self.num_q = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.num_s = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.num_o = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.num_a = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.dof = self.num_q + self.num_s + self.num_o + self.num_a
        self.foil_w = foil_w
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

        c1 = get_beam_size.getBeamSize(x_unnor, self.n_particles, self.madx, self.init_dist, self.foil_w, self.x)
        if self.foil_w == 0:
            a = (c1.get_beam_size())
        else:
            a = (c1.get_beam_size_foil())

        # If MAD-X has failed re-spawn process
        if a[4]:
            print("reset")
            self.reset(thin=self.thin)    # potential for problems
        # print("KL = " + str(round(a[7], 4)))
        print('SIG_x =' + str(round(a[0], 4)) + ', SIG_y=' + str(round(a[1], 4)) + ', SIG_z=' + str(round(a[2], 2)))
        # print('kurt_x =' + str(round(a[18], 4)) + ', kurt_y=' + str(round(a[19], 4)))
        print('NOM SIG_x =' + str(round(a[10], 4)) + ', SIG_y=' + str(round(a[11], 4)))
        print("LOSS = " + str(a[3]))

        self.parameters = [(a[0]-a[10]) * (a[3] + 1),  # beam size x matched
                           (a[1]-a[11]) * (a[3] + 1),  # beam size y macthed
                           a[0]+a[1], # beam size
                           a[1]-a[0], # beam size
                           a[8],  # dx
                           a[9],  # dx2
                           a[5],  # alfax
                           a[6],  # alfay
                           a[3]
                           ]
        self.targets = [0,  # beam size x
                        0,  # beam size y
                        0,   # x = y
                        0,   # x = y
                        0,  # dx
                        0,  # dx2
                        0,  # alfax
                        0,  # alfay
                        0
                        ]
        self.weights = [100,  # beam size x
                        100,  # beam size y
                        5,  # kurt x
                        50,  # kurt y
                        #
                        # 0,   # x = y
                        1,  # dx
                        1,  # dx2
                        100000,  # alfax
                        100000,  # alfay
                        10
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
        return output

    def step_MO(self, x_nor):
        self.counter = self.counter + 1
        print("iter = " + str(self.counter))

        x_unnor = self.unnorm_data(x_nor)  # normalise actions
        if np.size(self.x_all) == 0:
            self.x_all = x_nor
        else:
            self.x_all = np.vstack((self.x_all, x_nor))

        c1 = get_beam_size.getBeamSize(x_unnor, self.n_particles, self.madx, self.init_dist, self.foil_w, self.x)
        if self.foil_w == 0:
            a = (c1.get_beam_size_twiss())
        else:
            a = (c1.get_beam_size_foil())

        # If MAD-X has failed re-spawn process
        if a[4]:
            print("reset")
            self.reset(thin=self.thin)

        print('SIG_x =' + str(round(a[0], 4)) + ', SIG_y=' + str(round(a[1], 4)) + ', SIG_z=' + str(round(a[2], 2)))
        # print('WAIST SIG_x =' + str(round(a[-2], 4)) + ', SIG_y=' + str(round(a[-1], 4)))
        # print('NOM SIG_x =' + str(round(a[10], 4)) + ', SIG_y=' + str(round(a[11], 4)))
        print('FRAC_x =' + str(round(a[18], 4)) + ', FRAC_y=' + str(round(a[19], 4)))
        print('dx =' + str(round(a[8], 4)) + ', dx2=' + str(round(a[9], 4)))
        print('ax =' + str(round(a[5], 4)) + ', ay=' + str(round(a[6], 4)))
        print("LOSS = " + str(a[3]))

        self.parameters = [abs(a[0]-5.76) * (a[3] + 1) + 1000 * abs(a[8]) ,  # beam size x
                           abs(a[1]-5.76) * (a[3] + 1) + 1000 * abs(a[9]) ,  # beam size y
                           # a[8], # dx
                           # a[9] # dx2
                           # a[3] # loss
                           # a[5], # alfay
                           # a[6]
                           ]
        self.targets = [0,  # beam size x
                        0,  # beam size y
                        # 0,  # dx
                        # 0  # dx2
                        # 0  # alfax
                        # 0,  # alfay
                        # 0
                        ]
        self.weights = [1,  # beam size x
                        1,  # beam size y
                        # 0.01,  # dx
                        # 0.01  # dx2
                        # 1000  # loss
                        ]
        y_raw = np.log(np.multiply(np.array(self.parameters) - np.array(self.targets), self.weights) + 1)
        self.madx.input("delete, table = trackone;")
        self.madx.input("delete, table = trackloss;")
        self.madx.input("delete, table = tracksumm;")

        print("ymse = " + str(self._mse(y_raw)))
        output = self._mse(y_raw)
        if output < self.rew:
            self.rew = output
            self.x_best = x_unnor
            if np.size(self.output_all) == 0:
                self.output_all = [output, a[0], a[1], a[0], a[1], self._mse(y_raw), self.x_best]
            else:
                self.output_all = np.vstack((self.output_all, [output, a[0], a[1], a[0], a[1],  self._mse(y_raw), self.x_best]))
        else:
            if len(np.shape(self.output_all)) == 1:
                # self.output_all = np.vstack((self.output_all, [self.output_all, a[0], a[1], self.x_best]))
                pass
            else:
                temp = np.append(np.hstack((self.output_all[-1, :-4], a[0], a[1], self._mse(y_raw))), 0)
                temp[-1] = self.x_best
                self.output_all = np.vstack((self.output_all, temp))
        print("best = " + str(self.rew))
        if self.counter % 1000 == 0:
            print('automatic reset madx')
            self.kill_reset(thin=self.thin)
        if self.counter % 100000 == 0:
            pickle.dump(self.output_all, open("frac_output_all_" + str(time.time()) + ".p", "wb"))
        return y_raw, self._mse(y_raw)

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
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        if thin:
            self.madx.select(FLAG='makethin', THICK=True)
            self.madx.makethin(SEQUENCE='TT43', STYLE='teapot')
        self.madx.use(sequence='TT43')
        self.madx.twiss(BETX=5, ALFX=0, DX=0, DPX=0, BETY=5, ALFY=0, DY=0, dpy=0)

    def kill_reset(self, thin):
        """
         Kill madx, re-spawn process
         """
        self.madx.quit()
        del self.madx
        time.sleep(1)
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        if thin:
            self.madx.select(FLAG='makethin', THICK=True)
            self.madx.makethin(SEQUENCE='TT43', STYLE='teapot')
        self.madx.use(sequence='TT43')
        self.madx.twiss(BETX=5, ALFX=0, DX=0, DPX=0, BETY=5, ALFY=0, DY=0, dpy=0)

    def render(self, mode='human'):
        pass

    def seed(self, seed):
        np.random.seed(seed)
