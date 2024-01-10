import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


class Plot:
    def __init__(self, madx, x_best, x, init_dist, output_all, x_all):
        self.madx = madx
        self.output_all = output_all
        self.x_all = x_all
        self.x_best = x_best
        self.q = x_best
        self.ii = 0
        self.init_dist = init_dist
        x0 = self.init_dist[:, 0]
        px0 = self.init_dist[:, 3]
        y0 = self.init_dist[:, 1]
        py0 = self.init_dist[:, 4]

        self.name = [y['name'] for y in x.values()]
        self.num_q = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.num_s = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.num_o = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.num_a = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.dof = self.num_q + self.num_s + self.num_o + self.num_a

        temp = np.mean((x0 - np.mean(x0)) * (px0 - np.mean(px0)))
        self.emitx_before = np.sqrt(np.std(x0) ** 2 * np.std(px0) ** 2 - temp ** 2)
        temp = np.mean((y0 - np.mean(y0)) * (py0 - np.mean(py0)))
        self.emity_before = np.sqrt(np.std(y0) ** 2 * np.std(py0) ** 2 - temp ** 2)

        self.betx0 = np.divide(np.mean(np.multiply(x0, x0)), self.emitx_before)
        self.alfx0 = -np.divide(np.mean(np.multiply(x0, px0)), self.emitx_before)
        self.bety0 = np.divide(np.mean(np.multiply(y0, y0)), self.emity_before)
        self.alfy0 = -np.divide(np.mean(np.multiply(y0, py0)), self.emity_before)


    def plot1(self, *args):
        """
        Plot beam size and fraction vs. iteration
        Possibly not working
        """
        # self.output_all = args[0]
        fig = plt.figure(figsize=[20, 8])
        gs = fig.add_gridspec(7, 7)
        ax4 = plt.subplot(gs[0:3, 0:3])
        ax1= plt.subplot(gs[4:7, 0:3])
        ax3 = plt.subplot(gs[0:3, 4:7])
        ax2 = plt.subplot(gs[4:7, 4:7])
        label = "$\sigma_x$ [$\mu$m]"
        x = np.transpose(self.output_all)
        
        ax1.plot(x[1,:], linewidth=3.0, color="tab:blue")
        ax1.set_xlabel("Iteration", fontsize=34, usetex=True)
        ax1.set_ylabel(label, fontsize=38, usetex=True)
        # ax1.set_yscale('log')
        
        label = "$\sigma_y$ [$\mu$m]"
        ax2.plot(x[2, :], linewidth=3.0, color="tab:orange")
        ax2.set_xlabel("Iteration", fontsize=34, usetex=True)
        ax2.set_ylabel(label, fontsize=38, usetex=True)
        # ax2.set_yscale('log')

        label = "Objective"
        ax3.plot(x[0, :], linewidth=3.0, color="tab:green")
        ax3.set_xlabel("Iteration", fontsize=34, usetex=True)
        ax3.set_ylabel(label, fontsize=34, usetex=True)
        # ax3.set_yscale('log')

        x = self.x_all
        label = "Variables"
        ax4.plot(x, linewidth=3.0)
        ax4.set_xlabel("Iteration", fontsize=34, usetex=True)
        ax4.set_ylabel(label, fontsize=34, usetex=True)
        # ax4.set_yscale('log')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=28, pad=10)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2.5)
                ax.tick_params(width=2.5, direction='out')
        plt.show()

    def twiss(self, *args):
        """
        Twiss plots with synoptics
        """
        if args:
            for j in range(self.dof):
                self.madx.input(self.name[j] + "=" + str(args[0][j]) + ";")
                print(self.name[j][0] + self.name[j][-1] + "=" + str(args[0][j])+ ";")
        self.madx.use(sequence='GANTRY', range='#s/#e')
        self.madx.twiss(RMATRIX=True, BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0,file='twiss_one.txt')
        fig = plt.figure(figsize=(8,7))
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])

        self.madx.use(sequence='GANTRY')
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.select(flag='RMATRIX')
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 's', 'l','sigma_x', 'sigma_y', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'mux',
                                 'muy','RE56', 'RE16', 'TE166'])
        #self.madx.input(
         #   "sigma_x := 1000*(sqrt((table(twiss, betx)*6.81e-9) + (abs(table(twiss,dx))*0.002)*(abs(table(twiss,dx))*0.002)));")
        #self.madx.input(
          #  "sigma_y :=  1000*(sqrt((table(twiss, bety)*6.81e-9) + (abs(table(twiss,dy))*0.002)*(abs(table(twiss,dy))*0.002)));")
        twiss = self.madx.twiss(RMATRIX=True, BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0)
	
        for idx in range(np.size(twiss['l'])):
            if twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole' or twiss['keyword'][idx] == 'multipole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            # elif twiss['keyword'][idx] == 'marker':
            #     _ = ax1.add_patch(
            #         matplotlib.patches.Rectangle(
            #             (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #             facecolor='m', edgecolor='m'))
            # elif twiss['keyword'][idx] == 'kicker':
            #     _ = ax1.add_patch(
            #         matplotlib.patches.Rectangle(
            #             (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
            #             facecolor='c', edgecolor='c'))
        self.madx.select(flag='ptc_twiss',
                         column=['name', 'keyword', 's', 'l','sigma_x','sigma_y', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])

        # self.madx.select(flag='interpolate', step=0.05)
        self.madx.ptc_twiss(BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0,
                            file='ptc_twiss2.out')

        ptc_twiss = np.genfromtxt('ptc_twiss2.out', skip_header=90)


        twiss = self.madx.twiss(BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0)

        # ax.plot(twiss['s'], np.sqrt(twiss['betx'] * self.emitx_before + (0.002 * twiss['dx']) ** 2), 'k', label=r"$\beta_x$")
        # ax.plot(twiss['s'], np.sqrt(twiss['bety'] * self.emity_before + (0.002 * twiss['dy']) ** 2), 'r', label=r"$\beta_y$")
        ln = ax.plot(twiss['s'], twiss['betx'] , 'k',
                label=r"$\beta_x$")
        ln2 = ax.plot(twiss['s'], twiss['bety'] , 'r', label=r"$\beta_y$")
        
        ax2 = ax.twinx()
        ln3 = ax2.plot(twiss['s'], twiss['Dx'], 'g',  label=r"$D_x$")
        ln4 = ax2.plot(twiss['s'], twiss['Dy'], 'b', label=r"$D_y$")
        lns = ln + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=32)
        ax.set_xlabel("s [m]", fontsize=36, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta_x, \beta_y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$ [m]", fontsize=38, usetex=True, labelpad = 5)
        ax.tick_params(labelsize=32)
        ax2.tick_params(labelsize=32)
        # ax.set_ylim = (-1, 1)
        spine_color = 'gray'
        

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set(xlim=(0, twiss['s'][-1]))
        # ax.set(ylim=(-0.001, 0.025))
        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_position('center')
        ax1.xaxis.set_ticks([])

        for ax0 in [ax]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
                # ax0.tick_params('off')
                
                
        self.madx.use(sequence='GANTRY', range='#s/#e')
        self.madx.select(flag='twiss',clear=True)
        self.madx.select(flag='twiss',column=['name', 'keyword', 's', 'l','sigma_x', 'sigma_y', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy'])
        self.madx.twiss(BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0, file='twiss_one.txt')
                
                
                
        plt.show()
  

    def ptc_twiss_2(self):
        """
        Twiss plots with synoptics
        """

        # fig, ax = plt.subplots()
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])

        self.madx.use(sequence='GANTRY')
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])
        twiss = self.madx.twiss(BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0)

        for idx in range(np.size(twiss['l'])):
            if twiss['keyword'][idx] == 'quadrupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], np.sign(twiss['k1l'][idx]),
                        facecolor='k', edgecolor='k'))
            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9 * np.sign(twiss['k2l'][idx]),
                        facecolor='b', edgecolor='b'))
            elif twiss['keyword'][idx] == 'octupole':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8 * np.sign(twiss['k3l'][idx]),
                        facecolor='r', edgecolor='r'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'monitor':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='m', edgecolor='m'))
            elif twiss['keyword'][idx] == 'kicker':
                _ = ax1.add_patch(
                    matplotlib.patches.Rectangle(
                        (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
                        facecolor='c', edgecolor='c'))

        self.madx.select(flag='interpolate', step=0.05)
        self.madx.ptc_twiss(RMATRIX=True, BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0,
                            file='ptc_twiss2.out')
        ptc_twiss = np.genfromtxt('ptc_twiss2.out', skip_header=90)

        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])

        twiss = self.madx.twiss(BETX=8.24397, ALFX=0, DX=0, DPX=0, BETY=8.24397, ALFY=0, DY=0, dpy=0)
        ax.plot(twiss['s'], np.sqrt(twiss['betx'] * self.emitx_before + (0.001 * twiss['dx']) ** 2), 'k')
        ax.plot(twiss['s'], np.sqrt(twiss['bety'] * self.emity_before + (0.001 * twiss['dy']) ** 2), 'r')

        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g')
        ax2.plot(twiss['s'], twiss['Dy'], 'b')
        ax.set_xlabel("s [m]", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\sigma_x, \sigma_y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$ [m]", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)
        # ax.set_ylim = (-1, 1)
        spine_color = 'gray'

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set(xlim=(0, twiss['s'][-1]))
        plt.gcf().subplots_adjust(bottom=0.15)
        ax1.set(xlim=(0, twiss['s'][-1]), ylim=(-1, 1))
        ax1.axes.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position('center')

        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
                ax0.tick_params(labelsize=28, pad=10)
        plt.show()

