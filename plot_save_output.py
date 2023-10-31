import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


class Plot:
    def __init__(self, madx, x_best, x, init_dist, foil_w, output_all, x_all):
        self.madx = madx
        self.output_all = output_all
        self.x_all = x_all
        self.x_best = x_best
        self.q = x_best
        self.foil_width = 0e-6
        self.ii = 0
        self.init_dist = init_dist
        self.foil_w = foil_w
        x0 = self.init_dist[:, 0]
        px0 = self.init_dist[:, 3]
        y0 = self.init_dist[:, 1]
        py0 = self.init_dist[:, 4]

        self.foil_width = 0
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




        # x = self.sigmay_all
        # label = "$\sigma_y$"
        # ax2.plot(x[2:, ], linewidth=3.0, color=[1.00, 0.54, 0.00])
        # ax2.set_xlabel("Iteration", fontsize=34, usetex=True)
        # ax2.set_ylabel(label, fontsize=38, usetex=True)
        # ax2.annotate("%s = %.3f $\mu$m" % (label, x[-1]), (len(x), x[-1]), fontsize=38,
        #              xytext=(0.6 * len(x), x[-1] + 200))

        # x = np.sqrt(np.square(self.sigmay_all) + np.square(self.sigmax_all))
        # label = "$\sqrt{\sigma_x^2 + \sigma_y^2}$"
        # ax3.plot(x[2:, ], linewidth=3.0, color=[0.25, 0.80, 0.54])
        # ax3.set_xlabel("Iteration", fontsize=34, usetex=True)
        # ax3.set_ylabel(label, fontsize=35, usetex=True)
        # ax3.annotate("%s = %.3f $\mu$m" % (label, x[-1]), (len(x), x[-1]), fontsize=38,
        #              xytext=(0.25 * len(x), x[-1] + 500))
        #
        # x = self.fraction_all
        # label = "Fraction within 5 $\mu$m"
        # ax4.plot(x[2:, ], linewidth=3.0, color=[153 / 256, 102 / 256, 255 / 256])
        # ax4.set_xlabel("Iteration", fontsize=34, usetex=True)
        # ax4.set_ylabel(label, fontsize=35, usetex=True)
        # ax4.annotate("%s = %.3f " % (label, x[-1]), (len(x), x[-1]), fontsize=38,
        #              xytext=(0.25 * len(x), x[-1] - 20))


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

    def plot2(self):
        """
        Plot actions vs iteration
        Possibly not working
        """

        fig, ax = plt.subplots()
        ax.plot(self.y_all[1:, ], linewidth=3.0)
        ax.set_xlabel("Iteration", fontsize=34, usetex=True, labelpad=10)
        ax.set_ylabel("Normalised magnet strengths", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=28)
        plt.legend(['quad 1', 'quad 2', 'quad 3', 'quad 4', 'quad 5', 'quad 6', 'sextupole 1', 'sextupole 2',
                    'sextupole 3', 'sextupole 4', 'sextupole 5', 'sextupole 6', 'dist1', 'dist2', 'dist3', 'dist4',
                    'dist5', 'dist6'],
                   fontsize=30, edgecolor='white', loc='center left', bbox_to_anchor=(0.9, 0.5))
        # ax.set_ylim = (-1, 1)

        spine_color = 'gray'

        ax.tick_params(labelsize=28, pad=3)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)
        plt.show()

    def twiss(self, *args):
        """
        Twiss plots with synoptics
        """
        if args:
            for j in range(self.dof):
                self.madx.input(self.name[j] + "=" + str(args[0][j]) + ";")
                print(self.name[j][0] + self.name[j][-1] + "=" + str(args[0][j])+ ";")
        self.madx.use(sequence='TT43', range='#s/#e')
        self.madx.twiss(RMATRIX=True, BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        fig = plt.figure(figsize=(8,7), constrained_layout=True)
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])

        # self.madx.use(sequence='TT43')
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.select(flag='RMATRIX')
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 's', 'l','sigma_x', 'sigma_y', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'mux',
                                 'muy','RE56', 'RE16', 'TE166'])
        self.madx.input(
            "sigma_x := 1000*(sqrt((table(twiss, betx)*6.81e-9) + (abs(table(twiss,dx))*0.002)*(abs(table(twiss,dx))*0.002)));")
        self.madx.input(
            "sigma_y :=  1000*(sqrt((table(twiss, bety)*6.81e-9) + (abs(table(twiss,dy))*0.002)*(abs(table(twiss,dy))*0.002)));")
        twiss = self.madx.twiss(RMATRIX=True, BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * self.emitx_before + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * self.emitx_before + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=1.5)
            elif twiss['keyword'][idx] == 'quadrupole':
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
        self.madx.ptc_twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                            file='ptc_twiss2.out')

        ptc_twiss = np.genfromtxt('ptc_twiss2.out', skip_header=90)


        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        # ax.plot(twiss['s'], np.sqrt(twiss['betx'] * self.emitx_before + (0.002 * twiss['dx']) ** 2), 'k', label=r"$\beta_x$")
        # ax.plot(twiss['s'], np.sqrt(twiss['bety'] * self.emity_before + (0.002 * twiss['dy']) ** 2), 'r', label=r"$\beta_y$")
        ln = ax.plot(twiss['s'], twiss['betx'] , 'k',
                label=r"$\beta_x$")
        ln2 = ax.plot(twiss['s'], twiss['bety'] , 'r', label=r"$\beta_y$")

        # ax.legend(loc="upper left", fontsize=32)
        # ax.plot(twiss['s'], np.divide(twiss['dx'], np.sqrt(twiss['betx'] )), 'm')
        # ax.plot(twiss['s'], np.divide(twiss['dy'], np.sqrt(twiss['bety'] )), 'm')
        # ax.plot(twiss['s'], np.divide(np.sqrt(twiss['bety'] * self.emity_before + (0.002 * twiss['dy']) ** 2), np.sqrt(twiss['betx'] * self.emitx_before + (0.002 * twiss['dx']) ** 2)), 'm--')
        # ax.plot(twiss['s'], np.divide(np.sqrt(twiss['betx'] * self.emity_before + (0.002 * twiss['dx']) ** 2), np.sqrt(twiss['bety'] * self.emitx_before + (0.002 * twiss['dy']) ** 2)), 'g--')
        # ax.plot(twiss['s'], 4*np.sqrt(twiss['betx'] * 2*6.8e-9 + (0.002 * twiss['dx']) ** 2), 'k--')
        # ax.plot(twiss['s'], 4*np.sqrt(twiss['bety'] * 2*6.8e-9 + (0.002 * twiss['dy']) ** 2), 'r--')
        # ax.plot(ptc_twiss[:, 2], np.sqrt(ptc_twiss[:, 3] * 6.8e-9 + (0.002 * ptc_twiss[:, 15]) ** 2), 'k')
        # ax.plot(ptc_twiss[:, 2], np.sqrt(ptc_twiss[:, 6] * 6.8e-9 + (0.002 * ptc_twiss[:, 17]) ** 2), 'r')

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * self.emity_before + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * self.emity_before + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=2.5)

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
        #plt.gcf().subplots_adjust(bottom=0.15)
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

        self.madx.use(sequence='TT43')
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=1.5)
            elif twiss['keyword'][idx] == 'quadrupole':
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
        self.madx.ptc_twiss(RMATRIX=True, BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                            file='ptc_twiss2.out')
        ptc_twiss = np.genfromtxt('ptc_twiss2.out', skip_header=90)

        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])

        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        # ax.plot(twiss['s'], np.sqrt(twiss['betx'] * 6.8e-9 + (0.002 * twiss['dx']) ** 2), 'k')
        # ax.plot(twiss['s'], np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2), 'r')
        ax.plot(ptc_twiss[:, 2], np.sqrt(ptc_twiss[:, 3] * 6.8e-9 + (0.002 * ptc_twiss[:, 15]) ** 2), 'k')
        ax.plot(ptc_twiss[:, 2], np.sqrt(ptc_twiss[:, 6] * 6.8e-9 + (0.002 * ptc_twiss[:, 17]) ** 2), 'r')


        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=2.5)

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

    def phase(self):
        """
        Twiss plots with synoptics
        """

        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[0])

        self.madx.use(sequence='TT43')

        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                file="twiss_opt_out.out")
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                file="twiss_opt_out.out")

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * 6.2e-9 + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * 6.2e-9 + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=1.5)
            elif twiss['keyword'][idx] == 'quadrupole':
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
        self.madx.select(flag='twiss',
                         column=['name', 'keyword', 'k1l', 'k2l', 'k3l', 's', 'l', 'betx', 'bety', 'dx', 'dy', 'mux',
                                 'muy'])
        self.madx.select(flag='interpolate', step=0.01)
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        ax.plot(twiss['s'], twiss['mux'][-1] * 2 - twiss['mux'] * 2, 'k')
        ax.plot(twiss['s'], twiss['muy'][-1] * 2 - twiss['muy'] * 2, 'r')

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:0':
                ax.plot([twiss['s'][idx], twiss['s'][idx]],
                        [np.min(np.sqrt(twiss['bety'] * 6.2e-9 + (0.002 * twiss['dy']) ** 2)),
                         np.max(np.sqrt(twiss['bety'] * 6.2e-9 + (0.002 * twiss['dy']) ** 2))], color='m',
                        linestyle='--', linewidth=2.5)

        ax.set_ylabel("$\mu_x, \mu_y [\pi]$ ", fontsize=38, usetex=True, labelpad=10)
        ax.set_xlabel("$s$ [m]", fontsize=38, usetex=True, labelpad=10)
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

        for ax0 in [ax]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2.5, direction='out', color=spine_color)
                ax0.tick_params(labelsize=28, pad=10)
        plt.show()


    def plotmat(self, *args):
            """
            Plot beam distributions
            """
            i = 0
            print('Plotting distributions...')
            # if args:
            #     for j in range(self.dof):
            #         self.madx.input(self.name[j] + "=" + str(args[0][j]) + ";")
            #         print(self.name[j][0] + self.name[j][-1] + "=" + str(args[0][j]) + ";")
            # self.madx.use(sequence='TT43', range='#s/#e')
            # self.madx.twiss(RMATRIX=True, BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                    DY=0,
                                    dpy=0)
            # init_dist = self.init_dist
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
            self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                            dpy=0)
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
            self.madx.ptc_setswitch(fringe=True)
            self.madx.ptc_align()
            self.madx.ptc_observe(place='MERGE')
            self.madx.ptc_observe(place='FOIL1')
            self.madx.ptc_observe(place='FOIL2')

            # self.madx.input("ii =" + str(seed) + ";")

            # with self.madx.batch():
            #     for particle in self.init_dist:
            #         self.madx.ptc_start(x=-particle[0] , px=-particle[3],
            #                             y=particle[1] , py=particle[4],
            #                             t=1 * particle[2],
            #                             pt=2.09 * particle[5])
            #     self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
            #                         maxaper=[0.03, 0.03, 0.03, 0.03, 1.0, 1])
            #     self.madx.ptc_track_end()
            # ptc_output = self.madx.table.trackone
            #
            # if any(twiss['name'][:] == 'foil1:1'):
            #     for idx in range(np.size(twiss['name'])):
            #         if twiss['name'][idx] == 'foil1:1bu':
            #             s_foil = twiss['s'][idx]
            #     idx_temp = np.array(ptc_output.s == s_foil)

            with self.madx.batch():
                for particle in self.init_dist:
                    self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                        t=1.4 * particle[2],
                                        pt=2.09 * particle[5])
                self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                    maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone

            for idx in range(np.size(twiss['l'])):
                if twiss['name'][idx] == 'foil1:1':
                    s_foil = twiss['s'][idx]
            idx_temp = np.array(ptc_output.s == s_foil)

            beam_dist = np.vstack((self.madx.table.trackone['x'][idx_temp], self.madx.table.trackone['y'][idx_temp]))
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
            e_beta = 0.999995
            e_mom = 150
            rad_length = 8.9e-2

            if self.foil_w == 0:
                theta_0 = 0
            else:
                theta_temp = 13.6 / (e_beta * e_mom) / np.sqrt(rad_length)
                theta_0 = theta_temp * np.sqrt(self.foil_w) * (1. + 0.038 * np.log(self.foil_w / rad_length))

            x0 = self.madx.table.trackone['x'][idx_temp]
            y0 = self.madx.table.trackone['y'][idx_temp]
            t0 = self.madx.table.trackone['t'][idx_temp]
            px0 = self.madx.table.trackone['px'][idx_temp] + theta_0 * np.random.normal(
                size=np.shape(self.madx.table.trackone['px'][idx_temp]))
            py0 = self.madx.table.trackone['py'][idx_temp] + theta_0 * np.random.normal(
                size=np.shape(self.madx.table.trackone['py'][idx_temp]))
            pt0 = self.madx.table.trackone['pt'][idx_temp]

            self.madx.use(sequence='TT43', range="FOIL1/TT43$END")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
            self.madx.ptc_setswitch(fringe=True)
            self.madx.ptc_align()
            self.madx.ptc_observe(place='MERGE')
            self.madx.ptc_observe(place='FOIL1')
            self.madx.ptc_observe(place='FOIL2')
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)

            init_dist = np.transpose([x0, y0, t0, px0, py0, pt0])

            with self.madx.batch():
                for particle in init_dist:
                    self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                        t=1 * particle[2],
                                        pt=2.09 * particle[5])
                self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                    maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone

            for idx in range(np.size(twiss['l'])):
                if twiss['name'][idx] == 'foil2:1':
                    s_foil = twiss['s'][idx]
            idx_temp = np.array(ptc_output.s == s_foil)

            beam_dist = np.vstack((self.madx.table.trackone['x'][idx_temp], self.madx.table.trackone['y'][idx_temp]))
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
            e_beta = 0.999995
            e_mom = 150
            rad_length = 8.9e-2

            if self.foil_w == 0:
                theta_0 = 0
            else:
                theta_temp = 13.6 / (e_beta * e_mom) / np.sqrt(rad_length)
                theta_0 = theta_temp * np.sqrt(self.foil_w) * (1. + 0.038 * np.log(self.foil_w / rad_length))

            x0 = self.madx.table.trackone['x'][idx_temp]
            y0 = self.madx.table.trackone['y'][idx_temp]
            t0 = self.madx.table.trackone['t'][idx_temp]
            px0 = self.madx.table.trackone['px'][idx_temp] + theta_0 * np.random.normal(
                size=np.shape(self.madx.table.trackone['px'][idx_temp]))
            py0 = self.madx.table.trackone['py'][idx_temp] + theta_0 * np.random.normal(
                size=np.shape(self.madx.table.trackone['py'][idx_temp]))
            pt0 = self.madx.table.trackone['pt'][idx_temp]

            self.madx.use(sequence='TT43', range="FOIL2/TT43$END")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=100)
            self.madx.ptc_setswitch(fringe=True)
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)

            init_dist = np.transpose([x0, y0, t0, px0, py0, pt0])


            with self.madx.batch():
                for particle in init_dist:
                    self.madx.ptc_start(x=particle[0], px=particle[3], y=particle[1], py=particle[4], t=particle[2],
                                        pt=1 * particle[5])
                self.madx.ptc_observe(PLACE='MERGE')
                self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
                                    maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone

            for idx in range(np.size(twiss['l'])):
                if twiss['name'][idx] == 'merge:1':
                    s_merge = twiss['s'][idx]
            idx_temp = np.array(ptc_output.s == s_merge)

            print("input emittance x = " + str(round(self.emitx_before * 150000000 / 0.511, 4)) + " um")
            print("input emittance y = " + str(round(self.emity_before * 150000000 / 0.511, 4)) + " um")
            print("initial twiss parameters (beta - x,y; alpha - x,y; dx - inj, end):")
            print(self.betx0, self.bety0, self.alfx0, self.alfy0)

            x_out = self.madx.table.trackone['x'][idx_temp]
            y_out = self.madx.table.trackone['y'][idx_temp]
            z_out = self.madx.table.trackone['t'][idx_temp]
            px_out = self.madx.table.trackone['px'][idx_temp]
            py_out = self.madx.table.trackone['py'][idx_temp]
            pz_out = self.madx.table.trackone['pt'][idx_temp]

            _, x_ind = self.reject_outliers(x_out)
            _, y_ind = self.reject_outliers(y_out)
            _, z_ind = self.reject_outliers(z_out)
            _, px_ind = self.reject_outliers(px_out)
            _, py_ind = self.reject_outliers(py_out)
            _, pz_ind = self.reject_outliers(pz_out)
            # sum_ind = x_ind + y_ind + z_ind + px_ind + py_ind + pz_ind
            sum_ind = x_ind + y_ind + z_ind + px_ind + py_ind + pz_ind

            x_out = np.multiply(x_out, 1000000)
            y_out = np.multiply(y_out, 1000000)
            z_out = np.multiply(z_out, 1000000)
            px_out = np.multiply(px_out, 1000000)
            py_out = np.multiply(py_out, 1000000)
            pz_out = np.multiply(pz_out, 1000000)
            import scipy.stats as stats
            print("Kurtosises = " + str(stats.kurtosis(x_out[sum_ind], fisher=True)) + ", " + str(stats.kurtosis(y_out[sum_ind], fisher=True)))
            # print(length(x_out))

            beam_size_x = np.std(x_out)
            beam_size_y = np.std(y_out)
            beam_size_z = np.std(z_out)

            fig = plt.figure()
            gs = fig.add_gridspec(5, 5)
            ax1 = plt.subplot(gs[0:2, 0:2])
            ax2 = plt.subplot(gs[3:5, 0:2])
            ax3 = plt.subplot(gs[0:2, 3:5])
            ax4 = plt.subplot(gs[3:5, 3:5])
            # ax1.hist2d(x_out, px_out, bins=100, cmap='my_cmap')
            cmap = 'BuPu'

            # ax1.set_xlabel("$x$ [$\mu$m]", fontsize=34, usetex=True)
            # ax1.set_ylabel("$px$ $[10^{-6}]$", fontsize=38, usetex=True)
            # ax1.set_title("$\sigma_x$ = %.2f" % beam_size_x, fontsize=34, usetex=True)
            # ax1.tick_params(labelsize=28)
            # ax1y = ax1.twinx()
            # ax1y.spines['right'].set_visible(False)
            # ax1y.spines['top'].set_visible(False)
            # ax1y.set_yticks([])
            # # ax1y.plot([0, 0.9], '.', color='white', zorder=0)
            # ax1x = ax1.twiny()
            # ax1x.spines['bottom'].set_visible(False)
            # ax1x.spines['left'].set_visible(False)
            # ax1x.set_xticks([])
            # # ax1x.plot(0.002, 0, 'x', color='red', alpha=0)
            # sns.kdeplot(x_out, px_out, shade=True, ax=ax1, cmap=cmap)
            # sns.distplot(x_out, hist=False, ax=ax1y, color='indigo')
            # sns.distplot(px_out, vertical=True, hist=False, ax=ax1x, color='indigo')
            #
            # ax1.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
            # ax1.set_ylim([-5 * np.std(px_out), 5 * np.std(px_out)])
            # ax1.collections[0].set_alpha(0)
            #
            # ax2.set_xlabel("$y$ [$\mu$m]", fontsize=34, usetex=True)
            # ax2.set_ylabel("$py$ $[10^{-6}]$", fontsize=38, usetex=True)
            # ax2.set_title("$\sigma_y$ = %.2f" % beam_size_y, fontsize=34, usetex=True)
            # ax2.tick_params(labelsize=28)
            # ax2y = ax2.twinx()
            # ax2y.spines['right'].set_visible(False)
            # ax2y.spines['top'].set_visible(False)
            # ax2y.set_yticks([])
            # # ax2y.plot([0, 0.9], '.', color='white', alpha=0)
            # ax2x = ax2.twiny()
            # ax2x.spines['bottom'].set_visible(False)
            # ax2x.spines['left'].set_visible(False)
            # ax2x.set_xticks([])
            # # ax2x.plot(0.002, 0, '.', color='red', alpha=0)
            # sns.kdeplot(y_out, py_out, shade=True, ax=ax2, cmap=cmap)
            # sns.distplot(y_out, hist=False, ax=ax2y, color='indigo')
            # sns.distplot(py_out, vertical=True, hist=False, ax=ax2x, color='indigo')
            # ax2.set_xlim([-5 * np.std(y_out), 5 * np.std(y_out)])
            # ax2.set_ylim([-5 * np.std(py_out), 5 * np.std(py_out)])
            # ax2.collections[0].set_alpha(0)
            #
            # ax3.set_xlabel("$z$ [$\mu$m]", fontsize=34, usetex=True)
            # ax3.set_ylabel("$pz$ $[10^{-6}]$", fontsize=38, usetex=True)
            # ax3.set_title("$\sigma_z$ = %.3f" % beam_size_z, fontsize=34, usetex=True)
            # ax3.tick_params(labelsize=28)
            # ax3y = ax3.twinx()
            # ax3y.spines['right'].set_visible(False)
            # ax3y.spines['top'].set_visible(False)
            # ax3y.set_yticks([])
            # # ax3y.plot([0, 0.1], '.', color='white', alpha=0)
            # ax3x = ax3.twiny()
            # ax3x.spines['bottom'].set_visible(False)
            # ax3x.spines['left'].set_visible(False)
            # ax3x.set_xticks([])
            # # ax3x.plot(0.002, 0, 'o', color='green', alpha=0)
            # sns.kdeplot(z_out, pz_out, shade=True, ax=ax3, cmap=cmap)
            # sns.distplot(z_out, hist=False, ax=ax3y, color='indigo')
            # sns.distplot(pz_out, vertical=True, hist=False, ax=ax3x, color='indigo')
            # ax3.set_xlim([-5 * np.std(z_out), 4 * np.std(z_out)])
            # ax3.set_ylim([-4 * np.std(pz_out), 4 * np.std(pz_out)])
            # ax3.collections[0].set_alpha(0)
            #
            # ax4.set_xlabel("$x$ [$\mu$m]", fontsize=34, usetex=True)
            # ax4.set_ylabel("$y$ [$\mu$m]", fontsize=38, usetex=True)
            # # ax4.set_title("$\sigma_y$ = %.3f" % beam_size_y, fontsize=34, usetex=True)
            # ax4.tick_params(labelsize=28)
            # ax4y = ax4.twinx()
            # ax4y.spines['right'].set_visible(False)
            # ax4y.spines['top'].set_visible(False)
            # ax4y.set_yticks([])
            # # ax4y.plot([0, 0.4], '.', color='white', alpha=0)
            # ax4x = ax4.twiny()
            # ax4x.spines['bottom'].set_visible(False)
            # ax4x.spines['left'].set_visible(False)
            # ax4x.set_xticks([])
            # # ax4x.plot([-20, -20], '.', color='white', alpha=0)
            # sns.kdeplot(x_out, y_out, shade=True, ax=ax4, cmap=cmap)
            # sns.distplot(x_out, hist=False, ax=ax4y, color='indigo')
            # sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='indigo')
            # # print(np.shape(x_out))
            # x_out_1, _ = self.reject_outliers(x_out)
            # mean, std = norm.fit(x_out_1)
            # # print(std)
            # x = np.linspace(-25, 25, 100)
            # y = norm.pdf(x, mean, std)
            # ax4.plot(x, y * 700 - 5 * np.std(y_out))
            # ## ax4x.plot(350*y)
            # y_out_1, _ = self.reject_outliers(np.array(y_out))
            # mean, std = norm.fit(y_out_1)
            # # print(std)
            # x = np.linspace(-25, 25, 100)
            # y = norm.pdf(x, mean, std)
            #
            # ax4.plot(y * 700 - 5 * np.std(x_out), x)
            # ax4.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
            # ax4.set_ylim([-5 * np.std(y_out), 5 * np.std(y_out)])
            # ax4.collections[0].set_alpha(0)
            #
            # spine_color = 'gray'
            # for ax in [ax1, ax2, ax3, ax4]:
            #     ax.tick_params(labelsize=28)
            #     ax.xaxis.set_ticks_position('bottom')
            #     ax.yaxis.set_ticks_position('left')
            #     for spine in ['top', 'right']:
            #         ax.spines[spine].set_visible(False)
            #     for spine in ['left', 'bottom']:
            #         ax.spines[spine].set_color(spine_color)
            #         ax.spines[spine].set_linewidth(2.5)
            #         ax.tick_params(width=2.5, direction='out', color=spine_color)
            # plt.show()
            ax1.set_xlabel("$x$ [$\mu$m]", fontsize=32, usetex=True)
            ax1.set_ylabel(r"${p_x}/{p_0}$ $[10^{-3}]$", fontsize=32, usetex=True)
            ax1.set_title("$\sigma_x$ = %.2f $\mu$m" % beam_size_x, fontsize=32, usetex=True)
            ax1.tick_params(labelsize=30)
            ax1y = ax1.twinx()
            ax1y.spines['right'].set_visible(False)
            ax1y.spines['top'].set_visible(False)
            ax1y.set_yticks([])
            # ax1y.plot([0, 0.9], '.', color='white', zorder=0)
            ax1x = ax1.twiny()
            ax1x.spines['bottom'].set_visible(False)
            ax1x.spines['left'].set_visible(False)
            ax1x.set_xticks([])
            # ax1x.plot(0.002, 0, 'x', color='red', alpha=0)
            sns.kdeplot(x_out, px_out / 10 ** 3, shade=True, ax=ax1, cmap=cmap)
            # ax1.scatter(x_out, px_out/10**3, c=pz_out, cmap='coolwarm')
            sns.distplot(x_out, hist=False, ax=ax1y, color='navy')
            sns.distplot(px_out / 10 ** 3, vertical=True, hist=False, ax=ax1x, color='navy')

            ax1.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
            ax1.set_ylim([-5 * np.std(px_out / 10 ** 3), 5 * np.std(px_out / 10 ** 3)])
            from matplotlib.ticker import (MultipleLocator,
                                           FormatStrFormatter,
                                           AutoMinorLocator)
            # ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1d'))
            # ax1.collections[0].set_alpha(0)

            ax2.set_xlabel("$y$ [$\mu$m]", fontsize=32, usetex=True)
            ax2.set_ylabel(r"${p_y}/{p_0}$ $[10^{-3}]$", fontsize=32, usetex=True)
            ax2.set_title("$\sigma_y$ = %.2f $\mu$m" % beam_size_y, fontsize=32, usetex=True)
            ax2.tick_params(labelsize=30)
            ax2y = ax2.twinx()
            ax2y.spines['right'].set_visible(False)
            ax2y.spines['top'].set_visible(False)
            ax2y.set_yticks([])
            # ax2y.plot([0, 0.9], '.', color='white', alpha=0)
            ax2x = ax2.twiny()
            ax2x.spines['bottom'].set_visible(False)
            ax2x.spines['left'].set_visible(False)
            ax2x.set_xticks([])
            # ax2x.plot(0.002, 0, '.', color='red', alpha=0)
            sns.kdeplot(y_out, py_out / 10 ** 3, shade=True, ax=ax2, cmap=cmap)
            # ax2.scatter(y_out, py_out/10**3, c=pz_out, cmap='coolwarm')
            sns.distplot(y_out, hist=False, ax=ax2y, color='navy')
            sns.distplot(py_out / 10 ** 3, vertical=True, hist=False, ax=ax2x, color='navy')
            # sns.distplot(y_out, hist=False, ax=ax2y, color='indigo')
            # sns.distplot(py_out, vertical=True, hist=False, ax=ax2x, color='indigo')
            ax2.set_xlim([-5 * np.std(y_out), 5 * np.std(y_out)])
            ax2.set_ylim([-5 * np.std(py_out / 10 ** 3), 5 * np.std(py_out / 10 ** 3)])
            # ax2.collections[0].set_alpha(0)

            ax3.set_xlabel("$z$ [$\mu$m]", fontsize=32, usetex=True)
            ax3.set_ylabel(r"${\Delta E}/{pc}$ $[10^{-3}]$", fontsize=32, usetex=True)
            ax3.set_title("$\sigma_z$ = %.2f $\mu$m" % beam_size_z, fontsize=32, usetex=True)
            ax3.tick_params(labelsize=30)
            ax3y = ax3.twinx()
            ax3y.spines['right'].set_visible(False)
            ax3y.spines['top'].set_visible(False)
            ax3y.set_yticks([])
            # ax3y.plot([0, 0.1], '.', color='white', alpha=0)
            ax3x = ax3.twiny()
            ax3x.spines['bottom'].set_visible(False)
            ax3x.spines['left'].set_visible(False)
            ax3x.set_xticks([])
            # ax3x.plot(0.002, 0, 'o', color='green', alpha=0)
            sns.kdeplot(z_out, pz_out / 10 ** 3, shade=True, ax=ax3, cmap=cmap)
            # sns.distplot(z_out, hist=False, ax=ax3y, color='indigo')
            # sns.distplot(pz_out, vertical=True, hist=False, ax=ax3x, color='indigo')
            # ax3.scatter(z_out, pz_out/10**3, c=pz_out, cmap='coolwarm')
            sns.distplot(z_out, hist=False, ax=ax3y, color='navy')
            sns.distplot(pz_out / 10 ** 3, vertical=True, hist=False, ax=ax3x, color='navy')
            ax3.set_xlim([-5 * np.std(z_out), 4 * np.std(z_out)])
            ax3.set_ylim([-4 * np.std(pz_out / 10 ** 3), 4 * np.std(pz_out / 10 ** 3)])
            # ax3.collections[0].set_alpha(0)

            ax4.set_xlabel("$x$ [$\mu$m]", fontsize=32, usetex=True)
            ax4.set_ylabel("$y$ [$\mu$m]", fontsize=32, usetex=True)
            # ax4.set_title("$\sigma_y$ = %.3f" % beam_size_y, fontsize=34, usetex=True)
            ax4.tick_params(labelsize=30)
            ax4y = ax4.twinx()
            ax4y.spines['right'].set_visible(False)
            ax4y.spines['top'].set_visible(False)
            ax4y.set_yticks([])
            # ax4y.plot([0, 0.4], '.', color='white', alpha=0)
            ax4x = ax4.twiny()
            ax4x.spines['bottom'].set_visible(False)
            ax4x.spines['left'].set_visible(False)
            ax4x.set_xticks([])
            # ax4x.plot([-20, -20], '.', color='white', alpha=0)
            sns.kdeplot(x_out, y_out, shade=True, ax=ax4, cmap=cmap)
            # sns.distplot(x_out, hist=False, ax=ax4y, color='indigo')
            # sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='indigo')
            # ax4.scatter(x_out, y_out, c=pz_out, cmap='coolwarm')
            sns.distplot(x_out, hist=False, ax=ax4y, color='navy')
            sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='navy')
            # print(np.shape(x_out))
            x_out_1, _ = self.reject_outliers(x_out)
            mean, std = norm.fit(x_out_1)
            # print(std)
            x = np.linspace(-20, 20, 100)
            y = norm.pdf(x, mean, np.std(y_out))

            # ax4.plot(x, y * 860 - 5 * np.std(y_out), color="orange")
            ## ax4x.plot(350*y)
            y_out_1, _ = self.reject_outliers(np.array(y_out))
            mean, std = norm.fit(y_out_1)
            # print(std)
            x = np.linspace(-20, 20, 100)
            y = norm.pdf(x, mean, np.std(x_out))

            # ax4.plot(y * 860 - 5 * np.std(x_out), x, color="orange")
            ax4.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
            ax4.set_ylim([-5 * np.std(y_out), 5 * np.std(y_out)])
            # ax4.collections[0].set_alpha(0)

            spine_color = 'gray'
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(labelsize=28)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                for spine in ['left', 'bottom']:
                    ax.spines[spine].set_color(spine_color)
                    ax.spines[spine].set_linewidth(2.5)
                    ax.tick_params(width=2.5, direction='out', color=spine_color)
            plt.show()

    def plotmat_old(self, *args):
        """
        Plot beam distributions
        """
        i = 0
        print('Plotting distributions...')
        # if args:
        #     for j in range(self.dof):
        #         self.madx.input(self.name[j] + "=" + str(args[0][j]) + ";")
        #         print(self.name[j][0] + self.name[j][-1] + "=" + str(args[0][j]) + ";")
        self.madx.use(sequence='TT43', range='#s/#e')
        # self.madx.twiss(RMATRIX=True, BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                dpy=0)
        # init_dist = self.init_dist
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0)
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_align()
        self.madx.ptc_observe(place='MERGE')
        self.madx.ptc_observe(place='FOIL')

        # self.madx.input("ii =" + str(seed) + ";")

        # with self.madx.batch():
        #     for particle in init_dist:
        #         self.madx.ptc_start(x=-particle[0] , px=-particle[3],
        #                             y=particle[1] , py=particle[4],
        #                             t=1 * particle[2],
        #                             pt=2.09 * particle[5] )
        #     self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
        #                         maxaper=[0.03, 0.03, 0.03, 0.03, 1.0, 1])
        #     self.madx.ptc_track_end()
        # ptc_output = self.madx.table.trackone

        # if any(twiss['name'][:] == 'merge:1'):
        #     for idx in range(np.size(twiss['name'])):
        #         if twiss['name'][idx] == 'merge:1':
        #             s_foil = twiss['s'][idx]
        #     idx_temp = np.array(ptc_output.s == s_foil)

        with self.madx.batch():
            for particle in self.init_dist:
                self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                    t=1 * particle[2],
                                    pt=2.09 * particle[5])
            self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
            self.madx.ptc_track_end()
        ptc_output = self.madx.table.trackone

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'foil:1':
                s_foil = twiss['s'][idx]
        idx_temp = np.array(ptc_output.s == s_foil)

        beam_dist = np.vstack((self.madx.table.trackone['x'][idx_temp], self.madx.table.trackone['y'][idx_temp]))
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        e_beta = 0.999995
        e_mom = 150
        rad_length = 8.9e-2

        if self.foil_w == 0:
            theta_0 = 0
        else:
            theta_temp = 13.6 / (e_beta * e_mom) / np.sqrt(rad_length)
            theta_0 = theta_temp * np.sqrt(self.foil_w) * (1. + 0.038 * np.log(self.foil_w / rad_length))

        x0 = self.madx.table.trackone['x'][idx_temp]
        y0 = self.madx.table.trackone['y'][idx_temp]
        t0 = self.madx.table.trackone['t'][idx_temp]
        px0 = self.madx.table.trackone['px'][idx_temp] + theta_0 * np.random.normal(
            size=np.shape(self.madx.table.trackone['px'][idx_temp]))
        py0 = self.madx.table.trackone['py'][idx_temp] + theta_0 * np.random.normal(
            size=np.shape(self.madx.table.trackone['py'][idx_temp]))
        pt0 = self.madx.table.trackone['pt'][idx_temp]

        self.madx.use(sequence='TT43', range="FOIL/TT43$END")
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=2, exact=True, NST=15)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)

        init_dist = np.transpose([x0, y0, t0, px0, py0, pt0])

        with self.madx.batch():
            for particle in init_dist:
                self.madx.ptc_start(x=particle[0], px=particle[3], y=particle[1], py=particle[4], t=particle[2],
                                    pt=1 * particle[5])
            self.madx.ptc_observe(PLACE='MERGE')
            self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
                                maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
            self.madx.ptc_track_end()
        ptc_output = self.madx.table.trackone

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'merge:1':
                s_merge = twiss['s'][idx]
        idx_temp = np.array(ptc_output.s == s_merge)

        print("input emittance x = " + str(round(self.emitx_before * 150000000 / 0.511, 4)) + " um")
        print("input emittance y = " + str(round(self.emity_before * 150000000 / 0.511, 4)) + " um")
        print("initial twiss parameters (beta - x,y; alpha - x,y; dx - inj, end):")
        print(self.betx0, self.bety0, self.alfx0, self.alfy0)

        x_out = self.madx.table.trackone['x'][idx_temp]
        y_out = self.madx.table.trackone['y'][idx_temp]
        z_out = self.madx.table.trackone['t'][idx_temp]
        px_out = self.madx.table.trackone['px'][idx_temp]
        py_out = self.madx.table.trackone['py'][idx_temp]
        pz_out = self.madx.table.trackone['pt'][idx_temp]

        # Very temp
        # x_out = np.transpose(self.init_dist)[0]
        # y_out = np.transpose(self.init_dist)[1]
        # z_out = np.transpose(self.init_dist)[2]
        # px_out = np.transpose(self.init_dist)[3]
        # py_out = np.transpose(self.init_dist)[4]
        # pz_out = np.transpose(self.init_dist)[5]*2.09
        #

        x_out = np.multiply(x_out, 1000000)
        y_out = np.multiply(y_out, 1000000)
        z_out = np.multiply(z_out, 1000000)
        px_out = np.multiply(px_out, 1000000)
        py_out = np.multiply(py_out, 1000000)
        pz_out = np.multiply(pz_out, 1000000)
        # print(length(x_out))

        beam_size_x = np.std(x_out)
        beam_size_y = np.std(y_out)
        beam_size_z = np.std(z_out)

        fig = plt.figure(figsize=(10,10), constrained_layout=True)
        gs = fig.add_gridspec(5, 5)
        ax1 = plt.subplot(gs[0:2, 0:2])
        ax2 = plt.subplot(gs[3:5, 0:2])
        ax3 = plt.subplot(gs[0:2, 3:5])
        ax4 = plt.subplot(gs[3:5, 3:5])
        # ax1.hist2d(x_out, px_out, bins=100, cmap='my_cmap')
        cmap = 'BuPu'

        ax1.set_xlabel("$x$ [$\mu$m]", fontsize=34, usetex=True)
        ax1.set_ylabel(r"$\frac{p_x}{p_0}$ $[10^{-6}]$", fontsize=38, usetex=True)
        ax1.set_title("$\sigma_x$ = %.2f $\mu$m" % beam_size_x, fontsize=34, usetex=True)
        ax1.tick_params(labelsize=28)
        ax1y = ax1.twinx()
        ax1y.spines['right'].set_visible(False)
        ax1y.spines['top'].set_visible(False)
        ax1y.set_yticks([])
        # ax1y.plot([0, 0.9], '.', color='white', zorder=0)
        ax1x = ax1.twiny()
        ax1x.spines['bottom'].set_visible(False)
        ax1x.spines['left'].set_visible(False)
        ax1x.set_xticks([])
        # ax1x.plot(0.002, 0, 'x', color='red', alpha=0)
        sns.kdeplot(x_out, px_out, shade=True, ax=ax1, cmap=cmap)
        sns.distplot(x_out, hist=False, ax=ax1y, color='indigo')
        sns.distplot(px_out, vertical=True, hist=False, ax=ax1x, color='indigo')

        ax1.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
        ax1.set_ylim([-5 * np.std(px_out), 5 * np.std(px_out)])
        ax1.collections[0].set_alpha(0)

        ax2.set_xlabel("$y$ [$\mu$m]", fontsize=34, usetex=True)
        ax2.set_ylabel(r"$\frac{p_y}{p_0}$ $[10^{-6}]$", fontsize=38, usetex=True)
        ax2.set_title("$\sigma_y$ = %.2f $\mu$m" % beam_size_y, fontsize=34, usetex=True)
        ax2.tick_params(labelsize=28)
        ax2y = ax2.twinx()
        ax2y.spines['right'].set_visible(False)
        ax2y.spines['top'].set_visible(False)
        ax2y.set_yticks([])
        # ax2y.plot([0, 0.9], '.', color='white', alpha=0)
        ax2x = ax2.twiny()
        ax2x.spines['bottom'].set_visible(False)
        ax2x.spines['left'].set_visible(False)
        ax2x.set_xticks([])
        # ax2x.plot(0.002, 0, '.', color='red', alpha=0)
        sns.kdeplot(y_out, py_out, shade=True, ax=ax2, cmap=cmap)
        sns.distplot(y_out, hist=False, ax=ax2y, color='indigo')
        sns.distplot(py_out, vertical=True, hist=False, ax=ax2x, color='indigo')
        ax2.set_xlim([-5 * np.std(y_out), 5 * np.std(y_out)])
        ax2.set_ylim([-5 * np.std(py_out), 5 * np.std(py_out)])
        ax2.collections[0].set_alpha(0)

        ax3.set_xlabel("$z$ [$\mu$m]", fontsize=34, usetex=True)
        ax3.set_ylabel(r"$\frac{\Delta E}{pc}$ $[10^{-6}]$", fontsize=38, usetex=True)
        ax3.set_title("$\sigma_z$ = %.2f $\mu$m" % beam_size_z, fontsize=34, usetex=True)
        ax3.tick_params(labelsize=28)
        ax3y = ax3.twinx()
        ax3y.spines['right'].set_visible(False)
        ax3y.spines['top'].set_visible(False)
        ax3y.set_yticks([])
        # ax3y.plot([0, 0.1], '.', color='white', alpha=0)
        ax3x = ax3.twiny()
        ax3x.spines['bottom'].set_visible(False)
        ax3x.spines['left'].set_visible(False)
        ax3x.set_xticks([])
        # ax3x.plot(0.002, 0, 'o', color='green', alpha=0)
        sns.kdeplot(z_out, pz_out, shade=True, ax=ax3, cmap=cmap)
        sns.distplot(z_out, hist=False, ax=ax3y, color='indigo')
        sns.distplot(pz_out, vertical=True, hist=False, ax=ax3x, color='indigo')
        ax3.set_xlim([-5 * np.std(z_out), 4 * np.std(z_out)])
        ax3.set_ylim([-4 * np.std(pz_out), 4 * np.std(pz_out)])
        ax3.collections[0].set_alpha(0)

        ax4.set_xlabel("$x$ [$\mu$m]", fontsize=34, usetex=True)
        ax4.set_ylabel("$y$ [$\mu$m]", fontsize=38, usetex=True)
        # ax4.set_title("$\sigma_y$ = %.3f" % beam_size_y, fontsize=34, usetex=True)
        ax4.tick_params(labelsize=28)
        ax4y = ax4.twinx()
        ax4y.spines['right'].set_visible(False)
        ax4y.spines['top'].set_visible(False)
        ax4y.set_yticks([])
        # ax4y.plot([0, 0.4], '.', color='white', alpha=0)
        ax4x = ax4.twiny()
        ax4x.spines['bottom'].set_visible(False)
        ax4x.spines['left'].set_visible(False)
        ax4x.set_xticks([])
        # ax4x.plot([-20, -20], '.', color='white', alpha=0)
        sns.kdeplot(x_out, y_out, shade=True, ax=ax4, cmap=cmap)
        sns.distplot(x_out, hist=False, ax=ax4y, color='indigo')
        sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='indigo')
        # print(np.shape(x_out))
        x_out_1 = self.reject_outliers(x_out)
        mean, std = norm.fit(x_out_1)
        # print(std)
        x = np.linspace(-20, 20, 100)
        y = norm.pdf(x, mean, std)
        ax4.plot(x, y * 860 - 5 * np.std(y_out))
        ## ax4x.plot(350*y)
        y_out_1 = self.reject_outliers(np.array(y_out))
        mean, std = norm.fit(y_out_1)
        # print(std)
        x = np.linspace(-20, 20, 100)
        y = norm.pdf(x, mean, std)

        ax4.plot(y * 860 - 5 * np.std(x_out), x)
        ax4.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
        ax4.set_ylim([-5 * np.std(y_out), 5 * np.std(y_out)])
        ax4.collections[0].set_alpha(0)

        spine_color = 'gray'
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=28)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(spine_color)
                ax.spines[spine].set_linewidth(2.5)
                ax.tick_params(width=2.5, direction='out', color=spine_color)
        plt.show()

    def error(self):
        n_seeds = 25
        bsx_total = []
        bsy_total = []
        scale_all = []
        for scale in range(25):
            bsx_all = []
            bsy_all = []
            scale_all = np.append(scale_all, scale)
            for seed in range(n_seeds):
                self.madx.input("ii =" + str(seed) + ";")
                self.madx.input("scale =" + str(scale) + ";")
                self.madx.call("add_errors.madx")
                bsx, bsy = self.output_beamsize()
                bsx_all = np.append(bsx_all, bsx)
                bsy_all = np.append(bsy_all, bsy)
            print(np.mean(bsx_all), np.mean(bsy_all))
            bsx_total = np.append(bsx_total, np.mean(bsx_all))
            bsy_total = np.append(bsy_total, np.mean(bsy_all))
        print(bsx_total, bsy_total)
        plt.figure(figsize=(8,6), constrained_layout=True)
        params = {'axes.labelsize': 24,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 24,
                  'legend.fontsize': 24,  # was 10
                  'xtick.labelsize': 30,
                  'ytick.labelsize': 30,
                  'axes.linewidth': 1.5,
                  'lines.linewidth': 3,
                  'text.usetex': True,
                  'font.family': 'serif'
                  }
        plt.rcParams.update(params)
        plt.plot(scale_all, bsx_total, label="$\sigma_x$", linewidth=3)
        plt.plot(scale_all, bsy_total,  label="$\sigma_y$", linewidth=3)
        plt.xlabel("1$\sigma$ quadrupole misalignment [$\mu$m]", fontsize=30, labelpad=5)
        plt.ylabel("1$\sigma$ beam size [$\mu$m]", fontsize=30, labelpad=5)
        plt.legend()

    def output_beamsize(self):
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                DY=0,
                                dpy=0)
        # init_dist = self.init_dist
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                        dpy=0)
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_align()
        self.madx.ptc_observe(place='MERGE')
        self.madx.ptc_observe(place='FOIL')

        with self.madx.batch():
            for particle in self.init_dist:
                self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                    t=1 * particle[2],
                                    pt=2.09 * particle[5])
            self.madx.ptc_track(icase=56, element_by_element=True, dump=False, onetable=True, recloss=True,
                                maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
            self.madx.ptc_track_end()
        ptc_output = self.madx.table.trackone

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'merge:1':
                s_foil = twiss['s'][idx]
        idx_temp = np.array(ptc_output.s == s_foil)

        x_out = self.madx.table.trackone['x'][idx_temp]
        y_out = self.madx.table.trackone['y'][idx_temp]
        z_out = self.madx.table.trackone['t'][idx_temp]
        px_out = self.madx.table.trackone['px'][idx_temp]
        py_out = self.madx.table.trackone['py'][idx_temp]
        pz_out = self.madx.table.trackone['pt'][idx_temp]

        _, x_ind = self.reject_outliers(x_out)
        _, y_ind = self.reject_outliers(y_out)
        _, z_ind = self.reject_outliers(z_out)
        _, px_ind = self.reject_outliers(px_out)
        _, py_ind = self.reject_outliers(py_out)
        _, pz_ind = self.reject_outliers(pz_out)
        # sum_ind = x_ind + y_ind + z_ind + px_ind + py_ind + pz_ind
        sum_ind = x_ind + y_ind + z_ind + px_ind + py_ind + pz_ind

        x_out = np.multiply(x_out, 1000000)
        y_out = np.multiply(y_out, 1000000)
        z_out = np.multiply(z_out, 1000000)

        beam_size_x = np.std(x_out)
        beam_size_y = np.std(y_out)
        beam_size_z = np.std(z_out)
        return beam_size_x, beam_size_y

    def plotmat_twiss(self, *args):
        """
        Plot beam distributions
        """
        i = 0
        observe = ['MERGE']
        print('Plotting distributions...')
        if args:
            for j in range(self.dof):
                self.madx.input(self.name[j] + "=" + str(args[0][j]) + ";")
                print(self.name[j][0] + self.name[j][-1] + "=" + str(args[0][j]))
        # self.madx.use(sequence='TT43', range='#s/#e')
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0)
        try:
            with self.madx.batch():
                self.madx.track(onetable=True, recloss=True, onepass=True)
                for particle in self.init_dist:
                    self.madx.start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                    t=1 * particle[2], pt=2.09 * particle[5])
                for obs in observe:
                    self.madx.observe(place=obs)
                self.madx.run(turns=1, maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.endtrack()

        except RuntimeError:
            print('MAD-X Error occurred, re-spawning MAD-X process')
            loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
            sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        else:
            ptc_output = self.madx.table.trackone
            ptc_output = ptc_output.dframe()

        for idx in range(np.size(twiss['l'])):
            if twiss['name'][idx] == 'tt43$start:1':
                s_merge = twiss['s'][idx]
        idx_temp = np.array(ptc_output.s == s_merge)

        print("input emittance x = " + str(round(self.emitx_before * 150000000 / 0.511, 4)) + " um")
        print("input emittance y = " + str(round(self.emity_before * 150000000 / 0.511, 4)) + " um")
        print("initial twiss parameters (beta - x,y; alpha - x,y; dx - inj, end):")
        print(self.betx0, self.bety0, self.alfx0, self.alfy0)

        x_out = self.madx.table.trackone['x'][idx_temp]
        y_out = self.madx.table.trackone['y'][idx_temp]
        z_out = self.madx.table.trackone['t'][idx_temp]
        px_out = self.madx.table.trackone['px'][idx_temp]
        py_out = self.madx.table.trackone['py'][idx_temp]
        pz_out = self.madx.table.trackone['pt'][idx_temp]

        # Very temp
        # x_out = np.transpose(self.init_dist)[0]
        # y_out = np.transpose(self.init_dist)[1]
        # z_out = np.transpose(self.init_dist)[2]
        # px_out = np.transpose(self.init_dist)[3]
        # py_out = np.transpose(self.init_dist)[4]
        # pz_out = np.transpose(self.init_dist)[5]*2.09
        #

        x_out = np.multiply(x_out, 1000000)
        y_out = np.multiply(y_out, 1000000)
        z_out = np.multiply(z_out, 1000000)
        px_out = np.multiply(px_out, 1000000)
        py_out = np.multiply(py_out, 1000000)
        pz_out = np.multiply(pz_out, 1000000)
        # print(length(x_out))

        beam_size_x = np.std(x_out)
        beam_size_y = np.std(y_out)
        beam_size_z = np.std(z_out)

        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        gs = fig.add_gridspec(5, 5)
        ax1 = plt.subplot(gs[0:2, 0:2])
        ax2 = plt.subplot(gs[3:5, 0:2])
        ax3 = plt.subplot(gs[0:2, 3:5])
        ax4 = plt.subplot(gs[3:5, 3:5])
        # ax1.hist2d(x_out, px_out, bins=100, cmap='my_cmap')
        cmap = 'BuPu'

        ax1.set_xlabel("$x$ [$\mu$m]", fontsize=32, usetex=True)
        ax1.set_ylabel(r"${p_x}/{p_0}$ $[10^{-3}]$", fontsize=32, usetex=True)
        ax1.set_title("$\sigma_x$ = %.2f $\mu$m" % beam_size_x, fontsize=32, usetex=True)
        ax1.tick_params(labelsize=30)
        ax1y = ax1.twinx()
        ax1y.spines['right'].set_visible(False)
        ax1y.spines['top'].set_visible(False)
        ax1y.set_yticks([])
        # ax1y.plot([0, 0.9], '.', color='white', zorder=0)
        ax1x = ax1.twiny()
        ax1x.spines['bottom'].set_visible(False)
        ax1x.spines['left'].set_visible(False)
        ax1x.set_xticks([])
        # ax1x.plot(0.002, 0, 'x', color='red', alpha=0)
        sns.kdeplot(x_out, px_out/10**3, shade=True, ax=ax1, cmap=cmap)
        # ax1.scatter(x_out, px_out/10**3, c=pz_out, cmap='coolwarm')
        sns.distplot(x_out, hist=False, ax=ax1y, color='navy')
        sns.distplot(px_out/10**3, vertical=True, hist=False, ax=ax1x, color='navy')

        ax1.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
        ax1.set_ylim([-5 * np.std(px_out/10**3), 5 * np.std(px_out/10**3)])
        from matplotlib.ticker import (MultipleLocator,
                                       FormatStrFormatter,
                                       AutoMinorLocator)
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1d'))
        # ax1.collections[0].set_alpha(0)

        ax2.set_xlabel("$y$ [$\mu$m]", fontsize=32, usetex=True)
        ax2.set_ylabel(r"${p_y}/{p_0}$ $[10^{-3}]$", fontsize=32, usetex=True)
        ax2.set_title("$\sigma_y$ = %.2f $\mu$m" % beam_size_y, fontsize=32, usetex=True)
        ax2.tick_params(labelsize=30)
        ax2y = ax2.twinx()
        ax2y.spines['right'].set_visible(False)
        ax2y.spines['top'].set_visible(False)
        ax2y.set_yticks([])
        # ax2y.plot([0, 0.9], '.', color='white', alpha=0)
        ax2x = ax2.twiny()
        ax2x.spines['bottom'].set_visible(False)
        ax2x.spines['left'].set_visible(False)
        ax2x.set_xticks([])
        # ax2x.plot(0.002, 0, '.', color='red', alpha=0)
        sns.kdeplot(y_out, py_out/10**3, shade=True, ax=ax2, cmap=cmap)
        # ax2.scatter(y_out, py_out/10**3, c=pz_out, cmap='coolwarm')
        sns.distplot(y_out, hist=False, ax=ax2y, color='navy')
        sns.distplot(py_out/10**3, vertical=True, hist=False, ax=ax2x, color='navy')
        # sns.distplot(y_out, hist=False, ax=ax2y, color='indigo')
        # sns.distplot(py_out, vertical=True, hist=False, ax=ax2x, color='indigo')
        ax2.set_xlim([-5 * np.std(y_out), 5 * np.std(y_out)])
        ax2.set_ylim([-5 * np.std(py_out/10**3), 5 * np.std(py_out/10**3)])
        # ax2.collections[0].set_alpha(0)

        ax3.set_xlabel("$z$ [$\mu$m]", fontsize=32, usetex=True)
        ax3.set_ylabel(r"${\Delta E}/{pc}$ $[10^{-3}]$", fontsize=32, usetex=True)
        ax3.set_title("$\sigma_z$ = %.2f $\mu$m" % beam_size_z, fontsize=32, usetex=True)
        ax3.tick_params(labelsize=30)
        ax3y = ax3.twinx()
        ax3y.spines['right'].set_visible(False)
        ax3y.spines['top'].set_visible(False)
        ax3y.set_yticks([])
        # ax3y.plot([0, 0.1], '.', color='white', alpha=0)
        ax3x = ax3.twiny()
        ax3x.spines['bottom'].set_visible(False)
        ax3x.spines['left'].set_visible(False)
        ax3x.set_xticks([])
        # ax3x.plot(0.002, 0, 'o', color='green', alpha=0)
        sns.kdeplot(z_out, pz_out/10**3, shade=True, ax=ax3, cmap=cmap)
        # sns.distplot(z_out, hist=False, ax=ax3y, color='indigo')
        # sns.distplot(pz_out, vertical=True, hist=False, ax=ax3x, color='indigo')
        # ax3.scatter(z_out, pz_out/10**3, c=pz_out, cmap='coolwarm')
        sns.distplot(z_out, hist=False, ax=ax3y, color='navy')
        sns.distplot(pz_out/10**3, vertical=True, hist=False, ax=ax3x, color='navy')
        ax3.set_xlim([-5 * np.std(z_out), 4 * np.std(z_out)])
        ax3.set_ylim([-4 * np.std(pz_out/10**3), 4 * np.std(pz_out/10**3)])
        # ax3.collections[0].set_alpha(0)

        ax4.set_xlabel("$x$ [$\mu$m]", fontsize=32, usetex=True)
        ax4.set_ylabel("$y$ [$\mu$m]", fontsize=32, usetex=True)
        # ax4.set_title("$\sigma_y$ = %.3f" % beam_size_y, fontsize=34, usetex=True)
        ax4.tick_params(labelsize=30)
        ax4y = ax4.twinx()
        ax4y.spines['right'].set_visible(False)
        ax4y.spines['top'].set_visible(False)
        ax4y.set_yticks([])
        # ax4y.plot([0, 0.4], '.', color='white', alpha=0)
        ax4x = ax4.twiny()
        ax4x.spines['bottom'].set_visible(False)
        ax4x.spines['left'].set_visible(False)
        ax4x.set_xticks([])
        # ax4x.plot([-20, -20], '.', color='white', alpha=0)
        sns.kdeplot(x_out, y_out, shade=True, ax=ax4, cmap=cmap)
        # sns.distplot(x_out, hist=False, ax=ax4y, color='indigo')
        # sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='indigo')
        # ax4.scatter(x_out, y_out, c=pz_out, cmap='coolwarm')
        sns.distplot(x_out, hist=False, ax=ax4y, color='navy')
        sns.distplot(y_out, vertical=True, hist=False, ax=ax4x, color='navy')
        # print(np.shape(x_out))
        x_out_1, _ = self.reject_outliers(x_out)
        mean, std = norm.fit(x_out_1)
        # print(std)
        x = np.linspace(-20, 20, 100)
        y = norm.pdf(x, mean, np.std(y_out))

        # ax4.plot(x, y * 860 - 5 * np.std(y_out), color="orange")
        ## ax4x.plot(350*y)
        y_out_1, _ = self.reject_outliers(np.array(y_out))
        mean, std = norm.fit(y_out_1)
        # print(std)
        x = np.linspace(-20, 20, 100)
        y = norm.pdf(x, mean,  np.std(x_out))

        # ax4.plot(y * 860 - 5 * np.std(x_out), x, color="orange")
        ax4.set_xlim([-5 * np.std(x_out), 5 * np.std(x_out)])
        ax4.set_ylim([-5 * np.std(y_out), 5 * np.std(y_out)])
        # ax4.collections[0].set_alpha(0)

        spine_color = 'gray'
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(labelsize=28)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color(spine_color)
                ax.spines[spine].set_linewidth(2.5)
                ax.tick_params(width=2.5, direction='out', color=spine_color)
        plt.show()

    def plotheat(self):
        """
        Plot heatmaps of deltap vs s and transverse offset vs s
        """

        long = np.array([])
        betx = np.array([])
        bety = np.array([])
        dx = np.array([])
        ptout = np.array([])
        self.madx.select(flag='interpolate', step=0.01)

        twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                alfy=str(-2.1110),
                                deltap=str(-0.00))
        betx0 = twiss.betx
        bety0 = twiss.bety
        dx0 = twiss.dx
        for i in range(81):
            pt = -0.002 + i * 0.00005
            self.madx.select(flag='interpolate', step=0.01)
            twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                    alfy=str(-2.1110), deltap=str(pt))
            long = np.squeeze(twiss.s)
            betx = np.append(betx, np.divide(twiss.betx - betx0, betx0))
            bety = np.append(bety, np.divide(twiss.bety - bety0, bety0))
            dx = np.append(dx, np.divide(twiss.dx - dx0, 1))
            ptout = np.append(ptout, pt)

        params = {'axes.labelsize': 10,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 10,
                  'legend.fontsize': 10,  # was 10
                  'xtick.labelsize': 24,
                  'ytick.labelsize': 24,
                  'text.usetex': True,
                  'font.family': 'serif'
                  }
        plt.figure()
        matplotlib.rcParams.update(params)
        z = betx.reshape(len(ptout), len(betx0))

        z = z[:, 1700:]
        long = long[1700:]
        print(np.shape(z))
        print(np.shape(long), np.shape(ptout))

        ax = plt.subplot()
        ax.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)

        plt.contourf(long, ptout, z, 25, cmap='inferno')

        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_x - \beta_{x0} }{\beta_{x0}}$', fontsize=40, usetex=True)
        plt.show()

        plt.figure()
        matplotlib.rcParams.update(params)
        z = bety.reshape(len(ptout), len(betx0))
        z = z[:, 1700:]

        ax = plt.subplot()
        ax.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        plt.contourf(long, ptout, z, 25, cmap='inferno')
        # cbar.set_ticklabels([-10, 10, 200])
        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_y - \beta_{y0} }{\beta_{y0}}$', fontsize=40, usetex=True)
        plt.show()

        plt.figure()
        matplotlib.rcParams.update(params)
        z = dx.reshape(len(ptout), len(betx0))
        ax = plt.subplot()
        ax.set_ylabel(r'$\frac{\Delta P}{P}$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        plt.contourf(long, ptout, z, 25, cmap='inferno')

        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$D_x - D_{x0}$', fontsize=40, usetex=True)
        plt.show()
        # As a function of offset

        plt.figure()
        long = np.array([])
        betx = np.array([])
        bety = np.array([])
        dx = np.array([])
        ptout = np.array([])
        self.madx.select(flag='interpolate')

        twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                alfy=str(-2.1110),
                                deltap=str(-0.000), x=str(0))
        betx0 = twiss.betx
        bety0 = twiss.bety
        dx0 = twiss.dx
        for i in range(81):
            x = -1000e-6 + i * 25e-6
            self.madx.select(flag='interpolate')
            twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                    alfy=str(-2.1110), x=str(x))
            long = twiss.s
            betx = np.append(betx, np.divide(twiss.betx - betx0, betx0))
            bety = np.append(bety, np.divide(twiss.bety - bety0, bety0))
            dx = np.append(dx, np.divide(twiss.dx - dx0, 1))
            ptout = np.append(ptout, x)

        z = betx.reshape(len(ptout), len(betx0))
        ax = plt.subplot()
        ax.set_ylabel(r'$x$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        vmin = -10
        vmax = 200
        logthresh = 10
        maxlog = int(np.ceil(np.log10(vmax)))
        minlog = int(np.ceil(np.log10(-vmin)))
        plt.contourf(long, ptout, z, 40, cmap='inferno')
        tick_locations = (-10, 0, 10, 50, 100, 150)

        cbar = plt.colorbar(ticks=tick_locations, extend='both')
        # cbar.ax.set_yticklabels(['-10', '0', '10', '50', '100', '150'])
        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        # cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_x - \beta_{x0} }{\beta_{x0}}$', fontsize=40, usetex=True)
        plt.show()

        plt.figure()
        z = bety.reshape(len(ptout), len(betx0))
        ax = plt.subplot()
        ax.set_ylabel(r'$x$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        plt.contourf(long, ptout, z, 25, cmap='inferno')

        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_y - \beta_{y0} }{\beta_{y0}}$', fontsize=40, usetex=True)
        plt.show()

        plt.figure()
        long = np.array([])
        betx = np.array([])
        bety = np.array([])
        dx = np.array([])
        ptout = np.array([])
        twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                alfy=str(-2.1110),
                                deltap=str(-0.000))
        betx0 = twiss.betx
        bety0 = twiss.bety
        dx0 = twiss.dx
        for i in range(81):
            y = -1000e-6 + i * 25e-6
            pt = 0
            self.madx.select(flag='interpolate')
            twiss = self.madx.twiss(betx=str(11.3866), alfx=str(-2.1703), dx=str(0), dpx=str(0), bety=str(11.1824),
                                    alfy=str(-2.1110), deltap=str(pt), y=str(y))
            long = twiss.s
            betx = np.append(betx, np.divide(twiss.betx - betx0, betx0))
            bety = np.append(bety, np.divide(twiss.bety - bety0, bety0))
            dx = np.append(dx, np.divide(twiss.dx - dx0, 1))
            ptout = np.append(ptout, y)

        z = betx.reshape(len(ptout), len(betx0))
        ax = plt.subplot()
        ax.set_ylabel(r'$y$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        plt.contourf(long, ptout, z, 25, cmap='inferno')

        # cbar.tick_params(labelsize=38, pad=10)
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_x - \beta_{x0} }{\beta_{x0}}$', fontsize=40, usetex=True)
        plt.show()

        plt.figure()
        z = bety.reshape(len(ptout), len(betx0))
        ax = plt.subplot()
        ax.set_ylabel(r'$y$', fontsize=38, usetex=True)
        ax.set_xlabel("s [m]", fontsize=34, usetex=True)
        z = bety.reshape(len(ptout), len(betx0))
        vmin = -1
        vmax = 8
        logthresh = 0.5
        maxlog = int(np.ceil(np.log10(vmax)))
        minlog = int(np.ceil(np.log10(-vmin)))
        plt.contourf(long, ptout, z, 40, cmap='inferno')
        tick_locations = (-1, 0, 1, 2, 3, 4, 5, 6, 7)
        cbar = plt.colorbar(ticks=tick_locations, extend='both')
        # cbar.ax.set_yticklabels(['-1', '0', '1', '2', '3', '4', '5', '6', '7'])
        spine_color = 'gray'

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(2.5)
            ax.tick_params(width=2.5, direction='out', color=spine_color)

        # cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$\frac{\beta_y - \beta_{y0} }{\beta_{y0}}$', fontsize=40, usetex=True)
        plt.show()
        # Twiss for different deltaP

    def plot_pymoo(self, problem):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(problem.output_all[:, 1], label="$\sigma_x$")
        ax.plot(problem.output_all[:, 2], label="$\sigma_y$")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Beam size [$\mu$m]')
        ax.set_yscale('log')
        ax.legend()
        actors = np.zeros(shape=(len(problem.output_all[:, 1]), 10))
        for i in range(len(problem.output_all[:,0])):
            for j in range(10):
                actors[i, j] = problem.output_all[i, 5][j]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(actors / problem.norm_vect)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Normalised magnet strengths")

    def animate(self, output_all):
        import matplotlib.animation as animation
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        fig = plt.figure(figsize=(10, 6))
        plt.xlim(0, 0.5*10**6)
        plt.ylim(0, 0.5*10**6)
        plt.xlabel('f1', fontsize=20)
        container = []
        def anim(j):
            k = 80 + j * 180
            points = plt.plot(output_all[k+0:180, 3], output_all[k+0:180, 4], 'o')
            container.append(points)
            # p.tick_params(labelsize=17)
            # plt.setp(p.lines, linewidth=7)

        plt.xlabel('f2', fontsize=20)
        for j in range(166):
            anim(j)
        ani = animation.ArtistAnimation(fig, container, interval=50, blit=True,
                                        repeat_delay=1000)
        plt.show()

    def diffTwiss(self):
        """
        Twiss plots with momentum offsets
        """
        # params = {'axes.labelsize': 40,  # fontsize for x and y labels (was 10)
        #           'axes.titlesize': 40,
        #           'legend.fontsize': 40,  # was 10
        #           'xtick.labelsize': 40,
        #           'ytick.labelsize': 40,
        #           'axes.linewidth': 2,
        #           'lines.linewidth': 3,
        #           'text.usetex': True,
        #           'font.family': 'serif'
        #           }
        # plt.rcParams.update(params)
        self.madx.use(sequence='TT43')
        fig, ax = plt.subplots(figsize=(10,8))
        self.madx.select(flag='interpolate', step=0.02)
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k', label=r"$\beta_x$")
        ax.plot(twiss['s'], twiss['bety'], 'r', label=r"$\beta_y$")
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g', label="$D_x$")
        ax2.plot(twiss['s'], twiss['Dy'], 'b', label=r"$D_y$")
        ax2.legend(loc="upper right")
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(-0.002))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0.002))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=38, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta x, \beta y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=38)
        # ax.set_ylim = (-1, 1)
        spine_color = 'k'

        ax.tick_params(labelsize=38, pad=10)
        ax2.tick_params(labelsize=38, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.title('Chromatic effects')
        plt.show()




        fig, ax = plt.subplots(figsize=(9,7), constrained_layout=True)
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), x=str(0))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k', label=r"$\beta_x$")
        ax.plot(twiss['s'], twiss['bety'], 'r', label=r"$\beta_y$")
        ax.legend(loc="upper left", fontsize=34)
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g', label="$D_x$")
        ax2.plot(twiss['s'], twiss['Dy'], 'b', label=r"$D_y$")
        ax2.legend(loc="upper right", fontsize=34)
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), x=str(-0.00025), px=str(-0.00005), y=str(-0.00025), py=str(-0.00005))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), x=str(0.00025), px=str(0.00005), y=str(0.00025), py=str(0.00005))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=44, usetex=True, labelpad=5)
        ax.set_ylabel(r'$\beta_x, \beta_y$ [m]', fontsize=44, usetex=True, labelpad=5)
        ax2.set_ylabel("$D_x, D_y$ [m]", fontsize=44, usetex=True, labelpad=5)
        ax.tick_params(labelsize=38)
        # ax.set_ylim = (-1, 1)
        spine_color = 'k'

        ax.tick_params(labelsize=38, pad=10)
        ax2.tick_params(labelsize=38, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.title('Detuning with amplitude - x')
        plt.show()

        fig, ax = plt.subplots(figsize=(18,15))
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), y=str(0))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k', label=r"$\beta_x$")
        ax.plot(twiss['s'], twiss['bety'], 'r', label=r"$\beta_y$")
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(twiss['s'], twiss['Dx'], 'g', label="$D_x$")
        ax2.plot(twiss['s'], twiss['Dy'], 'b', label=r"$D_y$")
        ax2.legend(loc="upper right")

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), y=str(-500e-6), py=str(-0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')
        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                deltap=str(0), y=str(500e-6), py=str(0.5e-4))
        # print(twiss.betx[-1], twiss.bety[-1])
        ax.plot(twiss['s'], twiss['betx'], 'k--')
        ax.plot(twiss['s'], twiss['bety'], 'r--')

        ax2.plot(twiss['s'], twiss['Dx'], 'g--')
        ax2.plot(twiss['s'], twiss['Dy'], 'b--')

        ax.set_xlabel("s [m]", fontsize=38, usetex=True, labelpad=10)
        ax.set_ylabel(r'$\beta_x, \beta_y$ [m]', fontsize=38, usetex=True, labelpad=10)
        ax2.set_ylabel("$D_x, D_y$ [m]", fontsize=38, usetex=True, labelpad=10)
        ax.tick_params(labelsize=38)
        # ax.set_ylim = (-1, 1)
        spine_color = 'k'

        ax.tick_params(labelsize=38, pad=10)
        ax2.tick_params(labelsize=38, pad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for ax0 in [ax, ax2]:
            for spine in ['left', 'bottom', 'right']:
                ax0.spines[spine].set_color(spine_color)
                ax0.spines[spine].set_linewidth(2.5)
                ax0.tick_params(width=2, direction='out', color=spine_color)
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.title('Detuning with amplitude - y')
        plt.show()

    def MapTable(self):
        """
        Bar charts showing the contribution to the beam size from different orders
        """

        self.madx.use(sequence='TT43', range="TT43$START/MERGE")
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0)
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_normal('maptable', icase=56, no=4)
        twiss = self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0,
                                    file="ptc_twiss_out.out")
        self.madx.write(table='map_table', file='data/ptc_map_ptcnorm.tfs')
        self.madx.ptc_end()

        maptab = pd.read_csv('data/ptc_map_ptcnorm.tfs', header=7, delimiter=' ', usecols=[0, 1, 2, 4],
                             skipinitialspace=True, names=['coef', 'value', 'nv', 'order'])
        # coefficients = maptab.coef
        out_all = np.array([])
        order_all = np.array([])
        ord_all = np.array([])
        sixdim_all = np.array([])
        # print(np.shape(out))
        std = [277.1e-6, 5.779e-5, 272.4e-6, 5.732e-5, 2.09*0.001092, 6.0e-5]
        # try printing some data from newDF

        for idx, line in maptab.iterrows():
            # print(out)
            x = np.empty([0])
            # digits = [int(d) for d in str()]
            sixdim = (str(maptab.iloc[idx][0])[3:])
            sixdim_all = np.append(sixdim_all, sixdim)
            sep_dim = [int(d) for d in str(sixdim)]
            ord_all = np.append(ord_all, int(maptab.iloc[idx][3]))
            for i in range(6):
                x = np.append(x, std[i] ** sep_dim[i])

            out = maptab.iloc[idx][1] * np.prod(x)
            out_all = np.append(out_all, out)
            order_all = np.append(order_all, maptab.iloc[idx][2])
        # print(out_all)
        ens = np.vstack((order_all, out_all))
        x_ens = ens[1, (ens[0, :]) == 1]
        plt.figure()
        color = np.array([(0, 0, 0)] * x_ens.shape[0], dtype='float16')
        color[((ord_all[(ens[0, :]) == 1]) == 1)] = [0, 0.324219, 0.628906]
        color[((ord_all[(ens[0, :]) == 1]) == 2)] = [1, 0.54, 0]
        color[((ord_all[(ens[0, :]) == 1]) == 3)] = [0.25, 0.80, 0.54]
        color[((ord_all[(ens[0, :]) == 1]) == 4)] = [0.597656, 0.398438, 0.996094]

        print(np.sum(x_ens))

        plt.bar(np.linspace(1, np.shape(x_ens)[0], np.shape(x_ens)[0]), x_ens * 1e6, color=color)
        plt.xticks(np.arange(1, np.shape(x_ens)[0] + 1, step=1), sixdim_all[(ens[0, :]) == 1], rotation=90, fontsize=20)
        plt.xlim(0, 70)
        plt.xlabel("Orders", fontsize=28)
        plt.ylabel("1$\sigma$ values [$\mu$m]", fontsize=28)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('$x$', fontsize=30)
        plt.show()

        y_ens = ens[1, (ens[0, :]) == 3]
        print(np.sum(y_ens))

        color = np.array([(0, 0, 0)] * y_ens.shape[0], dtype='float16')
        color[((ord_all[(ens[0, :]) == 3]) == 1)] = [0, 0.324219, 0.628906]
        color[((ord_all[(ens[0, :]) == 3]) == 2)] = [1, 0.54, 0]
        color[((ord_all[(ens[0, :]) == 3]) == 3)] = [0.25, 0.80, 0.54]
        color[((ord_all[(ens[0, :]) == 3]) == 4)] = [0.597656, 0.398438, 0.996094]

        plt.figure()
        plt.bar(np.linspace(1, np.shape(y_ens)[0], np.shape(y_ens)[0]), y_ens * 1e6, color=color)
        plt.xticks(np.arange(1, np.shape(y_ens)[0] + 1, step=1), sixdim_all[(ens[0, :]) == 3], rotation=90, fontsize=20)
        plt.xlim(0, np.shape(y_ens)[0])
        plt.xlabel("Orders", fontsize=28)
        plt.ylabel("1$\sigma$ values [$\mu$m]", fontsize=28)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('$y$', fontsize=30)
        plt.show()

        t_ens = ens[1, (ens[0, :]) == 6]
        color = np.array([(0, 0, 0)] * t_ens.shape[0], dtype='float16')
        color[((ord_all[(ens[0, :]) == 6]) == 1)] = [0, 0.324219, 0.628906]
        color[((ord_all[(ens[0, :]) == 6]) == 2)] = [1, 0.54, 0]
        color[((ord_all[(ens[0, :]) == 6]) == 3)] = [0.25, 0.80, 0.54]
        color[((ord_all[(ens[0, :]) == 6]) == 4)] = [0.597656, 0.398438, 0.996094]

        plt.figure()
        plt.bar(np.linspace(1, np.shape(t_ens)[0], np.shape(t_ens)[0]), t_ens * 1e6, color=color)
        plt.xticks(np.arange(1, np.shape(t_ens)[0] + 1, step=1), sixdim_all[(ens[0, :]) == 6], rotation=90, fontsize=20)
        plt.xlim(0, np.shape(t_ens)[0])
        plt.xlabel("Orders", fontsize=28)
        plt.ylabel("1$\sigma$ values [$\mu$m]", fontsize=28)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('$z$', fontsize=30)
        plt.show()

    def survey(self):
        self.madx.use(sequence='TT43')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        # self.madx.select(flag='interpolate', step=0.05)
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        survey = self.madx.survey(x0=0,
                                  y0=0,
                                  z0=0,
                                  theta0=0,
                                  phi0=0,
                                  psi0=0, file='data/TT40TT41_Survey.tfs')

        fig, ax = plt.subplots()
        plt.axis([-18, 2, -1, 7.5])
        # plt.axis('equal')
        A = np.stack((twiss['s'], twiss['betx'], twiss['bety']), axis=1)
        A = np.sort(A, axis=0)
        # print(A)
        # ax.plot(twiss['s'], np.sqrt(twiss['betx']*6.2e-9), 'k--')
        # ax.plot(twiss['s'], np.sqrt(twiss['bety']*6.2e-9), 'r--')
        x = np.squeeze(np.array([-survey['z'][-1] + survey['z']]))
        y = np.squeeze(np.array([survey['x'][-1] - survey['x']]))
        xt = np.squeeze(np.array([np.sqrt(twiss['betx'] * 6.8e-9 + (0.002 * twiss['dx']) ** 2)]))
        yt = np.squeeze(np.array([np.sqrt(twiss['bety'] * 6.8e-9 + (0.002 * twiss['dy']) ** 2)]))
        theta = np.array(survey['theta'])

        ax.plot(-survey['z'][-1] + survey['z'] + np.squeeze(np.multiply(4, xt, np.sin(theta))),
                -survey['x'][-1] + survey['x'] + np.squeeze(np.multiply(4, xt, np.cos(theta))), 'r-', linewidth=1)
        ax.plot(-survey['z'][-1] + survey['z'] - np.squeeze(np.multiply(4, xt, np.sin(theta))),
                -survey['x'][-1] + survey['x'] - np.squeeze(np.multiply(4, xt, np.cos(theta))), 'r-', linewidth=1)

        ax.plot(-survey['z'][-1] + survey['z'] + np.squeeze(np.multiply(4, yt, np.sin(theta))),
                -survey['x'][-1] + survey['x'] + np.squeeze(np.multiply(4, yt, np.cos(theta))), 'b-', linewidth=1)
        ax.plot(-survey['z'][-1] + survey['z'] - np.squeeze(np.multiply(4, yt, np.sin(theta))),
                -survey['x'][-1] + survey['x'] - np.squeeze(np.multiply(4, yt, np.cos(theta))), 'b-', linewidth=1)

        _ = ax.add_patch(
            matplotlib.patches.Rectangle(
                (-11, -0.05), 10, 0.1, angle=0, facecolor='r', edgecolor='r'))

        _ = ax.add_patch(
            matplotlib.patches.Rectangle(
                (0.0, -0.05), 1, 0.1, angle=0, facecolor='r', edgecolor='r'))

        for idx in range(np.size(survey['l'])):
            if twiss['keyword'][idx] == 'quadrupole':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.4 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.4 * np.sin(theta[idx])),
                        0.5, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.675 * np.cos(-7.5 * np.pi / 180) + 0.25 * np.sin(
                            -7.5 * np.pi / 180),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(-7.5 * np.pi / 180)), 0.75, 0.5,
                        angle=-7.5, facecolor='c', edgecolor='c'))

            elif twiss['keyword'][idx] == 'sextupole':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.175 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.175 * np.sin(theta[idx])),
                        0.2, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='r', edgecolor='r'))

            elif twiss['keyword'][idx] == 'octupole':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.175 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.175 * np.sin(theta[idx])),
                        0.2, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='k'))

            elif twiss['keyword'][idx] == 'monitor':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.155 * np.cos(theta[idx]) + 0.15 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(theta[idx]) - 0.155 * np.sin(theta[idx])),
                        0.15, 0.3, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='m'))
            elif twiss['keyword'][idx] == 'kicker':
                _ = ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.125 * np.cos(theta[idx]) + 0.15 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(theta[idx]) - 0.125 * np.sin(theta[idx])),
                        0.1, 0.3, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='c'))

        ax.plot(-survey['z'][-1] + survey['z'], -survey['x'][-1] + survey['x'], 'k-', linewidth=1)
        ax.set_xlabel("z [m]", fontsize=34, usetex=True)
        ax.set_ylabel("x [m]", fontsize=34, usetex=True)

        axins = zoomed_inset_axes(ax, 2.3, loc=1)

        axins.plot(-survey['z'][-1] + survey['z'] + np.squeeze(np.multiply(4, xt, np.sin(theta))),
                   -survey['x'][-1] + survey['x'] + np.squeeze(np.multiply(4, xt, np.cos(theta))), 'r-', linewidth=1,
                   label='$\pm 4\sigma_x$')
        axins.plot(-survey['z'][-1] + survey['z'] - np.squeeze(np.multiply(4, xt, np.sin(theta))),
                   -survey['x'][-1] + survey['x'] - np.squeeze(np.multiply(4, xt, np.cos(theta))), 'r-', linewidth=1)

        axins.plot(-survey['z'][-1] + survey['z'] + np.squeeze(np.multiply(4, yt, np.sin(theta))),
                   -survey['x'][-1] + survey['x'] + np.squeeze(np.multiply(4, yt, np.cos(theta))), 'b-', linewidth=1,
                   label='$\pm 4\sigma_y$')
        axins.plot(-survey['z'][-1] + survey['z'] - np.squeeze(np.multiply(4, yt, np.sin(theta))),
                   -survey['x'][-1] + survey['x'] - np.squeeze(np.multiply(4, yt, np.cos(theta))), 'b-', linewidth=1)

        _ = axins.add_patch(
            matplotlib.patches.Rectangle(
                (-11, -0.05), 10, 0.1, angle=0, facecolor='r', edgecolor='r'))

        _ = axins.add_patch(
            matplotlib.patches.Rectangle(
                (0.0, -0.05), 1, 0.1, angle=0, facecolor='r', edgecolor='r'))

        for idx in range(np.size(survey['l'])):
            if twiss['keyword'][idx] == 'quadrupole':
                _ = axins.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.4 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.4 * np.sin(theta[idx])),
                        0.5, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='g', edgecolor='g'))
            elif twiss['keyword'][idx] == 'rbend':
                _ = axins.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.675 * np.cos(-7.5 * np.pi / 180) + 0.25 * np.sin(
                            -7.5 * np.pi / 180),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(-7.5 * np.pi / 180)), 0.75, 0.5,
                        angle=-7.5, facecolor='c', edgecolor='c'))

            elif twiss['keyword'][idx] == 'sextupole':
                _ = axins.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.175 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.175 * np.sin(theta[idx])),
                        0.2, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='r', edgecolor='r'))


            elif twiss['keyword'][idx] == 'octupole':
                _ = axins.add_patch(

                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.175 * np.cos(theta[idx]) + 0.25 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.25 * np.cos(theta[idx]) - 0.155 * np.sin(theta[idx])),
                        0.2, 0.5, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='k'))
            elif twiss['keyword'][idx] == 'monitor':
                _ = axins.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.155 * np.cos(theta[idx]) + 0.15 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(theta[idx]) - 0.175 * np.sin(theta[idx])),
                        0.15, 0.3, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='m'))
            elif twiss['keyword'][idx] == 'kicker':
                _ = axins.add_patch(
                    matplotlib.patches.Rectangle(
                        (-survey['z'][-1] + survey['z'][idx] - 0.125 * np.cos(theta[idx]) + 0.15 * np.sin(theta[idx]),
                         -survey['x'][-1] + survey['x'][idx] - 0.15 * np.cos(theta[idx]) - 0.125 * np.sin(theta[idx])),
                        0.1, 0.3, angle=np.float(theta[idx]) * 180 / np.pi, facecolor='w', edgecolor='c'))

        axins.plot(-survey['z'][-1] + survey['z'], -survey['x'][-1] + survey['x'], 'k-', linewidth=1)
        axins.set_xlim(-4, -0.5)  # apply the x-limits
        # axins.set_xlabel("$injection$-$point$", fontsize=20, usetex=True)
        axins.set_ylim(-0.5, 1.5)  # apply the y-limits
        axins.legend()

        #     elif twiss['keyword'][idx] == 'sextupole':
        #         _ = axins.add_patch(
        #             matplotlib.patches.Rectangle(
        #                 (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.9*np.sign(twiss['k2l'][idx]),
        #                 facecolor='b', edgecolor='b'))
        #     elif twiss['keyword'][idx] ==  'octupole':
        #         _ = axins.add_patch(
        #             matplotlib.patches.Rectangle(
        #                 (twiss['s'][idx] - twiss['l'][idx], 0), twiss['l'][idx], 0.8*np.sign(twiss['k3l'][idx]),
        #                 facecolor='r', edgecolor='r'))
        #     elif twiss['keyword'][idx] == 'rbend':
        #         _ = axins.add_patch(
        #             matplotlib.patches.Rectangle(
        #                 (twiss['s'][idx] - twiss['l'][idx], -1), twiss['l'][idx], 2,
        #                 facecolor='g', edgecolor='g'))
        # # ax.plot(-survey['z'][-1] + survey['z'] + 0*np.multiply(xt, np.sin(theta)), survey['x'][-1] - survey['x'] + 0*np.multiply(xt, np.cos(theta)), 'r-')
        plt.show()

    def plasma_focus(self):
        """
        Plot beam distributions
        """

        i = 0
        print('Modelling plasma...')
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        var = []
        f = open('distr/Ellipse_150MeV_nominal.tfs', 'r')  # initialize empty array
        for line in f:
            var.append(
                line.strip().split())
        f.close()

        init_dist = np.array(var[9:])
        init_dist = init_dist[0:5000, 0:6]
        init_dist = init_dist.astype(np.float)
        self.madx.use(sequence='TT43')
        self.madx.select(FLAG='makethin', THICK=True)
        self.madx.makethin(SEQUENCE='TT43', STYLE='teapot')
        # madx.flatten()
        self.madx.use(sequence='TT43')
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        with self.madx.batch():
            self.madx.track(onetable=True, recloss=True, onepass=True)
            for particle in init_dist:
                self.madx.start(x=particle[0], px=particle[3], y=particle[1], py=particle[4],
                                t=1 * particle[2],
                                pt=1 * particle[5])

            self.madx.run(turns=1, maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
            self.madx.endtrack()

        ptc_output = self.madx.table.trackone

        for idx in range(np.size(twiss['l'])):
            if twiss['s'][idx] == twiss['s'][-1]:
                s_EN = twiss['s'][idx]
        idx = np.array(ptc_output.s == s_EN)

        x_out = ptc_output.x[idx]
        y_out = ptc_output.y[idx]
        z_out = ptc_output.t[idx]
        px_out = ptc_output.px[idx]
        py_out = ptc_output.py[idx]
        pz_out = ptc_output.pt[idx]
        # emitx_before = np.sqrt(np.mean(np.multiply(x_out, x_out) * np.mean(np.multiply(px_out, px_out))) - np.multiply(
        #     np.mean(np.multiply(x_out, px_out)), np.mean(np.multiply(x_out, px_out))))
        # print(emitx_before * 150 / 0.511)
        #
        # for idx in range(np.size(twiss['l'])):
        #     if twiss['name'][idx] == 'merge:1':
        #         s_EN = twiss['s'][idx]
        # idx = np.array(ptc_output.s == s_EN)
        # x_out = ptc_output.x[idx]
        # y_out = ptc_output.y[idx]
        # z_out = ptc_output.t[idx]
        # px_out = ptc_output.px[idx]
        # py_out = ptc_output.py[idx]
        # pz_out = ptc_output.pt[idx]
        beam_size_x = np.multiply(self.rmsValue(x_out), 1000000)
        beam_size_y = np.multiply(self.rmsValue(y_out), 1000000)
        beam_size_z = np.multiply(self.rmsValue(z_out), 1000000)
        print(beam_size_x, beam_size_y, beam_size_z)
        # emitx_after = np.sqrt(np.mean(np.multiply(x_out, x_out) * np.mean(np.multiply(px_out, px_out))) - np.multiply(
        #     np.mean(np.multiply(x_out, px_out)), np.mean(np.multiply(x_out, px_out))))
        # print(emitx_after * 150 / 0.511)

    def rmsValue(self, arr):
        n = len(arr)
        square = 0
        # Calculate square
        for i in range(0, n):
            square += (arr[i] ** 2)
            # Calculate Mean
        mean = (square / (float)(n))
        # Calculate Root
        root = np.sqrt(mean)
        return root


    def reject_outliers(self, data):
        ind = abs(data - np.mean(data)) < 5 * np.std(data)
        return data[ind], ind


    def plotheatdp(self):
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[5, 1, 5])

        ax1 = fig.add_subplot(gs[0])

        self.madx.use(sequence='TT43')
        self.madx.select(flag='interpolate', step=0.02)

        twiss_0 = self.madx.twiss(RMATRIX=True, BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0, DELTAP=str(0))
        twiss_dp = np.empty((0, len(twiss_0['BETX'])))
        sigmax_0 = np.sqrt(twiss_0['betx'] * 6.8e-9 + (0.002 * twiss_0['dx']) ** 2)
        for step, dp in enumerate((np.linspace(0.002, -0.002,21))):
            # print(dp)
            twiss = self.madx.twiss(RMATRIX=True, BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0, dpy=0, DELTAP=str(dp))
            # print((twiss_0['BETX']))
            # print((twiss['BETX']))
            # ax.plot(np.divide(twiss['BETX'] -
            twiss_dp = np.vstack((twiss_dp, np.sqrt(twiss['betx'] * 6.8e-9 + ((0.002) * twiss['dx']) ** 2) - sigmax_0))

        #     twiss_dp = np.vstack((twiss['BETX'] - twiss_0['BETX']twiss_dp, twiss['BETX'][:]))
        #     # twiss_dp = np.vstack((twiss_dp, np.divide((twiss['BETX'][:] - twiss_0['BETX'][:]), twiss_0['BETX'][:])))
        # print(twiss_dp)
        ax = sns.heatmap(np.multiply(1, np.divide(twiss_dp, sigmax_0)), yticklabels=np.round(np.linspace(2, -2, 21),3), cmap='inferno', cbar_kws={'label': '$\Delta\sigma_x$ [$\mu m$]'})
        ax.set_ylabel('$\delta_p$ [\%]')
        ax.set_xlabel('$s$ [m]')
        plt.xticks(np.linspace(0,15/0.02,16), np.linspace(0,15,16))
        n = 2  # Keeps every 7th label
        [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]

        ax1 = fig.add_subplot(gs[2])
        twiss_dp = np.empty((0, len(twiss_0['BETY'])))
        sigmay_0 = np.sqrt(twiss_0['bety'] * 6.8e-9 + (0.002 * twiss_0['dy']) ** 2)
        for step, dp in enumerate((np.linspace(0.002, -0.002, 21))):
            # print(dp)
            twiss = self.madx.twiss(RMATRIX=True, BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0,
                                    ALFY=self.alfy0, DY=0, dpy=0, DELTAP=str(dp))
            # print((twiss_0['BETX']))
            # print((twiss['BETX']))
            # ax.plot(np.divide(twiss['BETX'] -
            twiss_dp = np.vstack((twiss_dp, np.sqrt(twiss['bety'] * 6.8e-9 + ((0.002) * twiss['dy']) ** 2) - sigmay_0))

        #     twiss_dp = np.vstack((twiss['BETX'] - twiss_0['BETX']twiss_dp, twiss['BETX'][:]))
        #     # twiss_dp = np.vstack((twiss_dp, np.divide((twiss['BETX'][:] - twiss_0['BETX'][:]), twiss_0['BETX'][:])))
        # print(twiss_dp)
        ax1 = sns.heatmap(np.multiply(1, np.divide(twiss_dp, sigmay_0)),
                         yticklabels=np.round(np.linspace(2, -2, 21), 3), cmap='inferno',  cbar_kws={'label': '$\Delta\sigma_y$ [$\mu m$]'})
        ax1.set_ylabel('$\delta_p$ [\%]')
        ax1.set_xlabel('$s$ [m]')
        plt.xticks(np.linspace(0, 15 / 0.02, 16), np.linspace(0, 15, 16))
        n = 2  # Keeps every 7th label
        [l.set_visible(False) for (i, l) in enumerate(ax1.yaxis.get_ticklabels()) if i % n != 0]
        plt.gcf().subplots_adjust(bottom=0.15)

