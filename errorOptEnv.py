import matplotlib.pyplot as plt
import numpy as np
import get_beam_size
import matplotlib
from scipy.optimize import minimize
import time
from cpymad.madx import Madx


class Error:
    def __init__(self, x_best, x, init_dist, n_particles):
        self.x_best = x_best
        self.rew = 10 ** 20
        self.q = x_best
        self.num_q = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.num_s = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.num_o = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.num_a = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.x = x
        self.dof = self.num_q + self.num_s + self.num_o + self.num_a
        self.x_best_err = np.zeros([1, self.dof])
        self.ii = 0
        self.n_seeds = 1
        self.shot = 0
        self.err_flag = False
        # Vector to normalise actions
        self.norm_vect = [y['norm'] for y in x.values()]
        self.name = [y['name'] for y in x.values()]
        self.init_dist = init_dist
        self.bpm_res = 0e-6 # 10
        self.btv_res = 0e-6 # 1
        self.pos_jit = 0*10e-6
        self.ang_jit = 0*1e-6
        self.corr_err = 0*1e-6
        self.offset_err = 0*1e-6
        self.n_particles = n_particles
        self.name = [y['name'] for y in x.values()]
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        for j in range(self.dof):
            print(self.name[j][0] + self.name[j][-1] + "=" + str(self.q[j]) + ";")
            self.madx.input(self.name[j] + "=" + str(self.q[j]) + ";")

        self.correctors = np.array([], dtype=str)

        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        for idx in range(np.size(twiss['x'][:])):
            if "corr" in twiss['name'][idx]:
                self.correctors = np.append(self.correctors, str(twiss['name'][idx][:-2]))
        self.nom_x_0 = twiss['x']
        self.nom_y_0 = twiss['y']
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                DELTAP=str(-0.002))
        self.nom_x_m1 = twiss['x']
        self.nom_y_m1 = twiss['y']
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                DELTAP=str(0.002))
        self.nom_x_1 = twiss['x']
        self.nom_y_1 = twiss['y']
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                DELTAP=str(0))

        self.waist = twiss['s'][twiss['betx'] == np.min(twiss['betx'][:])]
        bpm = np.zeros(np.shape(twiss['x'][:]))

        for idx in range(np.size(twiss['x'][:])):
            if "bpm" in twiss['name'][idx]:
                # print(twiss['name'][idx])
                bpm[idx] = 1
            elif "merge:1" in twiss['name'][idx]:
                # print(twiss['name'][idx])
                bpm[idx] = 2

        self.bpm = bpm.astype(int)
        self.norm = np.ones((1, np.count_nonzero(self.bpm)))
        self.norm[0][-1] = np.sqrt(50)
        # print(self.norm)
        self.init_dist = init_dist
        x0 = self.init_dist[:, 0]
        px0 = self.init_dist[:, 3]
        y0 = self.init_dist[:, 1]
        py0 = self.init_dist[:, 4]
        pz0 = self.init_dist[:, 5]
        z0 = self.init_dist[:, 2]

        self.emitx_before = np.sqrt(np.mean(np.multiply(x0, x0) * np.mean(np.multiply(px0, px0))) - np.multiply(
            np.mean(np.multiply(x0, px0)), np.mean(np.multiply(x0, px0))))
        self.emity_before = np.sqrt(np.mean(np.multiply(y0, y0) * np.mean(np.multiply(py0, py0))) - np.multiply(
            np.mean(np.multiply(y0, py0)), np.mean(np.multiply(y0, py0))))

        self.betx0 = np.divide(np.mean(np.multiply(x0, x0)), self.emitx_before)
        self.alfx0 = -np.divide(np.mean(np.multiply(x0, px0)), self.emitx_before)
        self.bety0 = np.divide(np.mean(np.multiply(y0, y0)), self.emity_before)
        self.alfy0 = -np.divide(np.mean(np.multiply(y0, py0)), self.emity_before)
        self.twiss_clean = self.ptc_twiss(False)

    def addPowerConv(self, power_jit, seed):
        self.madx.input("power_jit = " + str(power_jit) + ";")
        self.madx.input("ii =" + str(seed + 42) + ";")
        self.madx.call(file='add_errors_jitt.madx')
        self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

    def zeroCorr(self):
        corr_xy = np.zeros([2 * np.size(self.correctors)])
        for num, corr in enumerate(self.correctors):
            self.madx.input(str(corr) + ", VKICK = " + str(
                corr_xy[num + np.size(self.correctors)] + 0 * np.random.normal()) + ", HKICK = " + str(
                corr_xy[num] + 0 * np.random.normal()) + ";")

    def switch_off_higher_order(self):
        for j in range(self.dof):
            if (self.x[j]["type"] == "sextupole") or (self.x[j]["type"] == "octupole"):
                self.madx.input(self.name[j] + "=" + str(0) + ";")
        self.ptc_twiss(True)

    def switch_on_higher_order(self):
        for j in range(self.dof):
            if (self.x[j]["type"] == "sextupole") or (self.x[j]["type"] == "octupole"):
                self.madx.input(self.name[j] + "=" + str(self.q[j]) + ";")
                # print(self.name[j] + "=" + str(self.q[j]) + ";")
        self.ptc_twiss(True)

    def addError(self, seed):
        self.madx.use(sequence='TT43')
        self.zeroCorr()
        self.madx.input("ii =" + str(seed) + ";")
        self.madx.call(file='add_errors.madx')
        self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

    def calcOffsets(self, parameter):
        self.madx.use(sequence='TT43')
        # np.random.seed(0)
        # x0 = np.random.normal()*10**-5
        # np.random.seed(1)
        # y0 = np.random.normal()*10**-5
        # np.random.seed(2)
        # px0 = np.random.normal()*10**-5
        # np.random.seed(3)
        # py0 = np.random.normal()*10**-5
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

        # self.magnets = np.array(["mqawd.0:1", "mqawd.4:1", "mqawd.2:1", "mqawd.9:1", "mqawd.6:1", "mqawd.10:1", "mqawd.14:1", "mqawd.11:1"])
        # self.magnets = np.array(
        #     ["sd3:1", "sd1:1", "sd5:1", "sd2:1", "sd6:1", "sd4:1"])
        # self.magnets = np.array(["oct6:1", "oct7:1", "oct8:1", "oct11:1"])
        self.magnets = np.array(["mbawh.8", "mbawh.3"])
        self.magnets = np.transpose(self.magnets)
        magnets_to_offset = [any(b in s.lower() for b in self.magnets) for s in twiss["name"]]
        strengths = twiss['k0l'][magnets_to_offset]
        print(strengths)
        init_dist = self.init_dist
        init_dist_0 = init_dist - np.mean(init_dist, axis=0)
        x_particle = np.zeros(len(init_dist_0))
        y_particle = np.zeros(len(init_dist_0))
        for j, particle in enumerate(init_dist_0):
            # print(j)
            x0 = -particle[0] + 0e-6
            y0 = particle[1] + 0e-6
            px0 = -particle[3] + 0e-6
            py0 = particle[4] + 0e-6
            for i in range(len(self.magnets)):
                twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0,
                                        dpy=0, X=x0,
                                        Y=y0, PX=px0, PY=py0)
                x_offsets = twiss['x'][magnets_to_offset]
                self.madx.input("eoption, add = false, seed := " + str(i) + ";")
                # self.madx.input("Select, flag = ERROR, clear;")
                self.madx.input("Select, flag = ERROR, class = " + str(self.magnets[i][:-2]) + ";")
                self.madx.input("error_qu := " + str(-parameter * abs(x_offsets[i] ** 3) * strengths[i]) + ";")
                # print("error_qu := " + str(parameter * self.initial_x_offset[i]) + ";")

                self.madx.input("efcomp, order:=2, radius:=" + str(
                    x_offsets[i]) + ", dkn:={error_qu,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0};")
                # print("efcomp, order:=1, radius:=" + str(
                #     x_offsets[i]) + ", dknr:={0,error_qu,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};")
                self.madx.input("Select, flag = error, clear;")
                twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0,
                                        dpy=0, X=x0,
                                        Y=y0, PX=px0, PY=py0)
                y_offsets = twiss['y'][magnets_to_offset]
                # print(y_offsets)
                x_particle[j] = twiss['x'][twiss['name'] == "merge:1"]
                # print(x_particle[j])
                self.madx.input("eoption, add = false, seed := " + str(i + 100) + ";")
                # self.madx.input("Select, flag = ERROR, clear;")
                self.madx.input("Select, flag = ERROR, class = " + str(self.magnets[i][:-2]) + ";")
                self.madx.input("error_qu := " + str(-parameter * abs(y_offsets[i] ** 3) * strengths[i]) + ";")
                # print("error_qu := " + str(parameter * y_offsets[i]) + ";")
                self.madx.input("efcomp, order:=2, radius:=" + str(
                    y_offsets[i]) + ", dkn:={ error_qu,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};")
                # self.madx.input("select, flag = error, clear;")
                self.madx.input("select, flag = error, class = quadrupole;")
                self.madx.input("select, flag = error, class = rbend;")
                #
                self.madx.input("select, flag = error, class = oct;")
                # # self.madx.input("select, flag = error, class = monitor;")
                self.madx.input("select, flag = error, class = sext;")
                self.madx.input("esave;")
                y_particle[j] = twiss['y'][twiss['name'] == "merge:1"]
        errors = self.madx.table["efield"]["name"]
        print(x_offsets)
        print(errors)

        print(np.std(x_particle * 10 ** 6))
        print(np.std(y_particle * 10 ** 6))

        # self.madx.ptc_create_universe()
        # self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        # self.madx.ptc_setswitch(fringe=True)
        # self.madx.ptc_align()
        # self.madx.ptc_observe(place='MERGE')
        # # self.madx.input("ii =" + str(seed) + ";")
        #
        # with self.madx.batch():
        #
        #     for particle in init_dist_0:
        #         self.madx.ptc_start(x=-particle[0]+x0, px=-particle[3]+px0,
        #                             y=particle[1]+y0, py=particle[4]+py0,
        #                             t=1 * particle[2],
        #                             pt=2.09 * particle[5])
        #     self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
        #                         maxaper=[0.03, 0.03, 0.03, 0.03, 1.0, 1])
        #     self.madx.ptc_track_end()
        # ptc_output = self.madx.table.trackone
        # twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
        #                         dpy=0)
        # # twiss = self.ptc_twiss(False)
        # if any(twiss['name'][:] == 'merge:1'):
        #     for idx in range(np.size(twiss['name'])):
        #         if twiss['name'][idx] == 'merge:1':
        #             s_foil = twiss['s'][idx]
        #     idx_temp = np.array(ptc_output.s == s_foil)
        #
        #     x_0 = self.madx.table.trackone['x'][idx_temp]
        #     y_0 = self.madx.table.trackone['y'][idx_temp]
        #     offset_pt = np.mean(self.madx.table.trackone['pt'][idx_temp])
        #
        #     bsx = np.multiply(np.std(x_0), 1000000)
        #     bsy = np.multiply(np.std(y_0), 1000000)
        # print(bsx, bsy)

    def addHomogeneityError(self):
        pass

    def QuadShunt(self, gain, QuadScanCombo):
        """
        Vary quadrupole strength and use this to estimate the offset of the quadrupole
        """
        print('$$$$$$$$$$$$$$$           Quad shunt            $$$$$$$$$$')

        quads = QuadScanCombo[:, 1]
        bpms = QuadScanCombo[:, 0]
        strengths = QuadScanCombo[:, 2]
        np.random.seed(self.shot)
        x_jit = np.random.normal(scale=self.pos_jit, size=4)[0]
        y_jit = np.random.normal(scale=self.pos_jit, size=4)[1]
        px_jit = np.random.normal(scale=self.ang_jit, size=4)[2]
        py_jit = np.random.normal(scale=self.ang_jit, size=4)[3]
        self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                        X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

        twiss = self.ptc_twiss(True)
        names = twiss['name']
        names = [x[:-2] for x in names]
        names2 = np.array([y + ":1" for y in names])
        bpm_ind = [any(b in s.lower() for b in bpms) for s in names2]
        quad_ind = [any(b in s.lower() for b in quads) for s in names2]

        for idx3 in range(len(bpm_ind)):
            if bpm_ind[idx3] == True:
                bpm_ind[idx3 + 1] = False
            if quad_ind[idx3] == True:
                quad_ind[idx3 + 1] = False
        strengths_all = np.divide(twiss['k1l'][quad_ind], twiss['l'][quad_ind])

        bpm_x_before = np.zeros(shape=(len(quads)))
        bpm_y_before = np.zeros(shape=(len(quads)))
        quad_x_before = np.zeros(shape=(len(quads)))
        quad_y_before = np.zeros(shape=(len(quads)))
        bpm_x_after = np.zeros(shape=(len(quads)))
        bpm_y_after = np.zeros(shape=(len(quads)))
        quad_x_after = np.zeros(shape=(len(quads)))
        quad_y_after = np.zeros(shape=(len(quads)))
        for idx in range(len(bpms)):
            twiss = self.ptc_twiss(True)
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"

            if idx < len(bpms) - 1:
                bpm_x_before[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][
                                                                                        bpm_ind][
                                                                                        idx]))
                bpm_y_before[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][bpm_ind][
                                                                                        idx]))
            else:
                bpm_x_before[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][
                                                                                        bpm_ind][
                                                                                        idx]))
                bpm_y_before[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][bpm_ind][
                                                                                        idx]))

        for idx in range(len(bpms)):
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"

            strength = np.divide(twiss['k1l'][quad_ind][idx], twiss['l'][quad_ind][idx])
            twiss = self.ptc_twiss(True)
            # print(twiss['x'])
            # print(twiss['name'])
            quad_x_before[idx] = twiss['x'][quad_ind][idx]
            quad_y_before[idx] = twiss['y'][quad_ind][idx]

            if quads[idx] == "mqawd.9":
                self.madx.input("quad5=" + str(0) + ";")
                twiss = self.ptc_twiss(True)
            if quads[idx] == "mqawd.14":
                self.madx.input("quad1=" + str(0) + ";")
                twiss = self.ptc_twiss(True)
            twiss = self.ptc_twiss(True)
            the_quad = quads[idx] + ":1"
            the_bpm = bpms[idx] + ":1"

            betax_i = twiss['betx'][twiss["name"] == the_quad][0]
            betay_i = twiss['bety'][twiss["name"] == the_quad][0]

            betax_j = twiss['betx'][twiss["name"] == the_bpm][0]
            betay_j = twiss['bety'][twiss["name"] == the_bpm][0]
            mux_i = np.multiply(twiss['mu1'][twiss["name"] == the_quad][0], 2 * np.pi)
            muy_i = np.multiply(twiss['mu2'][twiss["name"] == the_quad][0], 2 * np.pi)
            mux_j = np.multiply(twiss['mu1'][twiss["name"] == the_bpm][0], 2 * np.pi)
            muy_j = np.multiply(twiss['mu2'][twiss["name"] == the_bpm][0], 2 * np.pi)
            Ax_ij = -np.sqrt(np.multiply(betax_i, betax_j)) * np.sin(mux_i - mux_j)
            Ay_ij = -np.sqrt(np.multiply(betay_i, betay_j)) * np.sin(muy_i - muy_j)

            strength = np.divide(twiss['k1l'][twiss["name"] == the_quad][0], twiss['l'][twiss["name"] == the_quad][0])
            offset_x = np.zeros(shape=(2, 2))
            offset_y = np.zeros(shape=(2, 2))

            for idx2, i in enumerate([0.8 * strength, 1 * strength]):
                self.madx.input(str(strengths[idx]) + "=" + str(i) + ";")
                twiss = self.ptc_twiss(True)
                offset_x[idx2, 1] = twiss['x'][twiss["name"] == the_bpm][0] + np.random.normal(
                    scale=self.bpm_res)
                offset_x[idx2, 0] = i
                offset_y[idx2, 1] = twiss['y'][twiss["name"] == the_bpm][0] + np.random.normal(
                    scale=self.bpm_res)
                offset_y[idx2, 0] = i
                self.shot += 1
                print("shot = " + str(self.shot))

            unnorm_offset_x = -np.divide(np.arctan(np.divide((offset_x[0, 1] - offset_x[1, 1]), Ax_ij))
                                         , (0.2 * strength * twiss['l'][twiss["name"] == the_quad][0]))
            guess_x = -(unnorm_offset_x) * 10 ** 6
            unnorm_offset_y = -np.divide(np.arctan(np.divide((offset_y[0, 1] - offset_y[1, 1]), Ay_ij))
                                         , (0.2 * strength * twiss['l'][twiss["name"] == the_quad][0]))
            guess_y = (unnorm_offset_y) * 10 ** 6

            twiss = self.OffsetQuadrupole(quad_temp, "y", np.multiply(gain, guess_y))
            twiss = self.OffsetQuadrupole(quad_temp, "x", np.multiply(gain, guess_x))
            self.shot += 1
            print("shot = " + str(self.shot))
            if quads[idx] == "mqawd.9":
                self.madx.input("quad5=" + str(strengths_all[idx + 1]) + ";")
                twiss = self.ptc_twiss(True)
            if quads[idx] == "mqawd.14":
                self.madx.input("quad1=" + str(strengths_all[idx + 1]) + ";")
                twiss = self.ptc_twiss(True)

            # self.madx.input(str(strengths[idx]) + "=" + str(0) + ";")

            if idx < len(bpms) - 1:
                bpm_x_after[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                               size=np.shape(
                                                                                   twiss['x'][
                                                                                       bpm_ind][
                                                                                       idx]))
                bpm_y_after[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                               size=np.shape(
                                                                                   twiss['x'][bpm_ind][
                                                                                       idx]))
            else:
                bpm_x_after[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                               size=np.shape(
                                                                                   twiss['x'][
                                                                                       bpm_ind][
                                                                                       idx]))
                bpm_y_after[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                               size=np.shape(
                                                                                   twiss['x'][bpm_ind][
                                                                                       idx]))

            twiss = self.ptc_twiss(True)
            quad_x_after[idx] = twiss['x'][quad_ind][idx]
            quad_y_after[idx] = twiss['y'][quad_ind][idx]

            print("True QUAD-BPM offset = " + str(np.multiply(10 ** 6, quad_x_before[idx])))
            print("Guess at QUAD-BPM offset = " + str(np.round(guess_x, 4)) + " um at quad " + str(quads[idx]))
            print("True QUAD-BPM offset = " + str(np.multiply(10 ** 6, quad_y_before[idx])))
            print("Guess at QUAD-BPM offset = " + str(np.round(guess_y, 4)) + " um at quad " + str(quads[idx]))

        bpm_pos = twiss['s'][bpm_ind]
        return quad_x_before, quad_y_before, quad_x_after, quad_y_after, bpm_x_before, bpm_y_before, bpm_x_after, bpm_y_after, bpm_pos, names

    def quadshunt_plot(self, n_iter, gain):
        print("$$$$$$$$$$$$$$        Quad Shunt        $$$$$$$$$$$$$$$$$")
        QuadScanCombo = np.array([("bpm.0", "mqawd.0", "quad0"),
                                  ("bpm.1", "mqawd.4", "quad4"),
                                  ("bpm.2", "mqawd.2", "quad2")])

        QuadScanCombo2 = np.array([("bpm.3", "mqawd.9", "quad1"),
                                   ("bpm.4", "mqawd.6", "quad5")])

        QuadScanCombo3 = np.array([("bpm.6", "mqawd.10", "quad3"),
                                   ("bpm.7", "mqawd.14", "quad5"),
                                   ("merge", "mqawd.11", "quad1")])


        self.switch_off_higher_order()
        quad_x_before, quad_y_before, quad_x_afterx, quad_y_afterx, bpm_x_before, bpm_y_before, bpm_x_afterx, bpm_y_afterx, _, names = self.QuadShunt(
            gain, np.vstack((QuadScanCombo, QuadScanCombo2, QuadScanCombo3)))

        for j in range(n_iter):
            bpm_pos = []
            for quads in [QuadScanCombo, QuadScanCombo2, QuadScanCombo3]:
                self.shot += 1
                print("shot = " + str(self.shot))
                if j == 0:
                    print(j)
                    quad_x_before0, quad_y_before0, quad_x_after0, quad_y_after0, bpm_x_before0, bpm_y_before0, bpm_x_after0, bpm_y_after0, bpm_pos0, names0 = self.QuadShunt(
                        gain, quads)
                else:
                    print(j)
                    _, _, quad_x_after0, quad_y_after0, _, _, bpm_x_after0, bpm_y_after0, bpm_pos0, names0 = self.QuadShunt(
                        gain, quads)
                if np.size(bpm_pos) == 0:
                    bpm_pos = bpm_pos0
                    quad_x_after = quad_x_after0
                    quad_y_after = quad_y_after0
                    bpm_x_after = bpm_x_after0
                    bpm_y_after = bpm_y_after0
                else:
                    bpm_pos = np.hstack((bpm_pos, bpm_pos0))
                    quad_x_after = np.hstack((quad_x_after, quad_x_after0))
                    quad_y_after = np.hstack((quad_y_after, quad_y_after0))
                    bpm_x_after = np.hstack((bpm_x_after, bpm_x_after0))
                    bpm_y_after = np.hstack((bpm_y_after, bpm_y_after0))

        self.switch_on_higher_order()

        return bpm_x_before, bpm_x_after, bpm_y_before, bpm_y_after, quad_x_before, quad_x_after, quad_y_before, quad_y_after, bpm_pos

    def SextOptScan(self, magnets):
        DFSCombo = np.array([("bpm.0", "corr.9", "mqawd.0", "quad0"),
                             ("bpm.1", "corr.0", "mqawd.4", "quad4"),
                             ("bpm.2", "corr.1", "mqawd.2", "quad2"),
                             ("bpm.3", "corr.2", "mqawd.9", "quad1"),
                             ("bpm.4", "corr.3", "mqawd.6", "quad5"),
                             ("bpm.6", "corr.4", "mqawd.10", "quad3"),
                             ("bpm.7", "corr.6", "mqawd.14", "quad5"),
                             ("merge", "corr.7", "mqawd.11", "quad1")])
        # ("merge", "corr.8", "merge", "none")])
        twiss = self.ptc_twiss(False)
        bpms = DFSCombo[:, 0]
        names = twiss['name']
        names = [x[:-2] for x in names]

        bpm_ind = [any(b in s.lower() for b in bpms) for s in names]
        twiss = self.ptc_twiss(False)
        # self.magnets = np.array(["mqawd.0:1", "mqawd.4:1", "mqawd.2:1", "mqawd.9:1", "mqawd.6:1", "mqawd.10:1", "mqawd.14:1", "mqawd.11:1"])
        # self.magnets = np.array(
        #     ["sd3:1", "sd1:1", "sd5:1", "sd2:1", "sd6:1", "sd4:1", "oct8:1", "oct7:1", "oct6:1", "oct11:1"])
        self.magnets = np.transpose(magnets)

        magnets_to_offset = [any(b in s.lower() for b in self.magnets) for s in self.madx.table["efield"]["name"]]
        self.initial_x_offset = self.madx.table["efield"]["dx"][magnets_to_offset]
        self.initial_y_offset = self.madx.table["efield"]["dy"][magnets_to_offset]
        self.track()

        # self.magnets = np.empty(shape=(len(self.x)), dtype=str)
        # idx = 0
        # for key, val in self.x.items():
        #     if idx == 0:
        #         self.magnets = val['name'] + ":1"
        #     else:
        #         self.magnets = np.hstack((self.magnets, val['name'] + ":1"))

        # print((val['name'] + ":1"))
        # idx = idx + 1

        offsets_x = np.zeros(len(self.magnets))
        offsets_y = np.zeros(len(self.magnets))
        bpm_x_after = np.zeros(shape=(len(bpms)))
        bpm_y_after = np.zeros(shape=(len(bpms)))
        np.hstack((offsets_x, offsets_y))
        self.counter = 0
        self.x_best_err = np.zeros([1, self.dof])
        self.rew = 10 ** 20
        solution = minimize(self.step, np.hstack((offsets_x, offsets_y)), method='Nelder-Mead',
                            options={'maxiter': 75})
        offsets_x = self.x_best_err[0:int(len(self.x_best_err) / 2)]
        offsets_y = self.x_best_err[int(len(self.x_best_err) / 2) - 1:-1]
        print(offsets_x, offsets_y)

        for i in range(len(offsets_x)):
            self.OffsetMagnet(self.magnets[i], offsets_x[i] + self.initial_x_offset[i],
                              offsets_y[i] + self.initial_y_offset[i])
        self.track()
        twiss = self.ptc_twiss(False)
        for idx in range(len(bpms)):
            bpm_x_after[idx] = twiss['x'][bpm_ind][idx]
            bpm_y_after[idx] = twiss['y'][bpm_ind][idx]
        return bpm_x_after, bpm_y_after

    def OctOptScan(self):
        twiss = self.ptc_twiss(False)
        # self.magnets = np.array(["mqawd.0:1", "mqawd.4:1", "mqawd.2:1", "mqawd.9:1", "mqawd.6:1", "mqawd.10:1", "mqawd.14:1", "mqawd.11:1"])
        self.magnets = np.array(
            ["oct8:1", "oct7:1", "oct6:1", "oct11:1"])
        self.magnets = np.transpose(self.magnets)

        magnets_to_offset = [any(b in s.lower() for b in self.magnets) for s in self.madx.table["efield"]["name"]]
        self.initial_x_offset = self.madx.table["efield"]["dx"][magnets_to_offset]
        self.initial_y_offset = self.madx.table["efield"]["dy"][magnets_to_offset]
        self.track()

        # self.magnets = np.empty(shape=(len(self.x)), dtype=str)
        # idx = 0
        # for key, val in self.x.items():
        #     if idx == 0:
        #         self.magnets = val['name'] + ":1"
        #     else:
        #         self.magnets = np.hstack((self.magnets, val['name'] + ":1"))

        # print((val['name'] + ":1"))
        # idx = idx + 1

        offsets_x = np.zeros(len(self.magnets))
        offsets_y = np.zeros(len(self.magnets))
        np.hstack((offsets_x, offsets_y))
        self.counter = 0
        solution = minimize(self.step, np.hstack((offsets_x, offsets_y)), method='Nelder-Mead',
                            options={'maxiter': 100})

        self.track()

    def OptScanMO(self):
        from pymoo.model.problem import Problem
        from pymoo.algorithms.nsga2 import NSGA2
        from pymoo.algorithms.so_genetic_algorithm import GA
        from pymoo.factory import get_sampling, get_crossover, get_mutation
        from pymoo.optimize import minimize
        from pymoo.util.termination.default import MultiObjectiveDefaultTermination
        from pymoo.util.termination.default import SingleObjectiveDefaultTermination
        from pymoo.visualization.scatter import Scatter
        from MO_env import kOptEnv

        twiss = self.ptc_twiss(False)
        # self.magnets = np.array(["mqawd.0:1", "mqawd.4:1", "mqawd.2:1", "mqawd.9:1", "mqawd.6:1", "mqawd.10:1", "mqawd.14:1", "mqawd.11:1"])
        self.magnets = np.array(
            ["sd3:1", "sd1:1", "sd5:1", "sd2:1", "sd6:1", "sd4:1"])
        self.magnets = np.transpose(self.magnets)

        magnets_to_offset = [any(b in s.lower() for b in self.magnets) for s in self.madx.table["efield"]["name"]]
        self.initial_x_offset = self.madx.table["efield"]["dx"][magnets_to_offset]
        self.initial_y_offset = self.madx.table["efield"]["dy"][magnets_to_offset]
        self.track()

        # self.magnets = np.empty(shape=(len(self.x)), dtype=str)
        # idx = 0
        # for key, val in self.x.items():
        #     if idx == 0:
        #         self.magnets = val['name'] + ":1"
        #     else:
        #         self.magnets = np.hstack((self.magnets, val['name'] + ":1"))

        # print((val['name'] + ":1"))
        # idx = idx + 1

        offsets_x = np.zeros(len(self.magnets))
        offsets_y = np.zeros(len(self.magnets))
        x_0 = np.concatenate((offsets_x, offsets_y))
        norm_vect = (100 * 10 ** -6) * np.ones((1, len(x_0)))
        x_n = np.divide(x_0, norm_vect)
        parameters = ['offset',
                      'beamsize']
        magnets = self.magnets
        target_values = [0,
                         0]
        init_dist = self.init_dist
        n_obj = len(parameters)
        weights = np.ones((1, n_obj))
        initial_x_offset = self.madx.table["efield"]["dx"][magnets_to_offset]
        initial_y_offset = self.madx.table["efield"]["dy"][magnets_to_offset]
        bpm_res = 0
        max_values = target_values * 10

        class MatchingProblem(kOptEnv, Problem):
            def __init__(self,
                         norm_vect,
                         target_values,
                         n_var=len(x_0),
                         n_obj=n_obj,
                         n_constr=0,
                         xl=None,
                         xu=None):

                kOptEnv.__init__(self, norm_vect, weights,
                                 target_values, magnets, initial_x_offset, initial_y_offset, bpm_res, init_dist)
                Problem.__init__(self,
                                 n_var=n_var,
                                 n_obj=n_obj,
                                 n_constr=n_constr,
                                 xl=-np.ones(np.shape(norm_vect))[0],
                                 xu=np.ones(np.shape(norm_vect))[0])

            def _evaluate(self, x_n, out, *args, **kwargs):
                f = []
                # x_u = np.multiply(x_n, norm_vect)
                for j in range(x_n.shape[0]):
                    y_raw_all, y_raw_single = kOptEnv.step(self, x_n[j, :])

                    if self.n_obj == 1:
                        f.append(y_raw_single)
                    else:
                        f.append(y_raw_all)

                out["F"] = np.vstack(f)
                print(out)

        problem = MatchingProblem(norm_vect=norm_vect,
                                  target_values=target_values,
                                  n_var=len(x_0),
                                  n_obj=n_obj,
                                  n_constr=0,
                                  xl=-np.ones(np.shape(norm_vect))[0],
                                  xu=np.ones(np.shape(norm_vect))[0])

        # problem.evaluate([0, 0])

        algorithm = NSGA2(
            pop_size=30,
            n_offsprings=30,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        termination = MultiObjectiveDefaultTermination(
            x_tol=1e-8,
            cv_tol=1e-6,
            f_tol=1e-7,
            nth_gen=5,
            n_last=30,
            n_max_gen=100,
            n_max_evals=3000
        )

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=False,
                       verbose=True)

        ps = problem.pareto_set(use_cache=False, flatten=False)
        pf = problem.pareto_front(use_cache=False, flatten=False)
        # plt.figure()
        # plt.plot(res.F, color="red")
        params = {'text.usetex': False,
                  'font.family': 'serif'
                  }
        matplotlib.rcParams.update(params)
        # plot = Scatter(title="Design Space", axis_labels="x")
        # plot.add(res.X, s=30, facecolors='none', edgecolors='r')
        # if ps is not None:
        #     plot.add(ps, plot_type="line", color="black", alpha=0.7)
        # plot.do()
        # plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
        # plot.apply(lambda ax: ax.set_ylim(-2, 2))
        # plot.show()

        # Objective Space
        plot = Scatter(title="Objective Space")
        plot.add(res.F)
        if pf is not None:
            plot.add(pf, plot_type="line", color="black", alpha=0.7)
        plot.show()

    def step(self, offsets):
        t0 = time.time()
        self.counter = self.counter + 1
        print("iter = " + str(self.counter))
        # print(self.magnets)
        offsets_x = offsets[0:int(len(offsets) / 2)]
        offsets_y = offsets[int(len(offsets) / 2) - 1:-1]
        # print(offsets_x, offsets_y)

        for i in range(len(offsets_x)):
            self.OffsetMagnet(self.magnets[i], offsets_x[i] + self.initial_x_offset[i],
                              offsets_y[i] + self.initial_y_offset[i])
            # self.OffsetMagnet(self.magnets[i], "y", offsets_y[i] + self.initial_y_offset[i])

        beam_size_x, beam_size_y, offset_x, offset_y = self.track()

        self.madx.input("delete, table = trackone;")
        self.madx.input("delete, table = trackloss;")
        self.madx.input("delete, table = tracksumm;")

        output = 10*abs((beam_size_x-3*5.75) + self.bpm_res * np.random.normal()) + 10*abs(
            (beam_size_y-3*5.75) + self.bpm_res * np.random.normal()) + abs(offset_x + self.bpm_res * np.random.normal()) + abs(
            offset_y + self.bpm_res * np.random.normal())
        print("beamsize x, y = " + str(beam_size_x) + ", " + str(beam_size_y) + ", " + str(offset_x) + ", " + str(
            offset_y))
        print("cost = " + str(output))

        # If objective function is best so far, update x_best with new best parameters
        if output < self.rew:
            self.x_best_err = offsets
            self.rew = output
        print("best = " + str(self.rew))
        print("---------------------------------")
        t1 = time.time()
        print("time taken = " + str((t1 - t0)))
        if self.counter % 50 == 0:
            self.err_flag = True
        if self.err_flag == True:
            self.reset()
        return output

    def SextScan(self, seed):
        twiss = self.ptc_twiss(False)
        self.track()
        SextScanCombo = np.array([("bpm.3:1", "sd3:1", "sext0"),
                                  ("bpm.4:1", "sd1:1", "sext4"),
                                  ("bpm.5:1", "sd5:1", "sext1"),
                                  ("bpm.6:1", "sd2:1", "sext5"),
                                  ("bpm.9:1", "sd6:1", "sext2"),
                                  ("merge:1", "sd4:1", "sext3")])

        Sexts = SextScanCombo[:, 1]
        bpms = SextScanCombo[:, 0]
        Strs = SextScanCombo[:, 2]
        sext_strengths = np.zeros(len(Sexts))
        twiss = self.ptc_twiss(True)
        names = twiss['name']
        names = [x[:-2] for x in names]
        names2 = [y + ":1" for y in names]
        bpm_ind = [any(b in s.lower() for b in bpms) for s in names2]
        sext_ind = [any(b in s.lower() for b in Sexts) for s in names2]
        i = 0
        for j in range(self.dof):
            if (self.x[j]["type"] == "sextupole") & ((self.x[j]["name"] in Strs)):
                sext_strengths[i] = self.q[j]
                i = i + 1

        for idx3 in range(len(bpm_ind)):
            if bpm_ind[idx3] == True:
                bpm_ind[idx3 + 1] = False
            if sext_ind[idx3] == True:
                sext_ind[idx3 + 1] = False

        for idx in range(len(Sexts)):
            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dx"]))
            print(errors)
            # self.switch_off_higher_order()
            # self.madx.input(str(Strs[idx]) + "=" + str(sext_strengths[idx]) + ";")
            twiss = self.ptc_twiss(True)
            sext_temp = str(Sexts[idx])
            print("x, " + str(sext_temp))
            ScanValues = np.zeros((21, 5))
            self.OffsetSextupole(sext_temp, "x", -110)

            for i in range(21):
                self.OffsetSextupole(sext_temp, "x", 10)

                self.switch_off_higher_order()
                self.madx.input(str(Strs[idx]) + "=" + str(sext_strengths[idx]) + ";")
                twiss = self.ptc_twiss(True)
                # np.random.seed(i + 42 + seed)

                BeamBPMOffset = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res)
                # print(BeamBPMOffset)
                self.switch_on_higher_order()
                beam_size_x, beam_size_y, offset_x, offset_y, _, _ = self.track()
                ScanValues[i, 0] = i * 10 - 100
                ScanValues[i, 1] = BeamBPMOffset
                ScanValues[i, 2] = beam_size_x
                ScanValues[i, 3] = beam_size_y
                ScanValues[i, 4] = BeamBPMOffset
                # ScanValues[i, 5] = offset_y

            p = np.polyfit(ScanValues[:, 0], ScanValues[:, 2], 4)
            x_fitted = np.linspace(-100, 100, 2001)
            y_fitted = np.polyval(p, x_fitted)

            if np.absolute(x_fitted[np.argmax(y_fitted)]) < np.absolute(x_fitted[np.argmin(y_fitted)]):
                self.OffsetSextupole(sext_temp, "x", x_fitted[np.argmax(y_fitted)] - 100)
            else:
                self.OffsetSextupole(sext_temp, "x", x_fitted[np.argmin(y_fitted)] - 100)

            self.track()

    def QuadScan(self):
        QuadScanCombo = np.array([("bpm.0:1", "mqawd.0:1", "quad0"),
                                  ("bpm.1:1", "mqawd.4:1", "quad4"),
                                  ("bpm.2:1", "mqawd.2:1", "quad2"),
                                  ("bpm.3:1", "mqawd.9:1", "quad1"),
                                  ("bpm.4:1", "mqawd.6:1", "quad5"),
                                  ("bpm.6:1", "mqawd.10:1", "quad3"),
                                  ("bpm.7:1", "mqawd.14:1", "quad5"),
                                  ("bpm.8:1", "mqawd.11:1", "quad1")])

        quads = QuadScanCombo[:, 1]
        bpms = QuadScanCombo[:, 0]
        Strs = QuadScanCombo[:, 2]
        # self.madx.call(file='add_errors.madx')
        self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        twiss = self.ptc_twiss(False)
        names = twiss['name']
        names = [x[:-2] for x in names]
        names2 = [y + ":1" for y in names]
        bpm_ind = [any(b in s.lower() for b in bpms) for s in names2]
        quad_ind = [any(b in s.lower() for b in quads) for s in names2]
        # for idx3 in range(len(bpm_ind)):
        #     if bpm_ind[idx3]==True:
        #         bpm_ind[idx3+1]=False
        #     if quad_ind[idx3] == True:
        #         quad_ind[idx3+1]=False
        errors = np.vstack((self.madx.table["efield"]["name"],
                            self.madx.table["efield"]["dx"]))

        for idx in range(len(quads)):
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dx"]))

            Strength = twiss['k1l'][quad_ind][idx]
            quad_temp = str(quads[idx])
            bpm_temp = str(bpms[idx])  #
            # print("x, " + str(quad_temp))
            quad_offset = np.float(errors[1][errors[0] == quad_temp])
            # print("Initial Offset at quad = " + str(twiss['x'][quad_ind][idx]-quad_offset ))

            quad_offset = np.float(errors[1][errors[0] == quad_temp])
            bpm_offset = np.float(errors[1][errors[0] == bpm_temp])
            # print("bpm off =  "+str(bpm_offset))

            self.madx.input(str(Strs[idx]) + "=" + str(0) + ";")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            BeamBPMOffset = twiss['x'][bpm_ind][idx] - bpm_offset
            np.random.normal(42)
            target = BeamBPMOffset + np.random.normal(scale=self.bpm_res)

            self.madx.input(str(Strs[idx]) + "=" + str(Strength / 0.3) + ";")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            BeamBPMOffset = twiss['x'][bpm_ind][idx] - bpm_offset
            twiss = self.OffsetQuadrupole(quad_temp, "x", -26e-6)
            BeamBPMOffset = twiss['x'][bpm_ind][idx] - bpm_offset

            OffsetatBPM = np.zeros((51, 2))
            #
            for i in range(51):
                twiss = self.OffsetQuadrupole(quad_temp, "x", 1e-6)
                np.random.normal(43 + i)
                BeamBPMOffset = twiss['x'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.bpm_res)
                OffsetatBPM[i, 0] = 1 * i - 25
                OffsetatBPM[i, 1] = BeamBPMOffset
                errors = np.vstack((self.madx.table["efield"]["name"],
                                    self.madx.table["efield"]["dx"]))
                quad_offset = np.float(errors[1][errors[0] == quad_temp])
            twiss = self.OffsetQuadrupole(quad_temp, "x", -25e-6)
            closest_ind = self.find_nearest(OffsetatBPM[:, 1], target)
            # print(OffsetatBPM[closest_ind, 0]*1e-6)
            twiss = self.OffsetQuadrupole(quad_temp, "x", OffsetatBPM[closest_ind, 0] * 1e-6)

            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dy"]))
            quad_offset = np.float(errors[1][errors[0] == quad_temp])
            bpm_offset = np.float(errors[1][errors[0] == bpm_temp])
            self.madx.input(str(Strs[idx]) + "=" + str(0) + ";")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            BeamBPMOffset = twiss['y'][bpm_ind][idx] - bpm_offset
            np.random.normal(42)
            target = BeamBPMOffset + np.random.normal(scale=self.bpm_res)

            self.madx.input(str(Strs[idx]) + "=" + str(Strength / 0.3) + ";")
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            BeamBPMOffset = twiss['y'][bpm_ind][idx] - bpm_offset
            twiss = self.OffsetQuadrupole(quad_temp, "y", -26e-6)
            BeamBPMOffset = twiss['y'][bpm_ind][idx] - bpm_offset

            for i in range(51):
                twiss = self.OffsetQuadrupole(quad_temp, "y", 1e-6)
                np.random.normal(43 + i)
                BeamBPMOffset = twiss['y'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.bpm_res)
                OffsetatBPM[i, 0] = 1 * i - 25
                OffsetatBPM[i, 1] = BeamBPMOffset
                errors = np.vstack((self.madx.table["efield"]["name"],
                                    self.madx.table["efield"]["dy"]))
                quad_offset = np.float(errors[1][errors[0] == quad_temp])
            twiss = self.OffsetQuadrupole(quad_temp, "y", -25e-6)
            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dy"]))

            closest_ind = self.find_nearest(OffsetatBPM[:, 1], target)
            twiss = self.OffsetQuadrupole(quad_temp, "y", OffsetatBPM[closest_ind, 0] * 1e-6)

    def OffsetQuadrupole(self, name, dim, offset):
        offset = np.round(offset, 1)
        if offset > 100:
            offset = 10
        elif offset < -100:
            offset = -10

        self.madx.input("eoption, add = true;")
        self.madx.input("Select, flag = ERROR, clear;")
        self.madx.input("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        # print("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        self.madx.input("ealign, d" + str(dim) + ":= " + str(offset * 10 ** -6) + ";")
        # print("ealign, d" + str(dim) + ":= " + str(offset*10**-6) + ";")
        self.madx.input("select, flag = error, clear;")
        self.madx.input("select, flag = error, class = quadrupole;")
        self.madx.input("select, flag = error, class = rbend;")
        self.madx.input("select, flag = error, class = sextupole;")
        self.madx.input("select, flag = error, class = octupole;")
        self.madx.input("select, flag = error, class = monitor;")
        self.madx.input("esave;")
        self.madx.input("select, flag = twiss, clear;")
        twiss = self.ptc_twiss(True)
        return twiss

    def OffsetSextupole(self, name, dim, offset):
        # offset = np.round(offset, 1)
        self.madx.input("eoption, add = true;")
        self.madx.input("Select, flag = ERROR, clear;")
        self.madx.input("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        # print("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        self.madx.input("ealign, d" + str(dim) + ":= " + str(offset * 10 ** -6) + ";")
        # print("ealign, d" + str(dim) + ":= " + str(offset * 10 ** -6) + ";")
        self.madx.input("select, flag = error, clear;")
        self.madx.input("select, flag = error, class = quadrupole;")
        self.madx.input("select, flag = error, class = rbend;")
        self.madx.input("select, flag = error, class = sextupole;")
        self.madx.input("select, flag = error, class = octupole;")
        self.madx.input("select, flag = error, class = monitor;")
        self.madx.input("esave;")
        self.madx.input("select, flag = twiss, clear;")
        twiss = self.ptc_twiss(True)
        return twiss

    def OffsetMagnet(self, name, offset_x, offset_y):
        # offset = np.round(offset, 1)
        self.madx.input("eoption, add = false;")
        self.madx.input("Select, flag = ERROR, clear;")
        # print("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        self.madx.input("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        # print("ealign, dx"  + ":= " + str(offset_x) + ";")
        self.madx.input("ealign, dx" + ":= " + str(offset_x + self.offset_err*np.random.normal()) + ",dy" + ":= " + str(offset_y + self.offset_err*np.random.normal()) + ";")
        # print("ealign, dx"  + ":= " + str(offset_x) + ", dy"  + ":= " + str(offset_y) + ";")
        # print("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        # self.madx.input("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        # print("dy" + ":= " + str(offset_y) + ";")
        # self.madx.input("dy"  + ":= " + str(offset_y) + ";")
        self.madx.input("select, flag = error, clear;")
        self.madx.input("select, flag = error, class = quadrupole;")
        self.madx.input("select, flag = error, class = rbend;")
        self.madx.input("select, flag = error, class = sextupole;")
        self.madx.input("select, flag = error, class = octupole;")
        self.madx.input("select, flag = error, class = monitor;")
        self.madx.input("esave;")
        self.madx.input("select, flag = twiss, clear;")
        twiss = self.ptc_twiss(True)
        return twiss

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def BBA(self):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$$              BBA                $$$$$$$$$$')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        "BBA"
        i = 0
        for j in range(len(self.q)):
            self.madx.input("quad" + str(j) + "=" + str(self.x_best[i]) + ";")
            i = i + 1
        for j in range(len(self.s)):
            self.madx.input("sext" + str(j) + "=" + str(self.x_best[i]) + ";")
            i = i + 1
        for j in range(len(self.o)):
            self.madx.input("oct" + str(j) + "=" + str(self.x_best[i]) + ";")
            i = i + 1
        for j in range(len(self.a)):
            self.madx.input("dist" + str(j) + "=" + str(self.x_best[i]) + ";")
            i = i + 1
        # self.madx.use(sequence='TT43')
        # self.madx.input("ii =" + str(0) + ";")
        # self.madx.call(file='add_errors.madx')
        # self.madx.input("USEKICK, Status = on;")
        # self.madx.input("USEMONITORS, Status = on;")
        # self.madx.input("eoption, add = true;")
        # self.madx.input("Select, flag = ERROR, clear;")
        # self.madx.input("Select, flag = ERROR, class = quadrupole;")
        # self.madx.input("ealign, dx := 0.2e-3;")
        # self.madx.input("eoption, add = true;")
        # self.madx.input("Select, flag = ERROR, clear;")
        # self.madx.input("Select, flag = ERROR, class = monitor;")
        # self.madx.input("ealign, dx := 0.1e-3;")
        # self.madx.input("Select, flag = error, clear;")
        # self.madx.input("select, flag = error, class = quadrupole;")
        # self.madx.input("select, flag = error, class = rbend;")
        # self.madx.input("select, flag = error, class = sextupole;")
        # self.madx.input("select, flag = error, class = octupole;")
        # self.madx.input("select, flag = error, class = monitor;")
        # self.madx.input("esave;")

        # self.madx.input("ESAVE, FILE=errors.txt;")

        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                file="confused")
        ## TRACK TO SEE BEAM SIZE AFTER CORRECTION
        var = []
        f = open('distr/Ellipse_150MeV_nominal.tfs', 'r')  # initialize empty array
        for line in f:
            var.append(
                line.strip().split())
        f.close()

        init_dist = np.array(var[9:])
        init_dist = init_dist[0:5000, 0:6]
        init_dist = init_dist.astype(np.float)

        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_align()
        self.madx.ptc_setswitch(fringe=True, exact_mis=True)
        # self.madx.ptc_observe(place='MERGE')
        #
        # with self.madx.batch():
        #     for particle in init_dist:
        #         self.madx.ptc_start(x=-particle[0], px=-particle[3],
        #                             y=particle[1], py=particle[4],
        #                             t=1 * particle[2],
        #                             pt=2.09 * particle[5])
        #     self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
        #                         maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
        #     self.madx.ptc_track_end()
        # ptc_output = self.madx.table.trackone

        twiss = self.ptc_twiss(True)
        for idx in range(np.size(twiss['name'])):
            if twiss['name'][idx] == 'merge:1':
                s_foil = twiss['s'][idx]
        # idx_temp = np.array(ptc_output.s == s_foil)
        BBACombo = np.array([("bpm.0:1", "mqawd.0:1", "quad0"),
                             ("bpm.1:1", "mqawd.4:1", "quad4"),
                             ("bpm.2:1", "mqawd.2:1", "quad2"),
                             ("bpm.3:1", "mqawd.9:1", "quad1"),
                             ("bpm.4:1", "mqawd.6:1", "quad5"),
                             ("bpm.6:1", "mqawd.10:1", "quad3"),
                             ("bpm.7:1", "mqawd.14:1", "quad5"),
                             ("bpm.8:1", "mqawd.11:1", "quad1")])
        quads = BBACombo[:, 1]
        bpms = BBACombo[:, 0]
        Strengths = BBACombo[:, 2]
        guess_m = np.zeros(shape=(len(quads), 1))

        names = twiss['name']
        names = [x[:-2] for x in names]
        names2 = [y + ":1" for y in names]
        bpm_ind = [any(b in s.lower() for b in bpms) for s in names2]
        quad_ind = [any(b in s.lower() for b in quads) for s in names2]
        for idx3 in range(len(bpm_ind)):
            if bpm_ind[idx3] == True:
                bpm_ind[idx3 + 1] = False
            if quad_ind[idx3] == True:
                quad_ind[idx3 + 1] = False
        for idx in range(len(quads)):
            # x0 = self.madx.table.trackone['x'][idx_temp]
            # y0 = self.madx.table.trackone['y'][idx_temp]
            # beam_size_x = np.multiply(np.std(x0), 1000000)
            # beam_size_y = np.multiply(np.std(y0), 1000000)
            # print(beam_size_x)
            # print(beam_size_y)
            twiss = self.ptc_twiss(True)
            betax_i = twiss['betx'][quad_ind][idx]
            betay_i = twiss['bety'][quad_ind][idx]
            betax_j = twiss['betx'][bpm_ind][idx]
            betay_j = twiss['bety'][bpm_ind][idx]
            mux_i = np.multiply(twiss['mu1'][quad_ind][idx], 2 * np.pi)
            muy_i = np.multiply(twiss['mu2'][quad_ind][idx], 2 * np.pi)
            mux_j = np.multiply(twiss['mu1'][bpm_ind][idx], 2 * np.pi)
            muy_j = np.multiply(twiss['mu2'][bpm_ind][idx], 2 * np.pi)
            Ax_ij = np.sqrt(np.multiply(betax_i, betax_j)) * np.sin(mux_i - mux_j)
            Ay_ij = np.sqrt(np.multiply(betay_i, betay_j)) * np.sin(muy_i - muy_j)

            strength = np.divide(twiss['k1l'][quad_ind][idx], twiss['l'][quad_ind][idx])
            offset = np.zeros(shape=(2, 2))
            separation = (+twiss['s'][quad_ind][idx] - twiss['s'][bpm_ind][idx])
            separation = 0.25
            beam_offset_at_quad = twiss['x'][quad_ind][idx]
            beam_offset_at_bpm = twiss['x'][twiss['name'][:] == bpms[idx]] + np.random.normal(scale=self.bpm_res)
            for idx2, i in enumerate(np.linspace(0.7 * strength, 1 * strength, 2)):
                errors = np.vstack((self.madx.table["efield"]["name"],
                                    self.madx.table["efield"]["dx"]))
                quad_offset = np.float(errors[1][errors[0] == quads[idx]][0])
                bpm_offset = np.float(errors[1][errors[0] == bpms[idx]][0])
                self.madx.input(str(Strengths[idx]) + "=" + str(i) + ";")
                # print("quad0 =" + str(i[0]) + ";")
                # self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0, file = "test.txt")
                twiss = self.ptc_twiss(True)
                offset[idx2, 1] = twiss['x'][twiss['name'][:] == bpms[idx]] - bpm_offset + np.random.normal(
                    scale=self.bpm_res)
                offset[idx2, 0] = i

            # print("change in k =" + str(1.0 * strength[0] - 0.0 * strength[0]))
            # print("change in x =" + str(offset[0, 1] - offset[1, 1]))
            # print("change in angle = " + str(np.arctan(np.divide((offset[0, 1] - offset[1, 1]), separation))))
            # print("test angle =" + str(np.divide((offset[0, 1] - offset[1, 1]), Ax_ij)))
            unnorm_offset = -np.divide(np.arctan(np.divide((offset[0, 1] - offset[1, 1]), -Ax_ij))
                                       , (1 * strength * twiss['l'][quad_ind][idx] - 0.7 * strength *
                                          twiss['l'][quad_ind][idx]))
            # print("change in angle = " + str(np.arctan(np.divide(twiss['x'][twiss['name'][:] == "test1:1"] - twiss['x'][twiss['name'][:] == "test2:1"], twiss['s'][twiss['name'][:] == "test1:1"] - twiss['s'][twiss['name'][:] == "test1:2"]))))

            # print("BEAM offset = " + str(beam_offset_at_quad))
            # print("QUAD offset = " + str(quad_offset))
            # print("BPM offset = " + str(bpm_offset))
            true_offset = (quad_offset - bpm_offset) * 10 ** 6
            # print("QUAD-BEAM offset = " + str(quad_offset - beam_offset_at_quad))
            # print("predicted QUAD-BEAM offset offset =" + str(unnorm_offset))

            print("QUAD-BPM offset = " + str(np.round(true_offset, 4)) + " um")
            # print("BEAM-BPM offset = " + str(-beam_offset_at_bpm + bpm_offset))
            guess = (unnorm_offset - bpm_offset + beam_offset_at_bpm) * 10 ** 6
            guess_m[idx] = (unnorm_offset - bpm_offset + beam_offset_at_bpm)[0]

            print("Guess at QUAD-BPM offset = " + str(np.round(guess[0], 4)) + " um at quad " + str(quads[idx]))
            # print("error = " + str(100*(true_offset-(guess))/true_offset) + "%")
            print("-------------")
        return guess_m

    def OneToOne(self):
        self.madx.input("USEKICK, Status = on;")
        self.madx.input("USEMONITORS, Status = on;")

        OneToOneCombo = np.array([("merge", "corr.8", "merge", "none")])
        # ("bpm.0", "corr.9", "mqawd.0", "quad0"),
        # ("bpm.1", "corr.0", "mqawd.4", "quad4"),
        # ("bpm.2", "corr.1", "mqawd.2", "quad2"),
        # ("bpm.3", "corr.2", "mqawd.9", "quad1"),
        # ("bpm.4", "corr.3", "mqawd.6", "quad5"),
        # ("bpm.6", "corr.4", "mqawd.10", "quad3"),
        # # ("bpm.6", "corr.5", "oct8"),
        # ("bpm.7", "corr.6", "mqawd.14", "quad5"),
        # ("bpm.8", "corr.7", "mqawd.11", "quad1"),

        quads = OneToOneCombo[:, 2]
        quads2 = [x + ":1" for x in quads]
        bpms = OneToOneCombo[:, 0]
        bpms2 = [x + ":1" for x in bpms]
        correctors = OneToOneCombo[:, 1]

        twiss = self.ptc_twiss(False)
        names = twiss['name']
        names = [x[:-2] for x in names]

        bpm_ind = [any(b in s.lower() for b in bpms) for s in names]
        quad_ind = [any(b in s.lower() for b in quads) for s in names]
        corr_ind = [any(b in s.lower() for b in correctors) for s in names]

        x_before = np.zeros(shape=(self.n_seeds, len(quads)))
        y_before = np.zeros(shape=(self.n_seeds, len(quads)))
        x_after = np.zeros(shape=(self.n_seeds, len(quads)))
        y_after = np.zeros(shape=(self.n_seeds, len(quads)))

        # for self.ii in range(self.n_seeds):
        # self.madx.use(sequence='TT43')

        # Zero all correctors
        corr_xy = np.zeros([2 * np.size(self.correctors)])
        for num, corr in enumerate(self.correctors):
            self.madx.input(str(corr) + ", VKICK = " + str(
                corr_xy[num + np.size(self.correctors)] + 0 * np.random.normal()) + ", HKICK = " + str(
                corr_xy[num] + 0 * np.random.normal()) + ";")

        # twiss = self.ptc_twiss(True)

        # self.madx.input("ii =" + str(self.ii) + ";")
        # print("ii =" + str(self.ii) + ";")
        # self.madx.call(file='add_errors.madx')

        twiss = self.ptc_twiss(False)
        # x_before[self.ii, :] = twiss['x'][quad_ind]
        # y_before[self.ii, :] = twiss['y'][quad_ind]

        ## Read previous corrector values out
        # corr_x_1 = twiss['hkick'][corr_ind]

        for idx in range(len(bpms)):
            print(quads[idx])
            twiss = self.ptc_twiss(False)
            y_before = twiss['y']
            x_before = twiss['x']
            betax_i = twiss['betx'][bpm_ind][idx]
            betay_i = twiss['bety'][bpm_ind][idx]
            betax_j = twiss['betx'][corr_ind][idx]
            betay_j = twiss['bety'][corr_ind][idx]
            mux_i = np.multiply(twiss['mu1'][bpm_ind][idx], 2 * np.pi)
            muy_i = np.multiply(twiss['mu2'][bpm_ind][idx], 2 * np.pi)
            mux_j = np.multiply(twiss['mu1'][corr_ind][idx], 2 * np.pi)
            muy_j = np.multiply(twiss['mu2'][corr_ind][idx], 2 * np.pi)
            Ax_ij = np.sqrt(np.multiply(betax_i, betax_j)) * np.sin(mux_i - mux_j)
            Ay_ij = np.sqrt(np.multiply(betay_i, betay_j)) * np.sin(muy_i - muy_j)
            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dx"]))
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"
            quad_offset = np.float(errors[1][errors[0] == quad_temp])
            bpm_offset = np.float(errors[1][errors[0] == bpm_temp])
            # print("BPM offset = " + str(bpm_offset))
            # print("QUAD offset = " + str(quad_offset))
            errors = np.vstack((self.madx.table["efield"]["name"],
                                self.madx.table["efield"]["dy"]))
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"
            quad_offset_y = np.float(errors[1][errors[0] == quad_temp])
            bpm_offset_y = np.float(errors[1][errors[0] == bpm_temp])
            # print("BPM offset y = " + str(bpm_offset_y))
            # print("QUAD offset y = " + str(quad_offset_y))
            # print(quad_offset)
            # print(bpm_offset)
            # print(Ax_ij, 1/Ax_ij)
            # guess = np.squeeze(args)
            # if args:
            #     if idx < len(bpms) - 1:
            #         offsets_x_bef = twiss['x'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.bpm_res,
            #                                                                                  size=np.shape(
            #                                                                                      twiss['x'][
            #                                                                                          bpm_ind][idx])) - guess[idx]
            #         offsets_y_bef = twiss['y'][bpm_ind][idx] - bpm_offset_y + np.random.normal(scale=self.bpm_res,
            #                                                                                    size=np.shape(
            #                                                                                        twiss['x'][bpm_ind][idx]))
            #     else:
            #         offsets_x_bef = twiss['x'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.btv_res,
            #                                                                                  size=np.shape(
            #                                                                                      twiss['x'][
            #                                                                                          bpm_ind][idx]))
            #         offsets_y_bef = twiss['y'][bpm_ind][idx] - bpm_offset_y + np.random.normal(scale=self.btv_res,
            #                                                                                    size=np.shape(
            #                                                                                        twiss['x'][bpm_ind][idx]))
            # else:
            if idx < len(bpms) - 1:
                offsets_x_bef = twiss['x'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.bpm_res,
                                                                                         size=np.shape(
                                                                                             twiss['x'][bpm_ind][idx]))

                offsets_y_bef = twiss['y'][bpm_ind][idx] - bpm_offset_y + np.random.normal(scale=self.bpm_res,
                                                                                           size=np.shape(
                                                                                               twiss['x'][bpm_ind][
                                                                                                   idx]))
            else:
                offsets_x_bef = twiss['x'][bpm_ind][idx] - bpm_offset + np.random.normal(scale=self.btv_res,
                                                                                         size=np.shape(
                                                                                             twiss['x'][bpm_ind][idx]))
                offsets_y_bef = twiss['y'][bpm_ind][idx] - bpm_offset_y + np.random.normal(scale=self.btv_res,
                                                                                           size=np.shape(
                                                                                               twiss['x'][bpm_ind][
                                                                                                   idx]))
            # print("offset bpm before = " + str(np.round(offsets_x_bef * 10 ** 6, 4)) + "um, " + str(
            # np.round(offsets_y_bef * 10 ** 6, 4)) + "um, " + str(quads[idx]))

            corr_x = np.sin(np.multiply(1 / Ax_ij, np.multiply(offsets_x_bef, -1)))
            corr_y = np.sin(np.multiply(1 / Ay_ij, np.multiply(offsets_y_bef, -1)))
            self.madx.input(str(correctors[idx]) + ", VKICK = " + str(
                corr_y + 0 * np.random.normal()) + ", HKICK = " + str(
                corr_x + 0 * np.random.normal()) + ";")

            self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            twiss = self.ptc_twiss(False)

            offsets_x_aft = twiss['x'][quad_ind][idx] - quad_offset
            offsets_x_aft_test = twiss['x'][bpm_ind][idx] - bpm_offset
            offsets_y_aft = twiss['y'][quad_ind][idx] - quad_offset_y
            offsets_y_aft_test = twiss['y'][bpm_ind][idx] - bpm_offset_y

    def SVD(self, gain):

        print('$$$$$$$$$$$$$$$              SVD                $$$$$$$$$$')

        SVDCombo = np.array([("bpm.0", "corr.9", "mqawd.0", "quad0"),
                             ("bpm.1", "corr.0", "mqawd.4", "quad4"),
                             ("bpm.2", "corr.1", "mqawd.2", "quad2"),
                             ("bpm.3", "corr.2", "mqawd.9", "quad1"),
                             ("bpm.4", "corr.3", "mqawd.6", "quad5"),
                             ("bpm.6", "corr.4", "mqawd.10", "quad3"),
                             ("bpm.7", "corr.6", "mqawd.14", "quad5"),
                             ("bpm.8", "corr.7", "mqawd.11", "quad1"),
                             ("merge", "corr.8", "merge", "none")])

        quads = SVDCombo[:, 2]
        bpms = SVDCombo[:, 0]
        correctors = SVDCombo[:, 1]
        twiss = self.ptc_twiss(False)
        names = twiss['name']
        names = [x[:-2] for x in names]

        bpm_ind = [any(b in s.lower() for b in bpms) for s in names]
        quad_ind = [any(b in s.lower() for b in quads) for s in names]
        corr_ind = [any(b in s.lower() for b in correctors) for s in names]

        bpm_x_before = np.zeros(shape=(len(quads)))
        bpm_y_before = np.zeros(shape=(len(quads)))
        quad_x_before = np.zeros(shape=(len(quads)))
        quad_y_before = np.zeros(shape=(len(quads)))
        bpm_x_after = np.zeros(shape=(len(quads)))
        bpm_y_after = np.zeros(shape=(len(quads)))
        quad_x_after = np.zeros(shape=(len(quads)))
        quad_y_after = np.zeros(shape=(len(quads)))

        # offset at quadrupoles before correction
        twiss = self.ptc_twiss(False)

        ## Read previous corrector values out
        corr_x_1 = twiss['hkick'][corr_ind]
        corr_y_1 = twiss['vkick'][corr_ind]

        # Determine beam-element offsets
        for idx in range(len(bpms)):
            twiss = self.ptc_twiss(False)
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"

            if idx < len(bpms) - 1:
                bpm_x_before[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][
                                                                                        bpm_ind][
                                                                                        idx]))

                bpm_y_before[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][bpm_ind][
                                                                                        idx]))

                quad_x_before[idx] = twiss['x'][quad_ind][idx]

                quad_y_before[idx] = twiss['y'][quad_ind][idx]
            else:
                bpm_x_before[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][
                                                                                        bpm_ind][
                                                                                        idx]))

                bpm_y_before[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                size=np.shape(
                                                                                    twiss['x'][bpm_ind][
                                                                                        idx]))

                quad_x_before[idx] = twiss['x'][quad_ind][idx]

                quad_y_before[idx] = twiss['y'][quad_ind][idx]
        b = self.getAij(0, bpm_ind, corr_ind)

        Axij = b[0]
        Ayij = b[1]
        Axij_inv = np.linalg.pinv(Axij)
        Ayij_inv = np.linalg.pinv(Ayij)

        corr_x = np.dot(Axij_inv, np.multiply(bpm_x_before, -1))
        corr_y = np.dot(Ayij_inv, np.multiply(bpm_y_before, -1))

        for num, corr in enumerate(correctors):
            self.madx.input(str(correctors[num]) + ", VKICK = " + str(corr_y_1[num] +
                                                                      corr_y[
                                                                          num] * gain + self.corr_err * np.random.normal()) + ", HKICK = " + str(
                corr_x_1[num] + corr_x[num] * gain + self.corr_err * np.random.normal()) + ";")
            print(str(correctors[num]) + ", VKICK = " + str(corr_y_1[num] +
                                                            corr_y[
                                                                num] * gain + self.corr_err * np.random.normal()) + ", HKICK = " + str(
                corr_x_1[num] + corr_x[num] * gain + self.corr_err * np.random.normal()) + ";")

        for idx in range(len(bpms)):
            twiss = self.ptc_twiss(False)

            quad_x_after[idx] = twiss['x'][quad_ind][idx]
            bpm_x_after[idx] = twiss['x'][bpm_ind][idx]
            quad_y_after[idx] = twiss['y'][quad_ind][idx]
            bpm_y_after[idx] = twiss['y'][bpm_ind][idx]

            # print("offset quad after = " + str(np.round(offsets_x_aft * 10 ** 6, 4)) + "um, " + str(
            #     np.round(offsets_y_aft * 10 ** 6, 4)) + "um, " + str(quads[idx]))
            # print("offset bpm after = " + str(np.round(offsets_x_aft_test * 10 ** 6, 4)) + "um, " + str(
            #     np.round(offsets_y_aft * 10 ** 6, 4)) + "um, " + str(quads[idx]))
        bpm_pos = twiss['s'][bpm_ind]
        return bpm_x_before, bpm_x_after, bpm_y_before, bpm_y_after, quad_x_before, quad_x_after, quad_y_before, quad_y_after, bpm_pos

    def svd_plot(self, n_iter, gain):
        # fig = plt.figure()
        # gs = matplotlib.gridspec.GridSpec(7, 1)
        # ax = fig.add_subplot(gs[0:3])
        # ax1 = fig.add_subplot(gs[4:7])
        # ax.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
        # ax.plot([], '-', color="darkorange", label="Corrected")
        # ax1.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
        # ax1.plot([], '-', color="darkorange", label="Corrected")
        # ax.legend()
        # ax1.legend()
        #
        # for i in range(nseeds):
        #     print("-------------------")
        #     print(i)
        #     self.addError(i)
        for j in range(n_iter):
            if j == 0:
                bpm_x_before, bpm_x_after, bpm_y_before, bpm_y_after, quad_x_before, quad_x_after, quad_y_before, quad_y_after, bpm_pos = self.SVD(
                    gain)
            else:
                _, bpm_x_after, _, bpm_y_after, _, quad_x_after, _, quad_y_after, bpm_pos = self.SVD(gain)

        return bpm_x_before, bpm_x_after, bpm_y_before, bpm_y_after, quad_x_before, quad_x_after, quad_y_before, quad_y_after, bpm_pos
        #         bs_temp = self.track()
        #         beam_size_x = bs_temp[0]
        #         beam_size_y = bs_temp[1]
        #     if i == 0:
        #         ax.plot(quad_pos, quad_x_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
        #         ax1.plot(quad_pos, quad_y_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
        #         beam_size_x_all = beam_size_x
        #         beam_size_y_all = beam_size_y
        #         quad_x_before_all = quad_x_before
        #         quad_y_before_all = quad_y_before
        #         quad_x_after_all = quad_x_after
        #         quad_y_after_all = quad_y_after
        #     else:
        #         ax.plot(quad_pos, quad_x_after * 10 ** 3, '-', color="darkorange", label=None)
        #         ax1.plot(quad_pos, quad_y_after * 10 ** 3, '-', color="darkorange", label=None)
        #         beam_size_x_all = np.vstack((beam_size_x_all, beam_size_x))
        #         beam_size_y_all = np.vstack((beam_size_y_all, beam_size_y))
        #         quad_x_before_all = np.vstack((quad_x_before_all, quad_x_before))
        #         quad_y_before_all = np.vstack((quad_y_before_all, quad_y_before))
        #         quad_x_after_all = np.vstack((quad_x_after_all, quad_x_after))
        #         quad_y_after_all = np.vstack((quad_y_after_all, quad_y_after))
        #
        # if nseeds == 1:
        #     ax.plot(quad_pos, quad_x_after * 10 ** 3, '-', color="darkorange", label=None)
        #     ax1.plot(quad_pos, quad_y_after * 10 ** 3, '-', color="darkorange", label=None)
        # ax1.set_xlabel("s [m]")
        # ax1.set_ylabel("Offset $y$ [mm]")
        # ax1.set_ylim([-2.5, 2.5])
        # ax.set_xlabel("s [m]")
        # ax.set_ylabel("Offset $x$ [mm]")
        # ax.set_ylim([-2.5, 2.5])
        # ax.set_title("SVD correction")
        #
        # if nseeds > 1:
        #     offsets_before = [quad_x_before_all, quad_y_before_all]
        #     offsets_after = [quad_x_after_all, quad_y_after_all]
        #     sizes = [beam_size_x_all, beam_size_y_all]
        #     for i in range(2):
        #         fig = plt.figure()
        #         gs = matplotlib.gridspec.GridSpec(3, 3, wspace=0.25, hspace=0.7)
        #         ax0 = fig.add_subplot(gs[0])
        #         ax1 = fig.add_subplot(gs[1])
        #         ax2 = fig.add_subplot(gs[2])
        #         ax3 = fig.add_subplot(gs[3])
        #         ax4 = fig.add_subplot(gs[4])
        #         ax5 = fig.add_subplot(gs[5])
        #         ax6 = fig.add_subplot(gs[6])
        #         ax7 = fig.add_subplot(gs[7])
        #         ax8 = fig.add_subplot(gs[8])
        #
        #         ax8.hist(sizes[i], alpha=0.8,
        #                  label="Mean = %.2f $\mu$m" % np.multiply(1, np.mean(sizes[i])))
        #         ax8.legend()
        #         ax8.set_xlabel('Beam size in $\mu$m')
        #         ax8.set_ylabel('Frequency')
        #         ax8.set_title('Beam size at merge')
        #         for idx2, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]):
        #             ax = fig.add_subplot(gs[idx2])
        #             ax.hist(1000000 * offsets_before[i][:, idx2], alpha=0.8,
        #                     label="Offsets before = %.2f $\mu$m" % np.multiply(1, np.std(1000000 * offsets_before[i][:, idx2])))
        #             ax.hist(1000000 * offsets_after[i][:, idx2], alpha=0.8,
        #                     label="Offsets after = %.2f $\mu$m" % np.multiply(1, np.std(1000000 * offsets_after[i][:, idx2])))
        #             ax.legend()
        #             ax.set_xlabel('Offset in $\mu$m')
        #             ax.set_ylabel('Frequency')
        #             ax.set_title('Quad = %s' % idx2)
        #         plt.show()

    def dfs_plot(self, weight, n_iter, gain, switch_off_HO=False):

        # fig = plt.figure()
        # gs = matplotlib.gridspec.GridSpec(7, 1)
        # ax = fig.add_subplot(gs[0:3])
        # ax1 = fig.add_subplot(gs[4:7])
        # ax.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
        # ax.plot([], '-', color="darkorange", label="Corrected")
        # ax1.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
        # ax1.plot([], '-', color="darkorange", label="Corrected")
        if switch_off_HO:
            self.switch_off_higher_order()

        for j in range(n_iter):
            self.shot += 1
            print("shot = " + str(self.shot))
            self.track()
            if j == 0:
                quad_x_before_0, quad_y_before_0, quad_x_before_m1, quad_y_before_m1, quad_x_before_1, \
                quad_y_before_1, quad_x_after_0, quad_y_after_0, quad_x_after_m1, quad_y_after_m1, \
                quad_x_after_1, quad_y_after_1, bpm_pos, names, quad_ind, bpm_x_before_0, bpm_y_before_0, bpm_x_after_0, bpm_y_after_0 = self.dfsEnvs(
                    weight, gain)
            else:
                _, _, _, _, _, _, quad_x_after_0, quad_y_after_0, quad_x_after_m1, quad_y_after_m1, \
                quad_x_after_1, quad_y_after_1, _, _, _, _, _, bpm_x_after_0, bpm_y_after_0 = self.dfsEnvs(weight, gain)
            if switch_off_HO:
                self.switch_on_higher_order()
            self.track()
        return bpm_x_before_0, bpm_x_after_0, bpm_y_before_0, bpm_y_after_0, quad_x_before_0, quad_x_after_0, \
               quad_y_before_0, quad_y_after_0, bpm_pos

    def dfsEnvs(self, weight, gain):
        print('$$$$$$$$$$$$$$$              DFS                $$$$$$$$$$')

        DFSCombo = np.array([("bpm.0", "corr.9", "mqawd.0", "quad0"),
                             ("bpm.1", "corr.0", "mqawd.4", "quad4"),
                             ("bpm.2", "corr.1", "mqawd.2", "quad2"),
                             ("bpm.3", "corr.2", "mqawd.9", "quad1"),
                             ("bpm.4", "corr.3", "mqawd.6", "quad5"),
                             ("bpm.6", "corr.4", "mqawd.10", "quad3"),
                             ("bpm.7", "corr.6", "mqawd.14", "quad5"),
                             ("merge", "corr.7", "mqawd.11", "quad1")])
        # ("merge", "corr.8", "merge", "none")])

        quads = DFSCombo[:, 2]
        bpms = DFSCombo[:, 0]
        correctors = DFSCombo[:, 1]
        np.random.seed(self.shot)
        x_jit = np.random.normal(scale=self.pos_jit, size=4)[0]
        y_jit = np.random.normal(scale=self.pos_jit, size=4)[1]
        px_jit = np.random.normal(scale=self.ang_jit, size=4)[2]
        py_jit = np.random.normal(scale=self.ang_jit, size=4)[3]
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                dpy=0,
                                X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
        names = twiss['name']
        names = [x[:-2] for x in names]

        bpm_ind = [any(b in s.lower() for b in bpms) for s in names]
        quad_ind = [any(b in s.lower() for b in quads) for s in names]
        corr_ind = [any(b in s.lower() for b in correctors) for s in names]

        bpm_x_before_0 = np.zeros(shape=(len(quads)))
        bpm_y_before_0 = np.zeros(shape=(len(quads)))
        bpm_x_before_m1 = np.zeros(shape=(len(quads)))
        bpm_y_before_m1 = np.zeros(shape=(len(quads)))
        bpm_x_before_1 = np.zeros(shape=(len(quads)))
        bpm_y_before_1 = np.zeros(shape=(len(quads)))

        quad_x_before_0 = np.zeros(shape=(len(quads)))
        quad_y_before_0 = np.zeros(shape=(len(quads)))
        quad_x_before_m1 = np.zeros(shape=(len(quads)))
        quad_y_before_m1 = np.zeros(shape=(len(quads)))
        quad_x_before_1 = np.zeros(shape=(len(quads)))
        quad_y_before_1 = np.zeros(shape=(len(quads)))

        bpm_x_after_0 = np.zeros(shape=(len(quads)))
        bpm_y_after_0 = np.zeros(shape=(len(quads)))
        bpm_x_after_m1 = np.zeros(shape=(len(quads)))
        bpm_y_after_m1 = np.zeros(shape=(len(quads)))
        bpm_x_after_1 = np.zeros(shape=(len(quads)))
        bpm_y_after_1 = np.zeros(shape=(len(quads)))

        quad_x_after_0 = np.zeros(shape=(len(quads)))
        quad_y_after_0 = np.zeros(shape=(len(quads)))
        quad_x_after_m1 = np.zeros(shape=(len(quads)))
        quad_y_after_m1 = np.zeros(shape=(len(quads)))
        quad_x_after_1 = np.zeros(shape=(len(quads)))
        quad_y_after_1 = np.zeros(shape=(len(quads)))

        # ## Read previous corrector values out
        corr_x_init = twiss['hkick'][corr_ind]
        corr_y_init = twiss['vkick'][corr_ind]

        # Determine beam-element offsets
        for idx in range(len(bpms)):
            twiss = self.ptc_twiss(False)
            quad_temp = str(quads[idx]) + ":1"
            bpm_temp = str(bpms[idx]) + ":1"

            if idx < len(bpms) - 1:
                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, DELTAP=str(-0.002), X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
                bpm_x_before_m1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                   size=np.shape(
                                                                                       twiss['x'][
                                                                                           bpm_ind][
                                                                                           idx]))

                bpm_y_before_m1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                   size=np.shape(
                                                                                       twiss['x'][
                                                                                           bpm_ind][
                                                                                           idx]))

                quad_x_before_m1[idx] = twiss['x'][quad_ind][idx]

                quad_y_before_m1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, DELTAP=str(0.002), X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

                bpm_x_before_1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                bpm_y_before_1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                quad_x_before_1[idx] = twiss['x'][quad_ind][idx]

                quad_y_before_1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

                bpm_x_before_0[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))
                bpm_y_before_0[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                quad_x_before_0[idx] = twiss['x'][quad_ind][idx]

                quad_y_before_0[idx] = twiss['y'][quad_ind][idx]

            else:
                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(-0.002))
                bpm_x_before_m1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                   size=np.shape(
                                                                                       twiss['x'][
                                                                                           bpm_ind][
                                                                                           idx]))

                bpm_y_before_m1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                   size=np.shape(
                                                                                       twiss['x'][
                                                                                           bpm_ind][
                                                                                           idx]))

                quad_x_before_m1[idx] = twiss['x'][quad_ind][idx]
                quad_y_before_m1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(0.002))
                bpm_x_before_1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                bpm_y_before_1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                quad_x_before_1[idx] = twiss['x'][quad_ind][idx]

                quad_y_before_1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
                bpm_x_before_0[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                bpm_y_before_0[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][idx]))

                quad_x_before_0[idx] = twiss['x'][quad_ind][idx]

                quad_y_before_0[idx] = twiss['y'][quad_ind][idx]

        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

        offsets_x = np.hstack(
            (bpm_x_before_0, np.multiply(bpm_x_before_m1 - bpm_x_before_0 - self.nom_x_m1[bpm_ind], weight),
             np.multiply(bpm_x_before_1 - bpm_x_before_0 - self.nom_x_1[bpm_ind], weight)))
        offsets_y = np.hstack(
            (bpm_y_before_0, np.multiply(bpm_y_before_m1 - bpm_y_before_0 - self.nom_y_m1[bpm_ind], weight),
             np.multiply(bpm_y_before_1 - bpm_y_before_0 - self.nom_y_1[bpm_ind], weight)))

        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                        DY=0, dpy=0, deltap=str(0), X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
        b = self.getAij_dfs(0, bpm_ind, corr_ind)
        Axij_0 = b[0]
        Ayij_0 = b[1]
        b = self.getAij_dfs(-0.002, bpm_ind, corr_ind)
        Axij_m1 = -b[0] + Axij_0
        Ayij_m1 = -b[1] + Ayij_0
        b = self.getAij_dfs(0.002, bpm_ind, corr_ind)
        Axij_1 = -b[0] + Axij_0
        Ayij_1 = -b[1] + Ayij_0
        Axij = np.vstack((Axij_0, np.multiply(Axij_m1, weight), np.multiply(Axij_1, weight)))
        Ayij = np.vstack((Ayij_0, np.multiply(Ayij_m1, weight), np.multiply(Ayij_1, weight)))

        Axij_inv = np.linalg.pinv(Axij)
        Ayij_inv = np.linalg.pinv(Ayij)

        corr_x = np.multiply(np.dot(Axij_inv, np.multiply(offsets_x, -1)), gain) + corr_x_init
        corr_y = np.multiply(np.dot(Ayij_inv, np.multiply(offsets_y, -1)), gain) + corr_y_init

        for num, corr in enumerate(correctors):
            self.madx.input(str(corr) + ", VKICK = " + str(
                corr_y[num] + self.corr_err * np.random.normal()) + ", HKICK = " + str(
                corr_x[num] + self.corr_err * np.random.normal()) + ";")
        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

        for idx in range(len(bpms)):
            if idx < len(bpms) - 1:
                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(-0.002))
                bpm_x_after_m1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][
                                                                                          idx]))

                bpm_y_after_m1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][
                                                                                          idx]))

                quad_x_after_m1[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_m1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(0.002))

                bpm_x_after_1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                bpm_y_after_1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                quad_x_after_1[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)

                bpm_x_after_0[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))
                bpm_y_after_0[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.bpm_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                quad_x_after_0[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_0[idx] = twiss['y'][quad_ind][idx]

            else:
                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(-0.002))
                bpm_x_after_m1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][
                                                                                          idx]))

                bpm_y_after_m1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                  size=np.shape(
                                                                                      twiss['x'][
                                                                                          bpm_ind][
                                                                                          idx]))

                quad_x_after_m1[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_m1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit, DELTAP=str(0.002))
                bpm_x_after_1[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                bpm_y_after_1[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                quad_x_after_1[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_1[idx] = twiss['y'][quad_ind][idx]

                twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                                        DY=0, dpy=0, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
                bpm_x_after_0[idx] = twiss['x'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                bpm_y_after_0[idx] = twiss['y'][bpm_ind][idx] + np.random.normal(scale=self.btv_res,
                                                                                 size=np.shape(
                                                                                     twiss['x'][
                                                                                         bpm_ind][idx]))

                quad_x_after_0[idx] = twiss['x'][quad_ind][idx]

                quad_y_after_0[idx] = twiss['y'][quad_ind][idx]
        bpm_pos = twiss['s'][bpm_ind]

        return quad_x_before_0, quad_y_before_0, quad_x_before_m1, quad_y_before_m1, quad_x_before_1, \
               quad_y_before_1, quad_x_after_0, quad_y_after_0, quad_x_after_m1, quad_y_after_m1, \
               quad_x_after_1, quad_y_after_1, bpm_pos, names, quad_ind, bpm_x_before_0, bpm_y_before_0, \
               bpm_x_after_0, bpm_y_after_0

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

    def getAij(self, deltap, bpm_ind, corr_ind):
        # twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
        #                         deltap=str(deltap))

        twiss = self.ptc_twiss(False)
        betaxi = (twiss['betx'][bpm_ind])
        betaxj = (twiss['betx'][corr_ind])
        betayi = (twiss['bety'][bpm_ind])
        betayj = (twiss['bety'][corr_ind])
        muxi = np.multiply(twiss['mu1'][bpm_ind], 2 * np.pi)
        muxj = np.multiply(twiss['mu1'][corr_ind], 2 * np.pi)

        si = twiss['s'][bpm_ind]
        sj = twiss['s'][corr_ind]

        muyi = np.multiply(twiss['mu2'][bpm_ind], 2 * np.pi)
        muyj = np.multiply(twiss['mu2'][corr_ind], 2 * np.pi)

        Axij = np.zeros((len(betaxi), len(betaxj)))
        for i in range(len(betaxi)):
            for j in range(len(betaxj)):
                if sj[j] >= si[i]:
                    pass
                else:
                    Axij[i][j] = np.sqrt(betaxi[i] * betaxj[j]) * np.sin(muxi[i] - muxj[j])

        # Axij_inv = np.linalg.pinv(Axij)

        Ayij = np.zeros((len(betayi), len(betayj)))
        for i in range(len(betayi)):
            for j in range(len(betayj)):
                if sj[j] >= si[i]:
                    pass
                else:
                    Ayij[i][j] = np.sqrt(betayi[i] * betayj[j]) * np.sin(muyi[i] - muyj[j])

        return Axij, Ayij

    def getAij_dfs(self, deltap, bpm_ind, corr_ind):
        twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0,
                                deltap=str(deltap))
        betaxi = twiss['betx'][bpm_ind]
        betaxj = twiss['betx'][corr_ind]
        dxi = twiss['dx'][bpm_ind]
        dxj = twiss['dx'][corr_ind]
        muxi = np.multiply(twiss['mux'][bpm_ind], 2 * np.pi)
        muxj = np.multiply(twiss['mux'][corr_ind], 2 * np.pi)

        si = twiss['s'][bpm_ind]
        sj = twiss['s'][corr_ind]

        betayi = twiss['bety'][bpm_ind]
        betayj = twiss['bety'][corr_ind]

        muyi = np.multiply(twiss['muy'][bpm_ind], 2 * np.pi)
        muyj = np.multiply(twiss['muy'][corr_ind], 2 * np.pi)

        Axij = np.zeros((len(betaxi), len(betaxj)))
        for i in range(len(betaxi)):
            for j in range(len(betaxj)):
                if sj[j] >= si[i]:
                    pass
                else:
                    Axij[i][j] = np.sqrt(betaxi[i] * betaxj[j]) * np.sin(muxi[i] - muxj[j])

        Ayij = np.zeros((len(betayi), len(betayj)))
        for i in range(len(betayi)):
            for j in range(len(betayj)):
                if sj[j] >= si[i]:
                    pass
                else:
                    Ayij[i][j] = np.sqrt(betayi[i] * betayj[j]) * np.sin(muyi[i] - muyj[j])
        return Axij, Ayij

    def reset(self):
        """
         If MAD-X fails, re-spawn process
         """
        print("Reseting MADX")
        self.err_flag = False
        self.madx.input('stop;')
        self.madx.quit()
        self.madx_init()
        self.zeroCorr()
        self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)

    def madx_init(self):
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
        for j in range(self.dof):
            print(self.name[j][0] + self.name[j][-1] + "=" + str(self.q[j]) + ";")
            self.madx.input(self.name[j] + "=" + str(self.q[j]) + ";")
        self.madx.use(sequence='TT43')

    def ptc_twiss(self, Bool):
        np.random.seed(self.shot)
        x_jit = np.random.normal(scale=self.pos_jit, size=4)[0]
        y_jit = np.random.normal(scale=self.pos_jit, size=4)[1]
        px_jit = np.random.normal(scale=self.ang_jit, size=4)[2]
        py_jit = np.random.normal(scale=self.ang_jit, size=4)[3]
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_align()
        self.madx.ptc_setswitch(fringe=True)
        self.madx.select(flag='ptc_twiss',
                         column=['name', 'keyword', 's', 'l','betx', 'bety', 'alfx', 'alfy',
                                 'mu1', 'mu2', 'mux',
                                 'muy', 'RE56', 'RE16', 'TE166'])
        self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                            dpy=0, CENTER_MAGNETS=Bool, X=x_jit, Y=y_jit, PX=px_jit, PY=py_jit)
        twiss = self.madx.table['ptc_twiss']
        return twiss

    def track(self):
        self.shot += 1
        print("shot = " + str(self.shot))
        np.random.seed(self.shot)
        x_jit = np.random.normal(scale=self.pos_jit, size=4)[0]
        y_jit = np.random.normal(scale=self.pos_jit, size=4)[1]
        px_jit = np.random.normal(scale=self.ang_jit, size=4)[2]
        py_jit = np.random.normal(scale=self.ang_jit, size=4)[3]
        init_dist = self.init_dist

        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_align()
        self.madx.ptc_observe(place='MERGE')
        # self.madx.input("ii =" + str(seed) + ";")

        with self.madx.batch():
            init_dist_0 = init_dist - np.mean(init_dist, axis=0)
            for particle in init_dist_0:
                self.madx.ptc_start(x=-particle[0] + x_jit, px=-particle[3] + px_jit,
                                    y=particle[1] + y_jit, py=particle[4] + py_jit,
                                    t=1 * particle[2],
                                    pt=2.09 * particle[5])
            self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
                                maxaper=[0.03, 0.03, 0.03, 0.03, 1.0, 1])
            self.madx.ptc_track_end()
        ptc_output = self.madx.table.trackone

        # self.madx.select(FLAG='makethin', THICK=True)
        # self.madx.makethin(SEQUENCE='TT43', STYLE='teapot')
        # # self.madx.align()
        # # madx.flatten()
        # self.madx.use(sequence='TT43')
        # twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
        # with self.madx.batch():
        #     self.madx.track(onetable=True, recloss=True, onepass=True)
        #     for particle in self.init_dist:
        #         self.madx.start(x=-particle[0] + x_jit, px=-particle[3] + px_jit,
        #                             y=particle[1] + y_jit, py=particle[4] + py_jit,
        #                             t=1 * particle[2],
        #                             pt=2.09 * particle[5])
        #
        #     self.madx.run(turns=1, maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
        #     self.madx.endtrack()
        #
        # ptc_output = self.madx.table.trackone

        twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                dpy=0)
        # twiss = self.ptc_twiss(False)
        if any(twiss['name'][:] == 'merge:1'):
            for idx in range(np.size(twiss['name'])):
                if twiss['name'][idx] == 'merge:1':
                    s_foil = twiss['s'][idx]
            idx_temp = np.array(ptc_output.s == s_foil)

            x0 = self.madx.table.trackone['x'][idx_temp]
            y0 = self.madx.table.trackone['y'][idx_temp]

            # all = x0 + y0
            # x0 = x0[~np.isnan(all)]
            # y0 = y0[~np.isnan(all)]

            offset_pt = np.mean(self.madx.table.trackone['pt'][idx_temp])

            beam_size_x = np.multiply(np.std(x0), 1000000)
            beam_size_y = np.multiply(np.std(y0), 1000000)
            offset_x = np.multiply(np.mean(x0), 1000000)
            offset_y = np.multiply(np.mean(y0), 1000000)
            twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0)
            dsx = twiss['dx'][twiss['name'] == "merge:1"]
            dsy = twiss['dy'][twiss['name'] == "merge:1"]
        else:
            beam_size_x = 50
            beam_size_y = 50
        print(beam_size_x, beam_size_y, offset_x, offset_y)

        return beam_size_x, beam_size_y, offset_x, offset_y
