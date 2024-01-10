"""
GET BEAM SIZE
R. Ramjiawan
Oct 2020
Track beam through line and extract beam parameters at merge-point
"""

import numpy as np
import scipy as scp
from scipy import stats
import sys

class getBeamSize:
    def __init__(self, q, n_particles, madx, init_dist, x):
        self.q = q
        self.x = x
        self.no_quad = sum(np.array([y['type'] for y in x.values()]) == 'quadrupole')
        self.no_sext = sum(np.array([y['type'] for y in x.values()]) == 'sextupole')
        self.no_oct = sum(np.array([y['type'] for y in x.values()]) == 'octupole')
        self.no_dist = sum(np.array([y['type'] for y in x.values()]) == 'distance')
        self.n_particles = n_particles
        self.madx = madx
        self.verbose = True
        self.ptc = False
        self.name = [y['name'] for y in x.values()]
        self.init_dist = init_dist
        x0 = self.init_dist[:, 0]
        px0 = self.init_dist[:, 3]
        y0 = self.init_dist[:, 1]
        py0 = self.init_dist[:, 4]
        self.dof = self.no_quad + self.no_sext + self.no_oct + self.no_dist
        self.gamma = np.sqrt(938e-3 ** 2 + 430e-3 ** 2) / 938e-3
        self.beta_nom = 8.24397
        
        # Calculate bunch parameters from input distribution
        self.emitx_before = self.calcEmit(x0, px0)
        self.emity_before = self.calcEmit(y0, py0)
        self.betx0 = np.divide(np.mean(np.multiply(x0, x0)), self.emitx_before)
        self.alfx0 = -np.divide(np.mean(np.multiply(x0, px0)), self.emitx_before)
        self.bety0 = np.divide(np.mean(np.multiply(y0, y0)), self.emity_before)
        self.alfy0 = -np.divide(np.mean(np.multiply(y0, py0)), self.emity_before)
        self.sig_nom_x = np.sqrt(np.multiply(self.emitx_before, self.beta_nom))
        self.sig_nom_y = np.sqrt(np.multiply(self.emity_before, self.beta_nom))
        print("input emittance x = {:.4f}, {:.4f} um".format(self.emitx_before * self.gamma * 10 ** 6,
                                                             self.emity_before * self.gamma * 10 ** 6))
        print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}: ".format(self.betx0, self.bety0, self.alfx0,
                                                                                   self.alfy0))

    def get_beam_size(self):
        _, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
        sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        # set magnet strengths/positions
        self.set_quads()
        try:
            ptc_output, error_flag = self.ptc_track('gantry$start',  'gantry$end', ['gantry$start','iso'],self.init_dist) 
            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,DY=0, dpy=0, file='twiss_verified.txt')
            
            maxbety = max(twiss['bety'])
            maxbetx = max(twiss['betx'])
            
            if maxbetx>1000 or maxbety>1000:
            	print("Betas are too high, " + "Betas are (betx,bety)=" + str(maxbetx) + "," + str(maxbetx))
            	loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
            sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
            else:
            	x0 = np.array(ptc_output['x'].loc['iso']) #in meters
            	y0 = np.array(ptc_output['y'].loc['iso'])
            	z0 = np.array(ptc_output['t'].loc['iso'])
            	px0 = np.array(ptc_output['px'].loc['iso'])
            	py0 = np.array(ptc_output['py'].loc['iso'])
            	loss = (self.n_particles - len(x0))/self.n_particles
            	
        except RuntimeError:
            error_flag = 1
        if error_flag == 1:
            loss = self.n_particles
        else:

            # Heavily penalise the loss of too many particle
            if loss > 0.8 * self.n_particles:
                print("loss is at " + str(100 * loss / self.n_particles) + "%")
                _, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, _, kl_divergence, \
                sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
            else:
                emitx = self.calcEmit(x0, px0) #in meters * rad
                print("output emittance x = {:.4f} um".format(emitx * self.gamma * 10 ** 6))
                sig_nom_x_after = np.sqrt(np.multiply(emitx, self.beta_nom))

                emity = self.calcEmit(y0, py0)
                print("output emittance y = {:.4f} um".format(emity * self.gamma * 10 ** 6))
                sig_nom_y_after = np.sqrt(np.multiply(emitx, self.beta_nom))

                beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y = self.get_beam_params(x0, y0, z0)
                
                # Calculate the KL divergence
                kl_divergence_x = self.calcKL(x0)
                kl_divergence_y = self.calcKL(y0)
                kl_divergence_px = self.calcKL(px0)
                kl_divergence_py = self.calcKL(py0)
                kl_divergence = np.sqrt(np.square(kl_divergence_x) + np.square(kl_divergence_y)
                                        + 0*np.square(kl_divergence_px) + 0*np.square(kl_divergence_py))
                if self.verbose:
                    print("KL divergences = {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(kl_divergence_x, kl_divergence_y,
                                                                                   kl_divergence_px, kl_divergence_py))

                ptc_twiss = self.ptc_twiss()
                #print(ptc_twiss['name'])
                
                # Calculate twiss parameters at merge-point
                if 'iso:1' in ptc_twiss['name']:
                    betx = twiss['betx'][ptc_twiss['name'] == 'iso:1'][0]
                    bety = twiss['bety'][ptc_twiss['name'] == 'iso:1'][0]

                    alfx = ptc_twiss['alfx'][ptc_twiss['name'] == 'iso:1'][0]
                    alfy = ptc_twiss['alfy'][ptc_twiss['name'] == 'iso:1'][0]

                    dx = twiss['dx'][twiss['name'][:] == 'iso:1'][0]
                    dx2 = twiss['dpx'][twiss['name'][:] == 'iso:1'][0]

                    if betx < 0 or bety < 0:
                        betx = 100
                        bety = 100
                else:  # if beam lost before merge
                    betx = 100
                    bety = 100
                    alfx = 1
                    alfy = 1
                    dx = 1
                    dx2 = 1
                
                print("beta - x,y = {:.4f}, {:.4f}; alpha - x,y = {:.4f}, {:.4f}; "
                      "dx dpx = {:.4f}, {:.4f}): ".format(betx, bety, alfx, alfy, dx, dx2))
        return beam_size_x, beam_size_y, beam_size_z, loss, error_flag, alfx, alfy, kl_divergence, dx, dx2, self.sig_nom_x, self.sig_nom_y, sig_nom_x_after, sig_nom_y_after, float(
            maxbetx), float(maxbety), beam_pos_x, beam_pos_y, frac_x, frac_y

    def ptc_track(self, start, end, observe, init_dist):
        error_flag = 0
        try:
            self.madx.use(sequence='GANTRY', range=start + "/" + end)
            self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                            DY=0, dpy=0)
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
            self.madx.ptc_align()
            self.madx.ptc_setswitch(fringe=True)
            for obs in observe:
                self.madx.ptc_observe(place=obs)
            self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)
            self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                dpy=0)
            with self.madx.batch():
                for particle in init_dist:
                    self.madx.ptc_start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4])
                self.madx.ptc_track(icase=4, element_by_element=True, dump=False, onetable=True, recloss=True,file='track.txt',ffile=1)
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone
            ptc_output = ptc_output.dframe()
            ptc_output = ptc_output[ptc_output['turn']==1]

        except RuntimeError or IndexError:  # If magnets overlap or twiss incomputable
            print('MAD-X Error occurred, re-spawning MAD-X process')
            error_flag = 1
            ptc_output = {}
        return ptc_output, error_flag

    def track(self, start, end, observe, init_dist):
        ptc_output = {}
        error_flag = 0
        try:
            with self.madx.batch():
                self.madx.track(onetable=True, recloss=True, onepass=True)
                for particle in init_dist:
                    self.madx.start(x=-particle[0], px=-particle[3], y=particle[1], py=particle[4],
                                    t=0 * particle[2])
                for obs in observe:
                    self.madx.observe(place=obs)
                #self.madx.run(turns=1, maxaper=[0.025, 0.025, 0.025, 0.025, 1.0, 1])
                self.madx.endtrack()

        except RuntimeError:
            print('MAD-X Error occurred, re-spawning MAD-X process')
            loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
            sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y = self.penalise()
        else:
            ptc_output = self.madx.table.trackone
            ptc_output = ptc_output.dframe()
        return ptc_output, error_flag

    def set_quads(self):
        for j in range(self.dof):
            self.madx.input(self.name[j] + "=" + str(self.q[j]) + ";")
            # if self.verbose:
            print(self.name[j][0] + self.name[j][-1] + "=" + str(self.q[j]))
        self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0,
                        DY=0, dpy=0)

    @staticmethod
    def calcEmit(u0, pu0):
        temp = np.mean((u0 - np.mean(u0)) * (pu0 - np.mean(pu0)))
        emit = np.sqrt(np.std(u0) ** 2 * np.std(pu0) ** 2 - temp ** 2)
        return emit

    @staticmethod
    def calcKL(u0):
        std = (np.std(u0))
        pk = scp.stats.norm(0, std).pdf(np.linspace(-4 * std, 4 * std, num=50))
        qk_temp = stats.gaussian_kde(np.array(u0))
        qk = qk_temp([np.linspace(-4 * std, 4 * std, num=50)])
        kl_div = stats.entropy(pk=pk, qk=qk)
        if kl_div == float("inf"):
            kl_div = 10
        return kl_div

    def get_beam_params(self, x0, y0, z0, rms=False):
    
        dim=0 #10**3 #Unit Converter
        if rms:
            beam_size_x = np.multiply(self.rmsValue(x0), 10 ** dim)
            beam_size_y = np.multiply(self.rmsValue(y0), 10 ** dim)
            beam_size_z = np.multiply(self.rmsValue(z0), 10 ** dim)
        else:
            beam_size_x = np.multiply(np.std(x0), 10 ** dim)
            beam_size_y = np.multiply(np.std(y0), 10 ** dim)
            beam_size_z = np.multiply(np.std(z0), 10 ** dim)
        beam_pos_x = np.multiply(np.mean(x0), 10 ** dim)
        beam_pos_y = np.multiply(np.mean(y0), 10 ** dim)
        # frac_x = np.divide(sum(abs(x0)<5.76*10**-6), len(x0))
        # frac_y = np.divide(sum(abs(y0)<5.76*10**-6), len(y0))
        frac_x = stats.kurtosis(self.reject_outliers(x0), fisher=True)
        frac_y = stats.kurtosis(self.reject_outliers(y0), fisher=True)
        return beam_size_x, beam_size_y, beam_size_z, beam_pos_x, beam_pos_y, frac_x, frac_y

    def reject_outliers(self, data):
        ind = abs(data - np.mean(data)) < 5 * np.std(data)
        return data[ind]


    def penalise(self):
        loss = self.n_particles
        frac_x = 1e8
        frac_y = 1e8
        beam_size_x = 1e8
        beam_size_y = 1e8
        beam_pos_x = 1e3
        beam_pos_y = 1e3
        beam_size_z = 1e3
        error_flag = 1
        kl_divergence = 1
        sig_nom_x_after = 0
        sig_nom_y_after = 0
        maxbetx = 1000
        maxbety = 1000
        alfx = 10
        alfy = 10
        dx = 1
        dx2 = 1
        return loss, beam_size_x, beam_size_y, beam_pos_x, beam_pos_y, beam_size_z, error_flag, kl_divergence, \
               sig_nom_x_after, sig_nom_y_after, maxbetx, maxbety, alfx, alfy, dx, dx2, frac_x, frac_y, 

    def rmsValue(self, arr):
        n = len(arr)
        square = 0
        for i in range(0, n):
            square += (arr[i] ** 2)
        mean = (square / float(n))
        root = np.sqrt(mean)
        return root

    def del_madx_tab(self):
        self.madx.input("delete, table = trackone;")
        self.madx.input("delete, table = trackloss;")
        self.madx.input("delete, table = tracksumm;")

    def ptc_twiss(self):
        self.madx.ptc_create_universe()
        self.madx.ptc_align()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                            dpy=0) 
        twiss = self.madx.table['ptc_twiss']
        return twiss
