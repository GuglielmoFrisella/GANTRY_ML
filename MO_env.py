import numpy as np
import gym
from cpymad.madx import Madx
import copy


class kOptEnv(gym.Env):

    def __init__(self, norm_vect, weights, targets, magnets, initial_x_offset, initial_y_offset, bpm_res, init_dist):

        self.rew = 10 ** 20  # Must be higher than initial cost function - definitely a better way to do this
        self.counter = 0
        self.w = weights
        self.targets = targets
        self.magnets = magnets
        # Collate all variables (quadrupoles, sextupoles, octupoles, distances)
        # Store actions, beam size, loss and fraction for every iteration
        # self.x_best = np.zeros([1, self.dof])
        # Vector to normalise actions
        self.norm_vect = norm_vect
        self.x_best = np.zeros([1, len(norm_vect)])
        self.initial_x_offset = initial_x_offset
        self.initial_y_offset = initial_y_offset
        self.bpm_res = 0
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
        # Number of particles to track
        # Max number of iterations
        # Spawn MAD-X process
        print("SPAWNING MADX AGAIN")
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)


    def step(self, offsets_norm):
        self.counter = self.counter + 1
        print("iteration = " + str(self.counter))
        # print(self.magnets)
        offsets = self.unnorm_data(offsets_norm)[0]
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

        output = abs(beam_size_x + self.bpm_res * np.random.normal()) + abs(
            beam_size_y + self.bpm_res * np.random.normal()) + abs(
            offset_x + self.bpm_res * np.random.normal()) + abs(offset_y + self.bpm_res * np.random.normal())
        print("beamsize x, y = " + str(beam_size_x) + ", " + str(beam_size_y) + ", " + str(offset_x) + ", " + str(
            offset_y))
        print("cost = " + str(output))

        # If objective function is best so far, update x_best with new best parameters
        if output < self.rew:
            self.x_best_err = offsets
            self.rew = output
        print("best = " + str(self.rew))
        print("---------------------------------")


        beamsizes = ((beam_size_x- 5.75)/700)**2 + ((beam_size_y- 5.75)/700)**2
        beam_offsets = (offset_x/100)**2 + (offset_y/100)**2
        parameters = [beam_offsets, beamsizes]
        y_raw = np.abs(np.array(parameters) - np.array(self.targets))

        # # If objective function is best so far, update x_best with new best parameters
        # if output < self.rew:
        #     self.x_best = x_unnor
        #     self.rew = output
        # print("best = " + str(self.rew))
        # print("---------------------------------")
        return y_raw, self._mse(y_raw)

    def track(self):
        init_dist = self.init_dist
        try:
            self.madx.ptc_create_universe()
            self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
            self.madx.ptc_setswitch(fringe=True)
            self.madx.ptc_align()
            self.madx.ptc_observe(place='MERGE')
            # self.madx.input("ii =" + str(seed) + ";")

            with self.madx.batch():
                init_dist_0 = init_dist - np.mean(init_dist, axis=0)
                for particle in init_dist_0:
                    self.madx.ptc_start(x=-particle[0], px=-particle[3],
                                        y=particle[1], py=particle[4],
                                        t=1 * particle[2],
                                        pt=2.09 * particle[5])
                self.madx.ptc_track(icase=56, element_by_element=True, dump=True, onetable=True, recloss=True,
                                    maxaper=[0.03, 0.03, 0.03, 0.03, 1.0, 1])
                self.madx.ptc_track_end()
            ptc_output = self.madx.table.trackone

            twiss = self.madx.twiss(BETX=self.betx0, ALFX=self.alfx0, DX=0, DPX=0, BETY=self.bety0, ALFY=self.alfy0, DY=0,
                                    dpy=0)
            if any(twiss['name'][:] == 'merge:1'):
                for idx in range(np.size(twiss['name'])):
                    if twiss['name'][idx] == 'merge:1':
                        s_foil = twiss['s'][idx]
                idx_temp = np.array(ptc_output.s == s_foil)

                x0 = self.madx.table.trackone['x'][idx_temp]
                y0 = self.madx.table.trackone['y'][idx_temp]
                offset_pt = np.mean(self.madx.table.trackone['pt'][idx_temp])

                beam_size_x = np.multiply(np.std(x0), 1000000)
                beam_size_y = np.multiply(np.std(y0), 1000000)
                offset_x = np.multiply(np.mean(x0), 1000000)
                offset_y = np.multiply(np.mean(y0), 1000000)
                twiss = self.madx.twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0,
                                        dpy=0)
                dsx = twiss['dx'][twiss['name'] == "merge:1"]
                dsy = twiss['dy'][twiss['name'] == "merge:1"]
        except:
            beam_size_x = 50
            beam_size_y = 50
            offset_x = 100
            offset_y = 100
            # twiss = self.ptc_twiss(False)
        #
        # else:
        #     beam_size_x = 50
        #     beam_size_y = 50
        print(beam_size_x, beam_size_y, offset_x, offset_y)

        return beam_size_x, beam_size_y, offset_x, offset_y



    def OffsetMagnet(self, name,  offset_x, offset_y):
        # offset = np.round(offset, 1)
        self.madx.input("eoption, add = false;")
        self.madx.input("Select, flag = ERROR, clear;")
        # print("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        self.madx.input("Select, flag = ERROR,class = " + str(name[0:-2]) + ";")
        self.madx.input("ealign, dx"  + ":= " + str(offset_x) + ",dy"  + ":= " + str(offset_y) + ";")
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

    def ptc_twiss(self, Bool):
        self.madx.ptc_create_universe()
        self.madx.ptc_create_layout(model=1, method=6, exact=True, NST=100)
        self.madx.ptc_align()
        self.madx.ptc_setswitch(fringe=True)
        self.madx.ptc_twiss(BETX=11.3866, ALFX=-2.1703, DX=0, DPX=0, BETY=11.1824, ALFY=-2.1110, DY=0, dpy=0, CENTER_MAGNETS=Bool)
        twiss = self.madx.table['ptc_twiss']
        return twiss

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

    def reset(self):
        """
         If MAD-X fails, re-spawn process
         """
        self.madx = Madx(stdout=False)
        self.madx.call(file='general_tt43_python.madx')
        self.madx.use(sequence='TT43')
        self.madx.option(echo=False, warn=True, info=False, debug=False, verbose=False)

    def _mse(self, values):
        return np.sum((self.w * values) ** 2) / len(values)

    def __deepcopy__(self, memo):
        return deepcopy_with_sharing(self,
                                     shared_attribute_names=['madx'],
                                     memo=memo)



def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
    '''
    Deepcopy an object, except for a given list of attributes, which should
    be shared between the original object and its copy.

    obj is some object
    shared_attribute_names: A list of strings identifying the attributes that
    should be shared between the original and its copy.
    memo is the dictionary passed into __deepcopy__.  Ignore this argument if
    not calling from within __deepcopy__.
    '''
    assert isinstance(shared_attribute_names, (list, tuple))
    shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}

    if hasattr(obj, '__deepcopy__'):
        # Do hack to prevent infinite recursion in call to deepcopy
        deepcopy_method = obj.__deepcopy__
        obj.__deepcopy__ = None

    for attr in shared_attribute_names:
        del obj.__dict__[attr]

    clone = copy.deepcopy(obj)

    for attr, val in shared_attributes.items():
        setattr(obj, attr, val)
        setattr(clone, attr, val)

    if hasattr(obj, '__deepcopy__'):
        # Undo hack
        obj.__deepcopy__ = deepcopy_method
        del clone.__deepcopy__

    return clone