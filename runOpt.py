"""
AWAKE Run 2 Electron Line Model
R. Ramjiawan
Jun 2020
Track beam through line and extract beam parameters at merge-point
"""

import OptEnv as opt_env
import errorOptEnv as errorEnv
import plot_save_output as plot
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import pickle
import sys


# Initial values for quadrupoles (q), sextupoles (s), octupoles (o) and distances (a)
# nominal

a0 = 0
a1 = 0
a2 = 0

qkmb = 0.01437
q0 = 2.323811628
q1 = -2.897820544
q2 = 5.80293764
q3 = -5.441585438
q4 = 1.948010225
q5 = -2.129893896
q6 = 1.966194965
q7 = 2.962078517
q8 = -3.876905437



# # enter the values here over which to optimise, otherwise hard-code them into MAD-X file
# Norm indicates the maximum strength of quad intensity
x = {
    0: {'name': 'kq0', 'strength': q0, 'type': 'quadrupole', 'norm': 7},
    1: {'name': 'kq1', 'strength': q1, 'type': 'quadrupole', 'norm': 7},
    2: {'name': 'kq2', 'strength': q2, 'type': 'quadrupole', 'norm': 7},
    3: {'name': 'kq3', 'strength': q3, 'type': 'quadrupole', 'norm': 7},
    4: {'name': 'kq4', 'strength': q4, 'type': 'quadrupole', 'norm': 7},
    5: {'name': 'kq5', 'strength': q5, 'type': 'quadrupole', 'norm': 7},
    6: {'name': 'kq6', 'strength': q6, 'type': 'quadrupole', 'norm': 7},
    7: {'name': 'kq7', 'strength': q7, 'type': 'quadrupole', 'norm': 7},
    8: {'name': 'kq8', 'strength': q8, 'type': 'quadrupole', 'norm': 7},
    9: {'name': 'kmb', 'strength': qkmb, 'type': 'quadrupole', 'norm': 7},
    # 15: {'name': 'dist0', 'strength': a0, 'type': 'distance', 'norm': 0.2},
    # 16: {'name': 'dist1', 'strength': a1, 'type': 'distance', 'norm': 0.2},
    # 17: {'name': 'dist2', 'strength': a2, 'type': 'distance', 'norm': 0.2},
    # 18: {'name': 'dist3', 'strength': a3, 'type': 'distance', 'norm': 0.1},
    # 19: {'name': 'dist4', 'strength': a4, 'type': 'distance', 'norm': 0.2},
    # 20: {'name': 'dist5', 'strength': a5, 'type': 'distance', 'norm': 0.3},
    # 21: {'name': 'dist6', 'strength': a6, 'type': 'distance', 'norm': 0.3},
    # 22: {'name': 'dist7', 'strength': a7, 'type': 'distance', 'norm': 0.3},
    # 23: {'name': 'dist8', 'strength': a8, 'type': 'distance', 'norm': 0.3}
}

# Specify parameters for optimisation
solver = 'Powell'
n_iter = 20
n_particles = 100  # Used to generate distribution to track
foil_w = 0*100e-6
init_dist = []
thin = False
file = 'distr/Ellipse_150MeV_nominal.tfs'

# Initialise environment
env = opt_env.kOptEnv(solver, n_particles, n_iter, init_dist, foil_w, x, thin=thin)

# Initialise input distribution
var = []
f = open(file, 'r')  # initialize empty array
for line in f:
    var.append(
        line.strip().split())
f.close()
init_dist = np.array(var)[0:n_particles, 0:6].astype(np.float)
env.init_dist = init_dist
del var

# Either use optimiser (solution) or just output as is (step)
# If don't use step, will run with values as in general_tt43_python
if solver == "pyMOO":
    env.step_MO(env.norm_data([y['strength'] for y in x.values()]))
    plot = plot.Plot(env.madx, env.x_best, x, init_dist, foil_w, env.output_all, env.x_all)
    plot.twiss()
else:
    env.step(env.norm_data([y['strength'] for y in x.values()]))

# Optimise
if solver != "pyMOO":
    # solution = minimize(env.step, env.norm_data([y['strength'] for y in x.values()]), method=solver, options={'maxfev':n_iter})
    plot = plot.Plot(env.madx, env.x_best, x, init_dist, foil_w, env.output_all, env.x_all)
    plot.twiss()
    # plot.plotmat_twiss()
    # plot.plot1()
    # env.step(env.norm_data([y['strength'] for y in x.values()]))
    # plot.error()
    # plot.plotmat()
    sys.exit()
else:
    from pymoo.model.problem import Problem
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    from pymoo.optimize import minimize
    from pymoo.factory import get_termination
    from pymoo.visualization.scatter import Scatter
    
    


    
    x_0 = env.norm_data([y['strength'] for y in x.values()])
    norm_vect = env.norm_data([y['norm'] for y in x.values()])
    n_obj = 1
    
    
    class MatchingProblem(opt_env.kOptEnv, Problem):
        def __init__(self,
                     norm_vect,
                     x_0,
                     n_var=len(x_0),
                     n_obj=n_obj,
                     n_constr=0,
                     xl=None,
                     xu=None):
            opt_env.kOptEnv.__init__(self, solver, n_particles, n_iter, init_dist, foil_w, x, thin=thin)
            Problem.__init__(self,
                             n_var=len(x_0),
                             n_obj=n_obj,
                             n_constr=n_constr,
                             xl=-np.ones(np.shape(norm_vect)),
                             xu=np.ones(np.shape(norm_vect)))

        def _evaluate(self, x_n, out, *args, **kwargs):
            f = []
            for j in range(x_n.shape[0]):
                y_raw_all, y_raw_single = self.step_MO(x_n[j, :])

                if self.n_obj == 1:
                    f.append(y_raw_single)
                else:
                    f.append(y_raw_all)
            out["F"] = np.vstack(f)

    problem = MatchingProblem(
        norm_vect=norm_vect,
        n_var=len(x_0),
        n_obj=n_obj,
        n_constr=0,
        x_0=x_0,
        xl=-np.ones(np.shape(norm_vect)),
        xu=np.ones(np.shape(norm_vect)))

    # problem.evaluate(np.vstack([problem.x_0, problem.x_0, -np.ones_like(problem.x_0)]))

    algorithm = GA(
        pop_size=200,
        n_offsprings=200,
        sampling=get_sampling("real_lhs"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=30),
        eliminate_duplicates=True
    )

    # termination = MultiObjectiveDefaultTermination(
    #     x_tol=1e-8,
    #     cv_tol=1e-6,
    #     f_tol=1e-7,
    #     nth_gen=5,
    #     n_last=30,
    #     n_max_gen=50000,
    #     n_max_evals=200000
    # )
    termination = get_termination("n_eval", n_iter)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   copy_algorithm=False,
                   verbose=True)

    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # Plotting functions
    import plot_save_output as plot
    name = [y['name'] for y in x.values()]
    plot = plot.Plot(env.madx, problem.x_best, x, init_dist, foil_w, problem.output_all, problem.x_all)
    for j in range(len(problem.x_best)):
        env.madx.input(name[j] + "=" + str(problem.x_best[j]) + ";")
        print(name[j] + "=" + str(problem.x_best[j]) + ";")
        env.madx.use(sequence='TT43', range='#s/#e')
    # plot.plotmat_twiss()
    # plot.twiss()
    # plot.plot1(problem.output_all)
    # plot2 = Scatter()
    # plot2.add(res.F, color="red")
    # plot2.show()

    fig = plt.figure(figsize=[8, 7], constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax1 = plt.subplot(gs[:])
    # ax1.plot(np.zeros(shape=  (algorithm.pop_size)), problem.output_all[0:algorithm.pop_size, 0], 'o')
    generations = int(len(problem.output_all[algorithm.pop_size:, -1]) / algorithm.n_offsprings)
    mean = np.zeros(shape=(generations))
    max = np.zeros(shape=(generations))
    min = np.zeros(shape=(generations))
    iterations = np.zeros(shape=(generations))
    for i in range(generations):
        v = problem.output_all[algorithm.pop_size + i * algorithm.n_offsprings:algorithm.pop_size + (
                i + 1) * algorithm.n_offsprings, -2]
        mean[i] = (np.mean(v[v < 10 ** 21]))
        min[i] = (np.min(v[v < 10 ** 21]))
        max[i] = (np.max(v[v < 10 ** 21]))
        iterations[i] = i +1
    ax1.fill_between(iterations, min, max, alpha=0.1, color="tab:blue")
    ax1.plot(iterations, mean, '-o', label="mean(objective)", color="tab:blue")
    ax1.plot(iterations, min, '-', linewidth=0.4, color="tab:blue")
    ax1.plot(iterations, max, '-', linewidth=0.4, color="tab:blue")

    # ax2 = ax1.twinx()
    # mean = np.zeros(shape=(generations))
    # max = np.zeros(shape=(generations))
    # min = np.zeros(shape=(generations))
    # iterations = np.zeros(shape=(generations))
    # for i in range(generations):
    #     v = problem.output_all[algorithm.pop_size + i * algorithm.n_offsprings:algorithm.pop_size + (
    #             i + 1) * algorithm.n_offsprings, 4]
    #     mean[i] = np.mean(v[v < 1 * 10 ** 6])
    #     min[i] = np.min(v[v < 1 * 10 ** 6])
    #     max[i] = np.max(v[v < 1 * 10 ** 6])
    #     iterations[i] = i +1
    # ax2.fill_between(iterations, min, max, alpha=0.05, color='tab:orange')
    # ax2.plot(iterations, mean, '-', color='tab:orange', label="mean($\sigma_x$)")
    # ax2.plot(iterations, min, '-', linewidth=0.2, color="tab:orange")
    # ax2.plot(iterations, max, '-', linewidth=0.2, color="tab:orange")
    # mean = np.zeros(shape=(generations))
    # max = np.zeros(shape=(generations))
    # min = np.zeros(shape=(generations))
    # iterations = np.zeros(shape=(generations))
    # for i in range(generations):
    #     v = problem.output_all[algorithm.pop_size + i * algorithm.n_offsprings:algorithm.pop_size + (
    #             i + 1) * algorithm.n_offsprings, 3]
    #     mean[i] = np.mean(v[v < 1 * 10 ** 6])
    #     min[i] = np.min(v[v < 1 * 10 ** 6])
    #     max[i] = np.max(v[v < 1 * 10 ** 6])
    #     iterations[i] = i +1
    # ax2.fill_between(iterations, min, max, alpha=0.05, color='tab:green')
    # ax2.plot(iterations, mean, '-', color='tab:green', label="mean($\sigma_y$)")
    # ax2.plot(iterations, min, '-', linewidth=0.2, color="tab:green")
    # ax2.plot(iterations, max, '-', linewidth=0.2, color="tab:green")
    # ax2.set_ylabel("Beam size [$\mu$m]")
    # fig.legend(loc="center")
    # ax2.set_yscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Objective function")
    # ax1.set_xlim([1, iterations[-1]])
    # ax2.set_ylim([0, 1000])



## ERROR STUDIES

    # errorEnv = errorEnv.Error(env.x_best, x, init_dist, n_particles, env.madx)
    # nseeds = 1


    # bsx_all = np.zeros(nseeds)
    # bsy_all = np.zeros(nseeds)
    # ox_all = np.zeros(nseeds)
    # oy_all = np.zeros(nseeds)
    # tsx_all = np.zeros(nseeds)
    # tsy_all = np.zeros(nseeds)
    # power_jit = 0*10e-6
    # mom_jit = 0*1e-3
    # input_jit = 0*10e-6
    # proton_jit = 0*56
    # for i in range(nseeds):
    #     print(i)
    #     scale = np.random.randint(1, 10000)
    #     # errorEnv.addError(scale)
    #     # errorEnv.OneToOne()
    #     # errorEnv.addPowerConv(power_jit, i)
    #     errorEnv.calcOffsets(8)
    #     # print("before" + str(bsx) +" "+str(bsy)+" "+str(osx)+" "+str( osy))
    #     # errorEnv.OneToOne()
    #     # errorEnv.track()
    #     # bsx_all[i] = bsx
    #     # bsy_all[i] = bsy
    #     # ox_all[i] = osx
    #     # oy_all[i] = osy

    #     print("after" + str(bsx) + " " + str(bsy) + " " + str(osx) + " " + str(osy))
    #     tsx_all[i] = (proton_jit*np.random.normal() - osx)
    #     tsy_all[i] = (proton_jit*np.random.normal() - osy)
    #     print(bsx, bsy, osx, osy, tsx_all[i], tsy_all[i])
    #     print((bsx < 6.9) & (bsy < 6.9), (tsx_all[i] < 13) & (tsy_all[i]< 13))
    # print(sum((bsx_all < 6.9) & (bsy_all < 6.9)), sum((-13<tsx_all ) & (-13<tsy_all )&(tsx_all < 13) & (tsy_all < 13)), sum((-13<tsx_all ) & (-13<tsy_all )&(tsx_all < 13) & (tsy_all < 13)
    #                                                                                 & (bsx_all < 6.9) & (bsy_all < 6.9)))
    # print(np.mean(bsx_all), np.mean(bsy_all), np.mean(tsy_all), np.mean(tsy_all))

nseeds = 2
# errorEnv = errorEnv.Error(env.x_best, x, init_dist, n_particles)
# Model steering algorithms
errorEnv = errorEnv.Error(env.x_best, x, init_dist, n_particles)
# errorEnv.track()
# print("Adding the errors")
# errorEnv.addError(0)
# errorEnv.track()
#
# # errorEnv.OptScanMO()
# print("doing svd")
# errorEnv.svd_plot(10, 1)  # Takes n_iter, gain as input
# print("doing dfs")
# errorEnv.dfs_plot(10, 2, 3, 0.95)    # Number of seeds, dfs weight, number of iterations of steering, gain

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(7, 1)
ax = fig.add_subplot(gs[0:3])
ax1 = fig.add_subplot(gs[4:7])
ax.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
ax.plot([], '-', color="darkorange", label="Corrected")
ax1.plot([], '-', color=[0, 0.324219, 0.628906], label="Uncorrected")
ax1.plot([], '-', color="darkorange", label="Corrected")
ax.legend()
ax1.set_xlabel("s [m]")
ax1.set_ylabel("Offset $y$ [mm]")
# ax1.set_ylim([-5, 5])
ax.set_xlabel("s [m]")
ax.set_ylabel("Offset $x$ [mm]")
# ax.set_ylim([-5, 5])
# ax.set_title("Quad shunt")
ax1.legend()

# # # Quad shunt, dfs
for i in range(0, nseeds):
    errorEnv.shot = 0
    errorEnv.addError(i)
    bs_temp = errorEnv.track()
    beam_size_before_x = bs_temp[0]
    beam_size_before_y = bs_temp[1]
    bpm_x_before, bpm_x_after, bpm_y_before, bpm_y_after, quad_x_before, quad_x_after, \
    quad_y_before, quad_y_after, bpm_pos = errorEnv.quadshunt_plot(2, 0.5)
    _, bpm_x_after, _, bpm_y_after, _, quad_x_after, \
    _, quad_y_after, bpm_pos = errorEnv.quadshunt_plot(1, 1)
    # _, bpm_x_after, _, bpm_y_after, _, quad_x_after, _, quad_y_after, _ = errorEnv.dfs_plot(0, 1, 1, False)
    _, bpm_x_after, _, bpm_y_after, _, quad_x_after, _, quad_y_after, _ = errorEnv.dfs_plot(1, 1, 0.95, True)

    # errorEnv.SextOptScan(np.array(
    #         ["mqawd.0:1", "mqawd.4:1", "mqawd.2:1", "mqawd.9:1", "mqawd.6:1", "mqawd.10:1", "mqawd.14:1", "mqawd.11:1"]))
    bpm_x_after, bpm_y_after =    errorEnv.SextOptScan(np.array(
            ["sd3:1", "sd1:1", "sd5:1", "sd2:1", "sd6:1", "sd4:1", "oct8:1", "oct7:1", "oct6:1", "oct11:1"]))
    # _, bpm_x_after, _, bpm_y_after, _, quad_x_after, _, quad_y_after, _ = errorEnv.dfs_plot(1, 1, 1, False)
    # errorEnv.OctOptScan()
    # Takes number of seeds as input
    #     # _, bpm_x_after, _, bpm_y_after, _, quad_x_after, \
    #     #     _, quad_y_after, _ = errorEnv.dfs_plot(i, 1, 2, 0.95)   # Takes number of seeds as input
    #     # _, bpm_x_after, _, bpm_y_after, _, quad_x_after, \errorEnv.dfs_plot(i, 1, 2, 0.95)
    #     # _, quad_y_after, _ = errorEnv.dfs_plot(i, 2, 1, 0.7)  # Takes number of seeds as input
    #     # _, bpm_x_after, _, bpm_y_after, _, quad_x_after, \
    #     #         _, quad_y_after, bpm_pos = errorEnv.quadshunt_plot(i, 1, 0.95)  # Takes number of seeds as input
    bs_temp = errorEnv.track()
    print("---------------------------")
    beam_size_after_x = bs_temp[0]
    beam_size_after_y = bs_temp[1]
    if i == 0:
        bpm_x_before_all = bpm_x_before
        bpm_y_before_all = bpm_y_before
        bpm_x_after_all = bpm_x_after
        bpm_y_after_all = bpm_y_after
        quad_x_before_all = quad_x_before
        quad_y_before_all = quad_y_before
        quad_x_after_all = quad_x_after
        quad_y_after_all = quad_y_after
        beam_size_x_before_all = beam_size_before_x
        beam_size_y_before_all = beam_size_before_y
        beam_size_x_after_all = beam_size_after_x
        beam_size_y_after_all = beam_size_after_y
    else:
        bpm_x_before_all = np.vstack((bpm_x_before_all, bpm_x_before))
        bpm_y_before_all = np.vstack((bpm_y_before_all, bpm_y_before))
        bpm_x_after_all = np.vstack((bpm_x_after_all, bpm_x_after))
        bpm_y_after_all = np.vstack((bpm_y_after_all, bpm_y_after))
        quad_x_before_all = np.vstack((quad_x_before_all, quad_x_before))
        quad_y_before_all = np.vstack((quad_y_before_all, quad_y_before))
        quad_x_after_all = np.vstack((quad_x_after_all, quad_x_after))
        quad_y_after_all = np.vstack((quad_y_after_all, quad_y_after))
        beam_size_x_before_all = np.vstack((beam_size_x_before_all, beam_size_before_x))
        beam_size_y_before_all = np.vstack((beam_size_y_before_all, beam_size_before_y))
        beam_size_x_after_all = np.vstack((beam_size_x_after_all, beam_size_after_x))
        beam_size_y_after_all = np.vstack((beam_size_y_after_all, beam_size_after_y))
    err_dict = {}
    err_dict['bpm_x_before_all'] = bpm_x_before
    err_dict['bpm_y_before_all'] = bpm_y_before
    err_dict['bpm_x_after_all'] = bpm_x_after
    err_dict['bpm_y_after_all'] = bpm_y_after
    err_dict['quad_x_before_all'] = quad_x_before
    err_dict['quad_y_before_all'] = quad_y_before
    err_dict['quad_x_after_all'] = quad_x_after
    err_dict['quad_y_after_all'] = quad_y_after
    err_dict['beam_size_x_before_all'] = beam_size_before_x
    err_dict['beam_size_y_before_all'] = beam_size_before_y
    err_dict['beam_size_x_after_all'] = beam_size_after_x
    err_dict['beam_size_y_after_all'] = beam_size_after_y
    err_dict['n_shots'] = errorEnv.shot
    err_dict['bpm_res'] = errorEnv.bpm_res
    err_dict['btv_res'] = errorEnv.btv_res
    err_dict['corr_err'] = errorEnv.corr_err
    err_dict['offset_err'] = errorEnv.offset_err


    filename = 'data/error_study_paper_' + str(i)
    outfile = open(filename, 'wb')
    # pickle.dump(err_dict, outfile)
    outfile.close()

    ax.plot(bpm_pos, bpm_x_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
    ax.plot(bpm_pos, bpm_x_after * 10 ** 3, '-', color="darkorange", label=None)
    ax1.plot(bpm_pos, bpm_y_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
    ax1.plot(bpm_pos, bpm_y_after * 10 ** 3, '-', color="darkorange", label=None)
# Plotting for the error studies
#

if nseeds > 1:
    offsets_before = [bpm_x_before, bpm_y_before]
    offsets_after = [bpm_x_after, bpm_y_after]
    sizes_before = [beam_size_x_before_all, beam_size_y_before_all]
    sizes_after = [beam_size_x_after_all, beam_size_y_after_all]
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.7)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    # ax2 = fig.add_subplot(gs[2])
    # ax3 = fig.add_subplot(gs[3])
    # ax4 = fig.add_subplot(gs[4])
    # ax5 = fig.add_subplot(gs[5])
    # ax6 = fig.add_subplot(gs[6])
    # ax7 = fig.add_subplot(gs[7])
    # ax8 = fig.add_subplot(gs[8])
    i = 0
    ax0.hist(sizes_before[i], alpha=0.8,
             label="Mean before = %.2f $\mu$m" % np.multiply(1, np.mean(sizes_before[i])))
    ax0.hist(sizes_after[i], alpha=0.8,
             label="Mean after = %.2f $\mu$m" % np.multiply(1, np.mean(sizes_after[i])))
    ax0.legend()

    ax0.set_xlabel('Beam size in $\mu$m')
    ax0.set_ylabel('Frequency')
    ax0.set_title('Horizontal')
    i = 1
    ax1.hist(sizes_before[i], alpha=0.8,
             label="Mean before = %.2f $\mu$m" % np.multiply(1, np.mean(sizes_before[i])))
    ax1.hist(sizes_after[i], alpha=0.8,
             label="Mean after = %.2f $\mu$m" % np.multiply(1, np.mean(sizes_after[i])))
    ax1.legend()

    ax1.set_xlabel('Beam size in $\mu$m')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Vertical')
    # fig = plt.figure()
    # for idx2, ax in enumerate([ax0, ax1]):
    #     ax = fig.add_subplot(gs[idx2])
    #     ax.hist(1000000 * offsets_before[i][idx2], alpha=0.8,
    #             label="Offsets before = %.2f $\mu$m" % np.multiply(1, np.std(
    #                 1000000 * offsets_before[i][idx2])))
    #     ax.hist(1000000 * offsets_after[i][:idx2], alpha=0.8,
    #             label="Offsets after = %.2f $\mu$m" % np.multiply(1, np.std(
    #                 1000000 * offsets_after[i][idx2])))
    #     ax.legend()
    #     ax.set_xlabel('Offset in $\mu$m')
    #     ax.set_ylabel('Frequency')
    #     ax.set_title('Quad = %s' % idx2)
    # plt.show()
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(1, 2, wspace=0.25, hspace=0.7)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    # ax2 = fig.add_subplot(gs[2])
    # ax3 = fig.add_subplot(gs[3])
    # ax4 = fig.add_subplot(gs[4])
    # ax5 = fig.add_subplot(gs[5])
    # ax6 = fig.add_subplot(gs[6])
    # ax7 = fig.add_subplot(gs[7])
    # ax8 = fig.add_subplot(gs[8])
    i = 0
    ax0.hist(offsets_before[i], alpha=0.8,
             label="Jitter before = %.2f $\mu$m" % np.multiply(1, np.std(offsets_before[i])))
    ax0.hist(offsets_after[i], alpha=0.8,
             label="Jitter after = %.2f $\mu$m" % np.multiply(1, np.std(offsets_after[i])))
    ax0.legend()

    ax0.set_xlabel('Beam jitter at injection-point in $\mu$m')
    ax0.set_ylabel('Frequency')
    ax0.set_title('Horizontal')
    i = 1
    ax1.hist(offsets_before[i], alpha=0.8,
             label="Jitter before = %.2f $\mu$m" % np.multiply(1, np.std(offsets_before[i])))
    ax1.hist(offsets_after[i], alpha=0.8,
             label="Jitter after = %.2f $\mu$m" % np.multiply(1, np.std(offsets_after[i])))
    ax1.legend()

    ax1.set_xlabel('Beam jitter at injection-point in $\mu$m')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Vertical')

