"""
AWAKE Run 2 Electron Line Model
R. Ramjiawan
Jun 2020
Track beam through line and extract beam parameters at merge-point
"""

import OptEnv as opt_env
#import errorOptEnv as errorEnv
import Beam_Generator as beam_gen
import plot_output as plot
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import pickle
import sys


# Initial values for quadrupoles (q), sextupoles (s), octupoles (o) and distances (a)
# nominal

qkmb = -0.01437*0
q0 = 2.323811628*0
q1 = -2.897820544*0
q2 = 5.80293764*0
q3 = -5.441585438*0
q4 = -1.948010225*0
q5 = 2.129893896*0
q6 = 1.966194965*0
q7 = 2.962078517*0
q8 = -3.876905437*0
a0 = 1.0;
a1 = 1.0;
a2 = 2;


# # enter the values here over which to optimise, otherwise hard-code them into MAD-X file
# Norm indicates the maximum strength of quad intensity
x = {
    0: {'name': 'kq0', 'strength': q0, 'type': 'quadrupole', 'norm': 20},
    1: {'name': 'kq1', 'strength': q1, 'type': 'quadrupole', 'norm': 20},
    2: {'name': 'kq2', 'strength': q2, 'type': 'quadrupole', 'norm': 50},
    3: {'name': 'kq3', 'strength': q3, 'type': 'quadrupole', 'norm': 50},
    4: {'name': 'kq4', 'strength': q4, 'type': 'quadrupole', 'norm': 20},
    5: {'name': 'kq5', 'strength': q5, 'type': 'quadrupole', 'norm': 20},
    6: {'name': 'kq6', 'strength': q6, 'type': 'quadrupole', 'norm': 50},
    7: {'name': 'kq7', 'strength': q7, 'type': 'quadrupole', 'norm': 50},
    8: {'name': 'kq8', 'strength': q8, 'type': 'quadrupole', 'norm': 50},
    9: {'name': 'kmb', 'strength': qkmb, 'type': 'quadrupole', 'norm': 5},
#    10: {'name': 'dist0', 'strength': a0, 'type': 'distance', 'norm': 3},
#    11: {'name': 'dist1', 'strength': a1, 'type': 'distance', 'norm': 3},
#    12: {'name': 'dist2', 'strength': a2, 'type': 'distance', 'norm': 3},
} #distance doesn't work as expected: they increase only the final drift ('after ISO') in theory this it also forbidden by seq lenght definition

### Specify parameters for optimisation ###

#Bounds on variables  are aviable for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, and COBYLA methods
solver = 'pyMOO'
n_iter = 1000
n_particles = 500 
init_dist = []
thin = False

# Beam Parameters Vector [betx,bety,alphax,alphay,emix,emiy]
beam_pars=[8.24397,8.24397,0.0,0.0,((7/np.sqrt(5))*10**(-6)),((7/np.sqrt(5))*10**(-6))]

#Generate a Gaussian Beam from beam_pars initialiazion
beam_gen.Beam_Generator(beam_pars)
file = 'distr/Beam_Distribution.tfs'

# Initialise environment
env = opt_env.kOptEnv(solver, n_particles, n_iter, init_dist, x, thin=thin)

# Initialise input distribution
var = []
f = open(file, 'r')  # initialize empty array
for line in f:
    var.append(
        line.strip().split())
f.close()
init_dist = np.array(var)[0:n_particles, 0:6].astype(float)
env.init_dist = init_dist
del var

# Either use optimiser (solution) or just output as is (step)
# If don't use step, will run with values as in general_tt43_python
if solver == "pyMOO":
    env.step(env.norm_data([y['strength'] for y in x.values()]))
    plot = plot.Plot(env.madx, env.x_best, x, init_dist,  env.output_all, env.x_all)
    plot.twiss()
else:
    env.step(env.norm_data([y['strength'] for y in x.values()]))

# Optimise
if solver != "pyMOO":
    bnds = [(l, u) for l, u in zip(-np.ones(len(x.values())), np.ones(len(x.values())))]
    solution = minimize(env.step, env.norm_data([y['strength'] for y in x.values()]), method=solver, bounds=bnds, options={'maxfev':n_iter})
    plot = plot.Plot(env.madx, env.x_best, x, init_dist, env.output_all, env.x_all)    
    #plot.twiss()
    #plot.ptc_twiss_2()
    #plot.plot1() #only if you are optimizing (plot the evolution of variables during optimization)
    
else:
    print('ciao')
    from pymoo.model.problem import Problem
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.algorithms.so_genetic_algorithm import GA
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    from pymoo.optimize import minimize
    from pymoo.factory import get_termination
    from pymoo.visualization.scatter import Scatter
    
    
    print('ciao')

    
    x_0 = env.norm_data([y['strength'] for y in x.values()])
    norm_vect = env.norm_data([y['norm'] for y in x.values()])
    n_obj = 1

    
    sys.exit()
    
    class MatchingProblem(opt_env.kOptEnv, Problem):
        def __init__(self,
                     norm_vect,
                     x_0,
                     n_var=len(x_0),
                     n_obj=n_obj,
                     n_constr=0,
                     xl=None,
                     xu=None):
            opt_env.kOptEnv.__init__(self, solver, n_particles, n_iter, init_dist, x, thin=thin)
            Problem.__init__(self,
                             n_var=len(x_0),
                             n_obj=n_obj,
                             n_constr=n_constr,
                             xl=-np.ones(np.shape(norm_vect)),
                             xu=np.ones(np.shape(norm_vect)))

        def _evaluate(self, x_n, out, *args, **kwargs):
            f = []
            for j in range(x_n.shape[0]):
                y_raw_all, y_raw_single = self.step(x_n[j, :])

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
    print(res)
    sys.exit()
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # Plotting functions
    import plot_save_output as plot
    name = [y['name'] for y in x.values()]
    plot = plot.Plot(env.madx, problem.x_best, x, init_dist, foil_w, problem.output_all, problem.x_all)
    for j in range(len(problem.x_best)):
        env.madx.input(name[j] + "=" + str(problem.x_best[j]) + ";")
        print(name[j] + "=" + str(problem.x_best[j]) + ";")
        env.madx.use(sequence='GANTRY', range='#s/#e')
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



