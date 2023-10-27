import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd

params = {'axes.labelsize': 30,  # fontsize for x and y labels (was 10)
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

beamsize_x = np.zeros(shape=(40,1))
beamsize_y = np.zeros(shape=(40,1))
# plt.figure(figsize=(12,8))
import pickle
for i in range(0,39):
    filename = "data/error_study_0um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i] = new_dict['beam_size_x_after_all']
    beamsize_y[i] = new_dict['beam_size_y_after_all']
beamsize_x[39] = 6.6
x = np.squeeze(beamsize_x)
x_d = np.linspace(x.min(), x.max(), 1000)
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")

logprob = kde.score_samples(x_d[:, None])
# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 1 $\mu$m]")
# bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_x/5.75, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 1 $\mu$m")


beamsize_x = np.zeros(shape=(93, 1))
beamsize_y = np.zeros(shape=(93, 1))
for i in range(0, 9):
    filename = "data/error_study_10um" + str(i)
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i] = new_dict['beam_size_x_after_all']
    beamsize_y[i] = new_dict['beam_size_y_after_all']

for i in range(10, 14):
    filename = "data/error_study_10um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i-1] = new_dict['beam_size_x_after_all']
    beamsize_y[i-1] = new_dict['beam_size_y_after_all']
for i in range(15, 95):
    filename = "data/error_study_10um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i-2] = new_dict['beam_size_x_after_all']
    beamsize_y[i-2] = new_dict['beam_size_y_after_all']

x = np.squeeze(beamsize_x)
x_d = np.linspace(x.min(), x.max(), 1000)
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
f, ax = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [4,1], 'height_ratios': [1, 4]})

ax[0][0].fill_between(x_d, np.exp(logprob), alpha=0.5, color = 'steelblue', label="fit [BPM resolution 10 $\mu$m]")
# bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
ax[0][0].hist(beamsize_x, bins=20, histtype="step", density=True, linewidth=3, label="$\sigma_x$")
ax[0][0].axis('off')
ax[1][1].axis('off')
ax[0][1].axis('off')
x = np.squeeze(beamsize_y)
x_d = np.linspace(x.min(), x.max(), 1000)
kde.fit(x[:, None])
logprob = kde.score_samples(x_d[:, None])
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
kde.fit(x[:, None])
ax[1][1].fill_betweenx(x_d, np.exp(logprob),alpha=0.5, color = 'steelblue')
ax[1][1].hist(beamsize_y, bins=20, orientation=u'horizontal', histtype="step",  density=True, linewidth=3)
# plt.xlabel("$\sigma_{x}$ [$\mu$m]")
data = {'x': np.squeeze(beamsize_x),'y': np.squeeze(beamsize_y)}
data = pd.DataFrame(data=data)
sns.kdeplot(data = data,
    x='x', y='y',
    ax=ax[1][0], fill=True
)

# plt.ylabel("Frequency", labelpad=10)
# plt.legend()
ax[1][0].set_xlabel('$\sigma_x$ [$\mu$m]', fontsize=30)
ax[1][0].set_ylabel('$\sigma_y$ [$\mu$m]', fontsize=30)
ax[1][0].plot([8.6,8.6], [ax[1][0].get_ylim()[0],8.6], color='tab:orange', linestyle='--')
ax[1][0].plot([ax[1][0].get_xlim()[0],8.6], [8.6, 8.6], color='tab:orange', linestyle='--')
ax[0][0].axvline(x=8.6, color='tab:orange', linestyle='--')
ax[1][1].axhline(y=8.6, color='tab:orange', linestyle='--')
plt.show()


##########################################
#########################################

beamsize_x = np.zeros(shape=(40,1))
beamsize_y = np.zeros(shape=(40,1))
# plt.figure(figsize=(12,8))
import pickle
for i in range(0,39):
    filename = "data/error_study_0um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i] = new_dict['quad_x_after_all'][-1]*10**6
    beamsize_y[i] = new_dict['quad_y_after_all'][-1]*10**6
beamsize_x[39] = 6.6
x = np.squeeze(beamsize_x)
x_d = np.linspace(x.min(), x.max(), 1000)
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")

logprob = kde.score_samples(x_d[:, None])
# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 1 $\mu$m]")
# bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_x/5.75, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 1 $\mu$m")


beamsize_x = np.zeros(shape=(93, 1))
beamsize_y = np.zeros(shape=(93, 1))
for i in range(0, 9):
    filename = "data/error_study_10um" + str(i)
    infile = open(filename, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i] = new_dict['bpm_x_after_all'][-1]*10**5 + np.random.normal(scale=53)
    beamsize_y[i] = new_dict['bpm_y_after_all'][-1]*10**5 + np.random.normal(scale=15)

for i in range(10, 14):
    filename = "data/error_study_10um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i-1] = new_dict['bpm_x_after_all'][-1]*10**5 + np.random.normal(scale=53)
    beamsize_y[i-1] = new_dict['bpm_y_after_all'][-1]*10**5 + np.random.normal(scale=15)
for i in range(15, 95):
    filename = "data/error_study_10um" +  str(i)
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    beamsize_x[i-2] = new_dict['bpm_x_after_all'][-1]*10**5 + np.random.normal(scale=53)
    beamsize_y[i-2] = new_dict['bpm_y_after_all'][-1]*10**5 + np.random.normal(scale=15)

x = np.squeeze(beamsize_x)
x_d = np.linspace(x.min(), x.max(), 1000)
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=10, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])

# plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
f, ax = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [4,1], 'height_ratios': [1, 4]})

ax[0][0].fill_between(x_d, np.exp(logprob), alpha=0.5, color = 'steelblue', label="fit [BPM resolution 10 $\mu$m]")
# bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
ax[0][0].hist(beamsize_x, bins=20, histtype="step", density=True, linewidth=3, label="$\sigma_x$")
ax[0][0].axis('off')
ax[1][1].axis('off')
ax[0][1].axis('off')

x = np.squeeze(beamsize_y)
x_d = np.linspace(x.min(), x.max(), 1000)
kde.fit(x[:, None])
logprob = kde.score_samples(x_d[:, None])
# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=10, kernel='gaussian')
kde.fit(x[:, None])
ax[1][1].fill_betweenx(x_d, np.exp(logprob),alpha=0.5, color = 'steelblue')
ax[1][1].hist(beamsize_y, bins=20, orientation=u'horizontal', histtype="step",  density=True, linewidth=3)
# plt.xlabel("$\sigma_{x}$ [$\mu$m]")
data = {'x': np.squeeze(beamsize_x),'y': np.squeeze(beamsize_y)}
data = pd.DataFrame(data=data)
sns.kdeplot(data = data,
    x='x', y='y',
    ax=ax[1][0], fill=True
)

ax[0][0].set_xlim(ax[1][0].get_xlim())
ax[1][1].set_ylim(ax[1][0].get_ylim())
# plt.ylabel("Frequency", labelpad=10)
# plt.legend()
ax[1][0].set_xlabel('$\Delta_x$ [$\mu$m]', fontsize=30)
ax[1][0].set_ylabel('$\Delta_y$ [$\mu$m]', fontsize=30)
ax[1][0].plot([10,10], [-10, 10], color='tab:orange', linestyle='--')
ax[1][0].plot([-10,10], [10, 10], color='tab:orange', linestyle='--')
ax[1][0].plot([-10,-10], [-10, 10], color='tab:orange', linestyle='--')
ax[1][0].plot([-10,10], [-10, -10], color='tab:orange', linestyle='--')

ax[0][0].axvline(x=10, color='tab:orange', linestyle='--')
ax[1][1].axhline(y=10, color='tab:orange', linestyle='--')
ax[0][0].axvline(x=-10, color='tab:orange', linestyle='--')
ax[1][1].axhline(y=-10, color='tab:orange', linestyle='--')
plt.show()
#
#
#
#
# # sns.kdeplot(
# #     data=geyser, x="waiting", y="duration",
# #     fill=True, thresh=0, levels=100, cmap="mako",
# # )
#
#
# plt.figure(figsize=(10, 10))
# a = beamsize_x/5.75
# # low charge
# # array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.hist(beamsize_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_x$')
# a = beamsize_y/5.75
# # low charge
# array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.hist(beamsize_y, 15, density=False, histtype='step', linewidth=3, label='$\sigma_y$')
# plt.xlabel('$\sigma$ [$\mu$m]')
# plt.ylabel('Density')
# plt.legend()
#
# #########################################
# ########################################
# plt.figure(figsize=(10, 10))
# a = beamsize_x/5.75
# # low charge
# array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.hist(array_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_x$')
# a = beamsize_y/5.75
# # low charge
# array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.hist(array_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_y$')
# plt.xlabel('$Q_t$')
# plt.ylabel('Density')
# plt.legend()
# plt.title('Low charge - Distribution of quality factors: {0:.0f} seeds'.format(len(beamsize_x)))
#
#
# plt.figure(figsize=(10, 10))
# a = beamsize_x/5.75
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.hist(array_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_x$')
# a = beamsize_y/5.75
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.hist(array_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_y$')
# plt.xlabel('$Q_t$')
# plt.ylabel('Density')
# plt.legend()
# plt.title('High charge - Distribution of quality factors: {0:.0f} seeds'.format(len(beamsize_x)))
#
#
#
# ###################################################
# ####################################################
# beamsize_x = np.zeros(shape=(93, 1))
# beamsize_y = np.zeros(shape=(93, 1))
# for i in range(0, 9):
#     filename = "data/error_study_10um" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
#
# for i in range(10, 14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-1] = new_dict['beam_size_y_after_all']
# for i in range(15, 95):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-2] = new_dict['beam_size_y_after_all']
# plt.figure()
#
#
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# a = beamsize_x/5.75
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(93)), array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# a = np.linspace(0.6, 3.6, 50)
# array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.plot(a, array_x, '-', color="tab:blue")
# # array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
# # plt.plot(a, array_x, '-', color="tab:blue")
# # a = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.6]
# # b = [0.45, 0.69, 0.85, 0.93, 0.92, 0.88, 0.79, 0.67, 0.52, 0.25]
# # plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# a = [0.6, 1, 1.2 ,1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
# b= [0.23, 0.54, 0.67, 0.78, 0.82, 0.74, 0.61, 0.495, 0.41, 0.34]
# plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# plt.legend()
# plt.xlabel('$\sigma/\sigma_{matched}$')
# plt.ylabel('$Q_t$')
#
#
#
# beamsize_x = np.zeros(shape=(22, 1))
# beamsize_y = np.zeros(shape=(22, 1))
# for i in range(22):
#     filename = "data/error_study_10um_2" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_x/5.75
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(22)), array_x+errors_q, 'o', color='tab:red', label='$2\sigma_{matched}$')
# # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$1\sigma_{matched}$')
#
#
#
#
# beamsize_x = np.zeros(shape=(29, 1))
# beamsize_y = np.zeros(shape=(29, 1))
# for i in range(29):
#     filename = "data/error_study_10um_3" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_x/5.75
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(29)), array_x+errors_q, 'o', color='tab:green', label='$3\sigma_{matched}$')
# # # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$1\sigma_{matched}$')
# plt.legend()
# plt.title('Low charge - horizontal')
# plt.show()
#
#
# ############################################################
# ############################################################
# beamsize_x = np.zeros(shape=(93, 1))
# beamsize_y = np.zeros(shape=(93, 1))
# for i in range(0, 9):
#     filename = "data/error_study_10um" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
#
# for i in range(10, 14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-1] = new_dict['beam_size_y_after_all']
# for i in range(15, 95):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-2] = new_dict['beam_size_y_after_all']
# plt.figure()
#
#
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# a = beamsize_x/5.75
# # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(93)), array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# a = np.linspace(0.6, 3.6, 50)
# # array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# # plt.plot(a, array_x, '-', color="tab:blue")
# array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
# plt.plot(a, array_x, '-', color="tab:blue")
# a = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.6]
# b = [0.45, 0.69, 0.85, 0.93, 0.92, 0.88, 0.79, 0.67, 0.52, 0.25]
# plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# # a = [0.6, 1, 1.2 ,1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
# # b= [0.23, 0.54, 0.67, 0.78, 0.82, 0.74, 0.61, 0.495, 0.41, 0.34]
# # plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# plt.legend()
# plt.xlabel('$\sigma/\sigma_{matched}$')
# plt.ylabel('$Q_t$')
#
#
#
# beamsize_x = np.zeros(shape=(22, 1))
# beamsize_y = np.zeros(shape=(22, 1))
# for i in range(22):
#     filename = "data/error_study_10um_2" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_x/5.75
# a = a[a<4]
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(22)), array_x+errors_q, 'o', color='tab:red', label='$2\sigma_{matched}$')
#
#
#
#
# beamsize_x = np.zeros(shape=(29, 1))
# beamsize_y = np.zeros(shape=(29, 1))
# for i in range(29):
#     filename = "data/error_study_10um_3" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_x/5.75
# a = a[a<4]
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# # # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot(np.mean((np.squeeze(beamsize_x) + errors)/5.75)*np.ones(shape=(29)), array_x+errors_q, 'o', color='tab:green', label='$3\sigma_{matched}$')
# plt.legend()
# plt.title('High charge- Horizontal')
# plt.show()
#
# ###############################################################
# ##############################################################
#
#
#
#
#
# ###################################################
# ####################################################
# beamsize_x = np.zeros(shape=(93, 1))
# beamsize_y = np.zeros(shape=(93, 1))
# for i in range(0, 9):
#     filename = "data/error_study_10um" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
#
# for i in range(10, 14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-1] = new_dict['beam_size_y_after_all']
# for i in range(15, 95):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-2] = new_dict['beam_size_y_after_all']
# plt.figure()
#
#
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# a = beamsize_y/5.75
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# a = np.linspace(0.6, 3.6, 50)
# array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# plt.plot(a, array_x, '-', color="tab:blue")
# # array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
# # plt.plot(a, array_x, '-', color="tab:blue")
# # a = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.6]
# # b = [0.45, 0.69, 0.85, 0.93, 0.92, 0.88, 0.79, 0.67, 0.52, 0.25]
# # plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# a = [0.6, 1, 1.2 ,1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
# b= [0.23, 0.54, 0.67, 0.78, 0.82, 0.74, 0.61, 0.495, 0.41, 0.34]
# plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# plt.legend()
# plt.xlabel('$\sigma/\sigma_{matched}$')
# plt.ylabel('$Q_t$')
#
#
#
# beamsize_x = np.zeros(shape=(22, 1))
# beamsize_y = np.zeros(shape=(22, 1))
# for i in range(22):
#     filename = "data/error_study_10um_2" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_y/5.75
# beamsize_y = beamsize_y[a<5]
# a = a[a<5]
#
# errors = np.random.normal(size=(len(a)))
# errors_q = np.random.normal(size=(len(a)))*0.05
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$2\sigma_{matched}$')
# # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$1\sigma_{matched}$')
#
#
#
#
# beamsize_x = np.zeros(shape=(29, 1))
# beamsize_y = np.zeros(shape=(29, 1))
# for i in range(29):
#     filename = "data/error_study_10um_3" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_x/5.75
# beamsize_y = beamsize_y[a<5]
# a = a[a<5]
# errors = np.random.normal(size=(len(a)))
# errors_q = np.random.normal(size=(len(a)))*0.05
# # low charge
# array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$3\sigma_{matched}$')
# # # high charge
# # array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$1\sigma_{matched}$')
# plt.legend()
# plt.title('Low charge - Vertical')
# plt.show()
#
#
# ############################################################
# ############################################################
# beamsize_x = np.zeros(shape=(93, 1))
# beamsize_y = np.zeros(shape=(93, 1))
# for i in range(0, 9):
#     filename = "data/error_study_10um" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
#
# for i in range(10, 14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-1] = new_dict['beam_size_y_after_all']
# for i in range(15, 95):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-2] = new_dict['beam_size_y_after_all']
# plt.figure()
#
#
# errors = np.random.normal(size=(len(beamsize_x)))
# errors_q = np.random.normal(size=(len(beamsize_x)))*0.05
# a = beamsize_y/5.75
# # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:purple', label='$1\sigma_{matched}$')
# a = np.linspace(0.6, 3.6, 50)
# # array_x = 0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102
# # plt.plot(a, array_x, '-', color="tab:blue")
# array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
# plt.plot(a, array_x, '-', color="tab:blue")
# a = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.4, 2.8, 3.6]
# b = [0.45, 0.69, 0.85, 0.93, 0.92, 0.88, 0.79, 0.67, 0.52, 0.25]
# plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# # a = [0.6, 1, 1.2 ,1.4, 1.6, 2, 2.4, 2.8, 3.2, 3.6]
# # b= [0.23, 0.54, 0.67, 0.78, 0.82, 0.74, 0.61, 0.495, 0.41, 0.34]
# # plt.plot(a, b, 'x', color="tab:orange", markersize=12, linewidth=3)
# plt.legend()
# plt.xlabel('$\sigma/\sigma_{matched}$')
# plt.ylabel('$Q_t$')
#
#
#
# beamsize_x = np.zeros(shape=(22, 1))
# beamsize_y = np.zeros(shape=(22, 1))
# for i in range(22):
#     filename = "data/error_study_10um_2" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_y/5.75
# beamsize_y = beamsize_y[a<5]
# a = a[a<5]
# errors = np.random.normal(size=(len(beamsize_y)))
# errors_q = np.random.normal(size=(len(beamsize_y)))*0.05
# # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:red', label='$2\sigma_{matched}$')
#
#
#
#
# beamsize_x = np.zeros(shape=(29, 1))
# beamsize_y = np.zeros(shape=(29, 1))
# for i in range(29):
#     filename = "data/error_study_10um_3" + str(i)
#     infile = open(filename, 'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# a = beamsize_y/5.75
# beamsize_y = beamsize_y[a<5]
# a = a[a<5]
# errors = np.random.normal(size=(len(beamsize_y)))
# errors_q = np.random.normal(size=(len(beamsize_y)))*0.05
# # # low charge
# # array_x = np.squeeze(0.03942*a**6 - 0.5635*a**5 + 3.18364*a**4 - 8.87607*a**3 + 12.26352*a**2 - 7.21457*a**1 + 1.70102)
# # plt.plot((np.squeeze(beamsize_x) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$1\sigma_{matched}$')
# # high charge
# array_x = np.squeeze(0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516)
# plt.plot((np.squeeze(beamsize_y) + errors)/5.75, array_x+errors_q, 'o', color='tab:green', label='$3\sigma_{matched}$')
# plt.legend()
# plt.title('High charge- Vertical')
# plt.show()
#
# ###############################################################
# ##############################################################
#
#
#
#
# a = beamsize_y/5.75
# # low charge
# array_x = 0.0394*a**6 - 0.5635*a**5 + 3.1836*a**4 - 8.8761*a**3 + 12.264*a**2 - 7.2146*a**1 + 1.701
# # high charge
# array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
#
# plt.figure()
# plt.hist(array_x, 15, density=False, histtype='step', linewidth=3, label='$\sigma_y$')
# plt.xlabel('$Q_t$')
# plt.ylabel('Density')
# plt.title('Distribution of quality factors: {0:.0f} seeds'.format(len(beamsize_x)))
# plt.legend()
#
# # a = beamsize_y/5.75
# # array_x = 0.0446*a**6 - 0.6417*a**5 + 3.6249*a**4 - 10.058*a**3 + 13.818*a**2 - 8.1383*a**1 + 1.8893
# # array_x = 0.0286*a**5 - 0.3599*a**4 + 1.8067*a**3 - 4.5598*a**2 + 5.4853*a**1 - 1.5516
# # plt.figure()
# # plt.hist(array_x)
# #########################################################
# #########################################################
# #########################################################
# #########################################################
#
# beamsize_x = np.zeros(shape=(39,1))
# beamsize_y = np.zeros(shape=(39,1))
# plt.figure(figsize=(12,8))
# import pickle
# for i in range(0,39):
#     filename = "data/error_study_0um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
# # beamsize_x[39] = 6.6
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 1 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_y/5.75, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 1 $\mu$m")
#
#
# beamsize_x = np.zeros(shape=(48,1))
# beamsize_y = np.zeros(shape=(48,1))
# for i in range(0,9):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['beam_size_x_after_all']
#     beamsize_y[i] = new_dict['beam_size_y_after_all']
#
# for i in range(10,14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-1] = new_dict['beam_size_y_after_all']
# for i in range(15,50):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['beam_size_x_after_all']
#     beamsize_y[i-2] = new_dict['beam_size_y_after_all']
#
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 10 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_y/5.75, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 10 $\mu$m")
# plt.xlabel("$\sigma_{y}$ [$\mu$m]")
# plt.ylabel("Frequency", labelpad=10)
# plt.legend()
#
#
#
#
# ##################################################################
# ##################################################################
# ##################################################################
# ##################################################################
#
# beamsize_x = np.zeros(shape=(40,1))
# beamsize_y = np.zeros(shape=(40,1))
# plt.figure(figsize=(12,8))
# import pickle
# for i in range(0,39):
#     filename = "data/error_study_0um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i] = new_dict['bpm_y_after_all'][-1]
# beamsize_x[39] = 0
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 1 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_x, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 1 $\mu$m")
#
#
# beamsize_x = np.zeros(shape=(48,1))
# beamsize_y = np.zeros(shape=(48,1))
# for i in range(0,9):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i] = new_dict['bpm_y_after_all'][-1]
#
# for i in range(10,14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i-1] = new_dict['bpm_y_after_all'][-1]
# for i in range(15,50):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i-2] = new_dict['bpm_y_after_all'][-1]
#
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 10 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_x, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 10 $\mu$m")
# plt.xlabel("$\Delta x$ [$\mu$m]")
# plt.ylabel("Frequency", labelpad=10)
# plt.legend()
#
# #########################################################
# #########################################################
# #########################################################
# #########################################################
#
# beamsize_x = np.zeros(shape=(39,1))
# beamsize_y = np.zeros(shape=(39,1))
# plt.figure(figsize=(12,8))
# import pickle
# for i in range(0,39):
#     filename = "data/error_study_0um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i] = new_dict['bpm_y_after_all'][-1]
# # beamsize_x[39] = 6.6
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 1 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_y, bins=20, histtype="step", density=True,linewidth=3, label="BPM resolution 1 $\mu$m")
#
#
# beamsize_x = np.zeros(shape=(48,1))
# beamsize_y = np.zeros(shape=(48,1))
# for i in range(0,9):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i] = new_dict['bpm_y_after_all'][-1]
#
# for i in range(10,14):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-1] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i-1] = new_dict['bpm_y_after_all'][-1]
# for i in range(15,50):
#     filename = "data/error_study_10um" +  str(i)
#     infile = open(filename,'rb')
#     new_dict = pickle.load(infile)
#     infile.close()
#     beamsize_x[i-2] = new_dict['bpm_x_after_all'][-1]
#     beamsize_y[i-2] = new_dict['bpm_y_after_all'][-1]
#
# x = np.squeeze(beamsize_x)
# x_d = np.linspace(x.min(), x.max(), 1000)
# # instantiate and fit the KDE model
# kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
# kde.fit(x[:, None])
#
# # score_samples returns the log of the probability density
# logprob = kde.score_samples(x_d[:, None])
#
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="$\sigma_y$")
#
# logprob = kde.score_samples(x_d[:, None])
# # plt.fill_between(x_d, np.exp(logprob), alpha=0.5, label="fit [BPM resolution 10 $\mu$m]")
# # bins = np.arange(beamsize_x.min()-0.025, beamsize_x.max() + 0.025, 0.05)
# plt.hist(beamsize_y, bins=20, histtype="step", density=True, linewidth=3, label="BPM resolution 10 $\mu$m")
# plt.xlabel("$\Delta y$ [$\mu$m]")
# plt.ylabel("Frequency", labelpad=10)
# plt.legend()
# # plt.figure()
# # plt.plot(bpm_pos, bpm_x_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
# # plt.plot(bpm_pos, bpm_x_after * 10 ** 3, '-', color="darkorange", label=None)
# # plt.plot(bpm_pos, bpm_y_before * 10 ** 3, '-', color=[0, 0.324219, 0.628906], label=None)
# # plt.plot(bpm_pos, bpm_y_after * 10 ** 3, '-', color="darkorange", label=None)
#
#
