import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym

#class BeamGeneration(Beam_pars):
def Beam_Generator(Beam_pars):
  # Define the mean and covariance matrix for the Gaussian distribution
  betx=Beam_pars[0]
  bety=Beam_pars[1]
  alphax=Beam_pars[2]
  alphay=Beam_pars[3]
  emix=Beam_pars[4]
  emiy=Beam_pars[5]
  c=299792458; #m/s^2
  E0=938.27208816 * c**2 * 10**6 #eV
  Ek= 430 * 10**6 #eV
  dp_p=0.1/100;
  mean = np.array([0, 0, 0, 0, 0, np.sqrt(((E0+Ek)**2 - E0**2)/(c**2))])
  print(mean[5])

  gammax = (1+alphax**2)/betx
  gammay = (1+alphay** 2)/bety
  covariance_matrix = np.array([
      [emix*betx, 0, 0, 0, 0, 0],
      [0, emix*gammax, 0, 0, 0, 0],
      [0, 0, emiy*bety, 0, 0, 0],
      [0, 0, 0, emiy*gammay, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, mean[5]*dp_p]
  ])

  # Number of samples to generate
  num_samples = 1000

  # Generate random samples from the Gaussian distribution
  samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)

  # Separate the dimensions
  x = samples[:, 0]
  px = samples[:, 1]
  y = samples[:, 2]
  py = samples[:, 3]
  z = samples[:, 4]
  pz = samples[:, 5]

  # Create a 4D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(x, y, px, c=py, cmap='viridis', marker='o')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('px')

  plt.show()

  #np.savetxt('gaussian_distribution.txt', samples, delimiter=', ')
  #Save the data 
  with open("distr/Beam_Distribution.tfs", "w") as file:
  	for i in range(len(x)):
  		line = f"{x[i]}\t{y[i]}\t{z[i]}\t{px[i]}\t{py[i]}\t{pz[i]}\n"
  		file.write(line)
