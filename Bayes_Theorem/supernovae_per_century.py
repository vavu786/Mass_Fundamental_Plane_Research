import matplotlib.pyplot as plt
import numpy as np
from math import factorial as fact
import scipy.integrate as integrate
	
def chose(n, k):
    return fact(n) / (fact(k) * fact(n-k))

def posterior(P, num_centuries, num_snovae):
	prior = np.random.uniform(0, 1)
	likelihood = lambda p: chose(num_centuries, num_snovae) * (p**num_snovae) * ((1 - p)**(num_centuries - num_snovae))
	norm_factor = integrate.quad(likelihood, 0, 1)[0] * prior
	posterior = (prior * likelihood(P)) / norm_factor
		
	return posterior
	
def main():
	num_centuries = 10
	num_snovae = 4
	
	x_ax = np.arange(0, 1, 0.01)
	y_ax = np.asarray([posterior(x, num_centuries, num_snovae) for x in x_ax])
	
	fig, ax = plt.subplots()
	
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 3)
	ax.set_xlabel("Number of supernovae per century")
	ax.set_ylabel("Probability density")
	
		
	ax.plot(x_ax, y_ax)
	ax.scatter(x_ax, y_ax, c='r')
	plt.show()
	
if __name__ == "__main__":
	main()
