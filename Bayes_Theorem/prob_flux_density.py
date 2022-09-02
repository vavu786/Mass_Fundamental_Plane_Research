import matplotlib.pyplot as plt
import numpy as np
from math import factorial as fact
import scipy.integrate as integrate

def posterior(S, sigma, data):
	non_norm_post = lambda s: np.exp((-1.0/(2.0 * sigma)) * sum([((datapt - s) ** 2.0) for datapt in data])) * (s ** (-5.0/2.0))
	K = integrate.quad(non_norm_post, 1, np.inf)[0]
	return (1/K) * non_norm_post(S)
	
def main():
	sigma = 1
	num_counts = 100
	data = [2, 1.3, 3, 1.5, 2, 1.8]
	
	x_ax = np.linspace(1, 4, 40)
	y_ax_n2 = np.asarray([posterior(x, sigma, data[:2]) for x in x_ax])
	y_ax_n4 = np.asarray([posterior(x, sigma, data[:4]) for x in x_ax])
	y_ax_n6 = np.asarray([posterior(x, sigma, data[:6]) for x in x_ax])
	
	fig, ax = plt.subplots()
	
	ax.set_xlim(1, 4)
	ax.set_ylim(0, 2)
	
	ax.set_xlabel("Flux density, S")
	ax.set_ylabel("Probability density")
		
	ax.plot(x_ax, y_ax_n2, c='r', label="data points = 2")
	ax.plot(x_ax, y_ax_n4, c='tab:orange', label="data points = 4")
	ax.plot(x_ax, y_ax_n6, c='g', label="data points = 6")
	
	fig.legend(*ax.get_legend_handles_labels(), loc=7)
	plt.show()
	
if __name__ == "__main__":
	main()
