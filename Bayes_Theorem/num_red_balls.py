import matplotlib.pyplot as plt
import numpy as np
from math import factorial as fact
	
def chose(n, k):
    return fact(n) / (fact(k) * fact(n-k))

def posterior(N, T, R, tot):
	prior = np.random.randint(1, tot+1)
	likelihood = lambda n: chose(T, R) * ((n / tot)**R) * (((tot - n) / tot)**(T - R))
	norm_factor = sum( likelihood(n) * prior for n in range(tot+1) )
	posterior = (prior * likelihood(N)) / norm_factor
		
	return posterior
	
def main():
	T = 5
	R = 3
	tot = 10
	
	x_ax = np.arange(0, tot+1, 1)
	y_ax = np.asarray([posterior(x, T, R, tot) for x in x_ax])
	
	fig, ax = plt.subplots()
	
	ax.set_xlim(0, 10)
	ax.set_ylim(0, 0.7)
	ax.set_xlabel("Number of red balls")
	ax.set_ylabel("Probability")
		
	ax.plot(x_ax, y_ax)
	ax.scatter(x_ax, y_ax, c='r')
	plt.show()
	
if __name__ == "__main__":
	main()
