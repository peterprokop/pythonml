import numpy as np
from numpy import genfromtxt

def computeCost(x, y, theta):
	result = 0
	m = x.shape[0]
	for j in range(m):
		tmp = theta[0] * x[j][0] + theta[1] * x[j][1] - y[j]
		result += tmp * tmp
	return result / (2 * m)

def gradientDescent(x, y, original_theta, alpha, iterations):
	theta = original_theta;
	m = x.shape[0] # number of training examples
	n = x.shape[1]
	for _x in range(iterations):
		tmp_theta = theta
		for j in range(n):
			sum = 0
			for i in range(m):
				tmp = theta[0] * x[i][0] + theta[1] * x[i][1];
				sum += (tmp - y[i]) * x[i][j];
			tmp_theta[j] -= sum * alpha / m
		theta = tmp_theta
		print("theta", theta)
		print("computeCost", computeCost(x, y, theta))

file_contents = genfromtxt('ex1data1.txt', delimiter=',')

x = np.empty([file_contents.shape[0], 2])
x[:, 0] = 1
x[:, 1] = file_contents[:, 0] 

y = file_contents[:, 1]

iterations = 1500
alpha = 0.01
theta = [0, 0]
gradientDescent(x, y, theta, alpha, iterations)
