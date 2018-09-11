# function g = sigmoid(z)
# %SIGMOID Compute sigmoid functoon
# %   J = SIGMOID(z) computes the sigmoid of z.

# % You need to return the following variables correctly 
# g = ones(size(z));

# % ====================== YOUR CODE HERE ======================
# % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
# %               vector or scalar).

# g = 1 + (exp(1) * g) .^ (-z);
# g = ones(size(z)) ./ g;


# % =============================================================

# end

import numpy as np

def sigmoid(z):
	# This will return array for scalar
	shape = 1
	if hasattr(z, "shape"):
		shape = z.shape
	g = np.ones(shape)
	g = np.exp(1) * g
	g = np.power(g, -z) + 1
	g = np.divide(np.ones(shape), g)

	return g

# function [J, grad] = costFunction(theta, X, y)
# %COSTFUNCTION Compute cost and gradient for logistic regression
# %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
# %   parameter for logistic regression and the gradient of the cost
# %   w.r.t. to the parameters.

# % Initialize some useful values
# m = length(y); % number of training examples
# n = size(X, 2)
# % You need to return the following variables correctly 
# J = 0;
# grad = zeros(size(theta));

# %theta
# %X
# %y
# h = sigmoid(X * theta);
# %h
# s1 =  -y .* log(h);
# s2 = - (1-y) .* log(1-h);
# J = 1/m * sum( s1 + s2 );
# %J

# for featureIndex = 1:n
#     %X(:, featureIndex)
#     grad(featureIndex) = 1/m * sum( (h-y)'*X(:, featureIndex) )
# end
# % =============================================================

def costFunction(theta, x, y):
	m = y.shape[0]
	n = x.shape[1]
	j = 0
	grad = np.zeros(theta.shape)
	h = sigmoid(x * theta)
	s1 = np.multiply(-y, np.log(h))
	s2 = -np.multiply(1 - y, np.log(1 - h))
	J = 1/m * np.sum(s1 + s2)
	for featureIndex in range(n):
		grad[featureIndex] = 1/m * np.sum(np.transpose(h - y) * x[:, featureIndex])
	return (J, grad)

print(sigmoid(-99))
print(sigmoid(1))
print(sigmoid(9))
print(sigmoid(50))
print(sigmoid(99))
# print(sigmoid(1.22))
# print(sigmoid(np.ones(1)))
# print(sigmoid(np.ones((1,1))))