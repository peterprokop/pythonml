import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from time import sleep

data = genfromtxt('ex2data1.txt', delimiter=',')

x = data[:, 0:2]
y = data[:, 2]

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.plot(x[neg], linestyle="", marker="x", color="r")
plt.plot(x[pos], linestyle="", marker="o", color="g")

plt.xlabel('label')

plt.show()
