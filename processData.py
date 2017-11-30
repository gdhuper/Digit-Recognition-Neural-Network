import numpy as np
import sys
from numba import jit


@jit
def processData():
	data = np.genfromtxt('./data/mnist_test.csv', delimiter=',')
	vectors = []

	for line in data[:2]:
		vectors.append(np.array(line))

	print(vectors)


if __name__ == '__main__':
	processData()