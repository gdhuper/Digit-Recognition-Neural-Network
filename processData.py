import numpy as np
import sys
from numba import jit


@jit
def processData():
	data = np.genfromtxt('./data/mnist_test.csv', delimiter=',')
	vectors = []

	for line in data[:1]:
		vectors.append(np.array(line))

	print(len(vectors[0]))


if __name__ == '__main__':
	processData()