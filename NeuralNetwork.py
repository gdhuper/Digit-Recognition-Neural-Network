import sys
import math
import numpy as np
from numba import jit
import time
from matrixMul import dotProduct


class NeuralNetwork():

	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.inodes = inputNodes
		self.hnodes = hiddenNodes
		self.onodes = outputNodes

		#learning rate 
		self.lr = learningRate



	def train(self):
		pass


	def query(self):
		pass


	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	def dotproduct(self, weights, inputs):
		#dot product using numpy
		return np.dot(weights, inputs) 

		# rowsWeights = len(weights)
		# colsWeights = len(weights[0])

		# rowsInputs = len(inputs)
		# colsInputs = len(inputs[0])

		# rowsWeights = weights.shape[0]
		# colsWeights = weights.shape[1]

		# rowsInputs = inputs.shape[0]
		# colsInputs = inputs.shape[1]


		# if colsWeights != rowsInputs:
		# 	return None

		# result = [[0 for r in range(colsInputs)] for c in range(rowsWeights)] #initialize result matrix

		# for i in range(rowsWeights):
		# 	for j in range(colsInputs):
		# 		for k in range(colsWeights):
		# 			result[i][j] += weights[i][k] * inputs[k][j]
		# return result

@jit
def main():
	nn = NeuralNetwork("", "", "", "")
	#print(nn.sigmoid(1.05))

	#numpy 2d array creation: clean up later
	# weights = np.matrix('1 2; 3 4')
	# inputs = np.matrix('5 6; 7 8')

	#2d array creation pure python: clean up later
	# weights = [[1, 2], [3, 4]]
	# inputs = [[1, 2], [3, 4]]


	a = np.random.randint(255, size=(100,784))
	b = np.random.randint(255, size=(784,1))

	
	start_time = time.time()
	print(nn.dotproduct(a, b))
	t1 = (time.time() - start_time)
	print("--- %s seconds ---" % t1)

	start_time2 = time.time()
	print(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
	dotProduct(a, b)
	t2 = (time.time() - start_time2)
	print("--- %s seconds ---" % t2)

	print("Pure Python: ", t1)
	print("OpenCL: ", t2)
	print("Pure Python is faster" if t1 < t2 else "OpenCl is faster")

if __name__ == '__main__':
	main()
