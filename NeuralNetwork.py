import sys
import math
import numpy as np

class NeuralNetwork():


	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.inodes = inputNodes
		self.hnodes = hiddenNodes
		self.onodes = outputNodes

		#learning rate 
		self.lr = learningRate



	def train(self):
		print("stuff")


	def query(self):
		pass


	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def dotproduct(self, weights, inputs):
		#dot product using numpy
		#return np.dot(weights, inputs) 
		rowsWeights = len(weights)
		colsWeights = len(weights[0])

		rowsInputs = len(inputs)
		colsInputs = len(inputs[0])

		if colsWeights != rowsInputs:
			return None

		result = [[0 for r in range(colsInputs)] for c in range(rowsWeights)] #initialize result matrix

		for i in range(rowsWeights):
			for j in range(colsInputs):
				for k in range(colsWeights):
					result[i][j] += weights[i][k] * inputs[k][j]

		return result


def main():
	nn = NeuralNetwork("", "", "", "")
	#print(nn.sigmoid(1.05))
	# weights = np.matrix('1 2; 3 4')
	# inputs = np.matrix('5 6; 7 8')

	weights = [[1, 2], [3, 4]]
	inputs = [[1, 2], [3, 4]]
	print(nn.dotproduct(weights, inputs))


if __name__ == '__main__':
	main()
