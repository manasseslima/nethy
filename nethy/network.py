import numpy as np
import math
from activations import *

__all__ = ('run_layer', 'update_weigths')


def run_layer(input, weight, bias, activation):
	lin = np.dot(weight, input.transpose()) + bias
	lout = activation(lin)
	return lin, lout


def update_weigths(w, d, r=0.01):
	return np.array([
		w[i][j] - (r * d[i][j])
		for i in range(3) 
		for j in range(3)
	])


class Layer():
	def __init__(self, size=3, activation='relu', input_size=3):
		self.size = size
		self.active = active.get(activation)
		self.derive = d_active.get(activation)
		self.w = np.random.random((size, input_size))
		self.bias = np.ones((input_size,1))
		self.input_size = input_size
		
	def fit(self):
		print(f'layer fit {self.index}')
		
	def forward(self, input):
		inp = np.array([input])
		self.input = input
		self.i = np.dot(self.input, self.w.transpose()) + self.bias
		self.o = self.active(self.i[0])
		return self.o
		
	def backpropagation(self, derr):
		oi = self.derive(self.input)
		iw = np.array([self.input])
		derr = np.array([derr])
		oi = np.array([oi])
		t = (derr * oi.transpose())
		ew = iw * derr.transpose()
		de = np.sum(t * self.w.transpose(), axis=1)
		self.w = self.w - (ew * 0.01)
		print(self.w)
		return de
		
	def get_out(self):
		pass


class Network():
	def __init__(self, input_size=0):
		self.layers = []
		self.input_size = input_size
		
	def add(self, size, activation='relu'):
		if len(self.layers) == 0:
			input_size = self.input_size
		else:
			input_size = self.layers[-1].size
		layer = Layer(size, activation, input_size)
		layer.index = len(self.layers)
		self.layers.append(layer)
		
	def fit(self, train, labels, loss='crossentropy', epochs=10):
		for (i, data) in enumerate(train):
			print(f'\nFORWARD {i}\n')
			out = data
			for (j, layer) in enumerate(self.layers):
				print(f'LAYER {j}')
				print(layer.w)
				out = layer.forward(out)
				print(out)
			error = losses.get(loss)(labels[i], out, len(train))
			print(f'ERROR {i} {error}')
			print(f'\nBACKPROPAGATION {i}\n')
			derr = d_losses.get(loss)(labels[i], out, len(train))
			print(derr)
			for layer in reversed(self.layers):
				print(f'LAYER {layer.index}')
				derr = layer.backpropagation(derr)
				print(derr)
	
	def run(self, data):
		pass
		
	def __str__(self):
		return f'{len(self.layers)}'
		
		
if __name__ == '__main__':
	import console
	console.clear()
	train = np.array((
		[0.1, 0.2, 0.7],
		[0.3, 0.5, 0.2],
		[0.6, 0.1, 0.3],
		[0.1, 0.5, 0.8],
	))
	labels = np.array((
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[0.0, 1.0],
	))
	nw = Network(3)
	nw.add(5, 'relu')
	nw.add(4, 'sigmoid')
	nw.add(2, 'softmax')
	nw.fit(train, labels)
	print(nw)
