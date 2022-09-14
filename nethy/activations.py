import numpy as np
import math


__all__ = [
	'relu', 'd_relu',
	'sigmoid', 'd_sigmoid',
	'softmax', 'd_softmax',
	'crossentropy', 'd_crossentropy',
	'logistic', 'd_logistic',
	'active','d_active',
	'losses','d_losses'
]


def relu(array):
	return np.array([max(0, x) for x in array])


def d_relu(array):
	return np.array([
		1 if x > 0 else 0 for x in array
	])
	
			
def sigmoid_x(x):
	return 1/(1 + math.exp(-x))	

		
def d_sigmoid_x(x):
	sig = sigmoid_x(x)
	return sig * (1 - sig)
	

def sigmoid(array):
	return np.array([sigmoid_x(x) for x in array])
	

def d_sigmoid(array):
	return np.array([d_sigmoid_x(x) for x in array])
	

def softmax(array):
	sum = np.array([math.exp(x) for x in array]).sum()
	return np.array([
		math.exp(x)/sum 
		for x in array
	])
	
	
def d_softmax(array):
	sum = np.array([math.exp(x) for x in array]).sum()
	psum = math.pow(sum, 2)
	return np.array([
		(math.exp(x) * (sum - math.exp(x))) / psum
		for x in array
	])


def crossentropy(tar, out, count):
	size = len(tar)
	res = np.array([
		(tar[i] * math.log10(out[i])) + 
		((1 - tar[i]) * math.log10(1 - out[i])) 
		for i in range(size)
	])
	return res.sum()*(-1/count)
	
	
def d_crossentropy(tar, out, count):
	size = len(tar)
	res = np.array([
		(tar[i] * (1/out[i])) +
		(1 - tar[i]) * (1 / (1 - out[i]))
		for i in range(size)
	]) * (-1)
	return res
	

def logistic(tar, out):
	pass
	
	
def d_logistic():
	pass
	
	
active = {
	'relu': relu,
	'sigmoid': sigmoid,
	'softmax': softmax,
}

d_active = {
	'relu': d_relu,
	'sigmoid': d_sigmoid,
	'softmax': d_softmax,
}

losses = {
	'crossentropy': crossentropy,
	'logistic': logistic
}

d_losses = {
	'crossentropy': d_crossentropy,
	'logistic': logistic
}
