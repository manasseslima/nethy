import numpy as np
import console
from activations import *
from network import *


console.clear()


inputs = np.array([0.1, 0.2, 0.7])
labels = np.array([1.0, 0.0, 0.0])
wij = np.array([
	[0.1, 0.2, 0.3],
	[0.3, 0.2, 0.7],
	[0.4, 0.3, 0.9],
	[0.5, 0.7, 0.6],
	[0.8, 0.3, 0.4],
])
b1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
wjk = np.array([
	[0.2, 0.3, 0.5, 0.3, 0.7],
	[0.3, 0.5, 0.7, 0.8, 0.2],
	[0.6, 0.4, 0.8, 0.5, 0.1],
	#[0.2, 0.8, 0.2, 0.4, 0.1],
])
b2 = np.array([1.0, 1.0, 1.0])
wkl = np.array([
	[0.1, 0.4, 0.8],
	[0.3, 0.7, 0.2],
	[0.4, 0.2, 0.1],
])
b3 = np.array([1.0, 1.0, 1.0])
	
	
h1in, h1out = run_layer(inputs, wij, b1, relu)
h2in, h2out = run_layer(h1out, wjk, b2, sigmoid)
oin, oout = run_layer(h2out, wkl, b3, softmax)
print('oout', oout)

error = crossentropy(labels, oout, 1)
print('error', error)


# BACKPROPQGATION
print('BACKPROPAGATION')
print('===========================')
print('\n', 'OUT', '+++++++++++++++++++++')
e_oo = d_crossentropy(labels, oout, 1)
print('e_oo:', e_oo)

oo_oi = d_softmax(oin)
print('oo_oi:', oo_oi)
oi_wkl = h2out * np.ones((3,3))
print('oi_wkl:', oi_wkl)
e_wkl = (e_oo * oo_oi).dot(oi_wkl)
print('e_wkl',e_wkl)
wkld = wkl - (e_wkl * 0.01)
print('wkld', wkld)



print('\n','H2', '+++++++++++++++++++++++')
e_h2o = np.sum(e_oo * oo_oi * wkl.transpose(), axis=1)

print('e_h2o:', e_h2o)
h2o_h2i = d_sigmoid(h2in)
print('h2o_h2i:', h2o_h2i)
h2i_wjk = h1out * np.ones((3,5))
print('h2i_wjk:', h2i_wjk)
e_wjk = (e_h2o *h2o_h2i).dot(h2i_wjk)
print('e_wjk', e_wjk)
wjkd = wjk - (e_wjk * 0.01)
print('wjkd', wjkd)



print('\n','H1', '+++++++++++++++++++++++')
e_h1o = np.sum(e_h2o * h2o_h2i * wjk.transpose(), axis=1)

print('e_h1o', e_h1o)
h1o_h1i = d_relu(h1in)
print('h1o_h1i', h1o_h1i)
h1i_wij = inputs * np.ones((5,3))
print('h1i_wij', h1i_wij)
e_wij = (e_h1o * h1o_h1i).dot(h1i_wij)
print('e_wij', e_wij)
wijd = wij - (e_wij * 0.01)
print('wijd', wijd)


'''
print('layer 1')
print('h1in:', h1in)
print('h1out:', h1out)

print('layer 2')
print('h2in:', h2in)
print('h2out:', h2out)

print('out')
print('oin:', oin)
print('oout:', oout)
'''

print('error:', error)



if __name__ == '__main__':
	print('Nethy')
