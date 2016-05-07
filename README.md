from __future__ import division
import numpy as np
np.set_printoptions(precision=2, suppress=True)
np.set_printoptions(threshold='nan')
import matplotlib.pyplot as plt

m = np.array([[1,0,0],[0,1,0],[0,0,1]])

def svd(M,k):
	M = np.dot(M,M.T)
	(n,m) = M.shape
	W,T,P = np.linalg.svd(M)
	numDim = k
	T = np.diag(T)[:numDim, :numDim]
	W = W[:, :numDim]
	
	lowDim = np.zeros((n,numDim))
	for i,v in enumerate(M):
		lowDim[i] = np.dot(np.linalg.inv(T), np.dot(W.T,v))
	return lowDim

def relax (M,k):
	numDim = k
	A = np.dot(M-M.T,M-M.T)
	D = np.diag([sum(a_i) for a_i in A])
	L = D-A
	evals, evecs = np.linalg.eig(L)
	edict =sorted(zip(evals,evecs), key=lambda x: x[0])
	lowDim = np.array([v[1] for v in edict[1:k+1]]).T
	return lowDim

def plot_low (V,low=svd,labels=None,form='b',mark='o'):
	W = low(V,2)
	if labels == None:
		plt.scatter(W[:,0],W[:,1],c=form,marker=mark)
		plt.show()
	else:
		i = 0
		plt.scatter(W[:,0], W[:,1],c=form,marker=mark,s=50.0)
		for label,x,y in zip(labels, W[:,0], W[:,1]):
			plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-10,10), textcoords='offset points', ha= 'center', va='bottom', bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0))#, arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
			i += 1
		plt.show()
		
#plot_low(m,relax)
