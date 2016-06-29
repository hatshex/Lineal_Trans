#-*-encoding: utf8 --*-
from __future__ import division
from pickle import load,dump
import numpy as np
from collections import defaultdict
from scipy.linalg import eig

EN, ES, en, es = load(open('W2Ven-es.p','r'))
seed = [w.split('\t') for w in open('seed.txt','r').read().lower().split('\n')]

seeded = []
for i,w in enumerate(seed):
	if w[0] in en.keys() and w[1] in es.keys():
		seeded.append(w)

def knn(M,k):
	if k == 0:
		words = en
	elif k == 1:
		words = es
	
	neigs = defaultdict(list)
	for u in words.keys():
		neigs[words[u]] = [words[w[k]] for w in seeded]
	return neigs

#Define un kernel euclideano inverso	
def ker(x,y,sigma=1):
	dist = np.linalg.norm(x-y)
	print dist
	if dist != 0.0:
		return sigma/(dist)
	else:
		return 0.0

#Genera la matriz Laplaciana	
def L(M,k):
	A = np.zeros((len(M),len(M)))
	H = knn(M,k)
	
	for u,vec in H.iteritems():
		for v in vec:
			d = ker(M[u],M[v])
			A[u,v] = d
	A = A + A.T
	D = np.diag( [sum(v) for v in A] )
	#print A
	#return A
	return D-A
	
L_EN = L(EN,0)
L_ES = L(ES,1)

evals1, enpvecs = eig(L_EN)
evals2, eshvecs = eig(L_ES)

New_EN = []
New_ES = []
for e,n in seeded:
	print e,n,evals1[en[e]].real, n,evals2[es[n]]
	b_en = enpvecs[en[e]]
	b_es = eshvecs[es[n]]
	New_EN.append(b_en)
	New_ES.append(b_es)
	
EN2 = np.array(New_EN).T
ES2 = np.array(New_ES).T

print EN2.shape, ES2.shape

out = [EN2, ES2, en, es]
file = open('dsm_seed.p','w')
dump(out,file)
file.close()