from we import *
import matplotlib.pyplot as plt
from lowDims import *
import numpy as np
from word2vec import *
from itertools import chain
from operator import itemgetter
from math import fabs
from collections import Counter
from pickle import dump

def embs (file):
	sents = open(file,'r').read().lower().split('\n')
	orac = [i.split(' ') for i in sents]
	words = list(chain(*[orac]))
	model = Word2Vec(words, size=100, window=5, min_count=10, workers=4)
	return model,words

esp, esp_sents = embs('todo.esp.lematizado')
nah, nah_sents = embs('todo.nah.segmentado')

print len(esp.vocab)
print len(nah.vocab)

def Mikolov2013(V1,V2):
	return np.linalg.lstsq(V1, V2, -1)[0].T


seed = open('seed.txt','r').read().lower().split('\n')
#print seed


V1 = np.zeros((len(seed),100))
V2 = np.zeros((len(seed),100))

for i,w in enumerate(seed):
	try:
		e,n = w.split('\t')
		V1[i] = esp[e]
		V2[i] = nah[n]
	except:
		pass
	 
#plot_words(svd(V1,2),v_es,'v')
#plot_words(svd(V2,2),v_en,'^')


W = Mikolov2013(V1,V2)
print W.shape

def get_close(s1,s2):
	cands = []
	for w1 in s1:
		sims = {}
		for w2 in s2:
			#print w1,w2 
			sim = np.dot(esp[w1],nah[w2]) / ( np.linalg.norm(esp[w1])*np.linalg.norm(nah[w2]) )
			sims[w2] = fabs(sim)
		#print type(sims)
		cands.append((w1,  max(sims.iteritems(), key=itemgetter(1))[0]) )

	return cands


def cos(u,v):
	return np.dot(u,v) / ( np.linalg.norm(u)*np.linalg.norm(v) )
#print esp.vocab
for w in esp.vocab:
	context = []
#	print w
	for s in zip(esp_sents,nah_sents):
#		print s
		if w in s[0]:
			context.append(s[1])
#
	cands = {}
	for w2,f in Counter( list(chain(*context)) ).items():
		try:
			cands[w2] = fabs(cos(esp[w], nah[w2]))
		except:
			pass

	vals = sorted(cands.iteritems(), key=itemgetter(1),reverse=True)
	print w, vals[0][0], vals[1][0], vals[2][0]


V_esp = {}
V_nah = {}
for w in esp.vocab.keys():
	V_esp[w] = esp[w]
for w in nah.vocab.keys():
	V_nah[w] = nah[w]

salida = [V_esp,V_nah,W]
	
out = open('datos.p','w')
dump(salida,out)
