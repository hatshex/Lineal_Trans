from we import *
import matplotlib.pyplot as plt
from lowDims import *
import numpy as np

sentences1 = ['gato huye perro', 'perro come gato', 'ruedor huye gato','gato come raton','ruedor come','raton come','raton huye','ruedor huye','gato come','perro come']

sentences2 = ['cat runs dog', 'dog eats cat', 'mouse run car','cat eat mouse','mouse eats','mouse ears','mouse runs','mouse runs','cat eat','dog eats']

def Mikolov2013(V1,V2):
	return np.linalg.lstsq(V1, V2, -1)[0].T

Sp, es = embs(sentences1)
En, ing = embs(sentences2)



v_es = ['gato','perro','raton']
v_en = ['cat','dog','mouse']

V1 = np.zeros((len(v_es),100))
V2 = np.zeros((len(v_es),100))

for i,w in enumerate(v_es):
	 V1[i] = Sp[es[w]]
	 V2[i] = En[ing[v_en[i]]]
	 
plot_words(svd(V1,2),v_es,'v')
plot_words(svd(V2,2),v_en,'^')


W = Mikolov2013(V1,V2)

T = np.zeros((len(v_es),100))

for i,v in enumerate(V1):
	T[i] = np.dot(W,v)
	
plot_words(svd(T,2),v_es,'o','red','top')

plt.show()
