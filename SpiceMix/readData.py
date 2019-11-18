import numpy as np
import os, pickle
from matplotlib import pyplot as plt
from util import dataFolder, path2dir

def readGeneExpression(filename):
	filename = os.path.join(path2dir, '..', 'data', filename)
	# print(filename)
	if os.path.exists(filename):
		return np.loadtxt(filename, dtype=np.float)
	if filename.endswith('.txt'):
		filename = filename[:-4] + '.pkl'
		# print(filename)
		if os.path.exists(filename):
			with open(filename, 'rb') as f:
				return pickle.load(f)
	return None

def readNeighborhood(filename, N):
	with open(os.path.join(path2dir, '..', 'data', filename)) as f:
		edges = np.array(list(list(map(int, _.strip().split())) for _ in f))
	assert np.all(0 <= edges) and np.all(edges < N)
	E = [[] for _ in range(N)]
	for (u, v) in edges:
		E[u].append(v)
		E[v].append(u)
	return E

def readGeneList(dataset, filename):
	with open(os.path.join(path2dir, '..', 'data', dataset, 'files', filename), 'r') as f:
		return np.array(f.read().strip().split())

def loadImage(dataset, filename):
	path2file = os.path.join(path2dir, '..', 'data', dataset, 'files', filename)
	if os.path.exists(path2file): return plt.imread(path2file)
	else: return None

def readDataSet(dataset, exper_list, use_spatial, neighbor_suffix=None, expression_suffix=None, **kwargs):
	if neighbor_suffix is None: neighbor_suffix = ''
	else: neighbor_suffix = '_' + neighbor_suffix
	if expression_suffix is None: expression_suffix = ''
	else: expression_suffix = '_' + expression_suffix

	YTs = [
		readGeneExpression(os.path.join(dataFolder(dataset), f'expression_{iexpr}{expression_suffix}.txt'))
		for iexpr in exper_list
	]
	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)
	Es = [
		readNeighborhood(os.path.join(dataFolder(dataset), f'neighborhood_{iexpr}{neighbor_suffix}.txt'), N)
		if u else
		[[] for _ in range(N)]
		for iexpr, N, u in zip(exper_list, Ns, use_spatial)
	]

	assert len(use_spatial) == len(YTs)

	Es_empty = [sum(map(len, E)) == 0 for E in Es]

	print(f'shapes of YTs = {[_.shape for _ in YTs]}')
	print('Hash of YTs = {}'.format('\t'.join([hex(hash(YT.tobytes())) for YT in YTs])))

	return YTs, Es, Es_empty
