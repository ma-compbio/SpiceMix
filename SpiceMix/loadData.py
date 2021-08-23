import os, pickle, logging
from matplotlib import pyplot as plt

import numpy as np

from util import print_datetime, parseSuffix


def loadExpression(filename):
	if filename.suffix == '.pkl':
		with open(filename, 'rb') as f:
			expr = pickle.load(f)
	elif filename.suffix == '.txt':
		expr = np.loadtxt(filename, dtype=np.float)
	else:
		raise ValueError(f'Invalid file format {filename}')
	N, G = expr.shape
	logging.info(f'{print_datetime()}Loaded {N} cells and {G} genes from {filename}')
	return expr


def loadEdges(filename, N):
	edges = np.loadtxt(filename, dtype=np.int)
	assert edges.shape[1] == 2, f'Detected an edge that does not contain two nodes'
	assert np.all(0 <= edges) and np.all(edges < N), f'Node ID exceeded range [0, N)'
	edges = np.sort(edges, axis=1)
	assert np.all(edges[:, 0] < edges[:, 1]), f'Detected {(edges[:, 0] == edges[:, 1]).sum()} self-loop(s)'
	if len(np.unique(edges, axis=0)) != len(edges):
		logging.warning(f'Detected {len(edges)-len(np.unique(edges, axis=0))} duplicate edge(s) from {len(edges)} loaded edges. Duplicate edges are discarded.')
		edges = np.unique(edges, axis=0)
	logging.info(f'{print_datetime()}Loaded {len(edges)} edges from {filename}')
	E = [[] for _ in range(N)]
	for (u, v) in edges:
		E[u].append(v)
		E[v].append(u)
	return E


def loadGeneList(filename):
	genes = np.loadtxt(filename, dtype=str)
	logging.info(f'{print_datetime()}Loaded {len(genes)} genes from {filename}')
	return genes


# def loadImage(dataset, filename):
# 	path2file = os.path.join(dataFolder(dataset), filename)
# 	if os.path.exists(path2file): return plt.imread(path2file)
# 	else: return None


def loadDataset(self, neighbor_suffix=None, expression_suffix=None):
	neighbor_suffix = parseSuffix(neighbor_suffix)
	expression_suffix = parseSuffix(expression_suffix)

	self.YTs = []
	for i in self.repli_list:
		for s in ['txt', 'tsv', 'pkl', 'pickle']:
			path2file = self.path2dataset / 'files' / f'expression_{i}{expression_suffix}.{s}'
			if not path2file.exists(): continue
			self.YTs.append(loadExpression(path2file))
	assert len(self.YTs) == len(self.repli_list)
	self.Ns, self.Gs = zip(*map(np.shape, self.YTs))
	self.GG = max(self.Gs)
	self.Es = [
		loadEdges(self.path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
		if u else [[] for _ in range(N)]
		for i, N, u in zip(self.repli_list, self.Ns, self.use_spatial)
	]
	self.Es_empty = [sum(map(len, E)) == 0 for E in self.Es]
	try:
		self.genes = [
			loadGeneList(self.path2dataset / 'files' / f'genes_{i}{expression_suffix}.txt')
			for i in self.repli_list
		]
	except:
		pass
