import pickle, logging

import numpy as np

from util import config_logger


logger = config_logger(logging.getLogger(__name__))


def load_expression(filename):
	if filename.suffix == '.pkl':
		with open(filename, 'rb') as f:
			expr = pickle.load(f)
	elif filename.suffix in ['.txt', '.tsv']:
		expr = np.loadtxt(filename, dtype=np.float)
	else:
		raise ValueError(f'Invalid file format {filename}')
	N, G = expr.shape
	logger.info(f'Loaded {N} cells and {G} genes from {filename}')
	return expr


def load_edges(filename, num_cells):
	edges = np.loadtxt(filename, dtype=np.int)
	assert edges.shape[1] == 2, f'Detected an edge that does not contain two nodes'
	assert np.all(0 <= edges) and np.all(edges < num_cells), f'Node ID exceeded range [0, N)'
	edges = np.sort(edges, axis=1)
	assert np.all(edges[:, 0] < edges[:, 1]), f'Detected {(edges[:, 0] == edges[:, 1]).sum()} self-loop(s)'
	if len(np.unique(edges, axis=0)) != len(edges):
		logger.warning(
			f'Detected {len(edges)-len(np.unique(edges, axis=0))} duplicate edge(s) from {len(edges)} loaded edges. '
			f'Duplicate edges are discarded.'
		)
		edges = np.unique(edges, axis=0)
	logger.info(f'Loaded {len(edges)} edges from {filename}')
	E = [[] for _ in range(num_cells)]
	for (u, v) in edges:
		E[u].append(v)
		E[v].append(u)
	return E


def load_genelist(filename):
	genes = np.loadtxt(filename, dtype=str)
	logger.info(f'Loaded {len(genes)} genes from {filename}')
	return genes
