import os, time, pickle, sys, psutil, resource, datetime, h5py, logging
from collections import Iterable

import numpy as np
import torch
import networkx as nx


pid = os.getpid()
psutil_process = psutil.Process(pid)


def print_datetime():
	return datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]\t')


def array2string(x):
	return np.array2string(x, formatter={'all': '{:.2e}'.format})


def parseSuffix(s):
	return '' if s is None or s == '' else '_' + s


def openH5File(filename, mode='a', num_attempts=5, duration=1):
	for i in range(num_attempts):
		try:
			return h5py.File(filename, mode=mode)
		except OSError as e:
			logging.warning(str(e))
			time.sleep(duration)
	return None


def encode4h5(v):
	if isinstance(v, str): return v.encode('utf-8')
	return v


def parseIiter(g, iiter):
	if iiter < 0: iiter += max(map(int, g.keys())) + 1
	return iiter


def zipTensors(*tensors):
	return np.concatenate([
		np.array(a).flatten()
		for a in tensors
	])


def unzipTensors(arr, shapes):
	assert np.all(arr.shape == (np.sum(list(map(np.prod, shapes))),))
	tensors = []
	for shape in shapes:
		size = np.prod(shape)
		tensors.append(arr[:size].reshape(*shape).squeeze())
		arr = arr[size:]
	return tensors


# PyTorchDType = torch.float
PyTorchDType = torch.double


def calcPermutation(sim):
	assert sim.ndim == 2
	B = nx.Graph()
	B.add_nodes_from([f'o{i}' for i in range(sim.shape[0])], bipartite=0)
	B.add_nodes_from([f't{i}' for i in range(sim.shape[1])], bipartite=1)
	B.add_edges_from([
		(f'o{i}', f't{j}', {'weight': sim[i, j]})
		for i in range(sim.shape[0]) for j in range(sim.shape[1])
	])
	assert nx.is_bipartite(B)
	matching = nx.max_weight_matching(B, maxcardinality=True)
	assert len(set(__ for _ in matching for __ in _)) == 2*min(sim.shape)
	matching = [_ if _[0][0] == 'o' else _[::-1] for _ in matching]
	matching = [tuple(int(__[1:]) for __ in _) for _ in matching]
	matching = sorted(matching, key=lambda x: x[1])
	perm, index = map(np.array, zip(*matching))
	return perm, index
