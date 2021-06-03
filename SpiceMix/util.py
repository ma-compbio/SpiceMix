import os, time, pickle, sys, psutil, resource, datetime, h5py, logging
from collections import Iterable

import numpy as np
import torch
import networkx as nx


pid = os.getpid()
psutil_process = psutil.Process(pid)

# PyTorchDType = torch.float
PyTorchDType = torch.double


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


def a2i(a, order=None, ignores=()):
	if order is None:
		order = list(set(a) - set(ignores))
	else:
		order = order[~np.isin(order, list(ignores))]
	d = dict(zip(order, range(len(order))))
	for k in ignores: d[k] = -1
	a = np.fromiter(map(d.get, a), dtype=int)
	return a, d, order


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
