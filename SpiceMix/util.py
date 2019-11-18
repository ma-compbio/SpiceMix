import numpy as np
import os, time, pickle, sys, psutil, resource
import matplotlib.markers
import torch
from collections import Iterable

path2dir = os.path.dirname(__file__)

pid = os.getpid()
psutil_process = psutil.Process(pid)

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

def dataFolder(dataset): return os.path.join(path2dir, '..', 'data', dataset, 'files')

# PyTorchDType = torch.float
PyTorchDType = torch.double
# force_cpu = True
force_cpu = False
if torch.cuda.is_available() and not force_cpu:
	PyTorchDevice = torch.device('cuda')
else:
	PyTorchDevice = torch.device('cpu')
	torch.set_num_threads(4)

class Logger:
	def __init__(self, dataset, tm=None):
		if tm is None: tm = time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + str(os.getpid())
		self.folder = os.path.join(path2dir, '..', 'data', dataset, 'logs', tm)
		os.makedirs(self.folder, exist_ok=True)
		print(f'log folder = {self.folder}')
	def log(self, filename, x):
		with open(os.path.join(self.folder, filename + '.pkl'), 'wb') as f: pickle.dump(x, f, protocol=2)

def loadLog(dataset, tm, limit=None):
	H_Theta_Q = [[], [], []]
	if isinstance(limit, list): limit = iter(limit)
	if isinstance(limit, Iterable): i = next(limit, None)
	else: i = 0
	while True:
		stop_flag = False

		for a, text in zip(H_Theta_Q, ['H', 'Theta', 'Q']):
			filename = os.path.join(path2dir, '..', 'data', dataset, 'logs', tm, f'{text}_{i}.pkl')

			if os.path.exists(filename):
				# print('loading file {}'.format(filename))
				with open(filename, 'rb') as f: a.append(pickle.load(f))
			else:
				stop_flag = True

			if stop_flag: break

		if isinstance(limit, Iterable):
			i = next(limit, None)
			if i is None: stop_flag = True
		elif isinstance(limit, int):
			i += 1
			if i > limit: stop_flag = True
		else:
			i += 1

		if stop_flag: break
	return tuple(H_Theta_Q)
