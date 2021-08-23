import sys, time, itertools, psutil, resource, logging, h5py, os
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from util import print_datetime, openH5File, encode4h5

import numpy as np
import gurobipy as grb
import torch

from loadData import loadDataset
from initialization import initializeByKMean
from estimateWeights import estimateWeightsICM, estimateWeightsWithoutNeighbor
from estimateParameters import estimateParametersX, estimateParametersY


class Model:
	def __init__(
			self, path2dataset, repli_list, use_spatial, neighbor_suffix, expression_suffix,
			K, lambda_SigmaXInv, betas, prior_x_modes,
			result_filename=None,
			PyTorch_device='cpu', num_processes=1,
	):
		self.PyTorch_device = PyTorch_device
		self.num_processes = num_processes

		self.path2dataset = Path(path2dataset)
		self.repli_list = repli_list
		self.use_spatial = use_spatial
		self.num_repli = len(self.repli_list)
		assert len(self.repli_list) == len(self.use_spatial)
		loadDataset(self, neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix)

		self.K = K
		self.YTs = [G / self.GG * self.K * YT / YT.sum(1).mean() for YT, G in zip(self.YTs, self.Gs)]
		self.lambda_SigmaXInv = lambda_SigmaXInv
		self.betas = betas
		self.prior_x_modes = prior_x_modes
		self.M_constraint = 'sum2one'
		self.X_constraint = 'none'
		self.dropout_mode = 'raw'
		self.sigma_yx_inv_mode = 'average'
		self.pairwise_potential_mode = 'normalized'

		if result_filename is not None:
			os.makedirs(self.path2dataset / 'results', exist_ok=True)
			self.result_filename = self.path2dataset / 'results' / result_filename
			logging.info(f'{print_datetime()}result file = {self.result_filename}')
		else:
			self.result_filename = None
		self.saveHyperparameters()

	# def __del__(self):
	# 	pass
		# if self.result_h5 is not None:
		# 	self.result_h5.close()

	def initialize(self, *args, **kwargs):
		ret = initializeByKMean(self, *args, **kwargs)
		self.saveWeights(iiter=0)
		self.saveParameters(iiter=0)
		return ret

	def estimateWeights(self, iiter):
		logging.info(f'{print_datetime()}Updating latent states')

		assert self.X_constraint == 'none'
		assert self.pairwise_potential_mode == 'normalized'

		rXTs = []
		with Pool(min(self.num_processes, self.num_repli)) as pool:
			for i in range(self.num_repli):
				if self.Es_empty[i]:
					rXTs.append(pool.apply_async(estimateWeightsWithoutNeighbor, args=(
						self.YTs[i],
						self.M[:self.Gs[i]], self.XTs[i], self.prior_xs[i], self.sigma_yx_invs[i],
						self.X_constraint, self.dropout_mode, i,
					)))
				else:
					rXTs.append(pool.apply_async(estimateWeightsICM, args=(
						self.YTs[i], self.Es[i],
						self.M[:self.Gs[i]], self.XTs[i], self.prior_xs[i], self.sigma_yx_invs[i], self.Sigma_x_inv,
						self.X_constraint, self.dropout_mode, self.pairwise_potential_mode, i,
					)))
					# rXTs.append(estimateWeightsICM(
					# 	self.YTs[i], self.Es[i],
					# 	self.M[:self.Gs[i]], self.XTs[i], self.prior_xs[i], self.sigma_yx_invs[i], self.Sigma_x_inv,
					# 	self.X_constraint, self.dropout_mode, self.pairwise_potential_mode, i,
					# ))
			self.XTs = [_.get(1e9) if isinstance(_, multiprocessing.pool.ApplyResult) else _ for _ in rXTs]
		pool.join()

		self.saveWeights(iiter=iiter)

	def estimateParameters(self, iiter):
		logging.info(f'{print_datetime()}Updating model parameters')

		self.Q = 0
		if self.pairwise_potential_mode == 'normalized' and all(
				prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']
				for prior_x in self.prior_xs):
			# pool = Pool(1)
			# Q_Y = pool.apply_async(estimateParametersY, args=([self])).get(1e9)
			# pool.close()
			# pool.join()
			Q_Y = estimateParametersY(self)
			self.Q += Q_Y

			Q_X = estimateParametersX(self, iiter)
			self.Q += Q_X
		else:
			raise NotImplementedError

		self.saveParameters(iiter=iiter)
		self.saveProgress(iiter=iiter)

		return self.Q

	def skipSaving(self, iiter):
		return iiter % 10 != 0

	def saveHyperparameters(self):
		if self.result_filename is None: return

		with h5py.File(self.result_filename, 'w') as f:
			f['hyperparameters/repli_list'] = [_.encode('utf-8') for _ in self.repli_list]
			for k in ['prior_x_modes']:
				for repli, v in zip(self.repli_list, getattr(self, k)):
					f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
			for k in ['lambda_SigmaXInv', 'betas', 'K']:
				f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))

	def saveWeights(self, iiter):
		if self.result_filename is None: return
		if self.skipSaving(iiter): return

		f = openH5File(self.result_filename)
		if f is None: return

		for repli, XT in zip(self.repli_list, self.XTs):
			f[f'latent_states/XT/{repli}/{iiter}'] = XT

		f.close()

	def saveParameters(self, iiter):
		if self.result_filename is None: return
		if self.skipSaving(iiter): return

		f = openH5File(self.result_filename)
		if f is None: return

		for k in ['M', 'Sigma_x_inv']:
			f[f'parameters/{k}/{iiter}'] = getattr(self, k)

		for k in ['sigma_yx_invs']:
			for repli, v in zip(self.repli_list, getattr(self, k)):
				f[f'parameters/{k}/{repli}/{iiter}'] = v

		for k in ['prior_xs']:
			for repli, v in zip(self.repli_list, getattr(self, k)):
				f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

		f.close()


	def saveProgress(self, iiter):
		if self.result_filename is None: return

		f = openH5File(self.result_filename)
		if f is None: return

		f[f'progress/Q/{iiter}'] = self.Q

		f.close()
