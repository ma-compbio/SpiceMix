import logging, h5py, os, pickle
from pathlib import Path
from util import openH5File, encode4h5, parse_suffix, config_logger

import numpy as np, pandas as pd
import torch

from load_data import load_expression, load_edges, load_genelist
from initialization import initialize_kmeans, initialize_Sigma_x_inv, initialize_svd, initialize_louvain, \
	initialize_post_clustering
from estimate_weights import estimate_weight_wonbr, estimate_weight_wnbr
from estimate_parameters import estimate_M, estimate_Sigma_x_inv, estimate_Sigma_x_inv_evolutionary


logger = config_logger(logging.getLogger(__name__))


class SpiceMix:
	def __init__(
			self,
			K, lambda_Sigma_x_inv, power_Sigma_x_inv, repli_list, betas=None, prior_x_modes=None,
			path2result=None, context=None, context_Y=None,
			batch_size_Sigma_x_inv=-1,
	):
		if context is None: context = dict(device='cpu', dtype=torch.float32)
		if context_Y is None: context_Y = context
		context.setdefault('device', 'cpu')
		context_Y.setdefault('device', 'cpu')
		if context['device'] != 'cpu':
			torch.cuda.set_device(context['device'])
		self.context = context
		self.context_Y = context_Y
		self.repli_list = repli_list
		self.num_repli = len(self.repli_list)

		self.K = K
		self.lambda_Sigma_x_inv = lambda_Sigma_x_inv
		self.power_Sigma_x_inv = power_Sigma_x_inv
		self.batch_size_Sigma_x_inv = batch_size_Sigma_x_inv
		if betas is None: betas = np.full(self.num_repli, 1/self.num_repli)
		else: betas = np.array(betas, copy=False) / sum(betas)
		self.betas = betas
		if prior_x_modes is None: prior_x_modes = ['exponential shared fixed'] * self.num_repli
		self.prior_x_modes = prior_x_modes
		self.M_constraint = 'simplex'
		# self.M_constraint = 'unit sphere'
		self.X_constraint = 'none'
		self.sigma_yx_inv_mode = 'average'
		self.pairwise_potential_mode = 'normalized'

		if path2result is not None:
			self.path2result = path2result
			logger.info(f'result file = {self.path2result}')
		else:
			self.path2result = None
		self.save_hyperparameters()

		self.Ys = self.meta = self.Es = self.Es_isempty = self.genes = self.Ns = self.Gs = self.GG = None
		self.Sigma_x_inv = self.Xs = self.sigma_yxs = None
		self.M = None
		self.Q = None
		self.optimizer_Sigma_x_inv = None
		self.prior_xs = None

	def load_dataset(self, path2dataset, neighbor_suffix=None, expression_suffix=None):
		path2dataset = Path(path2dataset)
		neighbor_suffix = parse_suffix(neighbor_suffix)
		expression_suffix = parse_suffix(expression_suffix)

		self.Ys = []
		for r in self.repli_list:
			flag_found = False
			for s in ['pkl', 'txt', 'tsv', 'pickle']:
				path2file = path2dataset / 'files' / f'expression_{r}{expression_suffix}.{s}'
				if not path2file.exists():
					continue
				self.Ys.append(load_expression(path2file))
				flag_found = True
				break
			if not flag_found:
				raise FileNotFoundError(r)
		assert len(self.Ys) == len(self.repli_list)
		self.Ns, self.Gs = zip(*map(np.shape, self.Ys))
		self.GG = max(self.Gs)
		self.genes = [
			load_genelist(path2dataset / 'files' / f'genes_{r}{expression_suffix}.txt')
			for r in self.repli_list
		]
		self.Ys = [G / self.GG * self.K * Y / Y.sum(1).mean() for Y, G in zip(self.Ys, self.Gs)]
		self.Ys = [
			torch.tensor(Y, **self.context_Y).pin_memory()
			if self.context['device'] != 'cpu' and self.context_Y['device'] == 'cpu' else
			torch.tensor(Y, **self.context_Y)
			for Y in self.Ys
		]

		self.Es = [
			load_edges(path2dataset / 'files' / f'neighborhood_{i}{neighbor_suffix}.txt', N)
			for i, N in zip(self.repli_list, self.Ns)
		]
		self.Es_isempty = [sum(map(len, E)) == 0 for E in self.Es]

		df_all = []
		for r in self.repli_list:
			path2file = path2dataset / 'files' / f'meta_{r}.pkl'
			if os.path.exists(path2file):
				with open(path2file, 'rb') as f: df_all.append(pickle.load(f))
				continue
			path2file = path2dataset / 'files' / f'meta_{r}.csv'
			if os.path.exists(path2file):
				df_all.append(pd.read_csv(path2file))
				continue
			# for legacy reason
			path2file = path2dataset / 'files' / f'celltypes_{r}.txt'
			if os.path.exists(path2file):
				df = pd.read_csv(path2file, header=None)
				df.columns = ['cell type']
				# df['repli'] = r
				df_all.append(df)
				continue
			raise FileNotFoundError(r)
		assert len(df_all) == len(self.repli_list)
		for r, df in zip(self.repli_list, df_all):
			df['repli'] = r
		df_all = pd.concat(df_all)
		self.meta = df_all

	def initialize(self, method='louvain', random_state=0, kwargs=None, precomputed_clusters=None):
		if kwargs is None:
			kwargs = dict()
		if method == 'kmeans':
			labels = initialize_kmeans(
				self.K, self.Ys,
				kwargs_kmeans=dict(random_state=random_state),
				**kwargs,
			)
			self.M, self.Xs = initialize_post_clustering(self.K, self.Ys, labels, self.context)
		elif method == 'louvain':
			labels = initialize_louvain(self.K, self.Ys, **kwargs)
			self.M, self.Xs = initialize_post_clustering(self.K, self.Ys, labels, self.context)
		elif method == 'svd':
			self.M, self.Xs = initialize_svd(
				self.K, self.Ys, context=self.context,
				M_nonneg=self.M_constraint == 'simplex', X_nonneg=True,
			)
		elif method == 'precomputed clusters':
			self.M, self.Xs = initialize_post_clustering(self.K, self.Ys, precomputed_clusters, self.context)
		else:
			raise NotImplementedError()

		if self.M_constraint == 'simplex':
			scale_fac = torch.linalg.norm(self.M, axis=0, ord=1, keepdim=True)
		elif self.M_constraint == 'unit sphere':
			scale_fac = torch.linalg.norm(self.M, axis=0, ord=2, keepdim=True)
		else:
			raise NotImplementedError()
		self.M.div_(scale_fac)
		for X in self.Xs: X.mul_(scale_fac)
		del scale_fac

		if all(_ == 'exponential shared fixed' for _ in self.prior_x_modes):
			# self.prior_xs = [(torch.ones(self.K, **self.context),) for _ in range(self.num_repli)]
			self.prior_xs = [(torch.zeros(self.K, **self.context),) for _ in range(self.num_repli)]
		else:
			raise NotImplementedError()

		self.estimate_sigma_yx()
		self.initialize_Sigma_x_inv()

		# self.save_weights(iiter=0)
		# self.save_parameters(iiter=0)

	def initialize_Sigma_x_inv(self):
		self.Sigma_x_inv = initialize_Sigma_x_inv(self.K, self.Xs, self.Es, self.betas, self.context)
		self.Sigma_x_inv.sub_(self.Sigma_x_inv.mean())
		self.optimizer_Sigma_x_inv = torch.optim.Adam(
			[self.Sigma_x_inv],
			lr=1e-2, # for scDesign2-based synthetic datasets
			# betas=(.9, .99), # for scDesign2-based synthetic datasets
			betas=(.5, .99), # for metagene-based synthetic datasets
		)

	def estimate_sigma_yx(self):
		d = np.array([
			torch.linalg.norm(torch.addmm(Y.to(X.device), X, self.M.T, alpha=-1), ord='fro').item() ** 2
			for Y, X in zip(self.Ys, self.Xs)
		])
		sizes = np.array([np.prod(Y.shape) for Y in self.Ys])
		if self.sigma_yx_inv_mode == 'separate':
			self.sigma_yxs = np.sqrt(d / sizes)
		elif self.sigma_yx_inv_mode == 'average':
			sigma_yx = np.sqrt(np.dot(self.betas, d) / np.dot(self.betas, sizes))
			self.sigma_yxs = np.full(self.num_repli, float(sigma_yx))
		else:
			raise NotImplementedError()
		Q = -np.dot(self.betas, sizes) / 2
		Q -= np.dot(self.betas, sizes) * np.log(2 * np.pi) / 2
		Q += (sizes * self.betas / np.log(self.sigma_yxs)).sum()
		return Q

	def estimate_weights(self, iiter, use_spatial):
		logger.info(f'Updating latent states')
		assert len(use_spatial) == self.num_repli

		assert self.X_constraint == 'none'
		assert self.pairwise_potential_mode == 'normalized'

		loss_list = []
		for i, (Y, X, sigma_yx, E, prior_x_mode, prior_x) in enumerate(zip(
				self.Ys, self.Xs, self.sigma_yxs, self.Es, self.prior_x_modes, self.prior_xs)):
			if self.Es_isempty[i] or not use_spatial[i]:
				loss = estimate_weight_wonbr(
					Y, self.M, X, sigma_yx, prior_x_mode, prior_x, context=self.context)
			else:
				loss = estimate_weight_wnbr(
					Y, self.M, X, sigma_yx, self.Sigma_x_inv, E, prior_x_mode, prior_x, context=self.context)
			loss_list.append(loss)

		self.save_weights(iiter=iiter)

		return loss_list

	def estimate_parameters(
			self, iiter, use_spatial, update_Sigma_x_inv=True,
			backend_Sigma_x_inv='GD',
	):
		logger.info(f'Updating model parameters')

		if update_Sigma_x_inv:
			if backend_Sigma_x_inv == 'GD':
				history, Q_spatial = estimate_Sigma_x_inv(
					self.Xs, self.Sigma_x_inv, self.Es, use_spatial, self.lambda_Sigma_x_inv, self.power_Sigma_x_inv,
					self.betas, self.optimizer_Sigma_x_inv, self.context,
					batch_size=self.batch_size_Sigma_x_inv,
				)
			elif backend_Sigma_x_inv == 'CMA-ES':
				history, Q_spatial = estimate_Sigma_x_inv_evolutionary(
					self.Xs, self.Sigma_x_inv, self.Es, use_spatial, self.lambda_Sigma_x_inv, self.power_Sigma_x_inv,
					self.betas, self.context,
					batch_size=self.batch_size_Sigma_x_inv,
					num_samples=20, ratio_samples=.3, gamma_cov=.9,
				)
			else:
				raise NotImplementedError()
		else:
			history, Q_spatial = [], 0
		estimate_M(
			self.Ys, self.Xs, self.M, self.sigma_yxs, self.betas,
			M_constraint=self.M_constraint, context=self.context,
			tol=None if any(use_spatial) else 1e-3,
		)
		Q_emission = self.estimate_sigma_yx()
		# TODO: add prior on X

		self.Q = Q_spatial + Q_emission
		logger.info(f'Q value = {self.Q}')

		self.save_parameters(iiter=iiter)
		self.save_metrics(iiter=iiter)

		return history, self.Q

	def skip_saving(self, iiter):
		return iiter % 10 != 0

	def save_hyperparameters(self):
		if self.path2result is None: return

		f = openH5File(self.path2result)
		if f is None: return
		f['hyperparameters/repli_list'] = [_.encode('utf-8') for _ in self.repli_list]
		# for k in ['prior_x_modes']:
		# 	for repli, v in zip(self.repli_list, getattr(self, k)):
		# 		f[f'hyperparameters/{k}/{repli}'] = encode4h5(v)
		for k in ['lambda_Sigma_x_inv', 'power_Sigma_x_inv', 'betas', 'K']:
			f[f'hyperparameters/{k}'] = encode4h5(getattr(self, k))
		f.close()

	def save_weights(self, iiter):
		if self.path2result is None: return
		if self.skip_saving(iiter): return

		f = openH5File(self.path2result)
		if f is None: return

		for repli, XT in zip(self.repli_list, self.Xs):
			f[f'latent_states/XT/{repli}/{iiter}'] = XT.cpu().numpy()

		f.close()

	def save_parameters(self, iiter):
		if self.path2result is None: return
		if self.skip_saving(iiter): return

		f = openH5File(self.path2result)
		if f is None: return

		for k in ['M', 'Sigma_x_inv']:
			v = getattr(self, k)
			if isinstance(v, torch.Tensor): v = v.cpu().numpy()
			f[f'parameters/{k}/{iiter}'] = v

		for k in ['sigma_yxs']:
			for repli, v in zip(self.repli_list, getattr(self, k)):
				f[f'parameters/{k}/{repli}/{iiter}'] = v

		# for k in ['prior_xs']:
		# 	for repli, v in zip(self.repli_list, getattr(self, k)):
		# 		f[f'parameters/{k}/{repli}/{iiter}'] = np.array(v[1:])

		f.close()

	def save_metrics(self, iiter):
		if self.path2result is None: return
		if self.skip_saving(iiter): return

		f = openH5File(self.path2result)
		if f is None: return

		for k in ['Q']:
			f[f'metrics/{k}/{iiter}'] = getattr(self, k)

		f.close()
