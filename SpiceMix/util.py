import time, h5py, logging, itertools
from tqdm.auto import tqdm, trange
from typing import Union, List, Tuple

import numpy as np, pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor

from anndata import AnnData
import scanpy as sc

import torch


def config_logger(logger, level=logging.INFO):
	if logger.hasHandlers():
		return logger
	logger.setLevel(level)
	ch = logging.StreamHandler()
	ch.setLevel(level)
	ch.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"))
	logger.addHandler(ch)
	return logger


logger = config_logger(logging.getLogger(__name__))


class Minibatcher:
	def __init__(
			self, dataset: Union[torch.Tensor, Tuple[torch.Tensor]], batch_size: int, shuffle=True, drop_last=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.num_samples = len(dataset) if isinstance(dataset, torch.Tensor) else len(dataset[0])
		if batch_size >= self.num_samples:
			shuffle = drop_last = False
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.shuffled_indices = None
		self.cur_idx = None
		self.reset()

	def reset(self):
		if self.shuffle:
			self.shuffled_indices = np.random.permutation(self.num_samples)
			self.cur_idx = 0
		else:
			self.cur_idx = 0
		return self

	def sample(self):
		if self.cur_idx >= self.num_samples:
			self.reset()
		if self.drop_last and self.cur_idx + self.batch_size > self.num_samples:
			self.reset()
		if self.shuffle:
			indices = self.shuffled_indices[self.cur_idx: self.cur_idx + self.batch_size]
		else:
			indices = slice(self.cur_idx, self.cur_idx + self.batch_size)
		self.cur_idx += self.batch_size
		if isinstance(self.dataset, torch.Tensor):
			return self.dataset[indices]
		else:
			return (tensor[indices] for tensor in self.dataset)


def calc_modularity(A, label, resolution=1):
	A = A.tocoo()
	n = A.shape[0]
	Asum = A.data.sum()
	score = A.data[label[A.row] == label[A.col]].sum() / Asum

	idx = np.argsort(label)
	label = label[idx]
	k = np.array(A.sum(0)).ravel() / Asum
	k = k[idx]
	idx = np.concatenate([[0], np.nonzero(label[:-1] != label[1:])[0] + 1, [len(label)]])
	score -= sum(k[i:j].sum() ** 2 for i, j in zip(idx[:-1], idx[1:])) * resolution
	return score


def clustering_louvain(X, *, kwargs_neighbors, kwargs_clustering, num_rs=100, method='louvain', verbose=True):
	if isinstance(X, AnnData):
		adata = X
	else:
		adata = AnnData(X, dtype=np.float32)
	if kwargs_neighbors is not None:
		sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
	best = {'score': np.nan}
	resolution = kwargs_clustering.get('resolution', 1)
	pbar = trange(num_rs, desc=f'Louvain clustering: res={resolution:.2e}', disable=not verbose)
	for rs in pbar:
		getattr(sc.tl, method)(adata, **kwargs_clustering, random_state=rs)
		cluster = np.array(list(adata.obs[method]))
		score = calc_modularity(
			adata.obsp['connectivities'], cluster, resolution=resolution)
		if not best['score'] >= score: best.update({'score': score, 'cluster': cluster.copy(), 'rs': rs})
		pbar.set_description(
			f'Louvain clustering: res={resolution:.2e}; '
			f"best: score = {best['score']:.2f} rs = {best['rs']} # of clusters = {len(set(best['cluster']))}"
		)
	pbar.close()
	y = best['cluster']
	y = pd.Categorical(y, categories=np.unique(y))
	return y


def clustering_louvain_nclust(
		X, n_clust_target, *, kwargs_neighbors, kwargs_clustering,
		resolution_boundaries=None,
		resolution_init=1, resolution_update=2,
		num_rs=100, method='louvain',
		verbose=True,
		force_n_cluster=True,
):
	adata = AnnData(X, dtype=np.float32)
	sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
	kwargs_clustering = kwargs_clustering.copy()
	y = None

	def do_clustering(res):
		y = clustering_louvain(
			adata,
			kwargs_neighbors=None,
			kwargs_clustering=dict(**kwargs_clustering, **dict(resolution=res)),
			method=method,
			num_rs=num_rs,
			verbose=verbose,
		)
		n_clust = len(set(y))
		return y, n_clust

	lb = rb = None
	n_clust = -1
	if resolution_boundaries is not None:
		lb, rb = resolution_boundaries
	else:
		res = resolution_init
		y, n_clust = do_clustering(res)
		if n_clust > n_clust_target:
			while n_clust > n_clust_target and res > 1e-2:
				rb = res
				res /= resolution_update
				y, n_clust = do_clustering(res)
			lb = res
		elif n_clust < n_clust_target:
			while n_clust < n_clust_target:
				lb = res
				res *= resolution_update
				y, n_clust = do_clustering(res)
			rb = res
		if n_clust == n_clust_target: lb = rb = res

	while rb - lb > .01 or lb == rb:
		mid = (lb * rb) ** .5
		y, n_clust = do_clustering(mid)
		if n_clust == n_clust_target or lb == rb: break
		if n_clust > n_clust_target: rb = mid
		else: lb = mid
	assert not force_n_cluster or n_clust == n_clust_target
	return y


def smooth_discrete(coor, y, rep, radius=2., radius_eps=1e-3):
	radius += radius_eps
	y_unique, y = np.unique(y, return_inverse=True)
	num_y = len(y_unique)
	p = np.zeros([len(y), num_y])
	for r in np.unique(rep):
		idx = np.where(rep == r)[0]
		y_u, yy = np.unique(y[idx], return_inverse=True)
		p[tuple(zip(*itertools.product(idx, y_u)))] = RadiusNeighborsClassifier(
			radius=radius, n_jobs=1).fit(coor[idx], yy).predict_proba(coor[idx]).ravel()
	# p[(range(len(y)), y)] -= 1
	# p /= p.sum(1, keepdims=True)
	z = p.argmax(1)
	return np.where(p.max(1) > .51, z, y)


def smooth_continuous(coor, y, rep, radius=2., radius_eps=1e-3, lam=1.):
	radius += radius_eps
	z = np.empty_like(y)
	for r in np.unique(rep):
		idx = np.where(rep == r)[0]
		z[idx] = RadiusNeighborsRegressor(radius=radius, n_jobs=1).fit(coor[idx], y[idx]).predict(coor[idx])
	return z * lam + y * (1-lam)


class NesterovGD:
	# https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
	def __init__(self, x, step_size):
		self.x = x
		self.step_size = step_size
		# self.y = x.clone()
		self.y = torch.zeros_like(x)
		# self.lam = 0
		self.k = 0

	def step(self, grad):
		# - method 1
		# lam_new = (1 + np.sqrt(1 + 4 * self.lam ** 2)) / 2
		# gamma = (1 - self.lam) / lam_new
		# - method 2
		self.k += 1
		gamma = - (self.k - 1) / (self.k + 2)
		# - method 3 - GD
		# gamma = 0
		# -
		# y_new = self.x - grad * self.step_size # use addcmul
		if isinstance(self.step_size, float):
			y_new = self.x.sub(grad, alpha=self.step_size)
		else:
			y_new = self.x.addcmul(grad, self.step_size, value=-1)
		self.x[:] = self.y.mul_(gamma).add_(y_new, alpha=1 - gamma)
		# self.lam = lam_new
		self.y[:] = y_new
		del y_new


def parse_suffix(s):
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
