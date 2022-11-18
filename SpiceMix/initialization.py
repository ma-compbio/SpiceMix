import logging, itertools
from typing import Dict, List
from util import clustering_louvain_nclust, config_logger

import torch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA


logger = config_logger(logging.getLogger(__name__))


def initialize_kmeans(K, Ys, kwargs_kmeans, num_pcs=None):
	assert 'random_state' in kwargs_kmeans
	Ys = [Y.cpu().numpy() for Y in Ys]
	Y = np.concatenate(Ys, axis=0)
	if num_pcs is not None:
		pca = PCA(n_components=num_pcs)
		Y = pca.fit_transform(Y)
	kmeans = KMeans(n_clusters=K, **kwargs_kmeans)
	label = kmeans.fit_predict(Y)
	labels = np.split(label, np.cumsum(list(map(len, Ys)))[:-1])
	return labels


def initialize_louvain(K, Ys, num_pcs=50, n_neighbors=20, resolution_boundaries=(.5, 2.), num_rs=10):
	Ys = [Y.cpu().numpy() for Y in Ys]
	x = np.concatenate(Ys, axis=0)
	if num_pcs is not None:
		pca = PCA(n_components=num_pcs)
		x = pca.fit_transform(x)
	while True:
		label = clustering_louvain_nclust(
			x, K,
			kwargs_neighbors=dict(n_neighbors=n_neighbors),
			kwargs_clustering=dict(),
			resolution_boundaries=resolution_boundaries,
			num_rs=num_rs,
			force_n_cluster=False,
		)
		label = np.array(label, dtype=int)
		if len(set(label)) == K:
			break
		n_neighbors = int(n_neighbors * 1.5)
		assert n_neighbors < len(label)
		logger.warning('Retrying clustering')
	labels = np.split(label, np.cumsum(list(map(len, Ys)))[:-1])
	return labels


def initialize_post_clustering(
		K: int, Ys: List[torch.Tensor], labels: List[np.ndarray], context: Dict, eps: float = 1e-10,
):
	eps = eps / K
	Y_cat = np.concatenate([Y.cpu().numpy() for Y in Ys], axis=0)
	label = np.concatenate(labels)
	assert set(label) == set(range(K)), set(label)
	M = np.stack([Y_cat[label == l].mean(0) for l in range(K)], axis=1)
	M = torch.tensor(M, **context)
	Xs = []
	for c in labels:
		X = np.full([len(c), K], eps)
		X[(range(len(c)), c)] = 1
		Xs.append(torch.tensor(X, **context))
	return M, Xs


def initialize_svd(K, Ys, context, M_nonneg=True, X_nonneg=True):
	Ns, Gs = zip(*[Y.shape for Y in Ys])
	GG = max(Gs)
	repli_valid = np.array(Gs) == GG
	Ys = [Y.cpu().numpy() for Y in Ys]
	Y_cat = np.concatenate(list(itertools.compress(Ys, repli_valid)), axis=0)
	svd = TruncatedSVD(K)
	X_cat = svd.fit_transform(Y_cat)
	M = svd.components_.T
	norm_p = np.ones([1, K])
	norm_n = np.ones([1, K])
	if M_nonneg:
		norm_p *= np.linalg.norm(np.clip(M, a_min=0, a_max=None), axis=0, ord=1, keepdims=True)
		norm_n *= np.linalg.norm(np.clip(M, a_min=None, a_max=0), axis=0, ord=1, keepdims=True)
	if X_nonneg:
		norm_p *= np.linalg.norm(np.clip(X_cat, a_min=0, a_max=None), axis=0, ord=1, keepdims=True)
		norm_n *= np.linalg.norm(np.clip(X_cat, a_min=None, a_max=0), axis=0, ord=1, keepdims=True)
	sign = np.where(norm_p >= norm_n, 1., -1.)
	M *= sign
	X_cat *= sign
	X_cat_iter = X_cat
	if M_nonneg:
		M = np.clip(M, a_min=1e-10, a_max=None)
	Xs = []
	for is_valid, N, Y in zip(repli_valid, Ns, Ys):
		if is_valid:
			X = X_cat_iter[:N]
			X_cat_iter = X_cat_iter[N:]
			if X_nonneg:
				# fill negative elements by zero
				# X = np.clip(X, a_min=1e-10, a_max=None)
				# fill negative elements by the average of nonnegative elements
				for x in X.T:
					idx = x < 1e-10
					x[idx] = x[~idx].mean()
		else:
			X = np.full([N, K], 1/K)
		Xs.append(X)
	M = torch.tensor(M, **context)
	Xs = [torch.tensor(X, **context) for X in Xs]
	return M, Xs


def initialize_Sigma_x_inv(K, Xs, Es, betas, context):
	Sigma_x_inv = torch.zeros([K, K], **context)
	for X, E, beta in zip(Xs, Es, betas):
		Z = X / torch.linalg.norm(X, dim=1, keepdim=True, ord=1)
		E = np.array([(i, j) for i, e in enumerate(E) for j in e])
		x = Z[E[:, 0]]
		y = Z[E[:, 1]]
		x = x - x.mean(0, keepdim=True)
		y = y - y.mean(0, keepdim=True)
		corr = (y / y.std(0, keepdim=True)).T @ (x / x.std(0, keepdim=True)) / len(x)
		Sigma_x_inv.add_(corr, alpha=-beta)
	Sigma_x_inv = (Sigma_x_inv + Sigma_x_inv.T) / 2
	Sigma_x_inv -= Sigma_x_inv.mean()
	Sigma_x_inv *= 10
	return Sigma_x_inv
