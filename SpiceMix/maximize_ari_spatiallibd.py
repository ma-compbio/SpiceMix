import pickle, itertools, sys, os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, \
	RadiusNeighborsRegressor, RadiusNeighborsClassifier

import scanpy as sc
from anndata import AnnData
import itertools
from pathlib import Path
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

n_jobs = 1
# n_jobs = 32


def smooth_discrete(coor, y, rep, radius=2., radius_eps=1e-3):
	radius += radius_eps
	y_unique, y = np.unique(y, return_inverse=True)
	num_y = len(y_unique)
	p = np.zeros([len(y), num_y])
	for r in np.unique(rep):
		idx = np.where(rep == r)[0]
		# nbrs = NearestNeighbors(radius=radius, n_jobs=1)\
		# 	.fit(coor[idx]).radius_neighbors_graph(coor[idx]).tocoo()
		# p[idx] = np.bincount(nbrs.row*num_y + y[idx[nbrs.col]], minlength=len(idx)*num_y).reshape(len(idx), num_y)
		y_u, yy = np.unique(y[idx], return_inverse=True)
		p[tuple(zip(*itertools.product(idx, y_u)))] = RadiusNeighborsClassifier(
			radius=radius, n_jobs=n_jobs).fit(coor[idx], yy).predict_proba(coor[idx]).ravel()
	# p[(range(len(y)), y)] -= 1
	# p /= p.sum(1, keepdims=True)
	z = p.argmax(1)
	return np.where(p.max(1) > .51, z, y)


def smooth_continuous(coor, y, rep, radius=2., radius_eps=1e-3, lam=1.):
	radius = radius + radius_eps
	z = np.empty_like(y)
	for r in np.unique(rep):
		idx = np.where(rep == r)[0]
		z[idx] = RadiusNeighborsRegressor(radius=radius, n_jobs=n_jobs).fit(coor[idx], y[idx]).predict(coor[idx])
	return z * lam + y * (1-lam)


def calc_modularity(A, label, resolution=1, normalize=False):
	num_nodes = A.shape[0]
	label_a2i = dict(zip(set(label), itertools.count()))
	num_labels = len(label_a2i)
	if num_labels == 1: return 0.
	label = np.fromiter(map(label_a2i.get, label), dtype=int)
	A = A.tocoo()
	assert (A.col != A.row).all() # Multiplying diagonal values by 2 might works
	Asum = A.data.sum()
	assert Asum > 0
	score = A.data[label[A.row] == label[A.col]].sum() / Asum

	k = np.bincount(label[A.row], weights=A.data, minlength=num_labels) / Asum
	kk = np.bincount(label[A.col], weights=A.data, minlength=num_labels) / Asum
	assert np.allclose(k, kk), (k, kk)
	score -= k @ k * resolution

	if normalize:
		max_score = k @ (1 - k*resolution)
		score /= max_score

	return score

