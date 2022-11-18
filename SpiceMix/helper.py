import time, h5py, logging, itertools
from tqdm.auto import tqdm, trange
from typing import Union, List, Tuple

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, f1_score
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor

from anndata import AnnData
import scanpy as sc

import torch

from matplotlib import pyplot as plt
import seaborn as sns
from umap import UMAP

from util import smooth_discrete, smooth_continuous, clustering_louvain_nclust


def evaluate_embedding_maynard2021(
		obj, do_plot=True, fn_normalize='z-score',
		num_nbrs=20, resolution_boundaries=(.1, 2.), num_rs=20, num_clust=7,
):
	M = obj.M.cpu().numpy()
	Xs = [X.cpu().numpy() for X in obj.Xs]
	x = np.concatenate(Xs, axis=0)
	print(f'normalization = {fn_normalize}')
	if fn_normalize == 'z-score':
		x = StandardScaler().fit_transform(x)
	elif fn_normalize == 'none':
		pass
	elif fn_normalize == 'l-2':
		x *= np.linalg.norm(M, axis=0, ord=2, keepdims=True)
	else:
		raise NotImplementedError()

	x = smooth_continuous(obj.meta[['coor X', 'coor Y']].values, x, obj.meta['repli'])

	def fn(n_clust):
		y = clustering_louvain_nclust(
			x.copy(), n_clust,
			kwargs_neighbors=dict(n_neighbors=num_nbrs),
			kwargs_clustering=dict(),
			resolution_boundaries=resolution_boundaries,
			num_rs=num_rs,
			force_n_cluster=False,
		)
		y_prev = y
		while True:
			y = smooth_discrete(obj.meta[['coor X', 'coor Y']].values, y, obj.meta['repli'])
			if np.all(y_prev == y):
				break
			y_prev = y
		return y

	obj.meta['label SpiceMixPlus'] = fn(num_clust)
	df = obj.meta[~obj.meta['cell type'].isna()]
	print('ari all = {:.2f}'.format(adjusted_rand_score(*df[['cell type', 'label SpiceMixPlus']].values.T)))
	for repli, df in obj.meta[~obj.meta['cell type'].isna()].groupby('repli'):
		print('ari {} = {:.2f}'.format(repli, adjusted_rand_score(*df[['cell type', 'label SpiceMixPlus']].values.T)))

	if do_plot:
		ncol = 1
		nrow =(obj.num_repli + ncol - 1) // ncol
		fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), squeeze=False)
		for ax, (repli, df) in zip(axes.flat, obj.meta.groupby('repli')):
			df = df.groupby(['cell type', 'label SpiceMixPlus']).size().unstack().fillna(0).astype(int)
			sns.heatmap(df.div(df.sum(1), 0), annot=df, mask=df == 0, ax=ax, fmt='d', cmap='Reds')
		plt.show()
		plt.close()
	else:
		print(obj.meta.groupby(['cell type', 'label SpiceMixPlus']).size().unstack().fillna(0).astype(int))

	t = np.quantile(x, np.linspace(0, 1, 11), axis=0)
	fig, ax = plt.subplots(figsize=(10, 6))
	sns.heatmap(ax=ax, data=t / t.max(0, keepdims=True), annot=t, fmt='.1e')
	plt.show()
	plt.close()
