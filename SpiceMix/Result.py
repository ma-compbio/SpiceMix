import sys, os, itertools, re, gc, copy
import h5py
from pathlib import Path
from collections import Iterable, OrderedDict, defaultdict
import pandas as pd
from util import print_datetime, openH5File, parseIiter, a2i

import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import networkx as nx
import umap

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# sns.set_style("white")
# plt.rcParams['font.family'] = "Liberation Sans"
# plt.rcParams['font.size'] = 16
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['svg.fonttype'] = 'none'

from loadData import loadExpression, loadDataset
from Model import Model


class Result:
	def __init__(
			self, path2dataset, result_filename,
			neighbor_suffix=None, expression_suffix=None,
			showHyperparameters=False,
	):
		self.path2dataset = Path(path2dataset)
		self.result_filename = self.path2dataset / 'results' / result_filename
		print(f'Result file = {self.result_filename}')

		with openH5File(self.result_filename, 'r') as f:
			for k in f['hyperparameters'].keys():
				v = f[f'hyperparameters/{k}']
				if isinstance(v, h5py.Group): continue
				v = v[()]
				if k in ['repli_list']:
					v = np.array([_.decode('utf-8') for _ in v])
				setattr(self, k, v)
				if showHyperparameters:
					print(f'{k} \t= {v}')
		self.num_repli = len(self.repli_list)
		self.use_spatial = [True] * self.num_repli
		loadDataset(self, neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix)
		self.columns_latent_states = np.array([f'latent state {i}' for i in range(self.K)])
		self.columns_exprs = np.array([f'expr {_}' for _ in self.genes[0]])
		self.data = pd.DataFrame(index=range(sum(self.Ns)))
		self.data[['coor X', 'coor Y']] = np.concatenate([
			loadExpression(self.path2dataset / 'files' / f'coordinates_{repli}.txt')
			for repli in self.repli_list
		], axis=0)
		self.data['cell type'] = np.concatenate([
			np.loadtxt(self.path2dataset / 'files' / f'celltypes_{repli}.txt', dtype=str)
			for repli in self.repli_list
		], axis=0)
		self.data['repli'] = sum([[repli] * N for repli, N in zip(self.repli_list, self.Ns)], [])
		self.data[self.columns_exprs] = np.concatenate(self.YTs, axis=0)
		self.scaling = [G / self.GG * self.K / YT.sum(1).mean() for YT, G in zip(self.YTs, self.Gs)]
		self.colors = {}
		self.orders = {}

		self.metagene_order = np.arange(self.K)

	def plotConvergenceQ(self, ax, **kwargs):
		with openH5File(self.result_filename, 'r') as f:
			g = f['progress/Q']
			k = np.fromiter(map(int, g.keys()), dtype=int)
			v = np.fromiter((g[str(_)][()] for _ in k), dtype=float)
		Q = np.full(k.max() - k.min() + 1, np.nan)
		Q[k - k.min()] = v
		for kk, linestyle in zip([1, 5, 25], ['-', '--', ':']):
			if kk in [25]: continue
			if kk >= len(Q): continue
			dQ = (Q[kk:] - Q[:-kk]) / kk
			# print(f'min dQ = {dQ[np.logical_not(np.isnan(dQ))].min():.2e}')
			# dQ += 1e-1
			ax.plot(np.arange(k.min(), k.max() + 1 - kk) + kk / 2 + 1, dQ, linestyle=linestyle, **kwargs)

	def reorderMetagenes(self, cidx):
		assert len(cidx) == len(set(cidx)) == self.K
		self.metagene_order = np.array(cidx)

	def loadLatentStates(self, iiter=-1):
		with openH5File(self.result_filename, 'r') as f:
			iiter = parseIiter(f[f'latent_states/XT/{self.repli_list[0]}'], iiter)
			print(f'Iteration {iiter}')
			XTs = [f[f'latent_states/XT/{repli}/{iiter}'][()] for repli in self.repli_list]
		XTs = [_ / __ for _, __ in zip(XTs, self.scaling)]
		self.data[[f'latent state {i}' for i in range(self.K)]] = np.concatenate(XTs)

	def clustering(self, ax, K_range, K_offset=0):
		XT = self.data[self.columns_latent_states].values
		XT = StandardScaler().fit_transform(XT)
		K_range = np.array(K_range)

		f = lambda K: AgglomerativeClustering(
			n_clusters=K,
			linkage='ward',
		).fit_predict(XT)
		CH = np.fromiter((calinski_harabasz_score(XT, f(K)) for K in K_range), dtype=float)
		K_opt = K_range[CH.argmax() + K_offset]

		ax.scatter(K_range, CH, marker='x', color=np.where(K_range == K_opt, 'C1', 'C0'))

		print(f'K_opt = {K_opt}')
		y = f(K_opt)
		print(f'#clusters = {len(set(y) - {-1})}, #-1 = {(y == -1).sum()}')
		self.data['cluster_raw'] = y
		self.data['cluster'] = list(map(str, y))

	def annotateClusters(self, clusteri2a):
		self.data['cluster'] = [clusteri2a[_] for _ in self.data['cluster_raw']]

	def assignColors(self, key, mapping):
		assert set(mapping.keys()) >= set(self.data[key])
		self.colors[key] = copy.deepcopy(mapping)

	def assignOrder(self, key, order):
		s = set(self.data[key])
		assert set(order) >= s and len(order) == len(set(order))
		order = list(filter(lambda _: _ in s, order))
		self.orders[key] = np.array(order)

	def UMAP(self, **kwargs):
		XT = self.data[self.columns_latent_states].values
		XT = StandardScaler().fit_transform(XT)
		XT = umap.UMAP(**kwargs).fit_transform(XT)
		self.data[[f'UMAP {i+1}' for i in range(XT.shape[1])]] = XT

	def visualizeFeatureSpace(self, ax, key, key_x='UMAP 1', key_y='UMAP 2', repli=None, **kwargs):
		if isinstance(repli, int): repli = self.repli_list[repli]
		if repli is None:
			data = self.data
		else:
			data = self.data.groupby('repli').get_group(repli)
		kwargs = copy.deepcopy(kwargs)
		if data[key].dtype == 'O':
			kwargs.setdefault('hue_order', self.orders.get(key, None))
			kwargs.setdefault('palette', self.colors.get(key, None))
			sns.scatterplot(ax=ax, data=data, x=key_x, y=key_y, hue=key, **kwargs)
		else:
			kwargs.setdefault('cmap', self.colors.get(key, None))
			sca = ax.scatter(data[key_x], data[key_y], c=data[key], **kwargs)
			cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.tick_params(axis='both', labelsize=10)

	def visualizeFeaturesSpace(self, axes, keys=(), key_x='coor X', key_y='coor Y', permute_metagenes=True, *args, **kwargs):
		if len(keys) == 0: keys = self.columns_latent_states
		keys = np.array(keys)
		keys_old = keys
		if tuple(keys) == tuple(self.columns_latent_states) and permute_metagenes: keys = keys[self.metagene_order]
		for ax, key, key_old in zip(np.array(axes).flat, keys, keys_old):
			self.visualizeFeatureSpace(ax, key, key_x, key_y, *args, **kwargs)
			ax.set_title(key_old)

	def visualizeLabelEnrichment(
			self, ax,
			key_x='cluster', order_x=None, ignores_x=(),
			key_y='cell type', order_y=None, ignores_y=(),
			**kwargs,
	):
		n_x = len(set(self.data[key_x].values) - set(ignores_x))
		n_y = len(set(self.data[key_y].values) - set(ignores_y))
		if order_x is None: order_x = self.orders.get(key_x)
		if order_y is None: order_y = self.orders.get(key_y)
		value_x, _, order_x = a2i(self.data[key_x].values, order_x, ignores_x)
		value_y, _, order_y = a2i(self.data[key_y].values, order_y, ignores_y)
		c = np.stack([value_x, value_y]).T
		c = c[~(c == -1).any(1)]
		c = c[:, 0] + c[:, 1] * n_x
		c = np.bincount(c, minlength=n_x * n_y).reshape(n_y, n_x)
		cp = c / c.sum(0, keepdims=True)

		im = ax.imshow(cp, vmin=0, vmax=1, aspect='auto', extent=(-.5, n_x - .5, -.5, n_y - .5), **kwargs)
		ax.set_xlabel(key_x)
		ax.set_ylabel(key_y)
		ax.set_xticks(range(n_x))
		ax.set_yticks(range(n_y)[::-1])
		ax.set_xticklabels(order_x, rotation=-90)
		ax.set_yticklabels(order_y)
		ax.set_ylim([-.5, n_y - .5])
		ax.set(frame_on=False)
		cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=.3)
		cbar.outline.set_visible(False)

		for i in range(n_y):
			for j in range(n_x):
				if c[i, j] == 0: continue
				text = ax.text(j, c.shape[0]-i-1, f'{c[i, j]:d}', ha="center", va="center", color="w" if cp[i, j] > .4 else 'k')

	def visualizeFeatureEnrichment(
			self, ax,
			keys_x=(), permute_metagenes=True,
			key_y='cluster', order_y=None, ignores_y=(),
			normalizer_raw=None,
			normalizer_avg=None,
			**kwargs,
	):
		n_y = len(set(self.data[key_y].values) - set(ignores_y))
		if order_y is None: order_y = self.orders.get(key_y)
		value_y, _, order_y = a2i(self.data[key_y].values, order_y, ignores_y)
		if len(keys_x) == 0: keys_x = self.columns_latent_states
		keys_x = np.array(keys_x)
		keys_x_old = keys_x
		if tuple(keys_x) == tuple(self.columns_latent_states) and permute_metagenes: keys_x = keys_x[self.metagene_order]
		n_x = len(keys_x)

		df = self.data[[key_y] + list(keys_x)].copy()
		if normalizer_raw is not None: df[keys_x] = normalizer_raw(df[keys_x].values)
		c = df.groupby(key_y)[keys_x].mean().loc[order_y].values
		if normalizer_avg is not None: c = normalizer_avg(c)

		if c.min() >= 0: vmin, vmax = 0, None
		else: vlim = np.abs(c).max(); vmin, vmax = -vlim, vlim
		im = ax.imshow(c, vmin=vmin, vmax=vmax, aspect='auto', extent=(-.5, n_x - .5, -.5, n_y - .5), **kwargs)
		ax.set_ylabel(key_y)
		ax.set_xticks(range(n_x))
		ax.set_yticks(range(n_y)[::-1])
		ax.set_xticklabels(keys_x_old, rotation=-90)
		ax.set_yticklabels(order_y)
		ax.set_ylim([-.5, n_y - .5])
		ax.set(frame_on=False)
		cbar = plt.colorbar(im, ax=ax, shrink=.3)
		cbar.outline.set_visible(False)

	def plotAffinityMetagenes(self, ax, iiter=-1, **kwargs):
		with openH5File(self.result_filename, 'r') as f:
			iiter = parseIiter(f['parameters/Sigma_x_inv'], iiter)
			print(f'Iteration {iiter}')
			Sigma_x_inv = f[f'parameters/Sigma_x_inv/{iiter}'][()]
		Sigma_x_inv = Sigma_x_inv[self.metagene_order, :]
		Sigma_x_inv = Sigma_x_inv[:, self.metagene_order]
		Sigma_x_inv = Sigma_x_inv - Sigma_x_inv.mean()
		vlim = np.abs(Sigma_x_inv).max()
		im = ax.imshow(Sigma_x_inv, vmin=-vlim, vmax=vlim, **kwargs)
		ticks = list(range(0, self.K - 1, 5)) + [self.K - 1]
		if len(ax.get_xticks()): ax.set_xticks(ticks)
		if ax.get_yticks: ax.set_yticks(ticks)
		ax.set_xticklabels(ticks)
		ax.set_yticklabels(ticks)
		ax.set_xlabel('metagene ID')
		ax.set_ylabel('metagene ID')
		cbar = plt.colorbar(im, ax=ax, pad=.01, shrink=.3, aspect=20)
		cbar.outline.set_visible(False)
		ax.set_frame_on(False)

	def plotAffinityClusters(self, ax, key='cluster', ignores=(), **kwargs):
		ignores = list(ignores)
		y, mapping, order = a2i(self.data[key].values, self.orders.get(key, None), ignores)
		y = y[y != -1]
		ncluster = len(set(y))
		n = np.bincount(y)  # number of cells in each cluster
		c = np.zeros([ncluster, ncluster])
		for repli, E in zip(self.repli_list, self.Es):
			yy = self.data.groupby('repli').get_group(repli)[key].values
			yy = np.fromiter(map(mapping.get, yy), dtype=int)
			c += np.bincount(
				[i * ncluster + j for i, e in zip(yy, E) if i != -1 for j in yy[e] if j != -1],
				minlength=c.size,
			).reshape(c.shape)
		assert (c == c.T).all(), (c - c.T)
		k = c.sum(0)  # degree of each cluster = sum of node deg
		m = c.sum()
		c -= np.outer(k, k / (m - 1))
		c.ravel()[::ncluster + 1] += k / (m - 1)
		c *= 2
		c.ravel()[::ncluster + 1] /= 2
		n = np.sqrt(n)
		c /= n[:, None]
		c /= n[None, :]

		vlim = np.abs(c).max()
		im = ax.imshow(c, vmax=vlim, vmin=-vlim, **kwargs)
		ax.set_xticks(range(ncluster))
		ax.set_yticks(range(ncluster))
		ax.set_xticklabels(order, rotation='270')
		ax.set_yticklabels(order)
		ax.set_xlabel(f'Cell clusters')
		ax.set_ylabel(f'Cell clusters')
		if key in self.colors:
			for tick_ind, tick in enumerate(ax.get_xticklabels()):
				bbox = dict(boxstyle="round", ec='none', fc=self.colors[key][order[tick_ind]], alpha=0.5, pad=.08)
				plt.setp(tick, bbox=bbox)
			for tick_ind, tick in enumerate(ax.get_yticklabels()):
				bbox = dict(boxstyle="round", ec='none', fc=self.colors[key][order[tick_ind]], alpha=0.5, pad=.08)
				plt.setp(tick, bbox=bbox)
		cbar = plt.colorbar(im, ax=ax, pad=.01, shrink=.3, aspect=20)
		cbar.outline.set_visible(False)
		ax.set_frame_on(False)
