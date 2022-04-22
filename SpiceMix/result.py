import sys, os, itertools, re, gc, copy, time
from tqdm.auto import tqdm
import h5py
from pathlib import Path
from collections import Iterable, OrderedDict, defaultdict, Counter
import pandas as pd
from util import print_datetime, openH5File, parseIiter, a2i

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr, linregress, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests
import umap
import anndata
import scanpy as sc
import networkx as nx
from networkx.exception import NetworkXUnfeasible
from multiprocessing import Pool

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from loadData import loadExpression, loadDataset
from maximize_ari_spatiallibd import smooth_discrete, smooth_continuous


def plotConvergenceQ(ax, path2dataset, result_filename, **kwargs):
	with openH5File(path2dataset / 'results' / result_filename, 'r') as f:
		g = f['progress/Q']
		k = np.fromiter(map(int, g.keys()), dtype=int)
		v = np.fromiter((g[str(_)][()] for _ in k), dtype=float)
	Q = np.full(k.max() - k.min() + 1, np.nan)
	Q[k - k.min()] = v
	print(f'Found {k.max()} iterations from {result_filename}')
	for kk, linestyle in zip([1, 5, 25], ['-', '--', ':']):
		if kk in [25]: continue
		if kk >= len(Q): continue
		dQ = (Q[kk:] - Q[:-kk]) / kk
		# print(f'min dQ = {dQ[np.logical_not(np.isnan(dQ))].min():.2e}')
		# dQ += 1e-1
		ax.plot(np.arange(k.min(), k.max() + 1 - kk) + kk / 2 + 1, dQ, linestyle=linestyle, **kwargs)


def plotQ(ax, path2dataset, result_filename, **kwargs):
	with openH5File(path2dataset / 'results' / result_filename, 'r') as f:
		g = f['progress/Q']
		k = np.fromiter(map(int, g.keys()), dtype=int)
		v = np.fromiter((g[str(_)][()] for _ in k), dtype=float)
	print(f'Found {k.max()} iterations from {result_filename}')
	idx = np.argsort(k)
	ax.plot(k[idx], v[idx], **kwargs)


def findBest(path2dataset, result_filenames, iiter=-1):
	Q = []
	for r in result_filenames:
		f = openH5File(path2dataset / 'results' / r, 'r')
		if f'progress/Q' not in f:
			Q.append(-np.inf)
			print(f'progress/Q not found')
			continue
		i = parseIiter(f[f'progress/Q'], iiter)
		print(f'Using iteration {i} from {r}')
		try:
			Q.append(f[f'progress/Q/{i}'][()])
		except:
			Q.append(-np.inf)
			print(f"Iteration {i} was not reached. Found {parseIiter(f[f'progress/Q'], -1)} iterations")
		f.close()
	i = np.argmax(Q)
	print(f'The best one is model #{i} - result filename = {result_filenames[i]}')
	return result_filenames[i]


def calc_modularity(A, label, resolution=1):
	n = A.shape[0]
	x = A.indices
	y = np.fromiter((
		i for i, l in enumerate(A.indptr[1:] - A.indptr[:-1])
		for _ in range(l)), dtype=int)
	Asum = A.data.sum()
	score = A.data[label[x] == label[y]].sum() / Asum

	idx = np.argsort(label)
	label = label[idx]
	k = np.array(A.sum(0)).ravel() / Asum
	k = k[idx]
	idx = np.concatenate([[0], np.nonzero(label[:-1] != label[1:])[0] + 1, [len(label)]])
	score -= sum(k[i:j].sum() ** 2 for i, j in zip(idx[:-1], idx[1:])) * resolution
	return score


def visualize_differential_features(axes, data, x, y, kwargs_plot=None, hatch=None):
	if kwargs_plot is None: kwargs_plot = {}
	l = []
	for _ in y:
		df = data[[x, _]].copy()
		df.columns = [x, 'expression']
		df['feature'] = _
		l.append(df)
	data = pd.concat(l, axis=0).reset_index()

	ax = axes[0]
	sns.boxplot(data=data[data['expression'] > 0], x='feature', y='expression', hue=x, ax=ax, **kwargs_plot)
	# sns.stripplot(data=data, x='feature', y='expression', hue=x, color='.25', size=4, dodge=True, ax=ax)
	ax.set_ylim(bottom=0)

	ax = axes[1]
	data['expression'] = data['expression'] > 0
	sns.barplot(data=data, x='feature', y='expression', hue=x, ci=None, ax=ax, **kwargs_plot, edgecolor='k')
	ax.set_ylim([0, 1])
	ax.set_yticks([0, .5, 1])

	if hatch is not None:
		for box in axes[0].artists:
			box.set_hatch(hatch)
		for bar in axes[1].patches:
			bar.set_hatch(hatch)
		axes[0].legend()
		axes[1].legend()


def visualize_total_counts(ax, data, x, y, kwargs_plot=None, hatch=None):
	if kwargs_plot is None: kwargs_plot = {}
	data = data.copy()
	data['total count'] = data[y].sum(1)
	sns.boxplot(data=data, x=x, y='total count', ax=ax, **kwargs_plot)
	if hatch is not None:
		for box in ax.artists:
			box.set_hatch(hatch)


def plot_dendrogram(
		self, ax, X=None, features=None, key='cluster', ignores_y=(),
		n_components=10, linkage_method=None, metric=None,
):
	ignores_y = set(ignores_y)

	mask = ~self.data[key].isin(ignores_y).values
	if X is None:
		cols = self.columns_exprs if features is None else features
		X = self.data[cols].values[mask].copy()
	else:
		X = X[mask].copy()
	if n_components: X = PCA(n_components=n_components, svd_solver='arpack').fit_transform(X)
	data = pd.DataFrame(data=X).groupby(self.data[key].values[mask]).mean()
	num_clusters = len(data)
	leaf_label = tuple(data.index.values)
	dist_compact = pdist(data.values, metric=metric)
	dist = pd.DataFrame(squareform(dist_compact), index=data.index, columns=data.index)
	Z = linkage(dist_compact, method=linkage_method)
	dendro = dendrogram(
		Z=Z, truncate_mode='lastp', p=num_clusters, distance_sort='ascending', no_plot=True,
		leaf_label_func=lambda i, arr=leaf_label: arr[i],
	)

	for xx, yy in zip(dendro['icoord'], dendro['dcoord']): ax.plot(xx, yy, c='black')
	ax.set_ylim([0, ax.get_ylim()[1]])
	t = np.array(dendro['icoord'])
	ax.set_xticks(np.linspace(t.min(), t.max(), num_clusters))
	ax.set_xticklabels(dendro['ivl'], rotation=-90)
	if key in self.colors:
		for tick_ind, tick in zip(dendro['leaves'], ax.get_xticklabels()):
			bbox = dict(boxstyle="round", ec='none', fc=self.colors[key][leaf_label[tick_ind]], alpha=0.5, pad=.5)
			plt.setp(tick, bbox=bbox)
	ax.tick_params(axis='both', which='both', length=0, labelsize=20)
	ax.set(frame_on=False)
	ax.set_yticks([])

	return dist


def visualizeContinuousFeatureScatter(
		self, ax, key, cell_mask=slice(None),
		radius=4, radius_thr=1e-5,
		**kwargs,
):
	# adapted from visualizeFeatureSpace for spatialLIBD
	df = self.data[['coor img X', 'coor img Y', key]].iloc[cell_mask].copy()
	coor = df[['coor img X', 'coor img Y']].values
	v = df[key].values.copy()
	v = np.clip(v, a_min=None, a_max=np.quantile(v, .99))
	s = v / v.max()
	idx = s > radius_thr
	sca = ax.scatter(*coor[idx].T, c=v[idx], s=s[idx]*radius, **kwargs)
	cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.tick_params(axis='both', labelsize=10)


def visualizeContinuousFeatureContour(
		self, ax, key, cell_mask=slice(None),
		kwargs_smooth_continuous=None,
		kwargs_kde=None,
		kwargs_contour=None,
):
	# adapted from visualizeFeatureSpace for spatialLIBD
	df = self.data[['coor img X', 'coor img Y', key]].iloc[cell_mask].copy()
	coor = df[['coor img X', 'coor img Y']].values
	coor2 = np.meshgrid(
		np.linspace(coor[:, 0].min(), coor[:, 0].max(), 100),
		np.linspace(coor[:, 1].min(), coor[:, 1].max(), 100),
	)
	y = df[key].values
	if kwargs_smooth_continuous is not 'none':
		if kwargs_smooth_continuous is None: kwargs_smooth_continuous = {}
		# radius=2
		y = smooth_continuous(df[['coor X', 'coor Y']].values, y, df['repli'].values, **kwargs_smooth_continuous)
	if kwargs_kde is None: kwargs_kde = {}
	# kernel='gaussian', bandwidth=10
	p = KernelDensity(**kwargs_kde) \
		.fit(coor, sample_weight=y) \
		.score_samples(np.stack(coor2).reshape(2, -1).T)
	p -= logsumexp(p)
	p = np.exp(p) * y.sum()
	p = p.reshape(coor2[0].shape)
	if kwargs_contour is None: kwargs_contour = {}
	h = ax.contourf(*coor2, p, **kwargs_contour)
	cbar = plt.colorbar(h, ax=ax, pad=.01, shrink=1, aspect=40)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.tick_params(axis='both', labelsize=10)


def calcPermutation(sim):
	"""
	maximum weight bipartite matching
	:param sim:
	:return: sim[perm, index], where index is sorted
	"""
	Ks = sim.shape
	B = nx.Graph()
	B.add_nodes_from(['o{}'.format(i) for i in range(Ks[0])], bipartite=0)
	B.add_nodes_from(['t{}'.format(i) for i in range(Ks[1])], bipartite=1)
	B.add_edges_from([
		('o{}'.format(i), 't{}'.format(j), {'weight': sim[i, j]})
		for i in range(Ks[0]) for j in range(Ks[1])
	])
	assert nx.is_bipartite(B)
	matching = nx.max_weight_matching(B, maxcardinality=True)
	assert len(set(__ for _ in matching for __ in _)) == Ks[0] * 2
	matching = [_ if _[0][0] == 'o' else _[::-1] for _ in matching]
	matching = [tuple(int(__[1:]) for __ in _) for _ in matching]
	matching = sorted(matching, key=lambda x: x[1])
	perm, index = tuple(map(np.array, zip(*matching)))
	return perm, index


def calcEarthMoversDistance(dist, dist_thr_quantile=1, verbose=True):
	# toooooo slow
	if dist.shape[0] > dist.shape[1]: dist = dist.T
	n, m = dist.shape
	tic = time.perf_counter()
	try:
		dist_thr_0 = np.quantile(dist, dist_thr_quantile, axis=1)
		dist_thr_1 = np.quantile(dist, dist_thr_quantile, axis=0)
		dist_thr_f = np.quantile(dist.ravel(), dist_thr_quantile)
		G = nx.DiGraph()

		G.add_nodes_from([f'x{i}' for i in range(n)], bipartite=0, demand=-m)
		G.add_nodes_from([f'y{j}' for j in range(m)], bipartite=1, demand=n)
		G.add_edges_from([
			(f'x{i}', f'y{j}', {'weight': dist[i, j]})
			for i in range(n) for j in range(m)
			if dist[i, j] <= max(dist_thr_0[i], dist_thr_1[j], dist_thr_f)
		],
			capacity=min(n, m),
			# capacity=np.inf,
		)

		# G.add_nodes_from(['s', 't'])
		# G.add_nodes_from([f'x{i}' for i in range(n)])
		# G.add_nodes_from([f'y{j}' for j in range(m)])
		# G.add_edges_from([('s', f'x{i}') for i in range(n)], capacity=m, weight=1)
		# G.add_edges_from([(f'y{j}', 't') for j in range(m)], capacity=n, weight=1)
		# G.add_edges_from([
		# 	(f'x{i}', f'y{j}', {'weight': dist[i, j]}) for i in range(n) for j in range(m)],
		# 	capacity=min(n, m),
		# 	# capacity=np.inf,
		# )

		# print(G.edges.data())
		# cost = nx.min_cost_flow_cost(G)
		# flow_dict = nx.max_flow_min_cost(G, 's', 't')
		# flow_value, flow_dict = nx.maximum_flow(G, 's', 't')
		flow_cost, flow_dict = nx.capacity_scaling(G)
		# assert flow_value == n*m
		# print(flow_dict)

		if verbose: print(f'{n}\t{m}\t{dist_thr_quantile:.2f}\tsucceeded\t{time.perf_counter() - tic:.2f}')
	except NetworkXUnfeasible as e:
		if verbose: print(f'{n}\t{m}\t{dist_thr_quantile:.2f}\tfailed\t{time.perf_counter() - tic:.2f}')
		return np.inf
	if verbose:
		print(
			f'# of nodes = {n} {m}',
			f'# of edges = {len(G.edges.data())}',
			f'# of used edges = {sum(__ != 0 for _ in flow_dict.values() for __ in _.values())}',
			sep='\t',
		)
	cost = nx.cost_of_flow(G, flow_dict)
	assert np.allclose(flow_cost, cost)
	return cost / n / m


def calcEarthMoversDistance2(x1, x2):
	# toooooo slow
	d = cdist(x1, x2)
	m, n = len(x1), len(x2)
	lcm = np.lcm(m, n)
	d = d.repeat(lcm // m, axis=0).repeat(lcm // n, axis=1)
	assignment = linear_sum_assignment(d)
	return d[assignment].sum() / lcm


def calcEarthMoversDistance_batch(
		obj, key_coor=('coor X', 'coor Y'), key_y='cell type', key_yhat='cluster',
		downsample_rate=1, nCPU=1, verbose=True,
):
	# toooooo slow
	dist = []

	pool = Pool(nCPU) if nCPU > 1 else None
	for r, df in obj.data.groupby('repli'):
		coor = df[list(key_coor)].values
		y = df[key_y].values
		yhat = df[key_yhat].values
		for a, b in itertools.product(np.unique(y), np.unique(yhat)):
			x1, x2 = coor[y == a], coor[yhat == b]
			x1 = x1[np.random.choice(len(x1), size=int(len(x1) * downsample_rate), replace=False)]
			x2 = x2[np.random.choice(len(x2), size=int(len(x2) * downsample_rate), replace=False)]
			if verbose: print(a, b, len(x1), len(x2), sep='\t')
			# for thr in np.linspace(.8, 1., 3):
			for thr in [1]:
				dist.append({
					'repli': r, 'cell type': a, 'cluster': b, 'thr': thr,
					'dist':
						pool.apply_async(calcEarthMoversDistance, (cdist(x1, x2), thr, verbose))
						if pool is not None else
						calcEarthMoversDistance(cdist(x1, x2), thr, verbose),
					# 'dist2': calcEarthMoversDistance2(x1, x2),
				})
			# break
	if pool is not None:
		for d in dist: d['dist'] = d['dist'].get(1e8)
		pool.close()
		pool.join()
	dist = pd.DataFrame(dist)
	return dist


def kde1d(x, w, xx, bw):
	if w is None: pass
	else:
		assert (w >= 0).all()
		idx = w > 0
		x = x[idx]
		w = w[idx]
	p = KernelDensity(kernel='gaussian', bandwidth=bw)\
		.fit(x[:, None], sample_weight=w)\
		.score_samples(xx[:, None])
	p -= logsumexp(p)
	p = np.exp(p)
	p *= len(xx) / (xx.max() - xx.min())
	return p


def visualize_feature_1d(
	obj,
	ax,
	layers, coef, intercepts,
	signals=(), signal_labels=(), bws=(),
	signals_twinx=None, signal_labels_twinx=None, bws_twinx=None,
	bg_signal=None,
	bg_signal_twinx=('cell count'),
	fac=None,
	vlines=(),
):
	if fac is None: fac = 1
	coef = coef.copy()
	intercepts = intercepts.copy()
#     coor = obj.data[['coor X', 'coor Y']].values.copy()
	coor = obj.data[['coor img X', 'coor img Y']].values.copy()
	x = coor @ coef
	xx = np.linspace(x.min(), x.max(), 500)
	y_bg = kde1d(x, None if bg_signal is None else obj.data[bg_signal], xx, 4)
	y_bg_twinx = kde1d(x, None if bg_signal_twinx is None else obj.data[bg_signal_twinx], xx, 4)
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(8, 4))
	ax_twinx = ax.twinx()
	if signals_twinx is None:
		ax_twinx.plot(xx * fac, y_bg, label=f'bg', c='grey', linestyle=':')
	else:
		for i, (signal, signal_label, bw) in enumerate(zip(signals_twinx, signal_labels_twinx, bws_twinx)):
			y = kde1d(x, signal, xx, bw) / y_bg_twinx
			y /= y.sum()
			y *= len(xx) / (xx.max() - xx.min()) / fac
			ax_twinx.plot(xx * fac, y, label=signal_label, linestyle=':', c=f'C{i+len(signals)}')
	for i, (signal, signal_label, bw) in enumerate(zip(signals, signal_labels, bws)):
		y = kde1d(x, signal, xx, bw) / y_bg
		y /= y.sum()
		y *= len(xx) / (xx.max() - xx.min()) / fac
		ax.plot(xx * fac, y, label=signal_label)
	ylim = ax.get_ylim()
	for l, lb, rb in zip(layers, intercepts[:-1], intercepts[1:]):
		ax.fill_between(
			[lb * fac, rb * fac], *ylim,
#             transform=ax.get_yaxis_transform(),
			alpha=.3, label=l, color=obj.colors['layer'][l], edgecolor='none',
		)
	ax.set_xlim(x.min() * fac, x.max() * fac)
	for vline in vlines:
		ax.axvline(vline * fac, linestyle=':', c='grey')
	ax.set_ylim(ylim)
	ax.legend(bbox_to_anchor=(1.05, .5), loc='center left')
	ax_twinx.legend(bbox_to_anchor=(1.35, .5), loc='center left')


class Result:
	def __init__(
			self, path2dataset, result_filename,
			neighbor_suffix=None, expression_suffix=None,
			showHyperparameters=False,
			expression_suffix4scaling=None,
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
		if not hasattr(self, 'lambda_x'): self.lambda_x = 1.
		self.num_repli = len(self.repli_list)
		self.use_spatial = [True] * self.num_repli
		if expression_suffix4scaling is not None:
			loadDataset(self, neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix4scaling)
			self.scaling = [G / self.GG * self.K / YT.sum(1).mean() for YT, G in zip(self.YTs, self.Gs)]
		loadDataset(self, neighbor_suffix=neighbor_suffix, expression_suffix=expression_suffix)
		self.columns_latent_states = np.array([f'latent state {i}' for i in range(self.K)])
		self.columns_exprs = np.array([f'expr {_}' for _ in self.genes[0]])
		self.data = pd.DataFrame(index=range(sum(self.Ns)))
		try:
			df = pd.concat([
				pd.read_csv(self.path2dataset / 'files' / f'meta_{repli}.csv', keep_default_na=False)
				for repli in self.repli_list
			], axis=0, ignore_index=True)
			self.data = pd.concat([self.data, df], axis=1)
			del df
		except OSError:
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
		if not hasattr(self, 'scaling'):
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
		XT = np.concatenate(XTs)
		self.data[[f'latent state {i}' for i in range(self.K)]] = XT
		return XT

	def loadMetagene(self, iiter=-1):
		with openH5File(self.result_filename, 'r') as f:
			iiter = parseIiter(f[f'parameters/M'], iiter)
			print(f'Iteration {iiter}')
			M = f[f'parameters/M/{iiter}'][()]
		self.M = M
		return M

	def clustering(
			self, ax, K_range, K_offset=0, min_cluster_size=1, threshold=None, criterion='CH',
			normalizer='z-score'):
		XT = self.data[self.columns_latent_states].values.copy()
		if threshold is not None:
			XT[XT < np.quantile(XT, threshold, axis=0, keepdims=True)] = 0.
		if normalizer == 'z-score':
			XT = StandardScaler().fit_transform(XT)
		elif normalizer is not None:
			XT = normalizer(XT)
		else:
			pass
		K_range = np.array(K_range)

		print(f'Creating linkage')
		Z = linkage(XT, method='ward', metric='euclidean')
		print(f'Creating clusters')
		f = lambda K: fcluster(Z, K, criterion='maxclust') - 1
		scores = []
		for K in K_range:
			y = f(K)
			if len(set(y)) == 1: scores.append(0)
			elif criterion == 'CH': scores.append(calinski_harabasz_score(XT, y))
			elif criterion == 'ARI': scores.append(adjusted_rand_score(self.data['cell type'], y))
			else: raise NotImplementedError
		scores = np.array(scores)
		K_opt = K_range[scores.argmax() + K_offset]

		if ax is not None:
			ax.scatter(K_range, scores, marker='x', color=np.where(K_range == K_opt, 'C1', 'C0'))

		print(f'K_opt = {K_opt}')
		y = f(K_opt)
		i = np.unique(y[y != -1], return_counts=True)[1]
		i = i >= min_cluster_size
		j = np.cumsum(i) - 1
		j[~i] = -1
		j = np.concatenate([j, [-1]])
		y = j[y]
		del i, j
		idx = y != -1
		print(f'#clusters = {len(set(y) - {-1})}, #-1 = {(y == -1).sum()}')
		self.data['cluster_raw'] = y
		self.data['cluster'] = list(map(str, y))
		return XT

	def clusteringLouvain(self, kwargs_neighbors, kwargs_clustering, threshold=None, num_rs=10, normalizer=None, method='louvain'):
		X = self.data[self.columns_latent_states].values
		if threshold is not None:
			X[X < np.quantile(X, threshold, axis=0, keepdims=True)] = 0.
		if normalizer is not None:
			X = normalizer(X)
		adata = anndata.AnnData(X)
		sc.pp.neighbors(adata, use_rep='X', **kwargs_neighbors)
		best = {'score': np.nan}
		for rs in tqdm(range(num_rs)):
			getattr(sc.tl, method)(adata, **kwargs_clustering, random_state=rs)
			cluster = np.array(list(adata.obs[method]))
			score = calc_modularity(
				adata.obsp['connectivities'], cluster, resolution=kwargs_clustering.get('resolution', 1))
			if not best['score'] >= score: best.update({'score': score, 'cluster': cluster.copy(), 'rs': rs})
		print(f"best: score = {best['score']}\trs = {best['rs']}\t# of clusters = {len(set(best['cluster']))}")
		self.data['cluster_raw'] = best['cluster'].copy()
		self.data['cluster'] = [f'c{_}' for _ in self.data['cluster_raw']]
		return X

	def annotateClusters(self, clusteri2a):
		self.data['cluster'] = [clusteri2a[_] for _ in self.data['cluster_raw']]

	def assignColors(self, key, mapping):
		assert set(mapping.keys()) >= set(self.data[key]), f'Values [{set(set(self.data[key]))-set(mapping.keys())}] are not assigned'
		self.colors[key] = copy.deepcopy(mapping)

	def assignOrder(self, key, order):
		s = set(self.data[key])
		assert set(order) >= s, f'The input set `order`=[{order}] must be a superset of [{s}]'
		assert len(order) == len(set(order)), f'The input set `order`=[{order}] contains duplicate elements.'
		order = list(filter(lambda _: _ in s, order))
		self.orders[key] = np.array(order)

	def getLatentState(self):
		return self.data[self.columns_latent_states[self.metagene_order]].values

	def getExpression(self):
		df = self.data[self.columns_exprs].copy()
		df.columns = [_[5:] for _ in df.columns]
		return df

	def UMAP(self, **kwargs):
		XT = self.data[self.columns_latent_states].values
		XT = StandardScaler().fit_transform(XT)
		XT = umap.UMAP(**kwargs).fit_transform(XT)
		self.data[[f'UMAP {i+1}' for i in range(XT.shape[1])]] = XT

	def visualizeFeatureSpace(
			self, ax, key, key_x='UMAP 1', key_y='UMAP 2', repli=None, annotate=False, cell_mask=slice(None),
			**kwargs,
	):
		if isinstance(repli, int): repli = self.repli_list[repli]
		if repli is None:
			data = self.data
		else:
			data = self.data.groupby('repli').get_group(repli)
		data = data.iloc[cell_mask]
		kwargs = copy.deepcopy(kwargs)
		if data[key].dtype == 'O':
			kwargs.setdefault('hue_order', self.orders.get(key, None))
			kwargs.setdefault('palette', self.colors.get(key, None))
			sns.scatterplot(ax=ax, data=data, x=key_x, y=key_y, hue=key, **kwargs)
		else:
			kwargs.setdefault('cmap', self.colors.get(key, None))
			sca = ax.scatter(data[key_x], data[key_y], c=data[key], **kwargs)
			cbar = plt.colorbar(sca, ax=ax, pad=.01, shrink=1, aspect=40)
		if annotate:
			for t, (x, y) in data[[key_x, key_y]].groupby(data[key]).mean().iterrows():
				ax.text(
					x, y, t, horizontalalignment='center', verticalalignment='center',
					color=kwargs['palette'].get(t) if isinstance(kwargs['palette'], dict) else None,
				)
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
			reorder_x_automatically=False,
			repli=None,
			**kwargs,
	):
		n_x = len(set(self.data[key_x].values) - set(ignores_x))
		n_y = len(set(self.data[key_y].values) - set(ignores_y))
		if order_x is None: order_x = self.orders.get(key_x)
		if order_y is None: order_y = self.orders.get(key_y)
		data = self.data if repli is None else self.data.groupby('repli').get_group(repli)
		value_x, _, order_x = a2i(data[key_x].values, order_x, ignores_x)
		value_y, _, order_y = a2i(data[key_y].values, order_y, ignores_y)
		c = np.stack([value_x, value_y]).T
		c = c[~(c == -1).any(1)]
		c = c[:, 0] + c[:, 1] * n_x
		c = np.bincount(c, minlength=n_x * n_y).reshape(n_y, n_x)
		cp = c / c.sum(0, keepdims=True)
		if reorder_x_automatically:
			idx = np.argsort(cp.argmax(0))
			c = c[:, idx]
			cp = cp[:, idx]
			order_x = order_x[idx]

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
				text = ax.text(j, c.shape[0] - i - 1, f'{c[i, j]:d}', ha="center", va="center",
							   color="w" if cp[i, j] > .4 else 'k')

	def visualizeFeatureEnrichment(
			self, ax,
			keys_x=(), permute_metagenes=True,
			key_y='cluster', order_y=None, ignores_y=(),
			reorder_x_automatically=False,
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
		c = df.groupby(key_y).mean().loc[order_y].values
		if normalizer_avg is not None: c = normalizer_avg(c)

		if reorder_x_automatically:
			idx = np.argsort(c.argmax(0))
			c = c[:, idx]
			keys_x_old = keys_x_old[idx]
			#         idx_inv = np.empty_like(idx)
			#         idx_inv[idx] = np.arange(len(idx))
			print(', '.join(map(str, idx)))

		if c.min() >= 0:
			vmin, vmax = 0, None
		else:
			vlim = np.abs(c).max(); vmin, vmax = -vlim, vlim
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

	def visualizeAffinityMetagenes(self, ax, iiter=-1, **kwargs):
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

	def visualizeAffinityClusters(self, ax, key='cluster', ignores=(), **kwargs):
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
