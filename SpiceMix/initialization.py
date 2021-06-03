import sys, logging, time, resource, gc, os
from multiprocessing import Pool
from util import print_datetime

import numpy as np
from sklearn.cluster import KMeans
import gurobipy as grb


def NMF_stepX(YT, M, XT, prior_x, X_constraint, dropout_mode):
	N, G = YT.shape
	K = M.shape[1]

	mX = grb.Model('init_X')
	mX.setParam('OutputFlag', False)
	mX.setParam('Threads', 1)
	vx = mX.addVars(K, lb=0.)
	if X_constraint == 'sum2one':
		mX.addConstr(vx.sum('*') == 1)
		raise NotImplementedError
	elif X_constraint == 'none':
		pass
	else:
		raise NotImplementedError(f'Constraint on X {X_constraint} is not implemented')

	obj_share = 0
	if dropout_mode == 'raw':
		MTM = M[:G].T @ M[:G] + 1e-5*np.eye(K)
		obj_share += grb.quicksum([MTM[i, j] * vx[i] * vx[j] for i in range(K) for j in range(K)])  # quadratic term of X and M
		del MTM
		YTM = YT @ M[:G] * -2
	else:
		raise NotImplementedError(f'Dropout mode {dropout_mode} is not implemented')

	for x, y, yTM in zip(XT, YT, YTM):
		obj = obj_share
		if dropout_mode != 'raw':
			raise NotImplementedError(f'Dropout mode {dropout_mode} is not implemented')
		obj = obj + grb.quicksum([yTM[k] * vx[k] for k in range(K)]) + np.dot(y, y)
		mX.setObjective(obj, grb.GRB.MINIMIZE)
		mX.optimize()
		x[:] = np.fromiter((vx[i].x for i in range(K)), dtype=float)
	return XT


def initializeMXTsByPartialNMF(self, prior_x_modes, num_NMF_iter, num_processes=1):
	self.XTs = [np.zeros([N, self.K], dtype=float) for N in self.Ns]

	self.sigma_yx_invs = [1/YT.std(0).mean() for YT in self.YTs]
	self.prior_xs = []
	for prior_x_mode, YT in zip(prior_x_modes, self.YTs):
		G = YT.shape[1]
		if prior_x_mode == 'Truncated Gaussian' or prior_x_mode == 'Gaussian':
			mu_x = np.full(self.K, YT.sum(1).mean() / self.K)
			sigma_x_inv = np.full(self.K, np.sqrt(self.K) / YT.sum(1).std())
			self.prior_xs.append((prior_x_mode, mu_x, sigma_x_inv, ))
		elif prior_x_mode in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
			lambda_x = np.full(self.K, G / self.GG * self.K / YT.sum(1).mean())
			self.prior_xs.append((prior_x_mode, lambda_x, ))
		else:
			raise NotImplementedError(f'Prior on X {prior_x_mode} is not implemented')

	mM = grb.Model('init_M')
	mM.setParam('OutputFlag', False)
	mM.setParam('Threads', 1)
	if self.M_constraint == 'sum2one':
		vm = mM.addVars(self.GG, self.K, lb=0.)
		mM.addConstrs((vm.sum('*', i) == 1 for i in range(self.K)))
	elif self.M_constraint == 'nonnegative':
		vm = mM.addVars(self.K, lb=0.)
	else:
		raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')

	niter2 = num_NMF_iter
	iiter2 = 0
	last_M = np.copy(self.M)
	last_re = np.nan

	while iiter2 < niter2:
		# update XT
		with Pool(min(num_processes, len(self.YTs))) as pool:
			self.XTs = pool.starmap(NMF_stepX, zip(
				self.YTs, [self.M]*self.num_repli, self.XTs, prior_x_modes,
				[self.X_constraint]*self.num_repli, [self.dropout_mode]*self.num_repli,
			))
		pool.join()
		del pool

		iiter2 += 1

		if True:
			Ns = self.Ns
			nXTs = [XT / (XT.sum(1, keepdims=True)+1e-30) for XT in self.XTs]
			logging.info(print_datetime() + 'At iter {}: X: #0 = {},\t#all0 = {},\t#<1e-10 = {},\t#<1e-5 = {},\t#<1e-2 = {},\t#>1e-1 = {}'.format(
				iiter2,
				', '.join(map(lambda x: '%.2f' % x, [(nXT == 0).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%d' % x, [(nXT == 0).all(axis=1).sum() for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-10).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-5 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-2 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT>1e-1 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				))
			del nXTs

		# update prior_x
		prior_xs_old = self.prior_xs
		self.prior_xs = []
		for prior_x, XT in zip(prior_xs_old, self.XTs):
			if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
				mu_x = XT.mean(0)
				sigma_x_inv = 1. / XT.std(0)
				# sigma_x_inv /= 2			# otherwise, σ^{-1} is overestimated ???
				# sigma_x_inv = np.minimum(sigma_x_inv, 1e2)
				prior_x = (prior_x[0], mu_x, sigma_x_inv, )
			elif prior_x[0] in ['Exponential', 'Exponential shared']:
				lambda_x = 1. / XT.mean(0)
				prior_x = (prior_x[0], lambda_x, )
			elif prior_x[0] == 'Exponential shared fixed':
				pass
			else:
				raise NotImplementedError(f'Prior on X {prior_x[0]} is not implemented')
			self.prior_xs.append(prior_x)
		if any(prior_x[0] == 'Exponential shared' for prior_x in self.prior_xs):
			raise NotImplementedError(f'Prior on X Exponential shared is not implemented')

		# update sigma_yx_inv
		ds = [YT - XT @ self.M[:G].T for YT, XT, G in zip(self.YTs, self.XTs, self.Gs)]
		if self.dropout_mode == 'raw':
			ds = [d.ravel() for d in ds]
			sizes = np.fromiter(map(np.size, self.YTs), dtype=float)
		else:
			raise NotImplementedError(f'Dropout mode {self.dropout_mode} is not implemented')
		ds = np.fromiter((np.dot(d, d) for d in ds), dtype=float)
		if self.sigma_yx_inv_mode == 'separate':
			sigma_yx_invs = ds/sizes
			re = np.sqrt(np.dot(sigma_yx_invs, self.betas))
			self.sigma_yx_invs = 1. / np.sqrt(sigma_yx_invs + 1e-10)
		elif self.sigma_yx_inv_mode == 'average':
			sigma_yx_invs = np.dot(self.betas, ds) / np.dot(self.betas, sizes)
			re = np.sqrt(sigma_yx_invs)
			self.sigma_yx_invs = np.full(self.num_repli, 1 / np.sqrt(sigma_yx_invs + 1e-10))
		elif self.sigma_yx_inv_mode.startswith('average '):
			idx = np.fromiter(map(int, self.sigma_yx_inv_str.split(' ')[1:]), dtype=int)
			sigma_yx_invs = np.dot(self.betas[idx], ds[idx]) / np.dot(self.betas[idx], sizes[idx])
			re = np.sqrt(sigma_yx_invs)
			self.sigma_yx_invs = np.full(self.num_repli, 1 / np.sqrt(sigma_yx_invs + 1e-10))
		else:
			raise NotImplementedError(f'σ_y|x mode {self.sigma_yx_inv_mode} is not implemented')

		if True:
			logging.info(f'{print_datetime()}At iter {iiter2}: re: RMSE = {re:.2e}, diff = {re-last_re:.2e},')

		if iiter2 >= niter2: break

		if self.M_constraint == 'sum2one':
			obj = 0
			for XT, YT, G, beta, sigma_yx_inv in zip(self.XTs, self.YTs, self.Gs, self.betas, self.sigma_yx_invs):
				if self.dropout_mode == 'raw':
					# quadratic term
					XXT = XT.T @ XT * (beta * sigma_yx_inv**2)
					obj += grb.quicksum(XXT[i, i] * vm[k, i] * vm[k, i] for k in range(G) for i in range(self.K))
					XXT *= 2
					obj += grb.quicksum(XXT[i, j] * vm[k, i] * vm[k, j] for k in range(G) for i in range(self.K) for j in range(i+1, self.K))
					# linear term
					YXT = YT.T @ XT * (-2 * beta * sigma_yx_inv**2)
					YTY = np.dot(YT.ravel(), YT.ravel()) * beta * sigma_yx_inv**2
				else:
					raise NotImplementedError(f'Dropout mode {self.dropout_mode} is not implemented')

				obj += grb.quicksum(YXT[i, j] * vm[i, j] for i in range(G) for j in range(self.K))
				obj += beta * YTY
			kk = 1e-2
			if kk != 0:
				obj += grb.quicksum([kk/2 * vm[k, i] * vm[k, i] for k in range(self.GG) for i in range(self.K)])
			mM.setObjective(obj)
			mM.optimize()
			self.M = np.array([[vm[i, j].x for j in range(self.K)] for i in range(self.GG)])
			if self.M_constraint == 'sum2one':
				pass
			# elif M_sum2one == 'L1':
			# 	M /= np.abs(M).sum(0, keepdims=True) + 1e-10
			# elif M_sum2one == 'L2':
			# 	M /= np.sqrt((M**2).sum(0, keepdims=True)) + 1e-10
			else:
				raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')
		else:
			YXTs = [(YT.T @ XT) * beta for YT, XT, beta in zip(self.YTs, self.XTs, self.betas)]
			obj_2s = []
			for XT, beta in zip(self.XTs, self.betas):
				XXT = XT.T @ XT * beta
				obj_2s.append(grb.quicksum([XXT[i,j]*vm[i]*vm[j] for i in range(self.K) for j in range(self.K)]))
			for g, Mg in enumerate(self.M):
				obj = []
				for G, YXT, XT, obj_2 in zip(self.Gs, YXTs, self.XTs, obj_2s):
					if g >= G: continue
					obj.append(obj_2)
					obj.append(grb.quicksum([-2*YXT[g, i]*vm[i] for i in range(self.K)]))
				mM.setObjective(sum(obj, []), grb.GRB.MINIMIZE)
				mM.optimize()
				Mg[:] = np.array([vm[i].x for i in range(self.K)])
			raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')

		dM = self.M-last_M

		iiter2 += 1

		if True:
			logging.info(
				f'{print_datetime()}'
				f'At iter {iiter2}: '
				f'Diff M: max = {np.abs(dM).max():.2e}, '
				f'RMS = {np.sqrt(np.mean(np.abs(dM)**2)):.2e}, '
				f'mean = {np.abs(dM).mean():.2e}\t'
			)
			# print(prior_xs)
			sys.stdout.flush()

		last_M = np.copy(self.M)
		last_re = re

	# return M, XTs, sigma_yx_invs, prior_xs


def initializeMByKMeans(YTs, K, random_seed4kmeans=0):
	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)
	YT = np.concatenate([YT for YT in YTs if YT.shape[1] == GG], axis=0)
	n_init = 10
	logging.info(f'{print_datetime()}random seed for K-Means = {random_seed4kmeans}')
	logging.info(f'{print_datetime()}n_init for K-Means = {n_init}')
	kmeans = KMeans(
		n_clusters=K,
		random_state=random_seed4kmeans,
		n_jobs=1,
		n_init=n_init,
		tol=1e-8,
	).fit(YT)
	return kmeans.cluster_centers_.T


def initializeByKMean(self, random_seed4kmeans, num_NMF_iter=10, Sigma_x_inv_mode='Constant'):
	logging.info(f'{print_datetime()}Initialization begins')
	# initialize M
	self.M = initializeMByKMeans(self.YTs, self.K, random_seed4kmeans=random_seed4kmeans)
	if self.M_constraint == 'sum2one':
		self.M = np.maximum(self.M, 0)
		self.M /= self.M.sum(0, keepdims=True)
	else:
		raise NotImplementedError(f'Constraint on M {self.M_constraint} is not implemented')
	logging.info(f'{print_datetime()}Initialized M with shape {self.M.shape}')

	# initialize XT and perhaps update M
	# sigma_yx is estimated from XT and M
	initializeMXTsByPartialNMF(self, prior_x_modes=self.prior_x_modes, num_NMF_iter=num_NMF_iter)

	if all(self.Es_empty): Sigma_x_inv_mode = 'Constant'
	logging.info(f'{print_datetime()}Sigma_x_inv_mode = {Sigma_x_inv_mode}')
	if Sigma_x_inv_mode == 'Constant':
		self.Sigma_x_inv = np.zeros([self.K, self.K])
		self.delta_x = np.zeros(self.K)
	elif Sigma_x_inv_mode.startswith('Identity'):
		kk = float(Sigma_x_inv_mode.split()[1])
		self.Sigma_x_inv = np.eye(self.K) * kk
		self.delta_x = np.zeros(self.K)
	elif Sigma_x_inv_mode.startswith('EmpiricalFromX'):
		kk = float(Sigma_x_inv_mode.split()[1])
		Sigma_x = np.zeros([self.K, self.K])
		for XT, E, beta in zip(self.XTs, self.Es, self.betas):
			t = np.zeros_like(Sigma_x)
			for XTi, Ei in zip(XT, E):
				t += np.outer(XTi, XT[Ei].sum(0))
			self.Sigma_x += t * beta
		Sigma_x /= np.dot(self.betas, [sum(map(len, E)) for E in self.Es])
		self.Sigma_x_inv = np.linalg.inv(Sigma_x)
		del Sigma_x
		self.Sigma_x_inv *= kk
		self.delta_x = np.zeros(self.K)
	else:
		raise NotImplementedError(f'Initialization for Sigma_x_inv {Sigma_x_inv_mode} is not implemented')
