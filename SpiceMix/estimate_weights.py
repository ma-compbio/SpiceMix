from tqdm.auto import tqdm, trange

import numpy as np
import torch

from sample_for_integral import project2simplex
from util import NesterovGD


@torch.no_grad()
def estimate_weight_wonbr(
		Y, M, X, sigma_yx, prior_x_mode, prior_x, context, n_epochs=10000, tol=1e-5, update_alg='gd', verbose=True):
	"""
	min 1/2σ^2 || Y - X MT ||_2^2 + lam || X ||_1
	grad = X MT M / σ^2 - Y MT / σ^2 + lam

	TODO: use (projected) Nesterov GD. not urgent
	"""
	MTM = M.T @ M / (sigma_yx ** 2)
	YM = Y.to(M.device) @ M / (sigma_yx ** 2)
	Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
	step_size = 1 / torch.linalg.eigvalsh(MTM).max().item()
	loss_prev, loss = np.inf, np.nan
	pbar = trange(n_epochs, leave=True, disable=not verbose, miniters=1000)
	for i_epoch in pbar:
		X_prev = X.clone()
		if update_alg == 'mu':
			X.clip_(min=1e-10)
			loss = ((X @ MTM) * X).sum() / 2 - X.view(-1) @ YM.view(-1) + Ynorm / 2
			numerator = YM
			denominator = X @ MTM
			if prior_x_mode == 'exponential shared fixed':
				# see sklearn.decomposition.NMF
				loss += (X @ prior_x[0]).sum()
				denominator.add_(prior_x[0][None])
			else:
				raise NotImplementedError()

			loss = loss.item()
			assert loss <= loss_prev * (1 + 1e-4), (loss_prev, loss, (loss_prev - loss) / loss)
			mul_fac = numerator / denominator
			X.mul_(mul_fac).clip_(min=1e-10)
		elif update_alg == 'gd':
			t = X @ MTM
			loss = (t * X).sum().item() / 2
			g = t
			t = YM
			if prior_x_mode == 'exponential shared fixed':
				t = t - prior_x[0][None]
			else:
				raise NotImplementedError()
			loss -= (t * X).sum().item()
			g -= t
			loss += Ynorm / 2
			X.add_(g, alpha=-step_size).clip_(min=1e-10)
		else:
			raise NotImplementedError()

		dX = X_prev.sub(X).div(torch.linalg.norm(X, dim=1, ord=1, keepdim=True)).abs().max().item()
		do_stop = dX < tol
		if i_epoch % 1000 == 0 or do_stop:
			pbar.set_description(
				f'Updating weight w/o nbrs: loss = {loss:.1e} '
				f'%δloss = {(loss_prev - loss) / loss:.1e} '
				f'%δX = {dX:.1e}'
			)
		loss_prev = loss
		if do_stop: break
	pbar.close()
	return loss


class IndependentSet:
	def __init__(self, adj_list, batch_size=50):
		self.N = len(adj_list)
		self.adj_list = adj_list
		self.batch_size = batch_size
		self.indices_remaining = None

	def __iter__(self):
		self.indices_remaining = set(range(self.N))
		return self

	def __next__(self):
		# make sure selected nodes are not adjacent to each other
		# i.e., find an independent set of `indices_candidates` in a greedy manner
		if len(self.indices_remaining) == 0:
			raise StopIteration
		indices = []
		indices_exclude = set()
		indices_candidates = np.random.choice(
			list(self.indices_remaining),
			size=min(self.batch_size, len(self.indices_remaining)),
			replace=False,
		)
		for i in indices_candidates:
			if i in indices_exclude:
				continue
			else:
				indices.append(i)
				indices_exclude |= set(self.adj_list[i])
		self.indices_remaining -= set(indices)
		return list(indices)


@torch.no_grad()
def estimate_weight_wnbr(
		Y, M, X, sigma_yx, Sigma_x_inv, E, prior_x_mode, prior_x, context,
		n_epochs=100,
		tol=1e-2,
):
	"""
	The optimization for all variables
	min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + lam || S ||_1 + sum_{ij in E} ziT Σx-1 zj

	for s_i
	min 1/2σ^2 || y - M z s ||_2^2 + lam s
	s* = max(0, ( yT M z / σ^2 - lam ) / ( zT MT M z / σ^2) )

	for Z
	min 1/2σ^2 || Y - diag(S) Z MT ||_2^2 + sum_{ij in E} ziT Σx-1 zj
	grad_i = MT M z s^2 / σ^2 - MT y s / σ^2 + sum_{j in Ei} Σx-1 zj

	TODO: Try projected Newton's method.
	TM: Inverse is precomputed once, and projection is cheap. Not sure if it works theoretically
	"""
	tol = tol / X.shape[1]
	MTM = M.T @ M / (sigma_yx ** 2)
	if Y.device == M.device:
		YM = Y.to(M) @ M / (sigma_yx ** 2)
	else:
		YM = torch.empty_like(X)
		batch_size = 64
		for i in range(0, len(X), batch_size):
			slc = slice(i, i + batch_size)
			YM[slc] = Y[slc].to(M) @ M
		YM /= sigma_yx ** 2
	Ynorm = torch.linalg.norm(Y, ord='fro').item() ** 2 / (sigma_yx ** 2)
	step_size_base = 1 / torch.linalg.eigvalsh(MTM).max().item()
	S = torch.linalg.norm(X, dim=1, ord=1, keepdim=True)
	Z = X / S
	N = len(Z)

	E_adj_list = np.array(E, dtype=object)

	def get_adj_mat(adj_list):
		edges = [(i, j) for i, e in enumerate(adj_list) for j in e]
		adj_mat = torch.sparse_coo_tensor(np.array(edges).T, np.ones(len(edges)), size=[len(adj_list), N], **context)
		return adj_mat

	E_adj_mat = get_adj_mat(E_adj_list)

	def update_s():
		S[:] = (YM * Z).sum(1, keepdim=True)
		if prior_x_mode == 'exponential shared fixed':
			S.sub_(prior_x[0][0] / 2)
		else:
			raise NotImplementedError()
		S.div_((Z @ MTM).mul_(Z).sum(1, keepdim=True))
		S.clip_(min=1e-10)

	def update_z_gd_nesterov(Z):
		def calc_grad(Z_batch, S_batch, quad, linear):
			g = (Z_batch @ quad).mul_(S_batch.square()).sub_(linear)
			# g.sub_(g.sum(1, keepdim=True))
			return g
		pbar = trange(N, leave=False, disable=True, desc='Updating Z w/ nbrs via Nesterov GD')
		for batch_indices in IndependentSet(E_adj_list, batch_size=1024):
			Z_batch = Z[batch_indices].contiguous()
			S_batch = S[batch_indices].contiguous()
			quad_batch = MTM
			linear_batch_spatial = - get_adj_mat(E_adj_list[batch_indices]) @ Z @ Sigma_x_inv
			linear_batch = linear_batch_spatial + YM[batch_indices] * S_batch
			optimizer = NesterovGD(Z_batch, step_size_base / S_batch.square())
			Z_batch_prev = Z_batch.clone()
			for i_iter in range(1, 31):
				grad = calc_grad(Z_batch, S_batch, quad_batch, linear_batch)
				optimizer.step(grad)
				project2simplex(Z_batch, dim=1)
				if i_iter % 10 == 0:
					dZ = Z_batch_prev.sub_(Z_batch).abs_().max().item()
					Z_batch_prev = Z_batch.clone()
					if dZ < tol: break
			Z[batch_indices] = Z_batch
			pbar.update(len(batch_indices))
		pbar.close()

	# def update_zs_gd_nesterov():
	# 	def calc_grad_z(Z, S, quad, linear):
	# 		g = (Z @ quad).mul_(S ** 2).sub_(linear)
	# 		# g.sub_(g.sum(1, keepdim=True))
	# 		return g
	#
	# 	def update_s(Z, S, YM):
	# 		S[:] = (YM * Z).sum(1, keepdim=True)
	# 		if prior_x_mode == 'exponential shared fixed':
	# 			# TODO: double check
	# 			S.sub_(prior_x[0][0] / 2)
	# 		else:
	# 			raise NotImplementedError()
	# 		S.div_(((Z @ MTM).mul_(Z)).sum(1, keepdim=True))
	# 		S.clip_(min=1e-10)
	#
	# 	pbar = trange(N, leave=False, disable=True, desc='Updating Z w/ nbrs via Nesterov GD')
	# 	for batch_indices in IndependentSet(E_adj_list, batch_size=256):
	# 		quad_batch = MTM
	# 		linear_batch_spatial = - get_adj_mat(E_adj_list[batch_indices]) @ Z @ Sigma_x_inv
	# 		Z_batch = Z[batch_indices].contiguous()
	# 		Z_batch_prev = Z_batch.clone()
	# 		S_batch = S[batch_indices].contiguous()
	# 		YM_batch = YM[batch_indices].contiguous()
	# 		optimizer = NesterovGD(Z_batch, step_size_base / S_batch.square())
	# 		valid_freq = 10
	# 		for i_iter in range(1, 101): # for scDesign2-based
	# 			update_s(Z_batch, S_batch, YM_batch)
	# 			linear_batch = linear_batch_spatial + YM_batch * S_batch
	# 			NesterovGD.step_size = step_size_base / S_batch.square() # TM: I think this converges as s converges
	# 			grad = calc_grad_z(Z_batch, S_batch, quad_batch, linear_batch)
	# 			optimizer.step(grad)
	# 			project2simplex(Z_batch, dim=1)
	# 			if i_iter % valid_freq == 0:
	# 				dZ = Z_batch_prev.sub_(Z_batch).abs_().max().item()
	# 				if dZ < tol * np.sqrt(valid_freq):
	# 					break
	# 				Z_batch_prev[:] = Z_batch
	# 		Z[batch_indices] = Z_batch
	# 		S[batch_indices] = S_batch
	# 		pbar.update(len(batch_indices))
	# 	pbar.close()

	def calc_loss(loss_prev):
		X = Z * S
		# loss = (X @ MTM).ravel() @ X.ravel() / 2 - X.ravel() @ YM.ravel() + Ynorm / 2
		loss = torch.addmm(YM, X, MTM, alpha=.5, beta=-1).ravel() @ X.ravel() + Ynorm / 2
		if prior_x_mode == 'exponential shared fixed':
			loss += prior_x[0][0] * S.sum()
		else:
			raise NotImplementedError()
		if Sigma_x_inv is not None:
			loss += ((E_adj_mat @ Z) @ Sigma_x_inv).ravel() @ Z.ravel() / 2
		loss = loss.item()
		return loss, loss_prev - loss

	loss = np.inf
	pbar = trange(n_epochs, desc='Updating weight w/ neighbors')

	for i_epoch in pbar:
		update_s()
		Z_prev = Z.clone().detach()
		update_z_gd_nesterov(Z)
		# update_zs_gd_nesterov()
		loss, dloss = calc_loss(loss)
		dZ = Z_prev.sub(Z).abs_().max().item()
		pbar.set_description(
			f'Updating weight w/ neighbors: loss = {loss:.1e} '
			f'δloss = {dloss:.1e} '
			f'δZ = {dZ:.1e}'
		)
		if dZ < tol:
			break

	X[:] = Z * S
	return loss
