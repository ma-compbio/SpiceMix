from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd
import torch

from sample_for_integral import integrate_of_exponential_over_simplex, project2simplex
from util import NesterovGD, Minibatcher


@torch.no_grad()
def project_M(M, M_constraint):
	if M_constraint == 'simplex':
		project2simplex(M, dim=0)
	elif M_constraint == 'unit sphere':
		M.div_(torch.linalg.norm(M, ord=2, dim=0, keepdim=True))
	elif M_constraint == 'nonneg unit sphere':
		M.clip_(1e-10).div_(torch.linalg.norm(M, ord=2, dim=0, keepdim=True))
	else:
		raise NotImplementedError()
	return M


@torch.no_grad()
def estimate_M(
		Ys, Xs, M, sigma_yxs, betas, M_constraint, context,
		n_epochs=10000, tol=None, verbose=True,
):
	"""
	M is shared across all replicates
	min || Y - X MT ||_2^2 / (2 σ_yx^2)
	s.t. || Mk ||_p = 1
	grad = (M XT X - YT X) / (σ_yx^2)

	Each replicate may have a slightly different M
	min || Y - X MT ||_2^2 / (2 σ_yx^2) + || M - M_base ||_2^2 λ_M / 2
	s.t. || Mk ||_p = 1
	grad = ( M XT X - YT X ) / ( σ_yx^2 ) + λ_M ( M - M_base )
	"""
	if tol is None:
		tol = 1e-5
	tol /= len(M)
	K = M.shape[1]
	quad = torch.zeros([K, K], **context)
	linear = torch.zeros_like(M)
	constant = (np.array([torch.linalg.norm(Y, ord='fro').item()**2 for Y in Ys]) * betas / (sigma_yxs**2)).sum()
	for Y, X, sigma_yx, beta in zip(Ys, Xs, sigma_yxs, betas):
		quad.addmm_(X.T, X, alpha=beta / (sigma_yx ** 2))
		linear.addmm_(Y.T.to(X.device), X, alpha=beta / (sigma_yx ** 2))

	loss_prev, loss = np.inf, np.nan
	pbar = trange(1, n_epochs+1, leave=True, disable=not verbose, desc='Updating M', miniters=1000)

	def calc_func(M):
		# f = (M @ quad).ravel() @ M.ravel() / 2
		# f -= linear.ravel() @ M.ravel()
		f = torch.addmm(linear, M, quad, alpha=.5, beta=-1).ravel() @ M.ravel() + constant / 2
		return f
	def calc_grad(M):
		g = torch.addmm(linear, M, quad, alpha=1, beta=-1)
		return g
	step_size = 1 / torch.linalg.eigvalsh(quad).max().item()
	optimizer = NesterovGD(M, step_size)
	valid_freq = 100
	M_prev = M.clone()
	func, func_prev, df, dM = np.nan, np.inf, np.nan, np.inf
	for i_epoch in pbar:
		grad = calc_grad(M)
		optimizer.step(grad)
		project_M(M, M_constraint)
		if i_epoch % valid_freq == 0 or i_epoch == n_epochs:
			func = calc_func(M).item()
			df = func_prev - func
			func_prev = func
			dM = M_prev.sub_(M).abs_().max().item()
			M_prev = M.clone()
			do_stop = dM < tol * np.sqrt(valid_freq) and i_epoch > 5
			assert np.isfinite(func)
			pbar.set_description(
				f'Updating M: loss = {func:.1e}, '
				f'%δloss = {df / func:.1e}, '
				f'δM = {dM:.1e}'
			)
			if do_stop:
				break


def estimate_Sigma_x_inv(
		Xs, Sigma_x_inv, Es, use_spatial, lambda_Sigma_x_inv, power_Sigma_x_inv, betas,
		optimizer, context, n_epochs=10000,
		batch_size=-1, valid_freq=50,
):
	if not any(sum(map(len, E)) > 0 and u for E, u in zip(Es, use_spatial)):
		return None, np.nan
	if batch_size == -1:
		batch_size = sum(map(len, Xs))
	use_minibatch = batch_size < sum(map(len, Xs))
	linear_term = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
	nu_list = []
	weight_list = []
	num_edges = 0
	for X, E, u, beta in zip(Xs, Es, use_spatial, betas):
		if not u:
			continue
		X = X.to(**context)
		Z = X / torch.linalg.norm(X, axis=1, ord=1, keepdim=True)
		E = [(i, j) for i, e in enumerate(E) for j in e]
		num_edges += beta * sum(map(len, E))
		E = torch.sparse_coo_tensor(np.array(E).T, np.ones(len(E)), size=[len(Z)]*2, **context)
		nu = E @ Z
		linear_term.addmm_(Z.T, nu, alpha=beta)
		nu_list.append(nu)
		weight_list.append(np.full(len(nu), beta))
	# linear_term = (linear_term + linear_term.T) / 2 # should be unnecessary as long as E is symmetric

	nu_all = torch.cat(nu_list, dim=0)
	weight_all = torch.tensor(np.concatenate(weight_list), **context)
	mini_batcher = Minibatcher((nu_all, weight_all), batch_size=batch_size)

	history = []
	Sigma_x_inv.requires_grad_(True)
	loss_prev, loss = np.inf, np.nan
	pbar = trange(1, n_epochs+1, desc='Updating Σx-1', miniters=valid_freq)
	Sigma_x_inv_best, loss_best, i_epoch_best = None, np.inf, -1
	Sigma_x_inv_prev = Sigma_x_inv.clone().detach_()

	def calc_func(mode):
		loss = Sigma_x_inv.view(-1) @ linear_term.view(-1)
		nu, weight = mini_batcher.sample() if mode == 'train' else (nu_all, weight_all)
		eta = nu @ Sigma_x_inv
		logZ = integrate_of_exponential_over_simplex(eta)
		# loss = loss + beta * len(nu_all) * logZ.mean()
		loss = loss + len(nu_all) / len(nu) * logZ @ weight
		loss = loss / num_edges
		loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.abs().pow(power_Sigma_x_inv).sum()
		return loss

	for i_epoch in pbar:
		optimizer.zero_grad()
		loss = calc_func('train')
		loss.backward()
		Sigma_x_inv.grad = (Sigma_x_inv.grad + Sigma_x_inv.grad.T) / 2
		optimizer.step()
		optimizer.zero_grad()

		with torch.no_grad():
			Sigma_x_inv -= Sigma_x_inv.mean()

		with torch.no_grad():
			if i_epoch % valid_freq == 0:
				if use_minibatch:
					loss = calc_func('eval')
				loss = loss.item()
				dloss = loss_prev - loss
				loss_prev = loss

				history.append({
					'epoch': i_epoch,
					'param': Sigma_x_inv.clone().detach_(),
					'loss': loss * num_edges,
				})
				if history[-1]['loss'] < loss_best:
					Sigma_x_inv_best = history[-1]['param']
					loss_best = history[-1]['loss']
					i_epoch_best = i_epoch
				dSigma_x_inv = Sigma_x_inv_prev.sub_(Sigma_x_inv).abs_().max().item()
				Sigma_x_inv_prev[:] = Sigma_x_inv
				pbar.set_description(
					f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
					f'δΣx-1 = {dSigma_x_inv:.1e} '
					f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
				)
				if dSigma_x_inv < 1e-1 or i_epoch - i_epoch_best >= 2 * valid_freq:
					break

	with torch.no_grad():
		Sigma_x_inv[:] = Sigma_x_inv_best
	Sigma_x_inv.requires_grad_(False)

	history = pd.DataFrame(history)
	return history, -history['loss'].min()


@torch.no_grad()
def estimate_Sigma_x_inv_evolutionary(
		Xs, Sigma_x_inv, Es, use_spatial, lambda_Sigma_x_inv, power_Sigma_x_inv, betas,
		context, n_epochs=10000,
		batch_size=-1, valid_freq=50,
		num_samples=100, ratio_samples=.1, eps_cov=1e-2, gamma_cov=.9,
):
	if not any(sum(map(len, E)) > 0 and u for E, u in zip(Es, use_spatial)):
		return None, np.nan
	if batch_size == -1:
		batch_size = sum(map(len, Xs))
	use_minibatch = batch_size < sum(map(len, Xs))
	linear_term = torch.zeros_like(Sigma_x_inv).requires_grad_(False)
	nu_list = []
	weight_list = []
	num_edges = 0
	for X, E, u, beta in zip(Xs, Es, use_spatial, betas):
		if not u:
			continue
		X = X.to(**context)
		Z = X / torch.linalg.norm(X, axis=1, ord=1, keepdim=True)
		E = [(i, j) for i, e in enumerate(E) for j in e]
		num_edges += beta * sum(map(len, E))
		E = torch.sparse_coo_tensor(np.array(E).T, np.ones(len(E)), size=[len(Z)]*2, **context)
		nu = E @ Z
		linear_term.addmm_(Z.T, nu, alpha=beta)
		nu_list.append(nu)
		weight_list.append(np.full(len(nu), beta))
	# linear_term = (linear_term + linear_term.T) / 2 # should be unnecessary as long as E is symmetric

	nu_all = torch.cat(nu_list, dim=0)
	weight_all = torch.tensor(np.concatenate(weight_list), **context)
	mini_batcher = Minibatcher((nu_all, weight_all), batch_size=batch_size)

	def calc_func(mode, Sigma_x_inv):
		loss = Sigma_x_inv.view(len(Sigma_x_inv), -1) @ linear_term.view(-1)
		nu, weight = mini_batcher.sample() if mode == 'train' else (nu_all, weight_all)
		eta = (nu[None, :, None, :] @ Sigma_x_inv[:, None, :, :]).squeeze(-2)
		logZ = integrate_of_exponential_over_simplex(eta)
		loss = loss + len(nu_all) / len(nu) * logZ @ weight
		loss = loss / num_edges
		loss = loss + lambda_Sigma_x_inv * Sigma_x_inv.abs().pow(power_Sigma_x_inv).sum((-1, -2))
		return loss

	history = []
	# Sigma_x_inv.requires_grad_(True)
	loss_prev, loss = np.inf, np.nan
	pbar = trange(1, n_epochs+1, desc='Updating Σx-1', miniters=valid_freq)
	Sigma_x_inv_best = Sigma_x_inv.clone()
	loss_best = calc_func('eval', Sigma_x_inv[None]).item()
	print(loss_best)
	i_epoch_best = -1
	Sigma_x_inv_prev = Sigma_x_inv.clone()
	mu = Sigma_x_inv.clone().view(-1)
	cov = torch.eye(len(mu), **context) * 1e-2

	for i_epoch in pbar:
		U = torch.linalg.cholesky(cov + torch.eye(len(mu), **context) * eps_cov, upper=True)
		samples = mu + torch.randn(num_samples, len(mu), **context) @ U

		loss = calc_func('train', samples.view(num_samples, *Sigma_x_inv.shape))
		indices = loss.topk(int(num_samples*ratio_samples), largest=False, sorted=False)[1]
		samples = samples[indices]
		loss = loss[indices]
		mu = samples.mean(0)
		cov = cov * gamma_cov + samples.T.cov() * (1-gamma_cov)
		loss, idx = loss.min(0)
		Sigma_x_inv[:] = samples[idx].reshape(*Sigma_x_inv.shape)

		with torch.no_grad():
			mu -= mu.mean()
			Sigma_x_inv -= Sigma_x_inv.mean()

		with torch.no_grad():
			if i_epoch % valid_freq == 0:
				if use_minibatch:
					loss = calc_func('eval', Sigma_x_inv[None])
				loss = loss.item()
				dloss = loss_prev - loss
				loss_prev = loss

				history.append({
					'epoch': i_epoch,
					'param': Sigma_x_inv.clone().detach_().cpu(),
					'loss': loss * num_edges,
				})
				if history[-1]['loss'] < loss_best:
					Sigma_x_inv_best = history[-1]['param']
					loss_best = history[-1]['loss']
					i_epoch_best = i_epoch
				dSigma_x_inv = Sigma_x_inv_prev.sub_(Sigma_x_inv).abs_().max().item()
				Sigma_x_inv_prev[:] = Sigma_x_inv
				pbar.set_description(
					f'Updating Σx-1: loss = {dloss:.1e} -> {loss:.1e} '
					f'δΣx-1 = {dSigma_x_inv:.1e} '
					f'Σx-1 range = {Sigma_x_inv.min().item():.1e} ~ {Sigma_x_inv.max().item():.1e}'
				)
				if dSigma_x_inv < 1e-1 or i_epoch - i_epoch_best >= 5 * valid_freq:
					break

	with torch.no_grad():
		Sigma_x_inv[:] = Sigma_x_inv_best
	Sigma_x_inv.requires_grad_(False)

	history = pd.DataFrame(history)
	return history, -history['loss'].min()
