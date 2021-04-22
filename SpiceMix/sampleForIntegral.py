import sys, subprocess, os, struct, time
from multiprocessing import Pool

import numpy as np
import torch
import scipy
from scipy.stats import truncnorm, multivariate_normal, mvn
from scipy.special import erf, loggamma

from util import PyTorchDType as dtype

n_cache = 2**14
tLogGamma_cache = None
tarange = None
# tLogGamma_cache = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=dtype, device=device)
# tarange = torch.arange(n_cache, dtype=dtype, device=device)


def sampleFromSimplex(n, D, seed=None):
	# should restore seed
	if seed != None: np.random.seed(seed)
	U = np.random.rand(n*D).reshape(n, D)
	E = np.log(U)
	S = np.sum(E, axis=1, keepdims=True)
	x = E / S
	return x


def sampleFromExponentialDistribution(n, D, lambda1, seed=None):
	if seed is not None: np.random.seed(seed)
	x = np.random.exponential(1./lambda1, [n, D])
	return x


def sampleFromHypercube(n, D, l, seed=None):
	if seed is not None: np.random.seed(seed)
	x = np.random.rand(n, D) * l
	return x


def sampleFromHyperball(n, D, l, seed=None):
	if seed is not None: np.random.seed(seed)
	x = np.random.normal(0, 1, [n, D])
	x /= np.linalg.norm(x, axis=1, keepdims=True)
	r = np.random.random(n) ** (1./D) * l
	x *= r[:, None]
	return x


def sampleFromSuperellipsoid(n, D, vec, seed=None):
	"""
	:param vec:		Every column is an eigenvector !!!
	"""
	x = sampleFromHyperball(n, D, 1., seed)
	x = x @ vec.T
	return x


def sampleFromSimplexPyTorch(n, D, seed=None, device='cpu'):
	if seed is not None: torch.manual_seed(seed)
	x = torch.rand([n, D], device=device, dtype=dtype).add_(1e-20)
	x.log_()
	x.div_(x.sum(1, keepdim=True))
	return x


def sampleFromTruncatedHyperballPyTorch(n, D, center, l, seed=None, device='cpu'):
	if seed is not None: torch.manual_seed(seed)
	m = torch.distributions.normal.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))
	m_uniform = torch.distributions.uniform.Uniform(torch.tensor(0., device=device), torch.tensor(1., device=device))
	ret = torch.zeros([n, D], dtype=dtype, device=device)
	zero_dim = (center < 1e-5)[None]
	cnt = 0
	nbatch = 0
	while cnt < n:
		nn = 2**20
		x = m.sample(torch.Size([nn, D]))
		x /= x.norm(dim=1, keepdim=True)
		r = m_uniform.sample(torch.Size([nn]))
		r.pow_(1./D)
		r *= l
		x *= r[:, None]
		x += center[None]
		x[zero_dim.expand_as(x)] = x[zero_dim.expand_as(x)].abs()
		mask = (x >= 0).all(dim=1)
		csmask = mask.cumsum(dim=0)
		num = min(csmask[-1], n - cnt)
		if num:
			mask.masked_fill_(csmask > num, 0)
			torch.masked_select(x, mask[:, None], out=ret[cnt: cnt+num])
		nbatch += 1
		cnt += num

	return ret


def sampleFromTruncatedSuperellipsoidPyTorch(n, D, center, vec, seed=None, device='cpu'):
	if seed is not None: torch.manual_seed(seed)
	# Below is slow !!!
	m = torch.distributions.normal.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device))
	m_uniform = torch.distributions.uniform.Uniform(torch.tensor(0., device=device), torch.tensor(1., device=device))
	ret = torch.zeros([n, D], dtype=dtype, device=device)
	cnt = 0
	nbatch = 0
	while cnt < n:
		x = m.sample(torch.Size([n, D]))
		x /= x.norm(dim=1, keepdim=True)
		r = m_uniform.sample(torch.Size([n]))
		r.pow_(1./D)
		x *= r[:, None]
		x = x @ vec.t()
		x += center[None]
		mask = (x >= 0).all(dim=1)
		csmask = mask.cumsum(dim=0)
		num = min(csmask[-1], n - cnt)
		if num:
			mask.masked_fill_(csmask > num, 0)
			torch.masked_select(x, mask[:, None], out=ret[cnt: cnt+num])
		nbatch += 1
		# if num > 0: print(nbatch, cnt)
		cnt += num

	# print(nbatch)
	return ret


def estimateMomentsForTruncatedMultivariateGaussianPyTorch(K, cov, means, func_args=None, device='cpu'):
	# https://doi.org/10.1080/10618600.2017.1322092
	assert K >= 3
	assert cov.shape == (K, K)
	assert means.shape[1] == K
	NN = means.shape[0]

	def estimatePartitionFunction(K, mean, cov):
		# https://github.com/scipy/scipy/blob/v1.2.1/scipy/stats/_multivariate.py
		# line 525, def _cdf
		assert cov.shape[1:] == (K, K)
		M = len(cov)
		assert mean.shape[1:] == (M, K)
		N = len(mean)

		arr = np.empty([N, M], dtype=np.float)
		for a, m, c in zip(arr.T, mean.transpose(1, 0, 2), cov):
			a[:] = [
				mvn.mvnun(
					lower=-x, upper=np.full(K, np.inf), means=np.zeros(K), covar=c,
					maxpts=1000000 * K,
					abseps=1e-6,
					releps=1e-6,
				)[0]
				for x in m
			]
		# [mvn.mvnun(lower=-m, upper=np.full(K, np.inf), means=np.zeros(K), covar=cov)[0] for m in mean]
		# [mvn.mvnun(lower=np.zeros(K), upper=np.full(K, np.inf), means=m, covar=cov)[0] for m in mean]
		# [mvn.mvnun(lower=np.full(K, -np.inf), upper=m, means=np.zeros(K), covar=cov)[0] for m in mean]
		# multivariate_normal(mean=None, cov=cov).cdf(mean)
		# multivariate_normal(mean=np.zeros(K), cov=cov).cdf(mean)
		# [multivariate_normal(mean=-m, cov=cov).cdf(np.zeros(K)) for m in mean]

		arr = torch.tensor(arr, dtype=dtype, device=device)
		return arr

	# covariance matrix is shared
	cov0 = cov
	cov1 = np.empty([K, K-1, K-1], dtype=np.float)
	cov2 = np.empty([int(K*(K-1)/2), K-2, K-2], dtype=np.float)
	idx = np.ones(K, dtype=np.bool)
	cov2_iter = iter(cov2)
	for i in range(K):
		idx[i] = False
		cov1[i] = cov[idx][:, idx] - np.outer(cov[i, idx], cov[i, idx]) / cov[i, i]
		assert np.abs(cov1[i] - cov1[i].T).max() < 1e-10
		for j in range(i+1, K):
			idx[j] = False
			next(cov2_iter)[:] = cov[idx][:, idx] - cov[idx][:, [i, j]] @ np.linalg.inv(cov[[i, j]][:, [i, j]]) @ cov[[i, j]][:, idx]
			# sanity check
			a = cov[idx][:, idx] - cov[idx][:, [i, j]] @ np.linalg.inv(cov[[i, j]][:, [i, j]]) @ cov[[i, j]][:, idx]
			assert np.abs(a - a.T).max() < 1e-10
			idx[j] = True
			# sanity check
			index = sorted(list(set(range(K)) - {j}))
			cov_j = cov[index][:, index] - np.outer(cov[j, index], cov[j, index]) / cov[j, j]
			index = sorted(list(set(range(K-1)) - {i}))
			cov_ij = cov_j[index][:, index] - np.outer(cov_j[i, index], cov_j[i, index]) / cov_j[i, i]
			# print(np.abs(cov_ij - a).max())
			assert np.abs(cov_ij - a).max() < 1e-10
		idx[i] = True
		assert idx.all()
	del idx
	assert next(cov2_iter, None) is None
	del cov2_iter
	tcov, tcov1, tcov2 = [torch.tensor(_, dtype=dtype, device=device) for _ in [cov, cov1, cov2]]
	tcov0 = tcov
	# tcov_2, tcov1_2, tcov2_2 = [torch.cholesky(_, upper=True) for _ in [tcov, tcov1, tcov2]]
	# tcov0_2 = tcov_2

	if func_args is None:
		moment0 = np.empty([NN], dtype=np.float)
		moment1 = np.empty([NN, K], dtype=np.float)
		moment2 = np.empty([NN, K, K], dtype=np.float)
	row_idx, col_idx = np.triu_indices(K, 1)

	batch_size = int(2**30 / (K*(K-1)*(K-2)/2))
	for idx_start in range(0, NN, batch_size):
		mean = means[idx_start: idx_start + batch_size]
		N = len(mean)

		mean0 = mean
		mean1 = mean[:, None, :] - cov * (mean / cov.flatten()[::K+1])[:, :, None]
		mean1 = mean1.reshape(N, -1)[:, :-1].reshape(N, K-1, K+1)[:, :, 1:].reshape(N, K, K-1)
		mean1_ = np.empty([N, K, K-1], dtype=np.float)
		mean2 = np.empty([int(K*(K-1)/2), N, K-2], dtype=np.float)
		mean2_iter = iter(mean2)
		idx = np.ones(K, dtype=np.bool)
		for i in range(K):
			idx[i] = False
			mean1_[:, i, :] = mean[:, idx] - cov[None, idx, i] * (mean[:, i] / cov[i, i])[:, None]
			for j in range(i+1, K):
				idx[j] = False
				next(mean2_iter, None)[:] = mean[:, idx] - ((cov[idx][..., [i, j]] @ np.linalg.inv(cov[[i, j]][:, [i, j]]))[None] @ mean[:, [i, j], None]).squeeze(-1)
				a = mean[:, idx] - ((cov[idx][..., [i, j]] @ np.linalg.inv(cov[[i, j]][..., [i, j]]))[None] @ mean[:, [i, j], None]).squeeze(-1)
				idx[j] = True
				# sanity check
				index = sorted(list(set(range(K)) - {j}))
				mean_j = mean[:, index] - cov[None, index, j] * (mean[:, j] / cov[j, j])[:, None]
				cov_j = cov[index][:, index] - np.outer(cov[j, index], cov[j, index]) / cov[j, j]
				index = sorted(list(set(range(K-1)) - {i}))
				mean_ij = mean_j[:, index] - cov_j[None, index, i] * (mean_j[:, i] / cov_j[i, i])[:, None]
				# print(np.abs(mean_ij - a).max())
				assert np.abs(mean_ij - a).max() < 1e-10
				# exit()

			idx[i] = True
			assert idx.all()
		del idx
		assert next(mean2_iter, None) is None
		del mean2_iter
		# print(np.abs(mean1 - mean1_).max())
		assert np.abs(mean1 - mean1_).max() < 1e-10
		mean2 = mean2.transpose(1, 0, 2)

		# using PyTorch
		# tmean2 = torch.tensor(mean2, dtype=dtype, device=device)
		# del mean2
		# t2_ = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K-2, tmean2, tcov2_2)	# (N, K*(K-1)/2)
		# del tmean2
		# using builtin
		# I_0^{n-2}		upper triangular part						shape: (N, K*(K-1)/2)
		t2_ = estimatePartitionFunction(K-2, mean2, cov2)
		del mean2
		# I_0^{n-2}		square matrix, zero diagonal				shape: (N, K, K-1)
		t2 = torch.zeros([N, K, K-1], dtype=dtype, device=device)
		# I_0^{n-2}		fill upper triangular part					shape: (N, K, K-1)
		t2[:, row_idx, col_idx-1] = t2_
		# I_0^{n-2}		fill lower triangular part					shape: (N, K, K-1)
		t2[:, col_idx, row_idx] = t2_
		del t2_
		# mu(-i) _j _ {row: i, column: j!=i}						shape: (N, K, K-1)
		tmean1 = torch.tensor(mean1, dtype=dtype, device=device)
		# t1 = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K-1, tmean1, tcov1_2)	# I_0^{n-1} (N, K)
		# Sigma(-i) _jj _ {row: i, column: j!=i}					shape: (1, K, K-1)
		tc = tcov1.view(K, -1)[None, :, ::K]	# (K, K-1, K-1) -> (K, (K-1)**2) -> (K, K-1) diagonal
		# phi_1( -i j ) _ {row: i, column: j!=i}					shape: (N, K, K-1)
		t = tmean1.pow(2).div_(tc).div_(-2).exp_().div_(tc.mul(2*np.pi).sqrt_())
		del tc
		# d_0^{n-1}		fill off-diagonal part						shape: (N, K, K-1)
		t2.mul_(t)
		del t
		# [ I^{n-1}_ej -i ] _ {row: i, column: j}	the second part	shape: (N, K, K-1)
		t2 = (t2[:, :, None, :] @ tcov1[None]).squeeze(-2)
		# I_0^{n-1} 												shape: (N, K)
		t1 = estimatePartitionFunction(K-1, mean1, cov1)
		# [ I^{n-1}_ej -i ] _ {row: i, column: j}	the first part	shape: (N, K, K-1)
		t2.addcmul_(t1[:, :, None], tmean1)
		# mu														shape: (N, K)
		tmean0 = torch.tensor(mean, dtype=dtype, device=device)
		# I_0^n														shape: (N)
		# t0 = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean0[:, None, :], tcov0_2[None]).squeeze(1)	# I_0^n (N, )
		t0 = estimatePartitionFunction(K, mean0[:, None, :], cov0[None]).squeeze(1)
		# Sigma_ii													shape: (1, K)
		tc = tcov0.view(-1)[None, ::K+1]
		# phi_1( i )												shape: (N, K)
		t = tmean0.pow(2).div_(tc).div_(-2).exp_().div_(tc.mul(2*np.pi).sqrt_())
		del tc
		# d_0^n														shape: (N, K)
		t1.mul_(t)
		# [ I^n_ei ]	the second part								shape: (N, K)
		t1 = t1 @ tcov
		# [ I^n_ei ]	the first part								shape: (N, K)
		t1.addcmul_(t0[:, None], tmean0)
		# [ d^n_ei j} ]												shape: (N, K, K)
		t2_ = torch.zeros([N, K, K], dtype=dtype, device=device)
		t2_[:, row_idx, col_idx] = t2[:, col_idx, row_idx]
		t2_[:, col_idx, row_idx] = t2[:, row_idx, col_idx-1]
		t2 = t2_
		del t2_
		# t2 = t2.transpose(1, 2).contiguous()
		# [ d^n_ei j} ]	off-diagonal part							shape: (N, K, K)
		t2.mul_(t[:, None, :])
		# [ d^n_ei j ] _ {row: i, column j}		diagonal part		shape: (N, K, K)
		t2.view(N, -1)[:, ::K+1].copy_(t0[:, None])
		# [ I^n_{ei+ek} ]	the second part							shape: (N, K, K)
		t2 = t2 @ tcov[None]
		# [ I^n_{ei+ek} ]	the first part							shape: (N, K, K)
		t2.addcmul_(t1[:, :, None], tmean0[:, None, :])
		# t2.addcmul_(tmean[:, :, None], t1[:, None, :])

		t1.div_(t0[..., None])
		t2.div_(t0[..., None, None])

		if func_args is None:
			moment0[idx_start: idx_start + batch_size] = t0.cpu().data.numpy()
			moment1[idx_start: idx_start + batch_size] = t1.cpu().data.numpy()
			moment2[idx_start: idx_start + batch_size] = t2.cpu().data.numpy()
		else:
			func, args = func_args
			args = func(
				slice(idx_start, idx_start+batch_size),
				(t0, t1, t2),
				*args,
			)
			func_args = (func, args)

	if func_args is None:
		return moment0, moment1, moment2	# not normalized
	else:
		func, args = func_args
		return args


def integrateOfExponentialOverSimplexRecurrence(teta, grad=1., requires_grad=False, device='cpu'):
	N, D = teta.shape
	# tarr[:, J:].sub_( t.exp().mul_(tarr[:, [J-1]]) ).div_(t.neg_())

	if requires_grad:
		tarr = torch.ones([N, D, D], dtype=dtype, device=device)

		for J in range(1, D):
			t = teta[:, J:] - teta[:, [J-1]]

			tt = t.exp().mul(tarr[:, J-1, [J-1]])
			tarr[:, J, J:] = tarr[:, J-1, J:].sub(tt).div(t.neg())
			del t, tt

		tret = tarr[:, -1, -1].mul(teta[:, -1].neg().exp())
		tret = tret.log()
	else:
		tarr = torch.ones([N, D, D], dtype=dtype, device=device)

		for J in range(1, D):
			t = teta[:, J:] - teta[:, [J-1]]

			tt = t.exp().mul_(tarr[:, J-1, [J-1]])
			t.neg_()
			torch.sub(tarr[:, J-1, J:], tt, out=tarr[:, J, J:]).div_(t)
			del t

		tret = tarr[:, -1, -1].mul(teta[:, -1].neg().exp_())

		# to be customized
		# grad * log Z
		tarr[:, 1, 0] = grad / tret	* teta[:, -1].neg().exp_()
		teta.grad[:, -1] = - grad
		# grad * Z
		# tarr[:, 1, 0] = grad * teta[:, -1].neg().exp_()
		# teta.grad[:, -1].sub_(grad * tret)

		tret = tret.log_()

		for L in range(1, D-1):
			tarr[:, L+1:, L-1] = teta[:, 1:-L].sub(teta[:, [-L]]).flip(-1).pow_(-1).cumprod(-1).mul_(tarr[:, [L], L-1])
			t = teta[:, -L:].sub(teta[:, [-L-1]])
			t = t.exp().div_(t)
			tarr[:, L+1, L] = t.mul_(tarr[:, L, :L].flip(-1)).sum(-1)
			del t
		for J in range(1, D):
			tarr[:, -J, :-J] = tarr[:, -J, :-J].flip(-1)

		for J in range(1, D):
			t = teta[:, J:] - teta[:, [J-1]]
			tt = t.exp().mul_(tarr[:, J-1, [J-1]])
			tt.sub_(tarr[:, J, J:]).div_(t).mul_(tarr[:, -J, :-J])
			del t
			teta.grad[:, J-1].sub_(tt.sum())
			teta.grad[:, J:].add_(tt)
			del tt

	return tret


def integrateOfExponentialOverSimplexInduction(teta, grad=None, requires_grad=False, device='cpu'):
	if grad is None: grad = torch.tensor([1.], dtype=dtype, device=device)
	N, D = teta.shape
	# tarr[:, J:].sub_( t.exp().mul_(tarr[:, [J-1]]) ).div_(t.neg_())

	# eps = 1e-5
	# eps0 = 1e-30
	eps = 0
	eps0 = 0

	nterm = 256
	chunk_size = 256

	if requires_grad:
		"""
		A = torch.empty([N, D], dtype=dtype, device=device)
		signs = torch.empty_like(A)
		for k in range(D):
			t = teta - teta[:, [k]]
			t[:, k] = 1
			tsign = t.sign()
			signs[:, k] = tsign.prod(-1)
			t = t.abs().log()
			t[:, k] = teta[:, k]
			A[:, k] = t.sum(-1).neg()
		toffset = A.max(-1, keepdim=True)[0]
		tret = A.sub(toffset).exp().mul(signs).sum(-1).log().add(toffset.squeeze(-1))
		# """

		tetas = teta
		trets = []
		tidx = tarange[D-1: D+nterm-1]
		tlg = tLogGamma_cache[D-1: D+nterm-1]
		for teta in tetas.split(chunk_size, 0):
		# tret = trets
		# teta = tetas
			A = torch.empty([len(teta), D], dtype=dtype, device=device)
			Asign = torch.empty_like(A)
			for k in range(D):
				t = teta - teta[:, [k]]
				t[:, k] = 1
				tsign = t.sign()
				Asign[:, k] = tsign.prod(-1)
				t = t.abs().log()
				A[:, k] = t.sum(-1).neg()

			tpoly = teta.abs().log()[:, :, None].mul(tidx[None, None, :]).sub(tlg[None, None, :])
			toffset = tpoly.max(-1, keepdim=True)[0]
			tpoly = tpoly.sub(toffset).exp().mul(teta.sign().neg()[:, :, None].pow(tidx[None, None, :]))
			tpoly_sum0 = tpoly.sum(-1)
			tpoly_sum0_sign = tpoly_sum0.sign()
			tpoly_sum0 = tpoly_sum0.abs().log().add(toffset.squeeze(-1))
			del toffset, tpoly

			Aexp_sign = Asign.mul(tpoly_sum0_sign)
			Aexp = A.add(tpoly_sum0)
			toffset = Aexp.max(-1, keepdim=True)[0]
			tret = Aexp.sub(toffset).exp().mul(Aexp_sign).sum(-1).log().add(toffset.squeeze(-1))
			del toffset, Aexp, Aexp_sign

			trets.append(tret)

		tret = torch.cat(trets)
	else:
		"""
		A = torch.empty([N, D], dtype=dtype, device=device)
		Asign = torch.empty_like(A)
		for k in range(D):
			t = teta - teta[:, [k]]
			t[:, k].fill_(1)
			tsign = t.sign()
			Asign[:, k] = tsign.prod(-1)
			t.abs_().log_()
			A[:, k].copy_(t.sum(-1).neg_())
		# the first (D-1) terms in the Taylar expansion cancel out
		# so this is numerically unstable
		Aexp = A.sub(teta)
		toffset = Aexp.max(-1, keepdim=True)[0]
		tret = Aexp.sub_(toffset).exp_().mul_(Asign).sum(-1).log_().add_(toffset.squeeze(-1))
		del toffset
		del Aexp

		grad = grad.log().sub_(tret)
		for i in range(D):
			texp_offset = torch.min(teta, dim=-1, keepdim=True)[0]
			texp = teta.neg().add_(texp_offset).exp_()
			texp.sub_(texp[:, [i]])
			texp[:, i].fill_(1)
			texp.abs_().log_().sub_(texp_offset)
			texp[:, i].copy_(teta[:, i]).neg_()

			td = teta[:, [i]].sub(teta).abs_()
			td[:, i].fill_(1)
			td.log_()
			texp.add_(A).sub_(td)
			del td
			texp.exp_().mul_(Asign)
			teta.grad[:, i].sub_(texp.sum(-1).mul_(grad.exp()))
		# """

		rank = teta.argsort().to(dtype).sub_(.5*D)
		teta.add_(1e-4, rank)
		# print(teta)
		tetas = teta
		trets = torch.empty(len(teta), dtype=dtype, device=device)
		tidx = tarange[D-1:D+nterm-1]
		tlg = tLogGamma_cache[D-1: D+nterm-1]
		for tret, teta, teta_grad in zip(trets.split(chunk_size, 0), tetas.split(chunk_size, 0), tetas.grad.split(chunk_size, 0)):
			A = torch.empty([len(teta), D], dtype=dtype, device=device)
			Asign = torch.empty_like(A)
			for k in range(D):
				t = teta - teta[:, [k]]
				t.add_(eps0)
				t[:, k].fill_(1)
				tsign = t.sign()
				Asign[:, k] = tsign.prod(-1)
				t.abs_().add_(eps).log_()
				A[:, k].copy_(t.sum(-1).neg_())

			tpoly = teta.abs().add_(eps).log_()[:, :, None].mul(tidx[None, None, :]).sub_(tlg[None, None, :])
			tpoly_sign = teta.add_(eps0).sign().neg_()[:, :, None].pow(tidx[None, None, :])
			toffset = tpoly.max(-1, keepdim=True)[0]
			# t = tpoly.sub_(toffset).exp_().mul_(teta.sign().neg_()[:, :, None].pow(tidx[None, None, :]))
			t = tpoly.sub(toffset).exp_().mul_(tpoly_sign)
			tpoly_sum1 = t[:, :, 1:].sum(-1)
			tpoly_sum0 = t[:, :, 0] + tpoly_sum1
			tpoly_sum1_sign = tpoly_sum1.sign()
			tpoly_sum0_sign = tpoly_sum0.sign()
			tpoly_sum1.abs_().add_(eps).log_().add_(toffset.squeeze(-1))
			tpoly_sum0.abs_().add_(eps).log_().add_(toffset.squeeze(-1))
			del toffset

			Aexp_sign = Asign.mul(tpoly_sum0_sign)
			Aexp = A.add(tpoly_sum0)
			toffset = Aexp.max(-1, keepdim=True)[0]
			tret.copy_(Aexp.sub_(toffset).exp_().mul_(Aexp_sign).sum(-1))
			assert (tret > 0).all()
			tret.log_().add_(toffset.squeeze(-1))
			del toffset, Aexp, Aexp_sign

			tgrad = grad.log().sub(tret)
			for i in range(D):
				t = teta[:, [i]] - teta
				t.add_(eps0)
				t[:, i].fill_(1)
				tsign = t.sign()
				t.abs_().add_(eps).log_().neg_()

				t.add_(A)
				tsign.mul_(Asign)

				#
				toffset = torch.max(tpoly_sum1[:, [i]], tpoly_sum1)
				tt = tpoly_sum1.sub(toffset).exp_().mul_(tpoly_sum1_sign)
				tt.sub_(tpoly_sum1[:, [i]].sub(toffset).exp_().mul_(tpoly_sum1_sign[:, [i]]))
				#
				# toffset = torch.max(torch.max(tpoly[:, :, 1:], dim=-1, keepdim=True)[0], torch.max(tpoly[:, [i], 1:], dim=-1, keepdim=True)[0])
				# tt = tpoly[:, :, 1:].sub(toffset).exp_().mul_(tpoly_sign[:, :, 1:])
				# tt.sub_(tpoly[:, [i], 1:].sub(toffset).exp_().mul_(tpoly_sign[:, [i], 1:]))
				# tt = tt.sum(-1)
				#
				tt[:, i].fill_(1)
				tsign.mul_(tt.sign())
				tt.abs_().log_().add_(toffset.squeeze(-1))
				del toffset
				tt[:, i].copy_(tpoly_sum0[:, i])
				tsign[:, i].mul_(tpoly_sum0_sign[:, i])
				t.add_(tt)

				teta_grad[:, i].sub_(t.exp_().mul_(tsign).sum(-1).mul_(tgrad.exp()))

		tret = trets

	return tret


def integrateOfExponentialOverSimplexInduction2(teta, grad=None, requires_grad=False, device='cpu'):
	if grad is None: grad = torch.tensor([1.], dtype=dtype, device=device)
	N, D = teta.shape

	global tLogGamma_cache, tarange
	if tLogGamma_cache is None:
		tLogGamma_cache = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=dtype, device=device)
	if tarange is None:
		tarange = torch.arange(n_cache, dtype=dtype, device=device)

	t_eta_offset = teta.max(-1, keepdim=True)[0] + 1e-5
	# nterm = 256
	nterm = (teta.max() - teta.min()).item()
	nterm = max(nterm+10, nterm*1.1)
	nterm = int(nterm)
	if D+nterm-1 > len(tLogGamma_cache):
		raise ValueError('Please increase the value of n_cache')
	tlg = tLogGamma_cache[D-1: D+nterm-1]

	if requires_grad:
		teta = teta - t_eta_offset
		teta = teta.neg()
		# teta = teta.sort()[0]

		f = torch.zeros([N, D], dtype=dtype, device=device)
		tret = torch.zeros([nterm, N], dtype=dtype, device=device)

		for m in range(1, nterm):
			f = f + teta.log()
			toffset = f.max(-1, keepdim=True)[0]
			f = f.sub(toffset).exp().cumsum(dim=-1).log().add(toffset)
			tret[m].copy_(f[:, -1])

		tret = tret.sub(tlg[:, None])
		toffset = tret.max(0, keepdim=True)[0]
		tret = tret.sub(toffset).exp().sum(0).log().add(toffset.squeeze(0)).sub(t_eta_offset.squeeze(-1))
	else:
		teta.sub_(t_eta_offset).neg_()
		tetas = teta
		trets = torch.empty(N, dtype=dtype, device=device)
		chunk_size = 32
		for teta, teta_grad, tretc in zip(tetas.split(chunk_size, 0), teta.grad.split(chunk_size, 0), trets.split(chunk_size, 0)):
			N = len(teta)
			teta_log = teta.log()
			tret = torch.zeros([nterm, N], dtype=dtype, device=device)
			tgrad = torch.full([nterm, N, D], -np.inf, dtype=dtype, device=device)
			f = torch.zeros([N, D], dtype=dtype, device=device)
			g = torch.full([N, D, D], -np.inf, dtype=dtype, device=device)
			for m in range(1, nterm):
				g.add_(teta_log[:, None, :])
				gd = g.view(N, D**2)[:, ::D+1]
				toffset = torch.max(gd, f)
				toffset[toffset == -np.inf] = 0
				assert (toffset != -np.inf).all()
				gd.sub_(toffset).exp_().add_(f.sub(toffset).exp_()).log_().add_(toffset)
				toffset = g.max(-1, keepdim=True)[0]
				assert (toffset != -np.inf).all()
				g = g.sub_(toffset).exp_().cumsum(dim=-1).log_().add_(toffset)
				tgrad[m].copy_(g[:, :, -1])

				f.add_(teta_log)
				toffset = f.max(-1, keepdim=True)[0]
				f = f.sub_(toffset).exp_().cumsum(dim=-1).log_().add_(toffset)
				tret[m].copy_(f[:, -1])

			tret.sub_(tlg[:, None])
			toffset = tret.max(0, keepdim=True)[0]
			tretc.copy_(tret.sub_(toffset).exp_().sum(0)).log_().add_(toffset.squeeze(0))

			tgrad.sub_(tlg[:, None, None])
			teta_grad.sub_(tgrad.sub_(tretc[None, :, None]).exp_().sum(0).mul_(grad))

		trets.sub_(t_eta_offset.squeeze(-1))
		tret = trets
		tetas.neg_().add_(t_eta_offset)

	return tret


def integrateOfExponentialOverSimplexSampling(teta, grad=None, requires_grad=False, seed=None, device='cpu'):
	# if grad is None: grad = torch.tensor([1.], dtype=dtype, device=device)
	N, D = teta.shape

	# teta_offset = teta.min(1, keepdim=True)[0]

	loggamma_D = loggamma(D)
	n = 2**8
	n = int(n)
	nround = 1

	if requires_grad:
		# teta = teta - teta_offset
		tlogZ = torch.zeros(N, dtype=dtype, device=device)
		for _ in range(nround):
			if seed is not None: seed = seed*nround+_
			t = teta.neg().matmul(sampleFromSimplexPyTorch(n, D, seed=seed).t())
			t_offset = t.max(1, keepdim=True)[0]
			t = t.sub(t_offset).exp().mean(-1).log().add(t_offset.squeeze(1))
			tlogZ = tlogZ + t
			# tlogZ = tlogZ + teta.neg().matmul(sampleFromSimplexPyTorch(n, D, seed=seed).t()).exp().mean(-1).log()
		tlogZ = tlogZ / nround
		tlogZ = tlogZ - loggamma_D
		# tlogZ = tlogZ - teta_offset.squeeze(1)
	else:
		tlogZ = torch.zeros(N, dtype=dtype, device=device)
		tgrad = torch.zeros_like(teta)

		# teta.sub_(teta_offset)

		for _ in range(nround):
			if seed is not None: seed = seed*nround+_
			tsample = sampleFromSimplexPyTorch(n, D, seed=seed)
			texp = teta.neg().matmul(tsample.t())
			t_offset = texp.max(1, keepdim=True)[0]
			texp.sub_(t_offset).exp_()
			tZ = texp.mean(-1)
			chunk_size = 64
			for tgradc, texpc, tZc in zip(tgrad.split(chunk_size, 0), texp.split(chunk_size, 0), tZ.split(chunk_size, 0)):
				tgradc.add_(1/n, texpc[:, None, :].matmul(tsample[None, :, :]).squeeze(1).div_(tZc[:, None]))
			tlogZ.add_(tZ.log_().add_(t_offset.squeeze(1)))

		# teta.add_(teta_offset)

		tlogZ.div_(nround).sub_(loggamma_D)
		tgrad.div_(nround).neg_()
		# tlogZ.sub_(teta_offset.squeeze(1))
		if grad is not None:
			teta.grad.addcmul_(grad, tgrad)
		else:
			teta.grad.add_(tgrad)

	return tlogZ
