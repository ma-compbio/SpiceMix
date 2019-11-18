from sampleForIntegral import *
import scipy.optimize
from util import zipTensors, unzipTensors, psutil_process, PyTorchDevice as device, PyTorchDType as dtype
import sys, timeit, itertools, psutil, resource
from multiprocessing import Pool
import gurobipy as grb
import torch
import numpy as np
import scipy
import scipy.spatial
import scipy.stats
import scipy.spatial
import sklearn.covariance

def E_step_expr(O, H, Theta, modelSpec, nsample4integral, iiter, dropout_str, **kwargs):
	raise NotImplementedError
	# If the samples for the integral are shared, pass the samples as another parameter
	# time_start = timeit.default_timer()
	assert dropout_str == 'origin'

	K, YT, YT_valid, E, E_empty, X_sum2one, M_sum2one, pairwise_potential_str, sigma_yx_inv_str = O
	(XT,) = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta
	N, G = YT.shape
	assert M.shape == (G, K)

	assert not X_sum2one
	assert M_sum2one

	# calculate Nu
	Nu = np.zeros([N, K], dtype=np.float)
	if not E_empty:
		for nu, e in zip(Nu, E): nu[:] = XT[e].sum(0)

	if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
		mu_x, sigma_x_inv = prior_x[1:]

		# calculate covariance of X
		prec = (M.T @ M) * sigma_yx_inv**2
		prec[np.diag_indices(K)] += sigma_x_inv**2

		# calculate mean of X
		mean = YT @ M * sigma_yx_inv**2 + mu_x * sigma_x_inv**2
		if not E_empty: mean -= (Nu - np.outer(np.array(list(map(len, E))), delta_x)) @ Sigma_x_inv

		del mu_x, sigma_x_inv
	elif prior_x[0] == 'Exponential':
		lambda_x, = prior_x[1:]

		# calculate covariance of X
		prec = (M.T @ M) * sigma_yx_inv ** 2

		# calculate mean of X
		mean = YT @ M * sigma_yx_inv ** 2 - lambda_x
		if not E_empty: mean -= (Nu - np.outer(np.array(list(map(len, E))), delta_x)) @ Sigma_x_inv
	else:
		assert False

	cov = np.linalg.inv(prec)
	mean = mean @ cov
	w, v = np.linalg.eigh(cov)
	for i, (ww, vv) in enumerate(zip(w, v.T)):
		print('eig {}: {:.2e}\t{}'.format(i, ww, np.array2string(vv, formatter={'all': '{:.2f}'.format}, max_line_width=10000)))

	kk = 20
	# print(f'prec =\n{prec}')
	# print(f'cov =\n{cov}')
	# print(f'YT @ M * sigma_yx_inv**2 =\n{YT[::kk] @ M * sigma_yx_inv**2}')
	# print(f'mean =\n{mean[::kk]}')

	"""
	talpha = torch.zeros([K], dtype=dtype, device=device)
	tA = torch.zeros([G, K], dtype=dtype, device=device)
	tB = torch.zeros([K, K], dtype=dtype, device=device)
	tC = torch.zeros([K, K], dtype=dtype, device=device)
	def func(idx, tms, taggregate, const):
		tm0, tm1, tm2 = tms
		talpha, tA, tB, tC, nSmallZ = taggregate
		Y, NvT = const
		tY = torch.tensor(Y[:, idx], dtype=dtype, device=device)
		tNuT = torch.tensor(NvT[:, idx], dtype=dtype, device=device)

		talpha.add_(tm1.sum(0))
		tA.addmm_(tY, tm1)
		tB.add_(tm2.sum(0))
		tC.addmm_(tNuT, tm1)
		nSmallZ += (tm0 < 1e-3).sum().item()

		return (talpha, tA, tB, tC, nSmallZ), (Y, Nu)
	(talpha, tA, tB, tC, nSmallZ), (Y, Nu) = estimateMomentsForTruncatedMultivariateGaussianPyTorch(
		K, cov, mean,
		func_args=(
			func, (
				(talpha, tA, tB, tC, 0),
				(YT.T, Nu.T),
			)
		),
	)
	print(f'nSmallZ = {nSmallZ}')
	assert nSmallZ == 0
	return talpha, tA.cpu().data.numpy(), tB.cpu().data.numpy(), tC, torch.tensor(Nu, dtype=dtype, device=device)
	# """

	# """
	cov_U = scipy.linalg.cholesky(cov, lower=False)
	prec_U = scipy.linalg.cholesky(prec, lower=False)
	ndrawn, m1, v1, m2, v2, entropy, ventropy = estimateMomentsForTruncatedMultivariateGaussianCPPBatch(
		nsample4integral, K,
		means=mean, cov=cov, cov_U=cov_U, prec=prec, prec_U=prec_U,
		init_xs=np.maximum(0, -mean+1e-10) @ np.linalg.inv(cov_U),
		seeds=np.arange(N)+iiter*N,
	)

	talpha = torch.tensor(m1.sum(0), dtype=dtype, device=device)
	talpha_e = torch.tensor(np.array(list(map(len, E))) @ m1, dtype=dtype, device=device)
	A = YT.T @ m1
	B = m2.sum(0)
	tC = torch.tensor(Nu.T @ m1, dtype=dtype, device=device)
	entropy = entropy.sum()
	# tC = torch.tensor(Nu.T @ np.full_like(m1, 1/lambda_x), dtype=dtype, device=device)
	return talpha, talpha_e, A, B, tC, torch.tensor(Nu, dtype=dtype, device=device), entropy
	# """

def E_step(O, H, Theta, modelSpec, nsample4integral, iiter, **kwargs):
	raise NotImplementedError
	time_start = timeit.default_timer()
	print('E-step begins')

	K, YTs, YT_valids, Es, Es_empty, betas, X_sum2one, M_sum2one, pairwise_potential_str, sigma_yx_inv_str = O
	(XTs,) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta

	talphas = []
	talpha_es = []
	As = []
	Bs = []
	tC = torch.zeros([K, K], dtype=dtype, device=device)
	tnus = []
	Q_entropy = 0
	for YT, YT_valid, E, E_empty, beta, XT, sigma_yx_inv, prior_x in zip(YTs, YT_valids, Es, Es_empty, betas, XTs, sigma_yx_invs, prior_xs):
		N, G = YT.shape
		talpha, talpha_e, A, B, tC_, tnu, Q_entropy_ = E_step_expr(
			(K, YT, E, E_empty, X_sum2one, M_sum2one),
			(XT, ),
			(M[:G], sigma_yx_inv, Sigma_x_inv, delta_x, prior_x),
			modelSpec,
			nsample4integral=nsample4integral,
			iiter=iiter,
			**modelSpec,
		)
		talphas.append(talpha)
		talpha_es.append(talpha_e)
		tnus.append(tnu)
		As.append(A)
		Bs.append(B)
		tC.add_(beta, tC_)
		Q_entropy += beta*Q_entropy_
		del talpha, talpha_e, tnu, A, B, tC_

	# tC.div_(2)

	pars = (As, Bs, tC, talphas, talpha_es, tnus)
	print('E-step ends in {}'.format(timeit.default_timer() - time_start))
	sys.stdout.flush()

	return pars, Q_entropy

def genPars(O, H, Theta, modelSpec, pairwise_potential_str, dropout_str, **kwargs):
	K, YTs, YT_valids, Es, Es_empty, betas = O
	(XTs,) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta

	assert pairwise_potential_str == 'normalized'

	talphas = []
	talpha_es = []
	As = []
	Bs = []
	tC = torch.zeros([K, K], dtype=dtype, device=device)
	tnus = []
	for YT, YT_valid, E, E_empty, XT, beta in zip(YTs, YT_valids, Es, Es_empty, XTs, betas):
		tXT = torch.tensor(XT, dtype=dtype, device=device)
		N, G = YT.shape
		talphas.append(tXT.sum(0))
		talpha_es.append(torch.tensor(list(map(len, E)), dtype=dtype, device=device) @ tXT)
		if dropout_str == 'origin':
			As.append(YT.T @ XT)
			Bs.append(XT.T @ XT)
		elif dropout_str == 'pass':
			A = np.empty([YT.shape[1], K], dtype=np.float)
			for y, y_valid, a in zip(YT.T, YT_valid.T, A):
				a[:] = y[y_valid] @ XT[y_valid]
			Bs.append(np.full([K, K], np.nan))
		else: assert False
		tXT.div_(tXT.sum(1, keepdim=True).add_(1e-30))
		tnu = torch.empty([N, K], dtype=dtype, device=device)
		for tnui, ei in zip(tnu, E):
			tnui.copy_(tXT[ei].sum(0))
		tnus.append(tnu)
		tC.add_(beta, tXT.t() @ tnu)
	pars = (As, Bs, tC, talphas, talpha_es, tnus)
	return pars

def M_step_Y(O, H, Theta, modelSpec, pars, M_sum2one, sigma_yx_inv_str, dropout_str, **kwargs):
	print('Optimizing M, σ_yx_inv ...', end='\n')
	sys.stdout.flush()
	time_start = timeit.default_timer()

	K, YTs, YT_valids, Es, Es_empty, betas = O
	(XTs,) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta
	As, Bs, tC, talphas, talpha_es, tnus = pars

	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)
	sizes = np.array([YT_valid.sum() for YT_valid in YT_valids])

	max_iter = 10
	max_iter = int(max_iter)

	m = grb.Model('M')
	m.setParam('OutputFlag', False)
	m.Params.Threads = 1
	if M_sum2one == 'sum':
		vM = m.addVars(GG, K, lb=0.)
		m.addConstrs((vM.sum('*', i) == 1 for i in range(K)))
	else:
		assert False
	for __t__ in range(max_iter):

		# print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		# print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

		obj = 0
		for beta, YT, YT_valid, sigma_yx_inv, A, B, G, XT in zip(betas, YTs, YT_valids, sigma_yx_invs, As, Bs, Gs, XTs):
			if dropout_str == 'origin':
				t = YT.flatten()
			elif dropout_str == 'pass':
				t = YT[YT_valid]
			else: assert False
			obj += beta * sigma_yx_inv**2 * np.dot(t, t)
			t = -2 * beta * sigma_yx_inv**2 * A
			obj += grb.quicksum([t[i, j] * vM[i, j] for i in range(G) for j in range(K)])
			if dropout_str == 'origin':
				t = beta * sigma_yx_inv**2 * B
				t[np.diag_indices(K)] += 1e-5
				obj += grb.quicksum([t[i, i] * vM[k, i] * vM[k, i] for k in range(G) for i in range(K)])
				t *= 2
				obj += grb.quicksum([t[i, j] * vM[k, i] * vM[k, j] for k in range(G) for i in range(K) for j in range(i+1, K)])
			elif dropout_str == 'pass':
				for g, (y, y_valid) in enumerate(zip(YT.T, YT_valid.T)):
					t = beta * sigma_yx_inv**2 * (XT[y_valid].T @ XT[y_valid])
					t[np.diag_indices(K)] += 1e-5
					obj += grb.quicksum([t[i, i] * vM[g, i] * vM[g, i] for i in range(K)])
					t *= 2
					obj += grb.quicksum([t[i, j] * vM[g, i] * vM[g, j] for i in range(K) for j in range(i+1, K)])
			else: assert False
			del t, beta, YT, A, B, G
		kk = 0
		if kk != 0:
			obj += grb.quicksum([kk/2 * vM[k, i] * vM[k, i] for k in range(GG) for i in range(K)])
		m.setObjective(obj, grb.GRB.MINIMIZE)
		m.optimize()
		M = np.array([[vM[i, j].x for j in range(K)] for i in range(GG)])
		if M_sum2one == 'sum' or M_sum2one == 'None':
			pass
		elif M_sum2one == 'L1':
			M /= np.abs(M).sum(0, keepdims=True)
		elif M_sum2one == 'L2':
			M /= np.sqrt((M ** 2).sum(0, keepdims=True))
		else:
			assert False

		last_sigma_yx_invs = np.copy(sigma_yx_invs)

		ds = np.array([
			np.dot(YT.flatten(), YT.flatten()) - 2 * np.dot(A.flatten(), M[:G].flatten()) + np.dot(B.flatten(), (
						M[:G].T @ M[:G]).flatten())
			for YT, A, B, G in zip(YTs, As, Bs, Gs)
		])
		if sigma_yx_inv_str == 'separate':
			sigma_yx_invs = ds / sizes
			sigma_yx_invs = 1. / np.sqrt(sigma_yx_invs)
			for YT, G, A, B, sigma_yx_inv in zip(YTs, Gs, As, Bs, sigma_yx_invs):
				assert np.abs(
					YT.size - sigma_yx_inv ** 2 * (
							np.dot(YT.flatten(), YT.flatten())
							- 2 * np.dot(A.flatten(), M[:G].flatten())
							+ np.dot(B.flatten(), (M[:G].T @ M[:G]).flatten())
					)
				) < 1e-7
		elif sigma_yx_inv_str == 'average':
			sigma_yx_invs = np.dot(betas, ds) / np.dot(betas, sizes)
			re = sigma_yx_invs
			sigma_yx_invs = np.full(len(YTs), 1 / (np.sqrt(sigma_yx_invs) + 1e-20))
		elif sigma_yx_inv_str.startswith('average '):
			idx = np.array(list(map(int, sigma_yx_inv_str.split(' ')[1:])))
			sigma_yx_invs = np.dot(betas[idx], ds[idx]) / np.dot(betas[idx], sizes[idx])
			re = sigma_yx_invs
			sigma_yx_invs = np.full(len(YTs), 1 / (np.sqrt(sigma_yx_invs) + 1e-20))
		else:
			assert False

		# print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		# print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

		d = sigma_yx_invs - last_sigma_yx_invs
		print(f"At iter {__t__}, σ_yxInv: {np.array2string(d, formatter={'all': '{:.2e}'.format})} -> {np.array2string(sigma_yx_invs, formatter={'all': '{:.2e}'.format})}")
		sys.stdout.flush()

		if (np.abs(d) / sigma_yx_invs).max() < 1e-5 or len(As) <= 1 or sigma_yx_inv_str.startswith('average'):
			break

	# emission
	Q_Y = -np.dot(betas, sizes) / 2
	# partition function - Pr [ Y | X, Theta ]
	Q_Y -= np.dot(betas, sizes) * np.log(2*np.pi) / 2
	Q_Y += (sizes * betas * np.log(sigma_yx_invs)).sum()

	print(f'Optimizing M, σ_yx_inv ends in {timeit.default_timer() - time_start:.2f}')

	return M, sigma_yx_invs, Q_Y


def M_step_X(O, H, Theta, modelSpec, pars, iiter, pairwise_potential_str, nsample4integral, lambda_SigmaXInv, **kwargs):
	print('Optimizing Sigma_x_inv, prior_xs ...')
	sys.stdout.flush()
	time_start = timeit.default_timer()

	K, YTs, YT_valids, Es, Es_empty, betas = O
	(XTs,) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta
	As, Bs, tC, talphas, talpha_es, tnus = pars

	Q_X = 0

	if all(prior_x[0] == 'Gaussian' for prior_x in prior_xs) and pairwise_potential_str == 'linear':
		Q_X = 0

		C = tC.cpu().data.numpy()
		C = (C + C.T) / 2
		kk = 1e3 * np.dot(betas, [sum(map(len, E)) for E in Es])
		Sigma_x_inv = - C / kk
		Q_X -= kk / 2 * np.dot(Sigma_x_inv.flatten(), Sigma_x_inv.flatten())
		Q_X -= np.dot(C.flatten(), Sigma_x_inv.flatten())

		prior_xs_old = prior_xs
		prior_xs = []
		for YT, beta, XT, prior_x, talpha, B in zip(YTs, betas, XTs, prior_xs_old, talphas, Bs):
			N = len(YT)
			if prior_x[0] == 'Gaussian':
				mu_x, sigma_x_inv = prior_x[1:]
				alpha = talpha.cpu().data.numpy()
				mu_x = alpha / N
				t = np.diag(B) - 2 * mu_x * alpha + N * mu_x ** 2
				sigma_x_inv = np.sqrt(N / t)
				# prior on X
				Q_X -= beta * XT.size / 2
				# partition function
				Q_X -= beta * N * K * np.sqrt(2 * np.pi)
				Q_X += beta * N * np.log(sigma_x_inv).sum()
				prior_xs.append((prior_x[0], mu_x, sigma_x_inv,))
				del mu_x, sigma_x_inv, alpha, t
			else:
				assert False
		del prior_xs_old
		assert False
	elif pairwise_potential_str in ['linear', 'linear w/ shift']:
		#         valid_diter = 1
		#		 valid_diter = 7
		# valid_diter = 97
		#         valid_diter = 997
		#         valid_diter = 3343
		valid_diter = 7177
		# valid_diter = 9973
		max_iter = 2e4
		max_iter = int(max_iter)
		batch_sizes = [512, ] * len(YTs)
		#         requires_grad = True
		requires_grad = False

		var_list = []
		optimizers = []
		tSigma_x_inv = torch.tensor(Sigma_x_inv, dtype=dtype, device=device, requires_grad=requires_grad)
		tdelta_x = torch.tensor(delta_x, dtype=dtype, device=device, requires_grad=requires_grad)
		var_list += [tSigma_x_inv, tdelta_x]
		if not all(Es_empty):
			optimizers += [
				#                 torch.optim.SGD([tSigma_x_inv], lr=1e-3, momentum=.0),
				torch.optim.Adam([tSigma_x_inv], lr=1e-5),
				#                 torch.optim.SGD([tdelta_x], lr=1e-3, momentum=.0),
				torch.optim.Adam([tdelta_x], lr=1e-5),
			]
		tprior_xs = []
		for prior_x in prior_xs:
			if prior_x[0] == 'Truncated Gaussian':
				mu_x, sigma_x_inv = prior_x[1:]
				tmu_x = torch.tensor(mu_x, dtype=dtype, device=device, requires_grad=requires_grad)
				tsigma_x_inv = torch.tensor(sigma_x_inv, dtype=dtype, device=device, requires_grad=requires_grad)
				tprior_xs.append((prior_x[0], tmu_x, tsigma_x_inv,))
				var_list += [tmu_x, tsigma_x_inv]
				optimizers += [
					#                     torch.optim.SGD([tmu_x], lr=1e-3, momentum=.0),
					torch.optim.Adam([tmu_x], lr=1e-4),
					#                     torch.optim.SGD([tsigma_x_inv], lr=1e-3, momentum=.0),
					torch.optim.Adam([tsigma_x_inv], lr=1e-4),
				]
				del mu_x, sigma_x_inv, tmu_x, tsigma_x_inv
			elif prior_x[0] == 'Exponential':
				lambda_x, = prior_x[1:]
				tlambda_x = torch.tensor(lambda_x, dtype=dtype, device=device, requires_grad=requires_grad)
				tprior_xs.append((prior_x[0], tlambda_x,))
				var_list.append(tlambda_x)
				optimizers += [
					#					 torch.optim.SGD([tlambda_x], lr=1e-3)
					torch.optim.Adam([tlambda_x], lr=1e-4)
				]
			else:
				assert False
		for t in var_list: t.grad = torch.zeros_like(t)

		tdiagBs = [torch.tensor(np.diag(B), dtype=dtype, device=device) for B in Bs]
		tNus = [tnu.sum(0) for tnu in tnus]
		#		 tNus = [None for tnu in tnus]
		#		 tNu2s = [tnu.t() @ tnu for tnu in tnus]
		tNu2s = [None for tnu in tnus]
		talpha_e_all = torch.zeros_like(talpha_es[0])
		for beta, talpha_e in zip(betas, talpha_es): talpha_e_all.add_(beta, talpha_e)
		tNu_all = torch.zeros_like(tNus[0])
		for beta, tNu in zip(betas, tNus): tNu_all.add_(beta, tNu)
		NEs = [sum(map(len, E)) for E in Es]
		tnEs = [torch.tensor(list(map(len, E)), dtype=dtype, device=device) for E in Es]

		print('Estimating parameters PyTorch initialized in {}'.format(timeit.default_timer() - time_start))
		sys.stdout.flush()

		row_idx, col_idx = np.triu_indices(K, 0)

		__t__, func, last_func = 0, None, torch.empty([], dtype=dtype, device=device).fill_(np.nan)
		best_func, best_iter = torch.empty([], dtype=dtype, device=device).fill_(np.nan), -1
		for __t__ in range(max_iter + 1):
			if not requires_grad:
				for t in var_list: t.grad.fill_(0)

			assert (tSigma_x_inv - tSigma_x_inv.t()).abs().max() < 1e-15

			func = torch.zeros([], dtype=dtype, device=device)
			if requires_grad:
				func_grad = torch.zeros([], dtype=dtype, device=device, requires_grad=True)

			# pairwise potential
			tSigma_x_inv.grad.add_(tC).addr_(alpha=-1, vec1=talpha_e_all, vec2=tdelta_x)
			t = talpha_e_all @ tSigma_x_inv
			func.add_(tC.view(-1) @ tSigma_x_inv.view(-1)).sub_(t @ tdelta_x)
			# func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1) - talpha_e_all @ tSigma_x_inv @ tdelta_x
			tdelta_x.grad.sub_(t)
			del t

			for N, E_empty, NE, tnE, beta, talpha, tnu, tNu, tNu2, tdiagB, tprior_x, batch_size in zip(
					map(len, YTs),
					Es_empty, NEs, tnEs, betas,
					talphas, tnus, tNus, tNu2s, tdiagBs,
					tprior_xs,
					batch_sizes,
			):
				# if N <= batch_size: idx = slice(None, None, None)
				# else: idx = sorted(np.random.choice(N, size=batch_size, replace=False))
				if tprior_x[0] == 'Truncated Gaussian':
					tmu_x, tsigma_x_inv = tprior_x[1:]

					# prior on X
					t = torch.zeros_like(tsigma_x_inv)
					t.add_(tdiagB).addcmul_(-2, tmu_x, talpha).add_(N, tmu_x.pow(2)).mul_(tsigma_x_inv)
					# t = t.add(tdiagB).addcmul(-2, tmu_x, talpha).add(N, tmu_x.pow(2)).mul(tsigma_x_inv)
					tsigma_x_inv.grad.add_(beta, t)
					func.add_(beta / 2, t @ tsigma_x_inv)
					# func_grad = func_grad.add(beta/2, t @ tsigma_x_inv)
					del t
					tmu_x.grad.add_(-beta, tsigma_x_inv.pow(2).mul_(talpha))
					tmu_x.grad.add_(beta * N, tsigma_x_inv.pow(2).mul_(tmu_x))

					# partition function - Pr [ X | Theta ]
					func.add_(np.log(2 * np.pi) * beta * N * K / 2.)
					# func_grad = func_grad.add(np.log(2 * np.pi) * beta * N * K / 2.)

					func.add_(-beta * N, tsigma_x_inv.log().sum())
					# func_grad = func_grad.add(-beta * N, tsigma_x_inv.log().sum())
					tsigma_x_inv.grad.add_(-beta * N, tsigma_x_inv.pow(-1))

					t = tsigma_x_inv * tmu_x
					tsigma_x_inv.grad.addcmul_(-beta * N, t, tmu_x)
					tmu_x.grad.addcmul_(-beta * N, t, tsigma_x_inv)
					func.add_(-beta * N / 2, t.pow_(2).sum())
					# func_grad = func_grad.add(-beta * N / 2, t.pow(2).sum())
					del t

					tnu_bar = tnu.addr(alpha=-1, vec1=tnE, vec2=tdelta_x)
					tXT_bar = tnu_bar.matmul(tSigma_x_inv).div(-tsigma_x_inv[None, :]).addcmul(tsigma_x_inv, tmu_x)
					func.add_(beta / 2, tXT_bar.pow(2).sum())
					# func_grad = func_grad.add(beta / 2, tXT_bar.pow(2).sum())
					t = tXT_bar.sum(0)
					tmu_x.grad.addcmul_(beta, tsigma_x_inv, t)
					tsigma_x_inv.grad.addcmul_(beta, tmu_x, t)
					del t
					t = tXT_bar.t() @ tnu_bar
					tSigma_x_inv.grad.addcdiv_(-beta, t, tsigma_x_inv[:, None])
					tsigma_x_inv.grad.addcmul_(beta, tsigma_x_inv.pow(-2), t.mul_(tSigma_x_inv).sum(1))
					del t
					tdelta_x.grad.add_(tnE.matmul(tXT_bar).div_(tsigma_x_inv).matmul(tSigma_x_inv))

					# using scipy.stats.norm which can calculate the logarithm of cdf
					"""
					ub = tSigma_x_inv_nu.div(-tsigma_x_inv[None]).add_(tmu_x.mul(tsigma_x_inv)[None])	# bar x / σ_x
					ub_np = ub.cpu().data.numpy()
					logcdf_np = scipy.stats.norm.logcdf(ub_np)
					logcdf = torch.tensor(logcdf_np, dtype=dtype, device=device)
					del logcdf_np
					# logpdf_np = scipy.stats.norm.logpdf(ub_np)
					# logpdf = torch.tensor(logpdf_np, dtype=dtype, device=device)
					# del logpdf_np
					logpdf = ub.pow_(2).div_(-2).sub_(np.log(2*np.pi)/2)
					del ub, ub_np
					func.add_(beta, logcdf.sum())
					t = logpdf.sub_(logcdf).exp_()		# 1 / denominator
					flag = (t >= 0).all()
					if __t__ % valid_diter == 0: print(t.max().item())
					if not flag:
						print('tmu_x', tmu_x.cpu().data.numpy(), 'tsigma_x_inv', tsigma_x_inv.cpu().data.numpy())
						print('t', t.min().item(), t.max().item())
					assert flag
					del logcdf, logpdf
					# """
					# using PyTorch
					# """
					tt = tXT_bar.div_(np.sqrt(2))
					# tt = tXT_bar.div(np.sqrt(2))
					t = tt.neg().erfc_().div_(2)  # cdf
					# t = tt.neg().erfc().div(2)  # cdf
					func.add_(beta, t.log().sum())
					# func_grad = func_grad.add(beta, t.log().sum())
					tt.pow_(2).neg_().exp_().div_(np.sqrt(2 * np.pi))  # pdf
					# tt = tt.pow(2).neg_().exp_().div_(np.sqrt(2 * np.pi))  # pdf
					t.pow_(-1).mul_(tt)  # 1 / denominator
					# t = t.pow(-1).mul_(tt)  # 1 / denominator
					flag = (t >= 0).all()
					if not flag:
						print('tmu_x', tmu_x.cpu().data.numpy(), 'tsigma_x_inv', tsigma_x_inv.cpu().data.numpy())
						print('tt', tt.min().item(), tt.max().item())
						print('t', t.min().item(), t.max().item())
					assert flag
					del tt
					# """
					# # #
					tt = t.sum(0)
					tsigma_x_inv.grad.addcmul_(beta, tmu_x, tt)
					tmu_x.grad.addcmul_(beta, tsigma_x_inv, tt)
					del tt
					tt = t.t() @ tnu_bar
					tSigma_x_inv.grad.addcmul_(-beta, tsigma_x_inv.pow(-1)[:, None], tt)
					tsigma_x_inv.grad.addcmul_(beta, tsigma_x_inv.pow(-2), tt.mul_(tSigma_x_inv).sum(1))
					tdelta_x.grad.add_(beta, tnE.matmul(t).div_(tsigma_x_inv).matmul(tSigma_x_inv))
					del t

					del tmu_x, tsigma_x_inv
				elif tprior_x[0] == 'Exponential':
					# prior
					tlambda_x, = tprior_x[1:]
					tlambda_x.grad.add_(beta, talpha)
					func.add_(beta, tlambda_x @ talpha)
					#					 func_grad = func_grad.add(beta, tlambda_x @ talpha)

					tnu_delta = tnu.addr(alpha=-1, vec1=tnE, vec2=tdelta_x)
					t = (tnu_delta @ tSigma_x_inv).add_(tlambda_x[None]).pow_(-1)  # t = 1 / bar lambda
					# t = (tnu_delta @ tSigma_x_inv).add(tlambda_x[None]).pow(-1)  # t = 1 / bar lambda
					# func.add_(beta, t.log().sum())
					func_grad = func_grad.add(beta, t.log().sum())
					flag = (t > 0).all()
					if not flag:
						print(__t__)
						print('min t', t.min().item())
						print('max t', t.max().item())
					assert flag
					# tlambda_x.grad.sub_(beta, t.sum(0))
					# tSigma_x_inv.grad.addmm_(alpha=-beta, mat1=t.t(), mat2=tnu_delta)
					# tdelta_x.grad.add_(beta, (tnE @ t).mul_(tsigma_x_inv.pow(-1)) @ tSigma_x_inv)

					del t
					del tlambda_x

			if requires_grad:
				func_grad.backward()
				func = func + func_grad

			# prior on parameters
			# prior on Σ_x^inv
			kk = 1e1 * np.dot(betas, list(map(len, Es)))
			tSigma_x_inv.grad.add_(kk, tSigma_x_inv)
			func.add_(kk / 2, tSigma_x_inv.pow(2).sum())
			# prior on δ_x
			kk = 1e1 * np.dot(betas, list(map(len, Es)))
			tdelta_x.grad.add_(kk, tdelta_x)
			func.add_(kk/2, tdelta_x.pow(2).sum())
			# prior on prior of X
			for tprior_x in tprior_xs:
				if tprior_x[0] == 'Truncated Gaussian':
					tmu_x, tsigma_x_inv = tprior_x[1:]
					# kk = 1e1 * np.dot(betas, list(map(len, YTs)))
					# tmu_x.grad.add_(kk, tmu_x)
					# func.add_(kk/2, tmu_x.pow(2).sum())
					del tmu_x, tsigma_x_inv
				elif tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]
					del tlambda_x
				else:
					assert False

			# normalize gradient by the weighted sizes of data sets
			for N, beta, tprior_x, batch_size in zip(map(len, YTs), betas, tprior_xs, batch_sizes):
				for tp in tprior_x[1:]:
					if tp.grad is not None:
						tp.grad.div_(N * beta)
			if not all(Es_empty):
				tSigma_x_inv.grad.div_(np.dot(betas, [sum(map(len, E)) for E in Es]))
				tdelta_x.grad.div_(np.dot(betas, [sum(map(len, E)) for E in Es]))
			func.div_(np.dot(betas, list(map(len, YTs))))

			tSigma_x_inv.grad.add_(tSigma_x_inv.grad.t()).div_(2)

			# for debug
			# tprior_xs[0][1].grad.zero_()
			# tprior_xs[0][1].grad[:-1].zero_()
			# tprior_xs[0][2].grad[:-1].zero_()
			#	 tSigma_x_inv.grad.zero_()
			#	 tSigma_x_inv.grad.triu_().tril_()

			# to change the variable
			if not requires_grad:
				pass
				# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv
				# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
				# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
				# tSigma_x_inv.pow_(-1)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Truncated Gaussian':
						tmu_x, tsigma_x_inv = tprior_x[1:]
						# tsigma_x_inv.grad.mul_(-1).mul_(tsigma_x_inv.pow(2))
						# tsigma_x_inv.pow_(-1)
						tsigma_x_inv.grad.mul_(tsigma_x_inv)
						tsigma_x_inv.log_()
						del tmu_x, tsigma_x_inv
					elif tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						tlambda_x.grad.mul_(tlambda_x)
						tlambda_x.log_()
						del tlambda_x
					else:
						assert False

			# setting flags
			stop_flag = True
			# stop_flag = False
			stop_flag &= (tSigma_x_inv.grad.abs() / (tSigma_x_inv.abs() + 1e-3)).abs().max().item() < 1e-2
			stop_flag &= (tdelta_x.grad.abs() / (tdelta_x.abs() + 1e-6)).abs().max().item() < 1e-2
			for tprior_x in tprior_xs:
				if tprior_x[0] == 'Truncated Gaussian':
					tmu_x, tsigma_x_inv = tprior_x[1:]
					stop_flag &= (tmu_x.grad / (
							tmu_x + 1e-3)).abs().max().item() < 1e-4 or tmu_x.grad.abs().max().item() < 1e-4
					stop_flag &= tsigma_x_inv.grad.abs().max().item() < 1e-4
					del tmu_x, tsigma_x_inv
				elif tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]
					stop_flag &= tlambda_x.grad.abs().max().item() < 1e-4
					del tlambda_x
				else:
					assert False
			stop_flag &= bool(func > last_func - 1e-5)
			# stop_flag |= best_func == func and __t__ > best_iter + 20

			if __t__ >= max_iter:
				stop_flag = True

			warning_flag = bool(func > last_func + 1e-15)

			if __t__ % valid_diter == 0 or stop_flag or warning_flag:
				# for tprior_x in tprior_xs:
				# 	for t in tprior_x[1:]:
				# 		print(np.array2string(t		.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
				# print()
				# for tprior_x in tprior_xs:
				# 	for t in tprior_x[1:]:
				# 		print(np.array2string(t.grad.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
				# print()
				# print(np.array2string(tSigma_x_inv		[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
				# print(np.array2string(tSigma_x_inv.grad	[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
				if warning_flag: print('Warning', end='\t')
				print(
					f'At iter {__t__},\t'
					f'func = {(func - last_func).item():.2e} -> {func.item():.2e}\t'
					f'Σ_x^inv: {tSigma_x_inv.max().item():.1e} - {tSigma_x_inv.min().item():.1e} = {tSigma_x_inv.max() - tSigma_x_inv.min():.1e} '
					f'grad = {tSigma_x_inv.grad.min().item():.2e} {tSigma_x_inv.grad.max().item():.2e}\t',
					f'δ_x: {tdelta_x.max().item():.1e} - {tdelta_x.min().item():.1e} = {tdelta_x.max() - tdelta_x.min():.1e} '
					f'grad = {tdelta_x.grad.min().item():.2e} {tdelta_x.grad.max().item():.2e}',
					end=''
				)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Truncated Gaussian':
						tmu_x, tsigma_x_inv = tprior_x[1:]
						print(
							f'\t'
							f'(μ {tmu_x.grad.min():.2e} {tmu_x.grad.max():.2e}) '
							f'(σ {tsigma_x_inv.grad.min():.2e} {tsigma_x_inv.grad.max():.2e});',
							end=''
						)
						del tmu_x, tsigma_x_inv
					elif tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						print(
							f'\t'
							f'(λ {tlambda_x.grad.min():.2e} {tlambda_x.grad.max():.2e});',
							end=''
						)
						del tlambda_x
					else:
						assert False
				print()
				sys.stdout.flush()

			# stop_flag = True

			if not stop_flag:
				for optimizer in optimizers: optimizer.step()
				if requires_grad:
					for optimizer in optimizers: optimizer.zero_grad()

			if not requires_grad:
				pass
				# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv
				# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
				# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
				# tSigma_x_inv.pow_(-1)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Truncated Gaussian':
						tmu_x, tsigma_x_inv = tprior_x[1:]
						# tsigma_x_inv.grad.mul_(-1).mul_(tsigma_x_inv.pow(2))
						# tsigma_x_inv.pow_(-1)
						tsigma_x_inv.exp_()
						tsigma_x_inv.grad.div_(tsigma_x_inv)
						del tmu_x, tsigma_x_inv
					elif tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						tlambda_x.exp_()
						tlambda_x.grad.div_(tlambda_x)
						del tlambda_x
					else:
						assert False

			if stop_flag: break

			for tprior_x in tprior_xs:
				if tprior_x[0] == 'Truncated Gaussian':
					tmu_x, tsigma_x_inv = tprior_x[1:]
					# assert all(tmu_x > 0 for tmu_x in tmu_xs)
					flag = (tsigma_x_inv > 0).all()
					if not flag:
						print(f'iter {__t__}')
						print(tsigma_x_inv)
						print(tsigma_x_inv.grad)
					assert flag
					del tmu_x, tsigma_x_inv
				elif tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]
					assert (tlambda_x > 0).all()
				else:
					assert False

			last_func = func
			if not best_func <= func:
				best_func, best_iter = func, __t__

		del prior_xs
		Sigma_x_inv = tSigma_x_inv.cpu().data.numpy()
		delta_x = tdelta_x.cpu().data.numpy()
		prior_xs = []
		for tprior_x in tprior_xs:
			if tprior_x[0] == 'Truncated Gaussian':
				tmu_x, tsigma_x_inv = tprior_x[1:]
				prior_xs.append((tprior_x[0], tmu_x.cpu().data.numpy(), tsigma_x_inv.cpu().data.numpy(),))
				del tmu_x, tsigma_x_inv
			elif tprior_x[0] == 'Exponential':
				tlambda_x, = tprior_x[1:]
				prior_xs.append((tprior_x[0], tlambda_x.cpu().data.numpy(),))
				del tlambda_x
			else:
				assert False
			del tprior_x
		del tprior_xs

		Q_X = -func.mul_(np.dot(betas, list(map(len, YTs)))).item()

		print('Estimating parameters Sigma_X_inv_2 ends with {} iterations in {}'.format(__t__ + 1,
																						 timeit.default_timer() - time_start))
	elif all(prior_x[0] in ['Exponential shared', 'Exponential shared fixed'] for prior_x in prior_xs) and pairwise_potential_str == 'normalized':
		prior_xs_old = prior_xs
		prior_xs = []
		for N, prior_x, talpha in zip(map(len, YTs), prior_xs_old, talphas):
			if prior_x[0] == 'Exponential shared':
				lambda_x, = prior_x[1:]
				lambda_x = talpha.mean().div_(N).pow(-1).cpu().data.numpy()
				Q_X -= lambda_x * talpha.sum().cpu().data.numpy()
				Q_X += N*K*np.log(lambda_x) - N*scipy.special.loggamma(K)
				prior_x = prior_x[:1] + (np.full(K, lambda_x), )
				prior_xs.append(prior_x)
			elif prior_x[0] == 'Exponential shared fixed':
				lambda_x, = prior_x[1:]
				Q_X -= lambda_x.mean() * talpha.sum().cpu().data.numpy()
				prior_xs.append(prior_x)
			else:
				assert False
		del prior_xs_old

		if not all(Es_empty):
			# valid_diter = 1
			# valid_diter = 7
			# valid_diter = 31
			# valid_diter = 97
			valid_diter = 331
			# valid_diter = 997
			# valid_diter = 3343
			# valid_diter = 7177
			# valid_diter = 9973
			max_iter = 1e4
			max_iter = int(max_iter)
			batch_sizes = [512, ] * len(YTs)
			# requires_grad = True
			requires_grad = False

			var_list = []
			optimizers = []
			schedulars = []
			tSigma_x_inv = torch.tensor(Sigma_x_inv, dtype=dtype, device=device, requires_grad=requires_grad)
			tdelta_x = torch.tensor(delta_x, dtype=dtype, device=device, requires_grad=requires_grad) # not used yet
			var_list += [
				tSigma_x_inv,
				tdelta_x,
			]
			if not all(Es_empty):
				schedular = None
				# optimizer = torch.optim.SGD([tSigma_x_inv], lr=1e1, momentum=.0)
				# optimizer = torch.optim.Adam([tSigma_x_inv], lr=1e-2 if iiter == 1 else 1e-3)
				optimizer = torch.optim.Adam([tSigma_x_inv], lr=1e-2)
				schedular = torch.optim.lr_scheduler.StepLR(optimizer, valid_diter, gamma=0.98)
				optimizers.append(optimizer)
				if schedular: schedulars.append(schedular)
				del optimizer, schedular
				"""
				# optimizer = torch.optim.SGD([tdelta_x], lr=1e-3, momentum=.0)
				# optimizer = torch.optim.Adam([tdelta_x], lr=1e-5)
				# optimizers.append(optimizer)
				# """
			tprior_xs = []
			for prior_x in prior_xs:
				if prior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
					lambda_x, = prior_x[1:]
					tlambda_x = torch.tensor(lambda_x, dtype=dtype, device=device, requires_grad=requires_grad)
					tprior_xs.append((prior_x[0], tlambda_x,))
					var_list.append(tlambda_x)
					# optimizers += [
					# 	torch.optim.SGD([tlambda_x], lr=1e-3)
						# torch.optim.Adam([tlambda_x], lr=1e-4)
					# ]
					del lambda_x
				else:
					assert False
			for t in var_list: t.grad = torch.zeros_like(t)

			tdiagBs = [torch.tensor(np.diag(B), dtype=dtype, device=device) for B in Bs]
			tNus = [tnu.sum(0) for tnu in tnus]
			# tNus = [None for tnu in tnus]
			tNu2s = [tnu.t() @ tnu for tnu in tnus]
			# tNu2s = [None for tnu in tnus]
			talpha_e_all = torch.zeros_like(talpha_es[0])
			for beta, talpha_e in zip(betas, talpha_es): talpha_e_all.add_(beta, talpha_e)
			# tNu_all = torch.zeros_like(tNus[0])
			# for beta, tNu in zip(betas, tNus): tNu_all.add_(beta, tNu)
			NEs = [sum(map(len, E)) for E in Es]
			tnEs = [torch.tensor(list(map(len, E)), dtype=dtype, device=device) for E in Es]
			tZTs = [torch.tensor(XT, dtype=dtype, device=device) for XT in XTs]
			for tZT in tZTs: tZT.div_(tZT.sum(1, keepdim=True))

			# Sigma_x_inv_ub = 1.
			# Sigma_x_inv_lb = -1.
			Sigma_x_inv_lb = None
			Sigma_x_inv_ub = None
			Sigma_x_inv_constraint = None				# complete matrix
			# Sigma_x_inv_constraint = 'diagonal'			# diagonal matrix
			# Sigma_x_inv_constraint = 'diagonal same'	# diagonal matrix, diagonal values are all the same

			print('Estimating parameters PyTorch initialized in {}'.format(timeit.default_timer() - time_start))
			sys.stdout.flush()

			row_idx, col_idx = np.triu_indices(K, 0)

			assumption_str = 'mean-field'
			# assumption_str = None
			# assumption_str = 'independent'
			random_flag = assumption_str in [
				'independent',
				'mean-field',
			]

			tsample = None
			tZe = None
			nbatch = 0
			batch_size = 0
			n_samples = 0
			regenerate_diter = int(1e10)
			tZes = [None] * len(YTs)
			tsample = None
			tsample_base = None
			n_samples_valid = n_samples
			n_samples_train = n_samples
			if assumption_str == None:
				regenerate_diter = 97
				torch.manual_seed(iiter)
				nbatch = 2**8
				batch_size = 2**6
				nbatch = int(nbatch)
				batch_size = int(batch_size)
				n_samples = nbatch * batch_size
				tZes = []
				for YT in YTs:
					tZes.append(torch.zeros((n_samples, K*K), dtype=dtype, device=device))
			elif assumption_str == 'mean-field':
				pass
			elif assumption_str == 'independent':
				n_samples_valid = 2**20
				n_samples_train = 2**10
				regenerate_diter = 1
				tsample_base = [torch.empty([n_samples_valid, K], dtype=dtype, device=device) for _ in range(2)]
			else:
				assert False

			if assumption_str in [None, 'independent']:
				tC.div_(2)

			loggamma_K = loggamma(K)

			# print(tC.div(np.dot(betas, [sum(map(len, E)) for E in Es])))

			__t__, func, last_func = 0, None, torch.empty([], dtype=dtype, device=device).fill_(np.nan)
			best_func, best_iter = torch.empty([], dtype=dtype, device=device).fill_(np.nan), -1
			tSigma_x_inv_best = None
			for __t__ in range(max_iter + 1):
				if not requires_grad:
					for t in var_list: t.grad.zero_()
				else:
					for optimizer in optimizers:
						optimizer.zero_grad()

				assert (tSigma_x_inv - tSigma_x_inv.t()).abs().max() < 1e-15
				if Sigma_x_inv_lb is not None:
					tSigma_x_inv.clamp_(min=Sigma_x_inv_lb)
				if Sigma_x_inv_ub is not None:
					tSigma_x_inv.clamp_(max=Sigma_x_inv_ub)
				if Sigma_x_inv_constraint in ['diagonal', 'diagonal same']:
					tSigma_x_inv.triu_().tril_()
				if Sigma_x_inv_constraint in ['diagonal same']:
					tSigma_x_inv[(range(K), range(K))] = tSigma_x_inv[(range(K), range(K))].mean()

				func = torch.zeros([], dtype=dtype, device=device)
				if requires_grad:
					func_grad = torch.zeros([], dtype=dtype, device=device, requires_grad=True)

				# pairwise potential
				# tSigma_x_inv.grad.add_(tC).addr_(alpha=-1, vec1=talpha_e_all, vec2=tdelta_x)
				# t = talpha_e_all @ tSigma_x_inv
				# func.add_(tC.view(-1) @ tSigma_x_inv.view(-1)).sub_(t @ tdelta_x)
				# func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1) - talpha_e_all @ tSigma_x_inv @ tdelta_x
				# tdelta_x.grad.sub_(t)
				# del t
				tSigma_x_inv.grad.add_(tC)
				func.add_(tC.view(-1) @ tSigma_x_inv.view(-1))
				# func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1)

				for N, E_empty, NE, tnE, E, beta, tZT, tZe, talpha, tnu, tNu, tNu2, tdiagB, tprior_x in zip(
						map(len, YTs),
						Es_empty, NEs, tnEs, Es, betas, tZTs, tZes,
						talphas, tnus, tNus, tNu2s, tdiagBs,
						tprior_xs,
				):
					# if N <= batch_size: idx = slice(None, None, None)
					# else: idx = sorted(np.random.choice(N, size=batch_size, replace=False))

					if E_empty:
						continue

					if assumption_str == 'mean-field':
						if tprior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
							if __t__ % valid_diter == 0:
								idx = slice(None)
							else:
								idx = np.random.choice(N, min(nsample4integral, N), replace=False)
							tnu = tnu[idx].contiguous()
							c = NE / tnE[idx].sum()
							# Z_z
							teta = tnu @ tSigma_x_inv
							teta.grad = torch.zeros_like(teta)
							# torch.manual_seed(iiter)
							# print((tSigma_x_inv.max() - tSigma_x_inv.min()).item(), (teta.max() - teta.min()).item())
							if iiter > 1 or __t__ > 100:
								# tlogZ = integrateOfExponentialOverSimplexSampling(teta, requires_grad=requires_grad, seed=iiter*max_iter+__t__)
								tlogZ = integrateOfExponentialOverSimplexInduction2(teta, grad=c, requires_grad=requires_grad)
							else:
								# tlogZ = integrateOfExponentialOverSimplexSampling(teta, requires_grad=requires_grad, seed=iiter*max_iter+__t__)
								tlogZ = integrateOfExponentialOverSimplexInduction2(teta, grad=c, requires_grad=requires_grad)
							if requires_grad:
								func_grad = func_grad.add(beta*c, tlogZ.sum())
							else:
								func.add_(beta*c, tlogZ.sum())
								tSigma_x_inv.grad.addmm_(alpha=beta, mat1=tnu.t(), mat2=teta.grad)
							pass
						else:
							assert False
					elif assumption_str == None:
						if tprior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
							if __t__ % regenerate_diter == 0:
								tZe = tZe.zero_().view(nbatch, batch_size, K, K)
								tsample = torch.empty([N, batch_size, K], dtype=dtype, device=device)
								for i in range(nbatch):
									tsample.exponential_()
									tsample.div_(tsample.sum(-1, keepdim=True))
									# if i == 1:
									# 	tsample.copy_(tZT[:, None, :])
									for u, e in enumerate(E):
										e = [_ for _ in e if _ > u]
										if e:
											tn = tsample[e].sum(0)
											tZe[i].baddbmm_(tsample[u, :, :, None], tn[:, None, :])
								tZe = tZe.view(n_samples, K*K)
							if requires_grad:
								tZe.neg_()
								tprob = tZe @ tSigma_x_inv.view(-1)
								# if __t__ % valid_diter == 0:
								# if True:
								# 	print(tprob)
								# 	print(tSigma_x_inv.view(-1) @ tC.view(-1))
								tprob_offset = tprob.max()
								tlogZ = tprob.sub(tprob_offset).exp().mean().log().add(tprob_offset)
								func_grad = func_grad.add(beta, tlogZ)
								# if __t__ % valid_diter == 0:
								# if True:
								# 	print(f'{func.item():.2e}\t{func_grad.item():.2e}\t{tprob_offset.item():.2e}')
							else:
								tprob = (tZe @ tSigma_x_inv.view(-1)).neg_()
								tprob_offset = tprob.max()
								tprob.sub_(tprob_offset).exp_()
								tlogZ = tprob.sum()
								tprob.div_(tlogZ)
								# print(tprob.max(), tprob.min())
								tlogZ.div_(n_samples).log_().add_(tprob_offset)
								func.add_(beta, tlogZ)
								tSigma_x_inv.grad.view(-1).addmv_(alpha=-beta, mat=tZe.t(), vec=tprob)
						else:
							assert False
					elif assumption_str == 'independent':
						if tprior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
							if __t__ % valid_diter == 0:
								tsample = tsample_base
								n_samples = n_samples_valid
							else:
								tsample = [_[:n_samples_train] for _ in tsample_base]
								n_samples = n_samples_train
							if __t__ % regenerate_diter == 0:
								# torch.manual_seed(0)
								for _ in tsample:
									_.exponential_()
									_.div_(_.sum(-1, keepdim=True))
							tprob = (tsample[0][:, None, :] @ tSigma_x_inv[None] @ tsample[1][:, :, None]).view(-1).neg_()
							tprob_offset = tprob.max()
							tlogZ = tprob.sub_(tprob_offset).exp_().sum()
							tprob.div_(tlogZ)
							tSigma_x_inv.grad.addmm_(alpha=-beta*NE/2, mat1=tsample[0].mul(tprob[:, None]).t(), mat2=tsample[1])
							tlogZ.div_(n_samples).log_().add_(tprob_offset)
							func.add_(beta*NE/2, tlogZ)
						else:
							assert False
					else: assert False

				if requires_grad:
					# print(tSigma_x_inv.grad[::3, ::3])
					func_grad.backward()
					# print(tSigma_x_inv.grad[::3, ::3])
					func = func + func_grad
					# break

				# prior on parameters
				# prior on Σ_x^inv
				"""
				num_burnin_iter = 200
				# if iiter <= num_burnin_iter:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iter-iiter+1)/num_burnin_iter)
				# else:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es)))
				kk = 1e-3 * np.dot(betas, NEs)
				tSigma_x_inv.grad.add_(kk, tSigma_x_inv.pow(2).mul_(tSigma_x_inv.sign()))
				func.add_(kk/3, tSigma_x_inv.pow(3).abs_().sum())
				# """
				# """
				num_burnin_iter = 200
				# if iiter <= num_burnin_iter:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iter-iiter+1)/num_burnin_iter)
				# else:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es)))
				kk = lambda_SigmaXInv * np.dot(betas, NEs)
				# kk = float(sys.argv[1]) * np.dot(betas, NEs)
				# print(kk)
				# exit()
				tSigma_x_inv.grad.add_(kk, tSigma_x_inv)
				func.add_(kk / 2, tSigma_x_inv.pow(2).sum())
				# """
				"""
				num_burnin_iter = 200
				# if iiter <= num_burnin_iter:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iter-iiter+1)/num_burnin_iter)
				# else:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es)))
				k = 1e-4 * np.dot(betas, NEs)
				threshold_Sigma_x_inv_reg = 1e-3
				kk = k / threshold_Sigma_x_inv_reg
				idx = (tSigma_x_inv.abs() < threshold_Sigma_x_inv_reg).to(dtype)
				tSigma_x_inv.grad.add_(k, tSigma_x_inv.sign().mul_(1-idx))
				func.add_(k, tSigma_x_inv.mul(1-idx).abs_().sum())
				tSigma_x_inv.grad.add_(kk, tSigma_x_inv.mul(idx))
				func.add_(kk/2, tSigma_x_inv.mul(idx).pow_(2).sum())
				# """
				# prior on δ_x
				# kk = 1e1 * np.dot(betas, list(map(len, Es)))
				# tdelta_x.grad.add_(kk, tdelta_x)
				# func.add_(kk/2, tdelta_x.pow(2).sum())
				# prior on prior of X
				for tprior_x in tprior_xs:
					if tprior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
						tlambda_x, = tprior_x[1:]
						del tlambda_x
					else:
						assert False

				# normalize gradient by the weighted sizes of data sets
				# for N, beta, tprior_x, batch_size in zip(map(len, YTs), betas, tprior_xs, batch_sizes):
				# 	for tp in tprior_x[1:]:
				# 		if tp.grad is not None:
				# 			tp.grad.div_(N * beta)q
				if not all(Es_empty):
					tSigma_x_inv.grad.div_(np.dot(betas, NEs))
					tdelta_x.grad.div_(np.dot(betas, NEs))
				func.div_(np.dot(betas, list(map(len, YTs))))

				tSigma_x_inv.grad.add_(tSigma_x_inv.grad.clone().t()).div_(2)
				# tSigma_x_inv.grad.sub_(tSigma_x_inv.grad.mean())

				if Sigma_x_inv_lb is not None:
					tSigma_x_inv.grad[(tSigma_x_inv <= Sigma_x_inv_lb) * (tSigma_x_inv.grad > 0)] = 0
				if Sigma_x_inv_ub is not None:
					tSigma_x_inv.grad[(tSigma_x_inv >= Sigma_x_inv_ub) * (tSigma_x_inv.grad < 0)] = 0
				if Sigma_x_inv_constraint in ['diagonal', 'diagonal same']:
					tSigma_x_inv.grad.triu_().tril_()
				if Sigma_x_inv_constraint in ['diagonal same']:
					tSigma_x_inv.grad[(range(K), range(K))] = tSigma_x_inv.grad[(range(K), range(K))].mean()

				# for debug
				# tprior_xs[0][1].grad.zero_()
				# tprior_xs[0][1].grad[:-1].zero_()
				# tprior_xs[0][2].grad[:-1].zero_()
				#	 tSigma_x_inv.grad.zero_()
				#	 tSigma_x_inv.grad.triu_().tril_()

				# to change the variable
				"""
				if not requires_grad:
					pass
					# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv
					# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
					# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
					# tSigma_x_inv.pow_(-1)
					for tprior_x in tprior_xs:
						if tprior_x[0] == 'Truncated Gaussian':
							tmu_x, tsigma_x_inv = tprior_x[1:]
							# tsigma_x_inv.grad.mul_(-1).mul_(tsigma_x_inv.pow(2))
							# tsigma_x_inv.pow_(-1)
							tsigma_x_inv.grad.mul_(tsigma_x_inv)
							tsigma_x_inv.log_()
							del tmu_x, tsigma_x_inv
						elif tprior_x[0] == 'Exponential':
							tlambda_x, = tprior_x[1:]
							tlambda_x.grad.mul_(tlambda_x)
							tlambda_x.log_()
							del tlambda_x
						else:
							assert False
				# """

				# setting flags
				best_flag = False
				if not random_flag or __t__ % valid_diter == 0:
					best_flag = not best_func <= func
					if best_flag:
						best_func, best_iter = func, __t__
						tSigma_x_inv_best = tSigma_x_inv.clone().detach()

				stop_flag = True
				# stop_flag = False
				stop_tSigma_x_inv_grad_pseudo = 1e-1
				stop_flag &= (tSigma_x_inv.grad.abs() / (tSigma_x_inv.abs() + stop_tSigma_x_inv_grad_pseudo)).abs().max().item() < 1e-2
				# stop_flag &= (tdelta_x.grad.abs() / (tdelta_x.abs() + 1e-6)).abs().max().item() < 1e-2
				for tprior_x in tprior_xs:
					if tprior_x[0] in ['Exponential shared', ]:
						tlambda_x, = tprior_x[1:]
						stop_flag &= tlambda_x.grad.abs().max().item() < 1e-4
						del tlambda_x
					elif tprior_x[0] in ['Exponential shared fixed', ]:
						pass
					else:
						assert False
				if random_flag:
					stop_flag &= not bool(func <= last_func - 1e-3*valid_diter)
				else:
					stop_flag &= not bool(func <= last_func - 1e-3)
				stop_flag |= random_flag and not __t__ < best_iter + 2*valid_diter
				# stop_flag |= best_func == func and __t__ > best_iter + 20
				if random_flag and __t__ % valid_diter != 0:
					stop_flag = False

				if __t__ >= max_iter:
					stop_flag = True

				warning_flag = bool(func > last_func + 1e-10)
				warning_flag &= not random_flag or __t__ % valid_diter == 0
				# warning_flag = True

				if __t__ % valid_diter == 0 or stop_flag or warning_flag or (regenerate_diter != 1 and (__t__ % regenerate_diter == 0 or (__t__+1) % regenerate_diter == 0)):
					# for tprior_x in tprior_xs:
					# 	for t in tprior_x[1:]:
					# 		print(np.array2string(t		.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
					# print()
					# for tprior_x in tprior_xs:
					# 	for t in tprior_x[1:]:
					# 		print(np.array2string(t.grad.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
					# print()
					# print(np.array2string(tSigma_x_inv		[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
					# print(np.array2string(tSigma_x_inv.grad	[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
					# if warning_flag: print('Warning', end='\t')
					print(
						f'At iter {__t__},\t'
						f'func = {(func - last_func).item():.2e} -> {func.item():.2e}\t'
						f'Σ_x^inv: {tSigma_x_inv.max().item():.1e} - {tSigma_x_inv.min().item():.1e} = {tSigma_x_inv.max() - tSigma_x_inv.min():.1e} '
						f'grad = {tSigma_x_inv.grad.min().item():.2e} {tSigma_x_inv.grad.max().item():.2e}\t'
						f'var/grad = {(tSigma_x_inv.grad.abs()/(tSigma_x_inv.abs() + stop_tSigma_x_inv_grad_pseudo)).abs().max().item():.2e}'
						# f'δ_x: {tdelta_x.max().item():.1e} - {tdelta_x.min().item():.1e} = {tdelta_x.max() - tdelta_x.min():.1e} '
						# f'grad = {tdelta_x.grad.min().item():.2e} {tdelta_x.grad.max().item():.2e}'
						, end=''
					)
					if warning_flag: print('\tWarning', end='')
					# for tprior_x in tprior_xs:
					# 	if tprior_x[0] == 'Truncated Gaussian':
					# 		tmu_x, tsigma_x_inv = tprior_x[1:]
					# 		print(
					# 			f'\t'
					# 			f'(μ {tmu_x.grad.min():.2e} {tmu_x.grad.max():.2e}) '
					# 			f'(σ {tsigma_x_inv.grad.min():.2e} {tsigma_x_inv.grad.max():.2e});',
					# 			end=''
					# 		)
					# 		del tmu_x, tsigma_x_inv
					# 	elif tprior_x[0] == 'Exponential':
					# 		tlambda_x, = tprior_x[1:]
					# 		print(
					# 			f'\t'
					# 			f'(λ {tlambda_x.grad.min():.2e} {tlambda_x.grad.max():.2e});',
					# 			end=''
					# 		)
					# 		del tlambda_x
					# 	else:
					# 		assert False
					if best_flag:
						print('\tbest', end='')
					print()
					# print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
					# print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
					sys.stdout.flush()

				# stop_flag = True

				# Sigma_x_inv_old = tSigma_x_inv.clone().detach().cpu().data.numpy()
				# Sigma_x_inv_sign_old = tSigma_x_inv.sign().cpu().data.numpy()
				# Sigma_x_inv_sign_old[np.abs(Sigma_x_inv_old) < threshold_Sigma_x_inv_reg] = 0
				if not stop_flag:
					for optimizer in optimizers: optimizer.step()
					for schedular in schedulars: schedular.step()
				# Sigma_x_inv_sign = tSigma_x_inv.sign().cpu().data.numpy()
				# Sigma_x_inv_sign[np.abs(tSigma_x_inv.cpu().data.numpy()) < threshold_Sigma_x_inv_reg] = 0
				# b = (Sigma_x_inv_sign_old * Sigma_x_inv_sign) == -1
				# if b.any():
				# 	print('-'*10)
				# 	print(b.sum())
				# 	print(Sigma_x_inv_old[b])
				# 	print(tSigma_x_inv.cpu().data.numpy()[b])

				"""
				if not requires_grad:
					pass
					# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv 
					# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
					# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
					# tSigma_x_inv.pow_(-1)
					for tprior_x in tprior_xs:
						if tprior_x[0] == 'Exponential shared':
							tlambda_x, = tprior_x[1:]
							tlambda_x.exp_()
							tlambda_x.grad.div_(tlambda_x)
							del tlambda_x
						else:
							assert False
				# """

				if stop_flag: break

				for tprior_x in tprior_xs:
					if tprior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
						tlambda_x, = tprior_x[1:]
						assert (tlambda_x > 0).all()
					else:
						assert False

				if not random_flag or __t__ % valid_diter == 0:
					last_func = func

			# del prior_xs
			tSigma_x_inv = tSigma_x_inv_best
			func = best_func
			Sigma_x_inv = tSigma_x_inv.cpu().data.numpy()
			delta_x = tdelta_x.cpu().data.numpy()
			# prior_xs = []
			# for tprior_x in tprior_xs:
			# 	if tprior_x[0] == 'Truncated Gaussian':
			# 		tmu_x, tsigma_x_inv = tprior_x[1:]
			# 		prior_xs.append((tprior_x[0], tmu_x.cpu().data.numpy(), tsigma_x_inv.cpu().data.numpy(),))
			# 		del tmu_x, tsigma_x_inv
			# 	elif tprior_x[0] == 'Exponential':
			# 		tlambda_x, = tprior_x[1:]
			# 		prior_xs.append((tprior_x[0], tlambda_x.cpu().data.numpy(),))
			# 		del tlambda_x
			# 	else:
			# 		assert False
			# 	del tprior_x
			# del tprior_xs

			Q_X -= func.mul_(np.dot(betas, list(map(len, YTs)))).item()

			print('Estimating parameters Sigma_X_inv_2 ends with {} iterations in {}'.format(__t__+1, timeit.default_timer() - time_start))
	elif all(prior_x[0] == 'Exponential' for prior_x in prior_xs) and pairwise_potential_str == 'normalized':
		valid_diter = 1
		# valid_diter = 7
		# valid_diter = 97
		# valid_diter = 997
		# valid_diter = 3343
		# valid_diter = 7177
		# valid_diter = 9973
		display_diter = 1
		# display_diter = 10
		max_iter = 2e4
		max_iter = int(max_iter)
		batch_sizes = [512, ] * len(YTs)
		# requires_grad = True
		requires_grad = False

		var_list = []
		optimizers = []
		tSigma_x_inv = torch.tensor(Sigma_x_inv, dtype=dtype, device=device, requires_grad=requires_grad)
		tdelta_x = torch.tensor(delta_x, dtype=dtype, device=device, requires_grad=requires_grad)
		var_list += [
			tSigma_x_inv,
			tdelta_x,
		]
		if not all(Es_empty):
			optimizers += [
				# torch.optim.SGD([tSigma_x_inv], lr=1e-3, momentum=.0),
				torch.optim.Adam([tSigma_x_inv], lr=1e-3),
				# torch.optim.SGD([tdelta_x], lr=1e-3, momentum=.0),
				# torch.optim.Adam([tdelta_x], lr=1e-3),
			]
		tprior_xs = []
		for prior_x in prior_xs:
			if prior_x[0] == 'Exponential':
				lambda_x, = prior_x[1:]
				tlambda_x = torch.tensor(lambda_x, dtype=dtype, device=device, requires_grad=requires_grad)
				tprior_xs.append((prior_x[0], tlambda_x,))
				var_list.append(tlambda_x)
				optimizers += [
					# torch.optim.SGD([tlambda_x], lr=1e-6)
					torch.optim.Adam([tlambda_x], lr=1e-4)
				]
				del lambda_x
			else:
				assert False
		for t in var_list: t.grad = torch.zeros_like(t)

		tdiagBs = [torch.tensor(np.diag(B), dtype=dtype, device=device) for B in Bs]
		# tNus = [tnu.sum(0) for tnu in tnus]
		tNus = [None for tnu in tnus]
		# tNu2s = [tnu.t() @ tnu for tnu in tnus]
		tNu2s = [None for tnu in tnus]
		talpha_e_all = torch.zeros_like(talpha_es[0])
		for beta, talpha_e in zip(betas, talpha_es): talpha_e_all.add_(beta, talpha_e)
		# tNu_all = torch.zeros_like(tNus[0])
		# for beta, tNu in zip(betas, tNus): tNu_all.add_(beta, tNu)
		NEs = [sum(map(len, E)) for E in Es]
		tnEs = [torch.tensor(list(map(len, E)), dtype=dtype, device=device) for E in Es]

		print('Estimating parameters PyTorch initialized in {}'.format(timeit.default_timer() - time_start))
		sys.stdout.flush()

		row_idx, col_idx = np.triu_indices(K, 0)

		__t__, func, last_func = 0, None, torch.empty([], dtype=dtype, device=device).fill_(np.nan)
		best_func, best_iter = torch.empty([], dtype=dtype, device=device).fill_(np.nan), -1
		for __t__ in range(max_iter + 1):
			valid_flag = __t__ % valid_diter == 0

			if not requires_grad:
				for t in var_list: t.grad.zero_()
			else:
				for optimizer in optimizers:
					optimizer.zero_grad()

			assert (tSigma_x_inv - tSigma_x_inv.t()).abs().max() < 1e-15

			func = torch.zeros([], dtype=dtype, device=device)
			if requires_grad:
				func_grad = torch.zeros([], dtype=dtype, device=device, requires_grad=True)

			# pairwise potential
			# tSigma_x_inv.grad.add_(tC).addr_(alpha=-1, vec1=talpha_e_all, vec2=tdelta_x)
			# t = talpha_e_all @ tSigma_x_inv
			# func.add_(tC.view(-1) @ tSigma_x_inv.view(-1)).sub_(t @ tdelta_x)
			# func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1) - talpha_e_all @ tSigma_x_inv @ tdelta_x
			# tdelta_x.grad.sub_(t)
			# del t
			tSigma_x_inv.grad.add_(tC)
			func.add_(tC.view(-1) @ tSigma_x_inv.view(-1))
			# func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1)

			tC.div_(2)

			for N, E, E_empty, NE, tnE, beta, talpha, tnu, tNu, tNu2, tdiagB, tprior_x in zip(
					map(len, YTs),
					Es, Es_empty, NEs, tnEs, betas,
					talphas, tnus, tNus, tNu2s, tdiagBs,
					tprior_xs,
			):
				# if N <= batch_size: idx = slice(None, None, None)
				# else: idx = sorted(np.random.choice(N, size=batch_size, replace=False))

				if tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]

					# prior
					tlambda_x.grad.add_(beta, talpha)
					func.add_(beta, tlambda_x @ talpha)

					# Z_Xif valid_flag:
					if valid_flag:
						nbatch = 2
						batch_size = 3
					else:
						nbatch = 7
						batch_size = 5
					nbatch = 2
					batch_size = 3
					nbatch = int(nbatch)
					batch_size = int(batch_size)
					n = nbatch * batch_size
					# tsample = torch.empty([batch_size, N, K], dtype=dtype, device=device)
					tsample = torch.empty([N, batch_size, K], dtype=dtype, device=device)
					tprob = torch.zeros(nbatch, batch_size, dtype=dtype, device=device)
					tlambda_x_grad = torch.zeros((nbatch, batch_size, ) + tlambda_x.shape, dtype=dtype, device=device)
					tSigma_x_inv_grad = torch.zeros((nbatch, batch_size, ) + tSigma_x_inv.shape, dtype=dtype, device=device)
					for i in range(nbatch):
						# print(i)
						tsample.exp_().div_(tlambda_x[None, None])
						# tlambda_x_grad[i].copy_(tsample.sum(1))
						tlambda_x_grad[i].copy_(tsample.sum(0))
						tsample.div_(tsample.sum(-1, keepdim=True))
						for u, e in enumerate(E):
							if e:
								# tn = tsample[:, e, :].sum(1)
								# tprob[i].sub_((tsample[:, [u], :] @ tSigma_x_inv[None] @ tn[:, :, None]).view(-1))
								# tSigma_x_inv_grad[i].baddbmm_(tsample[:, u, :, None], tn[:, None, :])
								tn = tsample[e, :, :].sum(0)
								tprob[i].sub_((tsample[u, :, None, :] @ tSigma_x_inv[None] @ tn[:, :, None]).view(-1))
								tSigma_x_inv_grad[i].baddbmm_(tsample[u, :, :, None], tn[:, None, :])
					tprob = tprob.view(-1)
					tlambda_x_grad = tlambda_x_grad.view(n, -1)
					tSigma_x_inv_grad = tSigma_x_inv_grad.view(n, -1)
					tSigma_x_inv_grad.div_(2)
					tprob_max = tprob.max()
					tprob.sub_(tprob_max).exp_()
					tlogZ = tprob.mean()
					tprob.div_(tlogZ).div_(n)
					tlogZ.log_().add_(tprob_max).sub_(N, tlambda_x.log().sum())
					tlambda_x.addmv_(alpha=-beta,  mat=tlambda_x_grad.t(), vec=tprob)
					tSigma_x_inv.grad.view(-1).addmv_(alpha=-beta, mat=tSigma_x_inv_grad.view(n, -1).t(), vec=tprob)
					"""
					teta = tnu @ tSigma_x_inv
					teta.grad = torch.zeros_like(teta)
					# loggamma_K = loggamma(K)
					if valid_flag:
						n = 2**10
						nround = 100
					else:
						n = 2**10
						nround = 10
					n = int(n)
					nround = int(nround)
					if requires_grad:
						tlogZ = torch.zeros(N, dtype=dtype, device=device)
						for _ in range(nround):
							if iiter is not None: torch.manual_seed(iiter*max_iter*nround + __t__*nround + _)
							# tsample = torch.empty([n, K], dtype=dtype, device=device).exponential_().div_(tlambda_x[None])
							tsample = torch.distributions.exponential.Exponential(tlambda_x).rsample(torch.Size([n]))
							tnsample = tsample.div(tsample.sum(1, keepdim=True))
							tlogvalue = teta.neg().matmul(tnsample.t())
							tlogvalue_max = tlogvalue.max(1, keepdim=True)[0]
							tlogZ = tlogZ + tlogvalue.sub(tlogvalue_max).exp().mean(1).log().add(tlogvalue_max.squeeze(1))
						tlogZ = tlogZ / nround
						tlogZ = tlogZ - tlambda_x.log().sum()
						func_grad = func_grad.add(beta, tlogZ.sum())
					else:
						tlogZ_sum = torch.zeros([], dtype=dtype, device=device)
						tlambda_x_grad = torch.zeros_like(tlambda_x)

						for _ in range(nround):
							if iiter is not None: torch.manual_seed(iiter*max_iter*nround + __t__*nround + _)
							# tsample = torch.empty([n, K], dtype=dtype, device=device).exponential_().div_(tlambda_x[None])
							tsample = torch.distributions.exponential.Exponential(tlambda_x).rsample(torch.Size([n]))

							tnsample = tsample.div(tsample.sum(1, keepdim=True))
							tprob = teta.neg().matmul(tnsample.t())
							tmax = tprob.max(1, keepdim=True)[0]
							tprob.sub_(tmax)
							tlogZ = tprob.exp_().mean(1)
							tprob.div_(tlogZ[:, None]).div_(n)
							tlogZ_sum.add_(tlogZ.log_().sum()).add_(tmax.sum())

							teta.grad.addmm_(mat1=tprob, mat2=tnsample)
							tlambda_x_grad.addmv_(mat=tsample.t(), vec=tprob.sum(0))

						tlogZ_sum.div_(nround).sub_(N, tlambda_x.log().sum())
						tlambda_x_grad.div_(nround)
						teta.grad.div_(nround)

						func.add_(beta, tlogZ_sum.sum())
						tSigma_x_inv.grad.addmm_(alpha=-beta, mat1=tnu.t(), mat2=teta.grad)
						tlambda_x.grad.sub_(beta, tlambda_x_grad)
					# """
				else:
					assert False

			if requires_grad:
				# t = tprior_xs[0][1].grad.clone()
				func_grad.backward()
				# print(tprior_xs[0][1].grad - t)
				# print(tlambda_x.pow(-1).mul(N))
				func = func + func_grad
			# print(func)
			# print(tSigma_x_inv.grad)
			# print(tprior_xs[0][1].grad)
			# exit()

			# prior on parameters
			# prior on Σ_x^inv
			# if iiter < 30:
			# kk = 1e-2 * np.dot(betas, list(map(len, Es)))
			# tSigma_x_inv.grad.add_(kk, tSigma_x_inv)
			# func.add_(kk / 2, tSigma_x_inv.pow(2).sum())
			# prior on δ_x
			# kk = 1e1 * np.dot(betas, list(map(len, Es)))
			# tdelta_x.grad.add_(kk, tdelta_x)
			# func.add_(kk/2, tdelta_x.pow(2).sum())
			# prior on prior of X
			for tprior_x in tprior_xs:
				if tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]
					del tlambda_x
				else:
					assert False

			# normalize gradient by the weighted sizes of data sets
			for N, beta, tprior_x, batch_size in zip(map(len, YTs), betas, tprior_xs, batch_sizes):
				for tp in tprior_x[1:]:
					if tp.grad is not None:
						tp.grad.div_(N * beta)
			if not all(Es_empty):
				tSigma_x_inv.grad.div_(np.dot(betas, [sum(map(len, E)) for E in Es]))
				tdelta_x.grad.div_(np.dot(betas, [sum(map(len, E)) for E in Es]))
			func.div_(np.dot(betas, list(map(len, YTs))))

			tSigma_x_inv.grad.add_(tSigma_x_inv.grad.clone().t()).div_(2)
			# exit()

			# for debug
			# tprior_xs[0][1].grad.zero_()
			# tprior_xs[0][1].grad[:-1].zero_()
			# tprior_xs[0][2].grad[:-1].zero_()
			#	 tSigma_x_inv.grad.zero_()
			#	 tSigma_x_inv.grad.triu_().tril_()

			# to change the variable
			# """
			if not requires_grad:
				pass
				# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv
				# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
				# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
				# tSigma_x_inv.pow_(-1)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Truncated Gaussian':
						tmu_x, tsigma_x_inv = tprior_x[1:]
						# tsigma_x_inv.grad.mul_(-1).mul_(tsigma_x_inv.pow(2))
						# tsigma_x_inv.pow_(-1)
						tsigma_x_inv.grad.mul_(tsigma_x_inv)
						tsigma_x_inv.log_()
						del tmu_x, tsigma_x_inv
					elif tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						tlambda_x.grad.mul_(tlambda_x)
						tlambda_x.log_()
						del tlambda_x
					else:
						assert False
			# """

			# setting flags
			stop_flag = False
			if valid_flag:
				stop_flag = True
				stop_flag &= (tSigma_x_inv.grad.abs() / (tSigma_x_inv.abs() + 1e-6)).abs().max().item() < 1e-4
				stop_flag &= (tdelta_x.grad.abs() / (tdelta_x.abs() + 1e-6)).abs().max().item() < 1e-3
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						stop_flag &= tlambda_x.grad.abs().max().item() < 1e-4
						del tlambda_x
					else:
						assert False
				# stop_flag &= not bool(func <= last_func - 1e-5)
				# stop_flag |= best_func == func and __t__ > best_iter + 20

				if not best_func <= func:
					best_func, best_iter = func, __t__
				if __t__ - best_iter > 1e3:
					stop_flag = True
			if __t__ >= max_iter:
				stop_flag = True

			warning_flag = False
			if valid_flag:
				warning_flag = bool(func > last_func + 1e-15)

			if __t__ % (valid_diter*display_diter) == 0 or stop_flag or (warning_flag and False):
				for tprior_x in tprior_xs:
					for t in tprior_x[1:]:
						print(np.array2string(t		.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
				print()
				for tprior_x in tprior_xs:
					for t in tprior_x[1:]:
						print(np.array2string(t.grad.cpu().data.numpy(), formatter={'all': '{:.2e}'.format}), end='\t')
				print()
				# print(np.array2string(tSigma_x_inv		[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
				# print(np.array2string(tSigma_x_inv.grad	[row_idx, col_idx].cpu().data.numpy()	, formatter={'all': '{:.2e}'.format}))
				# if warning_flag: print('Warning', end='\t')
				print(
					f'At iter {__t__},\t'
					f'func = {(func - last_func).item():.2e} -> {func.item():.2e}\t'
					f'Σ_x^inv: {tSigma_x_inv.max().item():.1e} - {tSigma_x_inv.min().item():.1e} = {tSigma_x_inv.max() - tSigma_x_inv.min():.1e} '
					f'grad = {tSigma_x_inv.grad.min().item():.2e} {tSigma_x_inv.grad.max().item():.2e}\t',
					# f'δ_x: {tdelta_x.max().item():.1e} - {tdelta_x.min().item():.1e} = {tdelta_x.max() - tdelta_x.min():.1e} '
					# f'grad = {tdelta_x.grad.min().item():.2e} {tdelta_x.grad.max().item():.2e}',
					end=''
				)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Truncated Gaussian':
						tmu_x, tsigma_x_inv = tprior_x[1:]
						print(
							f'\t'
							f'(μ {tmu_x.grad.min():.2e} {tmu_x.grad.max():.2e}) '
							f'(σ {tsigma_x_inv.grad.min():.2e} {tsigma_x_inv.grad.max():.2e});',
							end=''
						)
						del tmu_x, tsigma_x_inv
					elif tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						print(
							f'\t'
							f'(λ {tlambda_x.grad.min():.2e} {tlambda_x.grad.max():.2e});',
							end=''
						)
						del tlambda_x
					else:
						assert False
				print()
				sys.stdout.flush()

			# stop_flag = True

			# print('tSigma_x_inv', tSigma_x_inv)
			# print('tSigma_x_inv grad', tSigma_x_inv.grad)
			if not stop_flag:
				for optimizer in optimizers: optimizer.step()
			# print('tSigma_x_inv', tSigma_x_inv)
			# print('tSigma_x_inv grad', tSigma_x_inv.grad)

			if not requires_grad:
				pass
				# tSigma_x_inv.grad = - tSigma_x_inv @ tSigma_x_inv.grad @ tSigma_x_inv
				# tSigma_x_inv.copy_(torch.inverse(tSigma_x_inv))
				# tSigma_x_inv.grad.mul_(-1).mul_(tSigma_x_inv.pow(2))
				# tSigma_x_inv.pow_(-1)
				for tprior_x in tprior_xs:
					if tprior_x[0] == 'Exponential':
						tlambda_x, = tprior_x[1:]
						tlambda_x.exp_()
						tlambda_x.grad.div_(tlambda_x)
						del tlambda_x
					else:
						assert False

			if stop_flag: break

			for tprior_x in tprior_xs:
				if tprior_x[0] == 'Exponential':
					tlambda_x, = tprior_x[1:]
					if not (tlambda_x > 0).all():
						print(__t__)
						print(tlambda_x)
						print(tlambda_x.grad)
					assert (tlambda_x > 0).all()
				else:
					assert False

			if valid_flag:
				last_func = func

		# del prior_xs
		Sigma_x_inv = tSigma_x_inv.cpu().data.numpy()
		delta_x = tdelta_x.cpu().data.numpy()
		prior_xs = []
		for tprior_x in tprior_xs:
			if tprior_x[0] == 'Truncated Gaussian':
				tmu_x, tsigma_x_inv = tprior_x[1:]
				prior_xs.append((tprior_x[0], tmu_x.cpu().data.numpy(), tsigma_x_inv.cpu().data.numpy(),))
				del tmu_x, tsigma_x_inv
			elif tprior_x[0] == 'Exponential':
				tlambda_x, = tprior_x[1:]
				prior_xs.append((tprior_x[0], tlambda_x.cpu().data.numpy(),))
				del tlambda_x
			else:
				assert False
			del tprior_x
		del tprior_xs

		Q_X -= func.mul_(np.dot(betas, list(map(len, YTs)))).item()

		print('Estimating parameters Sigma_X_inv_2 ends with {} iterations in {}'.format(__t__ + 1, timeit.default_timer() - time_start))
	else:
		assert False

	return Sigma_x_inv, delta_x, prior_xs, Q_X

def estimateParameters(O, H, Theta, modelSpec, pairwise_potential_str, **kwargs):
	time_start_all = timeit.default_timer()
	print('Estimating parameters')
	sys.stdout.flush()

	K, YTs, YT_valids, Es, Es_empty, betas = O
	(XTs, ) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta

	assert M.shape[1] == K
	GG, K = M.shape
	Ns, Gs = zip(*[YT.shape for YT in YTs])
	assert max(Gs) == GG
	assert all(XT.shape == (N, K) for XT, N in zip(XTs, Ns))

	Q = 0

	if pairwise_potential_str == 'normalized' and all(prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed'] for prior_x in prior_xs):
		pars = genPars(O, H, Theta, modelSpec, **modelSpec)

		# print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		# print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

		M, sigma_yx_invs, Q_Y = M_step_Y(O, H, Theta, modelSpec, pars, **modelSpec)
		Q += Q_Y
		Theta = (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)
		del Q_Y

		# print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		# print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

		Sigma_x_inv, delta_x, prior_xs, Q_X = M_step_X(O, H, Theta, modelSpec, pars, **modelSpec)
		Q += Q_X
		Theta = (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)
		del Q_X
	else:
		raise NotImplementedError
		pars, Q_entropy = E_step(O, H, Theta, modelSpec, **modelSpec)
		Q += Q_entropy
		del Q_entropy

		M, sigma_yx_invs, Q_Y = M_step_Y(O, H, Theta, modelSpec, pars, **modelSpec)
		Q += Q_Y
		Theta = (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)
		del Q_Y

		Sigma_x_inv, delta_x, prior_xs, Q_X = M_step_X(O, H, Theta, modelSpec, pars, iiter, **modelSpec)
		Q += Q_X
		Theta = (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)
		del Q_X

	print('Estimating parameters all ends in {}'.format(timeit.default_timer() - time_start_all))

	return (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs), Q
