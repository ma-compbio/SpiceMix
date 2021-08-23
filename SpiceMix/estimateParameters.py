import sys, time, itertools, resource, logging
from multiprocessing import Pool, Process
from util import psutil_process, print_datetime, array2string, PyTorchDType as dtype

import torch
import numpy as np
import gurobipy as grb
from scipy.special import loggamma

from sampleForIntegral import integrateOfExponentialOverSimplexInduction2


def estimateParametersY(self, max_iter=10):
	logging.info(f'{print_datetime()}Estimating M and sigma_yx_inv')

	As = []
	Bs = []
	sizes = np.fromiter(map(np.size, self.YTs), dtype=float)
	for YT, E, XT in zip(self.YTs, self.Es, self.XTs):
		if self.dropout_mode == 'raw':
			As.append(YT.T @ XT)
			Bs.append(XT.T @ XT)
		else:
			raise NotImplementedError

	m = grb.Model('M')
	m.setParam('OutputFlag', False)
	m.Params.Threads = 1
	if self.M_constraint == 'sum2one':
		vM = m.addVars(self.GG, self.K, lb=0.)
		m.addConstrs((vM.sum('*', i) == 1 for i in range(self.K)))
	else:
		raise NotImplementedError

	for iiter in range(max_iter):
		obj = 0
		for beta, YT, sigma_yx_inv, A, B, G, XT in zip(self.betas, self.YTs, self.sigma_yx_invs, As, Bs, self.Gs, self.XTs):
			# constant terms
			if self.dropout_mode == 'raw':
				t = YT.ravel()
			else:
				raise NotImplementedError
			obj += beta * sigma_yx_inv**2 * np.dot(t, t)
			# linear terms
			t = -2 * beta * sigma_yx_inv**2 * A
			obj += grb.quicksum([t[i, j] * vM[i, j] for i in range(G) for j in range(self.K)])
			# quadratic terms
			if self.dropout_mode == 'raw':
				t = beta * sigma_yx_inv**2 * B
				t[np.diag_indices(self.K)] += 1e-5
				obj += grb.quicksum([t[i, i] * vM[k, i] * vM[k, i] for k in range(G) for i in range(self.K)])
				t *= 2
				obj += grb.quicksum([t[i, j] * vM[k, i] * vM[k, j] for k in range(G) for i in range(self.K) for j in range(i+1, self.K)])
			else:
				raise NotImplementedError
			del t, beta, YT, A, B, G
		kk = 0
		if kk != 0:
			obj += grb.quicksum([kk/2 * vM[k, i] * vM[k, i] for k in range(self.GG) for i in range(self.K)])
		m.setObjective(obj, grb.GRB.MINIMIZE)
		m.optimize()
		self.M = np.array([[vM[i, j].x for j in range(self.K)] for i in range(self.GG)])
		if self.M_constraint in ['sum2one', 'none']:
			pass
		elif self.M_constraint == 'L1':
			self.M /= np.abs(self.M).sum(0, keepdims=True)
		elif self.M_constraint == 'L2':
			self.M /= np.sqrt((self.M ** 2).sum(0, keepdims=True))
		else:
			raise NotImplementedError

		last_sigma_yx_invs = np.copy(self.sigma_yx_invs)

		ds = np.array([
			np.dot(YT.ravel(), YT.ravel()) - 2*np.dot(A.ravel(), self.M[:G].ravel()) + np.dot(B.ravel(), (self.M[:G].T @ self.M[:G]).ravel())
			for YT, A, B, G in zip(self.YTs, As, Bs, self.Gs)
		])
		if self.sigma_yx_inv_mode == 'separate':
			t = ds / sizes
			self.sigma_yx_invs = 1. / np.sqrt(t)
		elif self.sigma_yx_inv_mode == 'average':
			t = np.dot(self.betas, ds) / np.dot(self.betas, sizes)
			self.sigma_yx_invs = np.full(self.num_repli, 1 / (np.sqrt(t) + 1e-20))
		elif self.sigma_yx_inv_mode.startswith('average '):
			idx = np.array(list(map(int, self.sigma_yx_inv_mode.split(' ')[1:])))
			t = np.dot(self.betas[idx], ds[idx]) / np.dot(self.betas[idx], sizes[idx])
			self.sigma_yx_invs = np.full(self.num_repli, 1 / (np.sqrt(t) + 1e-20))
		else:
			raise NotImplementedError

		d = self.sigma_yx_invs - last_sigma_yx_invs
		logging.info(f"{print_datetime()}At iter {iiter}, σ_yxInv: {array2string(d)} -> {array2string(self.sigma_yx_invs)}")

		if (np.abs(d) / self.sigma_yx_invs).max() < 1e-5 or self.num_repli <= 1 or self.sigma_yx_inv_mode.startswith('average'):
			break

	# emission
	Q_Y = -np.dot(self.betas, sizes) / 2
	# partition function - Pr [ Y | X, Theta ]
	Q_Y -= np.dot(self.betas, sizes) * np.log(2*np.pi) / 2
	Q_Y += (sizes * self.betas * np.log(self.sigma_yx_invs)).sum()

	return Q_Y


def estimateParametersX(self, iiter):
	logging.info(f'{print_datetime()}Estimating Sigma_x_inv and prior_xs')

	device = self.PyTorch_device

	Bs = []
	talphas = []
	talpha_es = []
	tC = torch.zeros([self.K, self.K], dtype=dtype, device=device)
	tnus = []
	for YT, E, XT, beta in zip(self.YTs, self.Es, self.XTs, self.betas):
		tXT = torch.tensor(XT, dtype=dtype, device=device)
		N, G = YT.shape
		if self.dropout_mode == 'raw':
			Bs.append(XT.T @ XT)
		else:
			raise NotImplementedError
		talphas.append(tXT.sum(0))
		talpha_es.append(torch.tensor(list(map(len, E)), dtype=dtype, device=device) @ tXT)
		tXT.div_(tXT.sum(1, keepdim=True).add_(1e-30))
		tnu = torch.empty([N, self.K], dtype=dtype, device=device)
		for tnui, ei in zip(tnu, E):
			tnui.copy_(tXT[ei].sum(0))
		tnus.append(tnu)
		tC.add_(alpha=beta, other=tXT.t() @ tnu)
		del tXT

	Q_X = 0

	if all(prior_x[0] == 'Gaussian' for prior_x in self.prior_xs) and self.pairwise_potential_mode == 'linear':
		raise NotImplementedError
	elif self.pairwise_potential_mode in ['linear', 'linear w/ shift']:
		raise NotImplementedError
	elif all(prior_x[0] in ['Exponential shared', 'Exponential shared fixed'] for prior_x in self.prior_xs) and self.pairwise_potential_mode == 'normalized':
		prior_xs_old = self.prior_xs
		self.prior_xs = []
		for N, prior_x, talpha in zip(self.Ns, prior_xs_old, talphas):
			if prior_x[0] == 'Exponential shared':
				lambda_x, = prior_x[1:]
				lambda_x = talpha.mean().div_(N).pow(-1).cpu().data.numpy()
				Q_X -= lambda_x * talpha.sum().cpu().data.numpy()
				Q_X += N*self.K*np.log(lambda_x) - N*loggamma(self.K)
				prior_x = prior_x[:1] + (np.full(self.K, lambda_x), )
				self.prior_xs.append(prior_x)
			elif prior_x[0] == 'Exponential shared fixed':
				lambda_x, = prior_x[1:]
				Q_X -= lambda_x.mean() * talpha.sum().cpu().data.numpy()
				self.prior_xs.append(prior_x)
			else:
				raise NotImplementedError
		del prior_xs_old

		if not all(self.Es_empty):
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
			batch_sizes = [512, ] * self.num_repli
			# requires_grad = True
			requires_grad = False

			var_list = []
			optimizers = []
			schedulars = []
			tSigma_x_inv = torch.tensor(self.Sigma_x_inv, dtype=dtype, device=device, requires_grad=requires_grad)
			var_list += [
				tSigma_x_inv,
			]
			schedular = None
			optimizer = torch.optim.Adam([tSigma_x_inv], lr=1e-2)
			schedular = torch.optim.lr_scheduler.StepLR(optimizer, valid_diter, gamma=0.98)
			optimizers.append(optimizer)
			if schedular: schedulars.append(schedular)
			del optimizer, schedular
			tprior_xs = []
			for prior_x in self.prior_xs:
				if prior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
					lambda_x, = prior_x[1:]
					tlambda_x = torch.tensor(lambda_x, dtype=dtype, device=device, requires_grad=requires_grad)
					tprior_xs.append((prior_x[0], tlambda_x,))
					var_list.append(tlambda_x)
					del lambda_x
				else:
					raise NotImplementedError
			for t in var_list: t.grad = torch.zeros_like(t)

			tdiagBs = [torch.tensor(np.diag(B).copy(), dtype=dtype, device=device) for B in Bs]
			tNus = [tnu.sum(0) for tnu in tnus]
			tNu2s = [tnu.t() @ tnu for tnu in tnus]
			talpha_e_all = torch.zeros_like(talpha_es[0])
			for beta, talpha_e in zip(self.betas, talpha_es): talpha_e_all.add_(alpha=beta, other=talpha_e)
			NEs = [sum(map(len, E)) for E in self.Es]
			tnEs = [torch.tensor(list(map(len, E)), dtype=dtype, device=device) for E in self.Es]
			tZTs = [torch.tensor(XT, dtype=dtype, device=device) for XT in self.XTs]
			for tZT in tZTs: tZT.div_(tZT.sum(1, keepdim=True))

			# Sigma_x_inv_ub = 1.
			# Sigma_x_inv_lb = -1.
			Sigma_x_inv_lb = None
			Sigma_x_inv_ub = None
			Sigma_x_inv_constraint = None				# complete matrix
			# Sigma_x_inv_constraint = 'diagonal'			# diagonal matrix
			# Sigma_x_inv_constraint = 'diagonal same'	# diagonal matrix, diagonal values are all the same

			row_idx, col_idx = np.triu_indices(self.K, 0)

			assumption_str = 'mean-field'
			# assumption_str = None
			# assumption_str = 'independent'
			random_flag = assumption_str in [
				'independent',
				'mean-field',
			]

			n_samples = 0
			regenerate_diter = int(1e10)
			tZes = [None] * self.num_repli
			nsample4integral = 64
			if assumption_str == None:
				raise NotImplementedError
			elif assumption_str == 'mean-field':
				pass
			elif assumption_str == 'independent':
				raise NotImplementedError
			else:
				raise NotImplementedError

			if assumption_str in [None, 'independent']:
				tC.div_(2)

			loggamma_K = loggamma(self.K)

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
					tSigma_x_inv[(range(self.K), range(self.K))] = tSigma_x_inv[(range(self.K), range(self.K))].mean()

				func = torch.zeros([], dtype=dtype, device=device)
				# if requires_grad:
				func_grad = torch.zeros([], dtype=dtype, device=device, requires_grad=True)

				# pairwise potential
				tSigma_x_inv.grad.add_(tC)
				if requires_grad:
					func_grad = func_grad + tC.view(-1) @ tSigma_x_inv.view(-1)
				else:
					func.add_(tC.view(-1) @ tSigma_x_inv.view(-1))

				for N, E_empty, NE, tnE, E, beta, tZT, tZe, talpha, tnu, tNu, tNu2, tdiagB, tprior_x in zip(
						self.Ns,
						self.Es_empty, NEs, tnEs, self.Es, self.betas, tZTs, tZes,
						talphas, tnus, tNus, tNu2s, tdiagBs,
						tprior_xs,
				):
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
							if iiter > 1 or __t__ > 100:
								# tlogZ = integrateOfExponentialOverSimplexSampling(teta, requires_grad=requires_grad, seed=iiter*max_iter+__t__)
								tlogZ = integrateOfExponentialOverSimplexInduction2(teta, grad=c, requires_grad=requires_grad, device=device)
							else:
								# tlogZ = integrateOfExponentialOverSimplexSampling(teta, requires_grad=requires_grad, seed=iiter*max_iter+__t__)
								tlogZ = integrateOfExponentialOverSimplexInduction2(teta, grad=c, requires_grad=requires_grad, device=device)
							if requires_grad:
								func_grad = func_grad.add(beta*c, tlogZ.sum())
							else:
								func.add_(alpha=beta*c, other=tlogZ.sum())
								tSigma_x_inv.grad.addmm_(alpha=beta, mat1=tnu.t(), mat2=teta.grad)
						else:
							raise NotImplementedError
					elif assumption_str == None:
						raise NotImplementedError
					elif assumption_str == 'independent':
						raise NotImplementedError
					else:
						raise NotImplementedError

				if requires_grad:
					func_grad.backward()
					func = func + func_grad

				# prior on Σ_x^inv
				# num_burnin_iter = 200
				# if iiter <= num_burnin_iter:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es))) * 1e-1**((num_burnin_iter-iiter+1)/num_burnin_iter)
				# else:
				# 	kk = 1e-1 * np.dot(betas, list(map(len, Es)))
				kk = self.lambda_SigmaXInv * np.dot(self.betas, NEs)
				tSigma_x_inv.grad.add_(kk, tSigma_x_inv)
				func.add_(kk / 2, tSigma_x_inv.pow(2).sum())

				# normalize gradient by the weighted sizes of data sets
				tSigma_x_inv.grad.div_(np.dot(self.betas, NEs))
				func.div_(np.dot(self.betas, list(map(len, self.YTs))))

				tSigma_x_inv.grad.add_(tSigma_x_inv.grad.clone().t()).div_(2)

				if Sigma_x_inv_lb is not None:
					tSigma_x_inv.grad[(tSigma_x_inv <= Sigma_x_inv_lb) * (tSigma_x_inv.grad > 0)] = 0
				if Sigma_x_inv_ub is not None:
					tSigma_x_inv.grad[(tSigma_x_inv >= Sigma_x_inv_ub) * (tSigma_x_inv.grad < 0)] = 0
				if Sigma_x_inv_constraint in ['diagonal', 'diagonal same']:
					tSigma_x_inv.grad.triu_().tril_()
				if Sigma_x_inv_constraint in ['diagonal same']:
					tSigma_x_inv.grad[(range(self.K), range(self.K))] = tSigma_x_inv.grad[(range(self.K), range(self.K))].mean()

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
				for tprior_x in tprior_xs:
					if tprior_x[0] in ['Exponential shared', ]:
						tlambda_x, = tprior_x[1:]
						stop_flag &= tlambda_x.grad.abs().max().item() < 1e-4
						del tlambda_x
					elif tprior_x[0] in ['Exponential shared fixed', ]:
						pass
					else:
						raise NotImplementedError
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
					if best_flag:
						print('\tbest', end='')
					print()
					sys.stdout.flush()

				# stop_flag = True

				if not stop_flag:
					for optimizer in optimizers: optimizer.step()
					for schedular in schedulars: schedular.step()

				if stop_flag: break

				if not random_flag or __t__ % valid_diter == 0:
					last_func = func

			tSigma_x_inv = tSigma_x_inv_best
			func = best_func
			self.Sigma_x_inv = tSigma_x_inv.cpu().data.numpy()

			Q_X -= func.mul_(np.dot(self.betas, list(map(len, self.YTs)))).item()
	elif all(prior_x[0] == 'Exponential' for prior_x in self.prior_xs) and self.pairwise_potential_mode == 'normalized':
		raise NotImplementedError
	else:
		raise NotImplementedError

	return Q_X
