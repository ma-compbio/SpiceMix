import sys, logging, time, resource, gc, os
import multiprocessing
from multiprocessing import Pool
from util import print_datetime

import numpy as np
import gurobipy as grb
import torch


def estimateWeightsWithoutNeighbor(YT, M, XT, prior_x, sigma_yx_inv, X_constraint, dropout_mode, irepli):
	logging.info(f'{print_datetime()}Estimating weights without neighbors in repli {irepli}')
	K = XT.shape[1]

	res = []
	m = grb.Model('X w/o n')
	m.setParam('OutputFlag', False)
	m.Params.Threads = 1
	vx = m.addVars(K, lb=0.)
	assert X_constraint == 'none'

	# shared parts
	# quadratic term in log Pr[ Y | X, Theta ]
	obj_share = 0
	if dropout_mode == 'raw':
		# MTM = M.T @ M * (sigma_yx_inv**2 / 2.)
		MTM = (M.T @ M + 1e-6*np.eye(K)) * (sigma_yx_inv ** 2 / 2.)
		obj_share += grb.quicksum([vx[i] * MTM[i, i] * vx[i] for i in range(K)])
		MTM *= 2
		obj_share += grb.quicksum([vx[i] * MTM[i, j] * vx[j] for i in range(K) for j in range(i+1, K)])
		del MTM
		YTM = YT @ M * (-sigma_yx_inv**2)
	else:
		raise NotImplementedError

	# prior on X
	if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
		mu_x, sigma_x_inv = prior_x[1:]
		assert (sigma_x_inv > 0).all()
		t = sigma_x_inv ** 2 / 2.
		obj_share += grb.quicksum([t[i] * vx[i] * vx[i]	for i in range(K)])
		t *= - 2 * mu_x
		obj_share += grb.quicksum([t[i] * vx[i] 	    for i in range(K)])
		obj_share += np.dot(mu_x**2, sigma_x_inv**2) / 2.
		del t
	elif prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
		lambda_x, = prior_x[1:]
		assert (lambda_x >= 0).all()
		obj_share += grb.quicksum([lambda_x[i] * vx[i] for i in range(K)])
	else:
		raise NotImplementedError

	for y, yTM in zip(YT, YTM):
		obj = obj_share
		if dropout_mode != 'raw':
			raise NotImplemented
		obj = obj + grb.quicksum(yTM[i]*vx[i] for i in range(K)) + np.dot(y, y) * sigma_yx_inv / 2.
		m.setObjective(obj, grb.GRB.MINIMIZE)
		m.optimize()
		res.append([vx[i].x for i in range(K)])

	del m

	res = np.array(res)
	return res


def estimateWeightsICM(YT, E, M, XT, prior_x, sigma_yx_inv, Sigma_x_inv, X_constraint, dropout_mode, pairwise_potential_mode, irepli):
	logging.info(f'{print_datetime()}Estimating weights in repli {irepli} using ICM')
	N, G = YT.shape
	K = M.shape[1]
	MTM = None
	YTM = None
	if dropout_mode == 'raw':
		MTM = M.T @ M * sigma_yx_inv**2 / 2
		YTM = YT @ M * sigma_yx_inv**2 / 2
	else:
		raise NotImplementedError

	max_iter = 100
	max_iter = int(max_iter)
	max_iter_individual = 100

	m = grb.Model('ICM')
	m.setParam('OutputFlag', False)
	m.Params.Threads = 1
	m.Params.BarConvTol = 1e-6
	vz = m.addVars(K, lb=0.)
	m.addConstr(vz.sum() == 1)

	S = XT.sum(1, keepdims=True)
	ZT = XT / (S+1e-30)

	def calcObj(S, ZT):
		func = 0
		t = YT - S * ZT @ M.T
		if dropout_mode == 'raw':
			t = t.ravel()
		else:
			raise NotImplementedError
		func += np.dot(t, t) * sigma_yx_inv**2 / 2
		if pairwise_potential_mode == 'normalized':
			for e, z in zip(E, ZT):
				func += z @ Sigma_x_inv @ ZT[e].sum(0) / 2
		else:
			raise NotImplementedError
		if prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
			lambda_x, = prior_x[1:]
			func += lambda_x @ (S * ZT).sum(0)
			del lambda_x
		else:
			raise NotImplementedError
		func /= YT.size
		return func

	last_func = calcObj(S, ZT)
	best_func, best_iter = last_func, -1

	for iiter in range(max_iter):
		last_ZT = np.copy(ZT)
		last_S = np.copy(S)

		if pairwise_potential_mode == 'normalized':
			for i, (e, y, yTM, z, s) in enumerate(zip(E, YT, YTM, ZT, S)):
				eta = ZT[e].sum(0) @ Sigma_x_inv
				if dropout_mode != 'raw':
					raise NotImplementedError
				for iiiter in range(max_iter_individual):
					stop_flag = True

					a = z @ MTM @ z
					b = yTM @ z
					if prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
						lambda_x, = prior_x[1:]
						b -= lambda_x @ z / 2
						del lambda_x
					else:
						raise NotImplementedError
					s_new = b / (a+1e-30)
					s_new = np.maximum(s_new, 1e-15)
					ds = s_new - s
					stop_flag &= np.abs(ds) / (s + 1e-15) < 1e-3
					s = s_new

					obj = 0
					t = s**2 * MTM
					obj += grb.quicksum([vz[i] * t[i, i] * vz[i] for i in range(K)])
					t *= 2
					obj += grb.quicksum([vz[i] * t[i, j] * vz[j] for i in range(K) for j in range(i+1, K)])
					t = -2 * s * yTM
					t += eta
					if prior_x[0] in ['Exponential']:
						lambda_x, = prior_x[1:]
						t += lambda_x * s
						del lambda_x
					elif prior_x[0] in ['Exponential shared', 'Exponential shared fixed']:
						pass
					else:
						raise NotImplementedError
					obj += grb.quicksum([vz[i] * t[i] for i in range(K)])
					obj += y @ y * sigma_yx_inv**2 / 2
					m.setObjective(obj, grb.GRB.MINIMIZE)
					m.optimize()
					z_new = np.array([vz[i].x for i in range(K)])
					dz = z_new - z
					stop_flag &= np.abs(dz).max() < 1e-3
					z = z_new
					assert z.min() >= 0
					assert np.abs(z.sum()-1) < 1e-5

					if not stop_flag and iiiter == max_iter_individual-1:
						logging.warning(f'Cell {i} in the {irepli}-th repli did not converge in {max_iter_individual} iterations;\ts = {s:.2e}, ds = {ds:.2e}, max dz = {np.abs(dz).max():.2e}')

					if stop_flag:
						break

				ZT[i] = z
				S[i] = s
		else:
			raise NotImplementedError

		stop_flag = True

		dZT = ZT - last_ZT
		dS = S - last_S
		stop_flag &= np.abs(dZT).max() < 1e-2
		stop_flag &= np.abs(dS / (S + 1e-15)).max() < 1e-2

		func = calcObj(S, ZT)

		stop_flag &= func > last_func - 1e-4

		force_show_flag = False
		# force_show_flag |= np.abs(dZT).max() > 1-1e-5

		if iiter % 5 == 0 or stop_flag or force_show_flag:
			print(f'>{irepli} func at iter {iiter} = {func:.2e},\tdiff = {np.abs(dZT).max():.2e}\t{np.abs(dS).max():.2e}\t{func - last_func:.2e}')

			print(
				f'stat of XT: '
				f'#<0 = {(ZT < 0).sum().astype(np.float) / N:.1f}, '
				f'#=0 = {(ZT == 0).sum().astype(np.float) / N:.1f}, '
				f'#<1e-10 = {(ZT < 1e-10).sum().astype(np.float) / N:.1f}, '
				f'#<1e-5 = {(ZT < 1e-5).sum().astype(np.float) / N:.1f}, '
				f'#<1e-2 = {(ZT < 1e-2).sum().astype(np.float) / N:.1f}, '
				f'#>1e-1 = {(ZT > 1e-1).sum().astype(np.float) / N:.1f}'
			)

			print(
				f'stat of s: '
				f'#0 = {(S == 0).sum()}, '
				f'min = {S.min():.1e}, '
				f'max = {S.max():.1e}'
			)

			sys.stdout.flush()

		# print(func, last_func)
		assert not func > last_func + 1e-6
		last_func = func
		if not func >= best_func:
			best_func, best_iter = func, iiter

		if stop_flag:
			break

	del m

	XT = np.maximum(S, 1e-15) * ZT
	return XT
