import numpy as np
import timeit, sys, itertools, psutil, os, resource
from multiprocessing import Pool
import multiprocessing
from util import psutil_process, PyTorchDevice as device, PyTorchDType as dtype
import scipy.optimize
import gurobipy as grb
import torch

def estimateWeightsWithoutNeighbor(O, H, Theta, modelSpec, i, X_sum2one, dropout_str, **kwargs):
	K, YT, YT_valid, E, E_empty = O
	(XT, ) = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta
	N, G = YT.shape
	assert M.shape == (G, K)
	assert E_empty
	assert sigma_yx_inv > 0
	if YT.size == 0: return np.zeros([0, K], dtype=np.float)

	res = []
	m = grb.Model('X w/o n')
	m.setParam('OutputFlag', False)
	# m.Params.Threads = nCPU
	m.Params.Threads = 1
	vx = m.addVars(K, lb=0.)
	# if X_sum2one is True: m.addConstr(vx.sum() == 1)
	assert not X_sum2one

	# shared parts
	# quadratic term in log Pr[ Y | X, Theta ]
	obj_share = 0
	YTM = np.broadcast_to(np.nan, XT.shape)
	if dropout_str == 'origin':
		# MTM = M.T @ M * (sigma_yx_inv**2 / 2.)
		MTM = (M.T @ M + 1e-6*np.eye(K)) * (sigma_yx_inv ** 2 / 2.)
		obj_share += grb.quicksum([vx[i] * MTM[i, i] * vx[i] for i in range(K)])
		MTM *= 2
		obj_share += grb.quicksum([vx[i] * MTM[i, j] * vx[j] for i in range(K) for j in range(i+1, K)])
		del MTM
		YTM = YT @ M * (-sigma_yx_inv**2)
	elif dropout_str == 'pass': pass
	else: assert False

	# prior on X
	if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
		mu_x, sigma_x_inv = prior_x[1:]
		assert (sigma_x_inv > 0).all()
		t = sigma_x_inv ** 2 / 2.
		obj_share += grb.quicksum([t[i] * vx[i] * vx[i]	for i in range(K)])
		t *= - 2 * mu_x
		obj_share += grb.quicksum([t[i] * vx[i] 			for i in range(K)])
		obj_share += np.dot(mu_x**2, sigma_x_inv**2) / 2.
		del t
	elif prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
		lambda_x, = prior_x[1:]
		assert (lambda_x >= 0).all()
		obj_share += grb.quicksum([lambda_x[i] * vx[i] for i in range(K)])
	else:
		assert False

	for y, yTM, y_valid in zip(YT, YTM, YT_valid):
		obj = obj_share
		if dropout_str == 'pass':
			y = y[y_valid]
			Mi = M[y_valid]
			yTM = y @ Mi * (sigma_yx_inv**2 * -1)
			# MTM = Mi.T @ Mi * (sigma_yx_inv**2 / 2)
			MTM = (Mi.T @ Mi + 1e-6*np.eye(K)) * (sigma_yx_inv**2 / 2)
			obj = obj + grb.quicksum([MTM[i, i] * vx[i] * vx[i] for i in range(K)])
			MTM *= 2
			obj = obj + grb.quicksum([MTM[i, j] * vx[i] * vx[j] for i in range(K) for j in range(i+1, K)])
		obj = obj + grb.quicksum(yTM[i]*vx[i] for i in range(K)) + np.dot(y, y) * sigma_yx_inv / 2.
		m.setObjective(obj, grb.GRB.MINIMIZE)
		m.optimize()
		# print(m.ObjVal)
		res.append([vx[i].x for i in range(K)])

	del m

	res = np.array(res)
	return res

def estimateWeightsWithNeighbor(O, H, Theta, modelSpec, dropout_str, **kwargs):
	assert False
	K, YT, E, E_empty, X_sum2one, M_sum2one, pairwise_potential_str = O
	(XT, ) = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta
	N, G = YT.shape
	assert M.shape == (G, K)
	assert not E_empty
	assert sigma_yx_inv > 0
	if YT.size == 0: return np.zeros([0, K], dtype=np.float)

	print('constructing model ...', end='\t')
	time_start = timeit.default_timer()
	m = grb.Model('X w n')
	m.setParam('OutputFlag', False)
	# m.Params.Threads = nCPU
	m.Params.BarConvTol = 1e-12
	vXT = m.addVars(N, K, lb=0.)
	# if X_sum2one is True: m.addConstrs((vXT.sum(i,'*')==1 for i in range(N)))
	assert not X_sum2one

	obj = 0
	# emission y|x - quadratic term
	t = M.T @ M * sigma_yx_inv**2
	t[np.diag_indices(K)] /= 2
	t /= N*G
	# obj += grb.quicksum([t[i, j] * vXT[u, i] * vXT[u, j] for u in range(N) for i in range(K) for j in range(K)])
	obj = grb.quicksum([t[i, j] * vXT[u, i] * vXT[u, j] for u in range(N) for i in range(K) for j in range(i+1, K)])
	obj += grb.quicksum([t[i, i] * vXT[u, i] * vXT[u, i] for u in range(N) for i in range(K)])
	# emission y|x - linear term
	t = - YT @ M * sigma_yx_inv**2
	t /= N*G
	obj += grb.quicksum([t[u, i] * vXT[u, i] for u in range(N) for i in range(K)])
	# emission y|x - constant
	t = np.dot(YT.ravel(), YT.ravel()) * sigma_yx_inv**2 / 2
	t /= N*G
	obj += t
	# prior
	if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
		mu_x, sigma_x_inv = prior_x[1:]
		t = sigma_x_inv**2/2
		t /= N*G
		obj += grb.quicksum([t[i] * vXT[u, i] * vXT[u, i] for u in range(N) for i in range(K)])
		t = -sigma_x_inv**2 * mu_x
		t /= N*G
		obj += grb.quicksum([t[i] * vXT[u, i] for u in range(N) for i in range(K)])
		t = N/2 * np.dot(mu_x**2, sigma_x_inv**2)
		t /= N*G
		obj += t
		del mu_x, sigma_x_inv, t
	elif prior_x[0] in ['Exponential', 'Exponential shared']:
		lambda_x, = prior_x[1:]
		t = lambda_x / (N*G)
		obj += grb.quicksum([t[i] * vXT[u, i] for u in range(N) for i in range(K)])
		del lambda_x, t
	else:
		assert False
	# pairwise
	t = Sigma_x_inv / (N*G)
	obj += grb.quicksum([
		vXT[u, i] * t[i, j] * vXT[v, j]
		for u, eu in enumerate(E) for v in eu
		if v > u
		for i in range(K) for j in range(K)
	])
	t = -Sigma_x_inv @ delta_x / (N*G)
	obj += grb.quicksum([len(e) * t[i] * vXT[u, i] for u, e in enumerate(E) for i in range(K)])
	obj += delta_x @ Sigma_x_inv @ delta_x / (N*G) * sum(map(len, E))
	m.setObjective(obj, grb.GRB.MINIMIZE)
	print(timeit.default_timer() - time_start)
	print('solving model ...', end='\t')
	m.optimize()
	print(timeit.default_timer() - time_start)
	# print(m.ObjVal)
	XT = np.array([[vXT[i, j].x for j in range(K)] for i in range(N)])

	del m

	return XT

def solveQPbyGurobi(O, H, Theta, modelSpec, check_positive=True, gurobi_pars=None, reuse=False, **kwargs):
	assert False
	K, YT, E, E_empty, X_sum2one, M_sum2one, pairwise_potential_str = O
	assert pairwise_potential_str == 'normalized'
	XT, s = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta

	N, G = YT.shape
	assert M.shape == (G, K)
	assert not E_empty
	assert sigma_yx_inv > 0
	if YT.size == 0: return np.zeros([0, K], dtype=np.float)

	assert prior_x[0] in ['Exponential', 'Exponential shared']
	lambda1 = prior_x[1]
	lambda2 = None

	MTM = sigma_yx_inv**2 / 2. / (N*G) * (M.T @ M)
	YTM = YT @ M * (sigma_yx_inv**2 / 2. / (N*G))
	SyTy = np.dot(YT.ravel(), YT.ravel()) * sigma_yx_inv**2 / 2. / (N*G)
	Sigma_x_inv = Sigma_x_inv / (N*G)
	lambda1 = lambda1 / (N*G)
	if lambda2 is not None: lambda2 = lambda2 / (N*G)

	if gurobi_pars is None:
		m = grb.Model('qp')
		m.setParam('OutputFlag', False)
		# m.Params.Threads = nCPU
		m.Params.Threads = 1
		# m.Params.OptimalityTol = 1e-3
		m.Params.BarConvTol = 1e-12
		vXT = m.addVars(N, K, lb=0.)
		m.addConstrs((vXT.sum(i,'*')==1 for i in range(N)))
		dx = grb.quicksum([
			vXT[u,i] * Sigma_x_inv[i,j] * vXT[v,j]
			for u, eu in enumerate(E) for v in eu if v > u for i in range(K) for j in range(K)
		])
		gurobi_pars = (m, vXT, dx)
	else:
		(m, vXT, dx) = gurobi_pars
	if s is None:
		dy2 = grb.quicksum([vXT[u,i]*MTM[i,j]*vXT[u,j]			for u in range(N) for i in range(K) for j in range(K) ])
	else:
		dy2 = grb.quicksum([vXT[u,i]*MTM[i,j]*vXT[u,j]*s[u]**2	for u in range(N) for i in range(K) for j in range(K) ])
	if s is None:
		dy1 = grb.quicksum([-2*YTM[u,i]*vXT[u,i]		for u in range(N) for i in range(K) ])
	else:
		dy1 = grb.quicksum([-2*YTM[u,i]*vXT[u,i]*s[u]	for u in range(N) for i in range(K) ])
	if lambda1 is not None:
		if s is None:
			vx1 = grb.quicksum([lambda1[j]*vXT[i,j]		for i in range(N) for j in range(K)])
		else:
			vx1 = grb.quicksum([lambda1[j]*vXT[i,j]*s[i]	for i in range(N) for j in range(K)])
	else: vx1 = None
	if lambda2 is not None:
		if check_positive:
			ev = np.linalg.eig(MTM)[0]
			new_lambda2 = min(lambda2, ev.min()*sigma_yx_inv**2/2 * .95)
			if lambda2 == new_lambda2:
				print('w neighbor: min eigenvalue = {}, lambda2 = {}, new_lambda2 = {}'.format(ev.min()*sigma_yx_inv**2/2, lambda2, new_lambda2))
			else:
				print('GurobiWarning\tw neighbor: min eigenvalue = {}, lambda2 = {}, new_lambda2 = {}'.format(ev.min()*sigma_yx_inv**2/2, lambda2, new_lambda2))
				lambda2 = new_lambda2
		if s is None:
			vx2 = grb.quicksum([lambda2*vXT[i,j]*vXT[i,j] for i in range(N) for j in range(K)])
		else:
			vx2 = grb.quicksum([lambda2*vXT[i,j]*vXT[i,j]*s[i]**2 for i in range(N) for j in range(K)])
	else:
		vx2 = None
	assert vx2 is None
	kk = 1.
	kkk = .9
	while True:
		try:
			obj = dx*kk + dy2 + dy1 + SyTy + (vx1*kk if vx1 is not None else 0) + (vx2*kk if vx2 is not None else 0)
			m.setObjective(obj, grb.GRB.MINIMIZE)
			# time_start = timeit.default_timer()
			m.optimize()
			# print(timeit.default_timer() - time_start)
			# sys.exit()
			# print('Gurobi ObjVal = {:.8e} -> {:.8e}'.format(m.ObjVal, m.ObjVal*N*G))
			# sys.exit()
			# print(m.ObjVal + lambda1 * s.sum())
			break
		except Exception as e:
			print(f'{type(e).__name__}\t{e}\tw neighbor: min eigenvalue = {np.linalg.eig(MTM*sigma_yx_inv**2/2)[0].min():.2e}, kk = {kk:.2e}', end=';\t')
			kk *= kkk
			print(f'trying {kk:.2e}')

	ret = [np.array([[vXT[i,j].x for j in range(K)] for i in range(N)]), lambda2]
	if reuse: ret.append(gurobi_pars)

	del m

	return tuple(ret)


# def estimateWeightsByPytorch(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M, MTM, XT, YT, E, lambda1, lambda2):
# 	# dtype = torch.double
#
# 	assert False
#
# 	tSigma_x_inv_2 = torch.tensor(Sigma_x_inv_2, device=device, dtype=dtype)
# 	tSigma_x_inv = torch.tensor(Sigma_x_inv, device=device, dtype=dtype)
# 	tM = torch.tensor(M, device=device, dtype=dtype)
# 	tMTM = torch.tensor(MTM, device=device, dtype=dtype)
# 	tYT = torch.tensor(YT, device=device, dtype=dtype)
# 	vXT = torch.tensor(XT, device=device, dtype=dtype, requires_grad=False)
# 	tYTM = tYT @ tM
# 	tY2S = tYT.pow(2).sum()
# 	vs = vXT.sum(1)
# 	vXT /= vs[:, None]
#
# 	a = (vXT @ tMTM * vXT).sum(1) * sigma_yx_inv**2 / 2
# 	b = (tYT @ tM   * vXT).sum(1) * sigma_yx_inv**2 / 2
# 	if lambda1 != 0: b -= lambda1/2; b.clamp_(min=0)
# 	if lambda2 != 0: a.add_(lambda2, vXT.pow(2).sum(1))
# 	vs = b / (a + 1e-10)
# 	del a, b
#
# 	N, G = YT.shape
# 	K = M.shape[1]
#
# 	par_list = [vXT, vs]
# 	# optimizer = torch.optim.Adadelta(par_list, lr=1e-0)
# 	# optimizer = torch.optim.Adagrad(par_list, lr=1e-1)
# 	# optimizer = torch.optim.Adam(par_list, lr=1e-8, betas=(.9, .999))
# 	# optimizer = torch.optim.Adamax(par_list, lr=1e-2)
# 	# optimizer = torch.optim.ASGD(par_list, lr=1e-6)
# 	# optimizer = torch.optim.RMSprop(par_list, lr=1e-2)
# 	# optimizer = torch.optim.Rprop(par_list, lr=1e-3)
# 	# optimizer = torch.optim.SGD(par_list, lr=1e-6, momentum=0)
#
# 	lr_XT = 1e0
# 	lr_s = 0
#
# 	__t__, last_func, best_func, best_iter = None, np.nan, None, None
# 	time_start = timeit.default_timer()
# 	for __t__ in range(5):
#
# 		# print('stat of XT: shape = {}, min = {:.2e}, # <0 = {:.1f}, # zeros = {:.1f}, # <1e-2 = {:.1f}, # >1e-1 = {:.1f}'.format(
# 		# 	vXT.size(0),
# 		# 	vXT.min(),
# 		# 	float((vXT < 0).sum())/N,
# 		# 	float((vXT == 0).sum())/N,
# 		# 	float((vXT < 1e-2).sum())/N,
# 		# 	float((vXT > 1e-1).sum())/N,
# 		# ))
#
# 		vsXT = vs[:, None] * vXT
#
# 		func = (vsXT @ tMTM * vsXT).sum()
# 		vXT.grad = vs[:, None] * vsXT @ tMTM
# 		vs.grad = (vsXT @ tMTM * vXT).sum(1)
# 		func.sub_(2, torch.dot(tYTM.view(-1), vsXT.view(-1)))
# 		vXT.grad.addcmul_(-1, vs[:, None], tYTM)
# 		vs.grad.sub_((tYTM * vXT).sum(1))
# 		func += tY2S
#
# 		for Ei, vXTi, vXTgi in zip(E, vXT, vXT.grad):
# 			if Ei:
# 				tXNiT = vXT.index_select(0, torch.tensor(Ei, dtype=torch.long, device=device))
# 				dx = vXTi[None] - tXNiT
# 				func += (dx @ tSigma_x_inv_2).pow(2).sum()
# 				vXTgi.addmv_(mat=tSigma_x_inv, vec=dx.sum(0))
# 				del dx, tXNiT
#
# 		func.mul_(sigma_yx_inv ** 2 / 2)
# 		vXT.grad.mul_(sigma_yx_inv ** 2)
# 		vs.grad.mul_(sigma_yx_inv ** 2)
#
# 		if lambda1 != 0:
# 			func += lambda1 * vs.sum()
# 			vs.grad += lambda1 * K
# 		if lambda2 != 0:
# 			func += lambda2 * vsXT.pow(2).sum()
# 			vXT.grad.addcmul_(lambda2*2, vs[:, None], vsXT)
# 			vs.grad.addcmul_(lambda2*2, vs, vXT.pow(2).sum(1))
#
# 		del vsXT
#
# 		func.mul_(1. / G)
# 		vXT.grad.mul_(1. / G)
# 		vs.grad.mul_(1. / G)
#
# 		# print('stat of XT: shape = {}, # <0 = {:.1f}, # zeros = {:.1f}, # <1e-2 = {:.1f}, # >1e-1 = {:.1f}'.format(
# 		# 	vXT.size(0),
# 		# 	float((vXT < 0).sum())/N,
# 		# 	float((vXT == 0).sum())/N,
# 		# 	float((vXT < 1e-2).sum())/N,
# 		# 	float((vXT > 1e-1).sum())/N,
# 		# ))
#
# 		# print('stat of s: min = {:.2e}, max = {:2e}, #d>1.3 = {}, #d<0.7 = {}'.format(
# 		# 	vs.min(),
# 		# 	vs.max(),
# 		# 	(vs>1.3).sum(),
# 		# 	(vs<0.7).sum(),
# 		# ))
#
# 		print('=='*20)
#
# 		ii = 117
# 		print('vXT', vXT[ii])
# 		print('grad', vXT.grad[ii])
#
# 		index = (vXT == 0) & (vXT.grad > 0)
# 		dof = K - index.sum(1, keepdim=True).type(dtype)
# 		print('dof', dof[ii])
# 		print('index', index[ii])
# 		print('index vXT', vXT[ii] == 0)
# 		print('index grad', vXT.grad[ii] > 0)
# 		vXT.grad[index] = 0
# 		print('grad', vXT.grad[ii])
# 		vXT.grad -= vXT.grad.sum(1, keepdim=True) / dof
# 		print('grad', vXT.grad[ii])
# 		# del dof
# 		vXT.grad[index] = 0
# 		print('grad', vXT.grad[ii])
# 		# del index
#
# 		vs.grad[(vs == 0) & (vs.grad > 0)] = 0
#
# 		stop_flag = vXT.grad.abs().max() < 1e-5 and vs.grad.abs().max() < 1e-5 and func > last_func - 1e-5
# 		stop_flag |= best_func is not None and best_func == func and __t__ > best_iter + 20
# 		if __t__ % 1 == 0 or stop_flag:
# 			print('func_grad at iter {} = {}\t{}\t{}'.format(
# 				__t__, func,
# 				vXT.grad.abs().max().cpu().data,
# 				vs.grad.abs().max().cpu().data,
# 			))
# 			if stop_flag: break
#
# 		# optimizer.step()
#
# 		print('dof', dof[ii])
# 		print('index', index[ii])
# 		print('index vXT', vXT[ii] == 0)
# 		print('index grad', vXT.grad[ii] > 0)
# 		print('vXT', vXT[ii])
# 		print('grad', vXT.grad[ii])
# 		max_lr = vXT / vXT.grad
# 		print('lr', max_lr[ii])
# 		max_lr[vXT.grad <= 0] = lr_XT
# 		print('lr', max_lr[ii])
# 		max_lr = max_lr.min(1, keepdim=True)[0]
# 		print('lr', max_lr[ii])
# 		max_lr.clamp_(max=lr_XT)
# 		print('lr', max_lr[ii])
# 		d = max_lr * vXT.grad
# 		print('d', d[ii])
# 		print('grad', vXT.grad.abs().max(0))
# 		print('d', (d*1e10).abs().max(0))
# 		vXT.addcmul_(-1, max_lr, vXT.grad)
# 		print('vXT', vXT[ii])
# 		assert max_lr.min() >= 0
# 		del max_lr
# 		assert vXT.min() > -1e-10
# 		assert (vXT.sum(1) - 1.).abs().max() < 1e-10
# 		vXT.clamp_(min=0)
# 		vXT[vXT < 1e-14] = 0
# 		vXT /= vXT.sum(1, keepdim=True)
#
# 		max_lr = vs / vs.grad
# 		max_lr[vs.grad <= 0] = lr_s
# 		max_lr.clamp_(max=lr_s)
# 		vs.addcmul_(-1, max_lr, vs.grad)
# 		assert max_lr.min() >= 0
# 		del max_lr
# 		assert vs.min() > -1e-10
# 		vs.clamp_(min=0)
#
# 		if best_func is None or best_func > func:
# 			best_func, best_iter = func, __t__
#
# 	XT = vXT.cpu().data.numpy()
# 	s = vs.cpu().data.numpy()
# 	print(XT[:5])
# 	print(s[:5])
# 	sys.exit()
#
# 	vXT *= vs[:, None]
#
# 	return vXT.cpu().data.numpy()


# def estimateWeightsByPytorch2(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M, MTM, XT, YT, E, lambda1, lambda2):
# 	a = np.matmul((XT @ MTM)[:, None, :], XT[:, :, None]).flatten() * sigma_yx_inv ** 2 / 2
# 	b = np.matmul((YT @ M)[:, None, :], XT[:, :, None]).flatten() * sigma_yx_inv ** 2 / 2
# 	if lambda1 != 0: b -= lambda1/2; np.maximum(b, 0, out=b)
# 	if lambda2 != 0: a += lambda2 * np.matmul(XT[:, None, :], XT[:, :, None]).flatten()
# 	s = b / a
#
# 	tsigma_yx_inv = torch.tensor(sigma_yx_inv, device=device, dtype=dtype)
# 	tSigma_x_inv_2 = torch.tensor(Sigma_x_inv_2, device=device, dtype=dtype)
# 	tSigma_x_inv = torch.tensor(Sigma_x_inv, device=device, dtype=dtype)
# 	tM = torch.tensor(M, device=device, dtype=dtype)
# 	tMTM = torch.tensor(MTM, device=device, dtype=dtype)
# 	tYT = torch.tensor(YT, device=device, dtype=dtype)
# 	vXT = Variable(torch.tensor(XT, device=device, dtype=dtype))
# 	tYTM = tYT @ tM
# 	tY2S = tYT.pow(2).sum()
# 	vs = vXT.sum(1)
# 	vXT /= vs[:, None]
#
# 	# a = torch.matmul((vXT @ tMTM)[:, None, :], vXT[:, :, None]).view(-1) * sigma_yx_inv ** 2 / 2
# 	# b = torch.matmul((tYT @ tM)[:, None, :], vXT[:, :, None]).view(-1) * sigma_yx_inv ** 2 / 2
# 	# if lambda1 != 0: b -= lambda1/2; torch.max(b, 0, out=b)
# 	# if lambda2 != 0: a += lambda2 * torch.bmm(vXT[:, None, :], vXT[:, :, None]).view(-1)
# 	# vs = b / a
#
# 	N, G = YT.shape
# 	K = M.shape[1]
#
# 	par_list = [vXT, vs]
# 	scheduler = None
# 	# optimizer = torch.optim.Adadelta(par_list, lr=1e-3, rho=.1)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.4)
# 	# optimizer = torch.optim.Adagrad(par_list, lr=1e-2)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.97)
# 	optimizer = torch.optim.Adam(par_list, lr=1e-4)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=.9)
# 	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.95)
# 	# optimizer = torch.optim.Adamax(par_list, lr=1e-4)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.95)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,], gamma=.99)
# 	# optimizer = torch.optim.ASGD(par_list, lr=1e-8)
# 	# optimizer = torch.optim.RMSprop(par_list, lr=1e-6)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 400,], gamma=.3)
# 	# optimizer = torch.optim.Rprop(par_list, lr=1e-2)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 400,], gamma=.3)
# 	# optimizer = torch.optim.SGD(par_list, lr=1e-7, momentum=0)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=1e-1)
#
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
#
# 	fs = []
# 	gxmax_ = []
# 	gxrms_ = []
# 	gxmax = []
# 	gxrms = []
#
# 	__t__, last_func, best_func, best_param, best_iter = None, None, None, None, None
#
# 	time_start = timeit.default_timer()
#
# 	for __t__ in range(501):
# 		# time_start = timeit.default_timer()
# 		vXT.grad = vs.pow(2).view(N, 1) * vXT @ tMTM
# 		vs.grad = ((vXT@tMTM)[:, None, :] @ vXT[:, :, None]).view(-1) * vs
# 		vXT.grad -= vs.view(N, 1) * tYTM
# 		vs.grad[:, None, None].baddbmm_(tYTM[:, None, :], -vXT[:, :, None])
# 		vXT.grad *= tsigma_yx_inv ** 2
# 		vs.grad *= tsigma_yx_inv ** 2
# 		# print(timeit.default_timer() - time_start)
#
# 		for i, Ei in enumerate(E):
# 			if Ei:
# 				dx = vXT.index_select(0, torch.tensor(Ei, dtype=torch.long, device=device))
# 				dx.sub_(vXT[i][None])
# 				vXT.grad[i, :] -= (dx @ tSigma_x_inv).sum(0)*2
# 				del dx
# 		# print(timeit.default_timer() - time_start)
# 		# if __t__ == 10: sys.exit()
#
# 		gxmax_.append(vXT.grad.abs().max().cpu().data)
# 		gxrms_.append(vXT.grad.pow(2).mean().cpu().data)
#
# 		if lambda1 != 0:
# 			vs.grad += lambda1 * K
# 		if lambda2 != 0:
# 			vXT.grad += 2 * lambda2 * vs.pow(2)[:, None] * vXT
# 			vs.grad += 2 * lambda2 * vs * vXT.pow(2).sum(1)
#
# 		kk_sum2oneX = 1e4
# 		kk_nonnegX = 1e6
# 		kk_nonnegs = 1e4
#
# 		vXT.grad += (vXT.sum(1, keepdim=True) - 1) * kk_sum2oneX
# 		vXT.grad += vXT.clamp(max=0) * kk_nonnegX
# 		vs.grad += vs.clamp(max=0) * kk_nonnegs
#
# 		gxmax.append(vXT.grad.abs().max().cpu().data)
# 		gxrms.append(vXT.grad.pow(2).mean().cpu().data)
#
# 		stop_flag = False
# 		if __t__ % 1 == 0:
# 			func_obj = torch.bmm(((vs.view(N, 1) * vXT) @ tMTM)[:, None, :], (vs.view(N, 1) * vXT)[:, :, None]).sum()
# 			func_obj -= 2*torch.dot(tYTM.view(-1), (vs.view(N, 1)*vXT).view(-1))
# 			func_obj += tY2S
# 			func_obj *= tsigma_yx_inv ** 2 / 2
# 			for i, Ei in enumerate(E):
# 				if Ei:
# 					tXNiT = vXT.index_select(0, torch.tensor(Ei, dtype=torch.long, device=device))
# 					dx = vXT[i][None] - tXNiT
# 					func_obj += (dx @ tSigma_x_inv_2).pow(2).sum()
# 			func_reg = torch.tensor(0, dtype=dtype, device=device)
# 			if lambda1 != 0: func_reg += lambda1 * vs.sum()
# 			if lambda2 != 0: func_reg += lambda2 * (vs.view(N, 1)*vXT).pow(2).sum()
#
# 			func_constraints = (vXT.sum(1)-1).pow(2).sum() * kk_sum2oneX / 2
# 			func_constraints += vXT.clamp(max=0).pow(2).sum() * kk_nonnegX / 2
# 			func_constraints += vs.clamp(max=0).pow(2).sum() * kk_nonnegs / 2
#
# 			func = func_obj + func_reg + func_constraints
# 			fs.append(func.cpu().data.numpy())
# 			last_func = func
# 			if best_func is None or best_func > func:
# 				best_func, best_param, best_iter = func, (vXT.cpu().clone(), vs.cpu().clone()), __t__
#
# 			"""
# 			print('> func_grad at iter {} = {:.2f} + {:.2f} + {:.2f} = {:.2f},\t{:.2f}\t{:.2f}'.format(
# 				__t__, func_obj, func_reg, func_constraints, func,
# 				vXT.grad.abs().max().cpu().data,
# 				vs.grad.abs().max().cpu().data,
# 			))
#
# 			print('dev: non-neg X = {:.2e},\tnon-neg s = {:.2e},\tsum2one = {:.2e}'.format(
# 				vXT.min().cpu().data,
# 				vs.min().cpu().data,
# 				(vXT.sum(1) - 1).abs().max(),
# 			))
#
# 			print('stat of XT: #<0 = {:.1f}, #<1e-2 = {:.1f}, #>1e-1 = {:.1f}'.format(
# 				float((vXT <= 0).sum())/N,
# 				float((vXT < 1e-2).sum())/N,
# 				float((vXT > 1e-1).sum())/N,
# 			))
#
# 			print('stat of s: min = {:.2e}, max = {:2e}, #>1.3 = {}, #<0.7 = {}'.format(
# 				vs.min(),
# 				vs.max(),
# 				(vs>1.3).sum(),
# 				(vs<0.7).sum(),
# 			))
# 			# """
#
# 			stop_flag = vXT.grad.abs().max() < 1e-1 and vs.grad.abs().max() < 1e-3 and (last_func is not None and func > last_func - 1e-5)
# 			# stop_flag |= best_func is not None and best_func == func and __t__ > best_iter + 20
#
# 		if stop_flag: break
# 		if scheduler: scheduler.step()
# 		optimizer.step()
# 	print(timeit.default_timer() - time_start)
# 	# from matplotlib import pyplot as plt
# 	# plt.plot(np.arange(len(fs)), fs)
# 	# plt.show()
# 	# plt.plot(np.arange(len(gxmax_)), np.log10(gxmax_))
# 	# plt.plot(np.arange(len(gxrms_)), np.log(gxrms_)/2)
# 	# plt.show()
# 	# plt.plot(np.arange(len(gxmax)), np.log10(gxmax))
# 	# plt.plot(np.arange(len(gxrms)), np.log10(gxrms)/2)
# 	# plt.show()
# 	# sys.exit()
# 	vXT, vs = best_param
# 	vXT.clamp_(min=0)
# 	vXT *= vs[:, None]
# 	return vXT.cpu().data.numpy()

# def estimateWeightsByPytorch3(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M, MTM, XT, YT, E, lambda1, lambda2):
# 	tsigma_yx_inv = torch.tensor(sigma_yx_inv, device=device, dtype=dtype)
# 	tSigma_x_inv_2 = torch.tensor(Sigma_x_inv_2, device=device, dtype=dtype)
# 	tSigma_x_inv = torch.tensor(Sigma_x_inv, device=device, dtype=dtype)
# 	tM = torch.tensor(M, device=device, dtype=dtype)
# 	tMTM = torch.tensor(MTM, device=device, dtype=dtype)
# 	tYT = torch.tensor(YT, device=device, dtype=dtype)
# 	vXT = Variable(torch.tensor(XT, device=device, dtype=dtype))
# 	tYTM = tYT @ tM
# 	tY2S = tYT.pow(2).sum()
# 	vXT /= vXT.sum(1, keepdim=True)
#
# 	N, G = YT.shape
# 	K = M.shape[1]
#
# 	par_list = [vXT]
# 	scheduler = None
# 	# optimizer = torch.optim.Adadelta(par_list, lr=1e-3, rho=.1)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.4)
# 	# optimizer = torch.optim.Adagrad(par_list, lr=1e-2)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.97)
# 	optimizer = torch.optim.Adam(par_list, lr=1e-5)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,], gamma=.99)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.97)
# 	# optimizer = torch.optim.Adamax(par_list, lr=1e-4)
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.95)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,], gamma=.99)
# 	# optimizer = torch.optim.ASGD(par_list, lr=1e-8)
# 	# optimizer = torch.optim.RMSprop(par_list, lr=1e-6)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 400,], gamma=.3)
# 	# optimizer = torch.optim.Rprop(par_list, lr=1e-2)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 400,], gamma=.3)
# 	# optimizer = torch.optim.SGD(par_list, lr=1e-7, momentum=0)
# 	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=1e-1)
#
# 	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=.1)
#
# 	fs = []
# 	gxmax_ = []
# 	gxrms_ = []
# 	gxmax = []
# 	gxrms = []
#
# 	__t__, last_func, best_func, best_param, best_iter = None, None, None, None, None
# 	for __t__ in range(1001):
# 		# time_start = timeit.default_timer()
# 		t = vXT @ tMTM
# 		vXT.grad = t.clone()
# 		tYTMX = (tYTM * vXT).sum(1, keepdim=True)
# 		t.mul_(vXT)
# 		t = t.sum(1, keepdim=True)
# 		vXT.grad.mul_(tYTMX.pow(2) / t)
# 		vXT.grad.sub_(tYTM * tYTMX)
# 		vXT.grad.mul_(tsigma_yx_inv**2 / t)
# 		# print(timeit.default_timer() - time_start)
#
# 		for i, Ei in enumerate(E):
# 			if Ei:
# 				dx = vXT.index_select(0, torch.tensor(Ei, dtype=torch.long, device=device))
# 				dx.sub_(vXT[i][None])
# 				vXT.grad[i, :] -= (dx @ tSigma_x_inv).sum(0)*2
# 				del dx
# 		# print(timeit.default_timer() - time_start)
# 		# if __t__ == 10: sys.exit()
#
# 		gxmax_.append(vXT.grad.abs().max().cpu().data)
# 		gxrms_.append(vXT.grad.pow(2).mean().cpu().data)
#
# 		kk_sum2oneX = 1e4
# 		kk_nonnegX = 1e6
#
# 		vXT.grad += (vXT.sum(1, keepdim=True) - 1) * kk_sum2oneX
# 		vXT.grad += vXT.clamp(max=0) * kk_nonnegX
#
# 		gxmax.append(vXT.grad.abs().max().cpu().data)
# 		gxrms.append(vXT.grad.pow(2).mean().cpu().data)
#
# 		stop_flag = False
# 		if __t__ % 1 == 0:
# 			vs = (tYTM*vXT).sum(1) / ((vXT @ tMTM) * vXT).sum(1)
# 			func_obj = torch.bmm(((vs.view(N, 1) * vXT) @ tMTM)[:, None, :], (vs.view(N, 1) * vXT)[:, :, None]).sum()
# 			func_obj -= 2*torch.dot(tYTM.view(-1), (vs.view(N, 1)*vXT).view(-1))
# 			func_obj += tY2S
# 			func_obj *= tsigma_yx_inv ** 2 / 2
# 			for i, Ei in enumerate(E):
# 				if Ei:
# 					tXNiT = vXT.index_select(0, torch.tensor(Ei, dtype=torch.long, device=device))
# 					dx = vXT[i][None] - tXNiT
# 					func_obj += (dx @ tSigma_x_inv_2).pow(2).sum()
# 			func_reg = torch.tensor(0, dtype=dtype, device=device)
# 			if lambda1 != 0: func_reg += lambda1 * vs.sum()
# 			if lambda2 != 0: func_reg += lambda2 * (vs.view(N, 1)*vXT).pow(2).sum()
#
# 			func_constraints = (vXT.sum(1)-1).pow(2).sum() * kk_sum2oneX / 2
# 			func_constraints += vXT.clamp(max=0).pow(2).sum() * kk_nonnegX / 2
# 			# func_constraints += vs.clamp(max=0).pow(2).sum() * kk / 2
#
# 			func = func_obj + func_reg + func_constraints
# 			fs.append(func.cpu().data.numpy())
# 			last_func = func
# 			if best_func is None or best_func > func:
# 				best_func, best_param, best_iter = func, (vXT.cpu().clone(), vs.cpu().clone()), __t__
#
# 			"""
# 			print('> func_grad at iter {} = {:.2f} + {:.2f} + {:.2f} = {:.2f},\t{:.2f}'.format(
# 				__t__, func_obj, func_reg, func_constraints, func,
# 				vXT.grad.abs().max().cpu().data,
# 				# vs.grad.abs().max().cpu().data,
# 			))
#
# 			print('dev: non-neg X = {:.2e},\tnon-neg s = {:.2e},\tsum2one = {:.2e}'.format(
# 				vXT.min().cpu().data,
# 				vs.min().cpu().data,
# 				(vXT.sum(1) - 1).abs().max(),
# 			))
#
# 			print('stat of XT: #<0 = {:.1f}, #<1e-2 = {:.1f}, #>1e-1 = {:.1f}'.format(
# 				float((vXT <= 0).sum())/N,
# 				float((vXT < 1e-2).sum())/N,
# 				float((vXT > 1e-1).sum())/N,
# 			))
#
# 			print('stat of s: min = {:.2e}, max = {:2e}, #>1.3 = {}, #<0.7 = {}'.format(
# 				vs.min(),
# 				vs.max(),
# 				(vs>1.3).sum(),
# 				(vs<0.7).sum(),
# 			))
# 			# """
#
# 			stop_flag = vXT.grad.abs().max() < 1e-4 and vs.grad.abs().max() < 1e-3 and (last_func is not None and func > last_func - 1e-8)
# 			# stop_flag |= best_func is not None and best_func == func and __t__ > best_iter + 20
#
# 		if stop_flag: break
# 		if scheduler: scheduler.step()
# 		optimizer.step()
# 	# from matplotlib import pyplot as plt
# 	# plt.plot(np.arange(len(fs)), np.log(fs))
# 	# plt.show()
# 	# plt.plot(np.arange(len(gxmax_)), np.log(gxmax_))
# 	# plt.plot(np.arange(len(gxrms_)), np.log(gxrms_))
# 	# plt.show()
# 	# plt.plot(np.arange(len(gxmax)), np.log(gxmax))
# 	# plt.plot(np.arange(len(gxrms)), np.log(gxrms))
# 	# plt.show()
# 	# sys.exit()
# 	vXT, vs = best_param
# 	vXT.clamp_(min=0)
# 	vXT *= vs[:, None]
# 	return vXT.cpu().data.numpy()

def estimateWeightsICM(O, H, Theta, modelSpec, iexpr, pairwise_potential_str, dropout_str, **kwargs):
	K, YT, YT_valid, E, E_empty = O
	(XT,) = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta

	N, G = YT.shape
	K = M.shape[1]
	assert M.shape[0] == G
	MTM = None
	YTM = None
	size = YT_valid.sum()
	if dropout_str == 'origin':
		MTM = M.T @ M * sigma_yx_inv**2 / 2
		YTM = YT @ M * sigma_yx_inv**2 / 2
	elif dropout_str == 'pass':
		YTM = np.empty_like(XT)
		for yTM, y, y_valid in zip(YTM, YT, YT_valid):
			y = y[y_valid]
			Mi = M[y_valid]
			yTM[:] = y @ Mi * (sigma_yx_inv**2 / 2)
	else:
		assert False

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
		if dropout_str == 'origin':
			t = t.ravel()
		elif dropout_str == 'pass':
			t = t[YT_valid]
		else:
			assert False
		func += np.dot(t, t) * sigma_yx_inv**2 / 2
		if pairwise_potential_str == 'normalized':
			for e, z in zip(E, ZT):
				func += z @ Sigma_x_inv @ ZT[e].sum(0) / 2
		else:
			assert False
		if prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
			lambda_x, = prior_x[1:]
			func += lambda_x @ (S * ZT).sum(0)
			del lambda_x
		else:
			assert False
		func /= size
		return func

	__t__, last_func = -1, calcObj(S, ZT)
	best_func, best_iter = last_func, __t__

	for __t__ in range(max_iter):
		last_ZT = np.copy(ZT)
		last_S = np.copy(S)

		if pairwise_potential_str == 'normalized':
			for i, (e, y, yTM, y_valid, z, s) in enumerate(zip(E, YT, YTM, YT_valid, ZT, S)):
				eta = ZT[e].sum(0) @ Sigma_x_inv
				if dropout_str == 'pass':
					MTM = M[y_valid].T @ M[y_valid] * (sigma_yx_inv**2 / 2)
					y = y[y_valid]
					yTM = y@M[y_valid] * (sigma_yx_inv**2 / 2)
				for ___t___ in range(max_iter_individual):
					stop_flag = True

					a = z @ MTM @ z
					b = yTM @ z
					if prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
						lambda_x, = prior_x[1:]
						b -= lambda_x @ z / 2
						del lambda_x
					else:
						assert False
					b = np.maximum(b, 0)
					s_new = b / a
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
						assert False
					obj += grb.quicksum([vz[i] * t[i] for i in range(K)])
					obj += y @ y * sigma_yx_inv**2 / 2
					m.setObjective(obj, grb.GRB.MINIMIZE)
					m.optimize()
					z_new = np.array([vz[i].x for i in range(K)])
					dz = z_new - z
					stop_flag &= np.abs(dz).max() < 1e-3
					z = z_new

					if not stop_flag and ___t___ == max_iter_individual-1:
						print(f'Warning cell {i} in the {iexpr}-th expr did not converge in {max_iter_individual} iterations;\ts = {s:.2e}, ds = {ds:.2e}, max dz = {np.abs(dz).max():.2e}')

					if stop_flag:
						break

				ZT[i] = z
				S[i] = s
		else:
			assert False

		stop_flag = True

		dZT = ZT - last_ZT
		dS = S - last_S
		stop_flag &= np.abs(dZT).max() < 1e-2
		stop_flag &= np.abs(dS / (S + 1e-15)).max() < 1e-2

		func = calcObj(S, ZT)

		stop_flag &= func > last_func - 1e-4

		force_show_flag = False
		# force_show_flag |= np.abs(dZT).max() > 1-1e-5

		if __t__ % 5 == 0 or stop_flag or force_show_flag:
			print(f'>{iexpr} func at iter {__t__} = {func:.2e},\tdiff = {np.abs(dZT).max():.2e}\t{np.abs(dS).max():.2e}\t{func - last_func:.2e}')

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
			best_func, best_iter = func, __t__

		if stop_flag:
			break

	del m

	XT = np.maximum(S, 1e-15) * ZT
	return XT


def estimateWeightsAlternatively(O, H, Theta):
	assert False
	K, YT, E, E_empty, X_sum2one, M_sum2one, pairwise_potential_str = O
	(XT,) = H
	M, sigma_yx_inv, Sigma_x_inv, delta_x, prior_x = Theta

	N, G = YT.shape
	K = M.shape[1]
	assert M.shape[0] == G
	MTM = M.T @ M

	s = XT.sum(1)
	XT /= s.reshape(N, 1) + 1e-10
	c = np.dot(YT.ravel(), YT.ravel())*sigma_yx_inv**2/2

	gurobi_pars = None

	__t__, last_func, best_func, best_iter = None, np.nan, np.nan, -1
	stop_flag = False
	last_XT = np.nan
	def ff():
		XX = XT*s[:, None]
		ss = XT.sum(1)
		# print(ss.min(), ss.max())
		f = 0
		t = (YT - XX@M.T).ravel()
		f += np.dot(t, t) * sigma_yx_inv**2 / 2
		# print(f)
		t = 0
		for i, ei in enumerate(E):
			t += XT[i] @ Sigma_x_inv @ XT[ei].sum(0)
		tt = np.sum([
			XT[u, i] * Sigma_x_inv[i, j] * XT[v, j]
			for u, eu in enumerate(E) for v in eu for i in range(K) for j in range(K)
		])
		if not np.abs(t-tt) < 1e-5: print('dx', t, tt)
		assert np.abs(t-tt) < 1e-5
		f += t
		if prior_x[0] in ['Exponential', 'Exponential shared']:
			lambda_x, = prior_x[1:]
			f += lambda_x @ XX.sum(0)
			del lambda_x
		elif prior_x[0] == 'Gaussian' or prior_x[0] == 'Truncated Gaussian':
			mu_x, sigma_x_inv = prior_x[1:]
			del mu_x, sigma_x_inv
			assert False
		else:
			assert False
		f /= N*G
		return f
	for __t__ in range(10):
		last_last_XT = last_XT
		last_XT = XT
		last_s = s

		fff = ff()

		# XT, lambda2, gurobi_pars = solveQPbyGurobi(sigma_yx_inv, Sigma_x_inv, M, XT, s, YT, E, lambda1, lambda2, gurobi_pars=gurobi_pars, reuse=True)
		XT, lambda2, gurobi_pars = solveQPbyGurobi(O, (XT, s), Theta, gurobi_pars=gurobi_pars, reuse=True)
		# for numerical stability
		XT = np.maximum(XT, 0)
		XT[XT < 1e-10] = 0
		XT /= XT.sum(1, keepdims=True)

		ffff = ff()
		print(fff, ffff)
		# if fff < ffff + 1e-8:
		# 	XT = last_XT
		# 	break
		if not fff > ffff:
			print(fff, ffff)
			if __t__ == 0:
				XT = last_XT
				# break
		# assert fff > ffff - 1e-6

		a = (XT @ MTM * XT).sum(1) * sigma_yx_inv ** 2 / 2
		b = (YT @ M * XT).sum(1) * sigma_yx_inv ** 2 / 2
		func = c - (b**2/a).sum()
		if prior_x[0] == 'Exponential shared':
			lambda_x, = prior_x[1:]
			b -= lambda_x.mean()/2
			b = np.maximum(b, 0)
			del lambda_x
		elif prior_x[0] == 'Exponential':
			lambda_x, = prior_x[1:]
			b -= XT @ lambda_x / 2
			b = np.maximum(b, 0)
			del lambda_x
		elif prior_x[0] == 'Gaussian' or prior_x[0] == 'Truncated Gaussian':
			mu_x, sigma_x_inv = prior_x[1:]
			del mu_x, sigma_x_inv
			assert False
		else:
			assert False
		# if lambda1 != 0: b -= lambda1/2; np.maximum(b, 0, out=b)
		# if lambda2 != 0: a += (XT**2).sum(1) * lambda2
		s = b/a
		if prior_x[0] == 'Exponential shared':
			lambda_x, = prior_x[1:]
			func += lambda_x[0] * s.sum()
			del lambda_x
		elif prior_x[0] == 'Exponential':
			lambda_x, = prior_x[1:]
			func += s @ XT @ lambda_x
			del lambda_x

		for i, ei in enumerate(E):
			func += XT[i] @ Sigma_x_inv @ XT[ei].sum(0)

		func /= N*G
		# func_regularization = lambda1 * s.sum() + lambda2 * (s**2 * (XT**2).sum(1)).sum()
		# func_regularization /= N*G
		# func_obj = func + func_regularization

		dXT = np.abs(last_XT - XT).max()
		ds = np.abs(last_s - s).max()

		stop_flag = dXT < 1e-3 and ds < 1e-3
		stop_flag |= func > last_func - 1e-6
		if __t__ % 1 == 0 or stop_flag:
			print(__t__, func, dXT, ds, last_func)
			print(f'> func at iter {__t__} = {func:.2e},\tdiff = {dXT:.2e}\t{ds:.2e}\t{func-last_func:.2e}')

			print(
				f'stat of XT: '
				f'#<0 = {(XT < 0).sum().astype(np.float)/N:.1f}, '
				f'#=0 = {(XT == 0).sum().astype(np.float)/N:.1f}, '
				f'#<1e-10 = {(XT < 1e-10).sum().astype(np.float)/N:.1f}, '
				f'#<1e-5 = {(XT < 1e-5).sum().astype(np.float)/N:.1f}, '
				f'#<1e-2 = {(XT < 1e-2).sum().astype(np.float)/N:.1f}, '
				f'#>1e-1 = {(XT > 1e-1).sum().astype(np.float)/N:.1f}'
			)

			print(
				f'stat of s: '
				f'#0 = {(s == 0).sum()}, '
				f'min = {s.min():.1e}, '
				f'max = {s.max():.1e}'
			)

			sys.stdout.flush()

			if stop_flag: break
		last_func = func
		if not best_func <= func:
			best_func_obj, best_iter = func, __t__

	# print(ff())
	# print(XT[:5])
	# print(s[:5])

	XT *= s[:, None]

	# if not stop_flag:
	# XT = estimateWeightsByPytorch(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M, MTM, XT, YT, E, lambda1, lambda2)

	return XT

def estimateWeight(O, H, Theta, modelSpec, i, X_sum2one, pairwise_potential_str, **kwargs):
	time_start = timeit.default_timer()
	K, YT, YT_valid, E, E_empty = O
	for __ in range(2):
		assert __ < 1
		if YT.size == 0: XT = np.zeros([0, K]); break
		if E_empty: XT = estimateWeightsWithoutNeighbor(O, H, Theta, modelSpec, i, **modelSpec); break
		if X_sum2one: assert False
		if pairwise_potential_str == 'linear' or pairwise_potential_str == 'linear w/ shift':
			raise NotImplementedError
			XT = estimateWeightsWithNeighbor(O, H, Theta, modelSpec, i, **modelSpec)
			break
		elif pairwise_potential_str == 'normalized':
			# XT = estimateWeightsAlternatively(O, H, Theta)
			XT = estimateWeightsICM(O, H, Theta, modelSpec, i, **modelSpec)
			# XT = estimateWeightsByPytorch(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M[:G], MTM, XT, YT, E, lambda1, lambda2)
			# XT = estimateWeightsByPytorch2(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M[:G], MTM, XT, YT, E, lambda1, lambda2)
			# XT = estimateWeightsByPytorch3(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M[:G], MTM, XT, YT, E, lambda1, lambda2)
			# XT = estimateWeightsAlternatively(sigma_yx_inv, Sigma_x_inv_2, Sigma_x_inv, M[:G], MTM, XT, YT, E, lambda1, lambda2)
			break
		else:
			raise NotImplementedError
	print(f'Estimating weights {i} partial ends in {timeit.default_timer() - time_start}')
	sys.stdout.flush()
	return XT

def estimateWeights(O, H, Theta, modelSpec, X_sum2one, pairwise_potential_str, **kwargs):
	print('Estimating weights')
	sys.stdout.flush()
	K, YTs, YT_valids, Es, Es_empty, betas = O
	(XTs, ) = H
	M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs = Theta
	time_start_all = timeit.default_timer()

	assert M.shape[1] == K
	rXTs = []
	pool = Pool(min(8, len(YTs)))

	for i, (YT, YT_valid, E, E_empty, XT, sigma_yx_inv, prior_x) in enumerate(zip(YTs, YT_valids, Es, Es_empty, XTs, sigma_yx_invs, prior_xs)):
		# time_start = timeit.default_timer()
		N, G = YT.shape
		o = (K, YT, YT_valid, E, E_empty)
		h = (XT, )
		theta = (M[:G], sigma_yx_inv, Sigma_x_inv, delta_x, prior_x)
		rXTs.append(pool.apply_async(estimateWeight, args=(o, h, theta, modelSpec, i), kwds=modelSpec))

	rXTs = [_.get(1e9) if isinstance(_, multiprocessing.pool.ApplyResult) else _ for _ in rXTs]
	pool.close()

	print('Estimating weights ends in {}'.format(timeit.default_timer() - time_start_all))
	sys.stdout.flush()

	return (rXTs, )
