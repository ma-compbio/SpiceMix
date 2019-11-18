import numpy as np
import sklearn.cluster, sklearn.covariance
from util import zipTensors, unzipTensors, PyTorchDevice as device, PyTorchDType as dtype
import sys, itertools, timeit, resource, gc, pickle
import gurobipy as grb
from multiprocessing import Pool

def initializeMXTsByPartialNMF(YTs, YT_valids, M, betas, prior_x_strs, sigma_yx_inv_str, dropout_str, X_sum2one, M_sum2one, init_NMF_iter, **kwargs):
	Ns, Gs = map(np.array, zip(*[YT.shape for YT in YTs]))
	GG, K = M.shape
	assert GG == max(Gs)

	# XTs = [np.random.rand(N, K)+1e-2 for N in Ns]
	# if X_sum2one: XTs = [XT / np.mean(XT, axis=1, keepdims=True) for XT in XTs]
	XTs = [np.zeros([N, K], dtype=np.float) for N in Ns]

	mX = grb.Model('init_X')
	mX.setParam('OutputFlag', False)
	mX.setParam('Threads', 1)
	# mX.Params.Threads = nCPU
	# mX.Params.BarConvTol = 1e-12
	vx = mX.addVars(K, lb=0.)
	if X_sum2one: mX.addConstr(vx.sum('*') == 1)

	mM = grb.Model('init_M')
	mM.setParam('OutputFlag', False)
	mM.setParam('Threads', 1)
	# mM.Params.Threads = nCPU
	# mM.Params.BarConvTol = 1e-12
	if M_sum2one == 'sum':
		vm = mM.addVars(GG, K, lb=0.)
		mM.addConstrs((vm.sum('*', i) == 1 for i in range(K)))
	# elif M_sum2one in ['L1', 'L2']:
	# 	vm = mM.addVars(GG, K, lb=-1., ub=1.)
	# elif M_sum2one == 'L1':
	# 	assert False
	# 	vm = mM.addVars(GG, K, lb=-grb.GRB.INFINITY)
	# 	vam = mM.addVars(GG, K, lb=0.)
	# 	mM.addConstrs((vam.sum('*', i) == 1 for i in range(K)))
	# 	mM.addConstrs(vam[g, k] == grb.abs_(vm[g, k]) for g in range(GG) for k in range(K))
	# elif M_sum2one == 'L2':
	# 	assert False
	# 	vm = mM.addVars(GG, K, lb=-grb.GRB.INFINITY)
	# 	mM.addConstrs(grb.quicksum(vm[g, k]*vm[g, k] for g in range(GG)) == 1  for k in range(K))
	elif M_sum2one == 'None':
		vm = mM.addVars(K, lb=0.)
	else:
		assert False

	global updateX
	def updateX(N, G, XT, YT, YT_valid, sigma_yx_inv, prior_x):
		obj_share = 0
		YTM = np.broadcast_to(np.nan, XT.shape)
		if dropout_str == 'origin':
			MTM = M[:G].T @ M[:G] + 1e-5*np.eye(K)
			obj_share += grb.quicksum(
				[MTM[i, j] * vx[i] * vx[j] for i in range(K) for j in range(K)])  # quadratic term of X and M
			del MTM
			YTM = YT @ M[:G] * -2
		elif dropout_str == 'pass':
			pass
		else:
			assert False
		# if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
		# 	mu_x, sigma_x_inv = prior_x[1:]
		# 	sigma_x_inv = sigma_x_inv / 10
		# 	obj_share += grb.quicksum([ sigma_x_inv[i]**2/2.		* vx[i]*vx[i]	for i in range(K)])
		# 	obj_share += grb.quicksum([-sigma_x_inv[i]**2 * mu_x[i]	* vx[i]			for i in range(K)])
		# 	obj_share += np.dot(mu_x**2, sigma_x_inv**2) / 2.
		# 	del mu_x, sigma_x_inv
		# elif prior_x[0] == 'Exponential':
		# 	lambda_x, = prior_x[1:]
		# 	lambda_x = lambda_x / 10
		# 	obj_share += grb.quicksum([lambda_x[i]	* vx[i]			for i in range(K)])
		# 	del lambda_x
		# elif prior_x[0] == 'Exponential shared':
		# 	lambda_x, = prior_x[1:]
		# 	lambda_x = lambda_x
		# 	obj_share += grb.quicksum([lambda_x[i]	* vx[i]			for i in range(K)])
		# 	del lambda_x
		# elif prior_x[0] == 'Exponential shared fixed':
		# 	lambda_x, = prior_x[1:]
		# 	lambda_x = lambda_x
		# 	obj_share += grb.quicksum([lambda_x[i]	* vx[i]			for i in range(K)])
		# 	del lambda_x
		# else:
		# 	assert False
		for x, y, y_valid, yTM in zip(XT, YT, YT_valid, YTM):
			obj = obj_share
			if dropout_str == 'pass':
				y = y[y_valid]
				Mi = M[y_valid]
				yTM = y @ Mi * -2
				MTM = Mi.T @ Mi
				obj = obj + grb.quicksum([MTM[i, j] * vx[i] * vx[j] for i in range(K) for j in range(K)])  # quadratic term of X and M
			obj = obj + grb.quicksum([yTM[k] * vx[k] for k in range(K)]) + np.dot(y, y)
			mX.setObjective(obj, grb.GRB.MINIMIZE)
			mX.optimize()
			x[:] = np.array([vx[i].x for i in range(K)])
		del sigma_yx_inv, prior_x
		return XT

	niter2 = init_NMF_iter
	iiter2 = 0
	print('niter2 = {}'.format(niter2))

	last_M = np.copy(M)
	last_re = np.nan

	sigma_yx_invs = [1/YT.std(0).mean() for YT in YTs]
	prior_xs = []
	for prior_x_str, YT, G in zip(prior_x_strs, YTs, Gs):
		if prior_x_str == 'Truncated Gaussian' or prior_x_str == 'Gaussian':
			mu_x = np.full(K, YT.sum(1).mean() / K)
			sigma_x_inv = np.full(K, np.sqrt(K) / YT.sum(1).std())
			prior_xs.append((prior_x_str, mu_x, sigma_x_inv, ))
		elif prior_x_str in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
			lambda_x = np.full(K, G / GG * K / YT.sum(1).mean())
			prior_xs.append((prior_x_str, lambda_x, ))
		else:
			assert None
	# sigma_yx_invs = [None] * len(YTs)
	# prior_xs = [None] * len(YTs)

	sizes = np.array([YT_valid.sum() for YT_valid in YT_valids])

	while iiter2 < niter2:
		display_flag = niter2 <= 20 or iiter2 % 20 == 0 or iiter2 >= niter2 -2

		time_start = timeit.default_timer()

		# estimate XT from M, sigma_yx, prior_x
		# for _ in zip(Ns, Gs, XTs, YTs, YT_valids, sigma_yx_invs, prior_xs):
		# 	updateX(*_)
		pool = Pool(min(8, len(YTs)))
		XTs = pool.starmap_async(updateX, zip(Ns, Gs, XTs, YTs, YT_valids, sigma_yx_invs, prior_xs)).get(1e9)
		pool.close()

		iiter2 += 1

		if display_flag:
			nXTs = [XT / (XT.sum(1, keepdims=True)+1e-30) for XT in XTs]
			print('At iter {}: X: #0 = {},\t#all0 = {},\t#<1e-10 = {},\t#<1e-5 = {},\t#<1e-2 = {},\t#>1e-1 = {}\tin {:.4f} seconds'.format(
				iiter2,
				', '.join(map(lambda x: '%.2f' % x, [(nXT == 0).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%d' % x, [(nXT == 0).all(axis=1).sum() for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-10).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-5 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT<1e-2 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				', '.join(map(lambda x: '%.1f' % x, [(nXT>1e-1 ).sum()/N for N, nXT in zip(Ns, nXTs)])),
				timeit.default_timer() - time_start,
				))
			sys.stdout.flush()
			del nXTs

		# estimate prior_x
		prior_xs_old = prior_xs
		prior_xs = []
		for prior_x, XT in zip(prior_xs_old, XTs):
			if prior_x[0] == 'Truncated Gaussian' or prior_x[0] == 'Gaussian':
				mu_x = XT.mean(0)
				sigma_x_inv = 1. / XT.std(0)
				# sigma_x_inv /= 2			# otherwise, σ^{-1} is overestimated ???
				# sigma_x_inv = np.minimum(sigma_x_inv, 1e2)
				prior_x = (prior_x[0], mu_x, sigma_x_inv, )
			elif prior_x[0] == 'Exponential shared':
				# lambda_x = 1. / XT.mean(0)
				# lambda_x[:] = lambda_x.mean()
				# lambda_x /= 5
				lambda_x = np.full(K, 1 / XT.mean())
				prior_x = (prior_x[0], lambda_x, )
			elif prior_x[0] == 'Exponential':
				lambda_x = 1. / XT.mean(0)
				# lambda_x /= 5
				prior_x = (prior_x[0], lambda_x, )
			elif prior_x[0] == 'Exponential shared fixed':
				pass
			else:
				assert False
			prior_xs.append(prior_x)
		# print('prior_x')
		# print(prior_xs[0])

		# estimate sigma_yx_inv from Y - M X
		ds = [YT - XT @ M[:G].T for YT, XT, G in zip(YTs, XTs, Gs)]
		if dropout_str == 'origin':
			ds = [d.ravel() for d in ds]
		elif dropout_str == 'pass':
			ds = [d[YT_valid] for d, YT_valid in zip(ds, YT_valids)]
		else:
			assert False
		ds = np.array([np.dot(d, d) for d in ds])
		if sigma_yx_inv_str == 'separate':
			sigma_yx_invs = ds/sizes
			re = np.sqrt(np.dot(sigma_yx_invs, betas))
			sigma_yx_invs = 1. / np.sqrt(sigma_yx_invs + 1e-10)
		elif sigma_yx_inv_str == 'average':
			sigma_yx_invs = np.dot(betas, ds) / np.dot(betas, sizes)
			re = np.sqrt(sigma_yx_invs)
			sigma_yx_invs = np.full(len(YTs), 1 / np.sqrt(sigma_yx_invs + 1e-10))
		elif sigma_yx_inv_str.startswith('average '):
			idx = np.array(list(map(int, sigma_yx_inv_str.split(' ')[1:])))
			sigma_yx_invs = np.dot(betas[idx], ds[idx]) / np.dot(betas[idx], sizes[idx])
			re = np.sqrt(sigma_yx_invs)
			sigma_yx_invs = np.full(len(YTs), 1 / np.sqrt(sigma_yx_invs + 1e-10))
		else:
			assert False

		if display_flag:
			print(f'At iter {iiter2}: re: RMSE = {re:.2e}, diff = {re-last_re:.2e},')

		if iiter2 >= niter2: continue

		# time_start = timeit.default_timer()

		if M_sum2one != 'None':
			# print('constructing objective function ...', end='\t')
			sys.stdout.flush()
			time_start = timeit.default_timer()
			obj = 0
			for XT, YT, YT_valid, G, beta, sigma_yx_inv in zip(XTs, YTs, YT_valids, Gs, betas, sigma_yx_invs):
				if dropout_str == 'origin':
					XXT = XT.T @ XT * (beta * sigma_yx_inv**2)
					obj += grb.quicksum(XXT[i, i] * vm[k, i] * vm[k, i] for k in range(G) for i in range(K))
					XXT *= 2
					obj += grb.quicksum(XXT[i, j] * vm[k, i] * vm[k, j] for k in range(G) for i in range(K) for j in range(i+1, K))

					YXT = YT.T @ XT * (-2 * beta * sigma_yx_inv**2)
					YTY = np.dot(YT.ravel(), YT.ravel()) * beta * sigma_yx_inv**2
				elif dropout_str == 'pass':
					YXT = np.empty_like(M)
					for g, (y, y_valid, yx) in enumerate(zip(YT.T, YT_valid.T, YXT.T)):
						yx[:] = y[y_valid] @ XT[y_valid] * (beta * sigma_yx_inv**2 * 2)
						XXT = XT[y_valid].T @ XT[y_valid] * (beta * sigma_yx_inv**2)
						obj += grb.quicksum(XXT[k, k] * vm[g, k] * vm[g, k] for k in range(K))
						XXT *= 2
						obj += grb.quicksum(XXT[k, l] * vm[g, k] * vm[g, l] for k in range(K) for l in range(k+1, K))
					YTY = YT[YT_valid]
					YTY = np.dot(YTY, YTY) * (beta * sigma_yx_inv**2)
				else:
					assert False

				obj += grb.quicksum(YXT[i, j] * vm[i, j] for i in range(G) for j in range(K))
				obj += beta * YTY
			kk = 1e-2
			if kk != 0:
				obj += grb.quicksum([kk/2 * vm[k, i] * vm[k, i] for k in range(GG) for i in range(K)])
			mM.setObjective(obj)
			# print(timeit.default_timer() - time_start)

			# time_start = timeit.default_timer()
			mM.optimize()
			# print(timeit.default_timer() - time_start)

			M = np.array([[vm[i, j].x for j in range(K)] for i in range(GG)])
			if M_sum2one == 'sum':
				pass
			# elif M_sum2one == 'L1':
			# 	M /= np.abs(M).sum(0, keepdims=True) + 1e-10
			# elif M_sum2one == 'L2':
			# 	M /= np.sqrt((M**2).sum(0, keepdims=True)) + 1e-10
			else:
				assert False
		else:
			YXTs = [(YT.T @ XT) * beta for YT, XT, beta in zip(YTs, XTs, betas)]
			obj_2s = []
			for XT, beta in zip(XTs, betas):
				XXT = XT.T @ XT * beta
				obj_2s.append(grb.quicksum([XXT[i,j]*vm[i]*vm[j] for i in range(K) for j in range(K)]))
			for g, Mg in enumerate(M):
				obj = []
				for G, YXT, XT, obj_2 in zip(Gs, YXTs, XTs, obj_2s):
					if g >= G: continue
					obj.append(obj_2)
					obj.append(grb.quicksum([-2*YXT[g, i]*vm[i] for i in range(K)]))
				mM.setObjective(sum(obj, []), grb.GRB.MINIMIZE)
				mM.optimize()
				Mg[:] = np.array([vm[i].x for i in range(K)])
			assert False

		dM = M-last_M

		iiter2 += 1

		if display_flag:
			print(
				f'At iter {iiter2}: '
				f'Diff M: max = {np.abs(dM).max():.2e}, '
				f'RMS = {np.sqrt(np.mean(np.abs(dM)**2)):.2e}, '
				f'mean = {np.abs(dM).mean():.2e}\t'
				f'in {timeit.default_timer() - time_start:.4f} seconds'
			)
			# print(prior_xs)
			sys.stdout.flush()

		last_M = np.copy(M)
		last_re = re

	return M, XTs, sigma_yx_invs, prior_xs

def initializeMByKMeans(YTs, K, random_seed4kmeans, **kwargs):
	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)
	YT = np.concatenate([YT for YT in YTs if YT.shape[1] == GG], axis=0)
	n_init = 10
	print('random seed for K-Means = {}'.format(random_seed4kmeans))
	print('n_init for K-Means = {}'.format(n_init))
	kmeans = sklearn.cluster.KMeans(
		n_clusters=K,
		random_state=random_seed4kmeans,
		n_jobs=1,
		n_init=n_init,
		tol=1e-8,
	).fit(YT)

	return kmeans.cluster_centers_.T

def initializeByKMean(O, modelSpec, prior_x_strs, M_sum2one, **kwargs):
	print('Function initializeByKMean begins')
	K, YTs, YT_valids, Es, Es_empty, betas = O

	Ns, Gs = zip(*[YT.shape for YT in YTs])

	# initialize M
	M = initializeMByKMeans(YTs, K, **modelSpec)
	print('M.shape = {}'.format(M.shape))
	print(f'Hash of M = {hex(hash(M.tobytes()))}')

	if M_sum2one == 'sum':
		M = np.maximum(M, 0)
		M /= M.sum(0, keepdims=True)
	# elif M_sum2one == 'L1':
	# 	print((M < 0).sum())
	# 	print((M > 0).sum())
	# 	M /= np.abs(M).sum(0, keepdims=True)
	# elif M_sum2one == 'L2':
	# 	M /= np.sqrt((M**2).sum(0, keepdims=True))
	elif M_sum2one == 'None' or M_sum2one == None:
		pass
	else:
		assert False

	# initialize XT and perhaps update M
	# sigma_yx is estimated from XT and M
	M, XTs, sigma_yx_invs, prior_xs = initializeMXTsByPartialNMF(YTs, YT_valids, M, betas, prior_x_strs=prior_x_strs, **modelSpec)

	print('sigma_yx_inv = {}'.format(', '.join(map(lambda x: '%.2f' % x, sigma_yx_invs))))

	# method4Sigma_x_inv = 'Identity -1e-1'
	method4Sigma_x_inv = 'Constant'
	# method4Sigma_x_inv = 'EmpiricalFromX 1'
	if all(Es_empty): method4Sigma_x_inv = 'Constant'
	print('method4Sigma_x_inv = {}'.format(method4Sigma_x_inv))
	if method4Sigma_x_inv == 'Constant':
		Sigma_x_inv = np.zeros([K, K])
		delta_x = np.zeros(K)
	elif method4Sigma_x_inv.startswith('Identity'):
		kk = float(method4Sigma_x_inv.split()[1])
		Sigma_x_inv = np.eye(K) * kk
		delta_x = np.zeros(K)
	elif method4Sigma_x_inv.startswith('EmpiricalFromX'):
		kk = float(method4Sigma_x_inv.split()[1])
		Sigma_x = np.zeros([K, K])
		for XT, E, beta in zip(XTs, Es, betas):
			t = np.zeros_like(Sigma_x)
			for XTi, Ei in zip(XT, E):
				t += np.outer(XTi, XT[Ei].sum(0))
			Sigma_x += t * beta
		Sigma_x /= np.dot(betas, [sum(map(len, E)) for E in Es])
		Sigma_x_inv = np.linalg.inv(Sigma_x)
		del Sigma_x
		Sigma_x_inv *= kk
		delta_x = np.zeros(K)
	else:
		assert False

	print('max diff in Sigma_x_inv = {}'.format(Sigma_x_inv.max() - Sigma_x_inv.min()))

	# sys.exit()

	return (XTs, ), (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)

def initializeByRandomSpatial(O, prior_x_strs, random_noise=None):
	print('Function initializeByRandomSpatial begins')
	K, YTs, Es, Es_empty, betas, X_sum2one, M_sum2one, pairwise_potential_str = O
	# if random_noise == None: random_noise = 0
	# print('random noise at initial state = {}'.format(random_noise))

	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)

	M = np.random.rand(GG, K)	# [0, 1] i.i.d. r.v.
	M *= (np.random.rand(*M.shape) < 1/np.sqrt(GG))	# sparsity
	M /= M.max(0, keepdims=True)
	M = np.exp(M)
	M /= M.sum(0, keepdims=True)

	if all(Es_empty):
		Sigma_x_inv = np.zeros((K, K))
	else:
		row_idx, col_idx = np.triu_indices(K, 1)
		Sigma_x_inv = np.random.normal(loc=0, scale=1, size=(K, K))
		Sigma_x_inv_zero = np.random.rand(*Sigma_x_inv.shape) < 2/np.sqrt(K)
		Sigma_x_inv *= Sigma_x_inv_zero
		Sigma_x_inv[col_idx, row_idx] = Sigma_x_inv[row_idx, col_idx]

	delta_x = np.zeros(K)

	M, XTs, sigma_yx_invs, prior_xs = initializeMXTsByPartialNMF(
		YTs, M, betas, prior_x_strs, X_sum2one, M_sum2one, niter2=1,
	)
	# sigma_yx_invs = [1e3] * len(YTs)

	return (XTs,), (M, sigma_yx_invs, Sigma_x_inv, delta_x, prior_xs)
