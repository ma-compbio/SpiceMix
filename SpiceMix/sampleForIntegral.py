import numpy as np
import torch
from util import PyTorchDType as dtype
import sys, subprocess, os, struct, timeit
import scipy
from scipy.stats import truncnorm, multivariate_normal, mvn
from scipy.special import erf, loggamma
from multiprocessing import Pool

nCPU = 4

n_cache = 2**14
tLogGamma_cache = None
tarange = None
# tLogGamma_cache = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=dtype, device=PyTorch_device)
# tarange = torch.arange(n_cache, dtype=dtype, device=PyTorch_device)

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

def sampleFromSimplexPyTorch(n, D, seed=None):
	seed = 0
	if seed is not None: torch.manual_seed(seed)
	x = torch.rand([n, D], device=PyTorch_device, dtype=dtype).add_(1e-20)
	x.log_()
	x.div_(x.sum(1, keepdim=True))
	return x

def sampleFromTruncatedHyperballPyTorch(n, D, center, l, seed=None):
	if seed is not None: torch.manual_seed(seed)
	m = torch.distributions.normal.Normal(torch.tensor(0., device=PyTorch_device), torch.tensor(1., device=PyTorch_device))
	m_uniform = torch.distributions.uniform.Uniform(torch.tensor(0., device=PyTorch_device), torch.tensor(1., device=PyTorch_device))
	ret = torch.zeros([n, D], dtype=dtype, device=PyTorch_device)
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
		# if num > 0: print(nbatch, cnt)
		cnt += num

	# print(nbatch)
	return ret

def sampleFromTruncatedSuperellipsoidPyTorch(n, D, center, vec, seed=None):
	if seed is not None: torch.manual_seed(seed)
	# Below is slow !!!
	m = torch.distributions.normal.Normal(torch.tensor(0., device=PyTorch_device), torch.tensor(1., device=PyTorch_device))
	m_uniform = torch.distributions.uniform.Uniform(torch.tensor(0., device=PyTorch_device), torch.tensor(1., device=PyTorch_device))
	ret = torch.zeros([n, D], dtype=dtype, device=PyTorch_device)
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

def sampleFromTruncatedMultivariateGaussianGibbsPyTorch(n, D, mean, cov_2, seed=None):
	if seed is not None: torch.manual_seed(seed)
	mean = mean.cpu()
	cov_2 = cov_2.cpu()
	ret = torch.zeros([n, D], dtype=dtype)
	X = torch.zeros([D], dtype=dtype)
	nn = 16
	t = torch.zeros([nn], dtype=dtype)
	lb = - mean - cov_2 @ X
	nburnin = 10
	ndur = 10
	for __t__ in range(1, nburnin+ndur*(n-1)+1):
		for j in range(D):
			lb.add_(X[j], cov_2[:, j])
			mask = cov_2[:, j] > 0
			if mask.any(): l = (lb[mask]/cov_2[mask, j]).max()
			else: l = -1e5
			mask = cov_2[:, j] < 0
			if mask.any(): r = (lb[mask]/cov_2[mask, j]).min()
			else: r = 1e5
			if l > r: print(l, r)
			assert l <= r
			# if l < 0:
			# 	while True:
			# 		t.normal_()
			# 		mask = torch.nonzero((t >= l) & (t <= r))
			# 		if len(mask): break
			# elif l >= 0:
			# 	while True:
			# 		t.normal_().abs_()
			# 		mask = torch.nonzero((t >= l) & (t <= r))
			# 		if len(mask): break
			# X[j] = t[mask[0]]
			l = float(l)
			r = float(r)
			if r-l < 1e-10: X[j] = (l+r)/2
			else: X[j] = truncnorm.rvs(l, r)	# this is slow
			lb.add_(-X[j], cov_2[:, j])
		if __t__ >= nburnin and ((__t__ - nburnin) % ndur) == 0:
			i = int((__t__ - nburnin) / ndur)
			ret[i].copy_(X)
	ret = ret @ cov_2.t() + mean[None]
	return ret.to(device)

def sampleFromTruncatedMultivariateGaussianCPP(n, D, mean, cov_2, seed=None):
	folder = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'drawSampleFromTruncatedMultivariateGaussian')
	if seed is None: seed = 0
	suffix = '_' + str(seed) + '_' + str(os.getpid())
	infile = os.path.join(folder, 'tmp_parameters' + suffix)
	outfile = os.path.join(folder, 'tmp_samples' + suffix)
	assert not os.path.exists(infile) and not os.path.exists(outfile)
	with open(infile, 'wb') as f:
		for m in mean:
			f.write(struct.pack('<d', m))
		for ci in cov_2.T:
			for cij in ci:
				f.write(struct.pack('=d', cij))
	# print(np.fromfile(infile))
	subprocess.check_output([
		os.path.join(folder, 'main'),
		infile, outfile,
		str(n), str(D), str(seed),
	])
	x = np.fromfile(outfile).reshape(n, D)
	os.remove(infile)
	os.remove(outfile)
	return x

def sampleFromTruncatedMultivariateGaussianCPPBatch(N, n, D, means, cov_U, init_xs, seeds):
	# time_start = timeit.default_timer()
	assert N == len(means) == len(seeds)
	folder = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'drawSampleFromTruncatedMultivariateGaussian')
	infiles, outfiles = [], []
	batch_list = os.path.join(folder, 'batch_list_' + str(os.getpid()) + '.txt')
	with open(batch_list, 'w') as fb:
		for mean, init_x, seed in zip(means, init_xs, seeds):
			suffix = '_' + str(seed) + '_' + str(os.getpid())
			infile = os.path.join(folder, 'tmp_file', 'tmp_parameters' + suffix)
			outfile = os.path.join(folder, 'tmp_file', 'tmp_samples' + suffix)
			infiles.append(infile)
			outfiles.append(outfile)
			assert not os.path.exists(infile) and not os.path.exists(outfile)
			with open(infile, 'wb') as f:
				for m in mean:
					f.write(struct.pack('<d', m))
				for ci in cov_U:
					for cij in ci:
						f.write(struct.pack('=d', cij))
				for x in init_x:
					f.write(struct.pack('<d', x))
			fb.write('\t'.join([infile, outfile, str(n), str(D), str(seed)]) + '\n')
			# break
	# print(timeit.default_timer() - time_start)
	# time_start = timeit.default_timer()
	subprocess.Popen([
		'xargs',
		'-n5',
		'-a', batch_list,
		'-P', str(nCPU),
		os.path.join(folder, 'main'),
	]).wait()
	# sys.exit()
	# print(timeit.default_timer() - time_start)
	# time_start = timeit.default_timer()
	X = np.zeros([N, n, D], dtype=np.float)
	# print(X.shape)
	for Xi, infile, outfile in zip(X, infiles, outfiles):
		np.copyto(Xi, np.fromfile(outfile).reshape(n, D))
		os.remove(infile)
		os.remove(outfile)
	os.remove(batch_list)
	# print(timeit.default_timer() - time_start)
	# print(X[0,:3, :3])
	# sys.exit()
	return X

def estimateMomentsForTruncatedMultivariateGaussianCPPBatch(n, D, means, cov, cov_U, prec, prec_U, init_xs, seeds):
	print('Running Gibbs sampler (cpp) ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	assert len(means) == len(seeds) == len(init_xs)
	N = len(means)
	folder = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'TruncatedMultivariateGaussian')
	infiles, outfiles = [], []
	batch_list = os.path.join(folder, 'batch_list_' + str(os.getpid()) + '.txt')
	with open(batch_list, 'w') as fb:
		for mean, init_x, seed in zip(means, init_xs, seeds):
			suffix = f'_{os.getpid()}_{seed}'
			infile = os.path.join(folder, 'tmp_file', 'tmp_parameters' + suffix)
			outfile = os.path.join(folder, 'tmp_file', 'tmp_moments' + suffix)
			infiles.append(infile)
			outfiles.append(outfile)
			assert not os.path.exists(infile) and not os.path.exists(outfile)
			with open(infile, 'wb') as f:
				for m in mean:
					f.write(struct.pack('<d', m))
				for ci in cov_U:
					for cij in ci:
						f.write(struct.pack('=d', cij))
				for pi in prec_U:
					for pij in pi:
						f.write(struct.pack('=d', pij))
				for x in init_x:
					f.write(struct.pack('<d', x))
			fb.write('\t'.join([infile, outfile, str(n), str(D), str(seed)]) + '\n')
	subprocess.Popen([
		'xargs',
		'-n5',
		'-a', batch_list,
		'-P', str(nCPU),
		os.path.join(folder, 'bin', 'calcMoments'),
	]).wait()
	ndrawn = np.empty([N, 1], dtype=np.int)
	m1 = np.empty([N, D], dtype=np.float)
	v1 = np.empty([N, D], dtype=np.float)
	m2 = np.empty([N, D, D], dtype=np.float)
	v2 = np.empty([N, D, D], dtype=np.float)
	entropy = np.empty([N, 1], dtype=np.float)
	ventropy = np.empty([N, 1], dtype=np.float)
	for ni, m1i, v1i, m2i, v2i, ei, vei, infile, outfile in zip(ndrawn, m1, v1, m2, v2, entropy, ventropy, infiles, outfiles):
		with open(outfile, 'rb') as f:
			t = np.fromfile(f, count=1, dtype=np.int32)
			ni[:] = t[:1]
			assert len(t) == 1
			t = np.fromfile(f, count=-1, dtype=np.double)
			m1i[:] = t[:D]
			t = t[D:]
			v1i[:] = t[:D]
			t = t[D:]
			m2i[:] = t[:D**2].reshape(D, D)
			t = t[D**2:]
			assert np.abs(m2i - m2i.T).max() < 1e-15
			v2i[:] = t[:D**2].reshape(D, D)
			t = t[D**2:]
			assert np.abs(v2i - v2i.T).max() < 1e-15
			ei[:] = t[:1]
			t = t[1:]
			vei[:] = t[:1]
			assert len(t) == 1
		os.remove(infile)
		os.remove(outfile)
	os.remove(batch_list)
	ndrawn = ndrawn.flatten()
	entropy = entropy.flatten()
	ventropy = ventropy.flatten()
	print(timeit.default_timer() - time_start)
	print(f'rel eps: m1: {np.abs(v1/m1).max():.2e}\tm2: {np.abs(v2/m2).max():.2e}\tentropy: {np.max(ventropy/entropy):.2e}')
	print(f'# drawn: min = {ndrawn.min()}, max = {ndrawn.max()}, mean = {ndrawn.mean()}')
	# print(f'Gibbs sampler done in {timeit.default_timer() - time_start} seconds')
	# e = m2.reshape(N, D*D) @ prec.flatten() / 2 - (m1 * (means @ prec)).sum(-1) + ((means @ prec_U.T)**2).sum(-1) / 2
	# ve = (v2**2).reshape(N, D*D) @ (prec**2).flatten() / 4 + ((v1**2) * ((means @ prec)**2)).sum(-1)
	# ve = np.sqrt(ve)
	# print(np.abs(e-entropy).max())
	# print(np.abs(ve-ventropy).max())
	# for i in range(0, N, 100):
	# 	print(i, ve[i], ventropy[i])
	# exit()

	print('Calculating partition functions (Fortran) ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	entropy += D/2*np.log(2*np.pi) + np.log(cov_U[np.diag_indices(D)]).sum()
	# multi thread
	global pool_func
	# cdf
	def pool_func(mean, cov=cov, D=D):
		return mvn.mvnun(
			# lower=-mean, upper=np.full(D, np.inf),
			lower=np.full(D, -np.inf), upper=mean,
			means=np.zeros(D, dtype=np.float), covar=cov,
			maxpts=1000000*D,
			# maxpts=1*D,
			abseps=1e-2,
			releps=1e-3,
		)
	pool = Pool(nCPU)
	ret = pool.map(pool_func, list(means))
	pool.close()
	pool.join()
	Z = np.array(list(zip(*ret))[0])
	print(timeit.default_timer() - time_start)
	print(
		f'Z:\t'
		f'min = {Z[Z!=0].min():.2e}\t'
		f'#zeros = {(Z==0).sum()}\t'
		f'#<1e-10 = {(Z<1e-10).sum()}\t'
		f'#<1e-5 = {(Z<1e-5).sum()}\t'
		f'#<1e-3 = {(Z<1e-3).sum()}'
	)
	# ZZ = np.copy(Z)
	# print(means[Z==0])
	idx = Z == 0
	# idx = np.ones(len(Z), dtype=np.bool)
	logZ = np.empty_like(Z)
	logZ[~idx] = np.log(Z[~idx])
	if idx.any():
		print('Calculating partition functions (PyTorch) ...', end='\t')
		# print('Calculating partition functions (PyTorch) ...', end='\n')
		sys.stdout.flush()
		time_start = timeit.default_timer()
		tlogZ, tvlogZ = estimateLogPartitionFunctionForTruncatedMultivariateGaussianPyTorch(
			D,
			torch.tensor(means[idx][:, None], dtype=dtype, device=PyTorch_device),
			torch.tensor(cov_U.T[None], dtype=dtype, device=PyTorch_device),
			torch.tensor(prec_U.T[None], dtype=dtype, device=PyTorch_device),
			torch.tensor(m1[idx][:, None], dtype=dtype, device=PyTorch_device),
		)
		tlogZ = tlogZ.squeeze(-1)
		tvlogZ = tvlogZ.squeeze(-1)
		logZ[idx] = tlogZ.cpu().data.numpy()
		print(timeit.default_timer() - time_start)
		print(f'max σ = {tvlogZ.max().item()}, max rel σ = {(tvlogZ / tlogZ).abs_().max().item()}')
	# d = np.log(ZZ) - logZ
	# for i in range(0, len(d), 100):
	# for i in range(0, len(d)):
	# 	print(i, np.array2string(means[i], formatter={'all': '{:.2e}'.format}), logZ[i], np.log(ZZ[i]), d[i], d[i]/np.log(ZZ[i]))
	entropy += logZ
	# single thread
	# entropy += np.log([mvn.mvnun(lower=-mean, upper=np.full(D, np.inf), means=np.zeros(D), covar=cov)[0] for mean in means])

	# exit()

	return ndrawn, m1, v1, m2, v2, entropy, ventropy

def estimateLogPartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean, tcov_L, tprec_L, tmean_em):
	assert tcov_L.shape[1:] == (K, K)
	M = len(tcov_L)
	assert tmean.shape[1:] == (M, K)
	N = len(tmean)

	nround = 1e3 * K
	nsample = 2**10
	tsample = torch.empty([nsample, K], dtype=dtype, device=PyTorch_device)		# (n, K)
	tlogZ = torch.zeros([N, M], dtype=dtype, device=PyTorch_device)	# (N, M)
	tvlogZ = torch.zeros_like(tlogZ)

	# """
	# exponential
	tlambda = tmean.clamp(min=0).add_(2, tcov_L.view(M, K**2)[:, ::K+1].max(1)[0][None, :, None]).pow_(-1)	# (N, M, K)
	# tlambda = tmean.add(1e-3).pow(-1)  # (N, M, K)
	# tlambda.masked_scatter_(tmean < 0, -tmean / tcov_L.view(M, K**2)[:, ::K+1].max(1)[0].pow(2).mul(2)[None, :, None])
	# for m, l in zip(tmean.cpu().data.numpy(), tlambda.cpu().data.numpy()):
	# 	print(np.array2string(m, formatter={'all': '{:.2e}'.format}))
	# 	print(np.array2string(1/l, formatter={'all': '{:.2e}'.format}))
	for _ in range(int(nround)):
		torch.rand(*tsample.shape, out=tsample, dtype=dtype, device=PyTorch_device)
		tsample.log_().neg_()	# (n, K)
		chunk_size = int(2**30 / (M*nsample*K))
		for tmeanc, tlambdac, tlogZc, tvlogZc in zip(tmean.split(chunk_size, 0), tlambda.split(chunk_size, 0), tlogZ.split(chunk_size, 0), tvlogZ.split(chunk_size, 0)):
			Nc = len(tmeanc)
			t = torch.empty([Nc, M, nsample, K], dtype=dtype, device=PyTorch_device)
			t.copy_(tsample)
			e = t.sum(-1)
			t.div_(tlambdac[:, :, None, :])
			t.sub_(tmeanc[:, :, None, :])
			e.add_((t @ tprec_L[None]).pow_(2).sum(-1).div_(-2))			# (N, M, n)
			e_max = e.max(-1, keepdim=True)[0]
			t = e.sub_(e_max).exp_().mean(-1).log_().add_(e_max.squeeze(-1))
			tlogZc.add_(t)
			tvlogZc.add_(t.pow_(2))
	# """
	"""
	# Gamma
	# first- and second-order moments
	# ttheta = tcov_L.view(M, K**2)[:, ::K+1].max(1)[0][None] / tmean_em
	# tk = tmean_em / ttheta
	# exponential
	# ttheta = tmean.clamp(min=0).add_(2, tcov_L.view(M, K**2)[:, ::K+1].max(1)[0][None, :, None]).div_(2)
	# tk = torch.ones_like(tmean_em)
	# mode and mean
	# ttheta = tmean_em - tmean
	# tk = tmean_em / ttheta
	# print(ttheta)
	# print(tk)
	# exit()
	theta_np = ttheta.cpu().data.numpy()
	k_np = tk.cpu().data.numpy()
	for _ in range(int(nround)):
		sample_np = np.empty([N*M*K, nsample], dtype=np.float)
		logpdf_np = np.empty([N*M*K, nsample], dtype=np.float)
		for k, theta, s, lp in zip(k_np.flatten(), theta_np.flatten(), sample_np, logpdf_np):
			s[:] = np.random.gamma(k, theta, size=nsample)
			lp[:] = scipy.stats.gamma.logpdf(s, a=k, scale=theta)
		sample_np = sample_np.reshape(N, M, K, nsample).transpose(0, 1, 3, 2)
		logpdf_np = logpdf_np.reshape(N, M, K, nsample).sum(2)
		tsample = torch.tensor(sample_np, dtype=dtype, device=PyTorch_device)
		e = torch.tensor(logpdf_np, dtype=dtype, device=PyTorch_device).neg_()
		tsample.sub_(tmean[:, :, None, :])
		e.add_((tsample @ tprec_L[None]).pow_(2).sum(-1).div_(-2))
		e_max = e.max(-1, keepdim=True)[0]
		t = e.sub_(e_max).exp_().mean(-1).log_().add_(e_max.squeeze(-1))
		tlogZ.add_(t)
		tvlogZ.add_(t.pow_(2))

	# """

	tlogZ.div_(nround)
	tvlogZ.div_(nround)
	tvlogZ.sub_(tlogZ.pow(2))
	tlogZ.sub_(K/2 * np.log(2*np.pi))
	tlogZ.sub_(tcov_L.view(M, K**2)[:, ::K+1].log().sum(-1)[None])
	tlogZ.sub_(tlambda.log_().sum(-1))

	return tlogZ, tvlogZ

def estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean, tcov_2):
	assert tcov_2.shape[1:] == (K, K)
	M = len(tcov_2)
	assert tmean.shape[1:] == (M, K)
	N = len(tmean)

	nround = 1e3 * K
	nsample = 2**10
	tsample = torch.empty([nsample, K], dtype=dtype, device=PyTorch_device)	# (n, K)
	# tmean = torch.tensor(mean, dtype=dtype, device=PyTorch_device)	# (N, M, K)
	tZ = torch.zeros([N, M], dtype=dtype, device=PyTorch_device)	# (N, M)

	for _ in range(int(nround)):
		tsample.normal_()	# (n, K)
		t = tsample[None] @ tcov_2	# (M, n, K)
		chunk_size = int(2**30 / (M*nsample*K))
		for tmeanc, tZc in zip(tmean.split(chunk_size, 0), tZ.split(chunk_size, 0)):
			tZc.add_((tmeanc[:, :, None, :] + t[None, :, :, :] >= 0).all(-1).type(dtype).mean(-1))
			# (N, M, 1, K)-(1, M, n, K) -> (N, M, n, K) -> (N, M, n) -> (N, M)

	tZ.div_(nround)

	return tZ

def estimateMomentsForTruncatedMultivariateGaussianPyTorch(K, cov, means, func_args=None):
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

		arr = torch.tensor(arr, dtype=dtype, device=PyTorch_device)
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
	tcov, tcov1, tcov2 = [torch.tensor(_, dtype=dtype, device=PyTorch_device) for _ in [cov, cov1, cov2]]
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
		# tmean2 = torch.tensor(mean2, dtype=dtype, device=PyTorch_device)
		# del mean2
		# t2_ = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K-2, tmean2, tcov2_2)	# (N, K*(K-1)/2)
		# del tmean2
		# using builtin
		# I_0^{n-2}		upper triangular part						shape: (N, K*(K-1)/2)
		t2_ = estimatePartitionFunction(K-2, mean2, cov2)
		del mean2
		# I_0^{n-2}		square matrix, zero diagonal				shape: (N, K, K-1)
		t2 = torch.zeros([N, K, K-1], dtype=dtype, device=PyTorch_device)
		# I_0^{n-2}		fill upper triangular part					shape: (N, K, K-1)
		t2[:, row_idx, col_idx-1] = t2_
		# I_0^{n-2}		fill lower triangular part					shape: (N, K, K-1)
		t2[:, col_idx, row_idx] = t2_
		del t2_
		# mu(-i) _j _ {row: i, column: j!=i}						shape: (N, K, K-1)
		tmean1 = torch.tensor(mean1, dtype=dtype, device=PyTorch_device)
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
		tmean0 = torch.tensor(mean, dtype=dtype, device=PyTorch_device)
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
		t2_ = torch.zeros([N, K, K], dtype=dtype, device=PyTorch_device)
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

def integrateOfExponentialOverSimplexRecurrence(teta, grad=1., requires_grad=False):
	N, D = teta.shape
	# tarr[:, J:].sub_( t.exp().mul_(tarr[:, [J-1]]) ).div_(t.neg_())

	if requires_grad:
		tarr = torch.ones([N, D, D], dtype=dtype, device=PyTorch_device)

		for J in range(1, D):
			t = teta[:, J:] - teta[:, [J-1]]

			tt = t.exp().mul(tarr[:, J-1, [J-1]])
			tarr[:, J, J:] = tarr[:, J-1, J:].sub(tt).div(t.neg())
			del t, tt

		tret = tarr[:, -1, -1].mul(teta[:, -1].neg().exp())
		tret = tret.log()
	else:
		tarr = torch.ones([N, D, D], dtype=dtype, device=PyTorch_device)

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

def integrateOfExponentialOverSimplexInduction(teta, grad=None, requires_grad=False):
	if grad is None: grad = torch.tensor([1.], dtype=dtype, device=PyTorch_device)
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
		A = torch.empty([N, D], dtype=dtype, device=PyTorch_device)
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
			A = torch.empty([len(teta), D], dtype=dtype, device=PyTorch_device)
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
		A = torch.empty([N, D], dtype=dtype, device=PyTorch_device)
		Asign = torch.empty_like(A)
		for k in range(D):
			t = teta - teta[:, [k]]
			t[:, k].fill_(1)
			tsign = t.sign()
			Asign[:, k] = tsign.prod(-1)
			t.abs_().log_()
			A[:, k].copy_(t.sum(-1).neg_())
		# the first (D-1) terms in the Taylar expansion cancel out
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
		trets = torch.empty(len(teta), dtype=dtype, device=PyTorch_device)
		tidx = tarange[D-1:D+nterm-1]
		tlg = tLogGamma_cache[D-1: D+nterm-1]
		for tret, teta, teta_grad in zip(trets.split(chunk_size, 0), tetas.split(chunk_size, 0), tetas.grad.split(chunk_size, 0)):
			A = torch.empty([len(teta), D], dtype=dtype, device=PyTorch_device)
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
			# print(Aexp.sub(toffset).exp_().mul_(Aexp_sign).sum(-1))
			# exit()
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

def integrateOfExponentialOverSimplexInduction2(teta, grad=None, requires_grad=False, PyTorch_device='cpu'):
	if grad is None: grad = torch.tensor([1.], dtype=dtype, device=PyTorch_device)
	N, D = teta.shape

	global tLogGamma_cache, tarange
	if tLogGamma_cache is None:
		tLogGamma_cache = torch.tensor(loggamma(np.arange(1, n_cache)), dtype=dtype, device=PyTorch_device)
	if tarange is None:
		tarange = torch.arange(n_cache, dtype=dtype, device=PyTorch_device)

	t_eta_offset = teta.max(-1, keepdim=True)[0] + 1e-5
	# nterm = 256
	nterm = (teta.max() - teta.min()).item()
	nterm = max(nterm+10, nterm*1.1)
	nterm = int(nterm)
	tlg = tLogGamma_cache[D-1: D+nterm-1]

	if requires_grad:
		teta = teta - t_eta_offset
		teta = teta.neg()
		# teta = teta.sort()[0]

		f = torch.zeros([N, D], dtype=dtype, device=PyTorch_device)
		tret = torch.zeros([nterm, N], dtype=dtype, device=PyTorch_device)

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
		trets = torch.empty(N, dtype=dtype, device=PyTorch_device)
		chunk_size = 32
		for teta, teta_grad, tretc in zip(tetas.split(chunk_size, 0), teta.grad.split(chunk_size, 0), trets.split(chunk_size, 0)):
			N = len(teta)
			teta_log = teta.log()
			tret = torch.zeros([nterm, N], dtype=dtype, device=PyTorch_device)
			tgrad = torch.full([nterm, N, D], -np.inf, dtype=dtype, device=PyTorch_device)
			f = torch.zeros([N, D], dtype=dtype, device=PyTorch_device)
			g = torch.full([N, D, D], -np.inf, dtype=dtype, device=PyTorch_device)
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


def integrateOfExponentialOverSimplexSampling(teta, grad=None, requires_grad=False, seed=None):
	# if grad is None: grad = torch.tensor([1.], dtype=dtype, device=PyTorch_device)
	N, D = teta.shape

	# teta_offset = teta.min(1, keepdim=True)[0]

	loggamma_D = loggamma(D)
	n = 2**8
	n = int(n)
	nround = 1

	if requires_grad:
		# teta = teta - teta_offset
		tlogZ = torch.zeros(N, dtype=dtype, device=PyTorch_device)
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
		tlogZ = torch.zeros(N, dtype=dtype, device=PyTorch_device)
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

def test_sampleFromTruncatedMultivariateGaussianCPP_TruncatedNormal():
	import timeit
	folder = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'drawSampleFromTruncatedMultivariateGaussian')
	outfile = os.path.join(folder, 'test_TN')
	n = 1000000
	l, r = 3, 3.1
	time_start = timeit.default_timer()
	subprocess.call([
		os.path.join(folder, 'main'),
		outfile,
		str(l), str(r), str(n),
	])
	print(timeit.default_timer() - time_start)
	x = np.fromfile(outfile)
	print(x.size)
	from matplotlib import pyplot as plt
	plt.hist(x, density=True, bins=100)
	xx = np.linspace(l, r, 1001)
	plt.plot(xx, np.exp(-xx**2/2)*2 / ( erf(r/np.sqrt(2)) - erf(l/np.sqrt(2)) ) / np.sqrt(2*np.pi) )
	plt.show()

def test_dist():

	import timeit
	time_start = timeit.default_timer()
	x = sampleFromSimplex(10000, 3)
	# x = sampleFromHyperball(10000, 2, 10)
	# x = sampleFromSuperellipsoid(10000, 2, np.array([[1, 3], [1, -3]]))
	# x = sampleFromSimplexPyTorch(10000, 3).cpu().data.numpy()
	# x = sampleFromTruncatedSuperellipsoidPyTorch(100000, 2, torch.tensor([0., 0.,], dtype=dtype, device=PyTorch_device), torch.tensor([[1., 100.], [1., -100.]], dtype=dtype, device=PyTorch_device)).cpu().data.numpy()
	# D = 20; x = sampleFromTruncatedHyperballPyTorch(100000, D, torch.tensor([0.,]*D, dtype=dtype, device=PyTorch_device), 1).cpu().data.numpy()
	# D, D_ = 20, 10; x = sampleFromTruncatedHyperballPyTorch(100000, D, torch.tensor([1e-4]*D_ + [0.,]*(D-D_), dtype=dtype, device=PyTorch_device), 1).cpu().data.numpy()

	"""
	n, D = 1000, 2
	mean = np.array([1e-2, 7], dtype=np.float)
	cov = np.array([[3., -2], [-2, 3.]], dtype=np.float)
	# cov = np.eye(2)
	cov_2 = np.linalg.cholesky(cov)
	print(cov_2)
	# cov_2 = torch.tensor(cov_2, dtype=dtype, device=PyTorch_device)
	x = sampleFromTruncatedMultivariateGaussianGibbsPyTorch(n, D, torch.tensor(mean, dtype=dtype, device=PyTorch_device), torch.tensor(cov_2, dtype=dtype, device=PyTorch_device)).cpu().data.numpy()
	# x = sampleFromTruncatedMultivariateGaussianCPP(n*100, D, mean, cov_2)
	print(x[:, 0].min(), x[:, 1].min(), x.sum(1).min())
	# """

	print(timeit.default_timer() - time_start)
	from matplotlib import pyplot as plt
	plt.figure()
	plt.scatter(x[:,0], x[:,1], s=1)
	plt.show()
	plt.hist(x[:, 0], density=True, bins=100)
	plt.show()

def test_moments():
	np.random.seed(0)

	def brute_force(K, cov, mean):
		N = len(mean)
		nsample = 2**10
		# nround = 1000 * K
		nround = 10 * K
		t = torch.empty([nsample, K], dtype=dtype, device=PyTorch_device)
		tcov = torch.tensor(cov, dtype=dtype, device=PyTorch_device)
		tcov_2 = torch.cholesky(tcov, upper=True)
		tmean = torch.tensor(mean, dtype=dtype, device=PyTorch_device)
		tm0 = torch.zeros([N], dtype=dtype, device=PyTorch_device)
		tm1 = torch.zeros([N, K], dtype=dtype, device=PyTorch_device)
		tm2 = torch.zeros([N, K, K], dtype=dtype, device=PyTorch_device)
		for n in range(int(nround)):
			t.normal_()
			x = (t @ tcov_2)[None, :, :] + tmean[:, None, :]	# (N, n, K)
			index = (x >= 0).all(-1).type(dtype)
			tm0.add_(index.sum(-1))
			x.mul_(index[..., None])
			tm1.add_(x.sum(1))
			tm2.add_(x.permute(0, 2, 1) @ x)
		tm1.div_(tm0[:, None])
		tm2.div_(tm0[:, None, None])
		tm0.div_(nround * nsample)
		return (_.cpu().data.numpy() for _ in [tm0, tm1, tm2])

	import timeit
	t = torch.empty([10], dtype=dtype, device=PyTorch_device)
	torch.cuda.synchronize()
	del t
	N = 4
	K = 3
	row_idx, col_idx = np.triu_indices(K, 0)
	# cov = np.eye(K, dtype=np.float)*2
	cov = np.random.rand(K, K)
	cov[np.triu_indices(K, 1)] = 0
	cov = cov @ cov.T
	cov *= 1e-2
	prec = np.linalg.inv(cov)
	print(cov)
	# mean = np.zeros([N, K], dtype=np.float)
	# mean += np.array([5, 0, 0])[None]
	mean = np.random.rand(N, K)*1e-1 - 2e-1
	print(mean)

	print('recurrence ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	Z, mu, Sigma = estimateMomentsForTruncatedMultivariateGaussianPyTorch(K, cov, mean)
	# torch.cuda.synchronize()
	Sigma -= mu[:, :, None] * mu[:, None, :]
	print(timeit.default_timer() - time_start)

	print('recurrence parallel ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	Z_, mu_, Sigma_ = estimateMomentsForTruncatedMultivariateGaussianPyTorch(K, cov, mean)
	# torch.cuda.synchronize()
	Sigma_ -= mu_[:, :, None] * mu_[:, None, :]
	print(timeit.default_timer() - time_start)

	print('recurrence parallel ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	Z_, mu_, Sigma_ = estimateMomentsForTruncatedMultivariateGaussianPyTorch(K, cov, mean)
	# torch.cuda.synchronize()
	Sigma_ -= mu_[:, :, None] * mu_[:, None, :]
	print(timeit.default_timer() - time_start)

	print('brute force ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	m0, m1, m2 = brute_force(K, cov, mean)
	m2 -= m1[:, :, None] * m1[:, None, :]
	# torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)

	print('brute force parallel ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	m0_, m1_, m2_ = brute_force(K, cov, mean)
	m2_ -= m1_[:, :, None] * m1_[:, None, :]
	# torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)

	print('cpp Gibbs ...', end='\t')
	sys.stdout.flush()
	n = int(1e6)
	time_start = timeit.default_timer()
	cov_U = scipy.linalg.cholesky(cov, lower=False)
	prec_U = scipy.linalg.cholesky(prec, lower=False)
	ndrawn, cm1, cv1, cm2, cv2, entropy, ventropy = estimateMomentsForTruncatedMultivariateGaussianCPPBatch(n, K, mean, cov, cov_U, prec, prec_U, (-mean+1e-10) @ np.linalg.inv(cov_U), np.arange(len(mean)))
	cm2 -= cm1[:, :, None] * cm1[:, None, :]
	print(timeit.default_timer() - time_start)
	print(f'ndrawn =\t{ndrawn}')
	print(f'cv1 =\t{cv1.max(0)}')
	print(f'cv2 =\t{cv2[:, row_idx, col_idx].max(0)}')
	print(f'ventropy =\t{ventropy.max(0)}')

	print('cpp Gibbs parallel ...', end='\t')
	sys.stdout.flush()
	time_start = timeit.default_timer()
	cov_U = scipy.linalg.cholesky(cov, lower=False)
	ndrawn_, cm1_, cv1_, cm2_, cv2_, entropy_, ventropy_ = estimateMomentsForTruncatedMultivariateGaussianCPPBatch(n, K, mean, cov, cov_U, prec, prec_U, (-mean+1e-10) @ np.linalg.inv(cov_U), np.arange(len(mean))+len(mean))
	cm2_ -= cm1_[:, :, None] * cm1_[:, None, :]
	print(timeit.default_timer() - time_start)
	print(f'ndrawn =\t{ndrawn_}')
	print(f'cv1 =\t{cv1_.max(0)}')
	print(f'cv2 =\t{cv2_[:, row_idx, col_idx].max(0)}')
	print(f'ventropy =\t{ventropy_.max(0)}')

	abseps = 1e-7
	releps = 1e-7

	print('multivariate_normal.cdf(0, mean=mean, cov=cov) ...', end='\t')
	time_start = timeit.default_timer()
	cdf = np.array([multivariate_normal.cdf(np.zeros(K), mean=-m, cov=cov, maxpts=1000000*K, abseps=abseps, releps=releps) for m in mean])
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)

	print('multivariate_normal.cdf(mean, mean=0, cov=cov) ...', end='\t')
	time_start = timeit.default_timer()
	tcdf = multivariate_normal.cdf(mean, mean=np.zeros(K), cov=cov, maxpts=1000000*K, abseps=abseps, releps=releps)
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)
	print(np.abs(cdf - tcdf).max(0))

	print('multivariate_normal.cdf(mean, mean=None, cov=cov) ...', end='\t')
	time_start = timeit.default_timer()
	tcdf = multivariate_normal.cdf(mean, mean=None, cov=cov, maxpts=1000000*K, abseps=abseps, releps=releps)
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)
	print(np.abs(cdf - tcdf).max(0))

	print('multivariate_normal(-mean, cov).cdf(0) ...', end='\t')
	time_start = timeit.default_timer()
	tcdf = np.array([multivariate_normal(mean=-m, cov=cov).cdf(np.zeros(K)) for m in mean])
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)
	print(np.abs(cdf - tcdf).max(0))

	print('multivariate_normal(0, cov).cdf(mean) ...', end='\t')
	time_start = timeit.default_timer()
	tcdf = multivariate_normal(mean=np.zeros(K), cov=cov).cdf(mean)
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)
	print(np.abs(cdf - tcdf).max(0))

	print('multivariate_normal(None, cov).cdf(mean) ...', end='\t')
	time_start = timeit.default_timer()
	tcdf = multivariate_normal(mean=None, cov=cov).cdf(mean)
	print(timeit.default_timer() - time_start)
	# print(cdf - tcdf)
	print(np.abs(cdf - tcdf).max(0))

	print('mvn(lower=0, means=m) ...', end='\t')
	time_start = timeit.default_timer()
	mvnun = np.array([mvn.mvnun(lower=np.zeros(K), upper=np.full(K, np.inf), means=m, covar=cov, maxpts=1000000*K, abseps=abseps, releps=releps)[0] for m in mean])
	print(timeit.default_timer() - time_start)

	print('mvn(lower=-m, means=0) ...', end='\t')
	time_start = timeit.default_timer()
	tmvnun = np.array([mvn.mvnun(lower=-m, upper=np.full(K, np.inf), means=np.zeros(K), covar=cov, maxpts=1000000*K, abseps=abseps, releps=releps)[0] for m in mean])
	print(timeit.default_timer() - time_start)
	# print(mvnun - tmvnun)
	print(np.abs(mvnun - tmvnun).max(0))

	print('mvn(upper=m, means=0) ...', end='\t')
	time_start = timeit.default_timer()
	tmvnun = np.array([mvn.mvnun(lower=np.full(K, -np.inf), upper=m, means=np.zeros(K), covar=cov, maxpts=1000000*K, abseps=abseps, releps=releps)[0] for m in mean])
	print(timeit.default_timer() - time_start)
	# print(mvnun - tmvnun)
	print(np.abs(mvnun - tmvnun).max(0))

	print('PyTorch ...', end='\t')
	tmean = torch.tensor(mean, dtype=dtype, device=PyTorch_device)
	tcov_2 = torch.tensor(cov, dtype=dtype, device=PyTorch_device)
	tcov_2 = torch.cholesky(tcov_2, upper=True)
	torch.cuda.synchronize()
	time_start = timeit.default_timer()
	Z_pytorch = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean[:, None], tcov_2[None]).squeeze(1)
	torch.cuda.synchronize()
	Z_pytorch = Z_pytorch.cpu().data.numpy()
	torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)
	print('PyTorch parallel ...', end='\t')
	time_start = timeit.default_timer()
	Z_pytorch_ = estimatePartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean[:, None], tcov_2[None]).squeeze(1)
	torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)
	Z_pytorch_ = Z_pytorch_.cpu().data.numpy()
	torch.cuda.synchronize()
	print(np.abs(Z_pytorch - Z_pytorch_).max(0))

	print('PyTorch ...', end='\t')
	tmean = torch.tensor(mean, dtype=dtype, device=PyTorch_device)
	tcov_2 = torch.tensor(cov, dtype=dtype, device=PyTorch_device)
	tcov_2 = torch.cholesky(tcov_2, upper=True)
	tprec_2 = torch.tensor(prec, dtype=dtype, device=PyTorch_device)
	tprec_2 = torch.cholesky(tprec_2, upper=True)
	torch.cuda.synchronize()
	time_start = timeit.default_timer()
	Z_pytorch_log, vZ_pytorch_log = estimateLogPartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean[:, None], tcov_2[None].contiguous(), tprec_2[None].contiguous(), torch.tensor(cm1, dtype=dtype, device=PyTorch_device))
	Z_pytorch_log = Z_pytorch_log.squeeze(1)
	vZ_pytorch_log = vZ_pytorch_log.squeeze(1)
	torch.cuda.synchronize()
	Z_pytorch_log = Z_pytorch_log.exp_().cpu().data.numpy()
	torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)
	print('PyTorch parallel ...', end='\t')
	time_start = timeit.default_timer()
	Z_pytorch_log_, vZ_pytorch_log_ = estimateLogPartitionFunctionForTruncatedMultivariateGaussianPyTorch(K, tmean[:, None], tcov_2[None].contiguous(), tprec_2[None].contiguous(), torch.tensor(cm1, dtype=dtype, device=PyTorch_device))
	Z_pytorch_log_ = Z_pytorch_log_.squeeze(1)
	vZ_pytorch_log_ = vZ_pytorch_log_.squeeze(1)
	torch.cuda.synchronize()
	print(timeit.default_timer() - time_start)
	Z_pytorch_log_ = Z_pytorch_log_.exp_().cpu().data.numpy()
	torch.cuda.synchronize()
	print(np.abs(Z_pytorch_log - Z_pytorch_log_).max(0))


	print('=' * 10 + ' Z ' + '='*10)
	print(Z)
	print(Z_pytorch_log)
	print('variance')
	# print(Z - Z_)
	# print(m0 - m0_)
	print(np.abs(Z - Z_).max(0))
	print(np.abs(m0 - m0_).max(0))
	print(np.abs(Z_pytorch - Z_pytorch_).max(0))
	print(np.abs(Z_pytorch_log - Z_pytorch_log_).max(0))
	print('-'*10)
	print('bias')
	# print('Z_pytorch - mvnun\t', Z_pytorch - mvnun)
	# print('mvnun - cdf\t', mvnun - cdf)
	# print('cdf - Z\t', cdf - Z)
	# print('Z - m0\t', Z - m0)
	# print('Z_pytorch - mvnun\t', np.abs(Z_pytorch - mvnun).max(0))
	# print('mvnun - cdf\t', np.abs(mvnun - cdf).max(0))
	# print('cdf - Z\t', np.abs(cdf - Z).max(0))
	print('Z - m0\t', np.abs(Z - m0).max(0))
	print('Z - Z_pytorch\t', np.abs(Z - Z_pytorch).max(0))
	print('Z - Z_pytorch_log\t', np.abs(Z - Z_pytorch_log).max(0))

	print('=' * 10 + ' entropy ' + '=' * 10)
	print(entropy, entropy_)
	print('variance')
	print(entropy - entropy_)
	print('diff to par')
	hp = np.array([scipy.stats.multivariate_normal.entropy(mean=m, cov=cov) for m in mean])
	print('H(Gaussian) =', hp)
	print('diff =', entropy-hp)

	print('=' * 10 + ' mu ' + '='*10)
	print('variance')
	# print(mu - mu_)
	# print(m1 - m1_)
	print(np.abs(mu - mu_).max(0))
	print(np.abs(m1 - m1_).max(0))
	print(np.abs(cm1 - cm1_).max(0))
	print('-'*10)
	print('diff')
	# print(mu - m1)
	print(np.abs(mu - m1).max(0))
	print(np.abs(mu - cm1).max(0))
	print('ref diff')
	print(np.abs((mu - m1)/mu).max(0))
	print(np.abs((mu - cm1)/mu).max(0))
	# print('-'*10)
	# print('diff to par')
	# print(mu - mean)
	# print(m1 - mean)

	print('=' * 10 + ' Sigma ' + '='*10)
	print('sym')
	# print((Sigma - Sigma.transpose([0, 2, 1]))[:, row_idx, col_idx])
	# print((Sigma_ - Sigma_.transpose([0, 2, 1]))[:, row_idx, col_idx])
	print(np.abs((Sigma - Sigma.transpose([0, 2, 1]))[:, row_idx, col_idx]).max(0))
	print(np.abs((Sigma_ - Sigma_.transpose([0, 2, 1]))[:, row_idx, col_idx]).max(0))
	print('-'*10)
	print('variance')
	Sigma = (Sigma + Sigma.transpose([0, 2, 1])) / 2
	Sigma_ = (Sigma_ + Sigma_.transpose([0, 2, 1])) / 2
	Sigma = Sigma[:, row_idx, col_idx]
	Sigma_ = Sigma_[:, row_idx, col_idx]
	m2 = m2[:, row_idx, col_idx]
	m2_ = m2_[:, row_idx, col_idx]
	cm2 = cm2[:, row_idx, col_idx]
	cm2_ = cm2_[:, row_idx, col_idx]
	cov = cov[row_idx, col_idx]
	# print(Sigma - Sigma_)
	# print(m2 - m2_)
	print(np.abs(Sigma - Sigma_).max(0))
	print(np.abs(m2 - m2_).max(0))
	print(np.abs(cm2 - cm2_).max(0))
	print('-'*10)
	print('diff')
	# print(Sigma - m2)
	print(np.abs(Sigma - m2).max(0))
	print(np.abs(Sigma - cm2).max(0))
	print('rel diff')
	print(np.abs((Sigma - m2)/Sigma).max(0))
	print(np.abs((Sigma - cm2)/Sigma).max(0))
	print('-'*10)
	# print('diff to par')
	# print(Sigma - cov[None])
	# print(m2 - cov[None])
	# print(np.abs(Sigma - cov[None]).max(0))
	# print(np.abs(m2 - cov[None]).max(0))

def test_IntegralOfExponentialOverSimplex():
	N = 97
	K = 20
	# np.random.seed(0)
	# eta = np.zeros([N, K])
	# eta = np.array([[-1., 0., 1., 2., 3.]])[:, :K]
	# eta = np.array([0, 1e-10, 1], dtype=np.float)
	eta = np.random.rand(N, K)*2 - 1
	eta = eta.reshape(N, K)
	# eta *= 1e-4
	eta *= 1e2
	# print(f'η = {eta}')
	print(eta)
	t = np.sort(eta)
	print(f'min diff η = {np.min(t[:, 1:]-t[:, :-1])}')
	print(f'max range = {(t[:, -1] - t[:, 0]).max()}')
	teta = torch.tensor(eta, dtype=dtype, device=PyTorch_device)

	methods = [
		integrateOfExponentialOverSimplexSampling,
		# integrateOfExponentialOverSimplexRecurrence,
		integrateOfExponentialOverSimplexInduction,
		integrateOfExponentialOverSimplexInduction2,
	]
	logZ_manu = dict(zip(methods, [[] for _ in methods]))
	logZ_auto = dict(zip(methods, [[] for _ in methods]))
	grad_manu = dict(zip(methods, [[] for _ in methods]))
	grad_auto = dict(zip(methods, [[] for _ in methods]))

	for t in range(5):
		sys.stdout.flush()
		idx = np.random.permutation(K)
		# idx = np.arange(K)
		# print(idx)
		teta_ = torch.tensor(eta[:, idx], dtype=dtype, device=PyTorch_device)
		teta_.grad = torch.zeros_like(teta_)
		tgrad = torch.tensor([4], dtype=dtype, device=PyTorch_device)

		for method in methods:
			teta_.grad.zero_()
			teta_.requires_grad_(False)
			torch.cuda.synchronize()
			print(f'{method.__name__}\tmanual ...', end='\t')
			time_start = timeit.default_timer()
			logZ = method(teta_, grad=tgrad, requires_grad=False)
			torch.cuda.synchronize()
			print(timeit.default_timer() - time_start)

			logZ = logZ.cpu().data.numpy()
			logZ_manu[method].append(logZ)
			grad = np.empty_like(eta)
			grad[:, idx] = teta_.grad.cpu().data.numpy()
			grad_manu[method].append(grad)

			teta_.grad.zero_()
			teta_.requires_grad_()
			torch.cuda.synchronize()
			print(f'{method.__name__}\tauto ...', end='\t')
			time_start = timeit.default_timer()
			logZ = method(teta_, grad=tgrad, requires_grad=True)
			logZ.mul(tgrad).sum().backward()
			# logZ.mul(tgrad).sum().backward()
			torch.cuda.synchronize()
			print(timeit.default_timer() - time_start)

			logZ = logZ.cpu().data.numpy()
			logZ_auto[method].append(logZ)

			grad = np.empty_like(eta)
			grad[:, idx] = teta_.grad.cpu().data.numpy()
			grad_auto[method].append(grad)

	for d in [logZ_auto, logZ_manu, grad_auto, grad_manu]:
		for method in methods:
			d[method] = np.stack(d[method])

	def p(a, b):
		for method in methods:
			# print('{}\tmanu\t{}'.format(method.__name__, np.array2string(a[method], formatter={'all': '{:.2e}'.format})))
			# print('{}\tauto\t{}'.format(method.__name__, np.array2string(b[method], formatter={'all': '{:.2e}'.format})))
			pass
		print('='*15 + '\tvariance\t' + '='*15)
		for method in methods:
			print('{}\tmanu\tabs\t{}'.format(method.__name__, np.array2string(
				a[method].std(0).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tmanu\trel\t{}'.format(method.__name__, np.array2string(
				(a[method].std(0) / np.abs(a[method].mean(0))).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tauto\tabs\t{}'.format(method.__name__, np.array2string(
				b[method].std(0).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tauto\trel\t{}'.format(method.__name__, np.array2string(
				(b[method].std(0) / np.abs(b[method].mean(0))).max(),
				formatter={'all': '{:.2e}'.format}
			)))
		print('='*15 + '\tdiff\t\t' + '='*15)
		for method in methods:
			print('{}\tmanu\tabs\t{}'.format(method.__name__, np.array2string(
				np.abs(a[method].mean(0) - a[methods[0]].mean(0)).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tmanu\trel\t{}'.format(method.__name__, np.array2string(
				np.abs(a[method].mean(0) / a[methods[0]].mean(0) - 1).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tauto\tabs\t{}'.format(method.__name__, np.array2string(
				np.abs(b[method].mean(0) - a[methods[0]].mean(0)).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tauto\trel\t{}'.format(method.__name__, np.array2string(
				np.abs(b[method].mean(0) / a[methods[0]].mean(0) - 1).max(),
				formatter={'all': '{:.2e}'.format}
			)))
			print('{}\tdiff\tabs\t{}'.format(method.__name__, np.array2string(
				np.abs(a[method].mean(0) - b[method].mean(0)).max(),
				formatter={'all': '{:.2e}'.format}
			)))
	print('='*15 + '\tlog Z\t\t' + '='*15)
	p(logZ_manu, logZ_auto)
	print('='*15 + '\tgrad\t\t' + '='*15)
	p(grad_manu, grad_auto)


if __name__ == '__main__':
	np.set_printoptions(linewidth=100000)
	# device = torch.device('cpu')
	# test_dist()
	# test_sampleFromTruncatedMultivariateGaussianCPP_TruncatedNormal()
	# test_moments()
	# t = torch.tensor([[-.77, -.64, -.2]], dtype=dtype, device=PyTorch_device)
	# t.grad = torch.empty_like(t)
	# print(integrateOfExponentialOverSimplexInduction2(t, grad=None, requires_grad=True))
	# t.grad.zero_()
	# print(integrateOfExponentialOverSimplexInduction2(t, grad=None, requires_grad=False))
	# exit()
	test_IntegralOfExponentialOverSimplex()
