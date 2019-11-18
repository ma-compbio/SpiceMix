from util import Logger, dataFolder, psutil_process
import numpy as np
from readData import readDataSet
import os, sys, time, itertools, resource, gc, argparse
from estimateParameters import estimateParameters as estimateParameters
from estimateWeights import estimateWeights
import initializeState
import torch

def main(dataset, K, betas, dataset_parameter, logger, modelSpec, max_iter, **kwargs):
	print('dataset = {}'.format(dataset))
	print('K = {}'.format(K))
	print('betas = {}'.format(betas))

	YTs, Es, Es_empty = readDataSet(dataset, **dataset_parameter)
	Ns, Gs = zip(*[YT.shape for YT in YTs])
	GG = max(Gs)
	YTs = [G / GG * K * YT / YT.sum(1).mean() for YT, G in zip(YTs, Gs)]

	print('sizes of Es = {}, # of edges = {}'.format(list(map(len, Es)), [sum(map(len, E)) for E in Es]))

	if modelSpec['dropout_str'] == 'origin':
		YT_valids = [np.broadcast_to(True, YT.shape) for YT in YTs]
	elif modelSpec['dropout_str'] == 'pass':
		YT_valids = [YT != 0 for YT in YTs]
	else: assert False

	for k, v in modelSpec.items():
		print(f'{k}\t=\t{v}')

	print('begin!')
	sys.stdout.flush()

	O = (K, YTs, YT_valids, Es, Es_empty, betas)
	# prior_x_strs = ['Gaussian'] * len(YTs)
	# prior_x_strs = ['Truncated Gaussian'] * len(YTs)
	# prior_x_strs = ['Exponential shared'] * len(YTs)
	prior_x_strs = ['Exponential shared fixed'] * len(YTs)
	# prior_x_strs = ['Exponential'] * len(YTs)
	print(f"prior_x_str = {'	'.join(prior_x_strs)}")

	H, Theta = initializeState.initializeByKMean(
	# H, Theta = initializeState.initializeByRandomSpatial(
		O, modelSpec, **modelSpec, prior_x_strs=prior_x_strs,
	)
	if logger:
		logger.log('H_{:d}'.format(0), H)
		logger.log('Theta_{:d}'.format(0), Theta)
	else:
		# """
		M, sigma_yx_invs, Sigma_x_inv, delta_x = map(np.array, Theta[:-1])
		print('M = \n{}'.format(M[:4]))
		print('σ_yx_inv = {}'	.format(np.array2string(sigma_yx_invs	, formatter={'all': '{:.2e}'.format})))
		print('Σ_x_inv = \n{}'	.format(np.array2string(Sigma_x_inv		, formatter={'all': '{:.2e}'.format})))
		print('δ_x = {}'		.format(np.array2string(delta_x			, formatter={'all': '{:.2e}'.format})))
		del M, sigma_yx_invs, Sigma_x_inv, delta_x
		prior_xs = Theta[-1]
		for prior_x in prior_xs:
			if prior_x[0] in ['Truncated Gaussian', 'Gaussian']:
				mu_x, sigma_x_inv = map(np.array, prior_x[1:])
				print('μ_x = {}'		.format(np.array2string(mu_x		, formatter={'all': '{:.2e}'.format})), end='\t')
				print('σ_xInv = {}'		.format(np.array2string(sigma_x_inv	, formatter={'all': '{:.2e}'.format})), end='\n')
				del mu_x, sigma_x_inv
			elif prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
				lambda_x,  = map(np.array, prior_x[1:])
				print('lambda_x = {}'	.format(np.array2string(lambda_x, formatter={'all': '{:.2e}'.format})))
				del lambda_x
			else:
				assert False
			del prior_x
		del prior_xs
		# """
		pass


	torch.cuda.empty_cache()

	last_Q = np.nan
	# best_Q, best_iter = np.nan, -1

	print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
	print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')

	for iiter in range(max_iter):
		iiter += 1
		modelSpec['iiter'] = iiter

		H = estimateWeights(O, H, Theta, modelSpec, **modelSpec)
		print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
		if logger:
			logger.log(f'H_{iiter}', H)
			pass
		else:
			pass

		Theta, Q = estimateParameters(
			O, H, Theta, modelSpec,
			**modelSpec
		)
		print(f'Current RAM usage (%) is {psutil_process.memory_percent()}')
		print(f'Peak RAM usage till now is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}')
		if logger:
			logger.log(f'Theta_{iiter:d}', Theta)
			logger.log(f'Q_{iiter:d}', Q)
		else:
			# """
			M, sigma_yx_invs, Sigma_x_inv, delta_x = map(np.array, Theta[:-1])
			print('M = \n{}'.format(M[:4]))
			print('σ_yx_inv = {}'.format(np.array2string(sigma_yx_invs, formatter={'all': '{:.2e}'.format})))
			print('Σ_x_inv = \n{}'.format(np.array2string(Sigma_x_inv	, formatter={'all': '{:.2e}'.format})))
			print('δ_x = {}'		.format(np.array2string(delta_x			, formatter={'all': '{:.2e}'.format})))
			del M, sigma_yx_invs, Sigma_x_inv, delta_x
			prior_xs = Theta[-1]
			for prior_x in prior_xs:
				if prior_x[0] in ['Truncated Gaussian', 'Gaussian']:
					mu_x, sigma_x_inv = map(np.array, prior_x[1:])
					print('μ_x = {}'		.format(np.array2string(mu_x		, formatter={'all': '{:.2e}'.format})), end='\t')
					print('σ_xInv = {}'		.format(np.array2string(sigma_x_inv	, formatter={'all': '{:.2e}'.format})), end='\n')
					del mu_x, sigma_x_inv
				elif prior_x[0] in ['Exponential', 'Exponential shared', 'Exponential shared fixed']:
					lambda_x,  = map(np.array, prior_x[1:])
					print('lambda_x = {}'		.format(np.array2string(lambda_x	, formatter={'all': '{:.2e}'.format})))
					del lambda_x
				else:
					assert False
				del prior_x
			del prior_xs
			# """
			pass

		# stop_flag = False
		# if not Q <= best_Q:
		# 	best_Q, best_iter = Q, iiter
		# if iiter > best_iter + 30:
		# 	stop_flag = True

		print('Q = {:.4f},\tdiff Q = {:.2e}'.format(Q, Q-last_Q), end='')
		# if best_iter == iiter:
		# 	print('\tbest')
		# else:
		# 	print()
		print()

		last_Q = Q

		print('-'*10 + time.strftime(" %Y-%m-%d %H:%M:%S", time.localtime()) + '{:> 7d} '.format(iiter) + '-'*10)

		# if stop_flag:
		# 	break

def parse_arguments():
	parser = argparse.ArgumentParser()

	# dataset
	parser.add_argument(
		'--dataset', type=str,
		help='name of the dataset, ../data/<dataset> should be a folder containing a subfolder named \'files\''
	)
	parser.add_argument(
		'-K', type=int, default=20,
		help='Number of metagenes'
	)
	parser.add_argument(
		'--neighbor_suffix', type=str, default='',
		help='Suffix of the name of the file that contains interacting cell pairs'
	)
	parser.add_argument(
		'--expression_suffix', type=str, default='',
		help='Suffix of the name of the file that contains expressions'
	)
	parser.add_argument(
		'--exper_list', type=eval,
		help='list of names of the experiments, a Python expression, e.g., "[0,1,2]", "range(5)"'
	)
	parser.add_argument(
		'--use_spatial', type=eval,
		help='list of true/false indicating whether to use the spatial information in each experiment, '
			 'a Python expression, e.g., "[True,True]", "[False,False,False]", "[True]*5"'
	)

	# training & hyperparameters
	parser.add_argument('--lambda_SigmaXInv', type=float, default=1e-4, help='Regularization on Sigma_x^{-1}')
	parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of outer optimization iteration')
	parser.add_argument('--init_NMF_iter', type=int, default=10, help='2 * number of NMF iterations in initialization')
	parser.add_argument(
		'--beta', default=np.ones(1), type=np.array,
		help='Positive weights of the experiments; the sum will be normalized to 1; can be scalar (equal weight) or array-like'
	)

	parser.add_argument('--logger', default=False, action='store_true')

	return parser.parse_args()

if __name__ == '__main__':
	np.set_printoptions(linewidth=100000)

	args = parse_arguments()

	print(f'pid = {os.getpid()}')

	random_seed = 0
	np.random.seed(random_seed)
	print(f'random seed = {random_seed}')

	dataset = args.dataset
	K = args.K
	dataset_parameter = {
		'neighbor_suffix': args.neighbor_suffix,
		'expression_suffix': args.expression_suffix,
		'exper_list': args.exper_list,
		'use_spatial': args.use_spatial,
	}

	N = len(args.exper_list)
	betas = np.broadcast_to(args.beta, [N]).copy().astype(np.float)
	assert (betas>0).all()
	betas /= betas.sum()

	modelSpec = {}
	for key in ['lambda_SigmaXInv', 'max_iter', 'init_NMF_iter']:
		modelSpec[key] = getattr(args, key)

	modelSpec['nsample4integral'] = 64

	# modelSpec['X_sum2one'] = True		# not implemented
	modelSpec['X_sum2one'] = False
	modelSpec['M_sum2one'] = 'sum'
	# modelSpec['M_sum2one'] = 'L1'		# not implemented
	# modelSpec['M_sum2one'] = 'L2'		# not implemented
	# modelSpec['M_sum2one'] = 'None'	# not implemented
	# print('X_sum2one = {}'.format(X_sum2one))
	assert not modelSpec['X_sum2one'] or modelSpec['M_sum2one'] == 'None'

	# modelSpec['pairwise_potential_str'] = 'linear'
	# modelSpec['pairwise_potential_str'] = 'linear w/ shift'
	modelSpec['pairwise_potential_str'] = 'normalized'
	# print(f'pairwise_potential = {pairwise_potential_str}')

	# modelSpec['sigma_yx_inv_str'] = 'separate'
	modelSpec['sigma_yx_inv_str'] = 'average'
	# modelSpec['sigma_yx_inv_str'] = 'average 1'
	# print(f'sigma_yx_inv_str = {sigma_yx_inv_str}')

	modelSpec['dropout_str'] = 'origin'
	# modelSpec['dropout_str'] = 'pass'

	if args.logger:
		logger = Logger(dataset=dataset)
	else:
		logger = None

	main(
		dataset=dataset,
		K=K,
		betas=betas,
		dataset_parameter=dataset_parameter,
		modelSpec=modelSpec,
		logger=logger,
		**modelSpec,
	)
