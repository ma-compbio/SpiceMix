import os, argparse, logging
from tqdm.auto import tqdm, trange
from pathlib import Path

import numpy as np
import torch

from model import SpiceMix
from util import config_logger


logger = config_logger(logging.getLogger(__name__))


def parse_arguments():
	parser = argparse.ArgumentParser()

	# dataset
	parser.add_argument(
		'--path2dataset', type=Path,
		help='Name of the dataset, ../data/<dataset> should be a folder containing a subfolder named \'files\''
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
		'--repli_list', type=lambda x: list(map(str, eval(x))),
		help='list of names of the FOVs, a Python expression, e.g., "[0,1,2]", "range(5)"'
	)
	parser.add_argument(
		'--use_spatial', type=eval,
		help='list of True/False indicating whether to use the spatial information in each FOV, '
			 'a Python expression, e.g., "[True,True]", "[False,False,False]", "[True]*5"'
	)

	# hyperparameters
	parser.add_argument(
		'-K', type=int, default=15,
		help='Number of metagenes'
	)
	parser.add_argument(
		'--initialization_method', type=str, default='louvain',
		help='Supports `louvain`, `kmeans`, `svd`, and `precomputed clusters`'
	)
	parser.add_argument(
		'--initialization_kwargs', type=eval, default=dict(),
		help='A dictionary specifying arguments for the initialization'
	)
	parser.add_argument(
		'--lambda_Sigma_x_inv', type=float, default=1e-6,
		help='Regularization on Σx^{-1}')
	parser.add_argument(
		'--power_Sigma_x_inv', type=float, default=2, help='Regularization on Σx^{-1}')
	parser.add_argument(
		'--max_iter', type=int, default=200, help='Maximum number of outer optimization iteration')
	parser.add_argument(
		'--init_NMF_iter', type=int, default=10, help='Number of NMF iterations in initialization')
	parser.add_argument(
		'--betas', default=np.ones(1), type=np.array,
		help='Positive weights of the experiments; the sum will be normalized to 1; can be scalar (equal weight) or array-like'
	)
	parser.add_argument('--lambda_x', type=float, default=0., help='Prior of X, ignored')

	parser.add_argument('--random_seed', type=int, default=0)

	parser.add_argument(
		'--device', type=str, default='cuda:0',
		help="Which device to use. The value of this parameter will be passed to `torch.device` or equivalent functions."
	)
	parser.add_argument('--num_threads', type=int, default=1, help='Number of CPU threads for PyTorch')

	parser.add_argument(
		'--result_filename', default=None, help='The name of the h5 file to store results'
	)

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_arguments()
	logger.info(f'pid = {os.getpid()}')
	np.random.seed(args.random_seed)
	logger.info(f'random seed = {args.random_seed}')
	torch.set_num_threads(args.num_threads)

	betas = np.broadcast_to(args.betas, [len(args.repli_list)]).copy().astype(float)
	assert (betas > 0).all()
	betas /= betas.sum()

	context = dict(device=args.device, dtype=torch.float64)

	obj = SpiceMix(
		K=args.K, lambda_Sigma_x_inv=args.lambda_Sigma_x_inv, power_Sigma_x_inv=args.power_Sigma_x_inv,
		repli_list=args.repli_list, betas=betas,
		context=context,
		# context_Y=dict(dtype=torch.float32, device='cpu'),
		context_Y=context,
		path2result=Path(args.result_filename),
	)
	obj.load_dataset(args.path2dataset)
	obj.initialize(
		method=args.initialization_method, kwargs=args.initialization_kwargs,
		random_state=args.random_seed,
	)

	for iiter in range(args.init_NMF_iter):
		iiter = 0 if iiter == args.init_NMF_iter - 1 else -1
		obj.estimate_weights(iiter=iiter, use_spatial=[False] * obj.num_repli)
		obj.estimate_parameters(iiter=iiter, use_spatial=[False] * obj.num_repli)
	obj.initialize_Sigma_x_inv()
	for iiter in range(1, args.max_iter+1):
		logger.info(f'Iteration {iiter}')
		obj.estimate_parameters(iiter=iiter, use_spatial=args.use_spatial)
		obj.estimate_weights(iiter=iiter, use_spatial=args.use_spatial)
