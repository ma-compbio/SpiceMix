#!/usr/bin/env python
import random
import numpy as np

import scipy
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde, gamma, truncnorm, truncexpon, expon, bernoulli, dirichlet

from sklearn.decomposition import NMF

import umap
import pickle as pkl
import seaborn as sns
import pandas as pd
import networkx as nx
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from matplotlib.colors import ListedColormap
        
import seaborn as sns


def sample_gaussian(sigma, m, N=1):
    """

    TODO: Not exactly sure how this works.
    
    """
    K = len(sigma)
    assert sigma.shape[0] == sigma.shape[1]
    assert len(m) == K
    
    # Box-Muller Method
    L = np.linalg.cholesky(sigma)
    n_z = K + (K % 2)
    x = np.zeros((n_z, N))
    num_samples = 0
    while True:
        n_valid = 0
        while True:
            z = 2*np.random.rand(2) - 1
            if (z[0]**2 + z[1]**2 <= 1):
                r = np.linalg.norm(z)
                x[n_valid, num_samples] = z[0]*np.sqrt(-2*np.log(r**2)/r**2)
                x[n_valid + 1, num_samples] = z[1]*np.sqrt(-2*np.log(r**2)/r**2)
                n_valid += 2
            if n_valid == n_z:
                num_samples += 1
                break
        if num_samples == N:
            break
            
    # if K is odd, there will be one extra sample, so throw it away
    x = x[0:K, :]
    x = np.dot(L, x) + np.expand_dims(m, -1)
    
    return np.squeeze(x)

def sample_2D_points(num_points, minimum_distance):
    """Generate 2D samples that are at least minimum_distance apart from each other.
    
    """
    # TODO: Implement Poisson disc sampling for a vectorized operation
    
    points = np.zeros((num_points, 2))
    points[0] = np.random.random_sample(2)
    for index in range(1, num_points):
        while True:
            point = np.random.random_sample((1, 2))
            distances = cdist(points[:index], point)
            if np.min(distances) > minimum_distance:
                points[index] = point
                break
    return points


def generate_affinity_matrix(points, tau=1.0, method="delaunay"):
    """Given a set of 2D spatial coordinates, generate an affinity matrix.

    Can optionally use Delaunay triangulation or simple distance thresholding.
    """

    num_cells = len(points)
    if method == "delaunay":
        affinity_matrix = np.zeros((num_cells, num_cells))
        triangulation = Delaunay(points)
        for triangle in triangulation.simplices:
            affinity_matrix[triangle[0], triangle[1]] = 1
            affinity_matrix[triangle[1], triangle[2]] = 1
            affinity_matrix[triangle[2], triangle[0]] = 1
    else:
        disjoint_nodes = True
        while(disjoint_nodes):
            N = points.shape[0]
            # Construct graph
            distances = squareform(pdist(p))
            affinity_matrix = distances < tau
            identity_matrix = np.identity(N, dtype='bool')
            affinity_matrix = affinity_matrix * ~identity_matrix
            graph = nx.from_numpy_matrix(affinity_matrix)
            if not nx.is_connected(graph):
                # increase tau by 10% and repeat
                tau = 1.1*tau
                print('Graph is not connected, increasing tau to %s', tau)
            else:
                disjoint_nodes = False
                
    return affinity_matrix


def synthesize_metagenes(num_genes, num_real_metagenes, n_noise_metagenes, metagene_variation_probabilities, real_metagene_parameter, noise_metagene_parameter, normalize=True):
    """Synthesize related metagenes according to the metagene_variation_probabilities vector.
    
    Creates num_real_metagenes synthetic metagenes using a random Gamma distribution with
    shape parameter real_metagene_parameter. For each metagene i, if dropout_probabilities[i] != 0,
    randomly permutes a metagene_variation_probabilities[i] fraction of metagene i-1 to create metagene i;
    otherwise, creates a new random metagene. In addition, adds n_noise_metagenes parameterized by
    a Gamma distribution with shape parameter noise_metagene_parameter.
    """
    
    num_metagenes = num_real_metagenes + n_noise_metagenes 
    metagenes = np.zeros((num_metagenes, num_genes))
    
    for index in range(num_real_metagenes):
        variation_probability = metagene_variation_probabilities[index]
        if variation_probability == 0:
            metagene = gamma.rvs(real_metagene_parameter, size=num_genes)
            metagenes[index] = metagene
        else:
            mask = bernoulli.rvs(variation_probability, size=num_genes).astype('bool')
            perturbed_metagene = metagene
            perturbed_metagene[mask] = gamma.rvs(real_metagene_parameter, size=np.sum(mask))
        
            metagenes[index] = perturbed_metagene
            
    for index in range(num_real_metagenes, num_metagenes):
        metagenes[index] = gamma.rvs(noise_metagene_parameter, size=num_genes)
        
    metagenes = metagenes.T
    
    if normalize:
        metagenes = metagenes / np.sum(metagenes, axis=0)
        
    return metagenes


def synthesize_cell_embeddings(points, distributions, cell_type_definitions, mask_conditions, num_cells, n_noise_metagenes=3, signal_sigma_x=0.1, background_sigma_x=0.03, sigma_x_scale=1.0):
    """Generate synthetic cell embeddings.
    
    """
    
    num_patterns, num_cell_types, num_real_metagenes = cell_type_definitions.shape
    num_metagenes = num_real_metagenes + n_noise_metagenes
    
    sigma_x = np.concatenate([np.full(num_real_metagenes, signal_sigma_x), np.full(n_noise_metagenes, background_sigma_x)])
    sigma_x = sigma_x * sigma_x_scale

    cell_type = np.zeros((num_cells), dtype='int')
    Z = np.zeros((num_cells, num_metagenes))
    X = np.zeros((num_cells, num_metagenes))
    
    for pattern_index in range(len(mask_conditions)):
        pattern = mask_conditions[pattern_index](points)
        cell_type_definition = cell_type_definitions[pattern_index]
        distribution = distributions[pattern_index]
        
        cell_indices, = np.nonzero(pattern)
        random.shuffle(cell_indices)
        partition_indices = (np.cumsum(distribution) * len(cell_indices)).astype(int)
        partitions = np.split(cell_indices, partition_indices[:-1])
        
        for cell_type_index in range(len(cell_type_definition)):
            cell_type_composition = cell_type_definition[cell_type_index]
            partition = partitions[cell_type_index]
            cell_type[partition] = cell_type_index
            Z[partition, :num_real_metagenes] = cell_type_composition
        
   
    # Extrinsic factors
    Z[:, num_real_metagenes:num_metagenes] = 0.05

    # TODO: vectorize
    for cell in range(num_cells):
        for metagene in range(num_metagenes):
            X[cell, metagene] = sigma_x[metagene]*truncnorm.rvs(-Z[cell, metagene]/sigma_x[metagene], 100) + Z[cell, metagene]
    X = X * (Z > 0)
    X = X.T
    X = X / np.sum(X, axis=0)
    
    return X, cell_type

def perturb_genes(gene_expression, num_metagenes, first_threshold=.2, second_threshold=.2, shape=2.5):
    """Randomly perturb gene expression values.

    """
    genes, num_samples = gene_expression.shape
    random_sample_size = int(first_threshold * genes)
    for sample in range(num_samples):
        random_indices = random.sample(range(genes), random_sample_size)
        gene_expression[random_indices, sample] = gamma.rvs(shape, size=random_sample_size) / float(genes)
    random_sample_size = int(second_threshold * genes)
    indices = random.sample(range(genes), random_sample_size)
    gene_expression[indices, :] = (gamma.rvs(shape, size=(random_sample_size*num_samples)) / float(genes)).reshape((random_sample_size, num_samples))

    return gene_expression

class SyntheticDataset:
    """Synthetic mouse brain cortex dataset.
    
    This class provides methods for initializing a semi-random mouse cortex spatial
    transcriptomics dataset, as well as methods to visualize aspects of the dataset.
    
    """
    
    def __init__(self, distributions, cell_type_definitions, mask_conditions, metagene_variation_probabilities,
                 parameters, parent_directory, shared_metagenes=None, key=''):
        self.num_metagenes = parameters["num_real_metagenes"] + parameters['n_noise_metagenes']
        self.num_cells = parameters['n_cells']
        self.num_genes = parameters["n_genes"]
        
        # TODO: make color work for variable number of colors
        self.colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                       4: 'violet', 5: 'firebrick',
                       6: 'deepskyblue', 7: 'darkslateblue'}
        
        print('Synthesizing M...')
        self.metagenes = synthesize_metagenes(self.num_genes, parameters["num_real_metagenes"],
                                              parameters['n_noise_metagenes'], metagene_variation_probabilities,
                                              parameters["real_metagene_parameter"], parameters["noise_metagene_parameter"])
        
        print('Synthesizing X, A, and p...')
        self.num_replicates = parameters['num_replicates']
        self.sig_y = float(parameters['sigY_scale']) / self.num_genes
        self.Y = np.zeros((self.num_replicates, self.num_genes, self.num_cells))
        self.cell_embeddings = np.zeros((self.num_replicates, self.num_metagenes, self.num_cells))
        self.points = np.zeros((self.num_replicates, self.num_cells, 2))
        self.affinity_matrices = np.zeros((self.num_replicates, self.num_cells, self.num_cells))
        self.cell_types = np.zeros((self.num_replicates, self.num_cells))
        
        minimum_distance = 0.75 / np.sqrt(self.num_cells)
        tau = minimum_distance * 2.2
        for replicate in range(self.num_replicates):
            p_i = sample_2D_points(self.num_cells, minimum_distance)
            A_i = generate_affinity_matrix(p_i, tau)
            X_i, C_i = synthesize_cell_embeddings(p_i, distributions, cell_type_definitions, mask_conditions, self.num_cells)

            self.S = gamma.rvs(self.num_metagenes, scale=parameters['lambda_s'], size=self.num_cells)
            self.affinity_matrices[replicate] = A_i
            self.points[replicate] = p_i
            self.cell_embeddings[replicate] = X_i * self.S
            self.cell_types[replicate] = C_i

        print('Synthesizing Y...')
        for replicate in range(self.num_replicates):
            Y_i = np.matmul(self.metagenes, self.cell_embeddings[replicate])
            self.Y[replicate] = Y_i
            variance_y = (self.sig_y**2) * np.identity(self.num_genes)
            
            for cell in range(self.num_cells):
                self.Y[replicate][:, cell] = sample_gaussian(variance_y, Y_i[:, cell])
                
            if parameters['gene_replace_prob'] > 0 and parameters['element_replace_prob'] > 0:
                self.Y[replicate] = perturb_genes(Y_i, self.num_metagenes, first_threshold=parameters['gene_replace_prob'],
                                        second_threshold=parameters['element_replace_prob'],
                                        shape=self.num_metagenes)
                
        # gene_ind variable is just all genes -- we don't remove any
        # TODO: remove this field
        self.gene_ind = range(self.num_genes)
        
        # create empty Sig_x_inverse, since it is not used in this data generation
        self.sigma_x_inverse = np.zeros((self.num_metagenes, self.num_metagenes))

        data_subdirectory = 'synthetic_{}_{}_{:.0f}_{:.0f}_{:.0f}_{:.0f}_{}'
        data_subdirectory = data_subdirectory.format(self.num_cells, self.num_genes, parameters['sigY_scale']*10,
                                                         parameters['sigX_scale']*10, parameters['gene_replace_prob']*100,
                                                         parameters['element_replace_prob']*100, key)
        
        self.data_directory = Path(parent_directory) / data_subdirectory
        self.initialize_data_directory()
        
        print('Finished')

    def initialize_data_directory(self):
        """Initialize data directory structure on user file system.
        
        """
        (self.data_directory / "files").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "scripts").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "plots").mkdir(parents=True, exist_ok=True)

    def save_pickle(self):
        """Save dataset features as pickle files.
        
        """
        data_filename = 'synthetic_data.pkl'
        data_filepath = (self.data_directory / 'files' / data_filename)
        with open(data_filepath, 'wb') as f_out:
            pkl.dump([self.cell_embeddings, self.Y, self.affinity_matrices, self.points, self.metagenes, self.sig_y, self.sigma_x_inverse, self.gene_ind], f_out)
        
        cell_types_filename = 'synthetic_cell_types.pkl'
        cell_types_filepath = (self.data_directory / 'files' / cell_types_filename)
        with open(cell_types_filepath, 'wb') as f_out_C:
            pkl.dump([self.cell_types], f_out_C)

    def reformat(self):
        """Save dataset features to be compatible with downstream processing.
        
        """
        sigma_y_inverse = np.full(self.num_replicates, 1./self.sig_y)
        delta_x = np.zeros(self.num_metagenes)
        prior_xs = [('Exponential shared', np.ones(self.num_metagenes)) for _ in range(self.num_replicates)]
        scaling = [self.num_metagenes / self.Y[replicate].T.sum(axis=1).mean() for replicate in range(self.num_replicates)]
        scaled_X = [self.cell_embeddings[replicate].T * scaling[replicate] for replicate in range(self.num_replicates)]
        scaled_sigma_y_inverse = [sigma_y_inverse[replicate] / scaling[replicate] for replicate in range(self.num_replicates)]
        normalized_M = self.metagenes / self.metagenes.sum(axis=0, keepdims=True)

        np.savetxt(self.data_directory / 'files' / ('truth_M.txt'), self.metagenes, fmt='%.6f')
        for replicate in range(self.num_replicates):
            replicate_folder = self.data_directory / 'logs' / str(replicate)
            replicate_folder.mkdir(parents=True, exist_ok=True)

            Y_i = self.Y[replicate].T
            p_i = self.points[replicate]
            A_i = self.affinity_matrices[replicate]
            X_i = self.cell_embeddings[replicate].T
            
            _, lambda_x = prior_xs[replicate]
            lambda_x = lambda_x.mean()
            
            # TODO: these print statements are nice for debugging, but do we still need them?
            # print('G = {}'.format(self.num_genes))
            # print('mean Xik = {} ~ {}'.format(X_i.mean(), 1/lambda_x))
            # print('mean Yig = {} ~ {}'.format(Y_i.mean(), self.num_metagenes / lambda_x / self.num_genes))
            # print('mean power Yig = {}'.format((Y_i**2).mean()))
            # print('mean |Yi| = {} ~ {}'.format(Y_i.sum(1).mean(), self.num_metagenes/lambda_x))
            # print('RMSE Y = {} ~ {}'.format(np.sqrt(((Y_i.T - self.metagenes @ X_i.T)**2).mean()), self.sig_y))
            # print('SNR = {}'.format((Y_i**2).mean() / self.sig_y**2))
            # print('SNR (dB) = {}'.format(10*np.log((Y_i**2).mean() / (2*self.sig_y)**2)))
            
            np.savetxt(self.data_directory / 'files' / ('expression_{}.txt'.format(replicate)), Y_i, fmt='%.6f')
            np.savetxt(self.data_directory / 'files' / ('coordinates_{}.txt'.format(replicate)), p_i, fmt='%.6f')
            np.savetxt(self.data_directory / 'files'/ ('genes_{}.txt'.format(replicate)), np.arange(self.num_genes), fmt='%d')
            np.savetxt(self.data_directory/ 'files' / ('truth_X_{}.txt'.format(replicate)), X_i, fmt='%.6f')
            
            with open(self.data_directory / 'files' / ('neighborhood_{}.txt'.format(replicate)), 'w') as f:
                for (source, destination), adjacency in np.ndenumerate(A_i):
                    if adjacency == 1:
                        f.write('{}\t{}\n'.format(source, destination))

            np.savetxt(self.data_directory / 'files'/ ('genes_{}.txt'.format(replicate)), np.arange(self.num_genes), fmt='%d')

            with open(replicate_folder / 'H_0.pkl', 'wb') as f:
                pkl.dump((scaled_X[:replicate+1],), f, protocol=2,)

            with open(replicate_folder / 'Theta_0.pkl', 'wb') as f:
                pkl.dump(
                    (
                        normalized_M,
                        scaled_sigma_y_inverse[:replicate+1],
                        self.sigma_x_inverse,
                        delta_x,
                        prior_xs[:replicate+1],
                    ),f, protocol=2
                )

            with open(replicate_folder / 'Q_0.pkl', 'wb') as f:
                pkl.dump(0., f, protocol=2,)

    def plot_cells_UMAP(self, replicate=0, latent_space=False, cell_types=None,colors=None,save_figure=False, normalize=True):
        """Plot synthesized cells using UMAP.
        """
        # TODO: fix colors to be more compatible with variable cell types..
        
        if latent_space:
            gene_expression = self.cell_embeddings[replicate].T
        else:
            gene_expression = self.Y[replicate].T
            
        num_cells, num_features = gene_expression.shape
        C_i = self.cell_types[replicate]
        
        if not colors:
            colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                      4: 'violet', 5: 'firebrick', 6: 'gold',
                      7: 'deepskyblue', 8: 'darkslateblue', 9: 'gainsboro'}

        if normalize:
            gene_expression = (gene_expression - np.average(gene_expression, axis=0))
            gene_expression_std = gene_expression.std(axis=0)
            for feature in range(num_features):
                if gene_expression_std[feature] != 0:
                    gene_expression[:, feature] = np.divide(gene_expression[:, feature], gene_expression_std[feature])

        # TODO: cleanup unnecessary lines
        gene_expression_reduced = umap.UMAP(
                        n_components=2,
                        #         spread=1,
                        n_neighbors=10,
                        min_dist=0.3,
                        #         learning_rate=100,
                        #         metric='euclidean',
                        #         metric='manhattan',
                        #         metric='canberra',
                        #         metric='braycurtis',
                        #         metric='mahalanobis',
                        #         metric='cosine',
                        #         metric='correlation',
                        ).fit_transform(gene_expression)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0.1, 0.1, .8, .8])
        for color in np.unique(C_i):
            index = (C_i == color)
            ax.scatter(gene_expression_reduced[index, 0], gene_expression_reduced[index, 1], alpha=.7, c=colors[color])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")
        plt.show()
        
        if save_figure:
            plt.savefig(self.data_directory / "plots" / 'synthesized_data_umap.png')

    def plot_metagenes(self, order_genes=True):
        """Plot density map of metagenes in terms of constituent genes.
        
        """
        
        if order_genes:
            mclust = scipy.cluster.hierarchy.linkage(self.metagenes, 'ward')
            mdendro = scipy.cluster.hierarchy.dendrogram(mclust, no_plot=True)
            plt.imshow(self.metagenes[mdendro['leaves']], aspect='auto')
        else:
            plt.imshow(self.metagenes, aspect='auto')
            
        plt.xlabel('Metagene ID')
        plt.ylabel('Gene ID')
        plt.show()

    def plot_cell_types(self, replicate=0, save_figure=False, colors=None):
        """Plot cells in situ using cell type labels.
        
        """
        
        points = self.points[replicate]
        affinity_matrix = self.affinity_matrices[replicate]
        cell_types = self.cell_types[replicate]
        if not colors:
            colors = {0: 'sandybrown', 1: 'lightskyblue',
                      2: 'mediumspringgreen', 3: 'palegreen',
                      4: 'greenyellow', 5: 'darkseagreen',
                      6: 'burlywood', 7: 'orangered', 8: 'firebrick',
                      9: 'gold', 10: 'mediumorchid', 11: 'magenta',
                      12: 'palegoldenrod', 13: 'gainsboro', 14: 'teal',
                      15: 'darkslateblue'}
        df = pd.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'cell_type': cell_types})
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        
        for (source, destination) in zip(*np.where(affinity_matrix == 1)):
            plt.plot([points[source, 0], points[destination, 0]],
                [points[source, 1], points[destination, 1]], color="gray", linewidth=1)
        
        sns.scatterplot(data=df, x='X', y='Y', hue='cell_type', ax=ax, palette=colors,
                        legend=False, hue_order=list(set(cell_types)), size_norm=10.0)
        plt.show()

    def plot_metagenes_in_situ(self, replicate=0, save_figure=False):
        """Plot metagene values per cell in-situ.
        
        """
        
        points = self.points[replicate]
        cell_embeddings = self.cell_embeddings[replicate]
        for metagene in range(self.num_metagenes):
            fig = plt.figure(figsize=(5, 3))
            ax = fig.add_axes([.1, .1, .8, .8])
            ax.set_ylabel('Metagene {}'.format(metagene))
            sca = ax.scatter(points[:, 0], points[:, 1], c=cell_embeddings[metagene], s=23, cmap=plt.get_cmap('Blues'), vmin=0)
            fig.colorbar(sca)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
            if save_figure:
                plt.savefig(self.data_directory / "plots" / ('synthetic_metagene_{}.png'.format(metagene)))

    def plot_hidden_states(self, replicate=0, save_figure=False):
        """Plot synthetic cell embeddings.
        
        """
        cell_embeddings = self.cell_embeddings[replicate]
        image = plt.imshow(cell_embeddings, aspect='auto')
        plt.xlabel('Cell ID')
        plt.ylabel('Metagene ID')
        plt.colorbar(image)
        plt.show()


def synthesize_cell_expressions(points, distributions, cell_type_definitions, mask_conditions, num_genes, num_cells, signal_sigma_x=0.1, background_sigma_x=0.03, sigma_x_scale=1.0):
    """Generate synthetic cell expressions.
    
    """
    
    cell_types = np.zeros((num_cells), dtype=object)
    Y = np.zeros((num_cells, num_genes))
    
    partitioned_cells = 0
    for pattern_index in range(len(mask_conditions)):
        pattern = mask_conditions[pattern_index](points)
        cell_type_definition = cell_type_definitions[pattern_index]
        distribution = distributions[pattern_index]
        
        cell_indices, = np.nonzero(pattern)
        partition_indices = (np.cumsum(distribution) * len(cell_indices)).astype(int)
        partitions = np.split(cell_indices, partition_indices[:-1])
        
        cell_type_index = 0
        for cell_type_index, (cell_type, cell_type_composition) in enumerate(cell_type_definition.iteritems()):
            partition = partitions[cell_type_index]
            cell_types[partition] = cell_type
            partitioned_cells += len(partition)

            Y[partition, :num_genes] = cell_type_composition
    
    
   
    # # TODO: vectorize/ make add noise
    # for cell in range(num_cells):
    #     Y[cell, m] = sigma_x[metagene]*truncnorm.rvs(-Z[cell, metagene]/sigma_x[metagene], 100) + Z[cell, metagene]
    
    return Y.T, cell_types

class SyntheticEmpiricalDataset:
    """Synthetic mouse brain cortex dataset.
    
    This class provides methods for initializing a semi-random mouse cortex spatial
    transcriptomics dataset, as well as methods to visualize aspects of the dataset.
    
    """
    
    def __init__(self, distributions, cell_type_definitions, gene_names, mask_conditions,
                 parameters, parent_directory, shared_metagenes=None, key=''):
        self.num_cells = parameters['num_cells']
        self.num_genes = parameters["num_genes"]
        self.num_eigenvectors = 30
        
        # TODO: make color work for variable number of colors
        self.colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                       4: 'violet', 5: 'firebrick',
                       6: 'deepskyblue', 7: 'darkslateblue'}
        
        print('Synthesizing Y and p...')
        self.num_replicates = parameters['num_replicates']
        self.sig_y = float(parameters['sigY_scale']) / self.num_genes
        self.Y = np.zeros((self.num_replicates, self.num_genes, self.num_cells))
        self.points = np.zeros((self.num_replicates, self.num_cells, 2))
        self.affinity_matrices = np.zeros((self.num_replicates, self.num_cells, self.num_cells))
        self.cell_types = np.zeros((self.num_replicates, self.num_cells), dtype=object)
        self.cell_type_definitions = cell_type_definitions
        self.gene_names = gene_names
        self.cell_embeddings = np.zeros((self.num_replicates, self.num_cells, self.num_eigenvectors))
        
        minimum_distance = 0.75 / np.sqrt(self.num_cells)
        tau = minimum_distance * 2.2
        for replicate in range(self.num_replicates):
            print(f"Synthesizing replicate {replicate}")
            p_i = sample_2D_points(self.num_cells, minimum_distance)
            A_i = generate_affinity_matrix(p_i, tau)
            Y_i, C_i = synthesize_cell_expressions(p_i, distributions, self.cell_type_definitions, mask_conditions, self.num_genes, self.num_cells)

            
            self.S = gamma.rvs(self.num_genes, scale=parameters['lambda_s'], size=self.num_cells)
            self.points[replicate] = p_i
            self.affinity_matrices[replicate] = A_i
            self.Y[replicate] = Y_i * self.S
            variance_y = (self.sig_y**2) * np.identity(self.num_genes)
            
            for cell in range(self.num_cells):
                self.Y[replicate][:, cell] = sample_gaussian(variance_y, Y_i[:, cell])
                
            nmf = NMF(n_components=self.num_eigenvectors, max_iter=2000)
            embeddings = nmf.fit_transform(Y_i.T)

            self.cell_embeddings[replicate] = embeddings

            self.cell_types[replicate] = C_i
                
        # gene_ind variable is just all genes -- we don't remove any
        # TODO: remove this field
        self.gene_ind = range(self.num_genes)

        data_subdirectory = 'synthetic_{}_{}_{}'
        data_subdirectory = data_subdirectory.format(self.num_cells, self.num_genes, key)
        
        self.data_directory = Path(parent_directory) / data_subdirectory
        self.initialize_data_directory()
        
        print('Finished')

    def initialize_data_directory(self):
        """Initialize data directory structure on user file system.
        
        """
        (self.data_directory / "files").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "logs").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "scripts").mkdir(parents=True, exist_ok=True)
        (self.data_directory / "plots").mkdir(parents=True, exist_ok=True)

    def save_dataset(self):
        """Save dataset features to be compatible with downstream processing.
        
        """

        for replicate in range(self.num_replicates):
            replicate_folder = self.data_directory / 'logs' / str(replicate)
            replicate_folder.mkdir(parents=True, exist_ok=True)

            Y_i = self.Y[replicate].T
            p_i = self.points[replicate]
            A_i = self.affinity_matrices[replicate]
            cell_types_i = self.cell_types[replicate]
            gene_names_i = self.gene_names[replicate]
            
            np.savetxt(self.data_directory / 'files' / ('expression_{}.txt'.format(replicate)), Y_i, fmt='%.6f')
            np.savetxt(self.data_directory / 'files' / ('coordinates_{}.txt'.format(replicate)), p_i, fmt='%.6f')
            np.savetxt(self.data_directory / 'files'/ ('genes_{}.txt'.format(replicate)), gene_names_i, fmt='%s')
            np.savetxt(self.data_directory / 'files'/ ('labels_{}.txt'.format(replicate)), cell_types_i, fmt='%s')
            
            with open(self.data_directory / 'files' / ('neighborhood_{}.txt'.format(replicate)), 'w') as f:
                for (source, destination), adjacency in np.ndenumerate(A_i):
                    if adjacency == 1:
                        f.write('{}\t{}\n'.format(source, destination))

    def plot_cells_UMAP(self, replicate=0, latent_space=False, cell_types=None,colors=None,save_figure=False, normalize=True):
        """Plot synthesized cells using UMAP.
        """
        # TODO: fix colors to be more compatible with variable cell types..
        
        if latent_space:
            gene_expression = self.cell_embeddings[replicate]
        else:
            gene_expression = self.Y[replicate].T
            
        num_cells, num_features = gene_expression.shape
        C_i = self.cell_types[replicate]
        
        if not colors:
            colors = {0: 'darkkhaki', 1: 'mediumspringgreen', 2: 'greenyellow', 3: '#95bfa6',
                      4: 'violet', 5: 'firebrick', 6: 'gold',
                      7: 'deepskyblue', 8: 'darkslateblue', 9: 'gainsboro'}
            
        
        unique_cell_types = np.sort(np.unique(C_i))
        
        palette = sns.color_palette("husl", len(unique_cell_types))
        sns.set_palette(palette)
        
        colormap = ListedColormap(palette)

        if normalize:
            gene_expression = (gene_expression - np.average(gene_expression, axis=0))
            gene_expression_std = gene_expression.std(axis=0)
            for feature in range(num_features):
                if gene_expression_std[feature] != 0:
                    gene_expression[:, feature] = np.divide(gene_expression[:, feature], gene_expression_std[feature])

        # TODO: cleanup unnecessary lines
        gene_expression_reduced = umap.UMAP(
                        n_components=2,
                        #         spread=1,
                        n_neighbors=10,
                        min_dist=0.3,
                        #         learning_rate=100,
                        #         metric='euclidean',
                        #         metric='manhattan',
                        #         metric='canberra',
                        #         metric='braycurtis',
                        #         metric='mahalanobis',
                        #         metric='cosine',
                        #         metric='correlation',
                        ).fit_transform(gene_expression)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_axes([0.1, 0.1, .8, .8])
        for color, cell_type in np.ndenumerate(unique_cell_types):
            index = (C_i == cell_type)
            ax.scatter(gene_expression_reduced[index, 0], gene_expression_reduced[index, 1], alpha=.7, c=colormap(color), label=cell_type)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        plt.show()
        
        if save_figure:
            plt.savefig(self.data_directory / "plots" / 'synthesized_data_umap.png')

    def plot_cell_types(self, replicate=0, save_figure=False, colors=None):
        """Plot cells in situ using cell type labels.
        
        """
        
        points = self.points[replicate]
        affinity_matrix = self.affinity_matrices[replicate]
        cell_types = self.cell_types[replicate]

        unique_cell_types = np.sort(np.unique(cell_types))
        palette = sns.color_palette("husl", len(unique_cell_types))
        
        cell_type_mapping = {cell_type: index for index, cell_type in np.ndenumerate(unique_cell_types)}
        
        df = pd.DataFrame({'X': points[:, 0], 'Y': points[:, 1], 'cell_type': [cell_type_mapping[cell_type] for cell_type in cell_types]})
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        
        for (source, destination) in zip(*np.where(affinity_matrix == 1)):
            plt.plot([points[source, 0], points[destination, 0]],
                [points[source, 1], points[destination, 1]], color="gray", linewidth=1)
        
        top_ax = ax.twinx()
        top_ax.set_zorder(2)
        ax.set_zorder(1)
        sns.scatterplot(data=df, x='X', y='Y', hue='cell_type', ax=top_ax, palette=palette,
                        legend=False, s=80, size_norm=10.0)
        plt.show()
