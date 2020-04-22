import os
import pathlib

from random import shuffle
from itertools import combinations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

class Graph:

    def __init__(self, n_nodes, clique_size, p_edges=0.003, noise_ratio=10, save_path=None):
        self.n_nodes = n_nodes
        self.p_edges = p_edges
        self.clique_size = clique_size
        self.noise_ratio = noise_ratio
        self.save_path = save_path

        self.G = nx.gnp_random_graph(n=self.n_nodes, p=self.p_edges)

        self.n_noise_edges = np.ceil(self.noise_ratio / 100 * len(self.G.edges))
        self.nodes_clique, self.edges_clique, self.pos_dict, self.nodes_color = self.initialize_clique()

        if self.save_path is not None:
            pathlib.Path(os.path.join(self.save_path, "animation")).mkdir(parents=True, exist_ok=True)
            self.fig = plt.figure(figsize=(9.5, 7.5))


    def initialize_clique(self):
        nodes = list(self.G.nodes)

        shuffle(nodes)
        nodes_clique = nodes[0:self.clique_size]

        pos_list = [(-np.random.rand() * 5, -5 - np.random.rand() * 10) for i in range(self.n_nodes)]
        nodes_color = ['r' for i in range(self.n_nodes)]

        for i in range(len(nodes_clique)):
            pos_list[nodes_clique[i]] = (2 + np.random.rand() * 5, -5 - np.random.rand() * 10)
            nodes_color[nodes_clique[i]] = 'b'

        pos_dict = dict(zip(self.G.nodes(), pos_list))

        edges_clique = []

        for (u, v) in combinations(nodes_clique, r=2):
            edges_clique.append((u, v))

        return nodes_clique, edges_clique, pos_dict, nodes_color

    def evolve(self):
        """
        Evolves in time the graph G by adding and removing edges.

        Returns
        -------
        no_edges: list of tuples,
            edges that were added. Useful for plots
        """

        n_add = np.random.randint(0, self.n_noise_edges + 1)
        n_rem = np.random.randint(0, self.n_noise_edges + 1)

        edges = list(self.G.edges)
        shuffle(edges)
        self.G.remove_edges_from(edges[0:n_rem])

        no_edges = list(nx.non_edges(self.G))
        shuffle(no_edges)
        self.G.add_edges_from(no_edges[0:n_add])

        return no_edges[0:n_add]

    def create_clique(self):
        self.G.add_edges_from(self.edges_clique)
        return

    def update_plot(self, new_edges, eigenvec, newma, t_end):
        """
        Code to generate the animation. Still a WIP, can be written better.
        NOTE: create a folder named "animation" before running this.
        """

        data = pd.DataFrame(newma.log, columns=newma.data_columns)
        current_t = data["t"].iloc[-1]

        plt.clf()

        self.fig.suptitle("Time = {0:02d}".format(current_t), fontsize=16)
        gs = gridspec.GridSpec(2, 2)

        ax = self.fig.add_subplot(gs[0, 0])
        ax.scatter(list(self.G.nodes), np.abs(eigenvec * np.sqrt(self.n_nodes)), s=15, color=self.nodes_color)
        plt.grid(True)
        ax.set_xlabel('Component', fontsize=12)

        ax.set_xlim(-2, self.n_nodes)
        ax.set_ylim(bottom=0)
        ax.set_title('Eigenvector of 2nd largest eigenvalue')
        ax.axes.get_yaxis().set_ticks([])

        ax = self.fig.add_subplot(gs[0, 1])
        nx.draw_networkx_nodes(self.G, ax=ax, node_color=self.nodes_color, pos=self.pos_dict, node_size=15)
        nx.draw_networkx_edges(self.G, ax=ax, pos=self.pos_dict, width=0.1, edge_color='k')
        nx.draw_networkx_edges(self.G, ax=ax, pos=self.pos_dict, edgelist=new_edges, width=0.4, edge_color='r')

        ax.axis('off')

        ax = self.fig.add_subplot(gs[1, :])
        ax.plot(data["t"], data["norm"], label='Norm')
        ax.plot(data["t"], data["threshold"], label='Threshold')
        ax.plot(data["t"], data["n_edges"], label='Number of edges')

        ax.vlines(60, ymin=0, ymax=200, label='Community formation', color='k')

        plt.legend()
        plt.grid(True)
        ax.set_title('NEWMA', fontsize=20)
        ax.set_xlabel('Time')
        ax.set_xlim(0, t_end)

        if self.save_path is not None:
            img_name = os.path.join(self.save_path, "animation", 'anim_{0:02d}.png'.format(current_t))
            plt.savefig(img_name, format='png', bbox_inches='tight')

        return


    def __info__(self):
        print("n_nodes = {}, p_edges = {}, clique_size = {}, noise_ratio = {}"
              .format(self.n_nodes, self.p_edges, self.clique_size, self.noise_ratio))
        return

class NEWMA:
    def __init__(self, n_nodes, n_components, time_window=20, l_ratio=8.5, eta=0.99, rescale_tau=1.07, power_iter=2, save_path=None):
        self.n_nodes = n_nodes
        self.time_window = time_window
        self.l_ratio = l_ratio
        self.n_components = n_components
        self.eta = eta
        self.rescale_tau = rescale_tau
        self.power_iter = power_iter
        self.save_path = save_path
        self.data_columns = ["t", "n_edges", "norm", "norm_average", "threshold", "detection_flag"]

        self.lam = (self.l_ratio ** (1. / self.time_window) - 1) / (self.l_ratio ** ((self.time_window + 1) / self.time_window - 1))
        self.LAMBDA = self.l_ratio * self.lam

        self.threshold = 1
        self.norm = 0
        self.norm_average = 0
        self.z1, self.z2 = np.zeros((self.n_nodes, self.n_components)), np.zeros((n_nodes, self.n_components))
        self.detection_flag = False
        self.log = []

    def detect(self, Adj_matrix, t):

        self.z1 = (1 - self.lam) * self.z1 + self.lam * Adj_matrix
        self.z2 = (1 - self.LAMBDA) * self.z1 + self.LAMBDA * Adj_matrix

        self.norm = np.linalg.norm(self.z1 - self.z2)

        if self.norm > self.threshold:
            print('Flag at t =', t)
            self.detection_flag = True
            den_mean = 1
        else:
            self.detection_flag = False
            den_mean = t

        self.update_log(n_edges=np.sum(Adj_matrix)/2, t=t)

        self.threshold = self.rescale_tau * ((1 - self.eta) * self.norm_average + self.eta * self.norm)

        if den_mean == 1:
            self.norm_average = self.norm
        else:
            self.norm_average = (self.norm + den_mean * self.norm_average) / (den_mean + 1)

        return

    def compute_eigenvector(self, Adj_matrix):

        #Normalization

        d = np.sum(Adj_matrix, axis=1)
        D = np.diag(d)
        norm_A_clique = np.matmul(D, np.matmul(Adj_matrix, D))

        I_A = np.ones((self.n_nodes, self.n_nodes)) + norm_A_clique
        v1 = np.ones(self.n_nodes) * 1.0 / np.sqrt(self.n_nodes)

        #Power method

        x = np.random.randint(-1, 1, size=self.n_nodes)
        eigenvec = x - np.dot(v1, x) * v1
        for i in range(self.power_iter):
            eigenvec = np.dot(I_A, eigenvec)

        return eigenvec

    def update_log(self, n_edges, t):
        data = [t, n_edges, self.norm, self.norm_average, self.threshold, self.detection_flag]
        self.log.append(data)

        if self.save_path is not None:
            df = pd.DataFrame(self.log, columns=self.data_columns)
            df.to_csv(os.path.join(self.save_path, "newma_{}.csv".format(self.n_nodes)), sep='\t', index=False)