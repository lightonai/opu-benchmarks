import os
import pathlib

from random import shuffle
from itertools import combinations

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

class Graph:
    """
    Attributes
    ----------
    n_nodes: int, number of nodes.
    clique_size: int, number of nodes for the clique.
    clique_step: int, number of steps for the creation of the clique.
    p_edges: float, probability of drawing an edge between two nodes.
    noise_ratio: float, noise strength as a percentage of the starting number of edges.
    save_path: float or None, save path for the animation images

    G: nx graph, graph object.
    n_noise_edges: int, max number of noise edges that can be added/removed.
    nodes_clique: list of int, nodes that make up the clique.
    edges_clique: list of tuples (int, int), edges that make up the clique.
    pos_dict: dict, contains the position of the nodes in 2D space. keys: nodes(int), values: tuples (int, int)
    nodes_color: dict,
     contains the node colors. The clique nodes will have different color. keys: nodes (int) values:colors (str)
    fig: matplotlib figure, figure for the animation.

    Methods
    -------
    initialize_clique: generates the main parameters for the clique (nodes, edges and colors).
    evolve: evolves the graph in time.
    create_clique: creates the clique
    update_plot: generates another frame for the animation
    __info__: prints the main parameters of the graph.

    """
    def __init__(self, n_nodes, clique_size, clique_step=1, p_edges=0.003, noise_ratio=10, save_path=None):
        """

        Parameters
        ----------

        n_nodes: int, number of nodes.
        clique_size: int, number of nodes for the clique.
        clique_step: int, number of steps for the creation of the clique.
        p_edges: float, probability of drawing an edge between two nodes.
        noise_ratio: float, noise strength as a percentage of the starting number of edges.
        save_path: float or None, save path for the animation images
        """
        self.n_nodes = n_nodes
        self.p_edges = p_edges
        self.clique_size = clique_size
        self.clique_step = clique_step
        self.noise_ratio = noise_ratio
        self.save_path = save_path

        self.G = nx.gnp_random_graph(n=self.n_nodes, p=self.p_edges)

        self.n_noise_edges = np.ceil(self.noise_ratio / 100 * len(self.G.edges))
        self.nodes_clique, self.edges_clique, self.pos_dict, self.nodes_color = self.initialize_clique()

        if self.save_path is not None:
            pathlib.Path(os.path.join(self.save_path, "animation")).mkdir(parents=True, exist_ok=True)
            self.fig = plt.figure(figsize=(9.5, 7.5))


    def initialize_clique(self):
        """

        Returns
        -------

        nodes_clique: list of int, nodes that make up the clique.
        edges_clique: list of tuples (int, int), edges that make up the clique.
        pos_dict: dict, contains the position of the nodes in 2D space. keys: nodes(int), values: tuples (int, int)
        nodes_color: dict,
        contains the node colors. The clique nodes will have different color. keys: nodes (int) values:colors (str)

        """
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

    def create_clique(self, progression=0):
        """
        Creates (gradually) the clique.

        Parameters
        ----------
        progression: int, stage of creation of the clique

        """
        edges_to_add = len(self.edges_clique)//self.clique_step * progression
        self.G.add_edges_from(self.edges_clique[:edges_to_add])

        return

    def update_plot(self, new_edges, eigenvec, newma, t_end):
        """
        Code to generate the animation. There are 3 panels: eigenvector (top-left), graph (top-right) and NEWMA (bottom)

        Parameters
        ----------
        new_edges: list of tuples (int, int), edges added at the current iteration
        eigenvec: numpy array or torch tensor, eigenvector of 2nd largest eigenvalue
        newma: newma object, needed to access the logs of the simulation (norm, threshold and number of edges)
        t_end: int, total length of the simulation.
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
        """ Prints the main info of the graph"""

        print("n_nodes = {}\tp_edges = {}\nclique_size = {}\tclique_step = {}\tnoise_ratio = {}\n"
              .format(self.n_nodes, self.p_edges, self.clique_size, self.clique_step, self.noise_ratio))
        return

class NEWMA:
    """
    Newma object to perform change point detection on graphs

    Attributes
    ----------
    n_nodes: int, number of nodes of the graph.
    n_components: int, number of random projections.
    time_window: int, time window for Newma.
    l_ratio: float, ratio between the forgetting factors.
    eta: float, forgetting factor for the threshold update.
    rescale_tau: float, rescaling factor for the threshold
    power_iter: int, number of power iterations for the power method.
    verbose: int, verbose level: 0->just the time/flag | 1-> time, threshold and norms too.
    save_path: None or str, save path for the logs

    self.lam: float, forgetting factor 1
    self.LAMBDA: float, forgetting factor 2

    self.threshold: float, threshold value. Initial value is set at 1 for no particular reason.
    self.norm: float, norm value. Initial value is set at 0 for no particular reason.
    self.norm_average: norm average.

    self.z1: torch tensor, matrix for the first EWMA
    self.z2: torch tensor, matrix for the second EWMA

    self.detection_flag: boolean, if True, a change point was detected.
    self.log: list, contains the parameters to log.

    """
    def __init__(self, n_nodes, n_components, time_window=20, l_ratio=8.5, eta=0.99, rescale_tau=1.07, power_iter=2,
                 verbose=1, save_path=None):
        """

        Parameters
        ----------
        n_nodes: int, number of nodes of the graph.
        n_components: int, number of random projections.
        time_window: int, time window for Newma.
        l_ratio: float, ratio between the forgetting factors.
        eta: float, forgetting factor for the threshold update.
        rescale_tau: float, rescaling factor for the threshold
        power_iter: int, number of power iterations for the power method.
        verbose: int, verbose level: 0->just the time/flag | 1-> time, threshold and norms too.
        save_path: None or str, save path for the logs

        """
        self.n_nodes = n_nodes
        self.time_window = time_window
        self.l_ratio = l_ratio
        self.n_components = n_components
        self.eta = eta
        self.rescale_tau = rescale_tau
        self.power_iter = power_iter
        self.verbose = verbose
        self.save_path = save_path
        self.data_columns = ["t", "n_edges", "norm", "norm_average", "threshold", "detection_flag", "generation_time", "proj_time"]

        self.lam = (self.l_ratio ** (1. / self.time_window) - 1) / (self.l_ratio ** ((self.time_window + 1) / self.time_window - 1))
        self.LAMBDA = self.l_ratio * self.lam

        self.threshold = 1
        self.norm = 0
        self.norm_average = 0
        self.z1 = torch.zeros((self.n_nodes, self.n_components), requires_grad=False)
        self.z2 = torch.zeros((self.n_nodes, self.n_components), requires_grad=False)
        self.detection_flag = False
        self.log = []

    def detect(self, Adj_matrix, t):
        """
        Applies NEWMA to detect a change point in the graph.

        Parameters
        ----------
        Adj_matrix: torch tensor, Adjacency matrix of the graph.
        t: int, current time step.

        """
        self.z1 = (1 - self.lam) * self.z1 + self.lam * Adj_matrix
        self.z2 = (1 - self.LAMBDA) * self.z1 + self.LAMBDA * Adj_matrix

        self.norm = np.linalg.norm(self.z1 - self.z2)

        if self.norm > self.threshold:
            print('Flag at t =', t)
            self.detection_flag = True
            self.den_mean = 1
        else:
            self.detection_flag = False
            self.den_mean = t

        return

    def update_threshold(self):
        """
        Updates the threshold for the change point detection.
        """
        self.threshold = self.rescale_tau * ((1 - self.eta) * self.norm_average + self.eta * self.norm)

        if self.den_mean == 1:
            self.norm_average = self.norm
        else:
            self.norm_average = (self.norm + self.den_mean * self.norm_average) / (self.den_mean + 1)
        return

    def compute_eigenvector_numpy(self, Adj_matrix):
        """
        Computes the eigenvector of the second largest eigenvalue

        Parameters
        ----------
        Adj_matrix: torch tensor, Adjacency matrix of the graph.

        Returns
        -------
        eigenvec: torch tensor, eigenvector of 2nd largest eigenvalue

        """
        #Normalization
        print(type(Adj_matrix))
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


    def compute_eigenvector(self, Adj_matrix):
        """
        Computes the eigenvector of the second largest eigenvalue.

        Parameters
        ----------
        Adj_matrix: torch tensor,
            Adjacency matrix of the graph

        """
        #Normalization

        d = Adj_matrix.sum(dim=1)
        D = torch.diag(d)
        norm_A_clique = torch.mm(D, torch.mm(Adj_matrix, D))

        I_A = torch.ones((self.n_nodes, self.n_nodes)) + norm_A_clique
        v1 = torch.ones(self.n_nodes) * 1.0 / np.sqrt(self.n_nodes)

        #Power method

        x = torch.randint(-1, 1, size=(self.n_nodes,), dtype=torch.float)
        eigenvec = x - torch.dot(v1, x) * v1

        for i in range(self.power_iter):
            eigenvec = torch.matmul(I_A, eigenvec)

        return eigenvec

    def update_log(self, t, n_edges,  generation_time, proj_time):
        """
        Prints the current state of the simulation and updates the log with the relevant information.

        Parameters
        ----------
        t: int, current time step.
        n_edges; int, number of edges at time t.
        generation_time: float, time to generate the random matrix.
        proj_time: float, time for the projection.

        """

        data = [t, n_edges, self.norm, self.norm_average, self.threshold, self.detection_flag, generation_time, proj_time]
        self.log.append(data)

        if self.verbose == 0:
            print('t = {0:4d}\tFlag={1}'.format(t, self.detection_flag))
        elif self.verbose == 1:
            print('Flag={0}\tt = {1:4d}\t# edges = {2:6d}\tnorm = {3:4.1f}\tprev_average = {4:4.1f}\ttau = {5:4.1f}\tproj_time = {6:2.3f}'
                  .format(self.detection_flag, t, n_edges, self.norm, self.norm_average, self.threshold, proj_time))

        if self.save_path is not None:
            df = pd.DataFrame(self.log, columns=self.data_columns)
            df.to_csv(os.path.join(self.save_path, "newma_{}.csv".format(self.n_nodes)), sep='\t', index=False)

        return
