import os
import pathlib
import argparse
from argparse import RawTextHelpFormatter

from random import shuffle

import networkx as nx
import numpy as np
import time

from utils.graphs.newma import Graph, NEWMA

def parse_args():
    parser = argparse.ArgumentParser(description="Change point detection on graphs with NEWMA.",
                                     formatter_class=RawTextHelpFormatter)

    # Graph arguments
    parser.add_argument("-n", '--n_nodes', help="Number of nodes of the graph.", type=int, default=1000)
    parser.add_argument("-p", '--p_edges', help="Probability of drawing an edge between two nodes.", type=float, default=0.03)
    parser.add_argument("-csz", '--clique_size', help="Size of the clique.", type=int, default=20)
    parser.add_argument("-cst", '--clique_step', help="Steps to create a clique.", type=int, default=20)
    parser.add_argument("-nr", '--noise_ratio', help="percentage of the total number of edges to add/remove as noise.",
                        type=int, default=10)
    parser.add_argument("-t", '--t_end', help="Duration of the simulation.", type=int, default=100)
    parser.add_argument("-tc", '--t_clique', help="Time of creation of the clique.", type=int, default=60)

    #NEWMA arguments
    parser.add_argument("-tw", '--time_window', help="Time window for the detection of changes.", type=int, default=20)
    parser.add_argument("-lr", '--l_ratio', help="Ratio between the forgetting factors.", type=float, default=8.5)
    parser.add_argument("-eta", '--eta', help="Forgetting factor for threshold.", type=float, default=0.99)
    parser.add_argument("-r_tau", '--rescale_tau', help="rescaling factor for threshold.", type=float, default=1.07)
    parser.add_argument("-nc", '--n_components', help="Number of random projections.", type=int, default=1000)
    parser.add_argument("-pi", '--power_iter', help="Iterations for the Power method.", type=int, default=2)

    parser.add_argument("-s", "--save_path", help='Path to the save folder. If None, results will not be saved. Default=None.',
                        type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):

    if args.save_path is not None:
        folder_name = os.path.join(args.save_path, "graph_nodes_{}_clique_{}".format(args.n_nodes, args.clique_size))
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)


    graph = Graph(n_nodes=args.n_nodes, p_edges=args.p_edges, clique_size=args.clique_size, clique_step=args.clique_step,
                  noise_ratio=args.noise_ratio, save_path=folder_name)
    graph.__info__()

    newma = NEWMA(args.n_nodes, args.n_components, time_window=args.time_window, l_ratio=args.l_ratio, eta=args.eta,
                  rescale_tau=args.rescale_tau, power_iter=args.power_iter, save_path=folder_name)

    for t in range(1, args.t_end):
        new_edges = graph.evolve()
        #print('t = {0:4d}\t# edges = {1:6d}'.format(t, graph.G.number_of_edges()))

        if args.t_clique <= t <= args.t_clique + args.clique_step:
            graph.create_clique(progression=t - args.t_clique + 1)

        Adj_matrix = nx.to_numpy_array(graph.G)

        newma.detect(Adj_matrix, t=t)
        eigenvector = newma.compute_eigenvector(Adj_matrix)

        graph.update_plot(new_edges, eigenvector, newma, args.t_end)
        print(newma.log[-1])
        #print('t = {0:4d}\t# edges = {1:6d}\tnorm = {2:4.1f}\tprev_average = {3:4.1f}\ttau = {4:4.1f}'
        #      .format(t, graph.G.number_of_edges(), norm, S_bar, threshold))
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
