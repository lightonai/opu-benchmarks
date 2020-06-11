import os
import pathlib
import argparse
from argparse import RawTextHelpFormatter

import torch
import networkx as nx


from utils.graphs.newma import Graph, NEWMA
from utils.projections import get_random_features, GPU_matrix
from lightonml.projections.sklearn import OPUMap

def parse_args():
    parser = argparse.ArgumentParser(description="Change point detection on graphs with NEWMA.",
                                     formatter_class=RawTextHelpFormatter)

    # Graph arguments
    parser.add_argument("-n", '--n_nodes', help="Number of nodes of the graph.", type=int, default=1500)
    parser.add_argument("-p", '--p_edges', help="Probability of drawing an edge between two nodes.", type=float, default=0.002)
    parser.add_argument("-csz", '--clique_size', help="Size of the clique.", type=int, default=50)
    parser.add_argument("-cst", '--clique_step', help="Steps to create a clique.", type=int, default=10)
    parser.add_argument("-nr", '--noise_ratio', help="percentage of the total number of edges to add/remove as noise.",
                        type=int, default=1)
    parser.add_argument("-t", '--t_end', help="Duration of the simulation.", type=int, default=70)
    parser.add_argument("-tc", '--t_clique', help="Time of creation of the clique.", type=int, default=40)

    # For the plots
    parser.add_argument("-ts", '--t_start', help="Start of the plot. Useful if you want to skip the starting peak.",
                        type=int, default=0)

    #NEWMA arguments
    parser.add_argument("-tw", '--time_window', help="Time window for the detection of changes.", type=int, default=20)
    parser.add_argument("-lr", '--l_ratio', help="Ratio between the forgetting factors.", type=float, default=9.5)
    parser.add_argument("-eta", '--eta', help="Forgetting factor for threshold.", type=float, default=0.99)
    parser.add_argument("-r_tau", '--rescale_tau', help="rescaling factor for threshold.", type=float, default=1.1)
    parser.add_argument("-nc", '--n_components', help="Number of random projections.", type=int, default=20000)
    parser.add_argument("-pi", '--power_iter', help="Iterations for the Power method.", type=int, default=2)



    # Devices
    parser.add_argument("-d", "--device", help='Device for the Random projection.', type=str, choices=["cuda:0", "opu"],
                        default="opu")
    parser.add_argument("-m", "--GPU_memory",
                        help='Memory for the random projection if GPU is used. Used to optimize the matrix splits.',
                        type=int, default=10)


    parser.add_argument("-s", "--save_path", help='Path to the save folder. If None, results will not be saved. Default=None.',
                        type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):

    if args.save_path is not None:
        folder_name = os.path.join(args.save_path, "graph_nodes_{}_clique_{}_{}"
                                   .format(args.n_nodes, args.clique_size, args.device))
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    else:
        folder_name = None

    # Erase this, I hardcoded the number of components and the clique_size



    if args.device == "opu":
        R = None
        generation_time = 0.
        conv_blocks = 1
        opu_map = OPUMap(n_components=args.n_components)

    else:
        print("Generating random matrix of size ({} x {})".format(args.n_nodes, args.n_components))
        GPU_optimizer = GPU_matrix(n_samples=args.n_nodes, n_features=args.n_nodes, n_components=args.n_components,
                                   GPU_memory=args.GPU_memory)

        R, generation_time = GPU_optimizer.generate_RM()
        conv_blocks = args.n_nodes // GPU_optimizer.conv_blocks_size
        print("Generation time = {0:3.2f} s".format(generation_time))
        print("Splits size: R = {}\t conv = {}\n".format(GPU_optimizer.R_blocks_size, GPU_optimizer.conv_blocks_size))
        opu_map = None

    graph = Graph(n_nodes=args.n_nodes, p_edges=args.p_edges, clique_size=args.clique_size, clique_step=args.clique_step,
                  noise_ratio=args.noise_ratio, save_path=folder_name)
    graph.__info__()

    newma = NEWMA(args.n_nodes, args.n_components, time_window=args.time_window, l_ratio=args.l_ratio, eta=args.eta,
                  rescale_tau=args.rescale_tau, power_iter=args.power_iter, save_path=folder_name)


    with torch.no_grad():
        for t in range(1, args.t_end):
            new_edges = graph.evolve()

            if args.t_clique <= t <= args.t_clique + args.clique_step:
                graph.create_clique(progression=t - args.t_clique + 1)

            Adj_matrix = torch.FloatTensor(nx.to_numpy_array(graph.G))

            if args.device == "opu":
                Adj_matrix = Adj_matrix.bool()

            proj_time, random_features = get_random_features(Adj_matrix, args.n_components, opu_map=opu_map,
                                                             matrix=R, conv_blocks=conv_blocks, device=args.device)

            newma.detect(random_features.float(), t=t)

            eigenvector = newma.compute_eigenvector(Adj_matrix.float())

            newma.update_log(t, graph.G.number_of_edges(), generation_time, proj_time)
            if args.save_path is not None and t >= args.t_start:
                graph.update_plot(new_edges, eigenvector, newma, args.t_start, args.t_end, args.t_clique)
            newma.update_threshold()

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
