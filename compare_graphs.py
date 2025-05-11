import sys
import time
import numpy as np
from typing import List
from dfsgraphs import *
#(
#    Graph, SimpleEdge, MarkovChain, create_graph, add_edge, sort_adj_lists, print_graph,
#    get_next_graph, build_mcs, initial_hmms, euclidean_distance, manhattan_distance
#)
from hmm import * #HMM, baum_welch, save_hmm_txt, vector_hmm, MAX_VERTICES, MAX_EDGES, MAX_PROPERTIES, MAX_SYMBOLS, MAX_LABELS, MAX_SLICES_NEIGHBORHOOD, MAX_DEPTH

def main(args: List[str]) -> int:
    if len(args) != 11:
        print(f"Usage: {args[0]} graphFile NbProperties epsilon MaxIterations Threshold OutputFile SaveInitHMMs SaveFinalHMMs NbSlicesNeighborhood DepthMaxDFS")
        return -1

    nb_properties = int(args[2])
    epsilon = float(args[3])
    max_iterations = int(args[4])
    threshold = float(args[5])
    save_init_hmms = int(args[7])
    save_final_hmms = int(args[8])

    nb_slices_neighborhood = int(args[9])
    if not (2 <= nb_slices_neighborhood <= MAX_SLICES_NEIGHBORHOOD):
        nb_slices_neighborhood = MAX_SLICES_NEIGHBORHOOD

    depth_max_dfs = int(args[10])
    if not (0 <= depth_max_dfs <= MAX_DEPTH):
        depth_max_dfs = MAX_DEPTH

    np.random.seed(int(time.time()))

    try:
        f = open(args[6], 'w')
    except Exception as e:
        print(f"Error opening {args[6]}: {e}")
        return -1

    try:
        g = open(args[1], 'r')
    except Exception as e:
        print(f"Error opening {args[1]}: {e}")
        return -1

    id_graph = 0
    nb_vertices = [0]
    edges = [SimpleEdge(0, 0) for _ in range(MAX_EDGES)]
    nb_edges = [0]
    label_list = [0] * MAX_LABELS
    nb_labels = [0]
    labels = [0] * MAX_VERTICES
    T = [0] * MAX_VERTICES
    nb_mcs = [0] * MAX_PROPERTIES
    eucli_dist = np.zeros((MAX_GRAPHS, MAX_GRAPHS))
    manhat_dist = np.zeros((MAX_GRAPHS, MAX_GRAPHS))
    vectors = np.zeros((MAX_GRAPHS, MAX_PROPERTIES, MAX_SYMBOLS))
    vector = np.zeros((MAX_GRAPHS, MAX_PROPERTIES * MAX_SYMBOLS))
    mc_graph = [[MarkovChain() for _ in range(MAX_PROPERTIES)] for _ in range(MAX_VERTICES)]
    lambda_init = [HMM() for _ in range(MAX_PROPERTIES)]
    lambda_hmms = [HMM() for _ in range(MAX_PROPERTIES)]
    candidates = np.zeros((MAX_PROPERTIES, MAX_VERTICES, MAX_LENGTH), dtype=int)

    get_next_graph(id_graph, g, nb_vertices, edges, nb_edges, label_list, nb_labels, labels, f)

    while nb_vertices[0] != 0:
        graph = create_graph(nb_vertices[0], labels)
        for j in range(nb_edges[0]):
            add_edge(graph, edges[j].src, edges[j].dest)
        sort_adj_lists(graph)
        print_graph(graph, f, label_list)

        build_mcs(
            graph, mc_graph, candidates, nb_properties, depth_max_dfs, T, f,
            nb_slices_neighborhood, nb_mcs, label_list
        )
        initial_hmms(
            mc_graph, lambda_init, nb_mcs, depth_max_dfs, nb_properties,
            nb_slices_neighborhood, nb_labels[0]
        )

        nb_components = 0
        for index_prop in range(nb_properties):
            if save_init_hmms:
                hmm_file = f"HmmInit_P{index_prop}_g{id_graph + 1}.txt"
                save_hmm_txt(lambda_init[index_prop], hmm_file)

            f.write(f"Training (P{index_prop}) ... ")
            start_time = time.time()
            lambda_hmms[index_prop], iterations = baum_welch(
                lambda_init[index_prop], candidates[index_prop, :nb_mcs[index_prop]],
                T[:nb_mcs[index_prop]], epsilon, max_iterations, nb_mcs[index_prop], threshold
            )
            duration = (time.time() - start_time) * 1000
            f.write(f": {iterations} iterations\t Time = {duration:.6g} ms\n")

            if save_final_hmms:
                hmm_file = f"Hmm_P{index_prop}_g{id_graph + 1}.txt"
                save_hmm_txt(lambda_hmms[index_prop], hmm_file)

            vectors[id_graph, index_prop] = vector_hmm(lambda_hmms[index_prop])
            f.write(f"Vector(P{index_prop}): [")
            for j in range(lambda_hmms[index_prop].M):
                vector[id_graph, nb_components] = vectors[id_graph, index_prop, j]
                f.write(f"{vectors[id_graph, index_prop, j]:.6g}")
                f.write("," if j < lambda_hmms[index_prop].M - 1 else "]")
                nb_components += 1
            f.write("\n")

        id_graph += 1
        get_next_graph(id_graph, g, nb_vertices, edges, nb_edges, label_list, nb_labels, labels, f)

    g.close()
    nb_graphs = id_graph

    f.write(f"\n_________________________ Vectors ({nb_components} components) __________________________\n\n")
    for i in range(nb_graphs):
        f.write(f"G{i + 1}:[")
        f.write(','.join(f"{vector[i, j]:.6g}" for j in range(nb_components - 1)))
        f.write(f"{vector[i, nb_components - 1]:.6g}]\n")

    for i in range(nb_graphs):
        for j in range(i + 1):
            if i != j:
                eucli_dist[i, j] = euclidean_distance(vector[i], vector[j], nb_components)
                manhat_dist[i, j] = manhattan_distance(vector[i], vector[j], nb_components)
            else:
                eucli_dist[i, j] = manhat_dist[i, j] = 0.0

    f.write("\n_________________________ Euclidean dist. __________________________\n\n")
    f.write("\t" + "\t".join(f"[G{i + 1}]" for i in range(nb_graphs)) + "\n")
    for i in range(nb_graphs):
        f.write(f"[G{i + 1}]\t")
        f.write("\t".join(f"{eucli_dist[i, j]:.6g}" for j in range(i + 1)) + "\n")

    f.write("\n_________________________ Manhattan dist. __________________________\n\n")
    f.write("\t" + "\t".join(f"[G{i + 1}]" for i in range(nb_graphs)) + "\n")
    for i in range(nb_graphs):
        f.write(f"[G{i + 1}]\t")
        f.write("\t".join(f"{manhat_dist[i, j]:.6g}" for j in range(i + 1)) + "\n")

    f.close()
    input("Press Enter to continue...")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))