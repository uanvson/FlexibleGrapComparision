import numpy as np
from typing import List, Tuple, Optional
from hmm import * #HMM, MAX_VERTICES, MAX_LENGTH, MAX_PROPERTIES, MAX_STATES, EPSILON

# Constants
MAX_GRAPHS = 5
MAX_PROPERTIES = 3
MAX_LABELS = 3
MAX_SLICES_NEIGHBORHOOD = 5
MIN_DOUBLE = 1e-80

class Node:
    def __init__(self, vertex: int):
        self.vertex = vertex
        self.next = None

class SimpleEdge:
    def __init__(self, src: int, dest: int):
        self.src = src
        self.dest = dest

class Element:
    def __init__(self, state: int, symbol: int, vertex: int):
        self.state = state
        self.symbol = symbol
        self.vertex = vertex

class MarkovChain:
    def __init__(self):
        self.nb_elements = 0
        self.elements = [Element(0, 0, 0) for _ in range(MAX_LENGTH)]

class Graph:
    def __init__(self, num_vertices: int, labels: List[int]):
        self.num_vertices = num_vertices
        self.adj_lists = [None] * num_vertices
        self.v_labels = labels[:num_vertices]
        self.visited_vertices = [0] * num_vertices

def size_adj_lists(adj_list: Optional[Node]) -> int:
    n = 0
    temp = adj_list
    while temp:
        n += 1
        temp = temp.next
    return n

def sort_adj_lists(graph: Graph) -> None:
    for vertex in range(graph.num_vertices):
        # Collect vertices in adjacency list
        tmp = graph.adj_lists[vertex]
        table = []
        while tmp:
            table.append(tmp.vertex)
            tmp = tmp.next
        # Sort vertices
        table.sort()
        # Rebuild adjacency list
        new_list = None
        for v in reversed(table):
            new_node = Node(v)
            new_node.next = new_list
            new_list = new_node
        graph.adj_lists[vertex] = new_list

def existing_label(l: int, label_list: List[int], nb_labels: int) -> int:
    for i in range(nb_labels):
        if l == label_list[i]:
            return i
    return -1

def get_next_graph(
    id_graph: int,
    f,
    nb_vertices: List[int],
    edges: List[SimpleEdge],
    nb_edges: List[int],
    label_list: List[int],
    nb_labels: List[int],
    labels: List[int],
    g
) -> None:
    if id_graph == 0:
        line = f.readline().strip()
        label_list.clear()
        for val in line.split(','):
            label_list.append(int(val))
            g.write(f"{val},")
        nb_labels[0] = len(label_list)
        g.write("\n")

    nb_vertices[0] = nb_edges[0] = 0
    line = f.readline()
    if not line:
        return

    line = line.strip()
    parts = line.split('[')
    nb_vertices[0] = int(parts[0])
    g.write(f"\n_________________________ graph G{id_graph + 1} __________________________\n")
    g.write(f"\n{nb_vertices[0]}[")

    # Read labels
    label_str = parts[1].split(']')[0]
    labels.clear()
    for i, val in enumerate(label_str.split(',')):
        l = int(val)
        k = existing_label(l, label_list, nb_labels[0])
        if k == -1:
            print(f"\n'{l}' is not a valid label\n")
            return
        labels.append(k)
        g.write(f"{label_list[labels[-1]]}{',' if i < len(label_str.split(',')) - 1 else ']'}")

    # Read edges
    edge_str = line.split('(')[1:]
    edges.clear()
    nb_edges[0] = 0
    for edge in edge_str:
        src, dest = map(int, edge.strip(')').split(','))
        edges.append(SimpleEdge(src, dest))
        g.write(f"({src},{dest})")
        nb_edges[0] += 1
    g.write("\n\n")

def print_mc(delta: MarkovChain) -> None:
    for i in range(delta.nb_elements):
        print(f"({delta.elements[i].state},{delta.elements[i].symbol})\t", end='')
    print()

def print_mc_file(delta: MarkovChain, f) -> None:
    for i in range(delta.nb_elements):
        f.write(f"({delta.elements[i].state},{delta.elements[i].symbol})")
    f.write("\n")

def print_mc_file_vertices(delta: MarkovChain, f) -> None:
    for i in range(delta.nb_elements):
        f.write(f"({delta.elements[i].state},{delta.elements[i].symbol})")
    f.write("    ****   ")
    for i in range(delta.nb_elements):
        f.write(f"({delta.elements[i].vertex + 1})")
    f.write("\n")

def create_node(v: int) -> Node:
    return Node(v)

def create_graph(vertices: int, labels: List[int]) -> Graph:
    graph = Graph(vertices, labels)
    for i in range(vertices):
        graph.adj_lists[i] = None
    return graph

def add_edge(graph: Graph, src: int, dest: int) -> None:
    # Add edge from src to dest
    new_node = create_node(dest)
    new_node.next = graph.adj_lists[src]
    graph.adj_lists[src] = new_node
    # Add edge from dest to src if not self-loop
    if src != dest:
        new_node = create_node(src)
        new_node.next = graph.adj_lists[dest]
        graph.adj_lists[dest] = new_node

def print_graph(graph: Graph, f, label_list: List[int]) -> None:
    for v in range(graph.num_vertices):
        temp = graph.adj_lists[v]
        f.write(f"vertex {v}({label_list[graph.v_labels[v]]}): [")
        vertices = []
        while temp:
            vertices.append(str(temp.vertex))
            temp = temp.next
        f.write(','.join(vertices) + "]\n")
    f.write("\n")

def index_label_equal(graph: Graph, vertex: int, nb_slices_neighborhood: int, label_list: List[int]) -> int:
    temp = graph.adj_lists[vertex]
    x = label_list[graph.v_labels[vertex]]
    m = size_adj_lists(graph.adj_lists[vertex])
    n = 0
    while temp:
        y = label_list[graph.v_labels[temp.vertex]]
        if x == y:
            n += 1
        temp = temp.next
    if n == 0:
        return 0
    z = 100.0 * n / (m + MIN_DOUBLE)
    pas = 100.0 / nb_slices_neighborhood
    return int(z / pas)

def property(graph: Graph, src: int, index_prop: int, nb_slices_neighborhood: int, label_list: List[int]) -> int:
    if index_prop == 0:
        return size_adj_lists(graph.adj_lists[src]) % 2
    elif index_prop == 1:
        return graph.v_labels[src]
    elif index_prop == 2:
        return index_label_equal(graph, src, nb_slices_neighborhood, label_list)
    return 0

def build_mc_dfs(
    graph: Graph,
    vertex: int,
    depth: List[int],
    mc_graph: List[MarkovChain],
    index: List[int],
    nb_properties: int,
    depth_max: int,
    nb_slices_neighborhood: int,
    label_list: List[int]
) -> None:
    graph.visited_vertices[vertex] = 1
    temp = graph.adj_lists[vertex]
    while temp:
        connected_vertex = temp.vertex
        if not graph.visited_vertices[connected_vertex]:
            depth[0] += 1
            if depth[0] <= depth_max:
                for index_prop in range(nb_properties):
                    mc_graph[index_prop].elements[index[0]].state = depth[0]
                    mc_graph[index_prop].elements[index[0]].symbol = property(
                        graph, connected_vertex, index_prop, nb_slices_neighborhood, label_list
                    )
                    mc_graph[index_prop].elements[index[0]].vertex = connected_vertex
                index[0] += 1
                build_mc_dfs(
                    graph, connected_vertex, depth, mc_graph, index, nb_properties,
                    depth_max, nb_slices_neighborhood, label_list
                )
            depth[0] -= 1
        temp = temp.next

def existing_mc(
    src: List[MarkovChain],
    taille: int,
    liste: List[List[MarkovChain]],
    nb_mcs: int,
    property_idx: int
) -> bool:
    for i in range(nb_mcs):
        if src[property_idx].nb_elements == liste[i][property_idx].nb_elements:
            n = sum(
                1 for j in range(src[property_idx].nb_elements)
                if (src[property_idx].elements[j].state == liste[i][property_idx].elements[j].state and
                    src[property_idx].elements[j].symbol == liste[i][property_idx].elements[j].symbol)
            )
            if n == src[property_idx].nb_elements:
                return True
    return False

def build_mcs(
    graph: Graph,
    mc_graph: List[List[MarkovChain]],
    candidates: np.ndarray,
    nb_properties: int,
    depth_max: int,
    T: List[int],
    f,
    nb_slices_neighborhood: int,
    nb_mcs: List[int],
    label_list: List[int]
) -> None:
    mc_graph1 = [[MarkovChain() for _ in range(nb_properties)] for _ in range(MAX_VERTICES)]
    taille = [0] * MAX_VERTICES

    for vertex in range(graph.num_vertices):
        graph.visited_vertices = [0] * graph.num_vertices
        depth = [0]
        index = [1]
        for index_prop in range(nb_properties):
            mc_graph1[vertex][index_prop].elements[0].state = depth[0]
            mc_graph1[vertex][index_prop].elements[0].symbol = property(
                graph, vertex, index_prop, nb_slices_neighborhood, label_list
            )
            mc_graph1[vertex][index_prop].elements[0].vertex = vertex
        build_mc_dfs(
            graph, vertex, depth, mc_graph1[vertex], index, nb_properties,
            depth_max, nb_slices_neighborhood, label_list
        )
        for index_prop in range(nb_properties):
            mc_graph1[vertex][index_prop].nb_elements = index[0]
        taille[vertex] = index[0]

    for index_prop in range(nb_properties):
        nb_mcs[index_prop] = 0
        for vertex in range(graph.num_vertices):
            if taille[vertex] >= 2:
                if not existing_mc(mc_graph1[vertex], taille[vertex], mc_graph, nb_mcs[index_prop], index_prop):
                    mc_graph[nb_mcs[index_prop]][index_prop].nb_elements = mc_graph1[vertex][index_prop].nb_elements
                    for j in range(mc_graph[nb_mcs[index_prop]][index_prop].nb_elements):
                        mc_graph[nb_mcs[index_prop]][index_prop].elements[j].state = (
                            mc_graph1[vertex][index_prop].elements[j].state
                        )
                        mc_graph[nb_mcs[index_prop]][index_prop].elements[j].symbol = (
                            mc_graph1[vertex][index_prop].elements[j].symbol
                        )
                        mc_graph[nb_mcs[index_prop]][index_prop].elements[j].vertex = (
                            mc_graph1[vertex][index_prop].elements[j].vertex
                        )
                    T[nb_mcs[index_prop]] = taille[vertex]
                    nb_mcs[index_prop] += 1

    for index_prop in range(nb_properties):
        f.write(f"MCs for P{index_prop}:\n")
        for j in range(nb_mcs[index_prop]):
            f.write(f"{j+1}:\t")
            print_mc_file_vertices(mc_graph[j][index_prop], f)
            for i in range(T[j]):
                candidates[index_prop, j, i] = mc_graph[j][index_prop].elements[i].symbol
        f.write("\n")

def initial_hmms(
    mc_graph: List[List[MarkovChain]],
    lambda_hmms: List[HMM],
    nb_mcs: List[int],
    depth_max: int,
    nb_properties: int,
    nb_slices_neighborhood: int,
    nb_labels: int
) -> None:
    for index_prop in range(nb_properties):
        lambda_hmms[index_prop].N = depth_max + 1
        if index_prop == 0:
            lambda_hmms[index_prop].M = 2
        elif index_prop == 1:
            lambda_hmms[index_prop].M = nb_labels
        elif index_prop == 2:
            lambda_hmms[index_prop].M = nb_slices_neighborhood + 1

    for index_prop in range(nb_properties):
        from_count = np.zeros(MAX_STATES)
        use_of_state = np.zeros(MAX_STATES)
        lambda_hmms[index_prop].Pi.fill(0)
        lambda_hmms[index_prop].A.fill(0)
        lambda_hmms[index_prop].B.fill(0)

        for i in range(nb_mcs[index_prop]):
            for j in range(mc_graph[i][index_prop].nb_elements):
                if j < mc_graph[i][index_prop].nb_elements - 1:
                    state = mc_graph[i][index_prop].elements[j].state
                    next_state = mc_graph[i][index_prop].elements[j + 1].state
                    symbol = mc_graph[i][index_prop].elements[j].symbol
                    from_count[state] += 1
                    use_of_state[state] += 1
                    lambda_hmms[index_prop].A[state, next_state] += 1.0
                    lambda_hmms[index_prop].B[state, symbol] += 1.0
                    if j == 0:
                        lambda_hmms[index_prop].Pi[state] += 1.0
                else:
                    state = mc_graph[i][index_prop].elements[j].state
                    symbol = mc_graph[i][index_prop].elements[j].symbol
                    use_of_state[state] += 1
                    lambda_hmms[index_prop].B[state, symbol] += 1.0

        lambda_hmms[index_prop].Pi[:lambda_hmms[index_prop].N] /= (nb_mcs[index_prop] + EPSILON)
        for i in range(lambda_hmms[index_prop].N):
            lambda_hmms[index_prop].A[i, :lambda_hmms[index_prop].N] /= (from_count[i] + EPSILON)
            lambda_hmms[index_prop].B[i, :lambda_hmms[index_prop].M] /= (use_of_state[i] + EPSILON)

        # Normalize Pi
        sum_pi = sum(lambda_hmms[index_prop].Pi[:lambda_hmms[index_prop].N])
        delta = abs(1.0 - sum_pi) / lambda_hmms[index_prop].N
        lambda_hmms[index_prop].Pi[:lambda_hmms[index_prop].N] += delta

        # Normalize A
        for i in range(lambda_hmms[index_prop].N):
            sum_a = sum(lambda_hmms[index_prop].A[i, :lambda_hmms[index_prop].N])
            delta = abs(1.0 - sum_a) / lambda_hmms[index_prop].N
            lambda_hmms[index_prop].A[i, :lambda_hmms[index_prop].N] += delta

        # Normalize B
        for j in range(lambda_hmms[index_prop].N):
            sum_b = sum(lambda_hmms[index_prop].B[j, :lambda_hmms[index_prop].M])
            delta = abs(1.0 - sum_b) / lambda_hmms[index_prop].M
            lambda_hmms[index_prop].B[j, :lambda_hmms[index_prop].M] += delta

        stationary_distribution(lambda_hmms[index_prop])

def euclidean_distance(x: np.ndarray, y: np.ndarray, n: int) -> float:
    return np.sqrt(sum((x[i] - y[i]) ** 2 for i in range(n)))

def manhattan_distance(x: np.ndarray, y: np.ndarray, n: int) -> float:
    return sum(abs(x[i] - y[i]) for i in range(n))