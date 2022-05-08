from functools import cached_property
import networkx as nx
import matplotlib.pyplot as plt
from math import inf


def buildGraph(bandwidths, latencies, tasks, nodes):
    """Builds a graph from a nxn matrix of node latencies/bandwidths; a list of 
    dictionaries of nodes with `id` and (maybe) `cpu`; and a list of 
    dictionaries of tasks with `id`, `cpu`, `input nodes` list and `input sizes` list."""

    G = nx.DiGraph()
    # add source node 's' and sink node 't'
    G.add_nodes_from([("s", {"type": "source"}), \
                      ("t", {"type": "sink"}),])

    # add physical nodes
    node_list = []
    for node in nodes:
        cpu = node["cpu"]
        node_list.append(node["id"])
        name = "n" + str(node["id"])
        G.add_node(name, type = "node", capacity = cpu)
        G.add_edge(name, "t", capacity = cpu, weight = 0) # connect to sink
    
    # add task nodes and connect to physical nodes
    for task in tasks: 
        cpu = task["cpu"]
        name = "t" + str(task["id"])
        input_nodes = task["input nodes"]
        input_sizes = task["input sizes"]
        assert(len(input_nodes) == len(input_sizes))   

        G.add_node(name, type = "task", capacity = cpu)

        G.add_edge("s", name, capacity = cpu, weight = 0) # connect to source

        for node in node_list: 
            node_name = "n" + str(node)
            # if assignment not possible, don't add edge
            if G.nodes[node_name]["capacity"] < cpu:
                continue
            cost = 0
            for i in range(len(input_nodes)):
                input_node = input_nodes[i]
                if input_node != node:
                    # for any two nodes, take max(latency, size / bandwidth) as the cost
                    cost += max(input_sizes[i] / bandwidths[node][input_node], latencies[node][input_node])
            G.add_edge(name, node_name, capacity = cpu, weight = cost)
    return G

def mcmf(G):
    minCostFlow = nx.max_flow_min_cost(G, "s", "t") 
    return minCostFlow, nx.cost_of_flow(G, minCostFlow)

# an example
latencies = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
bandwidths = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
nodes = [{"id": 0, "cpu": 4}, {"id": 1, "cpu": 4}, {"id": 2, "cpu": 2}]
tasks = [{"id": 0, "cpu": 1, "input nodes": [1, 2], "input sizes" : [10, 10]}, {"id": 1, "cpu": 1, "input nodes": [1], "input sizes" : [10]}, {"id": 2, "cpu": 1, "input nodes": [1], "input sizes" : [50]}]
G = buildGraph(bandwidths, latencies, tasks, nodes)
print(mcmf(G))
