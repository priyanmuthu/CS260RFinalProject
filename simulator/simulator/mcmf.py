import networkx as nx
import matplotlib.pyplot as plt

def buildGraph(cluster, lnodes, pnodes):
    """Builds a graph from a nxn matrix of node latencies/bandwidths; a list of 
    dictionaries of nodes with `id` and (maybe) `cpu`; and a list of 
    dictionaries of tasks with `id`, `input nodes` list and `input sizes` list."""

    G = nx.DiGraph()
    # add source node 's' and sink node 't'
    G.add_nodes_from([("s", {"type": "source"}),
                        ("t", {"type": "sink"}), ])
    # add physical nodes
    for pnode in pnodes: 
        G.add_node(pnode.id, type = "pnode")
        G.add_edge(pnode.id, "t", capacity = 1, weight = 0)

    # add task nodes and connect to physical nodes
    for lnode in lnodes: 
        G.add_node(lnode.id, type = "lnode")
        G.add_edge("s", lnode, capacity = 1, weight = 0)

        for pnode in pnodes: 
            cost = 0
            for inp in lnode.input_q: 
                source = inp.from_pnode 
                bandwidth = cluster.get_bandwidth(pnode, source)
                latency = cluster.get_latency(pnode, source)
                cost += max(latency, inp.size / bandwidth)
            
            G.add_edge(lnode.id, pnode.id, capacity = 1, weight = cost)
    
    minCostFlow = nx.max_flow_min_cost(G, "s", "t")

    assignment = {}
    for lnode in lnodes:
        flow_edges = minCostFlow[lnode.id]
        for key in flow_edges:
            if flow_edges[key] > 0:
                assignment[lnode] = key
                break

    return assignment
