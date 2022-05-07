import networkx as nx
import matplotlib.pyplot as plt

def buildGraph(latencies, tasks, nodes):
    """Builds a graph from a nxn matrix of node latencies and a dictionary of task ids to 
    (compute, input node list) tuples"""

    G = nx.DiGraph()
    # add source node 's' and sink node 't'
    G.add_nodes_from([("s", {"type": "source"}), \
                      ("t", {"type": "sink"}),])

    # add physical nodes
    for i in range(len(latencies[0])): 
        G.add_node("n" + str(i), type = "node")

    # add physical nodes and connect to sink
    for key in nodes:
        compute = nodes[key]
        name = "n" + str(key)
        G.add_node(name, type = "node")
        G.add_edge(name, "t", capacity = compute, weight = 0)

    
    # add task nodes and connect to physical nodes
    for key in tasks: 
        compute = tasks[key][0]
        input_nodes = tasks[key][1]
        name = "t" + str(key)
        G.add_node("t" + str(key), type = "task")

        # connect source to task 
        G.add_edge("s", name, capacity = compute, weight = 0)

        for i in range(len(nodes)):
            cost = 0
            for node in input_nodes:
                cost += latencies[i][node]
            G.add_edge(name, "n" + str(i), capacity = compute, weight = cost)
    return G

def mcmf(G):
    minCostFlow = nx.max_flow_min_cost(G, "s", "t")
    return minCostFlow, nx.cost_of_flow(G, minCostFlow)




# def test_graph_build(num_nodes, num_tasks, )

latencies = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
tasks = {0: (1, [1, 2]), 1: (2, [1]), 2: (1, [0, 2])}
nodes = {0: 2, 1: 2, 2: 1}
G = buildGraph(latencies, tasks, nodes)
print(mcmf(G))

# Expected Output
