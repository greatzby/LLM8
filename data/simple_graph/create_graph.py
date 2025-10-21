import networkx as nx
import random
import os
import argparse
import numpy

def generate_random_directed_graph(num_nodes, edge_prob):
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges to the graph based on the probability
    for i in range(num_nodes):
        for j in range(num_nodes):
            if DAG:
                if i < j and random.random() < edge_prob: 
                    G.add_edge(i, j)
            else:
                if i != j and random.random() < edge_prob:
                    G.add_edge(i, j)
    return G

def get_reachable_nodes(G, target_node):  
    # Get the transitive closure of the graph  
    TC = nx.transitive_closure(G)  
    # Find the predecessors in the transitive closure (nodes that can reach the target_node)  
    reachable_from = TC.predecessors(target_node)  
    return list(reachable_from)

def obtain_reachability():
    reachability = {}  
    pairs = 0
    for node in random_digraph.nodes():  
        reachability[node] = get_reachable_nodes(random_digraph, node)
        pairs += len(reachability[node])
    return reachability, pairs

def random_walk(source_node, target_node):
    stack = [source_node]
    visited = [] # to eliminate cycles

    while stack != []:
        cur_node = stack.pop()
        visited.append(cur_node)
        if cur_node == target_node:
            return visited

        adj = list(random_digraph.successors(cur_node))
        anc = list(reachability[target_node])
        anc.append(target_node)
        
        remaining = [element for element in adj if element in anc and element not in visited] #if we want the path to contain cycles, we should remove "and element not in visited"
    
        if len(remaining) == 0:
            return random_walk(source_node, target_node) # for non-DAGs

        next_node = random.choice(remaining)
        stack.append(next_node)

    return visited

def create_dataset(i):
    train_set = []
    test_set = []
    train_num_per_pair = max(i,1)
    for target_node in range(num_nodes):
        cnt = 0 # to avoid some target not appear in training dataset
        for source_node in range(target_node):
            if (data[source_node][target_node] == 1):
                if random_digraph.has_edge(source_node, target_node):
                    train_set.append([source_node, target_node,source_node, target_node])
                for ii in range(train_num_per_pair):
                    train_set.append([source_node, target_node] + random_walk(source_node, target_node) )
            if (data[source_node][target_node] == -1):
                test_set.append([source_node, target_node] + random_walk(source_node, target_node))
                    
    return train_set, test_set

def add_x(train_set, test_set):
    cnt = 0
    for target_node in range(num_nodes):
        for source_node in range(target_node):
            if source_node not in reachability[target_node]:
                cnt += 1

    prob_in_test = len(test_set) / cnt * 0.2
    prob_in_train = min(len(train_set) / cnt * 0.2, 1 - prob_in_test)
    train_repeat = max(int(len(train_set) / cnt * 0.15 / prob_in_train), 1)
    print(prob_in_train, prob_in_test, train_repeat)

    for target_node in range(num_nodes):
        for source_node in range(target_node):
            if source_node not in reachability[target_node]:
                coin = random.random()
                if coin < prob_in_train:
                    for _ in range(train_repeat):
                        train_set.append([source_node, target_node, 'x'])
                        
                elif coin > 1 - prob_in_test:
                    test_set.append([source_node, target_node, 'x'])

    return train_set, test_set

def obtain_stats(dataset):
    max_len = 0
    pairs = set()

    for data in dataset:
        max_len = max(max_len, len(data))
        pairs.add((data[0],data[-1]))

    len_stats = [0]*(max_len + 1) 
    
    for data in dataset:
        length = len(data)
        len_stats[length] += 1
        
    print('number of source target pairs:', len(pairs))
    for ii in range(3, len(len_stats)):
        print(f'There are {len_stats[ii]} paths with length {ii-3}')


def format_data(data):
    return f"{data[0]} {data[1]} " + ' '.join(str(num) for num in data[2:])  + '\n'
        
def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Generate a random graph based on the given parameters.')  
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
    parser.add_argument('--edge_prob', type=float, default=0.1, help='Probability of creating an edge between two nodes')  
    parser.add_argument('--DAG', type=bool, default=True, help='Whether the graph should be a Directed Acyclic Graph')  
    parser.add_argument('--chance_in_train', type=float, default=0.5, help='Chance of a pair being in the training set')  
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  

    args = parser.parse_args()  

    num_nodes = args.num_nodes
    edge_prob = args.edge_prob
    DAG = args.DAG
    chance_in_train = args.chance_in_train
    num_of_paths = args.num_of_paths

    random_digraph = generate_random_directed_graph(num_nodes, edge_prob)
    reachability, feasible_pairs = obtain_reachability()

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}')
    if not os.path.exists(folder_name):  
        os.makedirs(folder_name)

    data = numpy.zeros([num_nodes,num_nodes])
    for target_node in range(num_nodes):
        cnt = 0 # to avoid some target not appear in training dataset
        for source_node in range(target_node):
            if source_node in reachability[target_node]:
                if (random_digraph.has_edge(source_node, target_node)) or random.random() < chance_in_train or cnt < 1:
                    data[source_node][target_node] = 1
                    cnt += 1 
                else:
                    data[source_node][target_node] = -1
                #if (random_digraph.has_edge(source_node, target_node)):
                #    data[source_node][target_node] = 1
                #else:
                #    data[source_node][target_node] = -1
                
    train_set, test_set = create_dataset(num_of_paths)
        
    obtain_stats(train_set)
    print('number of source target pairs:', len(test_set))

    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}/train_{num_of_paths}.txt') )
    write_dataset(test_set, os.path.join(os.path.dirname(__file__),  f'{num_nodes}/test.txt') )
    nx.write_graphml(random_digraph, os.path.join(os.path.dirname(__file__), f'{num_nodes}/path_graph.graphml') )

    