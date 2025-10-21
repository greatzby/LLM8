# ALPLINE Template Code
This is the modified NanoGPT code for planning on graphs. To configure the environment, we use

    conda create --name gptenv --file spec-file.txt

# Simple Graphs

## Data Preparations

To create the dataset, we can use

    python data/simple_graph/create_graph.py

The (optional) configurations include

    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
    parser.add_argument('--edge_prob', type=float, default=0.1, help='Probability of creating an edge between two nodes')  
    parser.add_argument('--DAG', type=bool, default=True, help='Whether the graph should be a Directed Acyclic Graph')  
    parser.add_argument('--chance_in_train', type=float, default=0.5, help='Chance of a pair being in the training set')  
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  
 

 

Then we convert txt files to bin files by 
 
    python data/simple_graph/prepare_minigpt.py

The (optional) configurations include

    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  



## Model Training and Testing
To train the model, we run

    python train.py


The  (optional) configurations include 

    parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use') 
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers (default: 1)') 
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads (default: 1)')  
    parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings (default: 120)')
    parser.add_argument('--max_iters', type=int, default=10000, help='Number of Iterations (default: 10000)')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of Nodes (default: 100)')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of Paths (default: 1)')

The models are stored in "out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'"



To test the model, we run

    python test_simple.py 
The  (optional) configurations include 

    parser.add_argument('--ckpt_iter', type=int, default=10000)
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_nodes', type=int, default=100)
    parser.add_argument('--num_of_paths', type=int, default=20)




The results will be stored in "out_dir = f'out/{dataset}_{config}_{num_nodes}/pred_test_{ckpt_iter}.txt'", and will show the error type.
    

 