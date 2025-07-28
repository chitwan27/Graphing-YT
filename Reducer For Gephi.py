import pandas as pd
import networkx as nx

# Load full graph
edges = pd.read_csv("output/edges.csv")
nodes = pd.read_csv("output/nodes.csv")

# Build graph to compute degree
G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# Compute weighted degree
degree_dict = dict(G.degree(weight='weight'))
degree_df = pd.DataFrame(list(degree_dict.items()), columns=['id', 'weighted_degree'])

# Merge with node data
nodes = nodes.merge(degree_df, on='id')

# Keep top N nodes
N = 1000
top_nodes = set(nodes.nlargest(N, 'weighted_degree')['id'])

# Filter edges to only those where both nodes are in top N
edges_reduced = edges[
    edges['source'].isin(top_nodes) & edges['target'].isin(top_nodes)
].reset_index(drop=True)

# Filter nodes to only those in top N (in case some had no edges left)
nodes_reduced = nodes[nodes['id'].isin(top_nodes)].reset_index(drop=True)

# Export
edges_reduced.to_csv("output/edges_reduced.csv", index=False)
nodes_reduced.to_csv("output/nodes_reduced.csv", index=False)

print(f"Reduced to {len(nodes_reduced)} nodes and {len(edges_reduced)} edges.")
