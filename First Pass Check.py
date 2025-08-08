import sqlite3
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, triu
import networkx as nx
import community as community_louvain
import os
import gc  # For garbage collection

def connect_db(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    return sqlite3.connect(db_path)

def load_subscriptions(conn):
    query = """
    SELECT from_channel AS subscriber_id, to_channel AS subscribed_to_id
    FROM subscriptions
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        raise ValueError("No subscription data found.")
    return df

def load_channel_names(conn):
    return pd.read_sql_query("SELECT channel_id, title FROM channel_names", conn)

def build_co_subscription_matrix(subs_df):
    subscribers, sub_idx = pd.factorize(subs_df['subscriber_id'])
    channels, chan_idx = pd.factorize(subs_df['subscribed_to_id'])
    print(f"Total unique subscribers: {len(sub_idx)}")
    print(f"Total unique channels: {len(chan_idx)}")
    
    A = csr_matrix(
        (np.ones_like(subscribers), (subscribers, channels)),
        shape=(len(sub_idx), len(chan_idx))
    )
    return A, chan_idx

def generate_co_subscription_edges_chunked(A, chan_idx, min_weight=1, chunk_size=1000):
    """
    Memory-efficient co-subscription edge generation using chunking.
    Processes the matrix in chunks to avoid creating huge dense matrices.
    """
    print(f"Processing {A.shape[1]} channels in chunks of {chunk_size}")
    edge_list = []
    n_channels = A.shape[1]
    
    # Pre-transpose A once to avoid repeated transposition
    A_T = A.T
    print("Matrix transposed, starting chunked processing...")
    
    for i in range(0, n_channels, chunk_size):
        end_i = min(i + chunk_size, n_channels)
        print(f"Processing chunk {i//chunk_size + 1}/{(n_channels-1)//chunk_size + 1} (channels {i}-{end_i-1})")
        
        # Get chunk of channels
        A_chunk = A_T[i:end_i, :]  # Shape: (chunk_size, n_subscribers)
        
        # Compute co-subscriptions between chunk channels and ALL channels
        # This gives us chunk_size × n_channels matrix
        C_chunk = A_chunk @ A  # Shape: (chunk_size, n_channels)
        
        # Only keep upper triangle to avoid duplicates
        # For chunks beyond the first, we only want connections to channels with index > current chunk start
        rows, cols = C_chunk.nonzero()
        weights = C_chunk.data
        
        # Filter to avoid duplicate edges and self-loops
        valid_mask = cols > (rows + i)  # Only upper triangle relative to global indices
        rows = rows[valid_mask]
        cols = cols[valid_mask] 
        weights = weights[valid_mask]
        
        # Filter by minimum weight
        weight_mask = weights >= min_weight
        rows = rows[weight_mask]
        cols = cols[weight_mask]
        weights = weights[weight_mask]
        
        if len(rows) > 0:
            # Convert back to global channel indices
            global_rows = rows + i
            
            chunk_edges = pd.DataFrame({
                'source': chan_idx[global_rows],
                'target': chan_idx[cols],
                'weight': weights
            })
            edge_list.append(chunk_edges)
        
        # Force garbage collection after each chunk
        del C_chunk
        gc.collect()
        
        if len(edge_list) > 0:
            print(f"  Found {len(edge_list[-1])} edges in this chunk (total so far: {sum(len(df) for df in edge_list)})")
    
    if not edge_list:
        return pd.DataFrame(columns=['source', 'target', 'weight'])
    
    final_edge_list = pd.concat(edge_list, ignore_index=True)
    print(f"Final edge count: {len(final_edge_list)}")
    return final_edge_list

def run_louvain(edge_list):
    if edge_list.empty:
        print("No edges found, cannot run community detection")
        return pd.DataFrame(columns=['id', 'community'])
    
    G = nx.Graph()
    for _, row in edge_list.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=['id', 'community'])
    
    partition = community_louvain.best_partition(G, weight='weight')
    return pd.DataFrame(list(partition.items()), columns=['id', 'community'])

def add_channel_names(nodes_df, names_df):
    if nodes_df.empty:
        return nodes_df
    return nodes_df.merge(names_df, how='left', left_on='id', right_on='channel_id') \
                   .drop(columns=['channel_id'])

def export_for_gephi(edge_list, nodes_df, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    edge_list.to_csv(f"{output_dir}/edges.csv", index=False)
    nodes_df.to_csv(f"{output_dir}/nodes.csv", index=False)
    print(f"Exported edges and nodes to '{output_dir}/'")

def estimate_memory_usage(n_subscribers, n_channels, chunk_size):
    """Estimate memory usage for different parameters"""
    # Original matrix A: subscribers × channels (sparse)
    A_memory_mb = (n_subscribers * n_channels * 0.01 * 8) / (1024 * 1024)  # Assume 1% density
    
    # Chunk processing: chunk_size × n_channels (can be dense)
    chunk_memory_mb = (chunk_size * n_channels * 8) / (1024 * 1024)  # 8 bytes per float64
    
    print(f"Estimated memory usage:")
    print(f"  Original matrix A: ~{A_memory_mb:.1f} MB")
    print(f"  Per chunk processing: ~{chunk_memory_mb:.1f} MB")
    print(f"  Recommended chunk_size for <8GB RAM: {min(chunk_size, int(1000 * 1024 * 1024 / (n_channels * 8)))}")

def main():
    db_path = "youtube_graph.db"
    
    # Parameters you can adjust
    min_weight = 3  # Minimum shared subscribers
    chunk_size = 500  # Reduce if still running out of memory
    
    try:
        conn = connect_db(db_path)
        
        # Load and process data
        subs_df = load_subscriptions(conn)
        names_df = load_channel_names(conn)
        
        # Optional: Sample data for testing
        # subs_df = subs_df.sample(frac=0.1).reset_index(drop=True)  # Use 10% of data
        
        A, chan_idx = build_co_subscription_matrix(subs_df)
        
        # Estimate memory usage
        estimate_memory_usage(A.shape[0], A.shape[1], chunk_size)
        
        # Generate edges using chunked approach
        edge_list = generate_co_subscription_edges_chunked(A, chan_idx, min_weight=min_weight, chunk_size=chunk_size)
        
        # Run community detection
        nodes_df = run_louvain(edge_list)
        nodes_df = add_channel_names(nodes_df, names_df)
        
        # Export results
        export_for_gephi(edge_list, nodes_df)
        
        print("\nAnalysis completed successfully!")
        print(f"Found {len(edge_list)} co-subscription relationships")
        print(f"Identified {nodes_df['community'].nunique()} communities among {len(nodes_df)} channels")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()