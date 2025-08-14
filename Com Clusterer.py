import sqlite3
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, triu
import networkx as nx
import os
import datetime

# Try to import available community detection libraries
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

class AnalysisConfig:
    """Configuration class for analysis parameters"""
    def __init__(self):
        # Database settings
        self.db_path = "youtube_graph.db"
        self.output_dir = "output"
        
        # Channel popularity filtering
        self.popularity_threshold = 100
        self.popularity_metric = "subscriber_count"  # "subscriber_count" or "custom"
        
        # Edge filtering
        self.min_weight = 3
        
        # Community detection
        self.algorithm = "louvain_networkx"  # "louvain_networkx", "leiden_igraph"
        self.resolution = 1.0
        self.random_seed = 42
        self.iterations = -1  # -1 for until convergence

def connect_db(db_path):
    """Connect to SQLite database"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    return sqlite3.connect(db_path)

def load_subscriptions(conn):
    """Load subscription data from database"""
    query = """
    SELECT from_channel AS subscriber_id, to_channel AS subscribed_to_id
    FROM subscriptions
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        raise ValueError("No subscription data found.")
    return df

def load_channel_names(conn):
    """Load channel names from database"""
    return pd.read_sql_query("SELECT channel_id, title FROM channel_names", conn)

def filter_popular_channels(subs_df, threshold=100, metric="subscriber_count"):
    """
    Filter channels based on popularity criteria
    
    Parameters:
    - subs_df: subscription dataframe
    - threshold: minimum threshold for filtering
    - metric: "subscriber_count" or "custom" for custom filtering logic
    """
    if metric == "subscriber_count":
        # Count subscribers per channel
        channel_counts = subs_df.groupby('subscribed_to_id').size()
        popular_channels = channel_counts[channel_counts > threshold].index
        filtered_df = subs_df[subs_df['subscribed_to_id'].isin(popular_channels)]
        
        print(f"Channels with >{threshold} subscribers: {len(popular_channels):,}")
        print(f"Subscriptions after filtering: {len(filtered_df):,} (was {len(subs_df):,})")
        
    elif metric == "custom":
        # Custom filtering logic - can be extended
        channel_counts = subs_df.groupby('subscribed_to_id').size()
        
        # Channels with sufficient subscribers
        min_subs = channel_counts > threshold
        
        popular_channels = channel_counts[min_subs].index
        filtered_df = subs_df[subs_df['subscribed_to_id'].isin(popular_channels)]
        
        print(f"Channels passing custom filter: {len(popular_channels):,}")
        print(f"Subscriptions after filtering: {len(filtered_df):,} (was {len(subs_df):,})")
    
    else:
        raise ValueError(f"Unknown popularity metric: {metric}")
    
    return filtered_df

def build_co_subscription_matrix(subs_df):
    """Build sparse co-subscription matrix"""
    subscribers, sub_idx = pd.factorize(subs_df['subscriber_id'])
    channels, chan_idx = pd.factorize(subs_df['subscribed_to_id'])
    
    print(f"Total unique subscribers: {len(sub_idx):,}")
    print(f"Total unique channels: {len(chan_idx):,}")
    
    A = csr_matrix(
        (np.ones_like(subscribers), (subscribers, channels)),
        shape=(len(sub_idx), len(chan_idx))
    )
    return A, chan_idx

def generate_co_subscription_edges(A, chan_idx, min_weight=1):
    """Generate co-subscription edges from matrix"""
    print("Computing co-subscription matrix...")
    C = A.T @ A  # co-subscription matrix (channels x channels)
    C = triu(C, k=1)  # upper triangle to avoid duplicates & self-loops
    
    rows, cols = C.nonzero()
    weights = C.data
    
    edge_list = pd.DataFrame({
        'source': chan_idx[rows],
        'target': chan_idx[cols],
        'weight': weights
    })
    
    edge_list = edge_list[edge_list['weight'] >= min_weight].reset_index(drop=True)
    print(f"Filtered edges (weight ≥ {min_weight}): {len(edge_list):,}")
    return edge_list

def run_community_detection(edge_list, algorithm="louvain_networkx", resolution=1.0, 
                                      random_seed=42, iterations=-1):
    """
    Run community detection with specified algorithm and parameters
    
    Parameters:
    - algorithm: "louvain_networkx", "leiden_igraph"
    - resolution: resolution parameter for community detection
    - random_seed: random seed for reproducibility
    - iterations: number of iterations (-1 for convergence)
    """
    if edge_list.empty:
        print("No edges found, cannot run community detection")
        return pd.DataFrame(columns=['id', 'community'])
    
    print(f"Running {algorithm} community detection...")
    print(f"Parameters: resolution={resolution}, seed={random_seed}")
    
    if algorithm == "louvain_networkx":
        return run_louvain_networkx(edge_list, resolution, random_seed)
    elif algorithm == "leiden_igraph":
        return run_leiden_igraph(edge_list, resolution, random_seed, iterations)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def run_louvain_networkx(edge_list, resolution=1.0, random_seed=42):
    """Run Louvain algorithm using NetworkX"""
    if not LOUVAIN_AVAILABLE:
        raise ImportError("python-louvain not available. Install with: pip install python-louvain")
    
    # Build NetworkX graph
    G = nx.Graph()
    for _, row in edge_list.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    print(f"Graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    
    # Run Louvain
    partition = community_louvain.best_partition(
        G, 
        weight='weight', 
        resolution=resolution,
        random_state=random_seed
    )
    
    communities = len(set(partition.values()))
    print(f"Found {communities:,} communities")
    
    return pd.DataFrame(list(partition.items()), columns=['id', 'community'])

def run_leiden_igraph(edge_list, resolution=1.0, random_seed=42, iterations=-1):
    """Run Leiden algorithm using igraph"""
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph not available. Install with: pip install python-igraph")

    # Build igraph
    all_nodes = pd.concat([edge_list['source'], edge_list['target']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    edges = [(node_to_idx[row['source']], node_to_idx[row['target']])
             for _, row in edge_list.iterrows()]
    weights = edge_list['weight'].tolist()

    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    g.vs['name'] = all_nodes

    print(f"Graph has {g.vcount():,} nodes and {g.ecount():,} edges")

    # Set seed for reproducibility, as Leiden uses a random element
    np.random.seed(random_seed)
    
    # Run Leiden
    partition = g.community_leiden(
        weights='weight',
        resolution_parameter=resolution,
        n_iterations=iterations
    )

    print(f"Found {len(partition):,} communities")
    print(f"Modularity: {partition.modularity:.3f}")

    return pd.DataFrame({'id': all_nodes, 'community': partition.membership})

def add_channel_names(nodes_df, names_df):
    """Add channel names to nodes dataframe"""
    if nodes_df.empty:
        return nodes_df
    
    result = nodes_df.merge(names_df, how='left', left_on='id', right_on='channel_id') \
                     .drop(columns=['channel_id'])
    
    named_count = result['title'].notna().sum()
    print(f"Added names for {named_count:,}/{len(result):,} channels")
    
    return result

def export_for_gephi(edge_list, nodes_df, output_dir='output'):
    """Export results for Gephi visualization with timestamped filenames"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    edge_path = f"{output_dir}/edges_{timestamp}.csv"
    node_path = f"{output_dir}/nodes_{timestamp}.csv"

    edge_list.to_csv(edge_path, index=False)
    nodes_df.to_csv(node_path, index=False)

    print(f"Exported {len(edge_list):,} edges and {len(nodes_df):,} nodes to '{output_dir}/' with timestamp {timestamp}")

def print_analysis_summary(edge_list, nodes_df):
    """Print summary of analysis results"""
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Edges: {len(edge_list):,}")
    print(f"Nodes: {len(nodes_df):,}")
    
    if not nodes_df.empty and 'community' in nodes_df.columns:
        communities = nodes_df['community'].nunique()
        print(f"Communities: {communities:,}")
        
        if communities > 0:
            community_sizes = nodes_df['community'].value_counts()
            print(f"Largest community: {community_sizes.iloc[0]:,} channels")
            print(f"Average community size: {community_sizes.mean():.1f}")
    
    if not edge_list.empty:
        print(f"Weight range: {edge_list['weight'].min()}-{edge_list['weight'].max()}")
        print(f"Average weight: {edge_list['weight'].mean():.1f}")
    
    print("="*50)

def create_config_from_input():
    """Create configuration from user input"""
    config = AnalysisConfig()
    
    print("YouTube Co-Subscription Network Analysis Configuration")
    print("-" * 55)
    
    # Database path
    db_input = input(f"Database path [{config.db_path}]: ").strip()
    if db_input:
        config.db_path = db_input
    
    # Popularity filtering
    print(f"\nChannel Popularity Filtering:")
    threshold_input = input(f"Minimum subscribers [{config.popularity_threshold}]: ").strip()
    if threshold_input:
        config.popularity_threshold = int(threshold_input)
    
    # Edge filtering
    print(f"\nEdge Filtering:")
    weight_input = input(f"Minimum co-subscription weight [{config.min_weight}]: ").strip()
    if weight_input:
        config.min_weight = int(weight_input)
    
    # Algorithm selection
    print(f"\nCommunity Detection Algorithm:")
    algorithms = []
    if LOUVAIN_AVAILABLE:
        algorithms.append("louvain_networkx")
    if IGRAPH_AVAILABLE:
        algorithms.append("leiden_igraph")
    
    if algorithms:
        for i, alg in enumerate(algorithms, 1):
            print(f"  {i}. {alg}")
        
        alg_choice = input(f"Choose algorithm [1]: ").strip()
        if alg_choice and alg_choice.isdigit():
            idx = int(alg_choice) - 1
            if 0 <= idx < len(algorithms):
                config.algorithm = algorithms[idx]
    else:
        print("No community detection libraries available!")
        return None
    
    # Advanced parameters
    advanced = input(f"\nConfigure advanced parameters? (y/n) [n]: ").strip().lower()
    if advanced == 'y':
        resolution_input = input(f"Resolution parameter [{config.resolution}]: ").strip()
        if resolution_input:
            config.resolution = float(resolution_input)
        
        seed_input = input(f"Random seed [{config.random_seed}]: ").strip()
        if seed_input:
            config.random_seed = int(seed_input)
    
    return config

def main():
    """Main analysis function"""
    print("YouTube Co-Subscription Network Analysis")
    print("=" * 40)
    
    # Get configuration
    use_config = input("Use interactive configuration? (y/n) [n]: ").strip().lower()
    
    if use_config == 'y':
        config = create_config_from_input()
        if config is None:
            print("Configuration failed. Exiting.")
            return
    else:
        # Use default configuration
        config = AnalysisConfig()
        print(f"Using default configuration:")
        print(f"  - Database: {config.db_path}")
        print(f"  - Min subscribers: {config.popularity_threshold}")
        print(f"  - Min weight: {config.min_weight}")
        print(f"  - Algorithm: {config.algorithm}")
    
    try:
        # Connect to database
        print(f"\nConnecting to database: {config.db_path}")
        conn = connect_db(config.db_path)
        
        # Load and process data
        print("Loading subscription data...")
        subs_df = load_subscriptions(conn)
        print(f"Loaded {len(subs_df):,} subscription records")
        
        print("Loading channel names...")
        names_df = load_channel_names(conn)
        print(f"Loaded {len(names_df):,} channel names")
        
        # Filter popular channels
        print(f"\nFiltering channels (>{config.popularity_threshold} subscribers)...")
        subs_df = filter_popular_channels(
            subs_df, 
            threshold=config.popularity_threshold,
            metric=config.popularity_metric
        )
        
        # Build co-subscription matrix
        print("\nBuilding co-subscription matrix...")
        A, chan_idx = build_co_subscription_matrix(subs_df)
        
        # Generate edges
        print(f"\nGenerating co-subscription edges...")
        edge_list = generate_co_subscription_edges(A, chan_idx, min_weight=config.min_weight)
        
        # Run community detection
        print(f"\nRunning community detection...")
        nodes_df = run_community_detection(
            edge_list, 
            algorithm=config.algorithm,
            resolution=config.resolution,
            random_seed=config.random_seed,
            iterations=config.iterations
        )
        
        # Add channel names
        print("\nAdding channel names...")
        nodes_df = add_channel_names(nodes_df, names_df)
        
        # Export results
        print(f"\nExporting results...")
        export_for_gephi(edge_list, nodes_df, config.output_dir)
        
        # Print summary
        print_analysis_summary(edge_list, nodes_df)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()
