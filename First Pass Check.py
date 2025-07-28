import sqlite3
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, triu
import networkx as nx
import community as community_louvain
import os

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

def generate_co_subscription_edges(A, chan_idx, min_weight=1):
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
    print(f"Filtered edges (weight ≥ {min_weight}): {len(edge_list)}")
    return edge_list

def run_louvain(edge_list):
    G = nx.Graph()
    for _, row in edge_list.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    partition = community_louvain.best_partition(G, weight='weight')
    return pd.DataFrame(list(partition.items()), columns=['id', 'community'])

def add_channel_names(nodes_df, names_df):
    return nodes_df.merge(names_df, how='left', left_on='id', right_on='channel_id') \
                   .drop(columns=['channel_id'])

def export_for_gephi(edge_list, nodes_df, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    edge_list.to_csv(f"{output_dir}/edges.csv", index=False)
    nodes_df.to_csv(f"{output_dir}/nodes.csv", index=False)
    print(f"Exported edges and nodes to '{output_dir}/'")

def main():
    db_path = "youtube_graph.db"

    try:
        conn = connect_db(db_path)

        # Load and process data
        subs_df = load_subscriptions(conn)
        names_df = load_channel_names(conn)

        A, chan_idx = build_co_subscription_matrix(subs_df)
        edge_list = generate_co_subscription_edges(A, chan_idx, min_weight=3)

        nodes_df = run_louvain(edge_list)
        nodes_df = add_channel_names(nodes_df, names_df)

        export_for_gephi(edge_list, nodes_df)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
