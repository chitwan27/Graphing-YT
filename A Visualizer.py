# -------------------------------------------------------------
# Optional command-line arguments:
#   --output-dir <dir>                Specify a different output directory.
#   --layout <spring|kamada_kawai|circular|random>
#                                     Choose the network layout algorithm.
#   --color-scheme <plotly|viridis|rainbow>
#                                     Choose the color scheme for communities.
#   --max-edges <N>                   Limit the number of edges displayed.
#   --save-html <filename>            Save the visualization as an HTML file.
#   --show-stats                      Print network statistics.
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import os
import glob
from datetime import datetime
import argparse
import math

class NetworkVisualizer:
    def __init__(self):
        self.edge_list = None
        self.nodes_df = None
        self.G = None
        self.pos = None
        self.community_colors = None
        
    def load_latest_results(self, output_dir="output"):
        """Load the most recent analysis results"""
        print(f"Looking for results in '{output_dir}/'...")
        
        # Find all edges files (handles timestamps)
        edge_files = glob.glob(os.path.join(output_dir, "edges*.csv"))
        node_files = glob.glob(os.path.join(output_dir, "nodes*.csv"))
        
        if not edge_files or not node_files:
            raise FileNotFoundError(f"No analysis results found in '{output_dir}/'")
        
        # Get the most recent files
        latest_edge_file = max(edge_files, key=os.path.getctime)
        latest_node_file = max(node_files, key=os.path.getctime)
        
        print(f"Loading edges from: {os.path.basename(latest_edge_file)}")
        print(f"Loading nodes from: {os.path.basename(latest_node_file)}")
        
        # Load data
        self.edge_list = pd.read_csv(latest_edge_file)
        self.nodes_df = pd.read_csv(latest_node_file)
        
        print(f"Loaded {len(self.edge_list):,} edges and {len(self.nodes_df):,} nodes")
        
        return self
    
    def load_specific_results(self, edges_file, nodes_file):
        """Load specific result files"""
        print(f"Loading edges from: {edges_file}")
        print(f"Loading nodes from: {nodes_file}")
        
        self.edge_list = pd.read_csv(edges_file)
        self.nodes_df = pd.read_csv(nodes_file)
        
        print(f"Loaded {len(self.edge_list):,} edges and {len(self.nodes_df):,} nodes")
        return self
    
    def build_network_layout(self, layout_algorithm="spring", iterations=50, k_spacing=1):
        """Build NetworkX graph and compute layout"""
        print(f"Building network graph...")
        
        # Create NetworkX graph
        self.G = nx.from_pandas_edgelist(
            self.edge_list, 
            source='source', 
            target='target', 
            edge_attr='weight'
        )
        
        print(f"Graph: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")
        
        # Add node attributes
        node_attrs = {}
        for _, row in self.nodes_df.iterrows():
            node_attrs[row['id']] = {
                'community': row['community'],
                'title': row.get('title', str(row['id']))
            }
        nx.set_node_attributes(self.G, node_attrs)
        
        # Compute layout
        print(f"Computing {layout_algorithm} layout...")
        
        if layout_algorithm == "spring":
            self.pos = nx.spring_layout(
                self.G, 
                iterations=iterations, 
                k=k_spacing/math.sqrt(self.G.number_of_nodes()),
                seed=42
            )
        elif layout_algorithm == "kamada_kawai":
            # Good for smaller networks
            if self.G.number_of_nodes() > 1000:
                print("Kamada-Kawai layout may be slow for large networks")
            self.pos = nx.kamada_kawai_layout(self.G)
        elif layout_algorithm == "circular":
            self.pos = nx.circular_layout(self.G)
        elif layout_algorithm == "random":
            self.pos = nx.random_layout(self.G, seed=42)
        else:
            print(f"Unknown layout: {layout_algorithm}, using spring")
            self.pos = nx.spring_layout(self.G, iterations=iterations, seed=42)
        
        print("Layout computed")
        return self
    
    def assign_community_colors(self, color_scheme="plotly"):
        """Assign colors to communities"""
        communities = self.nodes_df['community'].unique()
        n_communities = len(communities)
        
        print(f"Assigning colors for {n_communities} communities")
        
        if color_scheme == "plotly":
            # Use Plotly's default color sequence
            colors = px.colors.qualitative.Plotly
        elif color_scheme == "viridis":
            colors = px.colors.sequential.Viridis
        elif color_scheme == "rainbow":
            colors = px.colors.qualitative.Set3
        else:
            colors = px.colors.qualitative.Plotly
        
        # Extend colors if we have more communities than colors
        if n_communities > len(colors):
            colors = colors * (n_communities // len(colors) + 1)
        
        self.community_colors = {
            community: colors[i % len(colors)] 
            for i, community in enumerate(sorted(communities))
        }
        
        return self
    
    def create_interactive_plot(self, 
        plot_title="YouTube Channel Co-Subscription Network",
        node_size_column=None,
        show_edge_weights=False,
        max_edges_display=5000):
        """Create interactive Plotly visualization"""
        
        if self.pos is None:
            self.build_network_layout()
        
        if self.community_colors is None:
            self.assign_community_colors()
        
        print("Creating interactive visualization...")
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []
        hover_text = []
        
        for node in self.G.nodes():
            if node in self.pos:
                x, y = self.pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get node info
                node_info = self.G.nodes[node]
                community = node_info.get('community', 0)
                title = node_info.get('title', str(node))
                
                # Color by community
                node_colors.append(self.community_colors.get(community, '#888888'))
                
                # Node size (can be based on degree, weight, or custom column)
                if node_size_column and node_size_column in self.nodes_df.columns:
                    node_data = self.nodes_df[self.nodes_df['id'] == node]
                    if not node_data.empty:
                        size_val = node_data[node_size_column].iloc[0]
                        node_sizes.append(max(5, min(50, size_val / 10)))  # Scale between 5-50
                    else:
                        node_sizes.append(10)
                else:
                    # Size by degree (number of connections)
                    degree = self.G.degree(node)
                    node_sizes.append(max(5, min(50, degree * 2)))
                
                # Hover text
                degree = self.G.degree(node)
                hover_info = f"<b>{title}</b><br>"
                hover_info += f"Community: {community}<br>"
                hover_info += f"Connections: {degree}<br>"
                hover_info += f"Channel ID: {node}"
                hover_text.append(hover_info)
                
                node_text.append(title)
        
        # Prepare edge data (limit for performance)
        edge_x = []
        edge_y = []
        edge_weights = []
        
        edges_to_show = list(self.G.edges(data=True))
        if len(edges_to_show) > max_edges_display:
            print(f"Too many edges ({len(edges_to_show):,}), showing top {max_edges_display:,} by weight")
            # Sort by weight and take top edges
            edges_to_show.sort(key=lambda x: x[2].get('weight', 1), reverse=True)
            edges_to_show = edges_to_show[:max_edges_display]
        
        for edge in edges_to_show:
            x0, y0 = self.pos.get(edge[0], (0, 0))
            x1, y1 = self.pos.get(edge[1], (0, 0))
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        fig.add_trace(edge_trace)
        
        # Add nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if len(node_x) < 100 else 'markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=node_text if len(node_x) < 100 else None,
            textposition="middle center",
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=hover_text,
            name="Channels"
        )
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details. Colors represent communities.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=12, color="#888")
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
        )
        
        return fig
    
    def create_community_summary(self):
        """Create a summary visualization of communities"""
        if self.nodes_df is None:
            return None
        
        # Community statistics
        community_stats = self.nodes_df.groupby('community').agg({
            'id': 'count',
            'title': lambda x: x.dropna().iloc[0] if not x.dropna().empty else 'Unknown'
        }).rename(columns={'id': 'size'}).reset_index()
        
        community_stats = community_stats.sort_values('size', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            community_stats.head(20),  # Top 20 communities
            x='community',
            y='size',
            title="Top 20 Largest Communities",
            labels={'size': 'Number of Channels', 'community': 'Community ID'},
            color='size',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Community ID",
            yaxis_title="Number of Channels",
            showlegend=False
        )
        
        return fig
    
    def save_visualization(self, fig, filename="network_visualization.html"):
        """Save the visualization as an HTML file"""
        print(f"Saving visualization to {filename}")
        fig.write_html(filename)
        print(f"Saved! Open {filename} in your browser")
    
    def show_network_stats(self):
        """Print network statistics"""
        if self.G is None or self.nodes_df is None:
            return
        
        print("\n" + "="*50)
        print("NETWORK STATISTICS")
        print("="*50)
        
        # Basic stats
        print(f"Nodes: {self.G.number_of_nodes():,}")
        print(f"Edges: {self.G.number_of_edges():,}")
        print(f"Communities: {self.nodes_df['community'].nunique():,}")
        
        # Degree statistics
        degrees = [d for n, d in self.G.degree()]
        print(f"\nDegree Statistics:")
        print(f"   Average: {np.mean(degrees):.1f}")
        print(f"   Median: {np.median(degrees):.1f}")
        print(f"   Max: {max(degrees):,}")
        
        # Community size statistics
        community_sizes = self.nodes_df['community'].value_counts()
        print(f"\nCommunity Size Statistics:")
        print(f"   Largest: {community_sizes.iloc[0]:,} channels")
        print(f"   Average: {community_sizes.mean():.1f}")
        print(f"   Median: {community_sizes.median():.1f}")
        
        # Edge weight statistics
        if 'weight' in self.edge_list.columns:
            weights = self.edge_list['weight']
            print(f"\nEdge Weight Statistics:")
            print(f"   Average: {weights.mean():.1f}")
            print(f"   Median: {weights.median():.1f}")
            print(f"   Max: {weights.max():,}")
        
        print("="*50)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Visualize YouTube network analysis results')
    parser.add_argument('--output-dir', default='output', help='Directory containing analysis results')
    parser.add_argument('--layout', default='spring', choices=['spring', 'kamada_kawai', 'circular', 'random'], 
                        help='Layout algorithm')
    parser.add_argument('--color-scheme', default='plotly', choices=['plotly', 'viridis', 'rainbow'],
                        help='Color scheme for communities')
    parser.add_argument('--max-edges', type=int, default=5000, help='Maximum edges to display')
    parser.add_argument('--save-html', help='Save visualization as HTML file')
    parser.add_argument('--show-stats', action='store_true', help='Show network statistics')
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        viz = NetworkVisualizer()
        
        # Load results
        viz.load_latest_results(args.output_dir)
        
        # Show stats if requested
        if args.show_stats:
            viz.show_network_stats()
        
        # Build layout and create visualization
        viz.build_network_layout(layout_algorithm=args.layout)
        viz.assign_community_colors(color_scheme=args.color_scheme)
        
        # Create main network plot
        print("Creating network visualization...")
        fig = viz.create_interactive_plot(max_edges_display=args.max_edges)
        
        # Create community summary
        print("Creating community summary...")
        community_fig = viz.create_community_summary()
        
        # Save or show
        if args.save_html:
            viz.save_visualization(fig, args.save_html)
            if community_fig:
                viz.save_visualization(community_fig, args.save_html.replace('.html', '_communities.html'))
        else:
            print("Displaying visualization...")
            fig.show()
            if community_fig:
                community_fig.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage without command line
    print("YouTube Network Visualizer")
    print("="*40)
    
    try:
        viz = NetworkVisualizer()
        viz.load_latest_results()
        viz.show_network_stats()
        
        # Create visualization
        viz.build_network_layout()
        viz.assign_community_colors()
        
        fig = viz.create_interactive_plot()
        community_fig = viz.create_community_summary()
        
        # Show both plots
        fig.show()
        if community_fig:
            community_fig.show()
        
        # Optionally save
        save_choice = input("\nSave visualizations as HTML? [n]: ").strip().lower()
        if save_choice == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz.save_visualization(fig, f"network_viz_{timestamp}.html")
            if community_fig:
                viz.save_visualization(community_fig, f"communities_viz_{timestamp}.html")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()