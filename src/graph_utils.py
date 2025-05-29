import os
from collections import defaultdict
from typing import Optional, Dict, List, Any

import community as co
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.data_utils import calculate_and_normalize_ratios, \
    calculate_team_points


def normalize_to_0_1(value: float, min_val: float, max_val: float) -> float:
    """
    Normalizes a value from [min_val, max_val] to [0, 1].
    Handles cases where min_val == max_val to prevent division by zero.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def create_epl_network(df: pd.DataFrame, metric_name: str,
                       season: Optional[str] = None, start_year: Optional[int] = None,
                       end_year: Optional[int] = None) -> Optional[nx.Graph]:
    """
    Creates an UNDIRECTED graph network where each edge (TeamA - TeamB) represents
    the SIMILARITY in their TOTAL performance for a given 'metric_name'
    across the specified season(s). Higher edge weight means greater similarity.

    Args:
        df (pandas.DataFrame): The complete match DataFrame.
        metric_name (str): The specific metric to use for edge weights
                           ('goals', 'aggressiveness', 'control', 'points').
        season (str, optional): A specific season to analyze (e.g., '2016/17').
                                This parameter takes precedence over start_year/end_year if both are provided.
                                Defaults to None.
        start_year (int, optional): The starting year of the season range (e.g., 2014 for '2014/15').
                                    Defaults to None.
        end_year (int, optional): The ending year of the season range (e.g., 2017 for '2016/17').
                                  Defaults to None.

    Returns:
        nx.Graph: The undirected graph, or None if the data to process is empty, invalid,
                  or the metric_name is not supported.
    """

    df_to_process = df.copy()
    current_scope_name = "the entire dataset"

    # --- Data Filtering by Season/Year Range ---
    if season:
        df_to_process = df_to_process[df_to_process['Season'] == season].copy()
        current_scope_name = f"season {season}"
    elif start_year is not None and end_year is not None:
        if not (isinstance(start_year, int) and isinstance(end_year, int)):
            print("Error: 'start_year' and 'end_year' must be integers.")
            return None
        if start_year >= end_year:
            print("Error: For a valid season range (e.g., 2016/2017), 'end_year' must be greater than 'start_year'.")
            return None

        seasons_in_range = []
        for year in range(start_year, end_year):
            next_year_suffix = str(year + 1)[-2:]
            seasons_in_range.append(f"{year}/{next_year_suffix}")

        df_to_process = df_to_process[df_to_process['Season'].isin(seasons_in_range)].copy()
        current_scope_name = f"seasons from {start_year}/{str(start_year + 1)[-2:]} to {end_year - 1}/{str(end_year)[-2:]}"

    if df_to_process.empty:
        print(f"No data found for {current_scope_name}.")
        return None

    # --- Data Cleaning and Preprocessing ---
    # Ensure critical columns exist and are numeric
    cols_to_fill = ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'AwayShots',
                    'HomeCorners', 'AwayCorners',
                    'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards',
                    'HomeRedCards', 'AwayRedCards', 'FullTimeResult']

    for col in cols_to_fill:
        if col not in df_to_process.columns:
            print(
                f"Warning: Critical column '{col}' not found in the dataset for {current_scope_name}. Cannot proceed.")
            return None
        if df_to_process[col].dtype == 'object' and col not in ['HomeTeam', 'AwayTeam', 'FullTimeResult']:
            df_to_process[col] = pd.to_numeric(df_to_process[col], errors='coerce').fillna(0)
        elif col not in ['HomeTeam', 'AwayTeam', 'FullTimeResult']:
            df_to_process[col] = df_to_process[col].fillna(0)

    # Calculate points per match (needed for total points aggregation)
    df_to_process['HomePoints'] = df_to_process['FullTimeResult'].apply(
        lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df_to_process['AwayPoints'] = df_to_process['FullTimeResult'].apply(
        lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

    # --- Aggregation of TOTAL TEAM STATS across all their matches in the scope ---
    team_total_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'total_goals_scored': 0,
        'total_fouls': 0,
        'total_yellow_cards': 0,
        'total_red_cards': 0,
        'total_corners': 0,
        'total_shots': 0,
        'total_points': 0
    })

    for _, match in df_to_process.iterrows():
        ht = match['HomeTeam']
        at = match['AwayTeam']

        # Aggregate for Home Team
        team_total_stats[ht]['total_goals_scored'] += match['FullTimeHomeGoals']
        team_total_stats[ht]['total_fouls'] += match['HomeFouls']
        team_total_stats[ht]['total_yellow_cards'] += match['HomeYellowCards']
        team_total_stats[ht]['total_red_cards'] += match['HomeRedCards']
        team_total_stats[ht]['total_corners'] += match['HomeCorners']
        team_total_stats[ht]['total_shots'] += match['HomeShots']
        team_total_stats[ht]['total_points'] += match['HomePoints']

        # Aggregate for Away Team
        team_total_stats[at]['total_goals_scored'] += match['FullTimeAwayGoals']
        team_total_stats[at]['total_fouls'] += match['AwayFouls']
        team_total_stats[at]['total_yellow_cards'] += match['AwayYellowCards']
        team_total_stats[at]['total_red_cards'] += match['AwayRedCards']
        team_total_stats[at]['total_corners'] += match['AwayCorners']
        team_total_stats[at]['total_shots'] += match['AwayShots']
        team_total_stats[at]['total_points'] += match['AwayPoints']

    # --- Calculate Derived Metrics for Each Team's Totals ---
    calculated_team_metrics = {}
    for team, stats in team_total_stats.items():
        calculated_team_metrics[team] = {
            'goals': stats['total_goals_scored'],
            'aggressiveness': stats['total_fouls'] + stats['total_yellow_cards'] + (3 * stats['total_red_cards']),
            'control': stats['total_corners'] + stats['total_shots'],
            'points': stats['total_points']
        }

    # --- Validate the requested metric_name ---
    supported_metrics = ['goals', 'aggressiveness', 'control', 'points']
    if metric_name not in supported_metrics:
        print(f"Error: Unsupported metric_name '{metric_name}'. Supported metrics are: {supported_metrics}")
        return None

    all_teams_in_scope = list(calculated_team_metrics.keys())
    if not all_teams_in_scope:
        print(f"No teams found in the data for {current_scope_name}.")
        return None

    G = nx.Graph()  # Initialized as an Undirected Graph
    G.add_nodes_from(all_teams_in_scope)

    # --- Calculate and Add Edges based on Metric Difference ---
    # Store all raw differences to find min/max for normalization
    raw_metric_differences = []

    # Use a set to keep track of processed undirected edges
    processed_edges = set()

    for i, team_a in enumerate(all_teams_in_scope):
        for j, team_b in enumerate(all_teams_in_scope):
            if team_a == team_b:
                continue

            # Ensure each undirected edge is processed only once
            # Add a canonical representation (e.g., sort the tuple)
            edge_tuple = tuple(sorted((team_a, team_b)))
            if edge_tuple in processed_edges:
                continue

            val_a = calculated_team_metrics[team_a][metric_name]
            val_b = calculated_team_metrics[team_b][metric_name]

            raw_diff = abs(val_a - val_b)
            raw_metric_differences.append(raw_diff)

            # Add edge with raw difference as an attribute
            # We will update 'weight' after normalization
            G.add_edge(team_a, team_b, raw_diff=raw_diff)
            processed_edges.add(edge_tuple)  # Mark as processed

    # --- Normalize Edge Weights (Similarity) ---
    # Find min and max of all collected absolute differences
    min_raw_diff = min(raw_metric_differences) if raw_metric_differences else 0.0
    max_raw_diff = max(raw_metric_differences) if raw_metric_differences else 0.0

    for u, v, data in G.edges(data=True):
        # Normalize the absolute difference (dissimilarity) to a [0, 1] range.
        normalized_dissimilarity = normalize_to_0_1(data['raw_diff'], min_raw_diff, max_raw_diff)

        # For community detection and general network analysis, a higher 'weight'
        # typically implies a stronger connection (i.e., higher similarity).
        # So, convert dissimilarity (0=identical, 1=max_different) to similarity (1=identical, 0=max_different).
        G.edges[u, v]['weight'] = 1 - normalized_dissimilarity
        G.edges[u, v][
            'normalized_dissimilarity'] = normalized_dissimilarity  # Store normalized dissimilarity for clarity

    print(f"Undirected network created for {current_scope_name} based on '{metric_name}' metric.")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}.")
    return G


def calculate_and_print_centralities(graph: nx.Graph, metric: str) -> Dict[str, Dict[str, float]]:
    """
    Calculates and prints various centrality measures for a given UNDIRECTED graph.
    This graph is assumed to be ALREADY FILTERED and its edges possess a 'weight'
    attribute, which signifies similarity (e.g., 1 - normalized_absolute_difference).

    Args:
        graph (nx.Graph): The NetworkX undirected graph to analyze.
                          Edges are expected to have a 'weight' attribute.
        metric (str): The base name of the metric used for graph construction and analysis
                      (e.g., 'goals', 'aggressiveness', 'shot_accuracy', 'control', 'points').
                      This parameter is primarily used for logging and contextualizing the output.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the calculated centrality measures.
                                     Keys are centrality types (e.g., 'degree', 'strength'),
                                     and values are dictionaries mapping node names to their centrality scores.
    """
    centrality_scores = {}

    print(f"\n--- Calculating Centrality Measures for metric: {metric} ---")

    if graph.number_of_nodes() == 0:
        print(f"Warning: Input graph for metric '{metric}' is empty. Skipping centrality calculations.")
        return {
            'degree': {},
            'strength': {},
            'betweenness': {},
            'closeness': {},
            'eigenvector': {}
        }

    # For an undirected graph, 'strength' is the sum of incident edge weights.
    # The 'weight' attribute is assumed to be set by filter_graph_by_weight, representing similarity.
    weight_attribute_name = 'weight'

    # Check graph connectivity for path-based centralities (Betweenness and Closeness)
    if not nx.is_connected(graph):
        print(f"Warning: Graph for metric '{metric}' is not connected. "
              f"Betweenness and Closeness Centrality may be 0 for some nodes "
              f"or may represent only local paths.")
        print(f"Number of connected components: {nx.number_connected_components(graph)}")

    # 1. Degree Centrality (unweighted)
    # This measures the number of connections a node has.
    centrality_scores['degree'] = nx.degree_centrality(graph)

    # 2. Strength Centrality (weighted)
    # This measures the sum of the weights of edges incident to a node.
    centrality_scores['strength'] = dict(graph.degree(weight=weight_attribute_name))

    # --- Preparation for Betweenness and Closeness Centrality ---
    # These centralities are based on shortest paths, thus requiring edge weights to represent 'distance' or 'cost'.
    # Since our 'weight' attribute represents 'similarity' (higher weight = more similar/stronger connection),
    # we need to transform it into a 'distance' (lower distance = stronger connection).
    # A common transformation is (1 - similarity_weight) + epsilon.
    temp_graph_for_paths = graph.copy()
    epsilon = 1e-9  # A small constant to avoid zero distances, which can cause issues in path algorithms.

    # The 'weight' attribute is 1 - normalized_absolute_difference.
    # Therefore, (1 - 'weight') effectively gives us the normalized absolute difference.
    # Adding epsilon ensures distances are always positive.
    path_weight_attribute_name = 'distance_weight'

    for u, v, data in temp_graph_for_paths.edges(data=True):
        original_similarity_weight = data.get('weight', 0.0)  # Retrieve the similarity weight [0,1]

        # Transform similarity into distance: higher similarity -> lower distance.
        # If similarity is 1, distance is epsilon.
        # If similarity is 0, distance is 1 + epsilon.
        distance_val = (1 - original_similarity_weight) + epsilon
        temp_graph_for_paths[u][v][path_weight_attribute_name] = distance_val

    # 3. Betweenness Centrality
    # Measures the extent to which a node lies on shortest paths between other nodes.
    # The 'weight' parameter for NetworkX shortest path algorithms refers to the 'distance' attribute.
    if any(data.get(path_weight_attribute_name, float('inf')) < float('inf')
           for u, v, data in temp_graph_for_paths.edges(data=True)):
        centrality_scores['betweenness'] = nx.betweenness_centrality(temp_graph_for_paths,
                                                                     weight=path_weight_attribute_name)
    else:
        print(f"Warning: No finite paths for Betweenness Centrality for metric {metric}. Setting scores to 0.0.")
        centrality_scores['betweenness'] = {node: 0.0 for node in graph.nodes()}

        # 4. Closeness Centrality
    # Measures how close a node is to all other nodes in the network (inverse of average shortest path distance).
    if any(data.get(path_weight_attribute_name, float('inf')) < float('inf')
           for u, v, data in temp_graph_for_paths.edges(data=True)):
        centrality_scores['closeness'] = nx.closeness_centrality(temp_graph_for_paths,
                                                                 distance=path_weight_attribute_name)
    else:
        print(f"Warning: No finite paths for Closeness Centrality for metric {metric}. Setting scores to 0.0.")
        centrality_scores['closeness'] = {node: 0.0 for node in graph.nodes()}

        # 5. Eigenvector Centrality
    # Measures a node's influence based on the influence of its neighbors.
    try:
        # For Eigenvector centrality, the 'weight' attribute (similarity) is used directly.
        # If 'weight' does not exist, NetworkX defaults to unweighted (equivalent to unit weights).
        centrality_scores['eigenvector'] = nx.eigenvector_centrality(graph, weight=weight_attribute_name, max_iter=1000,
                                                                     tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        print(
            f"Warning: Eigenvector centrality did not converge for metric {metric}. This can happen in disconnected "
            f"graphs or specific weight distributions. Setting scores to 0.0.")
        centrality_scores['eigenvector'] = {node: 0.0 for node in graph.nodes()}
    except Exception as e:
        print(f"Error calculating eigenvector centrality for metric {metric}: {e}. Setting scores to 0.0.")
        centrality_scores['eigenvector'] = {node: 0.0 for node in graph.nodes()}

        # --- Print Centrality Results ---
    print(f"\n--- Centrality Measures for metric: {metric} (based on '{weight_attribute_name}' attribute) ---")

    if not centrality_scores['degree'] or all(v == 0.0 for v in centrality_scores['degree'].values()):
        print(f"No meaningful centrality results to display for '{metric}' (all values are zero or empty).")
        return centrality_scores

    print("\nTop 5 Degree Centrality:")
    for team, score in sorted(centrality_scores['degree'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {team}: {score:.4f}")

    print("\nTop 5 Strength Centrality:")
    for team, score in sorted(centrality_scores['strength'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {team}: {score:.4f}")

    print("\nTop 5 Betweenness Centrality:")
    for team, score in sorted(centrality_scores['betweenness'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {team}: {score:.6f}")

    print("\nTop 5 Closeness Centrality:")
    for team, score in sorted(centrality_scores['closeness'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {team}: {score:.6f}")

    print("\nTop 5 Eigenvector Centrality:")
    for team, score in sorted(centrality_scores['eigenvector'].items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {team}: {score:.6f}")

    return centrality_scores


def find_communities(graph: nx.Graph, resolution: float = 1.0, weight_key: str = 'weight') -> Optional[Dict[str, int]]:
    """
    Identifies communities within an undirected graph using the Louvain algorithm.

    Args:
        graph (nx.Graph): The undirected graph (typically an output from filter_graph_by_weight)
                          with edge weights. Edges are expected to have a 'weight' attribute
                          representing similarity, where higher values indicate stronger connections.
        resolution (float): A resolution parameter for the Louvain algorithm.
                            Higher values tend to yield a greater number of smaller communities.
                            Lower values tend to result in fewer, larger communities.
                            The default value is 1.0.
        weight_key (str): The name of the edge attribute that stores the weight used by Louvain.
                          The default value is 'weight', consistent with the output of filter_graph_by_weight.

    Returns:
        Optional[Dict[str, int]]: A dictionary mapping node (team) names to their assigned community ID.
                                  Returns None if the input graph is empty or if no communities are found.
    """
    if not graph or graph.number_of_nodes() == 0:
        print("Input graph is empty for community detection.")
        return None

    if graph.number_of_edges() == 0:
        print("No edges in the graph. Cannot detect communities.")
        # If there are no edges, each node is considered its own community.
        # This returns a partition where each node has a unique community ID.
        return {node: i for i, node in enumerate(graph.nodes())}

    print(f"Detecting communities using Louvain algorithm (resolution={resolution})...")

    try:
        # The Louvain algorithm (best_partition) maximizes modularity, where higher weights
        # indicate stronger connections. The 'weight_key' ensures the correct attribute is used.
        partition = co.best_partition(graph, resolution=resolution, weight=weight_key)

        # Check if communities were actually found (e.g., not all nodes in one community)
        if len(set(partition.values())) <= 1:
            print("No significant communities found (all nodes in one community or no clear separation).")
            return None

        return partition
    except Exception as e:
        print(f"Error during community detection: {e}.")
        return None


def analyze_centrality_vs_points(centrality_data: Dict[str, Dict[str, float]], team_points: Dict[str, int],
                                 graph_nodes: List[str]):
    """
    Performs and presents the Pearson correlation analysis between various network centrality measures
    and team league points.

    Args:
        centrality_data (Dict[str, Dict[str, float]]): A dictionary containing centrality scores for each
                                                        centrality type. Outer keys are centrality names
                                                        (e.g., 'degree', 'strength'), and inner dictionaries
                                                        map team names to their respective centrality scores.
                                                        Typically sourced from `calculate_and_print_centralities`.
        team_points (Dict[str, int]): A dictionary mapping team names to their accumulated league points.
                                      Typically sourced from `calculate_team_points`.
        graph_nodes (List[str]): A list of node (team) names that were present in the graph
                                 for which centrality measures were calculated. This ensures
                                 alignment of data points.
    """
    if not centrality_data or not team_points or not graph_nodes:
        print("\nCorrelation analysis cannot be performed: Essential data (centrality scores, team points, "
              "or graph nodes) is missing or empty.")
        return

    # Align teams and data: Only include teams that are present in the list of graph nodes
    # and have valid data in both the centrality dictionary and the team points dictionary.
    teams_for_correlation = [
        team for team in graph_nodes
        if team in team_points and all(team in c_scores for c_scores in centrality_data.values())
    ]

    if not teams_for_correlation:
        print("No common teams with valid centrality and points data found for correlation analysis. Skipping "
              "correlation.")
        return

    # Extract league points for the aligned teams
    points = [team_points.get(team, 0) for team in teams_for_correlation]

    # Prepare a Pandas DataFrame for correlation computation
    results_df_data: Dict[str, List[Any]] = {'Team': teams_for_correlation, 'Points': points}
    for c_type, c_values in centrality_data.items():
        results_df_data[c_type] = [c_values.get(team, 0.0) for team in teams_for_correlation]  # Ensure default is
        # float for scores

    results_df = pd.DataFrame(results_df_data)

    print(f"\nPearson Correlation between Centrality Measures and League Points:")
    # Iterate through each centrality type and calculate its correlation with league points
    for centrality_type in centrality_data.keys():
        # Correlation can only be calculated if there is variance in the centrality values
        if len(results_df[centrality_type].unique()) > 1:
            correlation = results_df['Points'].corr(results_df[centrality_type])
            print(f"  {centrality_type} Centrality vs. Points: {correlation:.4f}")
        else:
            print(f"  {centrality_type} Centrality vs. Points: Cannot calculate (insufficient variance in centrality "
                  f"values)")


def plot_metric_ratios_vs_points(normalized_ratios_df: pd.DataFrame, scope_description: str):
    """
    Generates a single figure with multiple subplots to visualize the relationship between
    specified normalized metrics and normalized total league points.
    Each subplot presents a different metric. Teams are differentiated by color.

    Args:
        normalized_ratios_df (pd.DataFrame): A DataFrame containing normalized metric ratios and
                                             normalized points, typically derived from
                                             `calculate_and_normalize_ratios` and joined with normalized points.
                                             Expected columns include 'Team', 'points_norm',
                                             and various normalized metric columns (e.g., 'goals_scored_norm').
        scope_description (str): A descriptive string indicating the scope of the analysis
                                 (e.g., "Season 2016/17" or "Seasons 2014-2017").
    """
    if normalized_ratios_df is None or normalized_ratios_df.empty:
        print("\nCannot plot metric ratios vs. points: Input DataFrame is None or empty.")
        return

    print(f"\n--- Plotting Metric Ratios vs. Normalized Points ({scope_description}) ---")

    metrics_to_plot = {
        'goals_scored_norm': 'Normalized Goals Scored',
        'aggressiveness_committed_norm': 'Normalized Aggressiveness Committed',
        'shot_accuracy_norm': 'Normalized Shot Accuracy',
        'control_norm': 'Normalized Control'
    }

    # Determine the number of unique teams to select an appropriate palette size
    num_teams = len(normalized_ratios_df.index.unique())

    # A custom palette designed to accommodate up to 46 distinct teams
    custom_palette_46 = [
        '#E60000', '#009900', '#0000CC', '#FFD700', '#8A2BE2', '#FFA500', '#008B8B', '#FF69B4', '#20B2AA', '#A52A2A',
        '#7FFF00', '#DAA520', '#C0C0C0', '#4682B4', '#D2691E', '#800000', '#00FF7F', '#800080', '#DDA0DD', '#F0E68C',
        '#1E90FF', '#FF4500', '#8B0000', '#2F4F4F', '#D8BFD8', '#BA55D3', '#B0C4DE', '#FAEBD7', '#7CFC00', '#FF00FF',
        '#BDB76B', '#ADFF2F', '#A0522D', '#CD853F', '#6B8E23', '#483D8B', '#FFEFD5', '#FFF0F5', '#F5DEB3', '#D2B48C',
        '#BC8F8F', '#A9A9A9', '#B8860B', '#3CB371', '#C71585', '#00BFFF'
    ]

    # Prepare the DataFrame by resetting its index once for plotting efficiency
    plot_data = normalized_ratios_df.reset_index()

    # Set a base font size for all text elements
    # Increased all font sizes slightly for better readability
    base_fontsize = 14  # Increased from 12
    label_fontsize = base_fontsize + 2  # Increased from 14
    tick_fontsize = base_fontsize  # Increased from 12
    legend_fontsize = base_fontsize + 2  # Increased from 14
    annotation_fontsize = base_fontsize  # Increased from 12

    # Create a single figure with 2 rows and 2 columns for the 4 subplots
    # Increased overall figure size slightly to accommodate larger fonts
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # Increased from 18x14
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    common_handles = []
    common_labels = []

    for i, (metric_col, title_suffix) in enumerate(metrics_to_plot.items()):
        ax = axes[i]  # Get the current subplot axis

        sns.scatterplot(
            data=plot_data,
            x=metric_col,
            y='points_norm',
            hue='Team',
            palette=custom_palette_46,
            s=150,
            alpha=0.8,
            ax=ax,
        )

        # Store legend handles and labels from the first subplot to create a common legend later
        if i == 0:  # Only capture once
            current_handles, current_labels = ax.get_legend_handles_labels()
            # Skip the first handle/label which is the legend title 'Team'
            common_handles = current_handles[1:]
            common_labels = current_labels[1:]

        # Remove the individual legend from each subplot
        if ax.legend_ is not None:
            ax.legend_.remove()

        sns.regplot(
            data=plot_data,
            x=metric_col,
            y='points_norm',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--', 'alpha': 0.7},
            ax=ax
        )

        ax.set_xlabel(title_suffix, fontsize=label_fontsize)
        ax.set_ylabel('Normalized Points', fontsize=label_fontsize)

        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

        correlation = normalized_ratios_df[metric_col].corr(normalized_ratios_df['points_norm'])
        ax.text(0.05, 0.95, f'Corr: {correlation:.2f}',
                transform=ax.transAxes, fontsize=annotation_fontsize,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    fig.suptitle(f'Metric Ratios vs. Normalized League Points ({scope_description})',
                 fontsize=label_fontsize + 8, y=0.97)  # Increased suptitle font size

    fig.legend(common_handles, common_labels, title='Team', loc='center right', bbox_to_anchor=(0.99, 0.5),
               fontsize=legend_fontsize, ncol=1, title_fontsize=label_fontsize)

    # Adjust layout to prevent overlapping elements, make space for suptitle/legend, and add subplot spacing
    plt.subplots_adjust(left=0.07, right=0.83, top=0.90, bottom=0.07,  # Adjusted general figure margins
                        wspace=0.2, hspace=0.3)  # Added horizontal and vertical space between subplots

    plt.show()


def plot_epl_analysis_results(
        epl_df: pd.DataFrame,
        scope_description: str
) -> None:
    """
    Generates scatter plots showing normalized team metric ratios (e.g., goals, aggressiveness)
    against normalized total points for a specified analysis scope (single season or range of seasons).

    This function expects the input DataFrame 'epl_df' to already be filtered to the desired
    analysis scope (e.g., a specific season or range of seasons). It then calculates and normalizes
    the relevant metric ratios and team points within this scope, finally calling
    'plot_metric_ratios_vs_points' to create the visualizations.

    Args:
        epl_df (pd.DataFrame): The DataFrame containing EPL match data, already filtered
                               to the specific season(s) or range being analyzed.
        scope_description (str): A string describing the analysis scope (e.g., "season 2016/17"
                                 or "seasons from 2012/13 to 2017/18"). This is used for plot titles.
    """
    print("\n--- Generating Metric Ratios vs. Normalized Points Plots ---")

    df_for_scope = epl_df.copy()

    if df_for_scope.empty:
        print("Skipping Metric Ratios vs. Points plot: Input DataFrame for scope is empty.")
        return

    normalized_ratios_df = calculate_and_normalize_ratios(df_for_scope)
    league_points = calculate_team_points(df_for_scope)  # Recalculate points for this scope

    if normalized_ratios_df is None or normalized_ratios_df.empty:
        print("Skipping Metric Ratios vs. Points plot: Issues in calculating or normalizing ratios.")
        return

    if not league_points:
        print("Skipping Metric Ratios vs. Points plot: League points data is missing or empty for this scope.")
        return

    league_points_for_plot = pd.DataFrame(list(league_points.items()), columns=['Team', 'Points']).set_index('Team')

    # Normalize points for plotting
    min_points = league_points_for_plot['Points'].min()
    max_points = league_points_for_plot['Points'].max()
    if max_points == min_points:
        league_points_for_plot['Normalized_Points'] = 0.5
    else:
        league_points_for_plot['Normalized_Points'] = (league_points_for_plot['Points'] - min_points) / (
                max_points - min_points)

    # Join normalized ratios with normalized points. Use 'inner' to ensure only common teams are plotted.
    plot_df = normalized_ratios_df.join(league_points_for_plot[['Normalized_Points']], how='inner', lsuffix='_ratio')

    if plot_df.empty:
        print("Skipping Metric Ratios vs. Points plot: No common data for plotting after joining ratios and points.")
        return

    # Call the existing plot_metric_ratios_vs_points function from graph_utils
    plot_metric_ratios_vs_points(plot_df, scope_description)
    print("Metric Ratios vs. Normalized Points Plots generated successfully.")


def _load_team_logos(team_names: List[str], logo_path: str, base_logo_pixel_dim: int = 200) -> Dict[str, Image.Image]:
    """
    Loads team logos from a specified folder, resizes them to a uniform base pixel dimension,
    and ensures consistent transparency handling by converting to RGBA.
    Returns PIL Image objects.

    Args:
        team_names (List[str]): List of team names in the graph.
        logo_path (str): Path to the directory containing logo files.
        base_logo_pixel_dim (int): The target side length in pixels for the base resized logos (e.g., 200x200).

    Returns:
        Dict[str, Image.Image]: A dictionary mapping team names to PIL Image objects,
                                where all images are of uniform pixel size (base_logo_pixel_dim x base_logo_pixel_dim).
    """
    logos = {}
    script_dir = os.path.dirname(__file__)
    absolute_logo_folder_path = os.path.normpath(os.path.join(script_dir, '..', logo_path))

    for team_name in team_names:
        standardized_filename = team_name.lower().replace(" ", "_")
        logo_file = os.path.join(absolute_logo_folder_path, f"{standardized_filename}.png")

        if not os.path.exists(logo_file):
            print(f"Logo not found for team: '{team_name}'. Expected file: '{logo_file}'. Skipping.")
            continue

        try:
            # CHANGE THIS LINE BACK TO RGBA
            img = Image.open(logo_file).convert("RGBA")

            img_resized = img.resize((base_logo_pixel_dim, base_logo_pixel_dim), Image.LANCZOS)

            logos[team_name] = img_resized
        except Exception as e:
            print(f"Error loading logo for {team_name} from {logo_file}: {e}")
    return logos


def plot_network_communities(graph: nx.Graph, communities: Dict[str, int], metric_name: str, scope_desc: str,
                             logo_folder: str = 'data/epl_logos/',
                             logo_display_zoom: float = 0.15,
                             node_base_size: int = 1600):
    """
    Plots the network graph with nodes represented by colored circles and team logos inside them.
    Nodes are colored and grouped by their detected community.

    Args:
        graph (nx.Graph): The NetworkX graph to visualize (undirected, filtered).
        communities (Dict[str, int]): A dictionary mapping node names (team names) to their community ID.
        metric_name (str): The name of the metric used for analysis (e.g., 'goals', 'aggressiveness').
        scope_desc (str): A string describing the analysis scope (e.g., "season 2016/17").
        logo_folder (str): Path to the directory containing team logo images.
        logo_display_zoom (float): The zoom factor for the logos. This value scales the logo
                                   relative to the data coordinates. Adjust this to make logos
                                   fit within the nodes. Smaller value = smaller logo.
        node_base_size (int): The base size for the colored circular nodes (in points^2).
                              This should be large enough to contain the logos.
    """
    if not graph or graph.number_of_nodes() == 0:
        print(f"Cannot plot empty graph for {metric_name}.")
        return

    # --- Font Size Adjustments ---
    title_fontsize = 24
    text_fallback_fontsize = 12
    legend_title_fontsize = 18
    legend_label_fontsize = 16

    # --- Figure Size Adjustment ---
    plt.figure(figsize=(18, 16))
    ax = plt.gca()
    plt.title(f"Network Communities for {metric_name} ({scope_desc})", fontsize=title_fontsize)

    team_names_in_graph = list(graph.nodes())
    loaded_logos = _load_team_logos(team_names_in_graph, logo_folder)

    categorical_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    unique_communities = sorted(list(set(communities.values())))
    color_map = {comm_id: categorical_palette[i % len(categorical_palette)]
                 for i, comm_id in enumerate(unique_communities)}
    node_colors = [color_map.get(communities.get(node), 'lightgrey') for node in graph.nodes()]

    pos = {}
    num_nodes = graph.number_of_nodes()

    if num_nodes == 1:
        node = list(graph.nodes())[0]
        pos[node] = np.array([0.0, 0.0])
        print(f"Plotting a single node graph for {metric_name}.")
    elif communities and len(unique_communities) > 1 and num_nodes > 1:
        community_centroid_graph = nx.Graph()
        community_centroid_graph.add_nodes_from(unique_communities)

        for u, v in graph.edges():
            comm_u = communities.get(u)
            comm_v = communities.get(v)
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                # Add edge to centroid graph or increase weight
                if community_centroid_graph.has_edge(comm_u, comm_v):
                    community_centroid_graph[comm_u][comm_v]['weight'] += 1
                else:
                    community_centroid_graph.add_edge(comm_u, comm_v, weight=1)

        try:
            # Use spring layout for centroids, spectral can be unstable with few nodes/edges
            centroid_pos = nx.spring_layout(community_centroid_graph, k=2.0, iterations=100, seed=42, weight='weight')
            if not centroid_pos:  # Fallback if spring_layout returns empty for some reason
                centroid_pos = {comm_id: np.random.rand(2) * 2 - 1 for comm_id in unique_communities}
        except Exception:
            print("Warning: Layout for centroids failed, falling back to random positions.")
            centroid_pos = {comm_id: np.random.rand(2) * 2 - 1 for comm_id in unique_communities}

        initial_pos = {}
        for node_name in graph.nodes():
            comm_id = communities.get(node_name)
            if comm_id is not None and comm_id in centroid_pos:
                # Add a small random jitter to initial positions within communities
                initial_pos[node_name] = centroid_pos[comm_id] + np.random.rand(
                    2) * 0.05 - 0.025  # Slightly more jitter
            else:
                initial_pos[node_name] = np.random.rand(2) * 2 - 1

        optimal_k = 0.7 / np.sqrt(num_nodes) if num_nodes > 0 else 0.15  # Adjusted optimal_k for more spread

        try:
            pos = nx.spring_layout(graph, pos=initial_pos, k=optimal_k, iterations=200, seed=42, weight='weight')
        except Exception as e:
            print(f"Error during spring layout refinement: {e}. Falling back to default spring layout.")
            pos = nx.spring_layout(graph, k=0.2, iterations=100, seed=42)  # Adjusted fallback k and iterations

    else:
        print("Less than 2 communities detected or 0 edges; using standard spring layout.")
        optimal_k = 0.7 / np.sqrt(num_nodes) if num_nodes > 0 else 0.15  # Adjusted optimal_k for more spread
        pos = nx.spring_layout(graph, k=optimal_k, iterations=150, seed=42, weight='weight')  # Increased iterations

    # Draw the colored nodes (circles) first
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_base_size,
                           alpha=0.9, linewidths=1.5, edgecolors='black')  # Increased linewidths for better definition

    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray')

    # Add logos on top of the nodes
    for node_name, p in pos.items():
        if node_name in loaded_logos:
            img_offset = OffsetImage(loaded_logos[node_name],
                                     zoom=logo_display_zoom,
                                     origin='upper',
                                     interpolation='bilinear'
                                     )
            ab = AnnotationBbox(img_offset, p,
                                frameon=False,
                                pad=0.0,
                                boxcoords="data",
                                xycoords="data",
                                bboxprops=dict(boxstyle="square,pad=0", fc="white", ec="none", alpha=1.0)
                                )
            ax.add_artist(ab)
        else:
            # Fallback for nodes without logos: draw text with increased font size
            ax.text(p[0], p[1], node_name, fontsize=text_fallback_fontsize, ha='center', va='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Adjust overall scaling of the plot area
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]

        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Increased padding slightly to ensure all nodes/labels fit
            x_pad = (max_x - min_x) * 0.20 if num_nodes > 1 else 0.25
            y_pad = (max_y - min_y) * 0.20 if num_nodes > 1 else 0.25

            ax.set_xlim(min_x - x_pad, max_x + x_pad)
            ax.set_ylim(min_y - y_pad, max_y + y_pad)
        else:
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
    else:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    legend_handles = []
    for comm_id in unique_communities:
        label = f"Community {comm_id}"
        handle = mpatches.Patch(color=color_map[comm_id], label=label)
        legend_handles.append(handle)

    if legend_handles:
        # Increased legend font sizes
        plt.legend(handles=legend_handles, title="Communities", loc='upper left', bbox_to_anchor=(1, 1),
                   fontsize=legend_label_fontsize, title_fontsize=legend_title_fontsize)

    # Adjusted tight_layout rect to ensure legend fits well
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Slightly more space on the right for the legend
    plt.axis('off')
    plt.show()
