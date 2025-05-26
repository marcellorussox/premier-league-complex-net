import os
from collections import defaultdict
from typing import Optional, Dict, List, Any
import numpy as np

import community as co
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.data_utils import normalize_val, normalize_diff_symmetric, calculate_and_normalize_ratios, \
    calculate_team_points


def create_epl_network(df: pd.DataFrame, season: Optional[str] = None, start_year: Optional[int] = None,
                       end_year: Optional[int] = None) -> Optional[nx.DiGraph]:
    """
    Creates a DIRECTED graph network where each edge (SourceTeam -> TargetTeam)
    represents the aggregated performance/action of the SourceTeam AGAINST the TargetTeam,
    along with the calculated differences in performance between Source and Target.

    Edge attributes reflect the SourceTeam's actions (e.g., goals scored by Source,
    fouls committed by Source) and the overall differences (Source vs Target).
    To obtain the TargetTeam's actions against the Source, one would inspect the reverse edge (TargetTeam -> SourceTeam).

    Args:
        df (pandas.DataFrame): The complete match DataFrame.
        season (str, optional): A specific season to analyze (e.g., '2016/17').
                                This parameter takes precedence over start_year/end_year if both are provided.
                                Defaults to None.
        start_year (int, optional): The starting year of the season range (e.g., 2014 for '2014/15').
                                    Defaults to None.
        end_year (int, optional): The ending year of the season range (e.g., 2017 for '2016/17').
                                  Defaults to None.

    Returns:
        nx.DiGraph: The directed graph, or None if the data to process is empty or invalid.
    """

    df_to_process = df.copy()
    current_scope_name = "the entire dataset"

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

    cols_to_fill = ['FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'AwayShots',
                    'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
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

    # Calculate points per match
    df_to_process['HomePoints'] = df_to_process['FullTimeResult'].apply(
        lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df_to_process['AwayPoints'] = df_to_process['FullTimeResult'].apply(
        lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

    directional_stats_agg = defaultdict(lambda: {
        'goals_scored_by_source': 0, 'goals_scored_by_target': 0,
        'sot_by_source': 0, 'sot_by_target': 0,
        'total_shots_by_source': 0, 'total_shots_by_target': 0,
        'fouls_by_source': 0, 'fouls_by_target': 0,
        'yc_by_source': 0, 'yc_by_target': 0,
        'rc_by_source': 0, 'rc_by_target': 0,
        'corners_by_source': 0, 'corners_by_target': 0,
        'points_scored_by_source': 0, 'points_scored_by_target': 0,
        'matches_played': 0
    })

    for _, match in df_to_process.iterrows():
        ht = match['HomeTeam']
        at = match['AwayTeam']

        # HOME TEAM -> AWAY TEAM (HT is SOURCE, AT is TARGET)
        directional_stats_agg[(ht, at)]['goals_scored_by_source'] += match['FullTimeHomeGoals']
        directional_stats_agg[(ht, at)]['goals_scored_by_target'] += match['FullTimeAwayGoals']
        directional_stats_agg[(ht, at)]['sot_by_source'] += match['HomeShotsOnTarget']
        directional_stats_agg[(ht, at)]['sot_by_target'] += match['AwayShotsOnTarget']
        directional_stats_agg[(ht, at)]['total_shots_by_source'] += match['HomeShots']
        directional_stats_agg[(ht, at)]['total_shots_by_target'] += match['AwayShots']
        directional_stats_agg[(ht, at)]['fouls_by_source'] += match['HomeFouls']
        directional_stats_agg[(ht, at)]['fouls_by_target'] += match['AwayFouls']
        directional_stats_agg[(ht, at)]['yc_by_source'] += match['HomeYellowCards']
        directional_stats_agg[(ht, at)]['yc_by_target'] += match['AwayYellowCards']
        directional_stats_agg[(ht, at)]['rc_by_source'] += match['HomeRedCards']
        directional_stats_agg[(ht, at)]['rc_by_target'] += match['AwayRedCards']
        directional_stats_agg[(ht, at)]['corners_by_source'] += match['HomeCorners']
        directional_stats_agg[(ht, at)]['corners_by_target'] += match['AwayCorners']
        directional_stats_agg[(ht, at)]['points_scored_by_source'] += match['HomePoints']
        directional_stats_agg[(ht, at)]['points_scored_by_target'] += match['AwayPoints']
        directional_stats_agg[(ht, at)]['matches_played'] += 1

        # AWAY TEAM -> HOME TEAM (AT is SOURCE, HT is TARGET)
        directional_stats_agg[(at, ht)]['goals_scored_by_source'] += match['FullTimeAwayGoals']
        directional_stats_agg[(at, ht)]['goals_scored_by_target'] += match['FullTimeHomeGoals']
        directional_stats_agg[(at, ht)]['sot_by_source'] += match['AwayShotsOnTarget']
        directional_stats_agg[(at, ht)]['sot_by_target'] += match['HomeShotsOnTarget']
        directional_stats_agg[(at, ht)]['total_shots_by_source'] += match['AwayShots']
        directional_stats_agg[(at, ht)]['total_shots_by_target'] += match['HomeShots']
        directional_stats_agg[(at, ht)]['fouls_by_source'] += match['AwayFouls']
        directional_stats_agg[(at, ht)]['fouls_by_target'] += match['HomeFouls']
        directional_stats_agg[(at, ht)]['yc_by_source'] += match['AwayYellowCards']
        directional_stats_agg[(at, ht)]['yc_by_target'] += match['HomeYellowCards']
        directional_stats_agg[(at, ht)]['rc_by_source'] += match['AwayRedCards']
        directional_stats_agg[(at, ht)]['rc_by_target'] += match['HomeRedCards']
        directional_stats_agg[(at, ht)]['corners_by_source'] += match['AwayCorners']
        directional_stats_agg[(at, ht)]['corners_by_target'] += match['HomeCorners']
        directional_stats_agg[(at, ht)]['points_scored_by_source'] += match['AwayPoints']
        directional_stats_agg[(at, ht)]['points_scored_by_target'] += match['HomePoints']
        directional_stats_agg[(at, ht)]['matches_played'] += 1

    all_teams_in_scope = pd.concat([df_to_process['HomeTeam'], df_to_process['AwayTeam']]).unique()
    G = nx.DiGraph()
    G.add_nodes_from(all_teams_in_scope)

    # Collect values for normalization ranges
    all_goals_scored_by_source = []
    all_aggressiveness_by_source = []
    all_shot_accuracy_by_source = []
    all_control_by_source = []
    all_points_scored_by_source = []

    # Collect RAW DIFFERENCES (can be negative or positive) for symmetric normalization
    all_goals_diff = []
    all_aggressiveness_diff = []
    all_shot_accuracy_diff = []
    all_control_diff = []
    all_points_diff = []

    processed_edges_data = {}
    for (source_team, target_team), agg_stats in directional_stats_agg.items():
        if source_team == target_team:
            continue

        # --- RAW METRICS FOR SOURCE TEAM'S ACTION AGAINST TARGET TEAM ---
        current_goals_scored = agg_stats['goals_scored_by_source']
        current_agg_by_source = agg_stats['fouls_by_source'] + agg_stats['yc_by_source'] + 3 * agg_stats['rc_by_source']
        current_sa_by_source = (agg_stats['sot_by_source'] / agg_stats['total_shots_by_source']) if agg_stats[
                                                                                                        'total_shots_by_source'] > 0 else 0.0
        current_ctrl_by_source = agg_stats['corners_by_source'] + agg_stats['total_shots_by_source']
        current_points_scored = agg_stats['points_scored_by_source']

        # --- CALCULATED DIFFERENCES (Source's performance relative to Target's performance against Source) ---
        goals_diff = current_goals_scored - agg_stats['goals_scored_by_target']
        aggressiveness_diff = current_agg_by_source - (
                agg_stats['fouls_by_target'] + agg_stats['yc_by_target'] + 3 * agg_stats['rc_by_target'])

        sa_of_target = (agg_stats['sot_by_target'] / agg_stats['total_shots_by_target']) if agg_stats[
                                                                                                'total_shots_by_target'] > 0 else 0.0
        shot_accuracy_diff = current_sa_by_source - sa_of_target

        ctrl_of_target = agg_stats['corners_by_target'] + agg_stats['total_shots_by_target']
        control_diff = current_ctrl_by_source - ctrl_of_target

        points_diff = current_points_scored - agg_stats['points_scored_by_target']

        processed_edges_data[(source_team, target_team)] = {
            'goals_scored': current_goals_scored,
            'aggressiveness_committed': current_agg_by_source,
            'shot_accuracy': current_sa_by_source,
            'control': current_ctrl_by_source,
            'points_scored': current_points_scored,

            'goals_diff': goals_diff,
            'aggressiveness_diff': aggressiveness_diff,
            'shot_accuracy_diff': shot_accuracy_diff,
            'control_diff': control_diff,
            'points_diff': points_diff,

            'matches_played': agg_stats['matches_played']
        }

        # Collect values for min-max normalization ranges
        all_goals_scored_by_source.append(current_goals_scored)
        all_aggressiveness_by_source.append(current_agg_by_source)
        all_shot_accuracy_by_source.append(current_sa_by_source)
        all_control_by_source.append(current_ctrl_by_source)
        all_points_scored_by_source.append(current_points_scored)

        # Append RAW (signed) differences for max_abs_diff calculation
        all_goals_diff.append(goals_diff)
        all_aggressiveness_diff.append(aggressiveness_diff)
        all_shot_accuracy_diff.append(shot_accuracy_diff)  # Corrected: append raw diff
        all_control_diff.append(control_diff)  # Corrected: append raw diff
        all_points_diff.append(points_diff)  # Corrected: append raw diff

    # Calculate global min/max for normalization of source metrics (range 0-1)
    min_goals_scored, max_goals_scored = (
        min(all_goals_scored_by_source), max(all_goals_scored_by_source)) if all_goals_scored_by_source else (0, 0)
    min_agg_by_source, max_agg_by_source = (
        min(all_aggressiveness_by_source), max(all_aggressiveness_by_source)) if all_aggressiveness_by_source else (
        0, 0)
    min_sa_by_source, max_sa_by_source = (
        min(all_shot_accuracy_by_source), max(all_shot_accuracy_by_source)) if all_shot_accuracy_by_source else (0, 0)
    min_ctrl_by_source, max_ctrl_by_source = (
        min(all_control_by_source), max(all_control_by_source)) if all_control_by_source else (0, 0)
    min_points_scored, max_points_scored = (
        min(all_points_scored_by_source), max(all_points_scored_by_source)) if all_points_scored_by_source else (0, 0)

    # Calculate GLOBAL MAX ABSOLUTE DIFFERENCE for symmetric normalization (-1 to 1)
    # This addresses the "squeezing" problem and ensures 0 difference maps to 0 normalized.
    max_abs_goals_diff = max(abs(val) for val in all_goals_diff) if all_goals_diff else 0
    max_abs_agg_diff = max(abs(val) for val in all_aggressiveness_diff) if all_aggressiveness_diff else 0
    max_abs_sa_diff = max(abs(val) for val in all_shot_accuracy_diff) if all_shot_accuracy_diff else 0
    max_abs_ctrl_diff = max(abs(val) for val in all_control_diff) if all_control_diff else 0
    max_abs_points_diff = max(abs(val) for val in all_points_diff) if all_points_diff else 0

    # Add edges to the graph with all calculated attributes
    for (source_team, target_team), stats in processed_edges_data.items():
        # Normalized totals (actions of source_team against target_team) - remain (0,1)
        goals_scored_norm = normalize_val(stats['goals_scored'], min_goals_scored, max_goals_scored,
                                          target_range=(0, 1))
        agg_committed_norm = normalize_val(stats['aggressiveness_committed'], min_agg_by_source, max_agg_by_source,
                                           target_range=(0, 1))
        sa_norm = normalize_val(stats['shot_accuracy'], min_sa_by_source, max_sa_by_source, target_range=(0, 1))
        ctrl_norm = normalize_val(stats['control'], min_ctrl_by_source, max_ctrl_by_source, target_range=(0, 1))
        points_scored_norm = normalize_val(stats['points_scored'], min_points_scored, max_points_scored,
                                           target_range=(0, 1))

        # Normalized Differences (range -1 to 1, sign retained)
        goals_diff_norm = normalize_diff_symmetric(stats['goals_diff'], max_abs_goals_diff)
        agg_diff_norm = normalize_diff_symmetric(stats['aggressiveness_diff'], max_abs_agg_diff)
        sa_diff_norm = normalize_diff_symmetric(stats['shot_accuracy_diff'], max_abs_sa_diff)
        ctrl_diff_norm = normalize_diff_symmetric(stats['control_diff'], max_abs_ctrl_diff)
        points_diff_norm = normalize_diff_symmetric(stats['points_diff'], max_abs_points_diff)

        G.add_edge(source_team, target_team,
                   # Raw totals (actions of source_team against target_team)
                   goals_scored=stats['goals_scored'],
                   aggressiveness_committed=stats['aggressiveness_committed'],
                   shot_accuracy=stats['shot_accuracy'],
                   control=stats['control'],
                   points_scored=stats['points_scored'],

                   # Normalized totals (actions of source_team against target_team) - remain (0,1)
                   goals_scored_norm=goals_scored_norm,
                   aggressiveness_committed_norm=agg_committed_norm,
                   shot_accuracy_norm=sa_norm,
                   control_norm=ctrl_norm,
                   points_norm=points_scored_norm,

                   # Raw Differences (signed)
                   goals_diff=stats['goals_diff'],
                   aggressiveness_diff=stats['aggressiveness_diff'],
                   shot_accuracy_diff=stats['shot_accuracy_diff'],
                   control_diff=stats['control_diff'],
                   points_diff=stats['points_diff'],

                   # Absolute Raw Differences
                   goals_diff_abs=abs(stats['goals_diff']),
                   aggressiveness_diff_abs=abs(stats['aggressiveness_diff']),
                   shot_accuracy_diff_abs=abs(stats['shot_accuracy_diff']),
                   control_diff_abs=abs(stats['control_diff']),
                   points_diff_abs=abs(stats['points_diff']),

                   # Normalized Differences (range -1 to 1, sign retained)
                   goals_diff_norm=goals_diff_norm,
                   aggressiveness_diff_norm=agg_diff_norm,
                   shot_accuracy_diff_norm=sa_diff_norm,
                   control_diff_norm=ctrl_diff_norm,
                   points_diff_norm=points_diff_norm,

                   # Absolute Normalized Differences (range 0-1, always positive)
                   goals_diff_norm_abs=abs(goals_diff_norm),
                   aggressiveness_diff_norm_abs=abs(agg_diff_norm),
                   shot_accuracy_diff_norm_abs=abs(sa_diff_norm),
                   control_diff_norm_abs=abs(ctrl_diff_norm),
                   points_diff_norm_abs=abs(points_diff_norm),

                   matches_played=stats['matches_played']
                   )

    print(f"Network created for {current_scope_name} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def filter_graph_by_weight(
        graph: nx.DiGraph,  # The input remains a directed graph (DiGraph)
        metric: str,
        threshold: Optional[float],
        use_normalized_abs_diff_for_filter: bool = True,  # True: _diff_norm_abs, False: _diff_abs
        keep_above: bool = True
) -> Optional[nx.Graph]:
    """
    Filters a directed graph based on the absolute intensity of the reciprocal relationship
    between two teams. It returns a new UNDIRECTED graph where an edge between u and v
    is included ONLY if their reciprocal relationship meets the filtering criteria.
    The 'weight' attribute on the resulting edge will be set to indicate similarity (1 - difference),
    making it suitable for Louvain community detection.

    Args:
        graph (nx.DiGraph): The original directed graph.
        metric (str): The base name of the metric for filtering (e.g., 'goals').
        threshold (float, optional): The threshold value for filtering. If None, no filtering is applied.
        use_normalized_abs_diff_for_filter (bool): If True, the filter will consider the
                                                    '{metric}_diff_norm_abs' attribute (range 0-1).
                                                    If False, it will consider the raw absolute
                                                    '{metric}_diff_abs' attribute.
        keep_above (bool): If True, keeps pairs where the intensity (difference) >= threshold.
                                     If False, keeps pairs where the intensity (difference) <= threshold.

    Returns:
        nx.Graph: A new UNDIRECTED graph with filtered reciprocal edges and a 'weight' attribute
                    suitable for Louvain (representing similarity).
                    Returns None if the input graph is empty or no edges remain after filtering.
    """
    if not graph or graph.number_of_nodes() == 0:
        print("Input graph is empty or invalid for filtering.")
        return None

    # Determine the attribute to use for filtering based on intensity
    # This attribute will be compared against the 'threshold'
    filter_attribute_name = f"{metric}_diff_norm_abs" if use_normalized_abs_diff_for_filter else f"{metric}_diff_abs"

    # Determine the source attribute for the 'weight' in the resulting graph (for Louvain)
    # Louvain requires a positive weight representing strength or similarity.
    # The normalized absolute difference ('_diff_norm_abs') is consistently used as the base
    # for this weight, as it is in the [0, 1] range and easily invertible to similarity.
    weight_source_attribute_name = f"{metric}_diff_norm_abs"

    # Initialize the new FILTERED UNDIRECTED graph
    filtered_G = nx.Graph()
    filtered_G.add_nodes_from(graph.nodes(data=True))  # Add all nodes to the new graph

    edges_retained = 0
    # This set will be used to process each {u,v} pair only once, as the resulting graph is undirected.
    processed_pairs = set()

    # Handle the case where no threshold is provided
    if threshold is None:
        print(f"No threshold provided for metric '{metric}'. Creating undirected graph with all reciprocal edges.")
        # When no filter is applied, include all existing reciprocal edges
        for u, v, data_uv in graph.edges(data=True):
            # Process only if the reverse edge exists and the pair has not been processed yet (to avoid duplicates in an undirected graph)
            if graph.has_edge(v, u):
                canonical_pair = tuple(sorted((u, v)))  # Order to get a unique identifier for the {u,v} pair
                if canonical_pair in processed_pairs:
                    continue
                processed_pairs.add(canonical_pair)

                # Retrieve the normalized absolute difference value from the u->v edge
                diff_norm_abs_value = data_uv.get(weight_source_attribute_name, 0.0)

                # Transform the difference into a similarity weight for Louvain (1 - difference)
                louvain_weight = 1 - diff_norm_abs_value

                filtered_G.add_edge(u, v, weight=louvain_weight)
                edges_retained += 1

        print(
            f"Unfiltered undirected graph for '{metric}' has {filtered_G.number_of_nodes()} nodes and {filtered_G.number_of_edges()} edges ({edges_retained} retained).")

        if filtered_G.number_of_edges() == 0:
            print("Warning: Unfiltered undirected graph has no edges.")
            return None
        return filtered_G

    # Logic for filtering with a specified threshold
    filter_condition_str = ">= threshold" if keep_above else "<= threshold"
    print(
        f"Applying reciprocal filter for '{metric}' with threshold {threshold} {filter_condition_str} on '{filter_attribute_name}'.")

    # Iterate over all edges in the original graph to find reciprocal pairs
    for u, v, data_uv in graph.edges(data=True):
        # Process each {u,v} pair only once (since we are creating an undirected graph)
        canonical_pair = tuple(sorted((u, v)))
        if canonical_pair in processed_pairs:
            continue
        processed_pairs.add(canonical_pair)

        # 4a. Get the filter value for the u -> v edge
        filter_value_uv = data_uv.get(filter_attribute_name)
        if filter_value_uv is None:
            continue  # Skip this pair if the filter attribute does not exist for u->v

        # 4b. Get the filter value for the v -> u edge
        # Check if the reverse edge exists, which is fundamental for a reciprocal relationship
        if graph.has_edge(v, u):
            data_vu = graph.get_edge_data(v, u)
            filter_value_vu = data_vu.get(filter_attribute_name)
            if filter_value_vu is None:
                continue  # Skip if the reverse edge exists but lacks the filter attribute
        else:
            # If the reverse edge does not exist, this pair cannot form a complete reciprocal relationship for filtering.
            continue

        # At this point, both u->v and v->u exist in the original graph and have the filter attribute.

        # 4c. Calculate the relationship intensity for the pair for filtering purposes.
        # Since _diff_norm_abs and _diff_abs are already absolute and symmetric values,
        # the pair's intensity is simply the value from one of the two directions.
        pair_intensity_for_filter = filter_value_uv

        # 4d. Apply the filter condition to the pair's intensity
        condition_met_for_pair = False
        if (keep_above and pair_intensity_for_filter >= threshold) or \
                (not keep_above and pair_intensity_for_filter <= threshold):
            condition_met_for_pair = True

        # 4e. If the pair does not meet the filter condition, skip it (do not add the undirected edge)
        if not condition_met_for_pair:
            continue

        # 4f. If the condition is met, add the UNDIRECTED edge to filtered_G
        # Retrieve the normalized absolute difference value from the u->v edge (or v->u, it's the same)
        diff_norm_abs_value = data_uv.get(weight_source_attribute_name, 0.0)

        # Transform the difference into a similarity weight for Louvain (1 - difference)
        louvain_weight = 1 - diff_norm_abs_value

        filtered_G.add_edge(u, v, weight=louvain_weight)  # Add a single undirected edge
        edges_retained += 1

    print(
        f"Undirected graph filtered by '{filter_attribute_name}' (threshold: {threshold}, {filter_condition_str}). Resulting undirected graph has {filtered_G.number_of_nodes()} nodes and {filtered_G.number_of_edges()} edges ({edges_retained} retained).")

    if filtered_G.number_of_edges() == 0:
        print(f"No edges remain after filtering. Consider adjusting the threshold or filter type.")
        return None

    return filtered_G


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
            f"Warning: Eigenvector centrality did not converge for metric {metric}. This can happen in disconnected graphs or specific weight distributions. Setting scores to 0.0.")
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
    Generates individual scatter plots to visualize the relationship between specified normalized metrics
    and normalized total league points. Each plot is presented as a separate figure.
    Teams are differentiated by color using a categorical palette.

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
    base_fontsize = 12
    title_fontsize = base_fontsize + 2  # Slightly larger for titles
    label_fontsize = base_fontsize + 2
    tick_fontsize = base_fontsize
    legend_fontsize = base_fontsize - 3.3
    annotation_fontsize = base_fontsize

    for i, (metric_col, title_suffix) in enumerate(metrics_to_plot.items()):
        # Create a NEW figure and a NEW set of axes for each individual plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Set appropriate dimensions for a single plot

        # Set a specific title for each plot
        ax.set_title(f'{title_suffix} vs. Normalized League Points ({scope_description})', fontsize=title_fontsize)

        sns.scatterplot(
            data=plot_data,  # Use the DataFrame with the reset index
            x=metric_col,
            y='points_norm',
            hue='Team',
            palette=custom_palette_46,
            s=150,
            alpha=0.8,
            ax=ax,
            legend='full'
        )

        # Adjust the legend position based on the number of teams
        if num_teams > 10:
            ax.legend(title='Team', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                      fontsize=legend_fontsize,
                      ncol=1, title_fontsize=label_fontsize)  # Set title_fontsize for the legend title
        else:
            ax.legend(title='Team', loc='best', fontsize=legend_fontsize,
                      title_fontsize=label_fontsize)  # Set title_fontsize for the legend title

        # Add a linear regression line to show general trend
        sns.regplot(
            data=plot_data,  # Use the DataFrame with the reset index
            x=metric_col,
            y='points_norm',
            scatter=False,
            color='gray',
            line_kws={'linestyle': '--', 'alpha': 0.7},
            ax=ax
        )

        ax.set_xlabel(title_suffix, fontsize=label_fontsize)
        ax.set_ylabel('Normalized Points', fontsize=label_fontsize)

        # Set tick label font size
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

        # Annotate the plot with the Pearson correlation coefficient
        correlation = normalized_ratios_df[metric_col].corr(normalized_ratios_df['points_norm'])
        ax.text(0.05, 0.95, f'Pearson Correlation: {correlation:.2f}',
                transform=ax.transAxes, fontsize=annotation_fontsize,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()  # Adjust the layout for this single figure
        plt.show()


# Example usage (assuming normalized_ratios_df and scope_description are defined)
# This part is just for demonstration and not part of the requested modification.
# if __name__ == '__main__':
#     # Create a dummy DataFrame for demonstration
#     data = {
#         'Team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H', 'Team I', 'Team J'],
#         'goals_scored_norm': [0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 0.75, 0.65, 0.85, 0.55],
#         'aggressiveness_committed_norm': [0.7, 0.8, 0.6, 0.9, 0.5, 0.85, 0.55, 0.75, 0.65, 0.95],
#         'shot_accuracy_norm': [0.75, 0.65, 0.85, 0.55, 0.95, 0.45, 0.8, 0.7, 0.9, 0.6],
#         'control_norm': [0.6, 0.7, 0.8, 0.9, 0.4, 0.75, 0.55, 0.85, 0.65, 0.95],
#         'points_norm': [0.9, 0.7, 0.8, 0.6, 1.0, 0.5, 0.85, 0.75, 0.95, 0.65]
#     }
#     normalized_df = pd.DataFrame(data).set_index('Team')
#     scope = "Test Season 2023/24"
#     plot_metric_ratios_vs_points(normalized_df, scope)


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


# Modified plot_network_communities function
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

    plt.figure(figsize=(16, 14))
    ax = plt.gca()
    plt.title(f"Network Communities for {metric_name} ({scope_desc})", fontsize=18)

    team_names_in_graph = list(graph.nodes())
    loaded_logos = _load_team_logos(team_names_in_graph, logo_folder)

    categorical_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    node_colors = []
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
                community_centroid_graph.add_edge(comm_u, comm_v)

        try:
            centroid_pos = nx.spectral_layout(community_centroid_graph, weight=None)
        except Exception:
            print("Warning: Spectral layout for centroids failed, falling back to spring layout.")
            centroid_pos = nx.spring_layout(community_centroid_graph, k=2.0, iterations=100, seed=42)

        if not centroid_pos:
            centroid_pos = {comm_id: np.random.rand(2) * 2 - 1 for comm_id in unique_communities}

        initial_pos = {}
        for node_name in graph.nodes():
            comm_id = communities.get(node_name)
            if comm_id is not None and comm_id in centroid_pos:
                initial_pos[node_name] = centroid_pos[comm_id] + np.random.rand(2) * 0.03 - 0.015
            else:
                initial_pos[node_name] = np.random.rand(2) * 2 - 1

        optimal_k = 0.5 / np.sqrt(num_nodes) if num_nodes > 0 else 0.15

        try:
            pos = nx.spring_layout(graph, pos=initial_pos, k=optimal_k, iterations=150, seed=42, weight='weight')
        except Exception as e:
            print(f"Error during spring layout refinement: {e}. Falling back to default spring layout.")
            pos = nx.spring_layout(graph, k=0.15, iterations=50, seed=42)

    else:
        print("Less than 2 communities detected or 0 edges; using standard spring layout.")
        optimal_k = 0.5 / np.sqrt(num_nodes) if num_nodes > 0 else 0.15
        pos = nx.spring_layout(graph, k=optimal_k, iterations=100, seed=42, weight='weight')

    # Draw the colored nodes (circles) first
    # Keeping alpha=0.9 for the background circle, but ensuring the logo is on top and unaffected.
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_base_size,
                           alpha=0.9, linewidths=1, edgecolors='black')

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
                                frameon=False,  # No frame around the image
                                pad=0.0,
                                boxcoords="data",  # Position image using data coordinates
                                xycoords="data",
                                # IMPORTANT: Add an explicit white background for the image to prevent color bleed
                                bboxprops=dict(boxstyle="square,pad=0", fc="white", ec="none", alpha=1.0)
                                )
            ax.add_artist(ab)
        else:
            # Fallback for nodes without logos: draw text
            ax.text(p[0], p[1], node_name, fontsize=8, ha='center', va='center', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Adjust overall scaling of the plot area
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]

        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            x_pad = (max_x - min_x) * 0.15 if num_nodes > 1 else 0.2
            y_pad = (max_y - min_y) * 0.15 if num_nodes > 1 else 0.2

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
        plt.legend(handles=legend_handles, title="Communities", loc='upper left', bbox_to_anchor=(1, 1),
                   fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.axis('off')
    plt.show()
