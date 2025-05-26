from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any

import networkx as nx
import pandas as pd

# Assuming these are correctly imported from your project structure
from src.data_utils import calculate_team_points
from src.graph_utils import find_communities, analyze_centrality_vs_points, plot_network_communities, \
    calculate_and_print_centralities, filter_graph_by_weight, create_epl_network


def perform_epl_network_analysis(
        epl_df: pd.DataFrame,
        analysis_season: Optional[str] = None,
        network_start_year: Optional[int] = None,
        network_end_year: Optional[int] = None,
        metrics_to_analyze: Optional[List[str]] = None,
        thresholds_for_analysis: Optional[List[float]] = None,
        keep_above_threshold: bool = True,
        community_resolution: float = 1.0,
        use_normalized_abs_diff_for_filter: bool = True
) -> Dict[str, Any]:
    """
    Performs a comprehensive network analysis of English Premier League data for multiple metrics.
    This includes graph creation, calculation of network properties, centrality measures,
    Pearson correlation analysis, and community detection.
    It collects all necessary data for plotting network communities and returns it,
    allowing for external plotting after all analyses are complete.

    Args:
        epl_df (pd.DataFrame): The main DataFrame containing EPL match data.
        analysis_season (Optional[str]): A specific season to analyze (e.g., "2016/2017").
                                         If provided, `network_start_year` and `network_end_year` are ignored.
        network_start_year (Optional[int]): The starting year of the season range for network construction.
                                            (e.g., 2015 for 2015/2016 season).
        network_end_year (Optional[int]): The ending year for the season range for network construction.
                                          If set to Y, the last season included will be (Y-1)/Y.
                                          (e.g., 2020 means up to 2019/2020 season).
        metrics_to_analyze (Optional[List[str]]): A list of metric base names to analyze (e.g., ['goals', 'control']).
                                                  Defaults to a predefined list if None.
        thresholds_for_analysis (Optional[List[float]]): A list of float thresholds, corresponding one-to-one
                                                         with `metrics_to_analyze`. If None, no filtering threshold
                                                         is applied for that metric.
        keep_above_threshold (bool): If True, edges where the filter metric is >= threshold are kept.
                                     If False, edges where the filter metric is <= threshold are kept.
        community_resolution (float): The resolution parameter for the Louvain community detection algorithm.
                                      Higher values lead to more, smaller communities. Default is 1.0.
        use_normalized_abs_diff_for_filter (bool): If True, filtering uses the normalized absolute difference
                                                   ('{metric}_diff_norm_abs'). If False, it uses the raw
                                                   absolute difference ('{metric}_diff_abs').

    Returns:
        Dict[str, Any]: A dictionary containing all analysis results. Keys include:
                        - 'network_metrics_results': Dict[str, Dict[str, Any]] (detailed per-metric network results, including graph and communities for plotting)
                        - 'df_for_scope': pd.DataFrame (the original dataframe sliced by scope)
                        - 'scope_description': str
                        - 'league_points': Dict[str, int]
                        Returns an empty dictionary if network creation fails.
    """
    if metrics_to_analyze is None:
        metrics_to_analyze = ['goals', 'aggressiveness', 'shot_accuracy', 'control', 'points']

    if thresholds_for_analysis is None:
        thresholds_for_analysis = [None] * len(metrics_to_analyze)

    if len(thresholds_for_analysis) != len(metrics_to_analyze):
        print(
            "Error: The length of 'thresholds_for_analysis' must match the length of 'metrics_to_analyze'. Aborting "
            "analysis.")
        return {}

    df_for_scope = epl_df.copy()
    scope_description = "the entire dataset"
    graph_to_analyze = None

    # Determine the scope of analysis (single season, season range, or entire dataset)
    if network_start_year is not None and network_end_year is not None:
        seasons_in_range = []
        for year in range(network_start_year, network_end_year):
            next_year_suffix = str(year + 1)[-2:]
            seasons_in_range.append(f"{year}/{next_year_suffix}")

        df_for_scope = epl_df[epl_df['Season'].isin(seasons_in_range)].copy()
        scope_description = f"seasons from {network_start_year}/{str(network_start_year + 1)[-2:]} to {network_end_year - 1}/{str(network_end_year)[-2:]}"
        print(f"\n--- Starting EPL Network Analysis for {scope_description} ---")
        graph_to_analyze = create_epl_network(epl_df, start_year=network_start_year, end_year=network_end_year)

    elif analysis_season:
        df_for_scope = epl_df[epl_df['Season'] == analysis_season].copy()
        scope_description = f"season {analysis_season}"
        print(f"\n--- Starting EPL Network Analysis for {scope_description} ---")
        graph_to_analyze = create_epl_network(epl_df, season=analysis_season)

    else:
        print(f"\n--- Starting EPL Network Analysis for {scope_description} ---")
        graph_to_analyze = create_epl_network(epl_df)

    if graph_to_analyze is None:
        print("Network creation failed. Aborting analysis.")
        return {}

    print("\n--- Overall Network Properties (Initial Directed Graph) ---")
    print(f"Number of nodes: {graph_to_analyze.number_of_nodes()}")
    print(f"Number of edges: {graph_to_analyze.number_of_edges()}")
    if graph_to_analyze.number_of_nodes() > 1:
        print(f"Network Density: {nx.density(graph_to_analyze):.4f}")
    else:
        print(f"Network Density: Not applicable for graph with 0 or 1 node.")

    league_points = calculate_team_points(df_for_scope)

    # This will store all the network analysis results for each metric
    network_metrics_results = {}

    print("\n--- Starting Unified Centrality and Community Analysis ---")

    for i, metric_base_name in enumerate(metrics_to_analyze):
        current_threshold = thresholds_for_analysis[i]

        print(f"\n### Analysis for Metric: '{metric_base_name}' ###")

        print(f"\n--- Filtering graph for '{metric_base_name}' with threshold {current_threshold} ---")

        analysis_graph_filtered = filter_graph_by_weight(
            graph=graph_to_analyze,
            metric=metric_base_name,
            threshold=current_threshold,
            use_normalized_abs_diff_for_filter=use_normalized_abs_diff_for_filter,
            keep_above=keep_above_threshold
        )

        if not analysis_graph_filtered or analysis_graph_filtered.number_of_nodes() == 0:
            print(
                f"No nodes or edges remaining for '{metric_base_name}' after filtering. Skipping centrality, "
                f"community detection, and plotting for this metric.")
            # Store None for graph/communities if not available
            network_metrics_results[metric_base_name] = {
                'filtered_graph': None,
                'centrality_scores': None,
                'communities': None,
            }
            continue

        print(f"\n--- Calculating Centralities for '{metric_base_name}' (on Filtered Undirected Graph) ---")

        centrality_scores = calculate_and_print_centralities(
            analysis_graph_filtered,
            metric=metric_base_name
        )

        communities = None
        print(f"\n--- Starting Community Detection for Metric: '{metric_base_name}' ---")

        if analysis_graph_filtered.number_of_edges() > 0:
            communities = find_communities(analysis_graph_filtered, resolution=community_resolution)

            if communities:
                print("\nDetected Communities:")
                communities_by_id = defaultdict(list)
                for team, comm_id in communities.items():
                    communities_by_id[comm_id].append(team)

                for comm_id in sorted(communities_by_id.keys()):
                    teams = communities_by_id[comm_id]
                    print(f"  Community {comm_id}: {', '.join(teams)}")
            else:
                print("No communities found in the filtered graph for this metric.")
        else:
            print(
                f"Skipping Community Detection for '{metric_base_name}' due to no edges after filtering or an empty graph.")

        # Store all relevant data for the current metric's network analysis
        network_metrics_results[metric_base_name] = {
            'filtered_graph': analysis_graph_filtered,
            'centrality_scores': centrality_scores,
            'communities': communities,
        }

        # Perform Pearson Correlation Analysis for this metric
        if centrality_scores and 'degree' in centrality_scores and league_points:
            teams_in_graph = list(analysis_graph_filtered.nodes())
            if not teams_in_graph:
                print("\nNo teams found in the filtered graph for centrality. Skipping correlation analysis.")
            else:
                print(
                    f"\n--- Performing Pearson Correlation Analysis for {metric_base_name} Centrality vs. League "
                    f"Points ---")
                analyze_centrality_vs_points(
                    centrality_scores,
                    league_points,
                    teams_in_graph
                )
        else:
            print(
                f"Skipping Pearson correlation analysis for '{metric_base_name}' due to no valid centrality results "
                f"or missing league points.")

    print("\n--- EPL Network Analysis Complete ---")

    return {
        'network_metrics_results': network_metrics_results,
        'df_for_scope': df_for_scope,
        'scope_description': scope_description,
        'league_points': league_points
    }
