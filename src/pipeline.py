from collections import defaultdict
from typing import Optional, List, Dict, Any

import networkx as nx
import pandas as pd

from src.data_utils import calculate_team_points
from src.graph_utils import find_communities, analyze_centrality_vs_points, calculate_and_print_centralities, create_epl_network


def perform_epl_network_analysis(
        epl_df: pd.DataFrame,
        analysis_season: Optional[str] = None,
        network_start_year: Optional[int] = None,
        network_end_year: Optional[int] = None,
        metrics_to_analyze: Optional[List[str]] = None,
        community_resolution: Optional[List[float]] = None,
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
        community_resolution (Optional[List[float]]): The resolution parameters for the Louvain community detection algorithm,
                                                      they correspond one-to-one to metrics_to_analyze.
                                                      Higher values lead to more, smaller communities. Default is 1.0.

    Returns:
        Dict[str, Any]: A dictionary containing all analysis results. Keys include:
                        - 'network_metrics_results': Dict[str, Dict[str, Any]] (detailed per-metric network results,
                            including graph and communities for plotting)
                        - 'df_for_scope': pd.DataFrame (the original dataframe sliced by scope)
                        - 'scope_description': str
                        - 'league_points': Dict[str, int]
                        Returns an empty dictionary if network creation fails.
    """
    if metrics_to_analyze is None:
        metrics_to_analyze = ['goals', 'aggressiveness', 'shot_accuracy', 'control', 'points']

    df_for_scope = epl_df.copy()
    scope_description = "the entire dataset"

    league_points = calculate_team_points(df_for_scope)

    # This will store all the network analysis results for each metric
    network_metrics_results = {}

    for i, metric_name in enumerate(metrics_to_analyze):

        # Determine the scope of analysis (single season, season range, or entire dataset)
        if network_start_year is not None and network_end_year is not None:
            seasons_in_range = []
            for year in range(network_start_year, network_end_year):
                next_year_suffix = str(year + 1)[-2:]
                seasons_in_range.append(f"{year}/{next_year_suffix}")

            df_for_scope = epl_df[epl_df['Season'].isin(seasons_in_range)].copy()
            scope_description = f"seasons from {network_start_year}/{str(network_start_year + 1)[-2:]} to {network_end_year - 1}/{str(network_end_year)[-2:]}"
            graph_to_analyze = create_epl_network(epl_df, start_year=network_start_year, end_year=network_end_year, metric_name=metric_name)

        elif analysis_season:
            df_for_scope = epl_df[epl_df['Season'] == analysis_season].copy()
            scope_description = f"season {analysis_season}"
            graph_to_analyze = create_epl_network(epl_df, season=analysis_season, metric_name=metric_name)

        else:
            graph_to_analyze = create_epl_network(epl_df, metric_name=metric_name)
        if graph_to_analyze is None:
            print("Network creation failed. Aborting analysis.")
            return {}

        print(f"\n--- Starting EPL Network Analysis for {scope_description} ---")

        print("\n--- Overall Network Properties (Initial Directed Graph) ---")
        print(f"Number of nodes: {graph_to_analyze.number_of_nodes()}")
        print(f"Number of edges: {graph_to_analyze.number_of_edges()}")
        if graph_to_analyze.number_of_nodes() > 1:
            print(f"Network Density: {nx.density(graph_to_analyze):.4f}")
        else:
            print(f"Network Density: Not applicable for graph with 0 or 1 node.")

        print("\n--- Starting Unified Centrality and Community Analysis ---")

        print(f"\n### Analysis for Metric: '{metric_name}' ###")

        if not graph_to_analyze or graph_to_analyze.number_of_nodes() == 0:
            print(
                f"No nodes or edges remaining for '{metric_name}' after filtering. Skipping centrality, "
                f"community detection, and plotting for this metric.")
            # Store None for graph/communities if not available
            network_metrics_results[metric_name] = {
                'filtered_graph': None,
                'centrality_scores': None,
                'communities': None,
            }
            continue

        print(f"\n--- Calculating Centralities for '{metric_name}' (on Undirected Graph) ---")

        centrality_scores = calculate_and_print_centralities(
            graph_to_analyze,
            metric=metric_name
        )

        communities = None
        print(f"\n--- Starting Community Detection for Metric: '{metric_name}' ---")

        if graph_to_analyze.number_of_edges() > 0:
            communities = find_communities(graph_to_analyze, resolution=community_resolution[i])

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
                f"Skipping Community Detection for '{metric_name}' due to no edges after filtering or an empty "
                f"graph.")

        # Store all relevant data for the current metric's network analysis
        network_metrics_results[metric_name] = {
            'filtered_graph': graph_to_analyze,
            'centrality_scores': centrality_scores,
            'communities': communities,
        }

        # Perform Pearson Correlation Analysis for this metric
        if centrality_scores and 'degree' in centrality_scores and league_points:
            teams_in_graph = list(graph_to_analyze.nodes())
            if not teams_in_graph:
                print("\nNo teams found in the filtered graph for centrality. Skipping correlation analysis.")
            else:
                print(
                    f"\n--- Performing Pearson Correlation Analysis for {metric_name} Centrality vs. League "
                    f"Points ---")
                analyze_centrality_vs_points(
                    centrality_scores,
                    league_points,
                    teams_in_graph
                )
        else:
            print(
                f"Skipping Pearson correlation analysis for '{metric_name}' due to no valid centrality results "
                f"or missing league points.")

    print("\n--- EPL Network Analysis Complete ---")

    return {
        'network_metrics_results': network_metrics_results,
        'df_for_scope': df_for_scope,
        'scope_description': scope_description,
        'league_points': league_points
    }
