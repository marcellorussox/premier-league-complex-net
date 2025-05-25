from collections import defaultdict
from typing import Optional, List, Dict, Any

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

# Assuming these are correctly imported from your project structure
from src.data_utils import calculate_and_normalize_ratios, calculate_team_points
from src.graph_utils import plot_metric_ratios_vs_points, find_communities, analyze_centrality_vs_points, \
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
):
    """
    Performs a comprehensive network analysis of English Premier League data, including:
    - Graph creation for specified seasons or the entire dataset.
    - Calculation of initial network properties (nodes, edges, density).
    - Team league points calculation.
    - Iterative filtering of the graph based on various metrics and thresholds.
    - Calculation and printing of centrality measures (Degree, Strength, Betweenness, Closeness, Eigenvector).
    - Pearson correlation analysis between centrality measures and league points.
    - Community detection using the Louvain algorithm.
    - Visualization of normalized metric ratios against normalized league points.

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
    """
    if metrics_to_analyze is None:
        metrics_to_analyze = ['goals', 'aggressiveness', 'shot_accuracy', 'control', 'points']

    if thresholds_for_analysis is not None and len(thresholds_for_analysis) != len(metrics_to_analyze):
        print("Error: The length of 'thresholds_for_analysis' must match the length of 'metrics_to_analyze'. Aborting "
              "analysis.")
        return

    df_for_scope = epl_df.copy()
    scope_description = "the entire dataset"
    graph_to_analyze = None

    # Determine the scope of analysis (single season, season range, or entire dataset)
    if network_start_year is not None and network_end_year is not None:
        seasons_in_range = []
        # Loop from network_start_year up to, but not including, network_end_year.
        # This aligns with the previous refinement where network_end_year=2020 means up to 2019/2020 season.
        for year in range(network_start_year, network_end_year):
            next_year_suffix = str(year + 1)[-2:]
            seasons_in_range.append(f"{year}/{next_year_suffix}")

        df_for_scope = epl_df[epl_df['Season'].isin(seasons_in_range)].copy()
        # The description reflects that network_end_year is the year *after* the last season's starting year.
        scope_description = f"seasons from {network_start_year}/{str(network_start_year + 1)[-2:]} to {network_end_year - 1}/{str(network_end_year)[-2:]}"
        print(f"\n--- Starting EPL Network Analysis for {scope_description} ---")
        # create_epl_network handles its internal season filtering consistently with this logic.
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
        return

    print("\n--- Overall Network Properties (Initial Directed Graph) ---")
    print(f"Number of nodes: {graph_to_analyze.number_of_nodes()}")
    print(f"Number of edges: {graph_to_analyze.number_of_edges()}")
    # Check for zero nodes to avoid division by zero in density calculation for empty graphs
    if graph_to_analyze.number_of_nodes() > 1:
        print(f"Network Density: {nx.density(graph_to_analyze):.4f}")
    else:
        print(f"Network Density: Not applicable for graph with 0 or 1 node.")


    league_points = calculate_team_points(df_for_scope)

    all_centrality_results_by_metric = {}

    print("\n--- Starting Unified Centrality and Community Analysis ---")

    for i, metric_base_name in enumerate(metrics_to_analyze):
        current_threshold = None
        if thresholds_for_analysis is not None:
            current_threshold = thresholds_for_analysis[i]

        print(f"\n### Analysis for Metric: '{metric_base_name}' ###")

        print(f"\n--- Filtering graph for '{metric_base_name}' with threshold {current_threshold} ---")

        # analysis_graph_filtered is the UNDIRECTED graph with edge weights correctly set for similarity.
        analysis_graph_filtered = filter_graph_by_weight(
            graph=graph_to_analyze,
            metric=metric_base_name,
            threshold=current_threshold,
            use_normalized_abs_diff_for_filter=use_normalized_abs_diff_for_filter,
            keep_above=keep_above_threshold
        )

        if not analysis_graph_filtered or analysis_graph_filtered.number_of_edges() == 0:
            print(
                f"No edges remaining for '{metric_base_name}' after filtering. Skipping centrality and community "
                f"detection for this metric.")
            continue

        print(f"\n--- Calculating Centralities for '{metric_base_name}' (on Filtered Undirected Graph) ---")

        centrality_scores = calculate_and_print_centralities(
            analysis_graph_filtered,
            metric=metric_base_name
        )

        if centrality_scores and 'degree' in centrality_scores: # Check for at least one centrality result
            all_centrality_results_by_metric[metric_base_name] = centrality_scores

            if league_points:
                teams_in_graph = list(analysis_graph_filtered.nodes())
                if not teams_in_graph:
                    print("\nNo teams found in the filtered graph for centrality. Skipping correlation analysis.")
                else:
                    print(
                        f"\n--- Performing Pearson Correlation Analysis for {metric_base_name} Centrality vs. League Points ---")
                    # analyze_centrality_vs_points iterates over all centralities in centrality_scores
                    analyze_centrality_vs_points(
                        centrality_scores,
                        league_points,
                        teams_in_graph
                    )
            else:
                print("\nSkipping Pearson correlation analysis due to missing league points.")
        else:
            print(f"Skipping correlation analysis for '{metric_base_name}' due to no valid centrality results.")

        print(f"\n--- Starting Community Detection for Metric: '{metric_base_name}' ---")

        # The graph is already prepared with correct weights by filter_graph_by_weight
        community_graph = analysis_graph_filtered

        if community_graph.number_of_edges() > 0:
            communities = find_communities(community_graph, resolution=community_resolution)

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
                f"Skipping Community Detection for '{metric_base_name}' due to no edges after filtering or an empty "
                f"graph.")

    # --- Plotting Metric Ratios vs. Points ---
    if df_for_scope is not None and not df_for_scope.empty:
        normalized_ratios_df = calculate_and_normalize_ratios(df_for_scope)
        if normalized_ratios_df is not None and not normalized_ratios_df.empty:
            league_points_for_plot = pd.DataFrame(list(league_points.items()), columns=['Team', 'Points']).set_index(
                'Team')

            # Normalize points for plotting
            min_points = league_points_for_plot['Points'].min()
            max_points = league_points_for_plot['Points'].max()
            if max_points == min_points:
                # Handle case with no variance in points
                league_points_for_plot['Normalized_Points'] = 0.5
            else:
                league_points_for_plot['Normalized_Points'] = (league_points_for_plot['Points'] - min_points) / (
                            max_points - min_points)

            # Join normalized ratios with normalized points
            plot_df = normalized_ratios_df.join(league_points_for_plot[['Normalized_Points']], how='inner', lsuffix='_ratio')

            if not plot_df.empty:
                plot_metric_ratios_vs_points(plot_df, scope_description)
                # plt.show() is called inside plot_metric_ratios_vs_points for each plot
            else:
                print("\nSkipping Metric Ratios vs. Points plot as no common data for plotting after join.")
        else:
            print("\nSkipping Metric Ratios vs. Points plot due to issues in calculating or normalizing ratios.")
    else:
        print("\nSkipping Metric Ratios vs. Points plot due to missing DataFrame for analysis scope.")

    print("\n--- EPL Network Analysis Complete ---")
