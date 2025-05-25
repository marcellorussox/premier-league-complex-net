from collections import defaultdict
from typing import Optional, Dict, Tuple

import pandas as pd


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a dataset from a specified CSV file path into a pandas DataFrame.
    It also attempts to convert a 'MatchDate' column to datetime objects if present.

    Args:
        file_path (str): The absolute or relative path to the CSV file.

    Returns:
        Optional[pd.DataFrame]: The loaded pandas DataFrame if successful.
                                Returns None if the specified file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from: {file_path}")
        print(f"Loaded dataset shape: {df.shape}")

        # Attempt to convert 'MatchDate' column to datetime objects
        if 'MatchDate' in df.columns:
            df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
            # 'errors='coerce'' will turn unparseable dates into NaT (Not a Time)

        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse the CSV file at {file_path}. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data from {file_path}. Details: {e}")
        return None


def calculate_and_normalize_ratios(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculates aggregated 'for' and 'against' statistics for various football metrics
    (e.g., goals, shot accuracy, aggressiveness, control) for every team across the provided
    match data. These aggregated totals are then normalized using Min-Max scaling.

    The function specifically returns a DataFrame containing only the normalized 'for' metrics
    (e.g., 'goals_scored_norm') and normalized total points.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw match data.
                           This DataFrame can span one or multiple seasons.
                           Expected columns include 'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals',
                           'FullTimeAwayGoals', 'HomeShots', 'AwayShots', 'HomeShotsOnTarget',
                           'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners', 'HomeFouls',
                           'AwayFouls', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards',
                           'AwayRedCards'.

    Returns:
        Optional[pd.DataFrame]: A DataFrame indexed by 'Team', containing the following normalized columns:
                                - 'goals_scored_norm'
                                - 'aggressiveness_committed_norm'
                                - 'shot_accuracy_norm'
                                - 'control_norm'
                                - 'points_norm'
                                Returns None if the input DataFrame is empty or critical columns are missing
                                after initial processing.
    """
    print("\n--- Calculating and Normalizing Team Totals (For/Against) across the provided data ---")

    if df.empty:
        print("Input DataFrame is empty. Cannot calculate totals.")
        return None

    # Ensure required columns are numeric and handle missing values by filling with zero.
    # This list includes all columns that will be used for aggregation.
    cols_to_process = [
        'FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'AwayShots',
        'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards'
    ]

    # Work on a copy to prevent modification of the original DataFrame and avoid SettingWithCopyWarning.
    df_working = df.copy()
    for col in cols_to_process:
        if col not in df_working.columns:
            # Add the column with zeros if it's missing in the DataFrame
            df_working[col] = 0
        else:
            # Convert column to numeric, coercing errors to NaN, then fill NaNs with 0
            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').fillna(0)

    # Initialize a defaultdict to store aggregated statistics for each team.
    # The lambda provides a default dictionary for new teams encountered.
    team_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        'goals_for': 0.0, 'goals_against': 0.0,  # Store 'for'/'against' for intermediate calculations
        'sot_for': 0.0, 'sot_against': 0.0,
        'total_shots_for': 0.0, 'total_shots_against': 0.0,
        'fouls_for': 0.0, 'fouls_against': 0.0,
        'yc_for': 0.0, 'yc_against': 0.0,
        'rc_for': 0.0, 'rc_against': 0.0,
        'corners_for': 0.0, 'corners_against': 0.0,
        'points': 0.0  # Store points for normalization later
    })

    # Iterate through each match to aggregate team statistics
    for _, match in df_working.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        # Update statistics for the Home Team
        team_stats[home_team]['goals_for'] += match.get('FullTimeHomeGoals', 0)
        team_stats[home_team]['goals_against'] += match.get('FullTimeAwayGoals', 0)
        team_stats[home_team]['sot_for'] += match.get('HomeShotsOnTarget', 0)
        team_stats[home_team]['sot_against'] += match.get('AwayShotsOnTarget', 0)
        team_stats[home_team]['total_shots_for'] += match.get('HomeShots', 0)
        team_stats[home_team]['total_shots_against'] += match.get('AwayShots', 0)
        team_stats[home_team]['fouls_for'] += match.get('HomeFouls', 0)
        team_stats[home_team]['fouls_against'] += match.get('AwayFouls', 0)
        team_stats[home_team]['yc_for'] += match.get('HomeYellowCards', 0)
        team_stats[home_team]['yc_against'] += match.get('AwayYellowCards', 0)
        team_stats[home_team]['rc_for'] += match.get('HomeRedCards', 0)
        team_stats[home_team]['rc_against'] += match.get('AwayRedCards', 0)
        team_stats[home_team]['corners_for'] += match.get('HomeCorners', 0)
        team_stats[home_team]['corners_against'] += match.get('AwayCorners', 0)

        # Update statistics for the Away Team
        team_stats[away_team]['goals_for'] += match.get('FullTimeAwayGoals', 0)
        team_stats[away_team]['goals_against'] += match.get('FullTimeHomeGoals', 0)
        team_stats[away_team]['sot_for'] += match.get('AwayShotsOnTarget', 0)
        team_stats[away_team]['sot_against'] += match.get('HomeShotsOnTarget', 0)
        team_stats[away_team]['total_shots_for'] += match.get('AwayShots', 0)
        team_stats[away_team]['total_shots_against'] += match.get('HomeShots', 0)
        team_stats[away_team]['fouls_for'] += match.get('AwayFouls', 0)
        team_stats[away_team]['fouls_against'] += match.get('HomeFouls', 0)
        team_stats[away_team]['yc_for'] += match.get('AwayYellowCards', 0)
        team_stats[away_team]['yc_against'] += match.get('HomeYellowCards', 0)
        team_stats[away_team]['rc_for'] += match.get('AwayRedCards', 0)
        team_stats[away_team]['rc_against'] += match.get('HomeRedCards', 0)
        team_stats[away_team]['corners_for'] += match.get('AwayCorners', 0)
        team_stats[away_team]['corners_against'] += match.get('HomeCorners', 0)

        # Update points based on match outcome
        if match.get('FullTimeHomeGoals', 0) > match.get('FullTimeAwayGoals', 0):
            team_stats[home_team]['points'] += 3
        elif match.get('FullTimeAwayGoals', 0) > match.get('FullTimeHomeGoals', 0):
            team_stats[away_team]['points'] += 3
        else:  # Draw
            team_stats[home_team]['points'] += 1
            team_stats[away_team]['points'] += 1

    # Convert the aggregated team statistics into a DataFrame
    df_ratios = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    df_ratios.rename(columns={'index': 'Team'}, inplace=True)

    # Calculate composite metrics
    # Aggressiveness: sum of fouls, yellow cards, and red cards (red cards weighted higher)
    df_ratios['aggressiveness_for'] = df_ratios['fouls_for'] + df_ratios['yc_for'] + 3 * df_ratios['rc_for']
    df_ratios['aggressiveness_against'] = df_ratios['fouls_against'] + df_ratios['yc_against'] + 3 * df_ratios['rc_against']

    # Shot Accuracy: Shots on Target divided by Total Shots. Handle division by zero.
    df_ratios['shot_accuracy_for'] = df_ratios.apply(
        lambda x: (x['sot_for'] / x['total_shots_for']) if x['total_shots_for'] > 0 else 0.0, axis=1)
    df_ratios['shot_accuracy_against'] = df_ratios.apply(
        lambda x: (x['sot_against'] / x['total_shots_against']) if x['total_shots_against'] > 0 else 0.0, axis=1)

    # Control: sum of corners and total shots. Often indicates team dominance.
    df_ratios['control_for'] = df_ratios['corners_for'] + df_ratios['total_shots_for']
    df_ratios['control_against'] = df_ratios['corners_against'] + df_ratios['total_shots_against']

    # Min-Max Scaling for normalization of all relevant totals ('for', 'against', and 'points').
    metrics_to_normalize = [
        'goals_for', 'goals_against',
        'aggressiveness_for', 'aggressiveness_against',
        'shot_accuracy_for', 'shot_accuracy_against',
        'control_for', 'control_against',
        'points'
    ]

    for metric in metrics_to_normalize:
        col_name_norm = f'{metric}_norm' if metric != 'points' else 'points_norm'

        # Apply Min-Max scaling. If max and min are the same (no variance), set to 0.5 to avoid division by zero.
        if df_ratios[metric].max() == df_ratios[metric].min():
            df_ratios[col_name_norm] = 0.5
        else:
            df_ratios[col_name_norm] = (df_ratios[metric] - df_ratios[metric].min()) / \
                                       (df_ratios[metric].max() - df_ratios[metric].min())

    print("Calculation and normalization complete.")

    # Rename the normalized 'for' columns to their final desired names for clarity.
    df_ratios.rename(columns={
        'goals_for_norm': 'goals_scored_norm',
        'aggressiveness_for_norm': 'aggressiveness_committed_norm',
        'shot_accuracy_for_norm': 'shot_accuracy_norm',
        'control_for_norm': 'control_norm'
    }, inplace=True)

    # Set the 'Team' column as the DataFrame index as required for subsequent analysis.
    df_ratios.set_index('Team', inplace=True)

    # Return only the specifically requested normalized columns.
    columns_to_return_strictly = [
        'goals_scored_norm',
        'aggressiveness_committed_norm',
        'shot_accuracy_norm',
        'control_norm',
        'points_norm'
    ]

    # Ensure all requested columns exist before attempting to select them,
    # in case any calculation resulted in an unexpected missing column.
    final_columns = [col for col in columns_to_return_strictly if col in df_ratios.columns]

    return df_ratios[final_columns]


def calculate_team_points(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculates the total league points for each team based on the provided DataFrame of match data.
    This function aggregates points for all matches present in the input DataFrame,
    regardless of specific seasons or temporal scope.

    Args:
        df (pd.DataFrame): The DataFrame containing match data. Expected columns include
                           'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', and 'FullTimeAwayGoals'.

    Returns:
        Dict[str, int]: A dictionary where keys are team names (str) and values are their
                        accumulated total points (int). Returns an empty dictionary if the
                        input DataFrame is empty or lacks essential columns for point calculation.
    """
    print("\n--- Calculating Total Team Points across the provided data ---")

    if df.empty:
        print("Input DataFrame is empty. Cannot calculate points.")
        return {}

    team_points: Dict[str, int] = defaultdict(int)

    required_point_cols = ['HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals']
    if not all(col in df.columns for col in required_point_cols):
        print(
            f"Error: Missing one or more required columns for point calculation: {required_point_cols}. Returning empty points dictionary.")
        return {}

    # Create a working copy of the DataFrame to prevent modifying the original
    # and to avoid potential SettingWithCopyWarning issues during type conversion.
    df_working = df.copy()

    # Ensure goal columns are numeric and handle any non-numeric values by coercing to NaN, then filling with 0.
    df_working['FullTimeHomeGoals'] = pd.to_numeric(df_working['FullTimeHomeGoals'], errors='coerce').fillna(0)
    df_working['FullTimeAwayGoals'] = pd.to_numeric(df_working['FullTimeAwayGoals'], errors='coerce').fillna(0)

    # Iterate through each match to calculate and aggregate points for home and away teams.
    for _, row in df_working.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FullTimeHomeGoals']
        away_goals = row['FullTimeAwayGoals']

        if home_goals > away_goals:
            team_points[home_team] += 3  # Home team wins
        elif away_goals > home_goals:
            team_points[away_team] += 3  # Away team wins
        else:  # Draw
            team_points[home_team] += 1
            team_points[away_team] += 1

    # Print the top 10 teams by total points for quick inspection.
    sorted_teams_by_points = sorted(team_points.items(), key=lambda item: item[1], reverse=True)
    print("Top 10 Teams by Total Points:")
    for team, points in sorted_teams_by_points[:10]:
        print(f"  {team}: {points} points")

    return dict(team_points)


def normalize_val(val: float, min_val: float, max_val: float, target_range: Tuple[float, float] = (0.0, 1.0)) -> float:
    """
    Normalizes a numerical value from its original range [min_val, max_val]
    to a specified target range, typically [0, 1], using Min-Max scaling.

    The formula for Min-Max scaling to a target range [A, B] is:
    Normalized_Value = A + (Value - min_val) * (B - A) / (max_val - min_val)

    Args:
        val (float): The input numerical value to be normalized.
        min_val (float): The minimum value of the original range.
        max_val (float): The maximum value of the original range.
        target_range (Tuple[float, float]): A tuple specifying the desired output range (min, max).
                                            Defaults to (0.0, 1.0).

    Returns:
        float: The normalized value within the specified target range.
               If `max_val` is equal to `min_val` (indicating no variance in the original range),
               the function returns the midpoint of the `target_range` to avoid division by zero.
    """
    if max_val == min_val:
        # Avoid division by zero when there is no variance in the data.
        # In this case, the normalized value is set to the midpoint of the target range.
        return (target_range[0] + target_range[1]) / 2.0

    # Scale the value to the [0, 1] range first
    normalized_0_1 = (val - min_val) / (max_val - min_val)

    # Then scale the [0, 1] value to the desired target range
    return normalized_0_1 * (target_range[1] - target_range[0]) + target_range[0]


def normalize_diff_symmetric(val: float, max_abs_val: float) -> float:
    """
    Normalizes a difference value symmetrically to a range of [-1, 1].
    This normalization is based on the maximum observed absolute difference, ensuring
    that a zero difference maps to a normalized value of 0, and the maximum
    positive/negative absolute differences map to +1 and -1, respectively.

    The formula applied is: Normalized_Difference = Value / max_abs_val.

    Args:
        val (float): The difference value to be normalized (can be positive, negative, or zero).
        max_abs_val (float): The maximum absolute difference observed across the dataset
                             for the specific metric. This value should be non-negative.

    Returns:
        float: The symmetrically normalized difference value, ranging from -1.0 to 1.0.
               Returns 0.0 if `max_abs_val` is 0, indicating no variation in differences.
    """
    if max_abs_val == 0:
        # If the maximum absolute difference is zero, it implies all differences are zero.
        # In this case, the normalized difference is also 0.0 to avoid division by zero.
        return 0.0

    return val / max_abs_val
