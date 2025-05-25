from collections import defaultdict

import pandas as pd


def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Dataset shape: {df.shape}")
        # Convert MatchDate to datetime if not already
        if 'MatchDate' in df.columns:
            df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def calculate_and_normalize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the "for" and "against" totals for various metrics (goals, shot_accuracy, aggressiveness, control)
    for every team across the provided DataFrame. Normalizes these totals.

    Args:
        df (pd.DataFrame): The DataFrame containing match data (can be one season or multiple seasons).

    Returns:
        pd.DataFrame: A DataFrame containing for each team:
                      - Only the requested normalized "for" metrics (e.g., 'goals_scored_norm')
                      - Normalized total points ('points_norm')
                      Returns None if the input DataFrame is empty or essential columns are missing.
    """
    print("\n--- Calculating and Normalizing Team Totals (For/Against) across the provided data ---")

    if df.empty:
        print("Input DataFrame is empty. Cannot calculate totals.")
        return None

    # Ensure required columns are numeric and handle NaNs
    cols_to_process = [
        'FullTimeHomeGoals', 'FullTimeAwayGoals', 'HomeShots', 'AwayShots',
        'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards'
    ]

    # We work on a copy to ensure original DataFrame is not modified and to avoid SettingWithCopyWarning
    df_working = df.copy()
    for col in cols_to_process:
        if col not in df_working.columns:
            df_working[col] = 0  # Add the column with zeros if missing
        else:
            df_working[col] = pd.to_numeric(df_working[col], errors='coerce').fillna(0)

    team_stats = defaultdict(lambda: {
        'goals_for': 0, 'goals_against': 0,  # Manteniamo for/against per i calcoli intermedi
        'sot_for': 0, 'sot_against': 0,
        'total_shots_for': 0, 'total_shots_against': 0,
        'fouls_for': 0, 'fouls_against': 0,
        'yc_for': 0, 'yc_against': 0,
        'rc_for': 0, 'rc_against': 0,
        'corners_for': 0, 'corners_against': 0,
        'points': 0  # Manteniamo i punti per la normalizzazione
    })

    for _, match in df_working.iterrows():
        ht = match['HomeTeam']
        at = match['AwayTeam']

        # Update stats for HomeTeam
        team_stats[ht]['goals_for'] += match.get('FullTimeHomeGoals', 0)
        team_stats[ht]['goals_against'] += match.get('FullTimeAwayGoals', 0)
        team_stats[ht]['sot_for'] += match.get('HomeShotsOnTarget', 0)
        team_stats[ht]['sot_against'] += match.get('AwayShotsOnTarget', 0)
        team_stats[ht]['total_shots_for'] += match.get('HomeShots', 0)
        team_stats[ht]['total_shots_against'] += match.get('AwayShots', 0)
        team_stats[ht]['fouls_for'] += match.get('HomeFouls', 0)
        team_stats[ht]['fouls_against'] += match.get('AwayFouls', 0)
        team_stats[ht]['yc_for'] += match.get('HomeYellowCards', 0)
        team_stats[ht]['yc_against'] += match.get('AwayYellowCards', 0)
        team_stats[ht]['rc_for'] += match.get('HomeRedCards', 0)
        team_stats[ht]['rc_against'] += match.get('AwayRedCards', 0)
        team_stats[ht]['corners_for'] += match.get('HomeCorners', 0)
        team_stats[ht]['corners_against'] += match.get('AwayCorners', 0)

        # Update stats for AwayTeam
        team_stats[at]['goals_for'] += match.get('FullTimeAwayGoals', 0)
        team_stats[at]['goals_against'] += match.get('FullTimeHomeGoals', 0)
        team_stats[at]['sot_for'] += match.get('AwayShotsOnTarget', 0)
        team_stats[at]['sot_against'] += match.get('HomeShotsOnTarget', 0)
        team_stats[at]['total_shots_for'] += match.get('AwayShots', 0)
        team_stats[at]['total_shots_against'] += match.get('HomeShots', 0)
        team_stats[at]['fouls_for'] += match.get('AwayFouls', 0)
        team_stats[at]['fouls_against'] += match.get('HomeFouls', 0)
        team_stats[at]['yc_for'] += match.get('AwayYellowCards', 0)
        team_stats[at]['yc_against'] += match.get('HomeYellowCards', 0)
        team_stats[at]['rc_for'] += match.get('AwayRedCards', 0)
        team_stats[at]['rc_against'] += match.get('HomeRedCards', 0)
        team_stats[at]['corners_for'] += match.get('AwayCorners', 0)
        team_stats[at]['corners_against'] += match.get('HomeCorners', 0)

        # Update points
        if match.get('FullTimeHomeGoals', 0) > match.get('FullTimeAwayGoals', 0):
            team_stats[ht]['points'] += 3
        elif match.get('FullTimeAwayGoals', 0) > match.get('FullTimeHomeGoals', 0):
            team_stats[at]['points'] += 3
        else:  # Draw
            team_stats[ht]['points'] += 1
            team_stats[at]['points'] += 1

    # Convert to DataFrame
    df_ratios = pd.DataFrame.from_dict(team_stats, orient='index').reset_index()
    df_ratios.rename(columns={'index': 'Team'}, inplace=True)

    # Calculate Aggressiveness (Fouls + YC + 3*RC)
    df_ratios['aggressiveness_for'] = df_ratios['fouls_for'] + df_ratios['yc_for'] + 3 * df_ratios['rc_for']
    df_ratios['aggressiveness_against'] = df_ratios['fouls_against'] + df_ratios['yc_against'] + 3 * df_ratios[
        'rc_against']

    # Calculate Shot Accuracy (SOT / Total Shots) - Handle division by zero
    df_ratios['shot_accuracy_for'] = df_ratios.apply(
        lambda x: (x['sot_for'] / x['total_shots_for']) if x['total_shots_for'] > 0 else 0, axis=1)
    df_ratios['shot_accuracy_against'] = df_ratios.apply(
        lambda x: (x['sot_against'] / x['total_shots_against']) if x['total_shots_against'] > 0 else 0, axis=1)

    # Calculate Control (Corners + Total Shots)
    df_ratios['control_for'] = df_ratios['corners_for'] + df_ratios['total_shots_for']
    df_ratios['control_against'] = df_ratios['corners_against'] + df_ratios['total_shots_against']

    # Min-Max Scaling for normalization of ALL relevant totals
    # Qui normalizziamo tutte le metriche 'for', 'against' e i 'points'
    metrics_to_normalize = [
        'goals_for', 'goals_against',  # Manteniamo questi per un calcolo completo, anche se non li ritorniamo
        'aggressiveness_for', 'aggressiveness_against',
        'shot_accuracy_for', 'shot_accuracy_against',
        'control_for', 'control_against',
        'points'
    ]

    for metric in metrics_to_normalize:
        col_name_norm = f'{metric}_norm' if metric != 'points' else 'points_norm'

        if df_ratios[metric].max() == df_ratios[metric].min():
            df_ratios[col_name_norm] = 0.5
        else:
            df_ratios[col_name_norm] = (df_ratios[metric] - df_ratios[metric].min()) / \
                                       (df_ratios[metric].max() - df_ratios[metric].min())

    print("Calculation and normalization complete.")

    # Rinominare le colonne normalizzate "for" come richiesto
    df_ratios.rename(columns={
        'goals_for_norm': 'goals_scored_norm',
        'aggressiveness_for_norm': 'aggressiveness_committed_norm',
        'shot_accuracy_for_norm': 'shot_accuracy_norm',
        'control_for_norm': 'control_norm'
    }, inplace=True)

    # IMPOSTA LA COLONNA 'Team' COME INDICE DEL DATAFRAME
    df_ratios.set_index('Team', inplace=True)

    # Restituisci SOLO le colonne richieste e i punti normalizzati
    columns_to_return_strictly = [
        'goals_scored_norm',
        'aggressiveness_committed_norm',
        'shot_accuracy_norm',
        'control_norm',
        'points_norm'
    ]

    # Assicurati che tutte le colonne richieste esistano prima di tentare di selezionarle
    final_columns = [col for col in columns_to_return_strictly if col in df_ratios.columns]

    return df_ratios[final_columns]


def calculate_team_points(df: pd.DataFrame) -> dict:
    """
    Calculates the total league points for each team across the provided DataFrame.
    This function aggregates points regardless of specific seasons, summing up
    points for all matches present in the input DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing match data (can be one season or multiple).

    Returns:
        dict: A dictionary where keys are team names and values are their total points.
              Returns an empty dict if the DataFrame is empty or required columns are missing.
    """
    print("\n--- Calculating Total Team Points across the provided data ---")

    if df.empty:
        print("Input DataFrame is empty. Cannot calculate points.")
        return {}

    team_points = defaultdict(int)

    required_point_cols = ['HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals']
    if not all(col in df.columns for col in required_point_cols):
        print(f"Error: Missing one or more required columns for point calculation ({required_point_cols}).")
        return {}

    # Ensure goal columns are numeric and fill NaNs.
    # Apply these conversions to a copy of the input df to avoid SettingWithCopyWarning.
    df_working = df.copy()
    df_working['FullTimeHomeGoals'] = pd.to_numeric(df_working['FullTimeHomeGoals'], errors='coerce').fillna(0)
    df_working['FullTimeAwayGoals'] = pd.to_numeric(df_working['FullTimeAwayGoals'], errors='coerce').fillna(0)

    for _, row in df_working.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FullTimeHomeGoals']
        away_goals = row['FullTimeAwayGoals']

        if home_goals > away_goals:
            team_points[home_team] += 3
        elif away_goals > home_goals:
            team_points[away_team] += 3
        else:  # Draw
            team_points[home_team] += 1
            team_points[away_team] += 1

    # No need to filter for 'active_teams_in_season' here.
    # Any team whose matches are in the input 'df' will have its points summed.

    sorted_teams_by_points = sorted(team_points.items(), key=lambda item: item[1], reverse=True)
    print("Top 10 Teams by Total Points:")
    for team, points in sorted_teams_by_points[:10]:
        print(f"  {team}: {points} points")

    return team_points


def normalize_val(val, min_val, max_val, target_range=(0, 1)):
    """Normalizes a value using Min-Max scaling to a target range."""
    if max_val == min_val:
        # Avoid division by zero, return midpoint if no variance
        return (target_range[0] + target_range[1]) / 2

    normalized_0_1 = (val - min_val) / (max_val - min_val)
    return normalized_0_1 * (target_range[1] - target_range[0]) + target_range[0]


def normalize_diff_symmetric(val, max_abs_val):
    """
    Normalizes a difference value symmetrically to a range of (-1, 1)
    based on the maximum absolute difference observed.
    Ensures that 0 difference maps to 0 normalized, and +/- max_abs_val maps to +/- 1.
    """
    if max_abs_val == 0:
        return 0.0  # No variance, so difference is always 0
    return val / max_abs_val