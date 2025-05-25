# Premier League Complex Network Analysis

-----

## ⚽ Unraveling the EPL's Hidden Dynamics with Network Science

The Premier League isn't just about goals and points; it's a complex web of interactions, rivalries, and underlying tactical battles. This project delves into the rich dataset of English Premier League matches to construct and analyze it as a **complex network**. By applying advanced network science methodologies—including centrality measures, community detection, and correlation analysis—we aim to uncover the unseen structural relationships that define team performance, strategic interactions, and league dynamics.

-----

## 🔍 Key Features

  * **Dynamic Network Creation**: Build a directed graph representing inter-team relationships (e.g., goals scored/conceded, aggression, control) based on match data over customisable seasons or specific campaigns.
  * **Comprehensive Centrality Analysis**: Calculate and interpret various centrality measures (In-Degree, Out-Degree, In-Strength, Out-Strength, Betweenness, Closeness, Eigenvector) to identify the most influential, central, or pivotal teams across different metrics.
  * **Correlation with League Performance**: Quantify the statistical relationship between a team's network centrality (across various metrics) and its final league points, revealing which aspects of team interaction are most tied to success.
  * **Community Detection**: Employ the Louvain algorithm to uncover natural groupings (communities) of teams based on their shared interaction patterns (e.g., teams that engage in similar levels of aggression or control when playing each other).
  * **Metric-Specific Filtering**: Apply dynamic thresholds to filter network edges, focusing analysis on significant interactions for specific metrics (e.g., only highly aggressive encounters, or matches with similar control levels).
  * **Visualizations**: (If applicable, describe your plotting capabilities) Generate insightful plots to visualize network structures, centrality distributions, and correlations, making complex data easily digestible.

-----

## 🛠️ Technologies & Libraries

  * **Python**: The core programming language for data processing and network analysis.
  * **Pandas**: Essential for data manipulation and preparation of match datasets.
  * **NetworkX**: The powerful library for creating, manipulating, and studying the structure, dynamics, and functions of complex networks.
  * **`python-louvain`**: For efficient and effective community detection using the Louvain algorithm.
  * **SciPy**: For statistical computations, particularly Pearson correlation.
  * **Matplotlib / Seaborn**: (If applicable) For creating compelling data visualizations.

-----

## 🚀 Getting Started

To explore the Premier League as a complex network, follow these steps:

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/marcellorussox/premier-league-complex-net.git
    cd premier-league-complex-net
    ```

2.  **Set Up Your Environment**:
    It's highly recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Your Data**:
    This project expects a Premier League match dataset.
    *(**Guidance**: Briefly explain where the user can get data or if it's included. E.g., "Place your `EPL_Match_Data.csv` file (or similar) in the `data/` directory. A sample dataset might be available [here](https://www.google.com/search?q=link_to_sample_data).")*

5.  **Run the Analysis**:
    The core logic resides in `main.py` (or your equivalent primary script). You can run it directly:

    ```bash
    python main.py
    ```

-----

## 📊 Project Structure

```
premier-league-complex-net/
├── data/
│   └── EPL_Match_Data.csv      # Placeholder for your raw data
├── src/
│   ├── __init__.py
│   ├── network_creation.py     # Functions to create the NetworkX graph
│   ├── centrality_analysis.py  # Functions for centrality calculations
│   ├── community_detection.py  # Functions for community detection
│   ├── utils.py                # Helper functions (e.g., data preprocessing, plotting)
│   └── main.py                 # Orchestrates the analysis workflow
├── notebooks/                  # (Optional) Jupyter notebooks for exploration/demonstration
│   └── exploration.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

-----

## 💡 How It Works (Under the Hood)

1.  **Data Ingestion & Preprocessing**: Raw match data (e.g., team names, goals, fouls, possession) is loaded and cleaned. Metrics like 'aggressiveness' (`fouls + yellow cards + red cards`), 'control' (`possession`), and 'shot accuracy' are calculated.
2.  **Network Construction**: A **directed graph (`networkx.DiGraph`)** is built where nodes represent Premier League teams. Edges `A -> B` represent interactions aggregated over matches where Team A played against Team B. Edge attributes store calculated metrics (e.g., `goals_for_A_vs_B`, `aggressiveness_A_vs_B`). Crucially, normalized difference metrics (e.g., `aggressiveness_diff_norm_abs`) are computed to quantify the *similarity* or *dissimilarity* of teams' performance across a specific metric when they interact.
3.  **Filtering for Analysis**: For each specific metric (e.g., 'aggressiveness'), a **filtered directed graph** is created. Only edges where the `_diff_norm_abs` for that metric falls within a specified threshold are retained. This ensures that centrality and community analysis focuses on meaningful and comparable interactions.
4.  **Centrality Measures**: Various centrality algorithms are applied to the *filtered directed graph* to understand team importance based on different network perspectives.
5.  **Community Detection**: The *filtered directed graph* is then converted into an **undirected graph**. For community detection, edge weights are often transformed to represent *similarity* (e.g., `1 - _diff_norm_abs`), where higher values indicate stronger ties, encouraging Louvain to group highly similar teams together.
6.  **Correlation Analysis**: The calculated centrality scores for each team are correlated with their total league points for the season. This provides quantitative insights into which network positions or interaction styles contribute most to league success.

-----

## 📈 Insights & Potential Applications

This project provides a robust framework to:

  * Identify **unexpectedly central teams** based on specific interaction styles (e.g., a mid-table team highly central in 'aggressiveness').
  * Uncover **hidden communities** of teams that share similar tactical approaches or competitive dynamics.
  * Understand which aspects of a team's **interaction profile (e.g., high control, high aggression)** are statistically linked to better league performance.
  * Serve as a foundation for **scouting insights**, **tactical analysis**, or even **predictive modeling** by providing a deeper, structural understanding of league dynamics beyond traditional statistics.

-----

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star\! Thanks\!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

-----