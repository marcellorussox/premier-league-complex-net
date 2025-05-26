## Multi-Season Analysis (2012/13 - 2017/18): Overall Network Properties

The initial network for this 6-season span (`network_start_year=2012`, `network_end_year=2018`) comprises 30 unique nodes (teams) and 762 edges. The high network density (0.8759) indicates that nearly all teams have direct connections, reflecting the comprehensive nature of the `perform_epl_network_analysis` function and the large number of potential team pairings over six seasons. The top 10 teams by total points unsurprisingly feature the "Big Six" (Man City, Chelsea, Man United, Tottenham, Arsenal, Liverpool) consistently at the top, followed by strong mid-table clubs like Everton and Southampton. This overall structure provides the context for our similarity analyses.

### Analysis for Metric: 'goals' (Normalized Threshold $\le$ 0.08)

The filtering for 'goals' similarity (normalized difference $\le$ 0.08) results in 162 edges, indicating a moderate level of sustained goal-scoring similarity across this six-year period.

**Centrality Measures:**
The top 5 teams in all centrality measures (Degree, Strength, Betweenness, Closeness, Eigenvector) are consistently **West Brom, Stoke, Crystal Palace, Middlesbrough, and Reading (or similar lower-mid table teams like Burnley, Wigan, Brighton, QPR, Norwich).** This is a significant finding. It suggests that teams with *less extreme* goal-scoring outputs (i.e., not consistently at the very top or very bottom of the scoring charts) act as central connectors in the similarity network over the long term. Their goal tallies are 'average' enough to be similar to a wider range of other teams. The strong negative correlation between centrality measures (especially Degree and Strength) and league points further supports this; teams that are "central" in terms of similar goal-scoring are generally *not* the highest-performing teams.

**Community Detection:**
The Louvain algorithm identifies four communities based on long-term goal similarity:
* **Community 0: Arsenal, Newcastle, Man United, Tottenham, Brighton, Huddersfield.** This community is intriguing. It groups several "Big Six" clubs (Arsenal, Man United, Tottenham) with teams that have had varying performances over this period (Newcastle, Brighton, Huddersfield). This suggests that over six seasons, the goal-scoring profiles of these top clubs, while generally high, were similar enough to each other and to some other teams that maintained a consistent, perhaps mid-to-high, goal tally.
* **Community 1: West Ham, Norwich, Sunderland, Leicester, Bournemouth, Watford, Middlesbrough.** This community contains teams that were either consistently mid-to-lower table or, in Leicester's case, had a highly variable performance (champions in one season, then lower). This group seems to represent a broad "mid-range" of goal-scoring similarity.
* **Community 2: QPR, Reading, West Brom, Chelsea, Aston Villa, Liverpool, Crystal Palace, Hull, Burnley.** This is a highly diverse group, notably including **Chelsea and Liverpool** alongside many teams that experienced relegation or struggled in certain seasons. This suggests that even elite clubs can, over a six-year period, show goal-scoring patterns similar to teams with much different trajectories, possibly due to periods of rebuilding, underperformance, or tactical shifts that momentarily align their goal output.
* **Community 3: Fulham, Man City, Wigan, Everton, Southampton, Swansea, Stoke, Cardiff.** This community, while containing **Manchester City** (the top-scoring team over this period), also includes teams like Wigan, Fulham, and Cardiff, which were often lower-table or relegated. This indicates that while Man City scored many goals, their *variance or specific scoring profile* over six seasons could be similar to teams with more fluctuating outputs.

### Analysis for Metric: 'aggressiveness' (Normalized Threshold $\le$ 0.15)

The aggressiveness similarity network has 203 edges, making it slightly denser than the goals network. This implies a higher degree of long-term similarity in aggressive playing styles across the league.

**Centrality Measures:**
**Arsenal** is surprisingly prominent, appearing in the top 5 for Degree, Strength, Betweenness, Closeness, and Eigenvector Centrality. This is unexpected for a team often associated with a "beautiful football" style. It suggests that, over the long run, Arsenal's aggressiveness level was consistently similar to a wide range of other teams. Other central teams include Hull, Cardiff, Wigan, and Middlesbrough – generally more combative sides. The weak correlation with league points indicates that sharing a similar aggression profile doesn't strongly predict success.

**Community Detection:**
Let's specifically consider **derby rivalries** in the context of aggressiveness communities:

* **Community 0: West Ham, Aston Villa, Stoke, Hull, Bournemouth.** This community appears to group teams that often play a direct and combative style.
* **Community 1: Arsenal, Fulham, Reading, Wigan, Everton, Chelsea, Norwich, Swansea, Cardiff.**
    * **London Derbies:** **Arsenal** and **Chelsea** are in the same community. This suggests that over this six-year period, despite their rivalry and different league positions, their **absolute levels of aggressiveness were broadly similar**. This is an interesting finding, as one might expect fierce rivals to deliberately differentiate their aggressive approaches.
    * **Merseyside Derby:** **Everton** is in this community. (Liverpool is in Community 2). This means the Merseyside rivals had *dissimilar* long-term aggressiveness profiles, which could be a significant tactical differentiator in their fierce derby matches.
* **Community 2: Newcastle, West Brom, Man City, Man United, Liverpool, Watford, Brighton, Huddersfield.**
    * **Manchester Derby:** **Man City** and **Man United** are grouped together. This is highly relevant! It implies that over these six seasons, the two Manchester giants, despite their intense rivalry and differing tactical philosophies, maintained **similar long-term aggressiveness levels**. This might point to a consistent underlying competitive spirit or a shared approach to physicality in the Premier League.
    * **Liverpool** is also in this community.
* **Community 3: QPR, Southampton, Tottenham, Sunderland, Crystal Palace, Leicester, Burnley, Middlesbrough.**
    * **London Derbies:** **Tottenham** is in this community. Given Arsenal (Community 1) and Chelsea (Community 1) are elsewhere, this means Tottenham's aggressiveness profile was distinct from both of its major London rivals, which is an interesting finding for North London and West London derbies.

This analysis of aggressiveness communities relative to derbies provides specific insights into how long-term stylistic similarity might (or might not) align with rivalry.

### Analysis for Metric: 'control' (Normalized Threshold $\le$ 0.13)

The 'control' (possession) similarity network has 182 edges, indicating a moderate level of sustained similarity in possession-based playing styles.

**Centrality Measures:**
**West Brom** and **Wigan** are highly central (high Degree and Strength). This mirrors the 'goals' analysis where mid-to-lower table teams often act as central connectors because their "average" stats are similar to a wider range of teams. **Huddersfield** and **Stoke** also show high betweenness and closeness centrality, suggesting their control profiles were instrumental in linking disparate groups. The strong negative correlation between control centrality and league points (-0.7838 for Degree centrality) implies that teams that are central in terms of similar possession stats are generally *not* the top-performing teams. This is a very powerful finding: **the most successful teams tend to have more distinctive (and likely higher) possession profiles, making them less "similar" to the average team.**

**Community Detection:**
* **Community 0: Arsenal, Man City, Wigan, Chelsea, Man United, Tottenham, Liverpool.** This is a very strong and intuitive community! It perfectly groups the **"Big Six" clubs** (Arsenal, Man City, Chelsea, Man United, Tottenham, Liverpool) that are consistently known for their emphasis on ball control and possession-based football, regardless of their specific tactical nuances. The inclusion of **Wigan**, a team that typically played a more direct style, is an outlier that warrants further investigation, possibly due to a specific season's anomaly in their possession statistics or shared periods of adapting their style. This community strongly confirms that over the long term, elite clubs share a fundamental similarity in their approach to controlling the game through possession.
* **Community 1:** A large cluster primarily composed of teams that were often in the lower half of the table or experienced relegation (Fulham, QPR, Reading, West Brom, West Ham, Aston Villa, Norwich, Stoke, Sunderland, Crystal Palace, Cardiff). This community likely represents teams that generally had **lower or more variable possession statistics**, making them similar to each other.
* **Community 2:** Another large community encompassing remaining mid-table and promoted teams (Newcastle, Everton, Southampton, Swansea, Hull, Leicester, Burnley, Bournemouth, Watford, Middlesbrough, Brighton, Huddersfield).

---

## Overall Academic Interpretation

This multi-season network analysis, leveraging normalized thresholds for comparability across different years, provides a robust framework for understanding long-term team dynamics in the Premier League.

1.  **"Average" as Central:** For both 'goals' and 'control', teams with non-extreme, more "average" statistical profiles (e.g., West Brom, Stoke for goals; West Brom, Wigan for control) tend to be more central in the similarity networks. This is because their metrics are closer to a wider range of other teams, acting as stylistic "bridges." Conversely, the strong negative correlations with league points suggest that **elite performance often correlates with a more distinctive (and thus less "similar" to the average) statistical profile**.

2.  **Aggressiveness: Shared Competitive Spirit:** The relatively high density of the aggressiveness network suggests that a common level of aggression is pervasive across the league over time. The co-occurrence of rival teams (e.g., Manchester City and Manchester United) in the same aggressiveness community indicates that intense competition might lead to **convergent aggressive styles** among top-tier rivals, despite differing tactical philosophies. The divergence of other rivals (e.g., Everton and Liverpool; Arsenal/Chelsea vs. Tottenham) could highlight areas where tactical or philosophical distinctions are more pronounced.

3.  **Control: Elite Distinctiveness:** The 'control' analysis strikingly confirms that the **"Big Six" exhibit a strong, long-term similarity in their possession-based approach**, clearly differentiating them from other clusters of teams. This reinforces the idea of a shared tactical philosophy among the top clubs that prioritizes ball control. The relative sparsity of the 'control' network further suggests that for many teams, their possession styles are not within narrow similarity bands, leading to more distinct groupings or even isolation (e.g., Wigan in Community 0 warrants further investigation).