# NFL Betting Market Prediction and Analysis

Sports betting is a multi-billion dollar per year industry.  Estimates vary, but the Nevada Gaming Commission reported over $3.2B in legal gambling in 2011, with 41% of that money being wagered on football alone. <sup id="a1">[__1__](#fn1)</sup>  According to NBA commissioner Adam Silver, if illegal gambling is counted the annual total for American gambling jumps to $400B. <sup id="a1">[__2__](#fn2)</sup>   The single biggest betting event in America is the Super Bowl, which now attracts over $100M in legal bets and potentially ten times that amount illegally. <sup id="a1">[__3__](#fn3)</sup>  With such sums of money flowing into the betting market, I was curious to see if I could isolate some -- _any_ -- notable inefficiency which would give a leg up on betting intelligently.  

The oddsmakers in Vegas use networks of supercomputers to set the odds, so expecting to beat them outright with a single machine learning model is a bad bet, no pun intended.  However, my hope was that systemic inefficiencies could be ferreted out as Vegas must also shift the odds based on __how the public bets__ in order to prevent a catastrophic loss of house money should an upset occur, granting the one-sided public bets all large payouts.  This opens the door for the informed bettor (us) to profit by the uninformed bettors (the public) making bad choices and forcing Vegas to alter the odds in an inefficient manner. In the interest of full disclosure, I have never bet a penny in my life (not even on slot machines -- sorry, casinos).  

<BR>

## Table of Contents
1. [Dataset](#dataset)
    + [Acquisition and Error Correction](#acquisition-and-error-correction) **tell # of PFR sheets**
    + [Feature Engineering](#feature-engineering)
    + [Advanced Metrics Limited by Years](#advanced-metrics-limited-by-years)
2. [NFL Betting Primer](#nfl-betting-primer)  
    + [Interpreting a Bet's Payout](#interpreting-a-bet's-payout)
    + [The Spread](#the-spread)
    + [The Over/Under](#the-over/under)
    + [The Money Line](#the-money-line)  
3. [Wise Bets](#wise-bets)  
4. [Model Selection](#model-selection)   
    + [Data Selection](#data-selection)
    + [Classification vs. Regression](#regression-vs-classification)
    + [Model Selection and Training](#model-selection-and-training)
      + [Tree-based Feature Importance](#feature-separation-and-importance)
5. [Results](#results)  **Try Seaborn RegPlot and lmplot with some targets/wise/proba/etc**
    + Overview -- Best and Worst
    + PCA _done_
    + t-SNE _done_
    + Feature Overlaps _done_
    + Spread             **Plot KDE/Curve of Spread target**
      + Features
      + Metrics
    + Over/Under         **Plot KDE/Curve of O/U target**  
      + Features        
      + Metrics
    + Money Line  **ROC**      **CMAT**
      + Features
      + Metrics
    + Weather **TEMP/wind/wc plots**
    + Wise Bets Actual Results
      + Spread
      + Over/Under
      + Money Line
        + 91.5% R2 with Spread when predicting proba...
    + Hypothetical Bettor Using This Model
      + Money Line
    + Clusters - Four Types
6. [Future Considerations](#future-considerations)  
    + Dynamic Web App
    + Player-specific Information
      + Injuries
      + Current-season Performance
    + Miles Traveled

<BR><BR>

## Dataset
The point of this entire project was to use team-level data to identify trends in how Vegas oddsmakers set the odds for a given game.  However, as this model aims to predict single game results, it requires the stats and information for each of the two teams in a given game up to that point in the given season.  This meant I needed to procure game-by-game detailed information for every statistic, and not merely season-long summary information.  The types of statistics and information I wanted to model included all in-game statistics, weather, stadium information, and advanced analytics.

Sports analytics has grown from a small cottage industry in the mid-1980s to a robust field unto itself in 2017.  My aim was to leverage as many 'advanced' as possible metrics to improve my model's accuracy.  Some of these metrics are proprietary and available only through subscriptions to their respective stat-owning websites, such as the _Defense-adjusted Value Over Average_ (DVOA) metrics from [Football Outsiders](www.footballoutsiders.com/) or _Clutch-weighted Quarterback Rating_ (QBR) and Brian Burke's _Football Power Index_ (FPI) from [ESPN Insider](www.espn.com/insider/).  Other metrics, such as _Pythagenport Win Expectancy_ -- a mildly revised descendent of baseball Sabremetrics godfather Bill James' famous _Pythagorean Win Expectancy_ metric -- or _Adjusted Net Yards Per Attempt_ (ANY/A), most recently modified by Chase Stuart, must be calculated.  Another excellent team-level advanced metric is _Expected Points Added_ (EPA), which originates from the seminal "Hidden Game of Football" published in the late 1980s by Bob Carroll and Pete Palmer.   

### Acquisition and Error Correction
Unfortunately, no single source exists which has all these statistics.  In an effort to use as many of these stats as possible I decided to scrape the desired single-game statistics from [Pro Football Reference](www.pro-football-reference.com) (PFR) using BeautifulSoup and urllib.  PFR is known as the online encyclopedia for all things pro football, and has detailed information for nearly each game played in pro football history, including stadium type, time of game, weather, and Vegas betting information.  Regarding scraping of their site, PFR makes the following pro-scraping statement on their [data use](http://www.sports-reference.com/data_use.html) page:
>We will not fulfill any requests for data for custom downloads, unless you are prepared to pay a minimum of $1,000 for any such request.
>
>We realize this will be insurmountable for any student requests. However, I would point out that learning how to accumulate data is often a more valuable skill than actually analyzing the data, so we encourage you as a student or professional to learn how.

In total, I obtained 181, 285 separate tables.  Around 30,000 of these were used in this project.  In order to utilize them, I first had to do modify them for uniformity and then create a table for each team that summarized their progress through a given season, game by game, with around 300 added features that covered both single-game and running total statistics.  Once completed, the final database was formed by stitching the statistics for each home team and road team together for every game from 1950-2016 into a single entry. For games that have Vegas-related information, which starts in 1978, this totals around 12,500 games.

Surprisingly, most of PFR's data was well-maintained.  There were, however, two notable errors.  First, _time of possession_ data for all post-season games from 1991-1998 was missing.  I looked up each of these games and manually entered the correct data.  Second, 87 games had missing weather data (temperature, wind chill, wind speed (mph), and humidity), which forced me to manually look up the weather in the city the game was played in on the date of the game, and insert into the database one-by-one.  (_Surprise_, it's hot and dry in Arizona).  

A total of 1817 games were played in a closed-roof, climate-controlled dome, starting in 1968.  The temperature was set to 67° F, no wind, and no humidity for these games.  Wind chill was also calculated for each game with a temperature below 50° F using the modern formula of 35.74 \+ (0.6215 \* Temp) - (35.75 \* Wind<sup>0.16</sup>) \+ (0.4275 \* Temp \* Wind<sup>0.16</sup>). <sup>[__4__](#fn4)</sup>

Last, as a long-time paying member at Football Outsiders, I was able to obtain all the DVOA data in their databases, which runs back to 1991 as of February 2017.

### Feature Engineering
I engineered features in seven distinct ways.
1. __Per-game averages__ for each team in every statistic.  This was necessary for two concrete reasons.
    1. Not every season in NFL history has had the same number of games.
    2. Within a season, teams regularly play an opponent that has not played the same number of games at they have at that point in the season.  Simply using season-long running-total values would skew things in favor of the team that had played more games for positive stats and in favor of the team that had played fewer games for negative stats.   By converting every statistic to a per-game value, we are comparing apples-to-apples.
2. __Deltas__ between the two teams in given statistics, resulting in a single statistic describing the relationship between the teams for the given game.   
    _Example:_ home team averages +103 more yards passing per game than the road team.
3. __Aggregation__ of similar statistics into a single metric.  
    _Example:_ find how many standard deviations a team is above or below league average in a given stat for a given week, and accumulate these for all relevant offensive or defensive stats into a single statistic, such as "Offensive Sigma".
4. __Binning__ of a given statistic into quintiles.
5. __Clustering__ of advanced metrics in an effort to identify "types" of game matchups.
6. __Dummying__ of categorical data, such as day of the week per game, type of stadium (dome or open), type of playing surface, and week of the season.
7. Calculating exact hours of rest between games, not merely days.  

### Advanced Metrics Limited by Years
Due to the proprietary nature of some of the advanced metrics, and the reliance upon more granular statistics for others, they are not available for the entirety of the dataset itself.  Vegas line information only goes back to the 1978 season, meaning 1978 is the earliest possible season for this model.  Time of possession information starts in 1983, 3rd and 4th Down success rates as well as DVOA begin in 1991, and EPA starts in 1994.  In the future as more old game logs and old game films are parsed and logged, it will be possible for these insightful advanced metrics to be extended further back into league history.  The selection of data from this parent dataset will be discussed below under [Model Selection](#model---data-selection).

<BR>


## NFL Betting Primer
There are three primary types of wagers made on NFL games:
1. The spread
2. The over/under
3. The money line  

#### Interpreting A Bet's Payout
When a money line or spread is negative for a given team, this means that team is favored to win.  As such, the payout for that bet is less favorable than for the underdog.  All odds are given relative to a wager of $100.  An example is easiest to demonstrate.  If Team A is the favorite and has odds (money line) of -200, this means you must bet $200 to win (net) $100 (the $200 you originally bet plus $100 in winnings).  Since Team A is this makes sense -- you must risk more money in order to profit since they're expected to win.  Conversely, if Team B is an underdog and has odds of +300, you will win (net) $300 with a wager of only $100.  Again, Team B is not expected to win, so to entice bettors to take the bet, the reward must be greater than the risk.

#### The Spread
The spread, also called the "line", is a measure of how much better Vegas thinks Team A is than Team B.  Vegas sets the spread in the amount of points the favored team is expected to win by.  A negative spread indicates a team is favored, positive an underdog.  For example, a spread of -3.0 means the favored team is expected to win by a field goal (3 points).  You can bet on either team, the favorite or underdog.  In order to win a bet on the spread, your team must exceed the spread in your favor.  So, if you bet on the favorite at -3.0, they must win by _more_ than 3 points for your bet to win.  If they win by exactly 3 points, the result is called a "push", and all money is returned to bettors, none having been won nor lost.  

Even moderate sports fans are doubtless familiar with the notion of "home field advantage," and we see it borne out in the history of the Vegas NFL spread.  The peaks in the distribution represent the most common increments of scoring in football: 3 points, 7 points, 10 points, and 13 points.  Note the aversion to setting the line at 0 points, as this is equivalent to simply picking the winner outright.  Also note the significant majority of lines are set favoring the home team, offering real evidence of the notion of "home field advantage."

<img src="images/road_spread_dist.png" alt="History of the Spread">  

<sub>__Figure 1:__ The historical distribution of the Vegas spread for NFL games from the perspective of the visiting team.  Excluding the intentional dip at 0 points, the spread conforms to a roughly normal distribution. </sub>

<BR>
<img src="images/over-under_dist.png" width="600" align="right" alt="History of the Over/Under">  

#### The Over/Under
The Over/Under is simply the total expected number of points scored by both teams in a game.  You can bet the Over or the Under, and will win if the combined score of the teams is either more than (over) or less than (under) the set Over/Under value, depending on your wager.  If the final combined score equals the Over/Under value exactly, the bet ends in a "push" and all money is returned.  

<BR><BR><BR><BR>
<div align="right">
<sub><b>Figure 2:</b> The historical distribution of the Vegas Over/Under for NFL games. The mean is denoted
<BR>
by the small dashed line at 42.2 points. Again, we observe here a gaussian distribution. </sub>
</div>

<BR>
<img src="images/money-line_dist.png" width="600" align="right" alt="History of the Money Line">  

#### The Money Line
The money line is simply the odds that a specific team will win the game, regardless of margin of victory (spread).  The money line is given in odds like the spread, where negative implies the favored team, and the odds themselves indicate what the [payout](#interpreting-a-bet's-payout) will be for a winning bet.  Again owing to the notion of "home field advantage", the average money line for a road team when they're favored is -230, while the average for a home favorite is -313.  

<BR><BR><BR><BR>
<div align="right">
<sub><b>Figure 3:</b> The historical distribution of the Vegas Money Line for NFL games. Split much like
<BR>
the spread, the money line shows the higher value and density of home favorites.
</sub>
</div>

All things considered, you must risk more money when betting on a home team as they're expected to win more frequently.  This trend holds true for underdogs as well; you win more money from the average road underdog (+248) than the average home underdog (+179).

<BR>




## Wise Bets
Games that pass a user-set threshold of deviation from the model's prediction, either in a point spread or in odds to win, are labeled as __wise bets__.

A game whose actual spread deviates from the predicted spread by the user-set point threshold or more will be labeled a "wise bet".  The underlying approach to finding mis-valued spreads works as follows.  The key factor for a spread is its _flexibility_. As Vegas receives more bets on a particular team at a given spread value, they can adjust the spread in order to balance the wagers on the opposing team, reducing the bookmakers' risk by taking near equal money on both sides.  (Vegas typically does not win big on any given game.  They win small amounts consistently by playing percentages very carefully).  

This flexibility in the line is the key component I aimed to use in snuffing out inefficiency in the spread.  If the betting public has a possibly inaccurate perception about a given team, they will either over- or under-bet for that team, forcing Vegas oddsmakers to compensate by artificially adjusting the spread in order to entice bettors to make wagers against their (inaccurate) perception and even out the money wagered.  

Because of this, the initial aim of this project was simple: I wanted __to identify which factors best predict games that have spreads that are incorrectly set, to label these games as potentially "wise bets," and to examine the results of these games in hopes of finding that a favorable percentage would be winning bets.__

Secondarily, we can do the same for the Over/Under as well as the Money Line.  The Money Line is slightly different, since it is concerned only with the binary outcome of win/lose.

<BR>

## Model Selection
Four models were tested in this project: two tree ensemble methods, Random Forests (RF) and Gradient Boosted Trees (GBT), as well as Support Vector Machines (SVM) and finally ElasticNet regression.  The GBT models are from __XGBoost__ while the rest are from __Sci-Kit Learn__.

### Data Selection
As mentioned above, there were three divisions of the original dataset features, as well as up to five progressively smaller year ranges of games to inspect.

#### Feature Separation and Importance
The feature-set division arises from my desire to answer the following question: _which single statistics are the most important in predicting X result for an NFL game?_  

The easiest and most direct way to do this is to use a model which has a feature importance attribute.  Gradient Boosted Trees do, and so served this role in this project.  There are two ways of finding the feature importance in ensemble tree models like Random Forests and Gradient Boosted Trees: first, each model has its own attribute which will tell you which features gave the highest return in purity or lowest return in error for the single fitting and run of the model.  Second, a more robust approach is to use Recursive Feature Elimination (RFE), which is calculated by fitting the model with all but one feature, measuring how well it predicts, and then repeating this for all features in the dataset until each one has had a turn being left out.  The features that caused the greatest drop in prediction accuracy are judged to be the most important.

However, if any two features are closely related, the importance for either one will be largely negated by the existence of the other.  For example, pretend we have the two following statistics in this database: _touchdowns in the first three quarters_ and _touchdowns for the entire game_.  When the the stat for the _entire game_ is removed and the model measures how accurate its predictions are, it will still have a feature included that provides three-quarters of the removed _entire game_ stat's information, and the prediction accuracy won't be severely impacted.  As a result, the _touchdowns for the entire game_ statistic will be reported as not being a very important feature.  However, the reality is that this would indeed be an important statistic, but having a correlated or, in this case, partially duplicated feature clouds our ability to determine its true importance.

Because of this fact, I split the feature set into three parts:
1. All statistics
2. Raw statistics
3. Matchup deltas

_Raw_ stats are simply every team's own per-game statistics coming into the given game of interest.  Examples include Touchdowns/game, 1st Downs/game, turnovers/game, etc.  Matchup deltas are the differences between the road and home teams in these respective stats.  So, the home team might average seven more first downs per game, but 1.3 more turnovers per game.  Since the matchup stats are technically derived from the raw statistics, I wanted to ensure proper evaluation of feature importances which led to their being optionally separated.  The top features for each target will be reported in the [Results](#results) section below.

Ideally, RFE allows you to trim the feature set of the model to only the most important variables in an effort to lower complexity and reduce variance.  This goal was not realistically achieved in this project, as any subset of N most important features (20, 40, 80) failed to match the accuracy of the full feature set, regardless of which subdivision (all, raw, matchup) of features was chosen.  

Regarding prediction, having more information tends to be better than having less, and we see that here as using _all_ the features did lead to the best prediction accuracy for any given model.  However, the difference between using all the data and only the matchup data was minor, commonly in the range of 0.5% to 1.5%.  The raw features alone were noticeably less predictive, sometimes giving up to 5% worse prediction accuracy.

#### Database Size
As noted above, the more advanced metrics do not extend back to the beginning of the recorded Vegas betting data.  In order to use data ranging back to 1978, the start of the Vegas data, we would have to exclude many statistics which begin in subsequent years.  So, this gives us realistically two options:
1. A 'longer' database that goes back to 1978, but is lacking any information-rich advanced metrics (called the _1978 database_).
    + 8,788 rows (games), 258 columns.  Total of 2,267,304 data points.
2. A 'wider' database that starts in 1994 but has all the info-rich metrics (called _Advanced database_ since all advanced metrics are included).
    + 5498 rows (games), 360 columns.  Total of 1,979,280 data points.  

Experiments with both databases showed the wider, Advanced database to give slightly better predictions in regression, up to 4% improved R<sup>2</sup> accuracy and a tenth of a point lower in MAE.  For predicting the winner of a game, this gap was near 0.5% in AUC but -1.3% in F1-score without SMOTE oversampling, and -3.8% in AUC and -4.7% in F1-score with SMOTE oversampling.  

This suggests that the information-rich advanced metrics more than make up for the loss of sample size when predicting the spread for a game but are either roughly equivalent to (without SMOTE), or do not compensate for the loss of (with SMOTE), fourteen extra years of data for predicting the winner outright.  Looking ahead, however, the advanced metrics will only continue to accrue and if they give better or roughly comparable predictions now with far fewer years of data, it would be sensible to use them going forward.

### Regression vs Classification
Since the Spread and the Over/Under are numeric, regression models were used to predict these targets.  Conversely, classification algorithms were used in modeling the Money Line binary winner/loser of a game.  

#### Regression
There are two possible regression targets: the _spread_ and the _over/under_ for each game. Each target is given in game points.  The spread extends back to 1978 while the over/under starts in 1979.

<BR>

Target | Mean | Minimum (abs) | Maximum (abs) | Range
-------|------|---------------|---------------|------
Spread | 2.6 | 0.0 | 26.5 | 26.5
Over/Under | 41.6 | 28.0 | 63.0 | 35.0
<sub>__Table 1:__ Summary statistics for the Vegas Spread and Over/Under, from 1978-2016.</sub>

#### Classification
Three classification targets are available:
1. Whether a team _covered the spread_ (won the spread bet),
2. Whether a game went _over or under_ the over/under point total, and simply
3. Whether a team _won or lost_ the game (the money line bet).  

The cover and over/under classes are very close to being ideally balanced, but the classes for a home team vs. a road team winning are slightly imbalanced.  Regardless of year range, the home team wins at roughly a 58% to 42% rate compared to the road team.  

This imbalance is not drastic, but does mean that stratification during train-test splitting to ensure both the train and the test splits receive an equal ratio of each class is wise.  Other class imbalance fixes attempted were using the cost-minimizing `'balanced'` class weighting in the Random Forest model, which made substantial difference in the efficacy of the model.  Alternatively, the oversampling SMOTE package by __imblearn__ was used for the GBC and SVC models.  Its impact was mixed, offering a few percentage points of improvement or negation for the Receiver Operating Characteristic Area Under the Curve (AUC).  In particular, the Gradient Boosted Tree classifier by __XGBoost__ handled the 58/42 class imbalance rather well out of the box.  The full breakdown of classes is shown below for reference.

##### Class Ratios:  
Target | Data | Majority Class | Minority Class | Majority % | Minority % | Counts
-------|------|----------------|----------------|------------|------------|-------
Road Cover | 1978 | Yes | No | 51.4% | 48.6% | 4514 - 4274
Road Cover | Advanced | Yes | No | 51.7% | 48.3% | 2843 - 2655
Over/Under Result | 1978 | Over | Under | 50.6% | 49.4% | 4449 - 4338
Over/Under Result | Advanced | Over | Under | 51.0% | 49.0% | 2805 - 2693
Home Team Win | 1978 | Win | Loss | 58.2% | 41.8% | 5114 - 3674
Home Team Win | Advanced | Win | Loss | 57.9% | 42.1% | 3183 - 2315
<sub>__Table 2:__ Breakdown of classification targets within the two primary datasets, 1978-2016 and 1994-2016 (_Advanced_).  Only the __Home Team Win__ target has a moderate class imbalance.  </sub>


### Model Selection and Training
In order to choose the models which performed best I optimized for the mean absolute error (MAE).  Compared to the root mean squared error (RMSE), the MAE is consistent across ranges of errors and doesn't 'flare' up in response to larger residuals.  For evaluating how many points a predicted NFL game's spread is  from the actual spread, there is no harsher penalty for being five points away than there is for being four points away.  There _are_ discontinuous jumps in importance of residual values, but these are _not_ progressively increasing with the value of the error itself.  Instead these recurring pronounced importance ranges are vestiges of the discrete scoring nature of football, with the vast majority of scores being multiples of three points or seven points, these being the two most common values of scoring plays (a field goal and a touchdown).  Using the absolute error ensures an easily interpretable metric for evaluating model accuracy with this data: a MAE of 3.5 means we have an average error of 3.5 points.

The two best regression performers were the Support Vector Machine and the Gradient Boosted Trees models, with the Random Forest a small step behind and the ElasticNet behind it.  Initial grid search cross-validation runs with the Gradient Boosted Trees regressor gave an overly optimistic result due to overfitting on the cross-validation set, with a drastically higher cross-validation score than subsequent test score.  This resulted in tweaking the parameters toward fewer trees, a medium tree depth, and a lower learning rate.  

For the point spread, both the SVM and GBT models had an MAE of 2.2 points (and an R<sup>2</sup> of near .740).  For the over/under, they converged to an MAE of 1.8 points and an R<sup>2</sup> of near .715.  This means that at our best, we can predict the point spread of an NFL game to within 2.2 points and the over/under to within 1.8 points.

The best performing model in all classification tasks was the Gradient Boosted Classifier. Classifying whether a team covered the spread or whether a game went over or under the over/under was not particularly responsive to model tuning.  Regardless of the model and its parameters, the AUC hovered close to an even score of 0.500.  I believe this is due to the nature of the categories -- the spread and over/under are designed by oddsmakers to be as close to the break-even point (50/50) as possible, to attract equal bets on both sides.  If anything, these results simply verify that Vegas is quite effective at calculating the expected margin of victory and total points scored per game, _en masse_.  Predicting the winner straight-up, however, is not a result contrived by Vegas and as such does have some appear to have some leeway in determining the outcome via machine learning.  Below are two tables summarizing the results of the model tuning and selection process.  

<br>

Target | Data | Model | Metrics | Score
-----|--------|-------|---------|-----
Game Spread | 1978 | GBR | MAE <br> R<sup>2</sup> | 2.41 <br> 0.699
Game Spread | Advanced | GBR | MAE <br> R<sup>2</sup> | 2.31 <br> 0.731
Over/Under Value | 1978 | GBR | MAE <br> R<sup>2</sup> | 1.85 <br> 0.716
Over/Under Value| Advanced | GBR | MAE <br> R<sup>2</sup> | 1.86 <br> 0.723  

<sub>__Table 3:__ The overview of results from the two regression models and targets in this project.

<BR><BR>

Target | Data | Model | Metrics | Score
-----|--------|-------|---------|-----
Road Team Cover | 1978 | GBC | AUC <br> AUC (SMOTE) | 0.516 <br> 0.498
Road Team Cover | Advanced | GBC | AUC <br> AUC (SMOTE)  | 0.502 <br> 0.530
Over/Under Result | 1978 | GBC | AUC <br> AUC (SMOTE)  | 0.510 <br> 0.509
Over/Under Result | Advanced | GBC | AUC <br> AUC (SMOTE)  | 0.494 <br> 0.497
Home Team Win | 1978 | GBC | AUC <br> AUC (SMOTE)  | 0.631 <br> 0.702
Home Team Win | Advanced | GBC | AUC <br> AUC (SMOTE)  | 0.636 <br> 0.666

<sub>__Table 4:__ The overview of results from the three classification targets using combinations of datasets and class-balancing oversampling (SMOTE).


<BR>

## Results
As mentioned in [Wise Bets](#wise-bets) above, the goal of predicting the spread and the over/under was be able to label games that had improperly set lines which could make them appealing bets.  This means we really wanted to regress against these targets in order to ultimately classify them.  Paired with the three classification targets, this results in the final goal for all models in this project being able to classify whether or not a game is one we should bet on.  A quick glance at __Tables 3__ and __4__ show a fairly pedestrian success rate at correctly predicting two of the three classification targets and a modest but not insignificant error on the regression targets.  

### Class Inspection
#### Principal Component Analysis
One tactic when struggling to find viable class separation is to analyze your data with dimensional reduction.  A popular method of this type of dimensionality reduction is Principal Component Analysis (PCA), which uses some higher-level mathematics to reduce the input data to core, or principal, components based on the amount of observed variance along a given rotational axis of the data.  The result is _not_ simply a set of input features, but rather the 'fundamental' relationships -- components -- between the features and the variance in the data.  If there exists a way to mathematically represent the data in a way that makes it separable in N-dimensions, PCA can tell us.  We can select for the number of components we want returned, which makes PCA ready-made for 2D and 3D visualizations.  

The results of using PCA to analyze the initial target and driving force of this project, the spread, were not encouraging.  Using the model's predictions for the spread to label games as potential "wise bets" or not, PCA showed a inseparable blob in two dimensions.
![PCA 2D Spread](images/2d3pwisebetPCA.png "2D PCA results for the spread")

<sub>__Figure 1:__ The first two principal components failed to give any viable separation for wise bets derived from the Vegas spread -- there is no line that can be drawn to reasonably divide the two classes.  

<br>

The classes are clearly inseparable in two dimensions, but what about three?  It is possible that there exists a hyperplane which can divide the classes in three dimensional space.  For example, picture in your mind the Great Pyramid at Giza, Egypt.  Pretend the limestone blocks that make up the pyramid are separated into two classes by being painted either red or blue.  Now, pretend the top fraction of the pyramid's peak is all red, and the rest of the structure is all blue.  We could divide the red from the blue blocks -- the classes -- by putting a massive sheet of, say, thin plywood, between them.  This sheet of plywood is called a _hyperplane_ and would perfectly separate the two classes of blocks, meaning we could predict mathematically whether a brick was in the red or blue class (no word yet on which class of block is filled with grain...).  

Now, imagine floating high directly above the pyramid and looking down upon it.  You'd see a smaller tip of red blocks in the center surrounded by blue blocks, because the pyramid itself would look like a two dimensional square, much the way mountains look 'flat' when you look directly down on them from a plane.  We would be wholly unable to divide the blue and red blocks in this flat, two-dimensional perspective.  This situation demonstrates the process of using PCA in two dimensions versus three dimensions.  Theoretically, PCA can be used for as many dimensions as there are features in your dataset, but we can only effectively visually represent it in two or three dimensions.

Unlike the simplistic pyramid example, applying PCA in three dimensions to the wise bets from the Vegas spread did not reveal any feasible hyperplane of separation.  

<img src="images/3d_pca_gifs1/3D_PCA.gif" width="600" alt="3D PCA for Vegas Spread wise bets">

<sub>__Figure 2:__ Three dimensions -- each axis is a principal component -- are unfortunately not enough to find a hyperplane of sufficient division between games that are wise bets and games that aren't.  There is no underlying structure to the classes, here.  They're distributed in a roughly globular manner, and almost randomly so.  The hyperplane was obtained by using a linear SVM model.

<br>

#### t-SNE
A second dimensional reduction algorithm, or manifold learner, that is commonly used for visualization is t-distributed Stochastic Neighbor Embedding (t-SNE).  Unlike PCA, t-SNE doesn't provide a Rosetta Stone for translating data into its fundamental components.  Instead it seeks to find local groupings of one data point to its neighbors in high dimensions and visually represent them in lower dimensions.  Its results will change slightly every time it is run, it is very sensitive to its parameters, and cannot be used for inference about unused, new data.  With proper tuning, however, it can reveal grouped relationships which might tell you if your data is actually separable.  Like PCA, t-SNE failed to reveal any underlying structure that could be separated.

<img src="images/model_spread_wb_non-pca_tsne/epsilon50/tsne_wise_bet.gif" width="600" alt="t-SNE for Vegas Spread">  

<sub> __Figure 3:__ The results of increasing levels of perplexity for t-SNE dimension reduction on the Vegas spread bets.  While there is eventual clustering, it is not separable.</sub>

<BR>

#### Important Feature Overlap
With no apparent real success in determining which games should be considered a wise bet, I decided to take a quick glance at the primary advanced metrics which routinely are designated the most important features in the model.  The hope is to see horizontal (x-axis) separation, showing that there are distinct means or groupings for the two classes in a given statistic.  As with PCA and t-SNE, the results were not encouraging as the both classes occupy very similar regions of each feature.

<img src="images/feature_overlap_vegas_spread.png" align="middle" alt="Important Feature Overlap" >

<sub>__Figure 4:__ The advanced metrics of DVOA, EPA, ANY/A, and PORT show very little horizontal separation for games that were labeled as a "wise bet" and those that weren't.  Note: Axis tick labels are removed to help focus simply on bin separation, and the counts have been normalized since the raw count of "wise bet" games is a mere fraction of the total games.</sub>



### Spread Results










Below are the results for each of the five Vegas-related targets investigated in this project.  

__90.4%__ of all spreads are <= +/- 10.



<BR>

## Future Considerations








<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>




### References:
<a name="fn1">1</a>: http://www.nbcnews.com/news/other/think-sports-gambling-isnt-big-money-wanna-bet-f6C10634316  
<a name="fn2">2</a>: https://www.inc.com/slate/jordan-weissmann-is-illegal-sports-betting-a-400-billion-industry.html  
<a name="fn3">3</a>: https://www.boydsbets.com/super-bowl-how-much-bet/  
<a name="fn4">4</a>: http://mentalfloss.com/article/26730/how-wind-chill-calculated
