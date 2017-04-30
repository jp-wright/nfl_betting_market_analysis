# NFL Betting Market Prediction and Analysis

Sports betting is a multi-billion dollar per year industry.  Estimates vary, but the Nevada Gaming Commission reported over $3.2B in legal gambling in 2011, with 41% of that money being wagered on football alone. <sup id="a1">[__1__](#fn1)</sup>  According to NBA commissioner Adam Silver, if illegal gambling is counted the annual total for American gambling jumps to $400B. <sup id="a1">[__2__](#fn2)</sup>   The single biggest betting event in America is the Super Bowl, which now attracts over $100M in legal bets and potentially ten times that amount illegally. <sup id="a1">[__3__](#fn3)</sup>  With such sums of money flowing into the betting market, I was curious to see if I could isolate some -- _any_ -- notable inefficiency which would give a leg up on betting intelligently.  

The oddsmakers in Vegas use networks of supercomputers to set the odds, so expecting to beat them outright with a single machine learning model is a bad bet, no pun intended.  However, my hope was that systemic inefficiencies could be ferreted out as Vegas must also shift the odds based on __how the public bets__ in order to prevent a catastrophic loss of house money should an upset occur, granting the one-sided public bets all large payouts.  This opens the door for the informed bettor (us) to profit by the uninformed bettors (the public) making bad choices and forcing Vegas to alter the odds in an inefficient manner. In the interest of full disclosure, I have never bet a penny in my life (not even on slot machines -- sorry, casinos).  

<BR>

## Table of Contents
1. [Dataset](#dataset)
    + [Acquisition and Error Correction](#dataset---acquisition-and-error-correction)
    + [Feature Engineering](#dataset---feature-engineering)
    + [Advanced Metrics Limited by Years](#dataset---advanced-metrics-limited-by-years)
2. [NFL Betting Primer](#nfl-betting-primer)  
    + [The Spread](#bets-the-spread)
    + [The Over/Under](#bets-the-over/under)
    + [The Money Line](#bets-the-money-line)  
3. [Wise Bets](#wise-bets)  
4. [Model Selection](#model-selection)   
    + [Data Selection](#model---data-selection)
    + [Classification v. Regression](#model---classification-v.-regression)
    + [Model Selection and Tuning](#model---model-selection-and-tuning)
      + Trees Feature Importance - Split Datasets  **CHECK RFE accuracy in main model**
    + Model Training - _EPA_ Dataset Used
        + In [157]: 5498*360 - epa
            Out[157]: 1979280
            In [158]: 8788*258 - spread
            Out[158]: 2267304
5. [Results](#results)  
    + Overview -- Best and Worst
    + Spread
      + Features
      + Metrics
    + Over/Under
      + Features
      + Metrics
    + Money Line
      + Features
      + Metrics
    + PCA
    + t-SNE
    + Wise Bets Actual Results
      + Spread
      + Over/Under
      + Money Line
    + Hypothetical Bettor Using This Model
      + Money Line
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

### Dataset - Acquisition and Error Correction
Unfortunately, no single source exists which has all these statistics.  In an effort to use as many of these stats as possible I decided to scrape the desired single-game statistics from [Pro Football Reference](www.pro-football-reference.com) (PFR) using BeautifulSoup and urllib.  PFR is known as the online encyclopedia for all things pro football, and has detailed information for nearly each game played in pro football history, including stadium type, time of game, weather, and Vegas betting information.  Regarding scraping of their site, PFR makes the following pro-scraping statement on their [data use](http://www.sports-reference.com/data_use.html) page:
>We will not fulfill any requests for data for custom downloads, unless you are prepared to pay a minimum of $1,000 for any such request.
>
>We realize this will be insurmountable for any student requests. However, I would point out that learning how to accumulate data is often a more valuable skill than actually analyzing the data, so we encourage you as a student or professional to learn how.

Surprisingly, most of PFR's data was well-maintained.  Apart from some random errors, there were 87 games that had missing weather data (temp, wind chill, wind speed, and humidity), which forced me to manually look up the weather in the city the game was played in on the date of the game, and insert into the database one-by-one.  (Surprise, it's hot and dry in Arizona).  

Last, as a long-time paying member at Football Outsiders, I was able to obtain all the DVOA data in their databases, which runs back to 1991 at present.

### Dataset - Feature Engineering
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

### Dataset - Advanced Metrics Limited by Years
Due to the proprietary nature of some of the advanced metrics, and the reliance upon more granular statistics for others, they are not available for the entirety of the dataset itself.  Vegas line information only goes back to the 1978 season.  Time of possession information starts in 1983, 3rd and 4th Down success rates as well as DVOA begin in 1991, and EPA starting in 1994.  In the future as more old game logs and old game films are parsed and logged, it will be possible for these insightful advanced metrics to be extended further back into league history.  The selection of data from this parent dataset will be discussed below under [Model Selection](#model-selection).

<BR>

## NFL Betting Primer
There are three primary types of wagers made on NFL games: the spread, the over/under, and the money line.  This section will be kept brief, but is necessary to understand the methodology used in model selection and the results found.

### Bets - The Spread
The spread, also called the "line", is a measure of how much better Vegas thinks Team A is than Team B.  Vegas sets the spread in the amount of points the favored team is expected to win by.  A negative spread indicates a team is favored, positive an underdog.  For example, a spread of -3.0 means the favored team is expected to win by a field goal (3 points).  You can bet on either team, the favorite or underdog.  In order to win a bet on the spread, your team must exceed the spread in your favor.  So, if you bet on the favorite at -3.0, they must win by _more_ than 3 points for your bet to win.  If they win by exactly 3 points, the result is called a "push", and all money is returned to bettors, none having been won nor lost.  

### Bets - The Over/Under
The Over/Under is simply the total expected number of points scored by both teams in a game.  You can bet the Over or the Under, and will win if the combined score of the teams is either more than (over) or less than (under) the set Over/Under value, depending on your wager.

### Bets - The Money Line
The money line is simply the odds that a specific team will win the game, regardless of margin of victory (spread).  The money line is given in odds like the spread, where negative implies the favored team, and the odds themselves indicate what the payout will be for a winning bet.

<BR>

## Wise Bets
Games that pass a user-set threshold of deviation from the model's prediction, either in a point spread or in odds to win, are labeled as __wise bets__.

A game whose actual spread deviates from the predicted spread by the user-set point threshold or more will be labeled a "wise bet".  The underlying approach to finding mis-valued spreads works as follows.  The key factor for a spread is its _flexibility_. As Vegas receives more bets on a particular team at a given spread value, they can adjust the spread in order to balance the wagers on the opposing team, reducing the bookmakers' risk by taking near equal money on both sides.  (Vegas typically does not win big on any given game.  They win small amounts consistently by playing percentages very carefully).  This flexibility in the line is the key component I aimed to use in snuffing out inefficiency in the spread.  _If the betting public has a possibly inaccurate perception about a given team, they will either over- or under-bet for that team, forcing Vegas oddsmakers to compensate by artificially adjusting the spread in order to entice bettors to make wagers against their (inaccurate) perception and even out the money wagered._  

__The goal of this model was to find out which factors best predict which games have spreads that are incorrectly set, by how many points they were off, and to predict the result.__  

Secondarily, we can do the same for the Over/Under as well as the Money Line.  The Money Line is slightly different, since it is concerned only with the binary outcome of win/lose.

<BR>

## Model Selection
Four models were tested in this project: Random Forests and Gradient Boosted Trees (GBT), were used initially, followed by Support Vector Machines (SVM) and finally ElasticNet regression.

### Model - Data Selection
As mentioned above, there were three divisions of the original dataset features, as well as up to five progressively smaller year ranges of games to inspect.  The maximum number of features, without dummies, is 345.  The feature-set division arises from my desire to answer the following question: _which single statistics are the most important in predicting X result for an NFL game?_  

The easiest and most direct way to do this is to use a model which has a feature importance attribute.  Gradient Boosted Trees do, and so served this role in this project.  There are two ways of finding the feature importance in ensemble tree models like Random Forests and Gradient Boosted Trees: first, each model has its own attribute which will tell you which features gave the highest return in purity or lowest return in error for the single fitting and run of the model.  Second, a more robust approach is to use Recursive Feature Elimination, which is calculated by fitting the model with all but one feature, measuring how well it predicts, and then repeating this for all features in the dataset until each one has had a turn being left out.  The features that caused the greatest drop in prediction accuracy are judged to be the most important.  

However, if any two features are closely related, the importance for either one will be largely negated by the existence of the other.  For example, pretend we have the two following statistics in this database: _touchdowns in the first three quarters_ and _touchdowns for the entire game_.  When the the stat for the entire game is removed and the model measures how accurate its predictions are, it will still have a feature included that provides three-quarters of the removed stat's information, and the prediction accuracy won't be severely impacted.  As a result, the _touchdowns for the entire game_ statistic will be reported as not being a very important feature.  However, the reality is that this would indeed be an important statistic, but having a correlated or, in this case, partially duplicated, feature clouds our ability to determine its true importance.

Because of this fact, I split the feature set into three parts: All, Raw, and Matchup.  Raw stats are simply every team's own per-game statistics coming into the given game of interest.  Examples include Touchdowns/game, 1st Downs/game, turnovers/gm, etc.  Matchup stats are the differences between the road and home teams in these respective stats.  So, the home team might average seven more first downs per game, but 1.3 more turnovers per game.  Since the matchup stats are technically derived from the raw statistics, I wanted to ensure proper evaluation of feature importances which led to their being optionally separated.

For prediction, having more information tends to be better than having less, and we see that here as using _all_ the features did lead to the best prediction accuracy for any given model.  However, the difference between using all the data and only the matchup data was minor, commonly in the range of 1% to 1.5%.  The raw features alone were noticeably less predictive, sometimes giving up to 5% worse prediction accuracy.


### Model - Classification v. Regression
Since the Spread and the Over/Under are numeric, regression models were used to predict these targets.  Conversely, classification algorithms were used in modeling the Money Line binary winner/loser of a game.  

### Model - Model Selection and Tuning
In order to choose the models which performed best I optimized for the mean absolute error (MAE).  Compared to the root mean squared error (RMSE), the MAE is consistent across ranges of errors and doesn't 'flare' up in response to larger residuals.  For evaluating how many points a predicted NFL game's spread is  from the actual spread, there is no harsher penalty for being five points away than there is for being four points away.  There _are_ discontinuous jumps in importance of residual values, but these are _not_ progressively increasing with the value of the error itself.  Instead these recurring pronounced importance ranges are vestiges of the discrete scoring nature of football, with the vast majority of scores being multiples of three points or seven points, these being the two most common values of scoring plays (a field goal and a touchdown).  Using the absolute error ensures an easily interpretable metric for evaluating model accuracy with this data: a MAE of 3.5 means we have an average error of 3.5 points.

##### Regression
For the regression targets, the two best performers were the Support Vector Machine and the Gradient Boosted Trees models, with the Random Forest a small step behind and the ElasticNet behind it.  Initial grid search cross-validation runs with the Gradient Boosted Trees regressor gave an overly optimistic result due to overfitting on the cross-validation set, with a drastically higher cross-validation score than subsequent test score.  This resulted in tweaking the parameters toward fewer trees, a medium tree depth, and a lower learning rate.  

For the point spread, both the SVM and GBT models had an MAE of 2.2 points (and an R<sup>2</sup> of near .740).  For the over/under, they converged to an MAE of 1.8 points and an R<sup>2</sup> of near .715.  This means that at our best, we can predict the point spread of an NFL game to within 2.2 points and the over/under to within 1.8 points.  

##### Classification


__EPA tends to be__

Data | Target | Model | Key Metrics
-----|--------|-------|------------

EPA
Here are some basic results comparing models.
#### EPA-HomeWin-All:
1. GBC:
_depth: 3, tree: 50, rate: .1 = .73 F1, .63 ROC (.67 Rec, .79 Pre) **_

#### EPA-Spread-All:
1. GBR:
_depth: 3, tree: 100, rate: .1 = 3.0 rmse, 2.3 mae, .73 r2 *_
2. SVR (rbf)
_C: 1000, gamma: .001 = 2.9 rmse, 2.2 mae, .74 r2 **_

#### EPA-O/U-All:
1. GBR
depth: 3, tree: 100, rate: .1 = 2.4 rmse, 1.8 mae, .71 r2 *

2. SVR (rbf)
C: 1000, gamma: .001 = 2.4 rmse, 1.8 mae, .72 r2  *


### Model - Training
+ In [157]: 5498*360 - epa
    Out[157]: 1979280
    In [158]: 8788*258 - spread
    Out[158]: 2267304



<BR>

## Results

<BR>

## Future Considerations








<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>




### References:
<a name="fn1">1</a>: http://www.nbcnews.com/news/other/think-sports-gambling-isnt-big-money-wanna-bet-f6C10634316  
<a name="fn2">2</a>: https://www.inc.com/slate/jordan-weissmann-is-illegal-sports-betting-a-400-billion-industry.html  
<a name="fn3">3</a>: https://www.boydsbets.com/super-bowl-how-much-bet/
