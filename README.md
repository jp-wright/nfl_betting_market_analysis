# NFL Betting Market Prediction and Analysis

Sports betting is a multi-billion dollar per year industry.  Estimates vary, but the Nevada Gaming Commission reported over $3.2B in legal gambling in 2011, with 41% of that money being wagered on football alone. <sup id="a1">[__1__](#fn1)</sup>  According to NBA commissioner Adam Silver, if illegal gambling is counted the annual total for American gambling jumps to $400B. <sup id="a1">[__2__](#fn2)</sup>   The single biggest betting event in America is the Super Bowl, which now attracts over $100M in legal bets and potentially ten times that amount illegally. <sup id="a1">[__3__](#fn3)</sup>  With such sums of money flowing into the betting market, I was curious to see if I could isolate some -- _any_ -- notable inefficiency which would give a leg up on betting intelligently.  

The oddsmakers in Vegas use networks of supercomputers to set the odds, so expecting to beat them outright with a single machine learning model is a bad bet, no pun intended.  However, my hope was that systemic inefficiencies could be ferreted out as Vegas must also shift the odds based on __how the public bets__ in order to prevent a catastrophic loss of house money should an upset occur, granting the one-sided public bets all large payouts.  This opens the door for the informed bettor (us) to profit by the uninformed bettors (the public) making bad choices and forcing Vegas to alter the odds in an inefficient manner. In the interest of full disclosure, I have never bet a penny in my life (not even on slot machines -- sorry, casinos).  

<BR>

## Table of Contents
1. [Dataset](#dataset)
    + [Acquisition and Error Correction](#dataset:-acquisition-and-error-correction)
    + [Feature Engineering](#dataset:-feature-engineering)
    + [Advanced Metrics Limited by Years](#dataset:-advanced-metrics-limited-by-years)
2. [NFL Betting Primer](#nfl-betting-primer)  
    + The Spread
    + The Over/Under
    + The Money Line  
3. [Wise Bets](#wise-bets)  
    + Thresholds
4. [Model Selection](#model-selection)   
    + Classification v. Regression Targets
    + SVMs, ElasticNet, and Ensemble Trees
    + Model Selection and Tuning
      + Trees Feature Importance - Split Datasets
    + Model Training - _EPA_ Dataset Used
5. [Results](#results)  
    + Overview -- Best and Worst
    + Spread
    + Over/Under
    + Money Line
    + Hypothetical Bettor Using This Model
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

### Dataset: Acquisition and Error Correction
Unfortunately, no single source exists which has all these statistics.  In an effort to use as many of these stats as possible I decided to scrape the desired single-game statistics from [Pro Football Reference](www.pro-football-reference.com) (PFR) using BeautifulSoup and URLLib.  PFR is known as the online encyclopedia for all things pro football, and has detailed information for nearly each game played in pro football history, including stadium type, time of game, and weather.  Regarding scraping of their site, PFR makes the following pro-scraping statement on their [data use](http://www.sports-reference.com/data_use.html) page:
>We will not fulfill any requests for data for custom downloads, unless you are prepared to pay a minimum of $1,000 for any such request.
>
>We realize this will be insurmountable for any student requests. However, I would point out that learning how to accumulate data is often a more valuable skill than actually analyzing the data, so we encourage you as a student or professional to learn how.

Surprisingly, most of PFR's data was well-maintained.  Apart from some random errors, there were 87 games that had missing weather data (temp, wind chill, wind speed, and humidity), which forced me to manually look up the weather in the city the game was played in on the date of the game, and insert into the database one-by-one.  (Surprise, it's hot and dry in Arizona).

As a long-time paying member at Football Outsiders, I was able to obtain all the DVOA data in their databases, which runs back to 1991 at present.

### Dataset: Feature Engineering
I engineered features in six distinct ways.
1. __Per-game averages__ for each team in every statistic.  This was necessary for two concrete reasons.
    1. Not every season in NFL history has had the same number of games.
    2. Within a season, teams regularly play an opponent that has not played the same number of games at that point in the season.  Simply using season-long running-total values would skew things in favor of the team that had played more games for positive stats and in favor of the team that had played fewer games for negative stats.   By converting every statistic to a per-game value, we are comparing apples-to-apples.
1. __Deltas__ between the two teams in given statistics, resulting in a single statistic describing the relationship between the teams for the given game.   
    _Example:_ home team averages +103 more yards passing per game than the road team.
2. __Aggregation__ of similar statistics into a single metric.  
    _Example:_ find how many standard deviations a team is above or below league average in a given stat for a given week, and accumulate these for all relevant offensive or defensive stats into a single statistic, such as "Offensive Sigma".
3. __Binning__ of a given statistic into quintiles.
4. __Clustering__ of advanced metrics in an effort to identify "types" of game matchups.
5. Calculating exact hours of rest between games, not merely days.  


### Dataset: Advanced Metrics Limited by Years







## NFL Betting Primer


## Wise Bets


## Model Selection

## Results

## Future Considerations








<BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR><BR>




### References:
<a name="fn1">1</a>: http://www.nbcnews.com/news/other/think-sports-gambling-isnt-big-money-wanna-bet-f6C10634316  
<a name="fn2">2</a>: https://www.inc.com/slate/jordan-weissmann-is-illegal-sports-betting-a-400-billion-industry.html  
<a name="fn3">3</a>: https://www.boydsbets.com/super-bowl-how-much-bet/
