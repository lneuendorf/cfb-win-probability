model limitations
- excludes overtime (for the time being), due to the format changes from 2013-2025
- does not account for turnover probability on kickoff or punt
- does not account for blocked punt, kickoff, fg
- does not account for proabiltiy of penalties
    - does account for pentalty yards indirectly in some models (e.g. punt or kickoff return yardline)
- lacking post-snap penalty modeling
- lacking some team specific tendency features (historical onside kick counts, go rates...)

metric ideas:
- sharp ration adapted to this context, to capture risk and expected value in one metric.
    football_sharpe = E[WP_after - WP_baseline] / std_dev_WP_after where WP_baseline is the avg eWP across all decisions



#TODO
1. Create tests for each model that simulate based on game state and inspect the output distribution, confirming that it aligns with expectation.


#todo: add timeout decision models to simulator
#todo: if offense or defenseive timeouts are 0, then skip the timeout model
#todo: if next play is kickoff or try attempt, then set yards to goal, distance down to none types so they are treated as such in the timeout model