model limitations
- does not account for turnover probability on kickoff or punt
- does not account for blocked punt, kickoff, fg
- does not account for proabiltiy of penalties
    - does account for pentalty yards indirectly in some models (e.g. punt or kickoff return yardline)
- lacking post-snap penalty modeling
- lacking some team specific tendency features (historical onside kick counts, go rates...)

metric ideas:
- sharp ration adapted to this context, to capture risk and expected value in one metric.
    football_sharpe = E[WP_after - WP_baseline] / std_dev_WP_after where WP_baseline is the avg eWP across all decisions


Todo
- create tests of each of the model functions to enusre they are returning logical probabilites / values