model limitations
- excludes overtime (for the time being), due to the format changes from 2013-2025
- does not account for turnover probability on kickoff or punt
- does not account for blocked punt, kickoff, fg
- does not account for proabiltiy of penalties
    - does account for pentalty yards indirectly in some models (e.g. punt or kickoff return yardline)
- lacking post-snap penalty modeling
- lacking some team specific tendency features (historical onside kick counts, go rates...)
- penalty modeling could use more work. Currently base rates.

metric ideas:
- sharp ration adapted to this context, to capture risk and expected value in one metric.
    football_sharpe = E[WP_after - WP_baseline] / std_dev_WP_after where WP_baseline is the avg eWP across all decisions



#TODO
4. Run Models
5. Pass Models
6. Time Runoff Logic between plays if rolling
7. handle for game stopages (2 minute warn, half, quarters)
7. Overtime Model
8. Test bench

Potential bugs
- distance is less than YTG (goal to go)
    - e.g. on field goal block return

Future improvements:
- incoperate in game stats into the models? (total rush yards, ...)
- better modeling of penalties (currently you do not model specifically for pre and pos snap penalties, and you disclude penalty plays from run and pass models)