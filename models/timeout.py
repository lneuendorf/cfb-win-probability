import pandas as pd
import xgboost as xgb

class Timeout():
    def __init__(self):
        model_path = 'models/raw/timeout/offense_timeout_classifier_xgb.bin'
        self.offense_model = xgb.Booster(model_file=model_path)

        model_path = 'models/raw/timeout/defense_timeout_classifier_xgb.bin'
        self.defense_model = xgb.Booster(model_file=model_path)

    def predict_offensive_timeout_proba(
            self,
            offense_timeouts: int,
            yards_to_goal: int,
            down: int,
            distance: int,
            pct_game_played: float,
            score_diff: int,
            diff_time_ratio: float,
            clock_rolling: bool,
            offense_pregame_elo: float,
            defense_pregame_elo: float,
            offense_is_home: bool,
            num_prior_plays_on_drive: int,

        ) -> float:
        """
        Predicts the probability of the defense calling a timeout.

        Args:
            offense_timeouts (int): Remaining timeouts for the offense.
            yards_to_goal (int): Yards to the opponent's end zone.
            down (int): Current down (1-4).
            distance (int): Yards needed for a first down.
            pct_game_played (float): Percentage of game played.
            score_diff (int): Score difference between teams.
            diff_time_ratio (float): e^(4*(3600-sec_left) / 3600) * score_diff
            clock_rolling (bool): Whether the clock is rolling.
            offense_pregame_elo (float): Offense pregame Elo rating.
            defense_pregame_elo (float): Defense pregame Elo rating.
            offense_is_home (bool): Whether the offense is playing at home.
            num_prior_plays_on_drive (int): Number of prior plays on the drive.
        
        Returns:
            float: Probability of the offense calling a timeout.
        """
        is_redzone, is_goal_to_go = 0, 0
        if yards_to_goal is not None:
            is_redzone = int(yards_to_goal <= 20)
            if distance is not None:
                is_goal_to_go = int(yards_to_goal <= distance)
        
        data = pd.DataFrame([{
            'prev_offense_timeouts': offense_timeouts,
            'yards_to_goal': yards_to_goal,
            'down': down,
            'distance': distance,
            'pct_game_played': pct_game_played,
            'score_diff': score_diff,
            'diff_time_ratio': diff_time_ratio,
            'clock_rolling_prior_to_play': clock_rolling,
            'offense_pregame_elo': offense_pregame_elo,
            'defense_pregame_elo': defense_pregame_elo,
            'offense_is_home': offense_is_home,
            'num_prior_plays_on_drive': num_prior_plays_on_drive,
            'is_redzone': is_redzone,
            'is_goal_to_go': is_goal_to_go
        }])
        dmatrix = xgb.DMatrix(data)
        proba = self.offense_model.predict(dmatrix)[0]
        return proba
    
    def predict_defensive_timeout_proba(
            self,
            defense_timeouts: int,
            yards_to_goal: int,
            down: int,
            distance: int,
            pct_game_played: float,
            score_diff: int,
            diff_time_ratio: float,
            clock_rolling: bool,
            offense_pregame_elo: float,
            defense_pregame_elo: float,
            defense_is_home: bool,
            num_prior_plays_on_drive: int,

        ) -> float:
        """
        Predicts the probability of the defense calling a timeout.

        Args:
            defense_timeouts (int): Remaining timeouts for the defense.
            yards_to_goal (int): Yards to the opponent's end zone.
            down (int): Current down (1-4).
            distance (int): Yards needed for a first down.
            pct_game_played (float): Percentage of game played.
            score_diff (int): Score difference between teams.
            diff_time_ratio (float): e^(4*(3600-sec_left) / 3600) * score_diff
            clock_rolling (bool): Whether the clock is rolling.
            offense_pregame_elo (float): Offense pregame Elo rating.
            defense_pregame_elo (float): Defense pregame Elo rating.
            offense_is_home (bool): Whether the offense is playing at home.
            num_prior_plays_on_drive (int): Number of prior plays on the drive.
        
        Returns:
            float: Probability of the defense calling a timeout.
        """
        is_redzone, is_goal_to_go = False, False
        if yards_to_goal is not None:
            is_redzone = yards_to_goal <= 20
            if distance is not None:
                is_goal_to_go = yards_to_goal <= distance
        
        data = pd.DataFrame([{
            'prev_defense_timeouts': defense_timeouts,
            'yards_to_goal': yards_to_goal,
            'down': down,
            'distance': distance,
            'pct_game_played': pct_game_played,
            'score_diff': score_diff,
            'diff_time_ratio': diff_time_ratio,
            'clock_rolling_prior_to_play': clock_rolling,
            'offense_pregame_elo': offense_pregame_elo,
            'defense_pregame_elo': defense_pregame_elo,
            'defense_is_home': defense_is_home,
            'num_prior_plays_on_drive': num_prior_plays_on_drive,
            'is_redzone': is_redzone,
            'is_goal_to_go': is_goal_to_go
        }])
        dmatrix = xgb.DMatrix(data)
        proba = self.defense_model.predict(dmatrix)[0]
        return proba