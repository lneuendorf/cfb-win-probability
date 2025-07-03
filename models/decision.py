import numpy as np
import pandas as pd
import xgboost as xgb

class Decision():
    def __init__(self):
        model_path = 'models/raw/decision/first_3_downs_decision_classifier_xgb.bin'
        self.first_3_downs_model = xgb.Booster(model_file=model_path)

        model_path = 'models/raw/decision/fourth_down_decision_classifier_xgb.bin'
        self.fourth_down_model = xgb.Booster(model_file=model_path)

        model_path = 'models/raw/decision/fourth_down_go_decision_classifier_xgb.bin'
        self.go_model = xgb.Booster(model_file=model_path)
        
    def predict_first_3_downs_decision(
        self,
        offense_timeouts: int,
        defense_timeouts: int,
        yards_to_goal: int,
        down: int,
        distance: int,
        seconds_remaining: int, 
        pct_game_played: float,
        diff_time_ratio: float,
        temperature: float,
        wind_speed: float,
        offense_elo: float,
        defense_elo: float,
        offense_last12_total_fg_poe_gaussian: float,
        offense_last6_pass_to_rush_ratio: float,
    ) -> str:
        """ 
        Predicts the action to take on first, second, or third down based on 
        game state and team based features.

        Args:
            offense_timeouts (int): Number of timeouts remaining for the offense.
            defense_timeouts (int): Number of timeouts remaining for the defense.
            yards_to_goal (int): Yards to the end zone from current position.
            down (int): Current down (1 to 3).
            distance (int): Yards needed for a first down.
            seconds_remaining (int): Seconds remaining in the game.
            pct_game_played (float): Percentage of game played.
            diff_time_ratio (float): e^(4 * (3600 - sec_left) / 3600) * score_diff
            temperature (float): Temperature in degrees Fahrenheit.
            wind_speed (float): Wind speed in miles per hour.
            offense_elo (float): ELO rating of the offense team.
            defense_elo (float): ELO rating of the defense team.
            offense_last12_total_fg_poe_gaussian (float): Total FG points of expected
                efficiency (POE) in last 12 games, with gaussian smoothing.
            offense_last6_pass_to_rush_ratio (float): Ratio of pass to run plays
                the last 6 games for the offense.  

        Returns:
            str: The predicted action ('pass', 'run', 'field_goal', 'qb_kneel').
        """
        data = pd.DataFrame([{
            "offense_timeouts": offense_timeouts,
            "defense_timeouts": defense_timeouts,
            "yards_to_goal": yards_to_goal,
            "down": down,
            "distance": distance,
            "pct_game_played": pct_game_played,
            "diff_time_ratio": diff_time_ratio,
            "is_two_minute_drill": self._is_two_minute_drill(seconds_remaining),
            "is_final_minute_of_half": self._is_final_minute_of_half(seconds_remaining),
            "down_x_distance": down * distance,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "offense_elo": offense_elo,
            "defense_elo": defense_elo,
            "offense_last12_total_poe_gaussian": offense_last12_total_fg_poe_gaussian,
            "offense_last6_pass_to_rush_ratio": offense_last6_pass_to_rush_ratio
        }])
        dmatrix = xgb.DMatrix(data)
        probas = self.first_3_downs_model.predict(dmatrix)[0]
        action = np.random.choice(
            ['pass', 'run', 'field_goal', 'qb_kneel'], 
            p=probas
        )
        return action

    def predict_4th_down_decision(
        self,
        yards_to_goal: int,
        down: int,
        distance: int,
        score_diff: int,
        pct_game_played: float,
        diff_time_ratio: float,
        offense_elo: float,
        defense_elo: float,
        offense_last12_longest_fg: float,
        offense_last12_total_fg_poe_gaussian: float,
        temperature: float,
        precipitation: float,
        wind_speed: float,
        offense_last6_pass_to_rush_ratio: float,
    ) -> str:
        """
        Predicts the action to take on 4th down based on various game state 
        features.

        Args:
            yards_to_goal (int): Yards to the end zone from current position.
            down (int): Current down (1 to 4).
            distance (int): Yards needed for a first down.
            score_diff (int): Score difference between teams.
            pct_game_played (float): Percentage of game played.
            diff_time_ratio (float): e^(4 * (3600 - sec_left) / 3600) * score_diff
            offense_elo (float): ELO rating of the offense team.
            defense_elo (float): ELO rating of the defense team.
            offense_last12_longest_fg (float): Longest field goal made by offense 
                in last 12 games.
            offense_last12_total_fg_poe_gaussian (float): Total FG points of expected
                efficiency (POE) in last 12 games, with gaussian smoothing.
            temperature (float): Temperature in degrees Fahrenheit.
            precipitation (float): Precipitation in inches.
            wind_speed (float): Wind speed in miles per hour.
            offense_last6_pass_to_rush_ratio (float): Ratio of pass to run plays
                the last 6 games for the offense. 

        Returns:
            str: The predicted action ('field_goal', 'punt', 'run', 'pass').
        """
        data = pd.DataFrame([{
            "yards_to_goal": yards_to_goal,
            "down": down,
            "distance": distance,
            "score_diff": score_diff,
            "pct_game_played": pct_game_played,
            "diff_time_ratio": diff_time_ratio,
            "down_x_distance": down * distance,
            "offense_elo": offense_elo,
            "defense_elo": defense_elo,
            "offense_last12_longest_fg": offense_last12_longest_fg,
            "offense_last12_total_poe_gaussian": offense_last12_total_fg_poe_gaussian,
            "temperature": temperature,
            "precipitation": precipitation,
            "wind_speed": wind_speed
        }])
        
        dmatrix = xgb.DMatrix(data)
        probas = self.fourth_down_model.predict(dmatrix)[0]
        action = np.random.choice(['go', 'field_goal', 'punt'], p=probas)

        if action == 'go':
            data = pd.DataFrame([{
                "yards_to_goal": yards_to_goal,
                "down": down,
                "distance": distance,
                "score_diff": score_diff,
                "pct_game_played": pct_game_played,
                "diff_time_ratio": diff_time_ratio,
                "offense_elo": offense_elo,
                "defense_elo": defense_elo,
                "offense_last6_pass_to_rush_ratio": offense_last6_pass_to_rush_ratio
            }])
            dmatrix = xgb.DMatrix(data)
            run_proba = self.go_model.predict(dmatrix)
            if np.random.rand() < run_proba:
                action = 'run'
            else:
                action = 'pass'
        
        return action
    
    def _is_two_minute_drill(self, seconds_remaining: int) -> bool:
        return (
            (seconds_remaining <= 120) |
            (1800 <= seconds_remaining <= 1920)
        )
    
    def _is_final_minute_of_half(self, seconds_remaining: int) -> bool:
        return (
            (seconds_remaining <= 60) |
            (1800 <= seconds_remaining <= 1860)
        )