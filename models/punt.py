from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

class Punt():
    def __init__(self):
        self.punt_receiving_ytg = {}
        for q in ["q025", "q50", "q975"]:
            model_path = f'models/raw/punt/punt_receiving_ytg_{q}_xgb.bin'
            self.punt_receiving_ytg[q] = xgb.Booster(model_file=model_path)

        self.blocked_punt_yards_gained = {}
        for q in ["q025", "q50", "q975"]:
            model_path = f'models/raw/punt/yards_gained_after_block_{q}_xgb.bin'
            self.blocked_punt_yards_gained[q] = xgb.Booster(model_file=model_path)

    def predict_if_punt_is_blocked(self, punt_team_ytg: int) -> bool:
        """
        Predicts if a punt attempt is likely to be blocked based on the
        yards to go for the punt team.

        Args:
            punt_team_ytg (int): Yards to go for the punt team.

        Returns:
            bool: True if the punt is likely to be blocked, False otherwise.
        """
        punt_block_proba = (
            0.0015 if punt_team_ytg < 30 else 0.00019 * punt_team_ytg - 0.00414
        )
        return np.random.rand() < punt_block_proba
    
    def predict_punt_receiving_yards(
        self,
        yards_to_goal: int,
        elevation: float,
        offense_elo: float,
        defense_elo: float,
        temperature: float,
        wind_speed: float,
    ) -> Tuple[int, int]:
        """
        Predicts the yards gained on a punt return and whether the punt is
        returned.

        Args:
            yards_to_goal (int): Distance of the punt in yards.
            elevation (float): Elevation of the stadium in feet.
            offense_elo (float): Elo rating of the offense.
            defense_elo (float): Elo rating of the defense.
            temperature (float): Temperature at the time of the punt in Fahrenheit.
            wind_speed (float): Wind speed at the time of the punt in mph.

        Returns:
            Tuple[int, int]: A tuple containing the predicted yards to goal
                of the receiving team and the time used for the punt return in 
                seconds.
        """
        min_ytg, max_ytg = 1, 99
        df = pd.DataFrame({
            'yards_to_goal': [yards_to_goal],
            'elevation': [elevation],
            'offense_elo': [offense_elo],
            'defense_elo': [defense_elo],
            'temperature': [temperature],
            'wind_speed': [wind_speed]
        })
        dmatrix = xgb.DMatrix(df)
        receiving_ytg = {}
        for q, model in self.punt_receiving_ytg.items():
            receiving_ytg[q] = model.predict(dmatrix)[0]
        
        # Sample from triangular distribution
        receiving_ytg_sample = int(np.random.triangular(
            np.clip(receiving_ytg['q025'], min_ytg, max_ytg),
            np.clip(receiving_ytg['q50'], min_ytg, max_ytg),
            np.clip(receiving_ytg['q975'], min_ytg, max_ytg),
            size=1
        )[0])

        # Calculate time used based on yards gained
        base_time = 5
        extra_seconds = int(abs(min(0, receiving_ytg_sample - 80)) * .15)
        time_used = base_time + extra_seconds

        return receiving_ytg_sample, time_used
    
    def predict_yards_gained_if_punt_blocked(
        self,
        yards_to_goal: int,
        offense_elo: float,
        defense_elo: float
    ) -> Tuple[int, int]:
        """
        Predicts the yards gained and time used if a punt is blocked.

        Args:
            yards_to_goal (int): Distance of the punt attempt in yards.
            offense_elo (float): Elo rating of the offense.
            defense_elo (float): Elo rating of the defense.

        Returns:
            Tuple[int, int]: A tuple containing the yards gained and time used 
                in seconds.
        """
        min_yards_gained = -yards_to_goal + 1  # 99 yards to goal at worst
        max_yards_gained = 100 - yards_to_goal  # touchdown at best
        input_data = pd.DataFrame({
            'yards_to_goal': [yards_to_goal],
            'offense_elo': [offense_elo],
            'defense_elo': [defense_elo],
        })
        dmatrix = xgb.DMatrix(input_data)
        yards_gained_dict = {}
        for q, model in self.blocked_punt_yards_gained.items():
            yards_gained_dict[q] = model.predict(dmatrix)[0]

        # Sample from triangular distribution using the quantiles
        yards_gained = int(np.random.triangular(
            np.clip(yards_gained_dict['q025'], min_yards_gained, max_yards_gained),
            np.clip(yards_gained_dict['q50'], min_yards_gained, max_yards_gained),
            np.clip(yards_gained_dict['q975'], min_yards_gained, max_yards_gained),
            size=1
        )[0])

        # Calculate time used based on yards gained
        base_time = 5
        extra_seconds = int(abs(yards_gained) // 10)
        time_used = base_time + extra_seconds

        return yards_gained, time_used