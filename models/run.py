from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

class Run():
    def __init__(self):
        self.rush_fumble_yards_lost_offense = pd.read_parquet(
            'models/raw/run/rush_fumble_yards_lost_offense.parquet'
        )
        self.rush_fumble_yards_lost_defense = pd.read_parquet(
            'models/raw/run/rush_fumble_yards_lost_defense.parquet'
        )
        self.rush_yards_gained = {}
        for q in ["q025", "q50", "q975"]:
            model_path = f'models/raw/run/rush_yards_gained_{q}_xgb.bin'
            self.rush_yards_gained[q] = xgb.Booster(model_file=model_path)

    def predict_rush_yards(
        self,
        yards_to_goal: int,
        down: int,
        distance: int,
        diff_time_ratio: float,
        elo_diff: float,
        last12_offense_rushing_plays_ppa: float,
        last12_offense_rushing_plays_success_rate: float,
        last12_defense_rushing_plays_ppa: float,
        last12_defense_rushing_plays_success_rate: float
    ) -> Tuple[int, int]:
        """
        Predicts the yards gained on a rush play and seconds used for the rush.

        Args:
            yards_to_goal (int): Distance to the end zone in yards.
            down (int): Current down (1, 2, 3, or 4).
            distance (int): Yards to go for a first down.
            diff_time_ratio (float): e^(4*(3600-sec_left) / 3600) * score_diff
            elo_diff (float): Difference in Elo between offense & defense.
            last12_offense_rushing_plays_ppa (float): Average PPA of the offense's
                last 12 rushing plays.
            last12_offense_rushing_plays_success_rate (float): Average success rate
                of the offense's last 12 rushing plays.
            last12_defense_rushing_plays_ppa (float): Average PPA of the defense's
                last 12 rushing plays.
            last12_defense_rushing_plays_success_rate (float): Average success rate
                of the defense's last 12 rushing plays.
        Returns:
            Tuple[int, int]: A tuple containing the predicted yards gained on the
                rush and the time used for the rush in seconds.
        """
        max_yards_lost, max_yards_gained = 100 - yards_to_goal, yards_to_goal
        data = pd.DataFrame([{
            'yards_to_goal': yards_to_goal,
            'down': down,
            'distance': distance,
            'diff_time_ratio': diff_time_ratio,
            'elo_diff': elo_diff,
            'last12_offense_rushing_plays_ppa': last12_offense_rushing_plays_ppa,
            'last12_offense_rushing_plays_success_rate': 
                last12_offense_rushing_plays_success_rate,
            'last12_defense_rushing_plays_ppa': last12_defense_rushing_plays_ppa,
            'last12_defense_rushing_plays_success_rate': 
                last12_defense_rushing_plays_success_rate
        }])

        q50 = int(self.rush_yards_gained['q50'].predict(xgb.DMatrix(data))[0])
        q025 = int(self.rush_yards_gained['q025'].predict(xgb.DMatrix(data))[0])
        q975 = int(self.rush_yards_gained['q975'].predict(xgb.DMatrix(data))[0])

        if q025 > q50:
            q025 = q50
        if q975 < q50:
            q975 = q50

        # Sample yards gained using a triangular distribution
        pred_rush_yards = int(np.random.triangular(
            left=np.clip(q025, max_yards_lost, max_yards_gained),
            mode=np.clip(q50, max_yards_lost, max_yards_gained),
            right=np.clip(q975, max_yards_lost, max_yards_gained),
            size=1
        )[0])

        # Calculate time used based on yards gained
        base_time = 5

        return pred_rush_yards, base_time
    
    def predict_if_rushing_fumble(
        self,
    ) -> bool:
        """
        Predicts whether a rushing play results in a fumble.

        Returns:
            bool: True if a fumble is predicted, False otherwise.
        """
        base_rush_fumble_rate = 0.0172 # CFB data 2013 - 2024
        return np.random.rand() < base_rush_fumble_rate

    def predict_recovery_team_on_rushing_fumble(
        self,
    ) -> str:
        """
        Predicts which team recovered a fumble after a rush.

        Returns:
            str: 'offense' or 'defense', indicating recovery team
        """
        return np.random.choice(
            ['offense', 'defense'],
            p=[0.5, 0.5]
        )
    
    def predict_rushing_fumble_yards_lost(
        self,
        is_offense_recovery: bool,
        yards_to_goal: int
    ) -> Tuple[int, int]:
        """
        Predicts the yards lost on a rushing fumble recovery.

        Args:
            is_offense_recovery (bool): True if the offense recovered the fumble,
                False if the defense recovered it.
            yards_to_goal (int): Distance of offense to the end zone in yards.

        Returns:
            Tuple[int, int]: A tuple containing the yards lost and the time used
                for the recovery in seconds.
        """
        max_yards_lost, max_yards_gained = 100 - yards_to_goal, yards_to_goal

        fumble_data = (
            self.rush_fumble_yards_lost_offense if is_offense_recovery 
            else self.rush_fumble_yards_lost_defense
        )

        # Sample a yardage value based on the empirical distribution
        sampled_yards = np.random.choice(
            fumble_data['yards_lost'], 
            p=fumble_data['proportion']
        )

        # Clamp the sampled value to [max_yards_lost, max_yards_gained]
        yards_lost = int(min(max(sampled_yards, max_yards_lost), max_yards_gained))

        time_used = 5

        return yards_lost, time_used