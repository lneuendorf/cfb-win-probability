from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import truncnorm

class Sack():
    def __init__(self):
        model_path = 'models/raw/sack/sack_classifier_xgb.bin'
        self.sack_model = xgb.Booster(model_file=model_path)

        self.sack_fumble_yards_lost_offense_dist = pd.read_parquet(
            'models/raw/sack/sack_fumble_yards_lost_offense.parquet'
        )
        self.sack_fumble_yards_lost_defense_dist = pd.read_parquet(
            'models/raw/sack/sack_fumble_yards_lost_defense.parquet'
        )
        self.normal_sack_yards_lost_dist = pd.read_parquet(
            'models/raw/sack/sack_yardage_loss_percentages.parquet'
        )

        # time used for sack (snap to sack)
        lower_bound = 2.0  # minimum plausible sack time
        upper_bound = 7.0  # maximum plausible sack time
        mean = 3.5         # average sack time
        std_dev = 0.8      # standard deviation
        a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
        self.sack_time_dist = truncnorm(a, b, loc=mean, scale=std_dev)

    def predict_if_sack(
        self,
        yards_to_goal: int,
        down: int,
        distance: int,
        diff_time_ratio: float,
        elo_diff: float,
        last6_offense_sacks_allowed_per_game: float,
        last6_defense_sacks_per_game: float
    ) -> bool:
        """
        Predicts whether a sack will occur

        Args:
            yards_to_goal (int): Distance to the end zone in yards.
            down (int): Current down (1, 2, 3, or 4).
            distance (int): Yards to go for a first down.
            diff_time_ratio (float):  e^(4*(3600-sec_left) / 3600) * score_diff
            elo_diff (float): Difference in Elo between offense & defense
            last6_offense_sacks_allowed_per_game (float): Average sacks allowed 
                by the offense in the last 6 games.
            last6_defense_sacks_per_game (float): Average sacks made by the 
                defense in the last 6 games.
        Returns:
            bool: True if a sack is predicted, False otherwise.
        """
        data = pd.DataFrame([{
            'yards_to_goal': yards_to_goal,
            'down': down,
            'distance': distance,
            'diff_time_ratio': diff_time_ratio,
            'elo_diff': elo_diff,
            'last6_offense_sacks_allowed_per_game': last6_offense_sacks_allowed_per_game,
            'last6_defense_sacks_per_game': last6_defense_sacks_per_game
        }])

        dmatrix = xgb.DMatrix(data)
        proba = self.sack_model.predict(dmatrix)[0]
        return np.random.rand() < proba
    
    def predict_if_sack_resulted_in_fumble(self) -> bool:
        """
        Predicts whether a sack resulted in a fumble (~8% of CFB sacks)

        Returns:
            bool: True if the sack resulted in a fumble, False otherwise.
        """
        return np.random.rand() < 0.08
    
    def predict_sack_fumble_recovery_team(self) -> str:
        """
        Predicts which team recovered a fumble after a sack.

        Returns:
            str: 'offense' or 'defense', indicating recovery team
        """
        p_offense = 0.535
        return np.random.choice(
            ['offense', 'defense'],
            p=[p_offense, 1 - p_offense]
        )
    
    def predict_sack_fumble_recovery_yards_lost(
        self,
        is_offense_recovery: bool,
        yards_to_goal: int
    ) -> Tuple[int, int]:
        """
        Predicts the yards lost on a sack fumble recovery.

        Args:
            is_offense_recovery (bool): True if the offense recovered the fumble,
                False if the defense recovered it.
            yards_to_goal (int): Distance of offense to the end zone in yards.

        Returns:
            Tuple[int, int]: A tuple containing the yards lost and the time used
                for the recovery in seconds.
        """
        max_yards_lost = 100 - yards_to_goal
        if is_offense_recovery:
            yards_lost = int(min(
                np.random.choice(
                    self.sack_fumble_yards_lost_offense_dist['yards_lost'],
                    p=self.sack_fumble_yards_lost_offense_dist['proportion']
                ),
                max_yards_lost
            ))
        else:
            yards_lost = int(min(
                np.random.choice(
                    self.sack_fumble_yards_lost_defense_dist['yards_lost'],
                    p=self.sack_fumble_yards_lost_defense_dist['proportion']
                ),
                max_yards_lost
            ))
        time_used = int(
            np.clip(
                3.5 + .13 * yards_lost,  # base time + yards lost factor
                4,
                12
            )
        )
        return yards_lost, time_used
    
    def predict_sack_yards_lost(
        self,
        yards_to_goal: int
    ) -> Tuple[int, int]:
        """
        Predicts the yards lost on a sack (without a fumble).

        Args:
            yards_to_goal (int): Distance of offense to the end zone in yards.

        Returns:
            Tuple[int, int]: A tuple containing the yards lost and the time used
                for the sack in seconds.
        """
        max_yards_lost = 100 - yards_to_goal
        yards_lost = int(min(
            np.random.choice(
                self.normal_sack_yards_lost_dist['yards_lost'],
                p=self.normal_sack_yards_lost_dist['percentage']
            ),
            max_yards_lost
        ))
        time_used = int(self.sack_time_dist.rvs())
        return yards_lost, time_used