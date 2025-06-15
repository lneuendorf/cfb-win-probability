from typing import Tuple
import numpy as np
import pandas as pd

receiving_team_ytg_bins = pd.read_parquet(
    'models/raw/kickoff_binned_probabilities.parquet'
)

class Kickoff():
    def __init__(self):
        df = pd.read_parquet(
            'models/raw/kickoff_binned_probabilities.parquet'
        )
        self.probas = df['empirical_proba'].values
        self.time_dict = df['seconds_used'].to_dict()
        self.ytg_bins = df['receiving_team_ytg'].values

    def predict_kickoff_ytg(self) -> Tuple[int, int]:
        """
        Predicts the yards to goal (YTG) after a kickoff based on historical rates.
        
        Returns:
            tuple: (yards_to_goal, seconds_used)
        """
        ytg = np.random.choice(
            self.ytg_bins,
            p=self.probas
        )
        
        return int(ytg), self.time_dict[ytg]