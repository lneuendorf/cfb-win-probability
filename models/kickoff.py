from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

class Kickoff():
    def __init__(self):
        # Load the kickoff return PDF with empirical probabilities
        df = pd.read_parquet(
            'models/raw/kickoffs/kickoff_binned_probabilities.parquet'
        )
        self.regular_kickoff = dict(
            ytg_bins = df['receiving_team_ytg'].values,
            probas = df['empirical_proba'].values,
            time_dict = df['seconds_used'].to_dict()
        )

        # Load the onside kickoff defense (recieving team) recovery PDF
        df = pd.read_parquet(
            'models/raw/kickoffs/onside_kick_defense_yardline_distribution.parquet'
        )
        self.onside_kickoff_defense = dict(
            ytg_bins = df.index.values,
            probas = df['probability'].values,
            time_dict = df['seconds_used'].to_dict()
        )

        # Load the onside kickoff offense (kicking team) recovery PDF
        df = pd.read_parquet(
            'models/raw/kickoffs/onside_kick_offense_yardline_distribution.parquet'
        )
        self.onside_kickoff_offense = dict(
            ytg_bins = df.index.values,
            probas = df['probability'].values,
            time_dict = df['seconds_used'].to_dict()
        )

        # Load the model predicting if a kickoff is regular or onside
        model_path = 'models/raw/kickoffs/onside_decision_xgb.bin'
        self.model = xgb.Booster(model_file=model_path)

    def predict_kickoff_ytg(
            self,
            score_diff: int,
            pct_game_played: float,
            diff_time_ratio: float,
            offense_timeouts: int,
        ) -> Tuple[int, int, str]:
        """
        Predicts if kick is onside or regular, and returns the yards to goal
        (YTG), seconds used, and recovery team.

        Args:
            score_diff (int): The score difference between the teams.
            pct_game_played (float): Percentage of game played.
            diff_time_ratio (float): e^(4 * (3600 - sec_left) / 3600) * score_diff
            offense_timeouts (int): Remaining timeouts for the offense.
        
        Returns:
            tuple: (yards_to_goal, seconds_used, recovery_team)
        """
        onside_proba = self._predict_if_onside(
            score_diff=score_diff,
            pct_game_played=pct_game_played,
            diff_time_ratio=diff_time_ratio,
            offense_timeouts=offense_timeouts
        )

        is_onside = np.random.rand() < onside_proba

        if is_onside:
            recovery_team = self._predict_onside_kick_recovery_team()
            ytg, seconds_used = self._predict_onside_kick_recovery_ytg(
                recovery_team=recovery_team
            )
        else:
            recovery_team = 'defense' # this is the receiving team
            ytg, seconds_used = self._predict_regular_kickoff_ytg()
        return ytg, seconds_used, recovery_team

    def _predict_if_onside(
        self,
        score_diff: int,
        pct_game_played: float,
        diff_time_ratio: float,
        offense_timeouts: int,
    ) -> float:
        """
        Predicts if a kickoff is an onside kick based on the game state.
        
        Args:
            score_diff (int): The score difference between the teams.
            pct_game_played (float): Percentage of game played.
            diff_time_ratio (float): e^(4 * (3600 - sec_left) / 3600) * score_diff
            offense_timeouts (int): Remaining timeouts for the offense.
        Returns:
            float: Probability of the kickoff being an onside kick.
        """
        data = pd.DataFrame([{
            "score_diff": score_diff,
            "pct_game_played": pct_game_played,
            "diff_time_ratio": diff_time_ratio,
            "offense_timeouts": offense_timeouts
        }])
        
        dmatrix = xgb.DMatrix(data)
        proba = self.model.predict(dmatrix)[0]
        
        return proba
    
    def _predict_onside_kick_recovery_team(
            self,
    ) -> str:
        """
        Predicts which team recovers an onside kick based on historical 
        probabilities.
        
        Returns:
            str: 'offense' or 'defense' indicating the recovering team.
        """

        # Value set referencing historical onside kick recovery rates 2013-2024
        P_OFFENSE_RECOVERY = 0.2
        
        return np.random.choice(
            ['offense', 'defense'],
            p=[P_OFFENSE_RECOVERY, 1 - P_OFFENSE_RECOVERY]
        )
        
    def _predict_regular_kickoff_ytg(self) -> Tuple[int, int]:
        """
        Predicts the yards to goal (YTG) after a regular kickoff based on 
        historical rates.
        
        Returns:
            tuple: (yards_to_goal, seconds_used)
        """
        ytg = np.random.choice(
            self.regular_kickoff['ytg_bins'],
            p=self.regular_kickoff['probas']
        )
        
        return int(ytg), self.regular_kickoff['time_dict'][ytg]
    
    def _predict_onside_kick_recovery_ytg(
        self,
        recovery_team: str
    ) -> Tuple[int, int]:
        """
        Predicts the yards to goal (YTG) after an onside kick recovery based on 
        historical rates.
        
        Args:
            recovery_team (str): 'offense' or 'defense' indicating recovery team.
        
        Returns:
            tuple: (yards_to_goal, seconds_used)
        """
        if recovery_team == 'offense':
            ytg = np.random.choice(
                self.onside_kickoff_offense['ytg_bins'],
                p=self.onside_kickoff_offense['probas']
            )
            return int(ytg), self.onside_kickoff_offense['time_dict'][ytg]
        else:
            ytg = np.random.choice(
                self.onside_kickoff_defense['ytg_bins'],
                p=self.onside_kickoff_defense['probas']
            )
            return int(ytg), self.onside_kickoff_defense['time_dict'][ytg]