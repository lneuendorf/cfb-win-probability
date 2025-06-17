import xgboost as xgb
import numpy as np
import pandas as pd

class TryAttemptDecision:
    def __init__(self):
        model_path = 'models/raw/try_attempt/decision_classifier_xgb.bin'
        self.model = xgb.Booster(model_file=model_path)

    def predict_xp_attempt_proba(
        self,
        score_diff: int,
        seconds_remaining: float,
        diff_time_ratio: float,
        pct_game_played: float,
    ):
        """
        Predicts the probability of attempting an extra point (XP) based on game 
        state verse attempting a two-point conversion.

        Args:
            score_diff (int): Score difference before the extra point attempt.
            seconds_remaining (float): Seconds remaining in the game.
            diff_time_ratio (float): e^(4 * (3600 - sec_left) / 3600) * score_diff
            pct_game_played (float): Percentage of the game played.

        Returns:
            float: Probability of attempting an extra point.
        """
        two_point_to_tie = int(np.where(score_diff == -2, 1, 0))
        two_point_to_lead = int(np.where(score_diff == -1, 1, 0))
        need_two_pt_to_tie = int(self._special_tie_condition(score_diff))
        data = pd.DataFrame([{
            "score_diff_before": score_diff,
            "pct_game_played": pct_game_played,
            "diff_time_ratio": diff_time_ratio,
            "two_point_to_tie": two_point_to_tie,
            "two_point_to_lead": two_point_to_lead,
            "will_need_two_pt_to_tie": need_two_pt_to_tie
        }])

        dmatrix = xgb.DMatrix(data)
        proba = self.model.predict(dmatrix)[0]
        return proba
    
    def _can_tie_with_3_and_7_one_fg(self, diff):
        target = -diff
        for fg in [0, 1]:  # 0 or 1 field goals (not allowing for more than 1 FG)
            for td in range((target - 3 * fg) // 7 + 1):
                if 7 * td + 3 * fg == target:
                    return True
        return False

    def _can_tie_with_3_7_8_one_fg(self, diff):
        target = -diff
        for fg in [0, 1]:  # 0 or 1 field goals
            for td_8 in range((target - 3 * fg) // 8 + 1):
                for td_7 in range((target - 3 * fg - 8 * td_8) // 7 + 1):
                    if 8 * td_8 + 7 * td_7 + 3 * fg == target:
                        return True
        return False

    def _special_tie_condition(self, diff):
        if diff >= 0:
            return False  # not trailing
        return (
            not self._can_tie_with_3_and_7_one_fg(diff)
            and self._can_tie_with_3_7_8_one_fg(diff)
        )


class ExtraPoint:
    def __init__(self):
        self.probas = pd.read_parquet(
            'models/raw/try_attempt/extra_point_success_rates.parquet'
        ).set_index('division').to_dict()['success_rate']

    def predict_xp_make_proba(
        self, 
        offense_division: str,
        offense_is_power_five: bool,
    ):
        """
        Returns the empirical probability of a successful extra point (XP) based 
        on the offense division and whether the offense is a Power Five team.

        Args:
            offense_division (str): either "FBS" or "FCS".
            offense_is_power_five (bool): If the offense is a Power Five team.

        Returns:
            float: Probability of a successful extra point.
        """
        non_fbs_or_fcs_proba = 0.85

        p5_indicator = "power_5" if offense_is_power_five else "non_power_5"
        key = f"{offense_division.lower()}"
        key = f"{key}_{p5_indicator}" if offense_division == "FBS" else key
        return self.probas.get(key, non_fbs_or_fcs_proba)

class TwoPointConversion:
    def __init__(self):
        self.probas = (
            pd.read_parquet(
                'models/raw/try_attempt/extra_point_success_rates.parquet'
            )
            .assign(
                division_matchup=(
                    lambda x: x.offense_division + "-" + x.defense_division
                )
            )
            .drop(columns=['offense_division', 'defense_division'])
        ).set_index('division_matchup').to_dict()['success_rate']
    
    def predict_two_pt_make_proba(
        self,
        offense_division: str,
        defense_division: str,
    ):
        """
        Returns the empirical probability of a successful two-point conversion 
        based on the offense and defense divisions.

        Args:
            offense_division (str): either "FBS" or "FCS".
            defense_division (str): either "FBS" or "FCS".

        Returns:
            float: Probability of a successful two-point conversion.
        """
        # Conservative since they will always be aginast FCS/FBS in this model
        non_fcs_or_fbs_proba = 0.3 

        key = f"{offense_division.lower()}-{defense_division.lower()}"
        return self.probas.get(key, non_fcs_or_fbs_proba)