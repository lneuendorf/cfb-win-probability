import numpy as np
import pandas as pd
import xgboost as xgb

class FieldGoal():
    def __init__(self):
        pass

    def predict_if_field_goal_is_blocked(self, kick_distance: int) -> bool:
        """
        Predicts if a field goal attempt is likely to be blocked based on the
        kick distance.
        
        Args:
            kick_distance (int): Distance of the field goal attempt in yards.
        
        Returns:
            bool: True if the field goal is likely to be blocked, False otherwise.
        """
        fg_block_proba = (
            0.059 if kick_distance >= 60 else 0.00115 * kick_distance - 0.0107
        )
        return np.random.rand() < fg_block_proba