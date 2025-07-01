import numpy as np
import pandas as pd
import xgboost as xgb

class FieldGoal():
    def __init__(self):
        model_path = 'models/raw/field_goal/make_proba_xgb.bin'
        self.fg_make_model = xgb.Booster(model_file=model_path)

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
    
    def predict_if_field_goal_is_made(
        self,
        yards_to_goal: int,
        pct_game_played: float,
        score_diff: float,
        elevation: float,
        offense_elo: float,
        temperature: float,
        wind_speed: float,
        offense_last12_total_poe_gaussian: float,
    ) -> bool:
        """
        Predicts if a field goal attempt is successful based on various factors.
        
        Args:
            yards_to_goal (int): Distance of the field goal attempt in yards.
            pct_game_played (float): Percentage of the game played.
            score_diff (float): Score difference at the time of the kick.
            elevation (float): Elevation of the stadium in feet.
            offense_elo (float): Elo rating of the offense.
            temperature (float): Temperature at the time of the kick in Fahrenheit.
            wind_speed (float): Wind speed at the time of the kick in mph.
            offense_last12_total_poe_gaussian (float): Total FG points of expected
                efficiency (POE) in last 12 games, with gaussian smoothing.
        Returns:
            bool: True if the field goal is likely to be made, False otherwise.
        """
        
        if yards_to_goal <= 48: # Maximum 65 yards FG distance for model
            tie_or_take_lead = 1 if (score_diff >= -3 and score_diff <= 0) else 0
            pressure_rating = self._pressure_rating(tie_or_take_lead, pct_game_played)
            
            input_data = pd.DataFrame({
                'yards_to_goal': [yards_to_goal],
                'pressure_rating': [pressure_rating],
                'elevation': [elevation],
                'offense_elo': [offense_elo],
                'temperature': [temperature],
                'wind_speed': [wind_speed],
                'offense_last12_total_poe_gaussian': [offense_last12_total_poe_gaussian],
                'tie_or_take_lead': [tie_or_take_lead],
            })
            
            dmatrix = xgb.DMatrix(input_data)
            proba = self.fg_make_model.predict(dmatrix)[0]
            return np.random.rand() < proba
        else:
            return False
        
    def _pressure_rating(
        self,
        tie_or_take_lead: int,
        pct_game_played: float,
    ) -> float:
        
        rating = 0
        if tie_or_take_lead == 1:
            if pct_game_played >= (58 / 60): # final 2 minutes
                rating = 4
            elif pct_game_played >= (55 / 60): # final 5 minutes
                rating = 3
            elif pct_game_played >= (50 / 60): # final 10 minutes
                rating = 2
            elif pct_game_played >= (45 / 60): # final 15 minutes
                rating = 1
        
        return rating