from typing import Tuple
import numpy as np
import pandas as pd

class Penalty():
    def __init__(self):
       self.PENALTY_RATE = 0.036
       self.penalty_yards = dict(
            yardage_bins = [5, 10, 15],
            probas = [0.49, 0.36, 0.15]
       )

    def predict_penalty_yards(
        self,
    ) -> int:
        """
        Predicts the yards lost due to a penalty based on historical rates.

        Returns:
            int: Yards lost due to penalty (0 if no penalty).
        """

        if np.random.rand() < self.PENALTY_RATE:
            yards = np.random.choice(
                self.penalty_yards['yardage_bins'],
                p=self.penalty_yards['probas']
            )
            return int(yards)
        else:
            return 0
        

    def defensive_penalty_is_automatic_first_down(
        self
    ) -> bool:
        """
        Determines if a defensive penalty results in an automatic first down.

        Returns:
            bool: True if automatic first down, False otherwise.
        """
        AUTOMATIC_FIRST_DOWN_RATE = 0.4 # ~40% chance of automatic first down

        return np.random.rand() < AUTOMATIC_FIRST_DOWN_RATE
    
    def offensive_penalty_is_loss_of_down(
        self
    ) -> bool:
        """
        Determines if an offensive penalty results in a loss of down.

        Returns:
            bool: True if loss of down, False otherwise.
        """
        LOSS_OF_DOWN_RATE = 0.05 # ~5% chance of loss of down

        return np.random.rand() < LOSS_OF_DOWN_RATE