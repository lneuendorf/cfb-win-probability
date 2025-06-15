import os
import numpy as np
import pandas as pd

from simulator.game_state import GameState
from models.kickoff import Kickoff

# import json; print(json.dumps(self.game_state, indent=2))

#NOTE: clock stops at 2700, 1800, 900, 120, and 0 seconds remaining

class Simulator:
    def __init__(self, game_state: GameState, next_action: str = "kickoff"):
        self.game_state = game_state
        self.next_action = next_action
        self.kickoff_model = Kickoff()

    def run(self) -> int:
        """
        Simulates a single game based on the current game state.
        
        Returns:
            int: 1 if home team wins, 0 if tie, -1 if away team wins.
        """

        # Game has not started -> coin toss
        if self.game_state['possession'] is None:
            self._coin_toss()

        # Simulate non-overtime quarters
        while self.game_state['seconds_remaining'] > 0:
            if self.next_action == "kickoff":
                self._kickoff()
            elif self.next_action == "play":
                self._simulate_play()
            elif self.next_action == "extra_point_or_two_point_conversion":
                self._extra_point_or_two_point_conversion()
            else:
                raise ValueError(f"Unknown action: {self.next_action}")
        
    def _coin_toss(self):
        self.game_state['possession'] = np.random.choice(
            ['home', 'away'],
            p=[0.5, 0.5]
        )

    def _kickoff(self):
        ytg, seconds_used = self.kickoff_model.predict_kickoff_ytg()
        self.game_state['seconds_remaining'] -= seconds_used

        if ytg == 0: # kick-off TD return
            self.game_state[self.game_state['possession']]['score'] += 6
            self.next_action = "extra_point_or_two_point_conversion"
        else:
            self.game_state['yards_to_goal'] = ytg
            self.game_state['down'] = 1
            self.game_state['distance'] = 10
            self.next_action = "play"