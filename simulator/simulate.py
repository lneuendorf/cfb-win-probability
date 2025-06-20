import os
import numpy as np
import pandas as pd

from simulator.game_state import GameState
from models.kickoff import Kickoff
from models.try_attempt import (
    TryAttemptDecision, 
    ExtraPoint, 
    TwoPointConversion
)

# import json; print(json.dumps(self.game_state, indent=2))
#NOTE: clock stops at 2700, 1800, 900, 120, and 0 seconds remaining

class Simulator:
    def __init__(self, game_state: GameState, next_action: str = "kickoff"):
        # Game State
        self.game_state = game_state
        self.next_action = next_action

        # Initialize models
        self.kickoff_model = Kickoff()
        self.try_attempt_model = TryAttemptDecision()
        self.extra_point_model = ExtraPoint()
        self.two_point_conversion_model = TwoPointConversion()

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
        '''
        Simulates a kickoff event in the game. This handles for the probability
        of an onside kick, the yards to goal (ytg) post kickoff, and
        whether the receiving team recovers the kickoff.
        '''
        ytg, seconds_used, recovered_by = self.kickoff_model.predict_kickoff_ytg(
            score_diff=self.game_state['score_diff'],
            pct_game_played=self.game_state['pct_game_played'],
            diff_time_ratio=self.game_state['diff_time_ratio'],
            offense_timeouts=self.game_state[self.game_state['possession']]['timeouts'],
        )
        self.game_state['seconds_remaining'] -= seconds_used
        self.game_state['clock_rolling'] = False

        # Flip possession if the defense (recieving team) recovers the kickoff
        if recovered_by == 'defense':
            self.game_state['possession'] = (
                'away' if self.game_state['possession'] == 'home' else 'home'
            )

        if ytg == 0: # kick-off TD return
            self.game_state[self.game_state['possession']]['score'] += 6
            self.next_action = "extra_point_or_two_point_conversion"
        else:
            self.game_state['yards_to_goal'] = ytg
            self.game_state['down'] = 1
            self.game_state['distance'] = 10
            self.next_action = "play"

    def _extra_point_or_two_point_conversion(self):
        offense = self.game_state['possession']
        defense = 'away' if offense == 'home' else 'home'
        proba_attempt_xp = self.try_attempt_model.predict_xp_attempt_proba(
            score_diff=self.game_state['score_diff'],
            seconds_remaining=self.game_state['seconds_remaining'],
            diff_time_ratio=self.game_state['diff_time_ratio'],
            pct_game_played=self.game_state['pct_game_played']
        )

        if np.random.rand() < proba_attempt_xp:
            # Attempt extra point
            xp_make_proba = self.extra_point_model.predict_xp_make_proba(
                offense_division=self.game_state[offense]['division'],
                offense_is_power_five=self.game_state[offense]['is_power_five']
            )
            if np.random.rand() < xp_make_proba:
                # Extra point successful
                self.game_state[offense]['score'] += 1
        else:
            # Attempt two-point conversion
            two_pt_make_proba = (
                self.two_point_conversion_model.predict_two_pt_make_proba(
                    offense_division=self.game_state[offense]['division'],
                    defense_division=self.game_state[defense]['division']
                )
            )
            if np.random.rand() < two_pt_make_proba:
                # Two-point conversion successful
                self.game_state[offense]['score'] += 2

        self.next_action = "kickoff"
        
    def _simulate_play(self):
        pass