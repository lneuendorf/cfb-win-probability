import numpy as np
import pandas as pd

from simulator.game_state import GameState
from models.kickoff import Kickoff
from models.try_attempt import (
    TryAttemptDecision, 
    ExtraPoint, 
    TwoPointConversion
)
from models.timeout import Timeout

# import json; print(json.dumps(self.game_state, indent=2))
#NOTE: clock stops at 2700, 1800, 900, 120, and 0 seconds remaining

class Simulator:
    def __init__(
        self, 
        game_state: GameState, 
        next_action: str = "kickoff"
    ):
        # Game State
        self.game_state = game_state
        self.next_action = next_action

        # Initialize models
        self.kickoff_model = Kickoff()
        self.try_attempt_model = TryAttemptDecision()
        self.extra_point_model = ExtraPoint()
        self.two_point_conversion_model = TwoPointConversion()
        self.timeout_model = Timeout()

    def run(self) -> int:
        """
        Simulates a single game based on the current game state.
        
        Returns:
            int: 1 if home team wins, 0 if tie, -1 if away team wins.
        """

        # Game has not started -> coin toss
        if self.game_state.get_possession() is None:
            self._coin_toss()

        # Simulate non-overtime quarters
        while self.game_state.get_seconds_remaining() > 0:
            self._timeout()
            if self.next_action == "kickoff":
                self._kickoff()
            elif self.next_action == "play":
                self._simulate_play()
            elif self.next_action == "extra_point_or_two_point_conversion":
                self._extra_point_or_two_point_conversion()
            else:
                raise ValueError(f"Unknown action: {self.next_action}")
                    
    def _coin_toss(self):
        self.game_state.set_possession(
            np.random.choice(['home', 'away'], p=[0.5, 0.5])
        )

    def _kickoff(self):
        '''
        Simulates a kickoff event in the game. This handles for the probability
        of an onside kick, the yards to goal (ytg) post kickoff, and
        whether the receiving team recovers the kickoff.
        '''
        ytg, seconds_used, recovered_by = self.kickoff_model.predict_kickoff_ytg(
            score_diff=self.game_state.get_score_diff(),
            pct_game_played=self.game_state.get_pct_game_played(),
            diff_time_ratio=self.game_state.get_diff_time_ratio(),
            offense_timeouts=self.game_state.get_offense_timeouts(),
        )
        self.game_state.decrement_seconds_remaining(seconds_used)
        self.game_state.stop_clock()

        # Flip possession if the defense (recieving team) recovers the kickoff
        if recovered_by == 'defense':
            self.game_state.switch_possession()

        if ytg == 0: # kick-off TD return
            self.game_state.increment_offense_score(6)
            self.next_action = "extra_point_or_two_point_conversion"
        else:
            self.game_state.set_yards_to_goal(ytg)
            self.game_state.set_down(1)
            self.game_state.set_distance(10)
            self.next_action = "play"

    def _extra_point_or_two_point_conversion(self):
        proba_attempt_xp = self.try_attempt_model.predict_xp_attempt_proba(
            score_diff=self.game_state.get_score_diff(),
            diff_time_ratio=self.game_state.get_diff_time_ratio(),
            pct_game_played=self.game_state.get_pct_game_played(),
        )

        if np.random.rand() < proba_attempt_xp:
            # Attempt extra point
            xp_make_proba = self.extra_point_model.predict_xp_make_proba(
                offense_division=self.game_state.get_offense_division(),
                offense_is_power_five=self.game_state.offense_is_power_five()
            )
            if np.random.rand() < xp_make_proba:
                # Extra point successful
                self.game_state.increment_offense_score(1)
        else:
            # Attempt two-point conversion
            two_pt_make_proba = (
                self.two_point_conversion_model.predict_two_pt_make_proba(
                    offense_division=self.game_state.get_offense_division(),
                    defense_division=self.game_state.get_defense_division(),
                )
            )
            if np.random.rand() < two_pt_make_proba:
                # Two-point conversion successful
                self.game_state.increment_offense_score(2)

        self.next_action = "kickoff"

    def _timeout(self):
        """
        Simulates a timeout event in the game. This will stop the clock and
        decrement the number of timeouts for the timeout calling team.
        """

        # Predict if the offense calls a timeout
        if self.game_state.get_offense_timeouts() > 0:
            proba = self.timeout_model.predict_offensive_timeout_proba(
                offense_timeouts=self.game_state.get_offense_timeouts(),
                yards_to_goal=self.game_state.get_yards_to_goal(),
                down=self.game_state.get_down(),
                distance=self.game_state.get_distance(),
                pct_game_played=self.game_state.get_pct_game_played(),
                score_diff=self.game_state.get_score_diff(),
                diff_time_ratio=self.game_state.get_diff_time_ratio(),
                clock_rolling=self.game_state.clock_is_rolling(),
                offense_pregame_elo=self.game_state.get_offense_elo_rating(),
                defense_pregame_elo=self.game_state.get_defense_elo_rating(),
                offense_is_home=self.game_state.get_offense_is_home(),
                num_prior_plays_on_drive=self.game_state.get_play_count()
            )
            if np.random.rand() < proba:
                # Offense calls a timeout
                self.game_state.decrement_offense_timeouts()
                self.game_state.stop_clock()
        
        # Predict if the defense calls a timeout
        if self.game_state.get_defense_timeouts() > 0:
            proba = self.timeout_model.predict_defensive_timeout_proba(
                defense_timeouts=self.game_state.get_defense_timeouts(),
                yards_to_goal=self.game_state.get_yards_to_goal(),
                down=self.game_state.get_down(),
                distance=self.game_state.get_distance(),
                pct_game_played=self.game_state.get_pct_game_played(),
                score_diff=self.game_state.get_score_diff(),
                diff_time_ratio=self.game_state.get_diff_time_ratio(),
                clock_rolling=self.game_state.clock_is_rolling(),
                offense_pregame_elo=self.game_state.get_offense_elo_rating(),
                defense_pregame_elo=self.game_state.get_defense_elo_rating(),
                defense_is_home=self.game_state.get_defense_is_home(),
                num_prior_plays_on_drive=self.game_state.get_play_count()
            )
            if np.random.rand() < proba:
                # Defense calls a timeout
                self.game_state.decrement_defense_timeouts
                self.game_state.stop_clock()
        
    def _simulate_play(self):
        pass