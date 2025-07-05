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
from models.penalty import Penalty
from models.decision import Decision
from models.field_goal import FieldGoal
from models.punt import Punt
from models.sack import Sack

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
        self.prev_action = None

        # Initialize models
        self.kickoff_model = Kickoff()
        self.try_attempt_model = TryAttemptDecision()
        self.extra_point_model = ExtraPoint()
        self.two_point_conversion_model = TwoPointConversion()
        self.timeout_model = Timeout()
        self.penalty_model = Penalty()
        self.decision_model = Decision()
        self.fg_model = FieldGoal()
        self.punt_model = Punt()
        self.sack_model = Sack()

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
            if self.game_state.clock_is_rolling():
                self._calculate_clock_runoff()
            self._penalty()
            if self.next_action == "kickoff":
                self._kickoff()
            elif self.next_action == "play":
                self._simulate_play()
            elif self.next_action == "extra_point_or_two_point_conversion":
                self._extra_point_or_two_point_conversion()
            else:
                raise ValueError(f"Unknown action: {self.next_action}")

        print(f"score: {self.game_state.get_offense_score()} - {self.game_state.get_defense_score()}")
                    
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
            self.prev_action = self.next_action
            self.next_action = "extra_point_or_two_point_conversion"
        else:
            self.game_state.set_yards_to_goal(ytg)
            self.game_state.set_down(1)
            self.game_state.set_distance(10)
            self.prev_action = self.next_action
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
        self.prev_action = self.next_action
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

    def _penalty(self):
        """
        Simulates a penalty event in the game.
        """
        yards_lost_offense = self.penalty_model.predict_penalty_yards()
        yards_lost_defense = self.penalty_model.predict_penalty_yards()
        current_ytg = self.game_state.get_yards_to_goal()

        if yards_lost_offense > 0:
            if current_ytg + yards_lost_offense > 100:
                # Half the distance to the goal penalty
                penalty_distance = (100 - current_ytg) / 2
                self.game_state.add_to_yards_to_goal(penalty_distance)
                self.game_state.add_to_distance(penalty_distance)
            else:
                # Regular penalty
                self.game_state.add_to_yards_to_goal(yards_lost_offense)
                self.game_state.add_to_distance(yards_lost_offense)
            if self.penalty_model.offensive_penalty_is_loss_of_down():
                # Loss of down for the offense
                if self.game_state.get_down() == 4:
                    # Turnover on downs
                    self.game_state.switch_possession()
                    self.game_state.set_down(1)
                    self.game_state.set_distance(10)
                else:
                    # Just decrement the down
                    self.game_state.increment_down()
        if yards_lost_defense > 0:
            if current_ytg - yards_lost_defense <= 0:
                # Half the distance to the goal penalty
                penalty_distance = current_ytg / 2
                self.game_state.add_to_yards_to_goal(-penalty_distance)
                self.game_state.add_to_distance(-penalty_distance)
            else:
                # Regular penalty
                self.game_state.add_to_yards_to_goal(-yards_lost_defense)
                self.game_state.add_to_distance(-yards_lost_defense)
            
            if self.penalty_model.defensive_penalty_is_automatic_first_down():
                # Automatic first down for the offense
                self.game_state.set_down(1)
                self.game_state.set_distance(10)
    
    def _simulate_play(self):
        down = self.game_state.get_down()

        if down <= 3:
            # Predict action for first 3 downs
            action = self.decision_model.predict_first_3_downs_decision(
                offense_timeouts=self.game_state.get_offense_timeouts(),
                defense_timeouts=self.game_state.get_defense_timeouts(),
                yards_to_goal=self.game_state.get_yards_to_goal(),
                down=down,
                distance=self.game_state.get_distance(),
                seconds_remaining=self.game_state.get_seconds_remaining(),
                pct_game_played=self.game_state.get_pct_game_played(),
                diff_time_ratio=self.game_state.get_diff_time_ratio(),
                temperature=self.game_state.get_temperature(),
                wind_speed=self.game_state.get_wind_speed(),
                offense_elo=self.game_state.get_offense_elo_rating(),
                defense_elo=self.game_state.get_defense_elo_rating(),
                offense_last12_total_fg_poe_gaussian=(
                    self.game_state.get_offense_last12_total_fg_poe_gaussian()
                ),
                offense_last6_pass_to_rush_ratio=(
                    self.game_state.get_offense_last6_pass_to_rush_ratio()
                )
            )
        else:
            action = self.decision_model.predict_4th_down_decision(
                yards_to_goal=self.game_state.get_yards_to_goal(),
                down=down,
                distance=self.game_state.get_distance(),
                score_diff=self.game_state.get_score_diff(),
                pct_game_played=self.game_state.get_pct_game_played(),
                diff_time_ratio=self.game_state.get_diff_time_ratio(),
                offense_elo=self.game_state.get_offense_elo_rating(),
                defense_elo=self.game_state.get_defense_elo_rating(),
                offense_last12_longest_fg=(
                    self.game_state.get_offense_last12_longest_fg()
                ),
                offense_last12_total_fg_poe_gaussian=(
                    self.game_state.get_offense_last12_total_fg_poe_gaussian()
                ),
                temperature=self.game_state.get_temperature(),
                precipitation=self.game_state.get_precipitation(),
                wind_speed=self.game_state.get_wind_speed(),
                offense_last6_pass_to_rush_ratio=(
                    self.game_state.get_offense_last6_pass_to_rush_ratio()
                )
            )

        if action == "pass":
            self._pass_play()
        elif action == "run":
            self._run_play()
        elif action == "field_goal":
            self._field_goal()
        elif action == "punt":
            self._punt()
        elif action == "qb_kneel":
            self._qb_kneel()

    def _run_play(self):
        yards_gained = 4
        self.game_state.add_rush_yards(yards_gained)
        self.game_state.decrement_seconds_remaining(7)
        self.game_state.start_clock()
        self.game_state.increment_down()
        self.game_state.add_to_distance(-yards_gained)
        self.game_state.add_to_yards_to_goal(-yards_gained)
        self.prev_action = "run_play"
        self.next_action = "play"

        if self.game_state.get_down() == 4:
            # Turnover on downs
            self.game_state.switch_possession()
            self.game_state.set_down(1)
            self.game_state.set_distance(10)
            self.game_state.set_yards_to_goal(
                100 - self.game_state.get_yards_to_goal()
            )
            self.game_state.stop_clock()

        if self.game_state.get_yards_to_goal() <= 0:
            # Touchdown
            self.game_state.increment_offense_score(6)
            self.prev_action = self.next_action
            self.next_action = "extra_point_or_two_point_conversion"

    def _pass_play(self):
        sack = self.sack_model.predict_if_sack(
            yards_to_goal=self.game_state.get_yards_to_goal(),
            down=self.game_state.get_down(),
            distance=self.game_state.get_distance(),
            diff_time_ratio=self.game_state.get_diff_time_ratio(),
            elo_diff=self.game_state.get_elo_diff(),
            last6_offense_sacks_allowed_per_game=(
                self.game_state.get_offense_last6_sacks_allowed_per_game()
            ),
            last6_defense_sacks_per_game=(
                self.game_state.get_defense_last6_sacks_per_game()
            )
        )
        ytg = self.game_state.get_yards_to_goal()
        down = self.game_state.get_down()
        if sack:
            sack_fumble = self.sack_model.predict_if_sack_resulted_in_fumble()
            if sack_fumble:
                # Sack resulted in a fumble
                is_offense_recovery = (
                    self.sack_model.predict_sack_fumble_recovery_team()
                )
                yards_lost, seconds_used = (
                    self.sack_model.predict_sack_fumble_recovery_yards_lost(
                        is_offense_recovery=is_offense_recovery,
                        yards_to_goal=self.game_state.get_yards_to_goal()
                    )
                )
                self.game_state.decrement_seconds_remaining(seconds_used)
                if is_offense_recovery:
                    # Offense recovers the fumble
                    if 100 - ytg - yards_lost <= 0:
                        # Safety on fumble recovery
                        self.game_state.stop_clock()
                        self.game_state.increment_defense_score(2)
                        # possession stays with offense for kickoff
                        self.prev_action = "safety"
                        self.next_action = "kickoff"
                    elif down == 4:
                        # Turnover on downs after fumble recovery
                        self.game_state.switch_possession()
                        self.game_state.set_down(1)
                        self.game_state.set_distance(
                            min(10, 100 - ytg - yards_lost)
                        )
                        self.game_state.set_yards_to_goal(
                            100 - self.game_state.get_yards_to_goal() - yards_lost
                        )
                        self.game_state.stop_clock()
                        self.prev_action = "sack_turnover"
                        self.next_action = "play"
                    else:
                        # Regular fumble recovery
                        self.game_state.start_clock()
                        self.game_state.add_to_distance(yards_lost)
                        self.game_state.add_to_yards_to_goal(yards_lost)
                        self.game_state.increment_down()
                        self.prev_action = "sack"
                        self.next_action = "play"
                else:
                    # Defense recovers the fumble
                    if 100 - ytg - yards_lost <= 0:
                        # Touchdown on fumble recovery
                        self.game_state.increment_defense_score(6)
                        self.game_state.stop_clock()
                        self.game_state.switch_possession()
                        self.prev_action = "sack_fumble_td"
                        self.next_action = "extra_point_or_two_point_conversion"
                    else:
                        # Regular fumble recovery by defense
                        self.game_state.switch_possession()
                        self.game_state.set_down(1)
                        self.game_state.set_distance(
                            min(10, 100 - ytg - yards_lost)
                        )
                        self.game_state.set_yards_to_goal(
                            100 - self.game_state.get_yards_to_goal() - yards_lost
                        )
                        self.game_state.stop_clock()
                        self.prev_action = "sack_fumble_recovery"
                        self.next_action = "play"
            else:
                # Sack without fumble
                yards_lost, seconds_used = (
                    self.sack_model.predict_sack_yards_lost(
                        yards_to_goal=self.game_state.get_yards_to_goal()
                    )
                )
                self.game_state.decrement_seconds_remaining(seconds_used)
                if 100 - ytg - yards_lost <= 0:
                    # Safety on sack
                    self.game_state.increment_defense_score(2)
                    self.game_state.stop_clock()
                    self.prev_action = "sack_safety"
                    self.next_action = "kickoff"
                elif down == 4:
                    # Turnover on downs after sack
                    self.game_state.switch_possession()
                    self.game_state.set_down(1)
                    self.game_state.set_distance(
                        min(10, 100 - ytg - yards_lost)
                    )
                    self.game_state.set_yards_to_goal(
                        100 - self.game_state.get_yards_to_goal() - yards_lost
                    )
                    self.game_state.stop_clock()
                    self.prev_action = "sack_turnover"
                    self.next_action = "play"
                else:
                    # Regular sack
                    self.game_state.start_clock()
                    self.game_state.add_to_distance(yards_lost)
                    self.game_state.add_to_yards_to_goal(yards_lost)
                    self.game_state.increment_down()
                    self.prev_action = "sack"
                    self.next_action = "play"
            return
        pass_completion = np.random.rand() < 0.7  # 70% completion rate
        yards_gained = 9 if pass_completion else 0
        self.game_state.decrement_seconds_remaining(7)
        self.game_state.add_pass_yards(yards_gained)
        if pass_completion:
            yards_gained = 6
            self.game_state.start_clock()
            self.game_state.add_to_distance(-yards_gained)
            self.game_state.add_to_yards_to_goal(-yards_gained)
        else:
            self.game_state.stop_clock()
        self.prev_action = "pass_play"
        self.next_action = "play"

        if self.game_state.get_down() == 4:
            # Turnover on downs
            self.game_state.switch_possession()
            self.game_state.set_down(1)
            self.game_state.set_distance(10)
            self.game_state.set_yards_to_goal(
                100 - self.game_state.get_yards_to_goal()
            )
            self.game_state.stop_clock()

        if self.game_state.get_yards_to_goal() <= 0:
            # Touchdown
            self.game_state.increment_offense_score(6)
            self.prev_action = self.next_action
            self.next_action = "extra_point_or_two_point_conversion"

    def _field_goal(self):
        fg_blocked = self.fg_model.predict_if_field_goal_is_blocked(
            kick_distance=self.game_state.get_yards_to_goal() + 17
        )
        if fg_blocked:
            # Field goal blocked
            yards_gained, seconds_used = (
                self.fg_model.predict_yards_gained_if_field_goal_blocked(
                    yards_to_goal=self.game_state.get_yards_to_goal(),
                    offense_elo=self.game_state.get_offense_elo_rating(),
                    defense_elo=self.game_state.get_defense_elo_rating()
                )
            )
            self.game_state.decrement_seconds_remaining(seconds_used)
            self.game_state.stop_clock()
            ytg = 100 - self.game_state.get_yards_to_goal() - yards_gained
            self.game_state.switch_possession()
            if ytg <= 0:
                # Touchdown on blocked field goal
                self.game_state.increment_defense_score(6)
                self.prev_action = "field_goal_blocked_td"
                self.next_action = "extra_point_or_two_point_conversion"
            else:
                # Field goal blocked but not a touchdown
                self.game_state.set_down(1)
                self.game_state.set_distance(min(10, ytg))
                self.game_state.set_yards_to_goal(ytg)
                self.prev_action = "field_goal_blocked"
                self.next_action = "play"
        else:
            # Predict if the field goal is made
            made_fg, seconds_used = self.fg_model.predict_if_field_goal_is_made(
                yards_to_goal=self.game_state.get_yards_to_goal(),
                pct_game_played=self.game_state.get_pct_game_played(),
                score_diff=self.game_state.get_score_diff(),
                elevation=self.game_state.get_elevation(),
                offense_elo=self.game_state.get_offense_elo_rating(),
                temperature=self.game_state.get_temperature(),
                wind_speed=self.game_state.get_wind_speed(),
                offense_last12_total_poe_gaussian=(
                    self.game_state.get_offense_last12_total_fg_poe_gaussian()
                )
            )
            self.game_state.decrement_seconds_remaining(seconds_used)
            self.game_state.stop_clock()
            
            if made_fg:
                self.game_state.increment_offense_score(3)
                self.prev_action = "field_goal"
                self.next_action = "kickoff"
            else:
                self.game_state.switch_possession()
                self.game_state.set_down(1)
                self.game_state.set_distance(10)
                self.game_state.set_yards_to_goal(
                    100 - self.game_state.get_yards_to_goal()
                )
                self.prev_action = "field_goal_miss"
                self.next_action = "play"

    def _punt(self):
        punt_blocked = self.punt_model.predict_if_punt_is_blocked(
            punt_team_ytg=self.game_state.get_yards_to_goal(),
        )
        
        if punt_blocked:
            # Punt blocked
            yards_gained, seconds_used = (
                self.punt_model.predict_yards_gained_if_punt_blocked(
                    yards_to_goal=self.game_state.get_yards_to_goal(),
                    offense_elo=self.game_state.get_offense_elo_rating(),
                    defense_elo=self.game_state.get_defense_elo_rating()
                )
            )
            self.game_state.decrement_seconds_remaining(seconds_used)
            self.game_state.stop_clock()
            ytg = 100 - self.game_state.get_yards_to_goal() - yards_gained
            if ytg <= 0:
                # Touchdown (70%) or safety (30%) on blocked punt
                is_td = np.random.rand() < 0.7
                if is_td:
                    self.game_state.increment_defense_score(6)
                    self.game_state.switch_possession()
                    self.prev_action = "punt_blocked_td"
                    self.next_action = "extra_point_or_two_point_conversion"
                else:
                    # Safety on blocked punt
                    self.game_state.increment_defense_score(2)
                    # Defense gets the ball -> dont switch possession
                    self.prev_action = "punt_blocked_safety"
                    self.next_action = "kickoff"
            else:
                # Punt blocked but not a touchdown
                self.game_state.switch_possession()
                self.game_state.set_down(1)
                self.game_state.set_distance(min(10, ytg))
                self.game_state.set_yards_to_goal(ytg)
                self.prev_action = "punt_blocked"
                self.next_action = "play"
        else:
            # Predict yards gained on punt
            receiving_ytg, seconds_used = (
                self.punt_model.predict_punt_receiving_yards(
                    yards_to_goal=self.game_state.get_yards_to_goal(),
                    offense_elo=self.game_state.get_offense_elo_rating(),
                    defense_elo=self.game_state.get_defense_elo_rating(),
                    elevation=self.game_state.get_elevation(),
                    temperature=self.game_state.get_temperature(),
                    wind_speed=self.game_state.get_wind_speed()
                )
            )
            self.game_state.decrement_seconds_remaining(seconds_used)
            self.game_state.stop_clock()
            self.game_state.switch_possession()
            if receiving_ytg <= 0:
                # Punt return touchdown
                self.game_state.increment_offense_score(6)
                self.prev_action = "punt_return_td"
                self.next_action = "extra_point_or_two_point_conversion"
            else:
                # Regular punt return
                self.game_state.set_down(1)
                self.game_state.set_distance(min(10, receiving_ytg))
                self.game_state.set_yards_to_goal(receiving_ytg)
                self.prev_action = "punt_return"
                self.next_action = "play"

    def _qb_kneel(self):
        yards_lost = -1
        self.game_state.add_rush_yards(yards_lost)
        self.game_state.decrement_seconds_remaining(3)
        self.game_state.start_clock()
        self.game_state.add_to_distance(yards_lost)
        self.game_state.add_to_yards_to_goal(-yards_lost)
        self.prev_action = "qb_kneel"
        self.next_action = "play"

        if self.game_state.get_down() == 4:
            # Turnover on downs
            self.game_state.switch_possession()
            self.game_state.set_down(1)
            self.game_state.set_distance(10)
            self.game_state.set_yards_to_goal(
                100 - self.game_state.get_yards_to_goal()
            )
            self.game_state.stop_clock()
        else:
            self.game_state.increment_down()

    def _calculate_clock_runoff(self):
        if self.prev_action == "qb_kneel":
            # QB kneel runs off 40 seconds
            self.game_state.decrement_seconds_remaining(40)
        else:
            #TODO: NEED TO FINISH THIS LOGIC
            # Regular play runs off 25 seconds
            self.game_state.decrement_seconds_remaining(25)



        # Changing game state features
            # possession
            # home.score
            # home.timeouts
            # away.score
            # away.timeouts
            # seconds_remaining
            # down
            # distance
            # yards_to_goal
            # clock_rolling
            # num_plays_on_drive
            # pass_yards
            # pass_attempts
            # rush_yards
            # rush_attempts