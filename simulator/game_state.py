import numpy as np

class GameState:
    def __init__(
        self,
        possession: str,
        home_score: int,
        away_score: int,
        home_elo_rating: float,
        away_elo_rating: float,
        home_timeouts: int,
        away_timeouts: int,
        home_division: str,
        away_division: str,
        home_is_power_five: bool,
        away_is_power_five: bool,
        seconds_remaining: int,
        down: int,
        distance: int,
        yards_to_goal: int,
        temperature: float,
        wind_speed: float,
        precipitation: float,
        elevation: float,
        clock_rolling: bool = False,
        neutral_site: bool = False
    ):
        # Core fields
        self.possession = possession
        self.home = {
            'score': home_score,
            'elo_rating': home_elo_rating,
            'timeouts': home_timeouts,
            'division': home_division,
            'is_power_five': home_is_power_five
        }
        self.away = {
            'score': away_score,
            'elo_rating': away_elo_rating,
            'timeouts': away_timeouts,
            'division': away_division,
            'is_power_five': away_is_power_five
        }

        self.seconds_remaining = seconds_remaining
        self.down = down
        self.distance = distance
        self.yards_to_goal = yards_to_goal
        self.weather = {
            'temperature': temperature,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        }
        self.elevation = elevation
        self.clock_rolling = clock_rolling
        self.neutral_site = neutral_site

        # Drive info
        self.num_plays_on_drive = 0

        # Derived
        self._update_score_diff()
        self._update_pct_game_played()
        self._update_diff_time_ratio()

        # Snapshot
        self._cache_initial_state()

    # ----- Derived Updates -----
    def _update_score_diff(self):
        offense = self.home if self.possession == 'home' else self.away
        defense = self.away if self.possession == 'home' else self.home
        self.score_diff = offense['score'] - defense['score']

    def _update_pct_game_played(self):
        self.pct_game_played = (
            (3600 - self.seconds_remaining) / 3600 
            if self.seconds_remaining is not None else 0
        )

    def _update_diff_time_ratio(self):
        self.diff_time_ratio = self.score_diff * np.exp(4 * self.pct_game_played)

    # ----- Setters (that also update derived) -----
    def set_possession(self, value: str):
        self.possession = value
        self._update_score_diff()
        self._update_diff_time_ratio()
        self.num_plays_on_drive = 0  # Reset drive

    def set_home_score(self, value: int):
        self.home['score'] = value
        self._update_score_diff()
        self._update_diff_time_ratio()

    def set_away_score(self, value: int):
        self.away['score'] = value
        self._update_score_diff()
        self._update_diff_time_ratio()

    def set_seconds_remaining(self, value: int):
        self.seconds_remaining = value
        self._update_pct_game_played()
        self._update_diff_time_ratio()

    def increment_play_count(self):
        self.num_plays_on_drive += 1

    # ----- Getters -----
    def get_possession(self): return self.possession
    def get_home_score(self): return self.home['score']
    def get_away_score(self): return self.away['score']
    def get_score_diff(self): return self.score_diff
    def get_pct_game_played(self): return self.pct_game_played
    def get_diff_time_ratio(self): return self.diff_time_ratio
    def get_play_count(self): return self.num_plays_on_drive
    def get_offense_is_home(self):
        return 0 if self.neutral_site else int(self.possession == 'home')

    # ----- Snapshot -----
    def _cache_initial_state(self):
        self.initial_state = {
            'possession': self.possession,
            'home': self.home.copy(),
            'away': self.away.copy(),
            'seconds_remaining': self.seconds_remaining,
            'down': self.down,
            'distance': self.distance,
            'yards_to_goal': self.yards_to_goal,
            'weather': self.weather.copy(),
            'elevation': self.elevation,
            'clock_rolling': self.clock_rolling,
            'neutral_site': self.neutral_site,
            'score_diff': self.score_diff,
            'pct_game_played': self.pct_game_played,
            'diff_time_ratio': self.diff_time_ratio,
            'num_plays_on_drive': self.num_plays_on_drive
        }

    def reset(self):
        for k, v in self.initial_state.items():
            setattr(self, k, v if not isinstance(v, dict) else v.copy())