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
        neutral_site: bool = False,
        clock_rolling: bool = False,
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
        if self.possession != value:
            self.possession = value
            self._update_score_diff()
            self._update_diff_time_ratio()
            self.num_plays_on_drive = 0  # Reset drive
    def switch_possession(self):
        if self.possession == 'home':
            self.possession = 'away'
        else:
            self.possession = 'home'
        self._update_score_diff()
        self._update_diff_time_ratio()
        self.num_plays_on_drive = 0
    def set_down(self, value: int):
        self.down = value
    def set_distance(self, value: int):
        self.distance = value
    def set_yards_to_goal(self, value: int):
        self.yards_to_goal = value
    def stop_clock(self):
        self.clock_rolling = False
    def start_clock(self):
        self.clock_rolling = True
    def decrement_offense_timeouts(self):
        if self.get_offense_timeouts() > 0:
            if self.possession == 'home':
                self.home['timeouts'] -= 1
            else:
                self.away['timeouts'] -= 1
            self.clock_rolling = False
    def decrement_defense_timeouts(self):
        if self.get_defense_timeouts() > 0:
            if self.possession == 'home':
                self.away['timeouts'] -= 1
            else:
                self.home['timeouts'] -= 1
            self.clock_rolling = False
    def increment_offense_score(self, value: int):
        self.home['score'] += value if self.possession == 'home' else 0
        self.away['score'] += value if self.possession == 'away' else 0
        self._update_score_diff()
        self._update_diff_time_ratio()
    def increment_defense_score(self, value: int):
        self.away['score'] += value if self.possession == 'home' else 0
        self.home['score'] += value if self.possession == 'away' else 0
        self._update_score_diff()
        self._update_diff_time_ratio()
    def increment_play_count(self):
        self.num_plays_on_drive += 1
    def decrement_seconds_remaining(self, value: int):
        if self.seconds_remaining is not None:
            self.seconds_remaining = max(0, self.seconds_remaining - value)
            self._update_pct_game_played()
            self._update_diff_time_ratio()

    # ----- Getters -----
    def get_possession(self): return self.possession
    def get_score_diff(self): return self.score_diff
    def get_pct_game_played(self): return self.pct_game_played
    def get_diff_time_ratio(self): return self.diff_time_ratio
    def get_play_count(self): return self.num_plays_on_drive
    def get_offense_is_home(self):
        return int(np.select(
            [self.neutral_site, self.possession == 'home'],
            [0, 1],
            default=-1
        ))
    def get_offense_timeouts(self):
        return (
            self.home['timeouts'] 
            if self.possession == 'home' 
            else self.away['timeouts']
        )
    def get_offense_score(self):
        return (
            self.home['score'] 
            if self.possession == 'home' 
            else self.away['score']
        )
    def offense_is_power_five(self):
        return int(
            self.home['is_power_five'] 
            if self.possession == 'home' 
            else self.away['is_power_five']
        )
    def get_offense_division(self):
        return (
            self.home['division'] 
            if self.possession == 'home' 
            else self.away['division']
        )
    def get_offense_elo_rating(self):
        return (
            self.home['elo_rating'] 
            if self.possession == 'home' 
            else self.away['elo_rating']
        )
    def get_defense_score(self):
        return (
            self.away['score'] 
            if self.possession == 'home' 
            else self.home['score']
        )
    def get_defense_timeouts(self):
        return (
            self.away['timeouts'] 
            if self.possession == 'home' 
            else self.home['timeouts']
        )
    def get_defense_is_home(self):
        return int(np.select(
            [self.neutral_site, self.possession == 'away'],
            [0, 1],
            default=-1
        ))
    def defense_is_power_five(self):
        return int(
            self.away['is_power_five'] 
            if self.possession == 'home' 
            else self.home['is_power_five']
        )
    def get_defense_division(self):
        return (
            self.away['division'] 
            if self.possession == 'home' 
            else self.home['division']
        )
    def get_defense_elo_rating(self):
        return (
            self.away['elo_rating'] 
            if self.possession == 'home' 
            else self.home['elo_rating']
        )
    def get_yards_to_goal(self): return self.yards_to_goal
    def get_seconds_remaining(self): return self.seconds_remaining
    def get_down(self): return self.down
    def get_distance(self): return self.distance
    def get_temperature(self): return self.weather['temperature']
    def get_wind_speed(self): return self.weather['wind_speed']
    def get_precipitation(self): return self.weather['precipitation']
    def get_elevation(self): return self.elevation
    def clock_is_rolling(self): return int(self.clock_rolling)

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