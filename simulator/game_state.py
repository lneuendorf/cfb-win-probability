class GameState(dict):
    def __init__(
        self,
        possession=None,
        home_score=None,
        home_elo_rating=None,
        home_timeouts=None,
        away_score=None,
        away_elo_rating=None,
        away_timeouts=None,
        seconds_remaining=None,
        down=None,
        distance=None,
        yards_to_goal=None,
        temperature=None,
        wind_speed=None,
        precipitation=None,
        elevation=None,
        home_division=None,
        home_is_power_five=None,
        away_division=None,
        away_is_power_five=None,
        clock_rolling=False
    ):
        super().__init__()
        self['possession'] = possession  # 'home' or 'away'
        self['home'] = {
            'score': home_score,
            'elo_rating': home_elo_rating,
            'timeouts': home_timeouts,
            'division': home_division,
            'is_power_five': home_is_power_five,
        }
        self['away'] = {
            'score': away_score,
            'elo_rating': away_elo_rating,
            'timeouts': away_timeouts,
            'division': away_division,
            'is_power_five': away_is_power_five,
        }
        self['seconds_remaining'] = seconds_remaining  # seconds remaining game
        self['down'] = down  # 1 to 4
        self['distance'] = distance  # yards to first down
        self['yards_to_goal'] = yards_to_goal  # yards to end zone
        self['weather'] = {
            'temperature': temperature,  # Fahrenheit
            'wind_speed': wind_speed,    # mph
            'precipitation': precipitation,  # inches of precipitation
        }
        self['elevation'] = elevation  # feet above sea level
        self['clock_rolling'] = clock_rolling

        self._cache_initial_state()
    
    def _cache_initial_state(self):
        self.initial_state = {
            'possession': self['possession'],
            'home': self['home'].copy(),
            'away': self['away'].copy(),
            'seconds_remaining': self['seconds_remaining'],
            'down': self['down'],
            'distance': self['distance'],
            'yards_to_goal': self['yards_to_goal'],
            'weather': self['weather'].copy(),
            'elevation': self['elevation'],
            'clock_rolling': self['clock_rolling'],
        }

    def reset(self):
        '''
        Resets the game state to the initial state.
        '''
        self['possession'] = self.initial_state['possession']
        self['home'] = self.initial_state['home'].copy()
        self['away'] = self.initial_state['away'].copy()
        self['seconds_remaining'] = self.initial_state['seconds_remaining']
        self['down'] = self.initial_state['down']
        self['distance'] = self.initial_state['distance']
        self['yards_to_goal'] = self.initial_state['yards_to_goal']
        self['weather'] = self.initial_state['weather'].copy()
        self['elevation'] = self.initial_state['elevation']
        self['clock_rolling'] = self.initial_state['clock_rolling']