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
    ):
        super().__init__()
        self['possession'] = possession  # 'home' or 'away'
        self['home'] = {
            'score': home_score,
            'elo_rating': home_elo_rating,
            'timeouts': home_timeouts,
        }
        self['away'] = {
            'score': away_score,
            'elo_rating': away_elo_rating,
            'timeouts': away_timeouts,
        }
        self['seconds_remaining'] = seconds_remaining  # seconds remaining game
        self['down'] = down  # 1 to 4
        self['distance'] = distance  # yards to first down
        self['yards_to_goal'] = yards_to_goal  # yards to end zone
        self['weather'] = {
            'temperature': temperature,  # degrees Fahrenheit
            'wind_speed': wind_speed,    # miles per hour
            'precipitation': precipitation,  # inches of rain
        }
        self['elevation'] = elevation  # feet above sea level

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