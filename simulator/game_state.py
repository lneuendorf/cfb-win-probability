import numpy as np

class GameState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # Initialize core fields (without triggering derived calculations)
        super().__setitem__('home', {
            'score': kwargs.get('home_score'),
            'elo_rating': kwargs.get('home_elo_rating'),
            'timeouts': kwargs.get('home_timeouts'),
            'division': kwargs.get('home_division'),
            'is_power_five': kwargs.get('home_is_power_five'),
        })
        super().__setitem__('away', {
            'score': kwargs.get('away_score'),
            'elo_rating': kwargs.get('away_elo_rating'),
            'timeouts': kwargs.get('away_timeouts'),
            'division': kwargs.get('away_division'),
            'is_power_five': kwargs.get('away_is_power_five'),
        })

        # Initialize derived fields (empty at first to avoid KeyError)
        self._derived = {
            'score_diff': 0,
            'pct_game_played': 0,
            'diff_time_ratio': 0
        }
        
        # Now set possession (will safely trigger derived calculations)
        self['possession'] = kwargs.get('possession')  # 'home' or 'away'
        
        # Initialize remaining fields
        self['seconds_remaining'] = kwargs.get('seconds_remaining')
        self['down'] = kwargs.get('down')
        self['distance'] = kwargs.get('distance')
        self['yards_to_goal'] = kwargs.get('yards_to_goal')
        self['weather'] = {
            'temperature': kwargs.get('temperature'),
            'wind_speed': kwargs.get('wind_speed'),
            'precipitation': kwargs.get('precipitation'),
        }
        self['elevation'] = kwargs.get('elevation')
        self['clock_rolling'] = kwargs.get('clock_rolling', False)
        
        # Update derived fields now that all dependencies are set
        self._derived = {
            'score_diff': self._calculate_score_diff(),
            'pct_game_played': self._calculate_pct_game_played(),
            'diff_time_ratio': self._calculate_diff_time_ratio()
        }
        
        self._cache_initial_state()

    def __setitem__(self, key, value):
        """Override setitem to update derived fields when dependencies change"""
        super().__setitem__(key, value)
        
        # Update score_diff if possession or scores change
        if key in ['possession', 'home', 'away']:
            self._derived['score_diff'] = self._calculate_score_diff()
            self._derived['diff_time_ratio'] = self._calculate_diff_time_ratio()
        
        # Update pct_game_played when time changes
        elif key == 'seconds_remaining':
            self._derived['pct_game_played'] = self._calculate_pct_game_played()
            self._derived['diff_time_ratio'] = self._calculate_diff_time_ratio()

    def __getitem__(self, key):
        """Override getitem to check derived fields first"""
        if key in self._derived:
            return self._derived[key]
        return super().__getitem__(key)

    def _calculate_score_diff(self):
        """score_diff = offense_score - defense_score"""
        offense = self['home'] if self['possession'] == 'home' else self['away']
        defense = self['away'] if self['possession'] == 'home' else self['home']
        return offense['score'] - defense['score']

    def _calculate_pct_game_played(self):
        """(3600 - seconds_remaining) / 3600"""
        return (
            (3600 - self['seconds_remaining']) / 3600 
            if self['seconds_remaining'] is not None 
            else 0
        )

    def _calculate_diff_time_ratio(self):
        """score_diff * exp(4 * pct_game_played)"""
        return self._derived['score_diff'] * np.exp(4 * self._derived['pct_game_played'])

    def _cache_initial_state(self):
        """Deep copy of all state including derived fields"""
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
            '_derived': self._derived.copy()
        }

    def reset(self):
        """Reset to initial state including derived fields"""
        for k, v in self.initial_state.items():
            if k == '_derived':
                self._derived = v.copy()
            else:
                self[k] = v