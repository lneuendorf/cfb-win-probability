import numpy as np 

from simulator.game_state import GameState
from simulator.simulate import Simulator

def simulate_full_game(
    home_elo_rating,
    away_elo_rating,
    temperature,
    wind_speed,
    precipitation,
    elevation,
    home_division,
    home_is_power_five,
    away_division,
    away_is_power_five,
    n_simulations=1000,
) -> np.ndarray:
    """
    Simulates a full game of college football based on the provided game state 
    arameters.

    Args:
        home_elo_rating (float): ELO rating of the home team.
        home_timeouts (int): Number of timeouts remaining for the home team.
        away_score (int): Current score of the away team.
        away_elo_rating (float): ELO rating of the away team.
        away_timeouts (int): Number of timeouts remaining for the away team.
        seconds_remaining (int): Seconds remaining in game.
        down (int): Current down (1 to 4).
        distance (int): Yards needed for a first down.
        yards_to_goal (int): Yards to the end zone from current position.
        temperature (float): Temperature in degrees Fahrenheit.
        wind_speed (float): Wind speed in miles per hour.
        precipitation (float): Precipitation in inches.
        elevation (float): Elevation in feet above sea level.
        home_division (str): Division of the home team (e.g., "FBS", "FCS").
        home_is_power_five (bool): Whether the home team is a Power Five team.
        away_division (str): Division of the away team (e.g., "FBS", "FCS").
        away_is_power_five (bool): Whether the away team is a Power Five team.
        n_simulations (int): Number of simulations to run.

    
    Returns:
        np.ndarray: A 2D array of win, tie, loss probabilities for the home team
            as np.array([win_prob, tie_prob, loss_prob]).
    """
    game_state = GameState(
        possession=None, # game not started -> coin flip decides possession
        home_score=0,
        home_elo_rating=home_elo_rating,
        home_timeouts=3,
        away_score=0,
        away_elo_rating=away_elo_rating,
        away_timeouts=3,
        seconds_remaining=3600,  # 60 minutes * 60 seconds
        down=None,  # Not applicable at game start
        distance=None,  # Not applicable at game start
        yards_to_goal=None,  # Not applicable at game start
        temperature=temperature,
        wind_speed=wind_speed,
        precipitation=precipitation,
        elevation=elevation,
        home_division=home_division,
        home_is_power_five=home_is_power_five,
        away_division=away_division,
        away_is_power_five=away_is_power_five,
        clock_rolling=False  # Clock starts rolling after kickoff
    )
    
    outcomes = []
    for _ in range(n_simulations):
        simulator = Simulator(game_state)
        outcomes.append(simulator.run())
        game_state.reset()

    outcomes = np.array(outcomes)
    win_prob = np.mean(outcomes == 1)
    tie_prob = np.mean(outcomes == 0)
    loss_prob = np.mean(outcomes == -1)
    return np.array([win_prob, tie_prob, loss_prob])
    

def simulate_from_4th_down():
    pass