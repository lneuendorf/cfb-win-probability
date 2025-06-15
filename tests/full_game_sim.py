from runner import simulate_full_game

simulate_full_game(
    n_simulations=1,
    home_elo_rating=1500,
    away_elo_rating=1400,
    temperature=70.0,
    wind_speed=5.0,
    precipitation=0.0,
    home_division="FBS",
    home_is_power_five=True,
    away_division="FBS",
    away_is_power_five=False,
    elevation=500.0
)