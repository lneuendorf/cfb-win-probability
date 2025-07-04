from runner import simulate_full_game

simulate_full_game(
    n_simulations=1,
    home_elo_rating=2500,
    away_elo_rating=500,
    temperature=70.0,
    wind_speed=5.0,
    precipitation=0.0,
    home_division="FBS",
    home_is_power_five=True,
    away_division="FBS",
    away_is_power_five=False,
    home_last12_total_fg_poe_gaussian=0.5,
    home_last12_longest_fg=50.0,
    home_last6_pass_to_rush_ratio=0.6,
    away_last12_total_fg_poe_gaussian=0.4,
    away_last12_longest_fg=45.0,
    away_last6_pass_to_rush_ratio=0.5,
    elevation=500.0,
    neutral_site=False,
)