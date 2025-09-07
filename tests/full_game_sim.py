from runner import simulate_full_game
from concurrent.futures import ProcessPoolExecutor

import time


def _run_single(config):
    simulate_full_game(**config)


def main():
    config = {
        "n_simulations": 1,
        "home_elo_rating": 2500,
        "away_elo_rating": 500,
        "temperature": 70.0,
        "wind_speed": 5.0,
        "precipitation": 0.0,
        "home_division": "FBS",
        "home_is_power_five": True,
        "away_division": "FBS",
        "away_is_power_five": False,
        "home_last12_total_fg_poe_gaussian": 0.5,
        "home_last12_longest_fg": 50.0,
        "home_last6_pass_to_rush_ratio": 0.6,
        "home_last6_offense_sacks_allowed_per_game": 1.5,
        "home_last6_defense_sacks_per_game": 2.0,
        "away_last12_total_fg_poe_gaussian": 0.4,
        "away_last12_longest_fg": 45.0,
        "away_last6_pass_to_rush_ratio": 0.5,
        "away_last6_offense_sacks_allowed_per_game": 1.2,
        "away_last6_defense_sacks_per_game": 1.8,
        "elevation": 500.0,
        "neutral_site": False,
    }

    s_time = time.time()
    n_jobs = 100
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(_run_single, [config] * n_jobs))
    e_time = time.time()
    print(f'Seconds: {e_time - s_time})')


if __name__ == "__main__":
    main()