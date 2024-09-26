import csv
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

# Set up parameter ranges
S0_range = (50, 150)  # Initial stock price range
K_range = (70, 130)  # Strike price range
T_range = (0.1, 2.0)  # Time to maturity in years (1 month to 2 years)
r_range = (0.01, 0.05)  # Risk-free rate (1% to 5%)
sigma_v_range = (0.1, 0.3)  # Volatility of volatility (10% to 30%)

# Heston model parameters
kappa = 2.0  # Speed of mean reversion
theta = 0.2  # Long-term mean of variance
rho_v = -0.5  # Correlation between stock price and volatility
v0 = 0.04  # Initial variance (i.e., initial volatility squared)

# Monte Carlo Parameters
N_iterations = 1000  # Fixed number of iterations per simulation
steps = 252  # Number of time steps (days in a year)
upper_barrier = 110  # Upper barrier for up-and-in and up-and-out options
lower_barrier = 90  # Lower barrier for double barrier options


def simulate_heston_gbm(S0, r, v0, T, steps, kappa, theta, sigma_v, rho_v):
    """
    Vectorized simulation of stock price paths using the Heston model.
    """
    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    prices = np.zeros(steps)
    volatilities = np.zeros(steps)
    prices[0] = S0
    volatilities[0] = v0

    Z1 = np.random.standard_normal(steps - 1)  # Wiener process for price
    Z2 = np.random.standard_normal(steps - 1)  # Wiener process for volatility
    Z2 = rho_v * Z1 + np.sqrt(1 - rho_v**2) * Z2  # Correlated Z2

    # Precompute constants
    kappa_dt = kappa * dt
    sigma_v_sqrt_dt = sigma_v * sqrt_dt

    for t in range(1, steps):
        # Update volatility using vectorized approach
        volatilities[t] = np.maximum(
            volatilities[t - 1]
            + kappa_dt * (theta - volatilities[t - 1])
            + sigma_v_sqrt_dt * np.sqrt(volatilities[t - 1]) * Z2[t - 1],
            0,
        )

        # Update price using vectorized approach
        prices[t] = prices[t - 1] * np.exp(
            (r - 0.5 * volatilities[t - 1]) * dt
            + np.sqrt(volatilities[t - 1]) * sqrt_dt * Z1[t - 1]
        )

    return prices, np.sqrt(volatilities)


def calculate_exotic_payoffs(price_path, K, upper_barrier, lower_barrier, r, T):
    """
    Vectorized calculation of exotic option payoffs.
    """
    ST = price_path[-1]
    avg_price = np.mean(price_path)
    min_price = np.min(price_path)
    max_price = np.max(price_path)

    # Precompute payoffs for different options
    up_in_call = max(ST - K, 0) if max_price >= upper_barrier else 0
    up_out_call = max(ST - K, 0) if max_price < upper_barrier else 0
    lookback_call = max(max_price - K, 0)
    asian_call = max(avg_price - K, 0)
    double_barrier_call = (
        max(ST - K, 0)
        if (min_price > lower_barrier and max_price < upper_barrier)
        else 0
    )

    return up_in_call, up_out_call, lookback_call, asian_call, double_barrier_call


def calculate_greeks(price_path, vol_path, S0, K, r, T):
    """
    Vectorized calculation of the Greeks using Black-Scholes formulas.
    """
    ST = price_path[-1]
    sigma = np.mean(vol_path)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Precompute Greeks using vectorized operations
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, theta, vega, rho


def run_monte_carlo(simulations, N_iterations, steps, theta):
    """
    Run Monte Carlo simulations in parallel and compute option payoffs and Greeks.
    """

    def run_single_simulation(sim_id, theta):
        S0 = random.uniform(*S0_range)
        K = random.uniform(*K_range)
        T = random.uniform(*T_range)
        r = random.uniform(*r_range)
        sigma_v = random.uniform(*sigma_v_range)

        # Initialize sums for averaging results
        up_in_call_sum = 0
        up_out_call_sum = 0
        lookback_call_sum = 0
        asian_call_sum = 0
        double_barrier_call_sum = 0
        delta_sum = 0
        gamma_sum = 0
        theta_greek_sum = 0
        vega_sum = 0
        rho_sum = 0

        # Loop over iterations
        for i in range(N_iterations):
            price_path, vol_path = simulate_heston_gbm(
                S0, r, v0, T, steps, kappa, theta, sigma_v, rho_v
            )
            # Calculate payoffs and Greeks
            up_in_call, up_out_call, lookback_call, asian_call, double_barrier_call = (
                calculate_exotic_payoffs(
                    price_path, K, upper_barrier, lower_barrier, r, T
                )
            )
            delta, gamma, theta_greek, vega, rho = calculate_greeks(
                price_path, vol_path, S0, K, r, T
            )

            # Accumulate results
            up_in_call_sum += up_in_call
            up_out_call_sum += up_out_call
            lookback_call_sum += lookback_call
            asian_call_sum += asian_call
            double_barrier_call_sum += double_barrier_call
            delta_sum += delta
            gamma_sum += gamma
            theta_greek_sum += theta_greek
            vega_sum += vega
            rho_sum += rho

        # Store averages
        return {
            "Simulation_ID": sim_id + 1,
            "S0": S0,
            "K": K,
            "T": T,
            "r": r,
            "sigma_v": sigma_v,
            "Up-and-In_Call": up_in_call_sum / N_iterations,
            "Up-and-Out_Call": up_out_call_sum / N_iterations,
            "European_Lookback_Call": lookback_call_sum / N_iterations,
            "Asian_Call": asian_call_sum / N_iterations,
            "Double_Barrier_Call": double_barrier_call_sum / N_iterations,
            "Delta": delta_sum / N_iterations,
            "Gamma": gamma_sum / N_iterations,
            "Theta": theta_greek_sum / N_iterations,
            "Vega": vega_sum / N_iterations,
            "Rho": rho_sum / N_iterations,
        }

    # Parallelize the simulations using joblib
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, theta) for i in range(simulations)
    )

    return pd.DataFrame(results)


def save_to_csv(df, filename):
    """
    Save the dataframe containing the simulation results to a CSV file.
    """
    df.to_csv(filename, index=False, encoding="utf-8", sep=",", float_format="%.6f")


# Number of Monte Carlo simulations to run
simulations = 100000  # Adjust this as needed

# Run the Monte Carlo simulations
df_results = run_monte_carlo(simulations, N_iterations, steps, theta)

# Save the results to CSV
save_to_csv(df_results, "monte_carlo_simulation_results_heston.csv")

print(
    "Monte Carlo simulations with the Heston model complete. Results saved to 'monte_carlo_simulation_results_heston.csv'."
)
