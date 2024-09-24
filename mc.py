import csv
import random

import numpy as np
import pandas as pd
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
    Simulate a stock price path using the Heston model for stochastic volatility.

    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        v0 (float): Initial variance.
        T (float): Time to maturity.
        steps (int): Number of time steps.
        kappa (float): Speed of mean reversion.
        theta (float): Long-term mean of variance.
        sigma_v (float): Volatility of volatility.
        rho_v (float): Correlation between stock price and volatility.

    Returns:
        tuple: Simulated price path and volatility path.
    """
    dt = T / steps
    prices = np.zeros(steps)
    volatilities = np.zeros(steps)
    prices[0] = S0
    volatilities[0] = v0  # Start with initial variance

    for t in range(1, steps):
        Z1 = np.random.standard_normal()  # Wiener process for price
        Z2 = np.random.standard_normal()  # Wiener process for volatility
        Z2 = rho_v * Z1 + np.sqrt(1 - rho_v**2) * Z2  # Correlated Z2

        # Heston stochastic volatility process
        volatilities[t] = max(
            volatilities[t - 1]
            + kappa * (theta - volatilities[t - 1]) * dt
            + sigma_v * np.sqrt(volatilities[t - 1]) * np.sqrt(dt) * Z2,
            0,
        )

        # Geometric Brownian Motion with stochastic volatility
        prices[t] = prices[t - 1] * np.exp(
            (r - 0.5 * volatilities[t - 1]) * dt
            + np.sqrt(volatilities[t - 1]) * np.sqrt(dt) * Z1
        )

    return prices, np.sqrt(volatilities)  # Returning price path and volatilities


def calculate_exotic_payoffs(price_path, K, upper_barrier, lower_barrier, r, T):
    """
    Calculate the payoffs for exotic options including Up-and-In, Up-and-Out, Lookback,
    Asian, and Double Barrier options.

    Args:
        price_path (np.array): Simulated stock price path.
        K (float): Strike price.
        upper_barrier (float): Upper barrier level for knock-in/knock-out options.
        lower_barrier (float): Lower barrier level for double barrier options.
        r (float): Risk-free interest rate.
        T (float): Time to maturity (years).

    Returns:
        tuple: Payoffs for (Up-and-In Call, Up-and-Out Call, European Lookback Call, Asian Call, Double Barrier Call)
    """
    ST = price_path[-1]
    avg_price = np.mean(price_path)
    min_price = np.min(price_path)
    max_price = np.max(price_path)

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
    Calculate the relevant Greeks (Delta, Gamma, Theta, Vega, Rho) for options using
    the Black-Scholes model with the average volatility from the Heston model.

    Args:
        price_path (np.array): Simulated stock price path.
        vol_path (np.array): Simulated volatility path.
        S0 (float): Initial stock price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time to maturity (years).

    Returns:
        tuple: (Delta, Gamma, Theta, Vega, Rho)
    """
    ST = price_path[-1]
    sigma = np.mean(vol_path)  # Use average volatility in the Black-Scholes formulas

    # Black-Scholes d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Greeks dynamically
    delta = norm.cdf(d1)  # Delta for a European call
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))  # Gamma formula
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * norm.cdf(
        d2
    )  # Theta for a call option
    vega = S0 * norm.pdf(d1) * np.sqrt(T)  # Vega formula
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)  # Rho formula

    return delta, gamma, theta, vega, rho


def run_monte_carlo(simulations, N_iterations, steps, theta):
    """
    Run multiple Monte Carlo simulations with the Heston model and calculate exotic
    option payoffs and Greeks.

    Args:
        simulations (int): Number of Monte Carlo simulations to run.
        N_iterations (int): Number of iterations for each simulation.
        steps (int): Number of time steps for each simulation.
        theta (float): Long-term mean of variance.

    Returns:
        pd.DataFrame: Dataframe containing results of all simulations.
    """
    results = []

    for sim in range(simulations):
        S0 = random.uniform(*S0_range)
        K = random.uniform(*K_range)
        T = random.uniform(*T_range)
        r = random.uniform(*r_range)
        sigma_v = random.uniform(*sigma_v_range)  # Volatility of volatility

        # Initialize accumulators for the payoffs and Greeks
        up_in_call_sum = 0
        up_out_call_sum = 0
        lookback_call_sum = 0
        asian_call_sum = 0
        double_barrier_call_sum = 0
        delta_sum = 0
        gamma_sum = 0
        theta_greek_sum = 0  # Changed variable name here
        vega_sum = 0
        rho_sum = 0

        # Perform N_iterations for the current Monte Carlo
        for i in range(N_iterations):
            price_path, vol_path = simulate_heston_gbm(
                S0, r, v0, T, steps, kappa, theta, sigma_v, rho_v
            )

            # Calculate exotic option payoffs
            up_in_call, up_out_call, lookback_call, asian_call, double_barrier_call = (
                calculate_exotic_payoffs(
                    price_path, K, upper_barrier, lower_barrier, r, T
                )
            )

            # Calculate Greeks
            delta, gamma, theta, vega, rho = calculate_greeks(
                price_path, vol_path, S0, K, r, T
            )

            # Accumulate the results
            up_in_call_sum += up_in_call
            up_out_call_sum += up_out_call
            lookback_call_sum += lookback_call
            asian_call_sum += asian_call
            double_barrier_call_sum += double_barrier_call
            delta_sum += delta
            gamma_sum += gamma
            theta_greek_sum += theta  # Use the renamed variable here
            vega_sum += vega
            rho_sum += rho

        # Calculate the averages for the current simulation
        results.append(
            {
                "Simulation_ID": sim + 1,
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
                "Theta": theta_greek_sum / N_iterations,  # Renamed here as well
                "Vega": vega_sum / N_iterations,
                "Rho": rho_sum / N_iterations,
            }
        )

    return pd.DataFrame(results)


def save_to_csv(df, filename):
    """
    Save the dataframe containing the simulation results to a CSV file.

    Args:
        df (pd.DataFrame): Dataframe containing the simulation results.
        filename (str): The name of the CSV file to save the data.
    """
    columns = [
        "Simulation_ID",
        "S0",
        "K",
        "T",
        "r",
        "sigma_v",
        "Up-and-In_Call",
        "Up-and-Out_Call",
        "European_Lookback_Call",
        "Asian_Call",
        "Double_Barrier_Call",
        "Delta",
        "Gamma",
        "Theta",
        "Vega",
        "Rho",
    ]
    df.to_csv(
        filename,
        columns=columns,
        index=False,
        encoding="utf-8",
        sep=",",
        float_format="%.6f",
    )


# User Input: Number of Monte Carlo simulations to run
simulations = 10  # Adjust this as needed

# Run the Monte Carlo simulations
df_results = run_monte_carlo(simulations, N_iterations, steps, theta)

# Save the results to CSV
save_to_csv(df_results, "monte_carlo_simulation_results_heston.csv")

print(
    f"Monte Carlo simulations with the Heston model complete. Results saved to 'monte_carlo_simulation_results_heston.csv'."
)
