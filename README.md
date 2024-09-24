# Independent Variables
| **Name**               | **Symbol**  | **Description**                                                                                                                                                    | **Range**                  |
|------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| Initial Stock Price     | `S0`        | The initial price of the stock or underlying asset at the start of the simulation.                                                                                  | 50 to 150                  |
| Strike Price            | `K`         | The price at which the option holder can buy or sell the underlying asset.                                                                                          | 70 to 130                  |
| Time to Maturity        | `T`         | The time (in years) until the option expires.                                                                                                                       | 0.1 to 2.0 (1 month to 2 years) |
| Risk-Free Interest Rate | `r`         | The risk-free interest rate used in the simulation. This represents the theoretical return of an investment with no risk.                                            | 0.01 to 0.05 (1% to 5%)    |
| Volatility of Volatility| `σ_v`       | The volatility of the variance process in the Heston model (also referred to as the volatility of volatility).                                                       | 0.1 to 0.3 (10% to 30%)    |
| Long-Term Mean Variance | `θ`         | The long-term mean level of variance (variance mean-reverts to this value in the Heston model).                                                                     | 0.2 (fixed)                |
| Speed of Mean Reversion | `κ`         | The rate at which volatility reverts to its long-term mean.                                                                                                         | 2.0 (fixed)                |
| Initial Variance        | `v0`        | The starting variance of the volatility process.                                                                                                                    | 0.04 (fixed)               |
| Correlation             | `ρ_v`       | The correlation between the stock price and the volatility process.                                                                                                | -0.5 (fixed)               |
| Upper Barrier           | N/A         | The upper price barrier at which an up-and-out option becomes worthless or an up-and-in option activates.                                                           | 110 (fixed)                |
| Lower Barrier           | N/A         | The lower price barrier used for double barrier options. The option becomes worthless if the price breaches this barrier.                                            | 90 (fixed)                 |
| Steps                   | `steps`     | The number of time steps (days in a year) for the stock price simulation.                                                                                           | 252 (fixed)                |
| Number of Iterations    | `N_iterations` | The number of iterations in each Monte Carlo simulation.                                                                                                           | 1000 (fixed)               |
| Number of Simulations   | `simulations` | The number of Monte Carlo simulations to run (each with N_iterations).                                                                                              | 10 (adjustable)            |

# Dependent Variables

| **Name**                 | **Symbol**  | **Description**                                                                                                                       | **Type**            |
|--------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| Up-and-In Call Payoff     | N/A         | Payoff of the up-and-in barrier option. The option becomes active only if the underlying asset price breaches the upper barrier.        | Payoff (numeric)    |
| Up-and-Out Call Payoff    | N/A         | Payoff of the up-and-out barrier option. The option becomes worthless if the underlying asset price breaches the upper barrier.         | Payoff (numeric)    |
| European Lookback Call    | N/A         | Payoff based on the maximum price of the underlying asset during the option's life.                                                     | Payoff (numeric)    |
| Asian Call Payoff         | N/A         | Payoff based on the average price of the underlying asset over the option's life.                                                       | Payoff (numeric)    |
| Double Barrier Call Payoff| N/A         | Payoff of the option if the asset price remains within both the upper and lower barriers throughout the option's life.                   | Payoff (numeric)    |
| Delta                    | Δ           | Sensitivity of the option price to small changes in the underlying asset's price.                                                       | Greek (numeric)     |
| Gamma                    | Γ           | Rate of change of Delta relative to changes in the underlying asset's price.                                                            | Greek (numeric)     |
| Theta                    | Θ           | Sensitivity of the option price to the passage of time (time decay).                                                                    | Greek (numeric)     |
| Vega                     | V           | Sensitivity of the option price to changes in the volatility of the underlying asset.                                                   | Greek (numeric)     |
| Rho                      | ρ           | Sensitivity of the option price to changes in the risk-free interest rate.                                                              | Greek (numeric)     |
