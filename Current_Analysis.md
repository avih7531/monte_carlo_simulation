# Report: Analysis of Monte Carlo Simulation Results Using the Heston Model

## Overview:
The dataset represents the results of a Monte Carlo simulation utilizing the Heston model for option pricing. It includes different types of options, including both standard and exotic varieties, as well as risk sensitivities known as the Greeks. The simulation varies key financial parameters across different scenarios to study the effects of market conditions on these financial instruments.

---

## Key Variables:

### 1. Independent Variables:
These are the variables that define the market conditions and options setup for each simulation.

- **S0 (Initial Stock Price):** The price of the underlying asset (e.g., stock) at the beginning of the simulation.
- **K (Strike Price):** The predetermined price at which the option holder can buy or sell the underlying asset.
- **T (Time to Maturity):** The length of time (in years) until the option expires.
- **r (Risk-Free Rate):** The rate of return on a risk-free investment, which is used in option pricing models.
- **sigma_v (Volatility):** The volatility of the underlying asset, which reflects the degree of price variation expected over time.

### 2. Exotic Options:
These are non-standard options that come with specific features or conditions, often making them more complex than vanilla options.

- **Up-and-In Call:** A type of barrier option that becomes active only if the underlying asset price reaches or exceeds a predetermined barrier during the option's life.
- **Up-and-Out Call:** A barrier option that becomes void if the underlying asset price reaches or exceeds a certain barrier level.
- **European Lookback Call:** A path-dependent option where the payoff is based on the maximum or minimum price of the underlying asset during the option's lifetime, allowing the holder to "look back" at price history.
- **Asian Call:** Another path-dependent option where the payoff is based on the average price of the underlying asset over a specified period, rather than just the price at maturity.
- **Double Barrier Call:** A barrier option that has both upper and lower price limits. The option becomes active or inactive depending on whether the asset price crosses either of the barriers.

### 3. Greeks:
The Greeks represent sensitivities in option pricing that describe how the price of an option is expected to change in response to different factors.

- **Delta:** Measures how much the price of the option will change in relation to a $1 change in the price of the underlying asset. A Delta close to 1 indicates the option closely follows the stock price.
- **Gamma:** Describes how Delta changes as the underlying asset price changes. It reflects the curvature in the relationship between the asset price and the option's price.
- **Theta:** Known as time decay, Theta measures how much the option's value decreases as time to maturity decreases. A negative Theta indicates that the option loses value as time progresses.
- **Vega:** Measures the sensitivity of the option price to changes in volatility. Higher Vega values indicate that the option price is highly sensitive to volatility changes.
- **Rho:** Reflects the sensitivity of the option price to changes in the risk-free interest rate. A higher Rho suggests that the option price increases when interest rates rise.

---

## Key Insights:
- The dataset demonstrates how different market variables such as volatility, stock price, and time to maturity influence the price and behavior of various options.
- Exotic options like Up-and-In and Up-and-Out calls are highly sensitive to market movements, with many instances where these options are priced at zero, indicating they are out of the money or inactive in specific simulations.
- Path-dependent options, such as Asian and Lookback calls, benefit from averaging or maximum price movements over time, making them more valuable in highly volatile market conditions.
- The Greeks help quantify the risk associated with the options, particularly in terms of how sensitive they are to underlying price changes (Delta and Gamma), time decay (Theta), and volatility (Vega).

---

## Conclusion:
The data provides a comprehensive look at how various options react under different market conditions, offering valuable insights into the pricing behavior of both vanilla and exotic options. The Greeks further help traders understand the risks and sensitivities of these financial instruments.
