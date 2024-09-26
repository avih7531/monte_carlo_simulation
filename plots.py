import pandas as pd
import seaborn as sns

df = pd.read_csv("monte_carlo_simulation_results_heston.csv")

sns.histplot(df["European_Lookback_Call"], log_scale=2, binrange=(0.5, 5))
import matplotlib.pyplot as plt

plt.show()
