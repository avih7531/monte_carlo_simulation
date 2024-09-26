import sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.ensemble import RandomForestRegressor
import time


class KNN:
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
    ):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, algorithm="auto"
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def test(self, X, y):
        return self.model.score(X, y)

    def __str__(self):
        return f"KNN with n_neighbors = {str(self.n_neighbors)}"


class LinearRegression:
    def __init__(self):
        self.model = SKLinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def test(self, X, y):
        return self.model.score(X, y)

    def __str__(self):
        return "LinearRegression"


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def test(self, X, y):
        return self.model.score(X, y)

    def __str__(self):
        return f"RandomForest with n_estimators = {str(self.n_estimators)}"


class Dataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path).drop(columns=["Simulation_ID"])

        self.X = self.data.drop(
            columns=[
                "Up-and-In_Call",
                "Up-and-Out_Call",
                "European_Lookback_Call",
                "Asian_Call",
                "Double_Barrier_Call",
            ]
        )

        self.train_x = self.X.iloc[: int(len(self.X) * 0.8)]
        self.train_y = self.data.iloc[: int(len(self.X) * 0.8)]
        self.val_x = self.X.iloc[int(len(self.X) * 0.8) : int(len(self.X) * 0.9)]
        self.val_y = self.data.iloc[int(len(self.X) * 0.8) : int(len(self.X) * 0.9)]
        self.test_x = self.X.iloc[int(len(self.X) * 0.9) : len(self.X)]
        self.test_y = self.data.iloc[int(len(self.X) * 0.9) : len(self.X)]


class EnsembleModel:
    def __init__(self, csv_path, models, target, verbose=True):
        self.models = models
        self.verbose = verbose
        if target not in [
            "Up-and-In_Call",
            "Up-and-Out_Call",
            "European_Lookback_Call",
            "Asian_Call",
            "Double_Barrier_Call",
        ]:
            raise ValueError("Invalid target")
        self.target = target
        if self.verbose:
            print(
                f"Creating ensemble model to target {self.target} with models {[str(model) for model in models]}"
            )
        self.data = Dataset(csv_path)
        if self.verbose:
            print(f"Data loaded from {csv_path}")

    def fit(self):
        if self.verbose:
            print("Fitting models")
        for model in self.models:
            time_begin = time.monotonic()
            if self.verbose:
                print(f"Fitting {str(model)}")
            model.fit(self.data.train_x, self.data.train_y[self.target])
            if self.verbose:
                print(
                    f"{str(model)} fitted in {round(time.monotonic() - time_begin, 5)} seconds"
                )

    def test(self):
        if self.verbose:
            print("Testing models")
        outputs = {}
        for model in self.models:
            time_begin = time.monotonic()
            outputs[str(model)] = [
                model.test(self.data.test_x, self.data.test_y[self.target]),
                time.monotonic() - time_begin,
            ]
            if self.verbose:
                print(
                    f"{str(model)}: {outputs[str(model)][0]} in {round(outputs[str(model)][1], 5)} seconds"
                )
        return outputs

    def __str__(self):
        return f"EnsembleModel with to target {self.target} with models {[str(model) for model in models]}"


if __name__ == "__main__":
    models = [
        LinearRegression(),
        KNN(),
        KNN(n_neighbors=10),
        KNN(n_neighbors=50),
        # RandomForest(),
        # RandomForest(n_estimators=200),
        RandomForest(n_estimators=50),
        RandomForest(n_estimators=10),
    ]

    Up_and_Out_Call = EnsembleModel(
        "/Users/orangoodman/dev/monte_carlo_simulation/monte_carlo_simulation_results_heston.csv",
        models,
        "Up-and-Out_Call",
    )
    Up_and_Out_Call.fit()
    Up_and_Out_Call.test()
    print("\n\n\n\n\n\n")
    Up_and_In_Call = EnsembleModel(
        "/Users/orangoodman/dev/monte_carlo_simulation/monte_carlo_simulation_results_heston.csv",
        [
            LinearRegression(),
            KNN(),
            KNN(n_neighbors=10),
            KNN(n_neighbors=50),
            RandomForest(n_estimators=50),
            RandomForest(n_estimators=10),
        ],
        "Up-and-In_Call",
    )
    Up_and_In_Call.fit()
    Up_and_In_Call.test()
    print("\n\n\n\n\n\n")
    European_Lookback_Call = EnsembleModel(
        "/Users/orangoodman/dev/monte_carlo_simulation/monte_carlo_simulation_results_heston.csv",
        [
            LinearRegression(),
            KNN(),
            KNN(n_neighbors=10),
            KNN(n_neighbors=50),
            RandomForest(n_estimators=50),
            RandomForest(n_estimators=10),
        ],
        "European_Lookback_Call",
    )
    European_Lookback_Call.fit()
    European_Lookback_Call.test()
    print("\n\n\n\n\n\n")
    Asian_Call = EnsembleModel(
        "/Users/orangoodman/dev/monte_carlo_simulation/monte_carlo_simulation_results_heston.csv",
        [
            LinearRegression(),
            KNN(),
            KNN(n_neighbors=10),
            KNN(n_neighbors=50),
            RandomForest(n_estimators=50),
            RandomForest(n_estimators=10),
        ],
        "Asian_Call",
    )
    Asian_Call.fit()
    Asian_Call.test()
    print("\n\n\n\n\n\n")
    Double_Barrier_Call = EnsembleModel(
        "/Users/orangoodman/dev/monte_carlo_simulation/monte_carlo_simulation_results_heston.csv",
        [
            LinearRegression(),
            KNN(),
            KNN(n_neighbors=10),
            KNN(n_neighbors=50),
            RandomForest(n_estimators=50),
            RandomForest(n_estimators=10),
        ],
        "Double_Barrier_Call",
    )
    Double_Barrier_Call.fit()
    Double_Barrier_Call.test()
