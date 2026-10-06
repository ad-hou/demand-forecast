"""Lance tous les modeles, journalise dans MLflow, sauvegarde les resultats."""
import mlflow
import pandas as pd

from src.backtest import HORIZON, INITIAL_TRAIN, STEP, rolling_origin, score
from src.models import ets, ets_prophet, lightgbm, naive, prophet, sarima, seasonal_naive

MODELS = {
    "naive": naive,
    "seasonal_naive_7": seasonal_naive(7),
    "seasonal_naive_365": seasonal_naive(365),
    "ets_weekly": ets,
    "sarima_weekly": sarima,
    "prophet": prophet,
    "lightgbm": lightgbm,
    "ets_prophet": ets_prophet,
}


def main():
    k = pd.read_parquet("data/daily.parquet")["kwh"]
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("demand-forecast")
    all_results = []
    for name, f in MODELS.items():
        res = rolling_origin(k, f)
        s = score(res)
        by_h = (res.dropna(subset=["y", "pred"])
                   .assign(err=lambda d: (d["y"] - d["pred"]).abs())
                   .groupby("h")["err"].mean())
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"initial_train": INITIAL_TRAIN,
                               "horizon": HORIZON, "step": STEP})
            mlflow.log_metrics({"mae": s["mae"], "rmse": s["rmse"], "smape": s["smape"]})
            for h, v in by_h.items():
                mlflow.log_metric("mae_by_horizon", float(v), step=int(h))
        all_results.append(res.assign(model=name))
        print(f"{name:<22} MAE {s['mae']:.2f}")
    pd.concat(all_results).to_parquet("data/backtest_results.parquet")
    print("Sauvegarde : data/backtest_results.parquet")


if __name__ == "__main__":
    main()
