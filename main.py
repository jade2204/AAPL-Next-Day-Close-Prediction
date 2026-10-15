import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid")


DATA_FILE = Path("AAPL_2006-01-01_to_2018-01-01.csv")
if not DATA_FILE.exists():  # allow running from the project root as well
    DATA_FILE = Path("1. Linear Regression") / DATA_FILE

MODEL_FILE = Path("aapl_next_close_linear_pipeline.joblib")
METADATA_FILE = Path("model_metadata.json")
MODEL_VERSION = "1.0.0"

TARGET = "Target_Next_Close"
CANDIDATE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Range_pct", "Open_Close_pct",
    "MA_5", "MA_20", "Volatility_5", "Volume_MA_5", "Volume_change",
]
PRICE_COLS = ["Open", "High", "Low", "Close"]
TRAIN_RATIO = 0.8
N_SPLITS = 5
RANDOM_STATE = 42

#Data loading and cleaning
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    print("Missing values per column:")
    print(df.isna().sum())

    # OHLC prices and volume must be strictly positive.
    valid = (df[PRICE_COLS] > 0).all(axis=1) & (df["Volume"] > 0)
    df = df.loc[valid].dropna(subset=PRICE_COLS + ["Volume"]).reset_index(drop=True)

    print(f"Period: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Sessions after cleaning: {len(df):,}")
    return df


#EDA plots
def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(df["Date"], df["Close"], color="royalblue", linewidth=1.4)
    axes[0].set_title("AAPL closing price over time")
    axes[0].set_ylabel("Close (USD)")
    axes[1].bar(df["Date"], df["Volume"], color="slategray", width=2)
    axes[1].set_title("Trading volume")
    axes[1].set_ylabel("Volume")
    axes[1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 5))
    sns.histplot(df["Close"], bins=40, kde=True, color="royalblue")
    plt.title("Distribution of AAPL closing price")
    plt.xlabel("Close (USD)")
    plt.show()


def detect_outliers(dataframe: pd.DataFrame, column: str):
    q1, q3 = dataframe[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (dataframe[column] < lower) | (dataframe[column] > upper)
    return dataframe[mask], lower, upper


def plot_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df[CANDIDATE_FEATURES + [TARGET]].corr(),
        cmap="coolwarm", center=0, annot=True, fmt=".2f",
    )
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()


#Feature engineering
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    #Create features using only information up to day t; target is Close at t+1.
    out = df.copy()

    # Momentum and intraday structure
    out["Return_1d"] = out["Close"].pct_change()
    out["Range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["Open_Close_pct"] = (out["Close"] - out["Open"]) / out["Open"]

    # Trend and volatility (past windows only)
    out["MA_5"] = out["Close"].rolling(5).mean()
    out["MA_20"] = out["Close"].rolling(20).mean()
    out["Volatility_5"] = out["Return_1d"].rolling(5).std()

    # Volume behavior
    out["Volume_MA_5"] = out["Volume"].rolling(5).mean()
    out["Volume_change"] = out["Volume"].pct_change()

    # shift(-1) is allowed ONLY for the target: row t holds Close of row t+1.
    out[TARGET] = out["Close"].shift(-1)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=CANDIDATE_FEATURES + [TARGET]).reset_index(drop=True)

    # Sanity check: target alignment
    first_date = out.loc[0, "Date"]
    idx = df.index[df["Date"].eq(first_date)][0]
    assert np.isclose(out.loc[0, TARGET], df.loc[idx + 1, "Close"]), "Target misaligned."
    assert np.isfinite(out[CANDIDATE_FEATURES + [TARGET]].to_numpy()).all()
    return out


#Time-based train/test split
def time_split(X, y, dates, train_ratio=TRAIN_RATIO):
    split_index = int(len(X) * train_ratio)
    return (
        X.iloc[:split_index], X.iloc[split_index:],
        y.iloc[:split_index], y.iloc[split_index:],
        dates.iloc[:split_index], dates.iloc[split_index:],
    )


def plot_split(dates_train, y_train, dates_test, y_test) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(dates_train, y_train, label="Train", color="steelblue")
    plt.plot(dates_test, y_test, label="Test", color="darkorange")
    plt.axvline(dates_test.iloc[0], color="black", linestyle="--", label="Split point")
    plt.title("Chronological train/test split")
    plt.ylabel("Next-day close (USD)")
    plt.legend()
    plt.show()


#Feature validation and selection (train data only)
def make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


def validate_features(X_train: pd.DataFrame) -> pd.DataFrame:
    quality = pd.DataFrame({
        "missing_count": X_train.isna().sum(),
        "infinite_count": np.isinf(X_train).sum(),
        "unique_count": X_train.nunique(),
        "variance": X_train.var(),
    })
    quality["is_valid"] = (
        (quality["missing_count"] == 0)
        & (quality["infinite_count"] == 0)
        & (quality["unique_count"] > 1)
    )
    return quality.sort_values("variance")


def mutual_information(X_train, y_train) -> pd.Series:
    values = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)
    return pd.Series(values, index=X_train.columns).sort_values(ascending=False)


def select_features(X_train, y_train, mi_scores):
    selector = RFECV(
        estimator=make_pipeline(),
        step=1,
        min_features_to_select=3,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
        importance_getter="named_steps.model.coef_",
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()

    ranking = pd.DataFrame({
        "Feature": X_train.columns,
        "Selected": selector.support_,
        "RFECV rank": selector.ranking_,
        "Mutual Information": mi_scores.reindex(X_train.columns).to_numpy(),
    }).sort_values(
        ["Selected", "RFECV rank", "Mutual Information"],
        ascending=[False, True, False],
    )
    return selected, ranking


def cross_validated_mae(X_train, y_train, feature_names):
    scores = cross_validate(
        make_pipeline(),
        X_train[feature_names],
        y_train,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
    )["test_score"]
    return -scores.mean(), scores.std()

#Evaluation metrics and plots
def evaluate_regression(y_true, y_pred, current_close, model_name) -> dict:
    y_true, y_pred, current_close = map(np.asarray, (y_true, y_pred, current_close))
    actual_dir = np.sign(y_true - current_close)
    pred_dir = np.sign(y_pred - current_close)
    # The naive baseline never predicts a direction, so the metric is N/A.
    directional_accuracy = (
        np.nan if np.all(pred_dir == 0) else (actual_dir == pred_dir).mean() * 100
    )
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Directional Accuracy (%)": directional_accuracy,
    }


def plot_predictions(dates_test, y_test, y_pred, baseline_pred, last_n=150) -> None:
    results = pd.DataFrame({
        "Date": dates_test.to_numpy(),
        "Actual": y_test.to_numpy(),
        "Predicted": y_pred,
        "Baseline": baseline_pred,
    }).tail(last_n)

    plt.figure(figsize=(14, 6))
    plt.plot(results["Date"], results["Actual"], label="Actual", linewidth=2)
    plt.plot(results["Date"], results["Predicted"], label="Linear Regression", alpha=0.9)
    plt.plot(results["Date"], results["Baseline"], label="Naive baseline",
             linestyle="--", alpha=0.8)
    plt.title(f"Actual vs predicted price over the last {last_n} sessions")
    plt.ylabel("Close (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], alpha=0.65)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--")
    axes[0].set(title="Actual vs Predicted", xlabel="Actual price", ylabel="Predicted price")

    residuals = y_test.to_numpy() - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], alpha=0.65)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(title="Residual plot", xlabel="Predicted price",
                ylabel="Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()


#Artifact export and verification
def export_artifact(pipeline, features, dates_train, scaled_metrics,
                    candidate_features, all_mae, selected_mae) -> dict:
    scaler = pipeline.named_steps["scaler"]
    linear_model = pipeline.named_steps["model"]

    standardized_weights = dict(zip(features, linear_model.coef_))
    effective_coefs = linear_model.coef_ / scaler.scale_
    effective_weights = dict(zip(features, effective_coefs))
    effective_intercept = linear_model.intercept_ - np.sum(effective_coefs * scaler.mean_)

    artifact = {
        "pipeline": pipeline,
        "features": features,
        "target": TARGET,
        "train_end_date": str(dates_train.max().date()),
        "model_version": MODEL_VERSION,
        "model_weights": {
            "standardized_coefficients": {k: float(v) for k, v in standardized_weights.items()},
            "pipeline_intercept": float(linear_model.intercept_),
            "effective_coefficients_original_scale": {k: float(v) for k, v in effective_weights.items()},
            "effective_intercept_original_scale": float(effective_intercept),
        },
    }
    joblib.dump(artifact, MODEL_FILE)

    metadata = {
        "name": "aapl-next-close-linear-regression",
        "version": MODEL_VERSION,
        "algorithm": "StandardScaler + LinearRegression",
        "symbol": "AAPL",
        "training_period": f"{dates_train.min().date()} to {dates_train.max().date()}",
        "train_end_date": str(dates_train.max().date()),
        "features": features,
        "candidate_features": candidate_features,
        "feature_selection": {
            "method": "RFECV",
            "cross_validation": f"TimeSeriesSplit(n_splits={N_SPLITS})",
            "scoring": "neg_mean_absolute_error",
            "selected_count": len(features),
            "candidate_count": len(candidate_features),
            "all_features_cv_mae": float(all_mae),
            "selected_features_cv_mae": float(selected_mae),
        },
        "test_mae": float(scaled_metrics["MAE"]),
        "test_rmse": float(scaled_metrics["RMSE"]),
        "test_r2": float(scaled_metrics["R2"]),
        "directional_accuracy_percent": float(scaled_metrics["Directional Accuracy (%)"]),
        "weights": artifact["model_weights"],
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model:    {MODEL_FILE.resolve()}")
    print(f"Metadata: {METADATA_FILE.resolve()}")
    return artifact


def verify_artifact(features, X_test) -> None:
    loaded = joblib.load(MODEL_FILE)
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    assert loaded["features"] == meta["features"], "Feature contract mismatch."
    assert set(meta["weights"]["standardized_coefficients"]) == set(features), "Missing weights."
    sample_pred = loaded["pipeline"].predict(X_test.iloc[[-1]])[0]
    assert np.isfinite(sample_pred), "Reloaded model cannot run inference."
    print("Artifact check passed: model reloads, metadata matches, weights present.")

#Main execution
def main() -> None:
    df = load_and_clean(DATA_FILE)
    plot_eda(df)

    df_features = build_features(df)
    for column in ["Return_1d", "Volume_change"]:
        outliers, lower, upper = detect_outliers(df_features, column)
        print(f"{column}: {len(outliers)} outliers | bounds [{lower:.4f}, {upper:.4f}]")
    plot_correlation(df_features)

    # Inputs / output (Date is excluded; the target never appears in X)
    X = df_features[CANDIDATE_FEATURES]
    y = df_features[TARGET]
    dates = df_features["Date"]

    X_train, X_test, y_train, y_test, dates_train, dates_test = time_split(X, y, dates)
    print(f"Train: {dates_train.min().date()} -> {dates_train.max().date()} ({len(X_train):,})")
    print(f"Test : {dates_test.min().date()} -> {dates_test.max().date()} ({len(X_test):,})")
    plot_split(dates_train, y_train, dates_test, y_test)

    # Feature validation and selection (train only)
    print(validate_features(X_train))
    mi_scores = mutual_information(X_train, y_train)
    selected_features, ranking = select_features(X_train, y_train, mi_scores)
    print(f"Selected {len(selected_features)}/{len(CANDIDATE_FEATURES)} features:")
    print(ranking)

    all_mae, all_std = cross_validated_mae(X_train, y_train, CANDIDATE_FEATURES)
    sel_mae, sel_std = cross_validated_mae(X_train, y_train, selected_features)
    print(pd.DataFrame([
        {"Feature set": "All candidates", "Feature count": len(CANDIDATE_FEATURES),
         "CV MAE": all_mae, "CV std": all_std},
        {"Feature set": "RFECV selected", "Feature count": len(selected_features),
         "CV MAE": sel_mae, "CV std": sel_std},
    ]).set_index("Feature set"))

    # Keep the current close (feature "Close") for baseline / direction metrics
    current_close_test = X_test["Close"].copy()
    features = selected_features
    X_train, X_test = X_train[features], X_test[features]
    print("Final feature contract:", features)

    # Naive baseline: tomorrow's close == today's close
    baseline_pred = current_close_test.to_numpy()
    baseline_metrics = evaluate_regression(y_test, baseline_pred, current_close_test, "Naive baseline")

    # Plain linear regression
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    linear_metrics = evaluate_regression(y_test, y_pred, current_close_test, "Linear Regression")

    rmse_gain = (baseline_metrics["RMSE"] - linear_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100
    print(f"RMSE improvement over baseline: {rmse_gain:.2f}%")
    plot_predictions(dates_test, y_test, y_pred, baseline_pred)

    coef_df = (
        pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        .assign(Abs_Coefficient=lambda d: d["Coefficient"].abs())
        .sort_values("Abs_Coefficient", ascending=False)
    )
    print(coef_df)

    # Final pipeline (scaler + model)
    pipeline = make_pipeline().fit(X_train, y_train)
    y_pred_scaled = pipeline.predict(X_test)
    scaled_metrics = evaluate_regression(
        y_test, y_pred_scaled, current_close_test, "Scaled Linear Regression"
    )
    print(pd.DataFrame([baseline_metrics, linear_metrics, scaled_metrics]).set_index("Model"))

    standardized_coef = pd.DataFrame({
        "Feature": features,
        "Standardized coefficient": pipeline.named_steps["model"].coef_,
    }).sort_values("Standardized coefficient", key=np.abs, ascending=False)
    print(standardized_coef)

    export_artifact(pipeline, features, dates_train, scaled_metrics,
                    CANDIDATE_FEATURES, all_mae, sel_mae)
    verify_artifact(features, X_test)


if __name__ == "__main__":
    main()