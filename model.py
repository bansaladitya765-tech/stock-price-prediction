import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "Return_1d", "Return_2d", "Return_5d", "Return_10d", "Gap", "Intraday_Range",
    "MA_5_Ratio", "MA_20_Ratio", "MA_50_Ratio", "MA_5_20_Spread", "MA_20_50_Spread",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram", "Volatility_5", "Volatility_20",
    "Volume_Ratio_5", "Volume_Ratio_20", "High_20_Ratio", "Low_20_Ratio", "Bollinger_Position",
]


def _rsi(close, period=14):
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.rolling(period).mean()
    average_loss = losses.rolling(period).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(average_loss != 0, 100).where(~((average_gain == 0) & (average_loss == 0)), 50)


def _create_features(data):
    df = data.copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in df.columns:
            raise ValueError(f"Downloaded data is missing the '{column}' column.")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Return_1d"] = df["Close"].pct_change()
    df["Return_2d"] = df["Close"].pct_change(2)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Close"]

    for window in (5, 20, 50):
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()
    df["MA_5_Ratio"] = df["Close"] / df["MA_5"]
    df["MA_20_Ratio"] = df["Close"] / df["MA_20"]
    df["MA_50_Ratio"] = df["Close"] / df["MA_50"]
    df["MA_5_20_Spread"] = df["MA_5"] / df["MA_20"] - 1
    df["MA_20_50_Spread"] = df["MA_20"] / df["MA_50"] - 1

    df["RSI_14"] = _rsi(df["Close"])
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    df["Volatility_5"] = df["Return_1d"].rolling(5).std()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["Volume_Ratio_5"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["Volume_Ratio_20"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["High_20_Ratio"] = df["Close"] / df["High"].rolling(20).max()
    df["Low_20_Ratio"] = df["Close"] / df["Low"].rolling(20).min()
    mid = df["Close"].rolling(20).mean()
    band = df["Close"].rolling(20).std() * 2
    df["Bollinger_Position"] = (df["Close"] - mid) / band

    # Predicting percentage return is more stable than predicting the raw price.
    df["Next_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Next_Close"] = df["Close"].shift(-1)
    return df.replace([np.inf, -np.inf], np.nan)


def _models():
    return {
        "Extra Trees": ExtraTreesRegressor(n_estimators=500, max_features=0.8, min_samples_leaf=3, random_state=42, n_jobs=1),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_features=0.8, min_samples_leaf=3, random_state=42, n_jobs=1),
    }


def _direction_models():
    """Classifiers explicitly trained to identify next-day direction."""
    return {
        "Extra Trees": ExtraTreesClassifier(n_estimators=500, max_features=0.8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=1),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_features=0.8, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=1),
    }


def train_and_predict(data):
    """Fit the best time-series-validated model and predict the next close."""
    df = _create_features(data)
    complete = df.dropna(subset=FEATURE_COLUMNS + ["Next_Return", "Next_Close"]).copy()
    if len(complete) < 150:
        raise ValueError("Not enough clean price history to train a reliable model.")

    split = int(len(complete) * 0.80)
    train = complete.iloc[:split]
    test = complete.iloc[split:]
    x_train, y_train = train[FEATURE_COLUMNS], train["Next_Return"]
    direction_train = (y_train >= 0).astype(int)

    # Select the model using only earlier periods; never shuffle market data.
    validation = TimeSeriesSplit(n_splits=4)
    scores = {}
    for name, candidate in _direction_models().items():
        fold_scores = []
        for train_index, validation_index in validation.split(x_train):
            candidate.fit(x_train.iloc[train_index], direction_train.iloc[train_index])
            prediction = candidate.predict(x_train.iloc[validation_index])
            fold_scores.append(accuracy_score(direction_train.iloc[validation_index], prediction))
        scores[name] = float(np.mean(fold_scores))

    chosen_name = min(scores, key=scores.get)
    model = _models()[chosen_name]
    model.fit(x_train, y_train)
    direction_model = _direction_models()[chosen_name]
    direction_model.fit(x_train, direction_train)

    # The regressor estimates move size; classifier supplies its direction.
    predicted_magnitudes = np.abs(model.predict(test[FEATURE_COLUMNS]))
    predicted_directions = direction_model.predict(test[FEATURE_COLUMNS])
    predicted_returns = predicted_magnitudes * np.where(predicted_directions == 1, 1, -1)
    predicted_closes = test["Close"].to_numpy() * (1 + predicted_returns)
    actual_closes = test["Next_Close"].to_numpy()
    direction_accuracy = accuracy_score(test["Next_Return"].to_numpy() >= 0, predicted_directions) * 100

    # Refit on all known data before making the latest prediction.
    final_model = _models()[chosen_name]
    final_model.fit(complete[FEATURE_COLUMNS], complete["Next_Return"])
    final_direction_model = _direction_models()[chosen_name]
    final_direction_model.fit(complete[FEATURE_COLUMNS], (complete["Next_Return"] >= 0).astype(int))
    latest = df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    current = float(latest["Close"].iloc[0])
    predicted_magnitude = abs(float(final_model.predict(latest[FEATURE_COLUMNS])[0]))
    predicted_direction = int(final_direction_model.predict(latest[FEATURE_COLUMNS])[0])
    predicted_return = predicted_magnitude if predicted_direction == 1 else -predicted_magnitude
    predicted = current * (1 + predicted_return)

    return {
        "current": current,
        "predicted": float(predicted),
        "predicted_return": predicted_return * 100,
        "model_name": chosen_name,
        "metrics": {
            "MAE": float(mean_absolute_error(actual_closes, predicted_closes)),
            "RMSE": float(np.sqrt(mean_squared_error(actual_closes, predicted_closes))),
            "R2": float(r2_score(actual_closes, predicted_closes)),
            "direction_accuracy": float(direction_accuracy),
        },
        "latest_indicators": {
            "RSI": float(latest["RSI_14"].iloc[0]),
            "MACD": float(latest["MACD"].iloc[0]),
            "MA_20": float(latest["MA_20"].iloc[0]),
            "Volatility": float(latest["Volatility_20"].iloc[0]),
        },
        "feature_importance": pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": final_model.feature_importances_}).sort_values("Importance", ascending=False),
    }
