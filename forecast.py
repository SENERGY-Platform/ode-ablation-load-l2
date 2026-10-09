"""The forecast itself, with no Ray, MLflow or Operator Lib in it.

Kept apart from training.py and op.py for two reasons: the training path and the
inference path share one definition of every feature instead of two that can
drift, and the arithmetic can be exercised in a cell against a synthetic series.

Time is counted in whole UTC hours since the epoch throughout. That is the
bucketing the evaluation scores in (Operator Lib's _bucket_start), so one hour
here is exactly one scored bucket.

The forecast for target hour T is issued from inside hour T-24. That hour is
still running, so the newest hour any feature may read is T-25. Nothing here
reads hour T-24 or later: that is what makes this a 24 hour ahead forecast and
not a nowcast of the hour being scored.
"""

import math
import typing

import numpy as np

HORIZON_H = 24
NEWEST_LAG_H = HORIZON_H + 1  # 25: the newest completed hour at issue time

# Feature order is the weight order. The first seven are in W, the last is 0/1.
FEATURES = (
    "lag25",      # newest completed hour
    "lag48",      # same hour, two days back
    "lag72",      # same hour, three days back
    "lag168",     # same hour, one week back
    "lag336",     # same hour, two weeks back
    "mean24",     # mean of the 24 newest completed hours (T-48 .. T-25)
    "samehour7",  # mean of the same hour over days 2..7 back
    "weekend",    # target hour falls on saturday or sunday (UTC calendar)
)
_N_POWER_FEATURES = 7
_LAGS = (25, 48, 72, 168, 336)

MAX_LAG_H = 336
# How many hourly means the online state keeps, and the model seeds it with:
# enough for every feature of a target HORIZON_H ahead of the newest hour.
SEED_HOURS = MAX_LAG_H + HORIZON_H + 2


def epoch_hour(seconds: float) -> int:
    """The UTC hour a unix time falls into, by truncation."""
    return int(math.floor(seconds / 3600.0))


def weekday_of_hour(hour: int) -> int:
    """Monday is 0. 1970-01-01 was a thursday.

    UTC calendar, not local: local midnight in Europe/Berlin is one or two hours
    earlier, so the weekend boundary is off by that much. Accepted to keep the
    image free of a tz database dependency.
    """
    return (hour // 24 + 3) % 7


def _nanmean(values: typing.Iterable[float]) -> float:
    total, count = 0.0, 0
    for v in values:
        if v is None or math.isnan(v):
            continue
        total += v
        count += 1
    return total / count if count else math.nan


def features_for(get: typing.Callable[[int], float], target: int) -> typing.List[float]:
    """The raw feature vector for target hour `target`.

    `get(hour)` returns the hourly mean for an epoch hour, or nan where there is
    none. Only hours <= target - NEWEST_LAG_H are ever asked for.
    """
    row = [get(target - k) for k in _LAGS]
    row.append(_nanmean(get(target - k) for k in range(NEWEST_LAG_H, NEWEST_LAG_H + 24)))
    row.append(_nanmean(get(target - 24 * d) for d in range(2, 8)))
    row.append(1.0 if weekday_of_hour(target) >= 5 else 0.0)
    return row


def usable(row: typing.Sequence[float]) -> bool:
    """A row needs at least one of the two averages to stand on."""
    return not (math.isnan(row[5]) and math.isnan(row[6]))


def impute(row: typing.Sequence[float], fallback: float) -> typing.List[float]:
    """A missing power feature takes the same-hour average, then the 24 h mean,
    then the training mean. The same rule in training and in inference."""
    out = list(row)
    if not math.isnan(out[6]):
        base = out[6]
    elif not math.isnan(out[5]):
        base = out[5]
    else:
        base = fallback
    for i in range(_N_POWER_FEATURES):
        if math.isnan(out[i]):
            out[i] = base
    return out


def _solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Least squares with an intercept and a small ridge on everything but it."""
    Xa = np.hstack([np.ones((X.shape[0], 1)), X])
    penalty = np.eye(Xa.shape[1]) * ridge * X.shape[0]
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def predict_row(weights: typing.Sequence[float], row: typing.Sequence[float]) -> float:
    return float(weights[0] + sum(w * x for w, x in zip(weights[1:], row)))


def design(hourly: typing.Dict[int, float], fallback: float):
    """Every target hour with a known actual and a usable feature row."""
    hours = sorted(hourly)
    get = lambda h: hourly.get(h, math.nan)
    rows, ys, ts = [], [], []
    for t in range(hours[0] + MAX_LAG_H + 1, hours[-1] + 1):
        y = hourly.get(t)
        if y is None or math.isnan(y):
            continue
        raw = features_for(get, t)
        if not usable(raw):
            continue
        rows.append(impute(raw, fallback))
        ys.append(y)
        ts.append(t)
    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float), np.asarray(ts, dtype=np.int64)


def fit(hourly: typing.Dict[int, float], ridge: float = 0.01, holdout_h: int = 28 * 24):
    """Fit the weights, and report an honest holdout on the newest `holdout_h`
    hours of the training history before refitting on all of it.

    Returns (weights, fallback, metrics).
    """
    values = np.asarray([v for v in hourly.values() if not math.isnan(v)], dtype=float)
    fallback = float(values.mean())
    X, y, ts = design(hourly, fallback)
    if len(y) < 2 * holdout_h:
        raise ValueError(f"only {len(y)} usable hours; need at least {2 * holdout_h}")

    split = ts.max() - holdout_h
    tr, va = ts <= split, ts > split
    w_holdout = _solve(X[tr], y[tr], ridge)
    Xa_va = np.hstack([np.ones((va.sum(), 1)), X[va]])
    metrics = {
        "val_rmse": _rmse(Xa_va @ w_holdout, y[va]),
        "val_rmse_baseline_samehour7": _rmse(X[va][:, 6], y[va]),
        "val_rmse_baseline_lag168": _rmse(X[va][:, 3], y[va]),
        "val_rmse_baseline_mean24": _rmse(X[va][:, 5], y[va]),
        "val_hours": float(va.sum()),
        "train_hours": float(len(y)),
    }
    weights = _solve(X, y, ridge)
    return [float(w) for w in weights], fallback, metrics


def hourly_from_frame(frame, column: str = "value") -> typing.Dict[int, float]:
    """Hourly means of one column of an Operator Lib history frame, keyed by epoch hour."""
    import pandas as pd

    t = pd.to_datetime(frame["time"], utc=True)
    v = pd.to_numeric(frame[column], errors="coerce")
    hours = ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(hours=1)).astype("int64")
    s = pd.Series(v.to_numpy(), index=hours.to_numpy()).dropna()
    means = s.groupby(level=0).mean()
    return {int(h): float(m) for h, m in means.items()}
