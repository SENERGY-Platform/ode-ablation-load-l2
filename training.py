"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The model is a linear day-ahead forecaster over lagged hourly means of net grid
power (see forecast.py for every feature and why none of them reads the hour being
forecast). Training turns the raw ~10 s history into hourly means, fits the
weights, reports an honest holdout on the newest four weeks of the history, and
hands the operator the newest hourly means as a seed so it can forecast from its
first message instead of waiting two weeks for its own lags.
"""

import datetime
import math
import typing

import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger

import forecast


# How much history one training pass reads. 120 days rather than a year, so the
# weights are fitted on the regime the forecast is used in: a year is dominated by
# winter's higher, heating-driven load, whose dynamics differ from summer and
# autumn. The first 14 days of the window only feed lags, so about 106 days of
# targets remain, of which the newest 28 are the holdout.
TRAINING_WINDOW = datetime.timedelta(days=120)


class OdeAblationLoadL2Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    It carries the fitted weights, the fallback level and the seed of hourly
    means ending where training ended. predict() takes a target epoch hour and a
    mapping of epoch hour to hourly mean; op.py calls forecast_hour() directly so
    it can pass its own lookup without building a payload per hour.
    """

    def __init__(self, weights: typing.List[float], fallback: float, seed: typing.Dict[int, float]) -> None:
        self.weights = weights
        self.fallback = fallback
        self.seed = seed

    def forecast_hour(self, get: typing.Callable[[int], float], target: int) -> float:
        raw = forecast.features_for(get, target)
        if not forecast.usable(raw):
            return self.fallback
        return forecast.predict_row(self.weights, forecast.impute(raw, self.fallback))

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when
        # the model is called directly, so the payload is taken from whichever
        # argument holds it.
        payload = model_input if model_input is not None else context
        history = {int(k): float(v) for k, v in payload.get("history", {}).items()}
        target = int(payload["target_hour"])
        return self.forecast_hour(lambda h: history.get(h, math.nan), target)


@ray.remote
def _fit(datasets: typing.List[typing.Any]) -> typing.Dict[str, typing.Any]:
    """The distributed part: hourly means, then the fit."""
    import pandas as pd

    frames = []
    for dataset in datasets:
        if isinstance(dataset, ray.ObjectRef):
            dataset = ray.get(dataset)
        frame = dataset.to_pandas()
        if "value" in frame.columns and len(frame):
            frames.append(frame[["time", "value"]])
    if not frames:
        return {}
    hourly = forecast.hourly_from_frame(pd.concat(frames, ignore_index=True))
    weights, fallback, metrics = forecast.fit(hourly)
    newest = max(hourly)
    seed = {h: v for h, v in hourly.items() if h > newest - forecast.SEED_HOURS}
    return {
        "weights": weights,
        "fallback": fallback,
        "metrics": metrics,
        "seed": seed,
        "hours": len(hourly),
        "newest_hour": newest,
    }


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        result = ray.get(_fit.remote(datasets))
    if not result:
        return None

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "horizon_h": forecast.HORIZON_H,
        "features": ",".join(forecast.FEATURES),
        "hourly_buckets": result["hours"],
        "newest_training_hour": datetime.datetime.fromtimestamp(
            result["newest_hour"] * 3600, tz=datetime.timezone.utc).isoformat(),
    })
    logger.log_metrics(result["metrics"])
    return OdeAblationLoadL2Model(
        weights=result["weights"], fallback=result["fallback"], seed=result["seed"])
