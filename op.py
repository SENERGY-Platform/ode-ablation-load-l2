"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

This operator forecasts the hourly mean of net grid power 24 hours ahead. Every
message it receives is folded into the mean of the hour it falls in; when an hour
closes, its mean joins the history the forecast reads. Every message answers with
the forecast for the hour 24 hours after its own, stamped with that time, and the
forecast is computed once per hour rather than once per message.
"""

import datetime
import math
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

import forecast
from training import train_model


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


def _unwrap(model: PyFuncModel):
    """The PythonModel behind a loaded pyfunc, which holds the weights and seed."""
    try:
        return model.unwrap_python_model()
    except Exception:
        return model._model_impl.python_model


class Operator(MLOperator):
    configType = CustomConfig

    # Which inputs this operator accepts. "args" are the mapping destinations the
    # pipeline is configured with, and the name is what infer() receives as
    # "selector", so one operator can treat several input shapes differently.
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._forecaster = None
        self._model_ref = None
        self._hist: typing.Dict[int, float] = {}
        self._cur_hour: typing.Optional[int] = None
        self._cur_sum = 0.0
        self._cur_n = 0
        self._cached_for: typing.Optional[int] = None
        self._cached: typing.Optional[float] = None
        super().init(*args, **kwargs)

    def _adopt(self, model: PyFuncModel) -> None:
        """Take the weights from a (new) model; keep any hours already observed,
        and fill in from the model's seed only where nothing was observed."""
        self._forecaster = _unwrap(model)
        self._model_ref = model
        for hour, value in self._forecaster.seed.items():
            self._hist.setdefault(int(hour), float(value))
        self._cached_for = None

    def _close_hour(self) -> None:
        if self._cur_hour is not None and self._cur_n > 0:
            self._hist[self._cur_hour] = self._cur_sum / self._cur_n
        self._cur_sum, self._cur_n = 0.0, 0

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        The result timestamp is the message's own time plus the horizon, so the
        forecast is filed under the hour it is about.
        """
        if model is None:
            return None, None, None
        if model is not self._model_ref:
            self._adopt(model)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        hour = forecast.epoch_hour(timestamp.timestamp())

        if self._cur_hour is None:
            self._cur_hour = hour
        elif hour > self._cur_hour:
            self._close_hour()
            self._cur_hour = hour
            floor = hour - forecast.SEED_HOURS
            for old in [h for h in self._hist if h < floor]:
                del self._hist[old]

        try:
            value = float(data.get("value"))
        except (TypeError, ValueError):
            value = math.nan
        # A late message (hour < current) is not folded into an hour already closed.
        if hour == self._cur_hour and math.isfinite(value):
            self._cur_sum += value
            self._cur_n += 1

        target = hour + forecast.HORIZON_H
        if self._cached_for != target:
            self._cached = self._forecaster.forecast_hour(
                lambda h: self._hist.get(h, math.nan), target)
            self._cached_for = target

        result_time = timestamp + datetime.timedelta(hours=forecast.HORIZON_H)
        return result_time, {"prediction": self._cached}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started — so params and metrics logged through "logger" land on the run
        the resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Daily, which is how often the lag weights could plausibly move. The online
        history is kept across a retrain, so a new model does not lose hours.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
