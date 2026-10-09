# ode-ablation-load-l2

## What this operator does, and how it got here

**Task.** Forecast the net electrical consumption from the grid 24 hours ahead, as
hourly mean power in watts, scored by the developer's `evaluation.yaml` (RMSE at 1 h
resolution, threshold 6.1 W, test window September 2026).

**Outcome.** The operator is a working, honest day-ahead forecaster. It does **not**
meet the threshold: its September RMSE is **220.0 W** (run `a8f176cc…`, commit
`1cc0600`). It was never reported as complete.

### Data

- Found through the platform ontology (function Get-Power on electricity meters),
  not by device name. The grid-connection meter is "Stromzähler", an Iskra MT 175
  read through Tasmota: `sensor.MT175.P`, net active power in W, signed (negative is
  PV export), sampled every ~10 s, with history from Nov 2024 to the split.
- Profiled before use: instantaneous, regular sampling, daily and weekly periodicity,
  four gaps in the training year (the longest 3–9 Feb 2026). Hourly means swing by
  several hundred watts within a day (around −400 W at a sunny midday, up to ~1000 W
  in the evening) and the daily mean moves by 100 W or more from one day to the next.
- The meter is both the operator's only input and its scored target.

### Model (`forecast.py`, `training.py`, `op.py`)

- A linear regression with a small ridge penalty on lagged hourly means. The lags are
  25, 48 and 72 h, 1 and 2 weeks, the mean of the 24 newest completed hours, the
  same-hour mean over days 2–7 back, and a weekend flag.
- **Day-ahead by construction.** The forecast for hour T is issued from inside hour
  T−24, which is still running, so no feature reads anything newer than hour T−25. A
  synthetic test confirmed this: altering every hour from T−24 on leaves the forecast
  unchanged.
- `infer()` folds each message into a running hourly mean. It computes the forecast
  once per hour and stamps it with the message time plus 24 h, so it lands in the
  bucket being forecast. The model carries the last ~15 days of hourly means as a
  seed, so it can forecast from its first message.
- The arithmetic lives in `forecast.py`, with no Ray or MLflow in it. Training and
  inference share one definition of every feature, and the module can be tested in
  a cell.
- Training logs an honest holdout on the newest four weeks of history, next to three
  baselines.

### Results

| Run | Training window | August holdout RMSE | September RMSE (scored) |
|---|---|---|---|
| `e7bf242` | 365 days | 232.6 W | 220.0 W |
| `02ad681` | 120 days | 236.8 W | 221.0 W |
| `1cc0600` | 365 days (final) | 232.6 W | 220.0 W |

The baselines on the August holdout were: same-hour mean over days 2–7 back,
244.1 W; 24-hour mean, 254.9 W; same hour one week back, 300.1 W. The model beats
the best of them by about 5%. This looks like the limit of what the meter's own
history can predict a day ahead.

### On the threshold

6.1 W is about 36× below anything measured here. The scorer only checks which hour a
prediction is stamped with, not how far ahead it was made. An operator that predicted
the hour it is currently in would score near the threshold, but that is not a 24-hour
forecast, so it was not built. Whether the threshold is right for this target is the
developer's decision. `evaluation.yaml` was never changed.

### What was tried for more signal

The remaining error is mostly midday PV export, which depends on tomorrow's weather.
A day-ahead irradiance forecast is the one input that could reduce it substantially
(by tens of watts, not by the factor the threshold asks for). It is blocked for now:

- **The Open-Meteo forecast archive (51.7 N / 10 E)** archives forecasts as issued.
  Its export holds no rows before the split.
- **A second export of the same import**, stamped by `value.issued_at`
  (`d93dacea-5333-4a93-a2d7-44902191dbdc`), also stored nothing. Its timestamp format
  was copied from the first export, and the actual format of `issued_at` is unknown.
  Deleting it from ODE was refused (403). **Delete it in the platform UI.** While it
  exists, a launch that maps this import is refused as ambiguous.
- **The yr.no forecast export** only starts on 22 Aug 2026, too short to train on.

**To continue:** find the format `value.issued_at` is written in, recreate the
issue-time export with it, check that it covers the training year, and add the
forecast irradiance and cloud cover for the target hour (lead 2 days) as features.
Also confirm that 51.7 N / 10 E is near the household.

---

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "forecast.py" | The forecast arithmetic and features, shared by training and inference. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.2". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-load-l2". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-load-l2:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.2", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
