"""ETS, SARIMAX/ARIMA and Theta wrappers (protocol sections 6.2-6.6 and 8).

Every wrapper offers the same small interface:

* ``fit_*(y, ...)`` -> a fitted object with ``forecast(H)`` (point forecasts for h = 1..H) and
  ``intervals(H, alphas)`` (model-based intervals, protocol section 8),
* ``fitted.forecast_from(prefix, H)`` -> h = 1..H forecasts from a SHORTER history ``prefix`` keeping the fitted
  parameters fixed (used for the split-conformal calibration, protocol section 8 step 1). For ETS and SARIMAX the
  state is re-filtered through ``prefix`` only; ``ThetaModel`` has no filtering interface, so for Theta the
  (closed-form / SES) parameters are re-estimated on ``prefix`` (recorded in the deviations log, section 15).

A fit that raises or returns non-finite output raises ``FitFailure``; callers apply the fallback of section 6.6.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEED = 20261016


class FitFailure(RuntimeError):
    """A model could not be fitted or returned non-finite forecasts."""


def _finite(a, what: str) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    if not np.all(np.isfinite(a)):
        raise FitFailure(f"non-finite {what}")
    return a


# --------------------------------------------------------------------------------------------------------- ETS
@dataclass(frozen=True)
class ETSConfig:
    error: str                 # 'add' | 'mul'
    trend: str | None          # None | 'add'
    damped: bool
    seasonal: str | None       # None | 'add' | 'mul'

    @property
    def name(self) -> str:
        e = "A" if self.error == "add" else "M"
        t = "N" if self.trend is None else ("Ad" if self.damped else "A")
        s = "N" if self.seasonal is None else ("A" if self.seasonal == "add" else "M")
        return f"ETS({e},{t},{s})"

    @classmethod
    def from_name(cls, name: str) -> "ETSConfig":
        for c in ets_grid():
            if c.name == name:
                return c
        raise KeyError(name)


def ets_grid() -> list[ETSConfig]:
    """The 18 configurations of protocol section 6.2 (fixed order: error, trend, seasonal)."""
    out = []
    for error, (trend, damped), seasonal in itertools.product(
            ("add", "mul"), ((None, False), ("add", False), ("add", True)), (None, "add", "mul")):
        out.append(ETSConfig(error, trend, damped, seasonal))
    return out


def _ets_model(y, cfg: ETSConfig, m: int) -> ETSModel:
    # statsmodels 0.15.0 raises AttributeError in ETSResults.get_prediction for numpy endog (it reads
    # ``predicted_mean.index``); a pandas Series with a RangeIndex avoids the bug and changes nothing else.
    y = pd.Series(np.asarray(y, dtype="float64"))
    return ETSModel(y, error=cfg.error, trend=cfg.trend, damped_trend=cfg.damped, seasonal=cfg.seasonal,
                    seasonal_periods=m if cfg.seasonal else None)


class ETSFit:
    def __init__(self, res, cfg: ETSConfig, m: int, n: int):
        self.res, self.cfg, self.m, self.n = res, cfg, m, n
        self.params = np.asarray(res.params, dtype="float64")

    def forecast(self, H: int) -> np.ndarray:
        return _finite(self.res.forecast(H), "ETS forecast")

    def intervals(self, H: int, alphas=(0.2, 0.05)) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Library prediction intervals (analytic where statsmodels has them, simulated otherwise, 1000 paths).
        ``numpy.random.seed(20261016)`` is set immediately before the call (protocol section 12) and the same seed is
        passed as ``rng`` because the library ignores the global seed."""
        np.random.seed(SEED)
        # statsmodels 0.15.0 no longer reads numpy's global seed in ``simulate`` (``rng=None`` draws fresh OS
        # entropy), so the seed is also passed explicitly through ``rng`` (deviations log, section 15).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = self.res.get_prediction(start=self.n, end=self.n + H - 1, simulate_repetitions=1000, rng=SEED)
        out = {}
        for a in alphas:
            sf = pr.summary_frame(alpha=a)
            out[a] = (_finite(sf["pi_lower"].to_numpy(), "ETS lower"), _finite(sf["pi_upper"].to_numpy(), "ETS upper"))
        return out

    def forecast_from(self, prefix, H: int) -> np.ndarray:
        r = _ets_model(np.asarray(prefix, dtype="float64"), self.cfg, self.m).smooth(self.params)
        return _finite(r.forecast(H), "ETS frozen forecast")


def fit_ets(y, cfg: ETSConfig, m: int) -> ETSFit:
    y = np.asarray(y, dtype="float64")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _ets_model(y, cfg, m).fit(disp=False, maxiter=1000)
        fit = ETSFit(res, cfg, m, len(y))
        _finite(fit.params, "ETS parameters")
        fit.forecast(1)
        return fit
    except FitFailure:
        raise
    except Exception as e:  # noqa: BLE001 - library failures all map to the protocol fallback
        raise FitFailure(f"{cfg.name}: {type(e).__name__}: {e}") from e


# ------------------------------------------------------------------------------------------------------ ARIMA
@dataclass(frozen=True)
class ARIMAConfig:
    order: tuple[int, int, int]
    seasonal: tuple[int, int, int]      # (P, D, Q); the period is m

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}x{self.seasonal}".replace(" ", "")

    @classmethod
    def from_name(cls, name: str) -> "ARIMAConfig":
        for c in arima_grid():
            if c.name == name:
                return c
        raise KeyError(name)


def arima_grid() -> list[ARIMAConfig]:
    """The 12 configurations of protocol section 6.3."""
    return [ARIMAConfig(o, s) for o in ((0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1))
            for s in ((0, 0, 0), (0, 1, 1), (1, 1, 1))]


def _sarimax(y, cfg: ARIMAConfig, m: int) -> SARIMAX:
    return SARIMAX(y, order=cfg.order, seasonal_order=cfg.seasonal + (m,), trend=None,
                   enforce_stationarity=True, enforce_invertibility=True)


class ARIMAFit:
    def __init__(self, res, cfg: ARIMAConfig, m: int):
        self.res, self.cfg, self.m = res, cfg, m
        self.params = np.asarray(res.params, dtype="float64")

    def forecast(self, H: int) -> np.ndarray:
        return _finite(self.res.get_forecast(H).predicted_mean, "ARIMA forecast")

    def intervals(self, H: int, alphas=(0.2, 0.05)) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        fc = self.res.get_forecast(H)
        out = {}
        for a in alphas:
            ci = np.asarray(fc.conf_int(alpha=a))
            out[a] = (_finite(ci[:, 0], "ARIMA lower"), _finite(ci[:, 1], "ARIMA upper"))
        return out

    def forecast_from(self, prefix, H: int) -> np.ndarray:
        r = _sarimax(np.asarray(prefix, dtype="float64"), self.cfg, self.m).smooth(self.params)
        return _finite(r.get_forecast(H).predicted_mean, "ARIMA frozen forecast")


def fit_arima(y, cfg: ARIMAConfig, m: int) -> ARIMAFit:
    y = np.asarray(y, dtype="float64")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _sarimax(y, cfg, m).fit(disp=False)
        fit = ARIMAFit(res, cfg, m)
        _finite(fit.params, "ARIMA parameters")
        fit.forecast(1)
        return fit
    except FitFailure:
        raise
    except Exception as e:  # noqa: BLE001
        raise FitFailure(f"{cfg.name}: {type(e).__name__}: {e}") from e


# ------------------------------------------------------------------------------------------------------- Theta
class ThetaFit:
    def __init__(self, res, m: int):
        self.res, self.m = res, m

    def forecast(self, H: int) -> np.ndarray:
        return _finite(np.asarray(self.res.forecast(H, theta=2)), "Theta forecast")

    def intervals(self, H: int, alphas=(0.2, 0.05)) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for a in alphas:
            pi = self.res.prediction_intervals(H, theta=2, alpha=a)
            out[a] = (_finite(pi["lower"].to_numpy(), "Theta lower"), _finite(pi["upper"].to_numpy(), "Theta upper"))
        return out

    def forecast_from(self, prefix, H: int) -> np.ndarray:
        """Theta has no frozen-parameter filter in statsmodels: re-estimate on ``prefix`` (see module docstring)."""
        return fit_theta(prefix, self.m).forecast(H)


def fit_theta(y, m: int) -> ThetaFit:
    y = np.asarray(y, dtype="float64")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ThetaModel(y, period=m, deseasonalize=True, method="auto").fit()
        fit = ThetaFit(res, m)
        fit.forecast(1)
        return fit
    except FitFailure:
        raise
    except Exception as e:  # noqa: BLE001
        raise FitFailure(f"Theta: {type(e).__name__}: {e}") from e


# --------------------------------------------------------------------------- analytic SNAIVE interval (fallback)
def snaive_analytic_interval(y, H: int, m: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fallback model-based interval, used only when a model's own interval is unavailable (see deviations log).

    ``point +/- z * sigma_h`` around the SNAIVE forecast with ``sigma_h^2 = S2 * (floor((h-1)/m) + 1)``, where S2 is
    the in-sample mean squared seasonal difference (the seasonal-naive residual variance), the standard formula."""
    from .baselines import snaive
    from .metrics import scale_mse
    a = np.asarray(y, dtype="float64")
    h = np.arange(1, H + 1)
    sig = np.sqrt(scale_mse(a, m) * (np.floor((h - 1) / m) + 1))
    z = stats.norm.ppf(1 - alpha / 2)
    p = snaive(a, H, m)
    return p - z * sig, p + z * sig
