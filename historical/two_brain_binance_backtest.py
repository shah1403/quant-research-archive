from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Data loading
# -----------------------------

def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 1000,
    market_data_only: bool = True,
) -> pd.DataFrame:
    """Fetch recent Binance spot klines.

    Note: this is meant to be run locally by the user. In this chat environment,
    outbound DNS/JSON fetches may be restricted.
    """
    import requests

    base = "https://data-api.binance.vision" if market_data_only else "https://api.binance.com"
    url = f"{base}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    num_cols = [c for c in cols if c not in ["open_time", "close_time"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


# -----------------------------
# Core math
# -----------------------------

def ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def poly_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    return np.vander(x, N=order + 1, increasing=True)


def fit_ols_poly(y: np.ndarray, order: int) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    X = poly_design_matrix(x, order)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_wls_ridge_poly(y: np.ndarray, order: int, ridge_lambda: float, decay: float) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    X = poly_design_matrix(x, order)
    # exponential recency weighting
    idx = np.arange(len(y), dtype=float)
    w = np.exp(decay * (idx - idx[-1]))
    W = np.diag(w)
    XT_W = X.T @ W
    gram = XT_W @ X
    reg = ridge_lambda * np.eye(order + 1)
    reg[0, 0] = 0.0  # do not regularize intercept
    beta = np.linalg.solve(gram + reg, XT_W @ y)
    return beta


def eval_poly(beta: np.ndarray, x: float) -> float:
    return float(sum(beta[i] * (x ** i) for i in range(len(beta))))


def poly_velocity(beta: np.ndarray, x: float) -> float:
    if len(beta) <= 1:
        return 0.0
    return float(sum(i * beta[i] * (x ** (i - 1)) for i in range(1, len(beta))))


def poly_acceleration(beta: np.ndarray, x: float) -> float:
    if len(beta) <= 2:
        return 0.0
    return float(sum(i * (i - 1) * beta[i] * (x ** (i - 2)) for i in range(2, len(beta))))


# -----------------------------
# Brain A state
# -----------------------------

@dataclass
class SubModelState:
    last_pred_dir: Optional[int] = None
    outcomes: List[int] = field(default_factory=list)

    def update_accuracy(self, actual_move: float, maxlen: int) -> None:
        if self.last_pred_dir is None:
            return
        if actual_move == 0:
            return
        actual_dir = 1 if actual_move > 0 else -1
        self.outcomes.append(1 if actual_dir == self.last_pred_dir else 0)
        if len(self.outcomes) > maxlen:
            self.outcomes.pop(0)

    @property
    def dir_acc(self) -> float:
        if not self.outcomes:
            return 50.0
        return 100.0 * float(sum(self.outcomes)) / len(self.outcomes)


@dataclass
class StrategyConfig:
    windows: Tuple[int, ...] = (10, 20, 40, 100, 200)
    orders: Tuple[int, ...] = (0, 1, 2)
    acc_lookback: int = 40
    penalty_factor: float = 0.4
    bonus_factor: float = 0.005
    ema_alpha: float = 0.3
    ridge_lambda: float = 1e-3
    wls_decay: float = 0.05
    trend_acc_threshold: float = 52.0
    dead_low: float = 50.0
    dead_high: float = 52.0
    trend_snr_threshold: float = 1.5
    noise_snr_threshold: float = 1.0
    mean_revert_band: float = 1.0
    cost_bps_per_side: float = 5.0  # rough placeholder for crossing costs


@dataclass
class BacktestResult:
    df: pd.DataFrame
    summary: Dict[str, float]


class TwoBrainPredictor:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.submodels: Dict[Tuple[int, int], SubModelState] = {
            (w, o): SubModelState() for w in cfg.windows for o in cfg.orders
        }

    def _brain_a_select(self, prices: np.ndarray, t: int) -> Tuple[int, int, float, float]:
        # Update past directional outcomes using the move that just happened.
        actual_move = prices[t] - prices[t - 1]
        for state in self.submodels.values():
            state.update_accuracy(actual_move, self.cfg.acc_lookback)

        best = None
        best_score = -1e18
        best_acc = 50.0

        for (w, o), state in self.submodels.items():
            if t < w - 1:
                continue
            y = prices[t - w + 1 : t + 1]
            beta = fit_ols_poly(y, o)
            pred_next = eval_poly(beta, w)
            pred_dir = 1 if pred_next > prices[t] else -1 if pred_next < prices[t] else 0
            state.last_pred_dir = pred_dir if pred_dir != 0 else state.last_pred_dir

            score = state.dir_acc - self.cfg.penalty_factor * o + self.cfg.bonus_factor * w
            if score > best_score:
                best_score = score
                best = (w, o)
                best_acc = state.dir_acc

        if best is None:
            # Fallback before enough data accumulates.
            valid_windows = [w for w in self.cfg.windows if t >= w - 1]
            w = min(valid_windows) if valid_windows else self.cfg.windows[0]
            return w, 1, 50.0, 50.0

        return best[0], best[1], best_acc, best_score

    def _brain_b_signal(self, prices: np.ndarray, t: int, w: int, o: int) -> Dict[str, float]:
        y = prices[t - w + 1 : t + 1]
        smooth = ema(y, self.cfg.ema_alpha)
        beta = fit_wls_ridge_poly(smooth, o, self.cfg.ridge_lambda, self.cfg.wls_decay)
        x_eval = w
        fair_value = eval_poly(beta, x_eval)
        velocity = poly_velocity(beta, x_eval)
        acceleration = poly_acceleration(beta, x_eval)
        local_vol = float(np.std(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
        snr = abs(velocity) / local_vol if local_vol > 1e-12 else 0.0
        deviation = prices[t] - fair_value
        return {
            "fair_value": fair_value,
            "velocity": velocity,
            "acceleration": acceleration,
            "local_vol": local_vol,
            "snr": snr,
            "deviation": deviation,
        }

    def _route(self, dir_acc: float, snr: float, velocity: float, deviation: float, local_vol: float) -> str:
        if self.cfg.dead_low < dir_acc < self.cfg.dead_high:
            return "flat_deadzone"

        if dir_acc >= self.cfg.trend_acc_threshold and snr >= self.cfg.trend_snr_threshold:
            return "trend_long" if velocity > 0 else "trend_short"

        # Noise / reversion mode
        if dir_acc >= self.cfg.trend_acc_threshold and snr < self.cfg.noise_snr_threshold:
            if deviation < -self.cfg.mean_revert_band * max(local_vol, 1e-12):
                return "mr_long"
            if deviation > self.cfg.mean_revert_band * max(local_vol, 1e-12):
                return "mr_short"
            return "flat_noise_wait"

        if dir_acc <= 50.0 and snr >= self.cfg.trend_snr_threshold:
            return "flat_danger"

        if dir_acc <= 50.0 and snr < self.cfg.noise_snr_threshold:
            if deviation < -self.cfg.mean_revert_band * max(local_vol, 1e-12):
                return "mr_long"
            if deviation > self.cfg.mean_revert_band * max(local_vol, 1e-12):
                return "mr_short"
            return "flat_mm_wait"

        return "flat_other"

    def backtest(self, df: pd.DataFrame, horizon: int = 1) -> BacktestResult:
        prices = df["close"].to_numpy(dtype=float)
        out_rows: List[Dict[str, float]] = []
        max_w = max(self.cfg.windows)

        for t in range(1, len(prices) - horizon):
            if t < max_w - 1:
                continue

            w, o, dir_acc, score = self._brain_a_select(prices, t)
            sig = self._brain_b_signal(prices, t, w, o)
            regime = self._route(dir_acc, sig["snr"], sig["velocity"], sig["deviation"], sig["local_vol"])

            future_ret = prices[t + horizon] - prices[t]
            side = 0
            if regime in ("trend_long", "mr_long"):
                side = 1
            elif regime in ("trend_short", "mr_short"):
                side = -1

            gross_pnl = side * future_ret
            # crude cost proxy for aggressive execution or crossing spread
            cost = 0.0
            if side != 0:
                cost = (self.cfg.cost_bps_per_side / 1e4) * prices[t] * 2
            net_pnl = gross_pnl - cost

            out_rows.append(
                {
                    "t": t,
                    "timestamp": df.iloc[t]["close_time"],
                    "price": prices[t],
                    "future_price": prices[t + horizon],
                    "future_ret": future_ret,
                    "winning_w": w,
                    "winning_o": o,
                    "dir_acc": dir_acc,
                    "selector_score": score,
                    **sig,
                    "regime": regime,
                    "side": side,
                    "gross_pnl": gross_pnl,
                    "cost": cost,
                    "net_pnl": net_pnl,
                }
            )

        bt = pd.DataFrame(out_rows)
        if bt.empty:
            return BacktestResult(bt, {"error": "Not enough data to run backtest."})

        trades = bt[bt["side"] != 0].copy()
        win_rate = float((trades["net_pnl"] > 0).mean()) if not trades.empty else math.nan
        summary = {
            "bars_evaluated": float(len(bt)),
            "trades": float(len(trades)),
            "trade_rate": float(len(trades) / len(bt)) if len(bt) else math.nan,
            "gross_pnl_sum": float(trades["gross_pnl"].sum()) if not trades.empty else 0.0,
            "net_pnl_sum": float(trades["net_pnl"].sum()) if not trades.empty else 0.0,
            "avg_net_pnl": float(trades["net_pnl"].mean()) if not trades.empty else math.nan,
            "win_rate": win_rate,
            "mean_dir_acc": float(bt["dir_acc"].mean()),
            "mean_snr": float(bt["snr"].mean()),
        }
        return BacktestResult(bt, summary)


def smoke_test() -> None:
    # Tiny sample only to verify plumbing, not strategy validity.
    rows = [
        [1774192200000, "68852.20", "68879.77", "68852.19", "68879.77", "3.14406", 1774192259999, "216538.49746980", 1154, "2.00173", "137859.80339110", "0"],
        [1774192260000, "68879.77", "68887.20", "68865.21", "68887.20", "2.03151", 1774192319999, "139921.33394240", 1851, "1.03049", "70977.11056270", "0"],
        [1774192320000, "68887.20", "68887.20", "68877.63", "68885.62", "1.50949", 1774192379999, "103973.76364340", 687, "0.63858", "43985.32128100", "0"],
        [1774192380000, "68885.61", "68885.62", "68880.14", "68880.15", "1.08122", 1774192439999, "74479.08908360", 562, "0.45270", "31183.80530060", "0"],
        [1774192440000, "68880.14", "68907.81", "68880.14", "68907.81", "5.62262", 1774192499999, "387322.76299560", 549, "5.42071", "373414.99275520", "0"],
    ]
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    cfg = StrategyConfig(windows=(3,), orders=(0,1,2), acc_lookback=3)
    model = TwoBrainPredictor(cfg)
    result = model.backtest(df, horizon=1)
    print(result.summary)
    if not result.df.empty:
        print(result.df[["timestamp", "price", "winning_w", "winning_o", "dir_acc", "snr", "regime", "side", "net_pnl"]])


if __name__ == "__main__":
    # Example local run:
    #   df = fetch_binance_klines("BTCUSDT", "1m", 1000)
    #   cfg = StrategyConfig()
    #   model = TwoBrainPredictor(cfg)
    #   result = model.backtest(df, horizon=1)
    #   print(result.summary)
    #   result.df.to_csv("two_brain_results.csv", index=False)
    smoke_test()
