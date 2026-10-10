from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "downloads" / "msft_1m_raw_kinematic_first2years_signals.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WINDOWS = [20, 40, 60]
ORDERS = [1, 2]
HORIZONS = [1, 5, 7, 10]
AR_ORDERS = [1, 2, 5, 10, 20]
SCORE_LOOKBACK = 150
TEST_FRACTION = 0.20
AR_HISTORY = 10_000
AR_REFIT_EVERY = 390


def projection(window: int, order: int) -> np.ndarray:
    tau = np.arange(-(window - 1), 1, dtype=float)
    weights = np.exp(tau / (window / 2))
    vand = np.vander(tau, order + 1, increasing=True)
    return np.linalg.solve(
        vand.T @ (vand * weights[:, None]),
        (vand * weights[:, None]).T,
    )


def kinematic_forecasts(close: np.ndarray):
    n = len(close)
    combos = [(w, o, h) for w in WINDOWS for o in ORDERS for h in HORIZONS]
    forecasts = np.full((len(combos), n), np.nan)
    scores = np.full((len(combos), n), -np.inf)

    coefficients: dict[tuple[int, int], np.ndarray] = {}
    for w in WINDOWS:
        windows = sliding_window_view(close, w)
        for o in ORDERS:
            coef = windows @ projection(w, o).T
            full = np.full((n, o + 1), np.nan)
            full[w - 1 :] = coef
            coefficients[(w, o)] = full

    for i, (w, o, h) in enumerate(combos):
        coef = coefficients[(w, o)]
        future_vector = np.array([h**power for power in range(o + 1)])
        pred = coef @ future_vector
        forecasts[i] = pred

        displacement = pred - close
        realized = np.full(n, np.nan)
        realized[: n - h] = close[h:] - close[: n - h]
        edge = np.nan_to_num(np.sign(displacement) * realized, nan=0.0)
        cumulative = np.concatenate([[0.0], np.cumsum(edge)])
        for t in range(h + SCORE_LOOKBACK, n):
            # Forecasts through t-h have matured. Score the latest 150.
            end = t - h + 1
            start = end - SCORE_LOOKBACK
            scores[i, t] = cumulative[end] - cumulative[start]

    return combos, forecasts, scores


def ar_design(returns: np.ndarray, p: int, start: int, end: int):
    # Rows predict r[t] from r[t-1],...,r[t-p]. The end is exclusive.
    idx = np.arange(max(start, p), end)
    x = np.column_stack([np.ones(len(idx))] + [returns[idx - lag] for lag in range(1, p + 1)])
    y = returns[idx]
    return x, y


def select_ar_order(returns: np.ndarray, train_end: int) -> tuple[int, pd.DataFrame]:
    rows = []
    start = max(1, train_end - 100_000)
    for p in range(1, 21):
        x, y = ar_design(returns, p, start, train_end)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        sse = float(residual @ residual)
        bic = len(y) * np.log(sse / len(y)) + (p + 1) * np.log(len(y))
        rows.append({"ar_order": p, "bic": bic})
    table = pd.DataFrame(rows)
    return int(table.loc[table["bic"].idxmin(), "ar_order"]), table


def rolling_ar_forecasts(
    close: np.ndarray,
    returns: np.ndarray,
    test_start: int,
    p: int,
) -> dict[int, np.ndarray]:
    n = len(close)
    output = {h: np.full(n, np.nan) for h in HORIZONS}
    beta = None
    for t in range(test_start, n - 1):
        if beta is None or (t - test_start) % AR_REFIT_EVERY == 0:
            start = max(p, t - AR_HISTORY + 1)
            x, y = ar_design(returns, p, start, t + 1)
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)

        history = list(returns[t - p + 1 : t + 1])
        predicted_returns = []
        for _ in range(max(HORIZONS)):
            lags = np.array(history[-p:][::-1])
            next_return = float(beta[0] + beta[1:] @ lags)
            predicted_returns.append(next_return)
            history.append(next_return)
        cumulative = np.cumsum(predicted_returns)
        for h in HORIZONS:
            output[h][t] = close[t] * np.exp(cumulative[h - 1])
    return output


def metrics(name: str, prediction: np.ndarray, target: np.ndarray, current: np.ndarray):
    valid = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(current)
    pred = prediction[valid]
    actual = target[valid]
    spot = current[valid]
    error_bps = 10_000 * (pred - actual) / spot
    actual_move = actual - spot
    predicted_move = pred - spot
    nonflat = actual_move != 0
    if np.all(predicted_move == 0):
        direction_accuracy = np.nan
    else:
        direction_accuracy = float(
            100 * np.mean(np.sign(predicted_move[nonflat]) == np.sign(actual_move[nonflat]))
        )
    return {
        "model": name,
        "n": int(valid.sum()),
        "mae_bps": float(np.mean(np.abs(error_bps))),
        "median_ae_bps": float(np.median(np.abs(error_bps))),
        "rmse_bps": float(np.sqrt(np.mean(error_bps**2))),
        "bias_bps": float(np.mean(error_bps)),
        "direction_accuracy_pct": direction_accuracy,
    }, valid, error_bps


def main():
    data = pd.read_csv(INPUT, usecols=["ts_event", "close"])
    data["ts_event"] = pd.to_datetime(data["ts_event"], utc=True)
    data = data.sort_values("ts_event").drop_duplicates("ts_event").reset_index(drop=True)
    close = data["close"].to_numpy(float)
    n = len(close)
    test_start = int(n * (1 - TEST_FRACTION))
    returns = np.diff(np.log(close), prepend=np.log(close[0]))

    combos, kin_forecasts, kin_scores = kinematic_forecasts(close)
    selected_dynamic = np.argmax(kin_scores, axis=0)
    dynamic_prediction = kin_forecasts[selected_dynamic, np.arange(n)]
    dynamic_h = np.array([combos[i][2] for i in selected_dynamic])
    dynamic_w = np.array([combos[i][0] for i in selected_dynamic])
    dynamic_o = np.array([combos[i][1] for i in selected_dynamic])

    _, bic_table = select_ar_order(returns, test_start)
    bic_table["included_in_comparison"] = bic_table["ar_order"].isin(AR_ORDERS)
    ar_by_order = {
        p: rolling_ar_forecasts(close, returns, test_start, p)
        for p in AR_ORDERS
    }

    target_dynamic = np.full(n, np.nan)
    ar_dynamic = {p: np.full(n, np.nan) for p in AR_ORDERS}
    martingale_dynamic = np.full(n, np.nan)
    for h in HORIZONS:
        idx = np.where((dynamic_h == h) & (np.arange(n) >= test_start) & (np.arange(n) + h < n))[0]
        target_dynamic[idx] = close[idx + h]
        for p in AR_ORDERS:
            ar_dynamic[p][idx] = ar_by_order[p][h][idx]
        martingale_dynamic[idx] = close[idx]

    rows = []
    prediction_rows = []
    dynamic_error = {}
    dynamic_models = [("Dynamic kinematic", dynamic_prediction)]
    dynamic_models += [(f"AR({p})", ar_dynamic[p]) for p in AR_ORDERS]
    dynamic_models += [("Martingale", martingale_dynamic)]
    for label, pred in dynamic_models:
        row, valid, errors = metrics(label, pred, target_dynamic, close)
        row["evaluation"] = "Dynamic selected horizon"
        rows.append(row)
        dynamic_error[label] = (valid, errors)

    for h in HORIZONS:
        combo_idx = [i for i, combo in enumerate(combos) if combo[2] == h]
        local_scores = kin_scores[combo_idx]
        chosen_local = np.argmax(local_scores, axis=0)
        chosen_global = np.array(combo_idx)[chosen_local]
        pred_kin = kin_forecasts[chosen_global, np.arange(n)]
        target = np.full(n, np.nan)
        target[: n - h] = close[h:]
        fixed_models = [("Kinematic", pred_kin)]
        fixed_models += [(f"AR({p})", ar_by_order[p][h]) for p in AR_ORDERS]
        fixed_models += [("Martingale", close)]
        for label, pred in fixed_models:
            masked = pred.copy()
            masked[:test_start] = np.nan
            row, _, _ = metrics(label, masked, target, close)
            row["evaluation"] = f"H={h} rows"
            rows.append(row)

    results = pd.DataFrame(rows)[
        ["evaluation", "model", "n", "mae_bps", "median_ae_bps", "rmse_bps", "bias_bps", "direction_accuracy_pct"]
    ]
    results.to_csv(OUT / "comparison_results.csv", index=False)
    bic_table.to_csv(OUT / "ar_order_selection.csv", index=False)

    valid_test = np.arange(n) >= test_start
    valid_test &= np.arange(n) + dynamic_h < n
    pred_frame = pd.DataFrame({
        "ts_event": data.loc[valid_test, "ts_event"].astype(str).to_numpy(),
        "close_t": close[valid_test],
        "selected_window": dynamic_w[valid_test],
        "selected_order": dynamic_o[valid_test],
        "selected_horizon_rows": dynamic_h[valid_test],
        "actual_close_t_plus_h": target_dynamic[valid_test],
        "kinematic_prediction": dynamic_prediction[valid_test],
    })
    for p in AR_ORDERS:
        pred_frame[f"ar_{p}_prediction_same_horizon"] = ar_dynamic[p][valid_test]
    pred_frame.to_csv(OUT / "walk_forward_predictions.csv", index=False)

    dynamic_results = results[results["evaluation"] == "Dynamic selected horizon"].set_index("model")
    kin_name = "Dynamic kinematic"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    labels = [kin_name] + [f"AR({p})" for p in AR_ORDERS] + ["Martingale"]
    x = np.arange(len(labels))
    width = 0.36
    axes[0, 0].bar(x - width / 2, dynamic_results.loc[labels, "mae_bps"], width, label="MAE")
    axes[0, 0].bar(x + width / 2, dynamic_results.loc[labels, "rmse_bps"], width, label="RMSE")
    axes[0, 0].set_xticks(x, labels, rotation=12)
    axes[0, 0].set_ylabel("Error (basis points of current price)")
    axes[0, 0].set_title("Dynamic-horizon forecast error")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.25)

    ar_rows = dynamic_results.loc[[f"AR({p})" for p in AR_ORDERS]]
    axes[0, 1].plot(AR_ORDERS, ar_rows["mae_bps"], marker="o", label="AR MAE")
    axes[0, 1].plot(AR_ORDERS, ar_rows["rmse_bps"], marker="o", label="AR RMSE")
    axes[0, 1].axhline(dynamic_results.loc[kin_name, "rmse_bps"], color="tab:red", linestyle="--", label="Kinematic RMSE")
    axes[0, 1].axhline(dynamic_results.loc["Martingale", "rmse_bps"], color="tab:gray", linestyle=":", label="Martingale RMSE")
    axes[0, 1].set_xlabel("AR order p")
    axes[0, 1].set_xticks(AR_ORDERS)
    axes[0, 1].set_ylabel("Error (bps)")
    axes[0, 1].set_title("Error across AR orders")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].grid(alpha=0.25)

    fixed_names = ["Kinematic"] + [f"AR({p})" for p in AR_ORDERS] + ["Martingale"]
    fixed = results[results["evaluation"].str.startswith("H=") & results["model"].isin(fixed_names)]
    for model, group in fixed.groupby("model"):
        hs = group["evaluation"].str.extract(r"H=(\d+)")[0].astype(int)
        axes[1, 0].plot(hs, group["rmse_bps"], marker="o", label=model)
    axes[1, 0].set_xticks(HORIZONS)
    axes[1, 0].set_xlabel("Forecast horizon (rows)")
    axes[1, 0].set_ylabel("RMSE (bps)")
    axes[1, 0].set_title("Error by fixed horizon")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.25)

    sel = pd.Series(dynamic_h[valid_test]).value_counts(normalize=True).sort_index() * 100
    axes[1, 1].bar(sel.index.astype(str), sel.values)
    axes[1, 1].set_xlabel("Selected horizon (rows)")
    axes[1, 1].set_ylabel("Selections (%)")
    axes[1, 1].set_title("Kinematic model horizon selection")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("MSFT 1-minute walk-forward comparison, final 20% holdout", fontsize=15)
    fig.savefig(OUT / "kinematic_vs_ar_plots.png", dpi=180)
    plt.close(fig)

    summary = {
        "dataset_rows": n,
        "dataset_start": str(data["ts_event"].min()),
        "dataset_end": str(data["ts_event"].max()),
        "test_start": str(data.loc[test_start, "ts_event"]),
        "test_rows": n - test_start,
        "ar_orders": AR_ORDERS,
        "kinematic_config": {"windows": WINDOWS, "orders": ORDERS, "horizons": HORIZONS, "score_lookback": SCORE_LOOKBACK},
        "ar_config": {"orders": AR_ORDERS, "history": AR_HISTORY, "refit_every_rows": AR_REFIT_EVERY, "modeled_series": "log returns"},
        "selection_pct": {
            "window": (pd.Series(dynamic_w[valid_test]).value_counts(normalize=True).sort_index() * 100).round(3).to_dict(),
            "order": (pd.Series(dynamic_o[valid_test]).value_counts(normalize=True).sort_index() * 100).round(3).to_dict(),
            "horizon": sel.round(3).to_dict(),
        },
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
