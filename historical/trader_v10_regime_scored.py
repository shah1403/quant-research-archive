
from datamodel import Order
import math


class Trader:
    def __init__(self):
        self.products = ["EMERALDS", "TOMATOES"]

        self.position_limits = {
            "EMERALDS": 50,
            "TOMATOES": 80,
        }

        self.max_hist = 450
        self.alpha = 0.30

        self.series = {}
        for p in self.products:
            self.series[p] = {
                "vwap": [],
                "ma2": [],
                "ema": [],
                "shock_cooldown": 0,
            }

        self.feature_window = 40
        self.anchor_window = 20
        self.window_pool = [10, 20, 40, 80, 120, 200]

        self.flat_dr = 0.10
        self.flat_dp = 0.25

        self.trend_dp = 0.60
        self.trend_dr = 0.18
        self.trend_ms = 1.00

        self.shock_threshold = 2.0
        self.shock_cooldown_ticks = 8

        self.pred_clip_k = 1.2
        self.pred_conf_threshold = 0.20

        self.w_fit = 0.20
        self.w_dir = 0.40
        self.w_kin = 0.20
        self.w_resid = 0.10
        self.w_curve = 0.07
        self.w_shock = 0.03

        self.eps = 1e-9

    def trim(self, arr):
        if len(arr) > self.max_hist:
            del arr[0: len(arr) - self.max_hist]

    def mean(self, arr):
        return sum(arr) / len(arr) if arr else 0.0

    def std(self, arr):
        if len(arr) <= 1:
            return 0.0
        mu = self.mean(arr)
        return math.sqrt(sum((x - mu) * (x - mu) for x in arr) / len(arr))

    def sign(self, x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def diff(self, arr):
        out = []
        for i in range(1, len(arr)):
            out.append(arr[i] - arr[i - 1])
        return out

    def abs_mean(self, arr):
        return self.mean([abs(x) for x in arr])

    def clip(self, x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def half_up_round(self, x):
        return int(math.floor(x + 0.5))

    def ac1(self, arr):
        if len(arr) < 3:
            return 0.0
        mu = self.mean(arr)
        num = 0.0
        den = 0.0
        for i in range(len(arr) - 1):
            num += (arr[i] - mu) * (arr[i + 1] - mu)
        for x in arr:
            den += (x - mu) * (x - mu)
        if abs(den) < self.eps:
            return 0.0
        return num / den

    def compute_vwap(self, order_depth):
        total_pq = 0.0
        total_q = 0.0

        for price, qty in order_depth.buy_orders.items():
            q = abs(qty)
            total_pq += price * q
            total_q += q

        for price, qty in order_depth.sell_orders.items():
            q = abs(qty)
            total_pq += price * q
            total_q += q

        if total_q <= 0:
            return None
        return total_pq / total_q

    def get_best_bid_ask(self, order_depth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def update_series(self, product, vwap):
        S = self.series[product]

        S["vwap"].append(vwap)
        self.trim(S["vwap"])

        if len(S["vwap"]) >= 2:
            ma2 = 0.5 * (S["vwap"][-1] + S["vwap"][-2])
        else:
            ma2 = vwap
        S["ma2"].append(ma2)
        self.trim(S["ma2"])

        if not S["ema"]:
            ema = ma2
        else:
            ema = self.alpha * ma2 + (1.0 - self.alpha) * S["ema"][-1]
        S["ema"].append(ema)
        self.trim(S["ema"])

        if S["shock_cooldown"] > 0:
            S["shock_cooldown"] -= 1

        return ma2, ema

    def compute_window_stats(self, series, W):
        if len(series) < max(W, 5):
            return None

        x = series[-W:]
        d = self.diff(x)
        if len(d) < 3:
            return None

        dd = self.diff(d) if len(d) >= 3 else []

        dp = abs(self.mean([self.sign(v) for v in d]))
        ms = self.std([abs(v) for v in d]) / (self.abs_mean(d) + self.eps)
        dr = abs(self.mean(d)) / (self.std(d) + self.eps)
        rm = self.abs_mean(d) / (abs(self.mean(x)) + self.eps)

        if dd:
            cp = abs(self.mean([self.sign(v) for v in dd]))
            ca = abs(self.mean(dd)) / (self.std(dd) + self.eps)
        else:
            cp = 0.0
            ca = 0.0

        shock = abs(d[-1] - self.mean(d)) / (self.std(d) + self.eps)

        return {
            "dp": dp,
            "ms": ms,
            "dr": dr,
            "rm": rm,
            "cp": cp,
            "ca": ca,
            "shock": shock,
            "sigma_d": self.std(d),
            "mean_d": self.mean(d),
            "mean_dd": self.mean(dd) if dd else 0.0,
            "d": d,
            "dd": dd,
        }

    def allowed_orders(self, stats):
        if (
            stats["dp"] > 0.65
            and stats["dr"] > 0.18
            and stats["ms"] < 1.00
            and stats["ca"] > 0.10
        ):
            return [1, 2]
        return [1]

    def stable_windows(self, series):
        if len(series) < self.anchor_window:
            return [10]

        anchor = self.compute_window_stats(series, self.anchor_window)
        if anchor is None:
            return [10]

        stable = []
        for W in self.window_pool:
            if len(series) < W:
                continue
            ws = self.compute_window_stats(series, W)
            if ws is None:
                continue

            se = (
                0.40 * abs(ws["dp"] - anchor["dp"])
                + 0.30 * abs(ws["ms"] - anchor["ms"])
                + 0.30 * abs(ws["dr"] - anchor["dr"])
            )

            if W <= 20 or se < 0.25:
                stable.append(W)

        if not stable:
            return [10]

        stable = sorted(stable)
        if len(stable) <= 3:
            return stable
        return stable[-3:]

    def fit_linear_wls(self, series, W):
        if len(series) < W:
            return None

        y = series[-W:]
        x = list(range(W))
        w = [math.exp(-0.04 * (W - 1 - i)) for i in range(W)]

        S = sum(w)
        Sx = sum(w[i] * x[i] for i in range(W))
        Sy = sum(w[i] * y[i] for i in range(W))
        Sxx = sum(w[i] * x[i] * x[i] for i in range(W))
        Sxy = sum(w[i] * x[i] * y[i] for i in range(W))

        denom = S * Sxx - Sx * Sx
        if abs(denom) < 1e-12:
            return None

        a = (S * Sxy - Sx * Sy) / denom
        b = (Sy - a * Sx) / S
        return ("linear", a, b)

    def fit_quadratic_wls(self, series, W):
        if len(series) < W:
            return None

        y = series[-W:]
        x = list(range(W))
        w = [math.exp(-0.04 * (W - 1 - i)) for i in range(W)]

        S0 = S1 = S2 = S3 = S4 = 0.0
        T0 = T1 = T2 = 0.0

        for i in range(W):
            wi = w[i]
            xi = x[i]
            yi = y[i]
            x2 = xi * xi

            S0 += wi
            S1 += wi * xi
            S2 += wi * x2
            S3 += wi * x2 * xi
            S4 += wi * x2 * x2

            T0 += wi * yi
            T1 += wi * xi * yi
            T2 += wi * x2 * yi

        A = [
            [S4, S3, S2, T2],
            [S3, S2, S1, T1],
            [S2, S1, S0, T0],
        ]

        for col in range(3):
            pivot = col
            for row in range(col + 1, 3):
                if abs(A[row][col]) > abs(A[pivot][col]):
                    pivot = row
            if abs(A[pivot][col]) < 1e-12:
                return None
            if pivot != col:
                A[col], A[pivot] = A[pivot], A[col]

            div = A[col][col]
            for j in range(col, 4):
                A[col][j] /= div

            for row in range(3):
                if row == col:
                    continue
                factor = A[row][col]
                for j in range(col, 4):
                    A[row][j] -= factor * A[col][j]

        a = A[0][3]
        b = A[1][3]
        c = A[2][3]
        return ("quadratic", a, b, c)

    def fit_model(self, series, W, order):
        if order == 1:
            return self.fit_linear_wls(series, W)
        if order == 2:
            return self.fit_quadratic_wls(series, W)
        return None

    def eval_model(self, model, x):
        if model is None:
            return 0.0
        if model[0] == "linear":
            _, a, b = model
            return a * x + b
        _, a, b, c = model
        return a * x * x + b * x + c

    def kinematics(self, model, W):
        x = W - 1
        if model is None:
            return 0.0, 0.0, 0.0

        cur = self.eval_model(model, x)

        if model[0] == "linear":
            _, a, _ = model
            return cur, a, 0.0

        _, qa, qb, _ = model
        v = 2.0 * qa * x + qb
        a = 2.0 * qa
        return cur, v, a

    def residual_metrics(self, series, model, W):
        y = series[-W:]
        resid = []
        for i in range(W):
            resid.append(y[i] - self.eval_model(model, i))

        resid_std = self.std(resid)
        resid_ac1 = self.ac1(resid)

        flip_count = 0
        for i in range(1, len(resid)):
            if self.sign(resid[i]) != self.sign(resid[i - 1]):
                flip_count += 1
        flip_rate = float(flip_count) / float(max(1, len(resid) - 1))

        return {
            "resid_std": resid_std,
            "resid_ac1": resid_ac1,
            "resid_flip_rate": flip_rate,
            "resid": resid,
        }

    def directional_accuracy(self, series, model, W):
        if len(series) < W + 4:
            return 0.0

        y = series[-(W + 4):]
        sub = y[:W]
        temp_model = self.fit_model(sub, W, 1 if model[0] == "linear" else 2)
        if temp_model is None:
            return 0.0

        weights = [0.5, 0.3, 0.2]
        horizons = [1, 2, 3]
        score = 0.0

        pred_now = self.eval_model(temp_model, W - 1)
        current_actual = y[W - 1]

        for i in range(3):
            h = horizons[i]
            pred_future = self.eval_model(temp_model, (W - 1) + h)
            pred_dir = self.sign(pred_future - pred_now)
            act_dir = self.sign(y[W - 1 + h] - current_actual)
            if pred_dir == act_dir:
                score += weights[i]

        return score

    def score_candidate(self, series, stats, model, W, order):
        if model is None:
            return -10**9

        resid = self.residual_metrics(series, model, W)
        _, v, a = self.kinematics(model, W)
        sigma_d = stats["sigma_d"] + self.eps

        fit_quality = -resid["resid_std"] / sigma_d
        dir_use = self.directional_accuracy(series, model, W)

        kin_score = 0.0
        if self.sign(v) == self.sign(stats["mean_d"]):
            kin_score += 0.7
        if self.sign(a) == self.sign(stats["mean_dd"]):
            kin_score += 0.3

        resid_pen = (
            resid["resid_std"] / sigma_d
            + max(0.0, abs(resid["resid_ac1"]) - 0.15)
        )

        curve_pen = 0.0
        if order == 2:
            curve_pen = abs(a) / sigma_d

        shock_pen = 1.0 if stats["shock"] > self.shock_threshold else 0.0

        score = (
            self.w_fit * fit_quality
            + self.w_dir * dir_use
            + self.w_kin * kin_score
            - self.w_resid * resid_pen
            - self.w_curve * curve_pen
            - self.w_shock * shock_pen
        )

        return score

    def choose_best_model(self, series):
        base_stats = self.compute_window_stats(series, min(self.feature_window, len(series)))
        if base_stats is None:
            return None

        orders = self.allowed_orders(base_stats)
        windows = self.stable_windows(series)

        best = None
        best_score = -10**9

        for order in orders:
            for W in windows:
                if len(series) < W + 4:
                    continue
                ws = self.compute_window_stats(series, W)
                if ws is None:
                    continue
                model = self.fit_model(series, W, order)
                if model is None:
                    continue
                score = self.score_candidate(series, ws, model, W, order)

                if score > best_score:
                    residuals = self.residual_metrics(series, model, W)
                    cur, v, a = self.kinematics(model, W)
                    best_score = score
                    best = {
                        "model": model,
                        "window": W,
                        "order": order,
                        "score": score,
                        "stats": ws,
                        "residuals": residuals,
                        "current_fit": cur,
                        "velocity": v,
                        "acceleration": a,
                    }

        return best

    def classify_regime(self, product, best):
        if best is None:
            return "FLAT_MR"

        S = self.series[product]
        stats = best["stats"]
        resid = best["residuals"]

        residual_std_ratio = resid["resid_std"] / (stats["sigma_d"] + self.eps)
        shock_cooldown = S["shock_cooldown"]

        if stats["shock"] > self.shock_threshold:
            S["shock_cooldown"] = self.shock_cooldown_ticks
            shock_cooldown = S["shock_cooldown"]

        if (
            stats["ms"] > 1.10
            or stats["shock"] > self.shock_threshold
            or residual_std_ratio > 1.20
            or shock_cooldown > 0
        ):
            return "VARIABLE_RW"

        if (
            stats["dp"] > self.trend_dp
            and stats["dr"] > self.trend_dr
            and stats["ms"] < self.trend_ms
            and residual_std_ratio < 1.00
        ):
            return "SMOOTH_TREND"

        if stats["dr"] < self.flat_dr and stats["dp"] < self.flat_dp:
            return "FLAT_MR"

        return "FLAT_MR"

    def predict_next(self, best):
        if best is None:
            return {
                "pred_next": None,
                "signal": 0.0,
                "strength": 0.0,
            }

        x = best["current_fit"]
        v = best["velocity"]
        a = best["acceleration"]
        sigma = best["stats"]["sigma_d"] + self.eps

        delta_hat = v + 0.5 * a
        delta_hat = self.clip(delta_hat, -self.pred_clip_k * sigma, self.pred_clip_k * sigma)

        conf = abs(v) / sigma
        w = min(1.0, conf / self.pred_conf_threshold)

        signal = w * delta_hat
        pred_next = x + signal
        strength = abs(signal) / sigma

        return {
            "pred_next": pred_next,
            "signal": signal,
            "strength": strength,
        }

    def build_orders(self, product, order_depth, position, fv, regime, pred):
        orders = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        limit = self.position_limits[product]
        buy_cap = max(0, limit - position)
        sell_cap = max(0, limit + position)

        if best_bid >= best_ask:
            return orders

        signal = pred["signal"]
        strength = pred["strength"]
        directional_bias = self.sign(signal)

        base_size = 8 if product == "EMERALDS" else 5
        buy_size = base_size
        sell_size = base_size

        bid_price = best_bid + 1
        ask_price = best_ask - 1

        if bid_price >= ask_price:
            bid_price = best_bid
            ask_price = best_ask

        inv_frac = float(position) / float(limit) if limit > 0 else 0.0

        if regime == "FLAT_MR":
            if inv_frac > 0.50:
                buy_size = max(1, buy_size - 3)
                sell_size = sell_size + 2
                ask_price = max(best_bid + 1, ask_price - 1)
            elif inv_frac < -0.50:
                buy_size = buy_size + 2
                sell_size = max(1, sell_size - 3)
                bid_price = min(best_ask - 1, bid_price + 1)

        elif regime == "SMOOTH_TREND":
            if directional_bias > 0:
                buy_size += 1
                if position < 0:
                    bid_price = min(best_ask - 1, bid_price + 1)
            elif directional_bias < 0:
                sell_size += 1
                if position > 0:
                    ask_price = max(best_bid + 1, ask_price - 1)

        elif regime == "VARIABLE_RW":
            buy_size = max(1, buy_size - 2)
            sell_size = max(1, sell_size - 2)

            if inv_frac > 0.30:
                buy_size = 1
                sell_size += 2
                ask_price = max(best_bid + 1, ask_price - 1)
            elif inv_frac < -0.30:
                sell_size = 1
                buy_size += 2
                bid_price = min(best_ask - 1, bid_price + 1)

        if buy_cap > 0 and buy_size > 0:
            orders.append(Order(product, bid_price, min(buy_size, buy_cap)))
        if sell_cap > 0 and sell_size > 0:
            orders.append(Order(product, ask_price, -min(sell_size, sell_cap)))

        if regime == "SMOOTH_TREND" and strength > 0.50:
            if directional_bias > 0 and best_ask < fv and buy_cap > 0:
                take_qty = min(3, buy_cap)
                if take_qty > 0:
                    orders.append(Order(product, best_ask, take_qty))
            elif directional_bias < 0 and best_bid > fv and sell_cap > 0:
                take_qty = min(3, sell_cap)
                if take_qty > 0:
                    orders.append(Order(product, best_bid, -take_qty))

        elif regime == "FLAT_MR":
            if best_ask + 1 < fv and buy_cap > 0:
                take_qty = min(2, buy_cap)
                if take_qty > 0:
                    orders.append(Order(product, best_ask, take_qty))
            if best_bid - 1 > fv and sell_cap > 0:
                take_qty = min(2, sell_cap)
                if take_qty > 0:
                    orders.append(Order(product, best_bid, -take_qty))

        return orders

    def run(self, state):
        result = {}
        conversions = 0
        traderData = ""

        for product in state.order_depths:
            if product not in self.products:
                result[product] = []
                continue

            order_depth = state.order_depths[product]
            vwap = self.compute_vwap(order_depth)

            if vwap is None:
                result[product] = []
                continue

            ma2, _ = self.update_series(product, vwap)
            fv = self.half_up_round(ma2)

            best = self.choose_best_model(self.series[product]["ema"])
            regime = self.classify_regime(product, best)
            pred = self.predict_next(best)

            position = state.position.get(product, 0)
            orders = self.build_orders(
                product=product,
                order_depth=order_depth,
                position=position,
                fv=fv,
                regime=regime,
                pred=pred,
            )
            result[product] = orders

        return result, conversions, traderData
