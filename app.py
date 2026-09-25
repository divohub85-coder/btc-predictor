import time
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template

app = Flask(__name__)

SYMBOL = "BTCUSDT"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DB_FILE = "predictor.db"

TIMEFRAMES = ["5m", "15m", "1h", "1d"]


# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            candle_open_time INTEGER NOT NULL,
            candle_close_time INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            bullish_prob REAL NOT NULL,
            bearish_prob REAL NOT NULL,
            confidence REAL NOT NULL,
            price_at_prediction REAL NOT NULL,
            created_at TEXT NOT NULL,
            actual_direction TEXT,
            next_close REAL,
            is_correct INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicator_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            indicator TEXT NOT NULL,
            vote TEXT NOT NULL,
            is_correct INTEGER,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uniq_pred
        ON predictions(symbol, interval, candle_open_time)
    """)
    conn.commit()
    conn.close()


def save_prediction(interval, open_time, close_time, pred, price, indicator_votes):
    try:
        conn = get_db()
        cur = conn.execute("""
            INSERT OR IGNORE INTO predictions
            (symbol, interval, candle_open_time, candle_close_time,
             predicted_direction, bullish_prob, bearish_prob, confidence,
             price_at_prediction, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            SYMBOL, interval, open_time, close_time,
            pred["direction"], pred["bullish_prob"], pred["bearish_prob"],
            pred["confidence"], price,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

        pred_id = cur.lastrowid
        if pred_id and indicator_votes:
            for ind, vote in indicator_votes.items():
                conn.execute("""
                    INSERT INTO indicator_votes (prediction_id, indicator, vote)
                    VALUES (?, ?, ?)
                """, (pred_id, ind, vote))
            conn.commit()

        conn.close()
    except Exception as e:
        print(f"Save prediction error: {e}")


def evaluate_predictions():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE is_correct IS NULL AND candle_close_time < ?
            ORDER BY candle_close_time DESC LIMIT 50
        """, (int(time.time() * 1000),)).fetchall()

        for row in rows:
            try:
                raw = requests.get(
                    BINANCE_KLINES_URL,
                    params={
                        "symbol": SYMBOL,
                        "interval": row["interval"],
                        "startTime": row["candle_close_time"],
                        "limit": 2
                    },
                    timeout=10
                ).json()

                if not raw:
                    continue

                next_close = float(raw[0][4])
                price_at_pred = float(row["price_at_prediction"])

                actual = "BULLISH" if next_close > price_at_pred else "BEARISH"
                is_correct = 1 if actual == row["predicted_direction"] else 0

                conn.execute("""
                    UPDATE predictions
                    SET actual_direction = ?, next_close = ?, is_correct = ?
                    WHERE id = ?
                """, (actual, next_close, is_correct, row["id"]))

                # Update indicator votes
                votes = conn.execute("""
                    SELECT * FROM indicator_votes WHERE prediction_id = ?
                """, (row["id"],)).fetchall()

                for v in votes:
                    if v["vote"] == "NEUTRAL":
                        conn.execute("""
                            UPDATE indicator_votes SET is_correct = 0 WHERE id = ?
                        """, (v["id"],))
                    else:
                        ind_correct = 1 if v["vote"] == actual else 0
                        conn.execute("""
                            UPDATE indicator_votes SET is_correct = ? WHERE id = ?
                        """, (ind_correct, v["id"]))

            except Exception as e:
                print(f"Eval error: {e}")
                continue

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Evaluate error: {e}")


def get_accuracy_stats():
    try:
        conn = get_db()
        stats = {}
        for limit in [50, 100, 500]:
            rows = conn.execute("""
                SELECT is_correct FROM predictions
                WHERE is_correct IS NOT NULL
                ORDER BY candle_close_time DESC LIMIT ?
            """, (limit,)).fetchall()

            if not rows:
                stats[f"last_{limit}"] = None
                continue

            correct = sum(1 for r in rows if r["is_correct"] == 1)
            total = len(rows)
            stats[f"last_{limit}"] = {
                "correct": correct,
                "total": total,
                "accuracy": round((correct / total) * 100, 1)
            }

        for tf in TIMEFRAMES:
            rows = conn.execute("""
                SELECT is_correct FROM predictions
                WHERE interval = ? AND is_correct IS NOT NULL
                ORDER BY candle_close_time DESC LIMIT 100
            """, (tf,)).fetchall()

            if rows:
                correct = sum(1 for r in rows if r["is_correct"] == 1)
                stats[f"tf_{tf}"] = {
                    "correct": correct,
                    "total": len(rows),
                    "accuracy": round((correct / len(rows)) * 100, 1)
                }
            else:
                stats[f"tf_{tf}"] = None

        conn.close()
        return stats
    except Exception as e:
        return {"error": str(e)}


def get_indicator_accuracy():
    """Har indicator ki accuracy"""
    try:
        conn = get_db()
        indicators = ["ema", "rsi", "macd", "volume", "candle",
                      "psar", "bb", "williams_r", "adx", "vwap"]

        result = {}
        for ind in indicators:
            rows = conn.execute("""
                SELECT is_correct FROM indicator_votes
                WHERE indicator = ? AND is_correct IS NOT NULL
                ORDER BY id DESC LIMIT 200
            """, (ind,)).fetchall()

            if not rows:
                result[ind] = None
                continue

            correct = sum(1 for r in rows if r["is_correct"] == 1)
            total = len(rows)
            result[ind] = {
                "correct": correct,
                "total": total,
                "accuracy": round((correct / total) * 100, 1)
            }

        conn.close()
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# BINANCE DATA
# ============================================================
def get_klines(symbol, interval, limit=300):
    try:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Binance API failed: {e}")

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(data, columns=cols)

    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])

    return df


# ============================================================
# INDICATORS
# ============================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def calc_williams_r(df, period=14):
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def calc_adx(df, period=14):
    high = df["high"]; low = df["low"]; close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


def calc_parabolic_sar(df, af=0.02, max_af=0.2):
    high = df["high"].values; low = df["low"].values
    n = len(df)

    sar = np.zeros(n); trend = np.zeros(n)
    ep = np.zeros(n); af_arr = np.zeros(n)

    sar[0] = low[0]; trend[0] = 1
    ep[0] = high[0]; af_arr[0] = af

    for i in range(1, n):
        prev_sar = sar[i-1]; prev_trend = trend[i-1]
        prev_ep = ep[i-1]; prev_af = af_arr[i-1]

        cur_sar = prev_sar + prev_af * (prev_ep - prev_sar)

        if prev_trend == 1:
            lookback = low[i-1] if i < 2 else min(low[i-1], low[i-2])
            cur_sar = min(cur_sar, lookback)
            if low[i] < cur_sar:
                trend[i] = -1; cur_sar = prev_ep
                ep[i] = low[i]; af_arr[i] = af
            else:
                trend[i] = 1
                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af_arr[i] = min(prev_af + af, max_af)
                else:
                    ep[i] = prev_ep; af_arr[i] = prev_af
        else:
            lookback = high[i-1] if i < 2 else max(high[i-1], high[i-2])
            cur_sar = max(cur_sar, lookback)
            if high[i] > cur_sar:
                trend[i] = 1; cur_sar = prev_ep
                ep[i] = high[i]; af_arr[i] = af
            else:
                trend[i] = -1
                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af_arr[i] = min(prev_af + af, max_af)
                else:
                    ep[i] = prev_ep; af_arr[i] = prev_af

        sar[i] = cur_sar

    return sar, trend


def calc_bollinger(df, period=20, std_dev=2):
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower


def calc_vwap(df, period=20):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]
    vwap = tp_vol.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)
    return vwap


def add_indicators(df):
    df = df.copy()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["rsi"] = calc_rsi(df["close"], 14)
    df["rsi_prev"] = df["rsi"].shift(1)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    df["atr"] = calc_atr(df, 14)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
    df["buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)

    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = df["body"] / df["range"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    psar, psar_trend = calc_parabolic_sar(df)
    df["psar"] = psar
    df["psar_trend"] = psar_trend

    upper, mid, lower = calc_bollinger(df)
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower

    df["williams_r"] = calc_williams_r(df, 14)
    df["williams_r_prev"] = df["williams_r"].shift(1)

    adx, plus_di, minus_di = calc_adx(df, 14)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["vwap"] = calc_vwap(df, 20)

    return df


def get_trend(row):
    if row["close"] > row["ema_9"] > row["ema_21"] > row["ema_50"]:
        return "UPTREND"
    if row["close"] < row["ema_9"] < row["ema_21"] < row["ema_50"]:
        return "DOWNTREND"
    if row["close"] > row["ema_50"]:
        return "WEAK_UPTREND"
    if row["close"] < row["ema_50"]:
        return "WEAK_DOWNTREND"
    return "SIDEWAYS"


def get_market_regime(row, trend):
    high_vol = row["atr_pct"] >= 1.0
    low_vol = row["atr_pct"] <= 0.3
    if high_vol and trend in ["UPTREND", "DOWNTREND"]:
        return "TRENDING_HIGH_VOL"
    if trend in ["UPTREND", "DOWNTREND"]:
        return "TRENDING"
    if low_vol:
        return "LOW_VOL_RANGE"
    return "SIDEWAYS_CHOPPY"


# ============================================================
# PREDICTION ENGINE (11-indicator voting)
# ============================================================
def analyze_candle(df, is_live=False):
    if len(df) < 60:
        return None

    last = df.iloc[-1]
    trend = get_trend(last)
    regime = get_market_regime(last, trend)

    bull = 0.0
    bear = 0.0
    votes_bull = 0
    votes_bear = 0
    total_votes = 0
    reasons = []
    warnings = []
    ind_votes = {}  # Har indicator ka vote

    # 1. EMA
    total_votes += 1
    if trend == "UPTREND":
        bull += 18; votes_bull += 1
        ind_votes["ema"] = "BULLISH"
        reasons.append("EMA bullish")
    elif trend == "DOWNTREND":
        bear += 18; votes_bear += 1
        ind_votes["ema"] = "BEARISH"
        reasons.append("EMA bearish")
    elif trend == "WEAK_UPTREND":
        bull += 8; votes_bull += 1
        ind_votes["ema"] = "BULLISH"
    elif trend == "WEAK_DOWNTREND":
        bear += 8; votes_bear += 1
        ind_votes["ema"] = "BEARISH"
    else:
        ind_votes["ema"] = "NEUTRAL"

    # 2. RSI
    rsi = float(last["rsi"]); rsi_prev = float(last["rsi_prev"])
    total_votes += 1
    if 52 <= rsi <= 72 and rsi > rsi_prev:
        bull += 11; votes_bull += 1
        ind_votes["rsi"] = "BULLISH"
        reasons.append(f"RSI {rsi:.1f} rising")
    elif 28 <= rsi <= 48 and rsi < rsi_prev:
        bear += 11; votes_bear += 1
        ind_votes["rsi"] = "BEARISH"
        reasons.append(f"RSI {rsi:.1f} falling")
    elif rsi < 28:
        bull += 5; votes_bull += 1
        ind_votes["rsi"] = "BULLISH"
        reasons.append(f"RSI {rsi:.1f} oversold")
    elif rsi > 72:
        bear += 5; votes_bear += 1
        ind_votes["rsi"] = "BEARISH"
        reasons.append(f"RSI {rsi:.1f} overbought")
    else:
        ind_votes["rsi"] = "NEUTRAL"

    # 3. MACD
    total_votes += 1
    if last["macd"] > last["macd_signal"] and last["macd_hist"] > last["macd_hist_prev"]:
        bull += 12; votes_bull += 1
        ind_votes["macd"] = "BULLISH"
        reasons.append("MACD bullish")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < last["macd_hist_prev"]:
        bear += 12; votes_bear += 1
        ind_votes["macd"] = "BEARISH"
        reasons.append("MACD bearish")
    else:
        ind_votes["macd"] = "NEUTRAL"

    # 4. Volume
    vol_ratio = float(last["vol_ratio"])
    buy_ratio = float(last["buy_ratio"])
    is_green = last["close"] > last["open"]
    total_votes += 1
    if vol_ratio >= 1.2:
        if is_green and buy_ratio >= 0.52:
            bull += 12; votes_bull += 1
            ind_votes["volume"] = "BULLISH"
            reasons.append(f"Green + {vol_ratio:.2f}x vol")
        elif not is_green and buy_ratio <= 0.48:
            bear += 12; votes_bear += 1
            ind_votes["volume"] = "BEARISH"
            reasons.append(f"Red + {vol_ratio:.2f}x vol")
        else:
            ind_votes["volume"] = "NEUTRAL"
    else:
        ind_votes["volume"] = "NEUTRAL"

    # 5. Candle
    body_ratio = float(last["body_ratio"])
    upper_wick = float(last["upper_wick"])
    lower_wick = float(last["lower_wick"])
    cr = float(last["range"]) if last["range"] > 0 else 1
    total_votes += 1
    if is_green and body_ratio >= 0.55 and lower_wick <= cr * 0.2:
        bull += 6; votes_bull += 1
        ind_votes["candle"] = "BULLISH"
        reasons.append("Strong bullish body")
    elif not is_green and body_ratio >= 0.55 and upper_wick <= cr * 0.2:
        bear += 6; votes_bear += 1
        ind_votes["candle"] = "BEARISH"
        reasons.append("Strong bearish body")
    else:
        ind_votes["candle"] = "NEUTRAL"

    # 6. PSAR
    total_votes += 1
    if last["psar_trend"] == 1:
        bull += 10; votes_bull += 1
        ind_votes["psar"] = "BULLISH"
        reasons.append("PSAR bullish")
    else:
        bear += 10; votes_bear += 1
        ind_votes["psar"] = "BEARISH"
        reasons.append("PSAR bearish")

    # 7. BB
    total_votes += 1
    if pd.notna(last["bb_upper"]):
        if last["close"] > last["bb_upper"]:
            bear += 8; votes_bear += 1
            ind_votes["bb"] = "BEARISH"
            reasons.append("Above BB upper")
        elif last["close"] < last["bb_lower"]:
            bull += 8; votes_bull += 1
            ind_votes["bb"] = "BULLISH"
            reasons.append("Below BB lower")
        elif last["close"] > last["bb_mid"]:
            bull += 4; votes_bull += 1
            ind_votes["bb"] = "BULLISH"
        else:
            bear += 4; votes_bear += 1
            ind_votes["bb"] = "BEARISH"
    else:
        ind_votes["bb"] = "NEUTRAL"

    # 8. Williams %R
    total_votes += 1
    wr = float(last["williams_r"]) if pd.notna(last["williams_r"]) else -50
    if wr < -80:
        bull += 10; votes_bull += 1
        ind_votes["williams_r"] = "BULLISH"
        reasons.append(f"Williams %R {wr:.0f} oversold")
    elif wr > -20:
        bear += 10; votes_bear += 1
        ind_votes["williams_r"] = "BEARISH"
        reasons.append(f"Williams %R {wr:.0f} overbought")
    elif -50 < wr < -20:
        bear += 4; votes_bear += 1
        ind_votes["williams_r"] = "BEARISH"
    elif -80 < wr < -50:
        bull += 4; votes_bull += 1
        ind_votes["williams_r"] = "BULLISH"
    else:
        ind_votes["williams_r"] = "NEUTRAL"

    # 9. ADX
    total_votes += 1
    adx = float(last["adx"]) if pd.notna(last["adx"]) else 0
    plus_di = float(last["plus_di"]) if pd.notna(last["plus_di"]) else 0
    minus_di = float(last["minus_di"]) if pd.notna(last["minus_di"]) else 0
    if adx > 25:
        if plus_di > minus_di:
            bull += 8; votes_bull += 1
            ind_votes["adx"] = "BULLISH"
            reasons.append(f"ADX {adx:.0f} strong uptrend")
        else:
            bear += 8; votes_bear += 1
            ind_votes["adx"] = "BEARISH"
            reasons.append(f"ADX {adx:.0f} strong downtrend")
    else:
        ind_votes["adx"] = "NEUTRAL"
        warnings.append(f"ADX {adx:.0f} weak trend")

    # 10. VWAP
    total_votes += 1
    vwap = float(last["vwap"]) if pd.notna(last["vwap"]) else last["close"]
    price = float(last["close"])
    vwap_diff_pct = ((price - vwap) / vwap) * 100 if vwap > 0 else 0

    if vwap_diff_pct > 0.3:
        bull += 10; votes_bull += 1
        ind_votes["vwap"] = "BULLISH"
        reasons.append(f"Price {vwap_diff_pct:.2f}% above VWAP")
    elif vwap_diff_pct < -0.3:
        bear += 10; votes_bear += 1
        ind_votes["vwap"] = "BEARISH"
        reasons.append(f"Price {abs(vwap_diff_pct):.2f}% below VWAP")
    elif vwap_diff_pct > 0.1:
        bull += 5; votes_bull += 1
        ind_votes["vwap"] = "BULLISH"
    elif vwap_diff_pct < -0.1:
        bear += 5; votes_bear += 1
        ind_votes["vwap"] = "BEARISH"
    else:
        ind_votes["vwap"] = "NEUTRAL"

    # Probability
    total = bull + bear
    if total <= 0:
        bull_prob = 50.0; bear_prob = 50.0
    else:
        bull_prob = (bull / total) * 100
        bear_prob = (bear / total) * 100

    vote_strength = abs(votes_bull - votes_bear) / total_votes if total_votes > 0 else 0
    diff = abs(bull_prob - bear_prob) / 100
    confidence = (vote_strength * 0.6 + diff * 0.4) * 100
    confidence = max(0, min(100, round(confidence, 1)))

    if bull_prob > bear_prob:
        direction = "BULLISH"
    elif bear_prob > bull_prob:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "is_live": is_live,
        "price": round(float(last["close"]), 2),
        "bullish_prob": round(bull_prob, 1),
        "bearish_prob": round(bear_prob, 1),
        "confidence": confidence,
        "direction": direction,
        "trend": trend,
        "regime": regime,
        "rsi": round(rsi, 1),
        "adx": round(adx, 1),
        "williams_r": round(wr, 1),
        "vwap": round(vwap, 2),
        "vwap_diff_pct": round(vwap_diff_pct, 2),
        "atr_pct": round(float(last["atr_pct"]), 2),
        "vol_ratio": round(vol_ratio, 2) if pd.notna(vol_ratio) else None,
        "buy_ratio": round(buy_ratio, 3) if pd.notna(buy_ratio) else None,
        "votes_bull": votes_bull,
        "votes_bear": votes_bear,
        "total_votes": total_votes,
        "indicator_votes": ind_votes,  # NEW
        "reasons": reasons[:6],
        "warnings": warnings,
    }


def analyze_timeframe(interval):
    raw = get_klines(SYMBOL, interval)
    now_ms = int(time.time() * 1000)

    closed_df = raw[raw["close_time"] < now_ms].copy()
    live_df = raw.copy()

    closed = add_indicators(closed_df)
    live = add_indicators(live_df)

    closed_result = analyze_candle(closed, is_live=False)
    live_result = analyze_candle(live, is_live=True)

    if closed_result and len(closed_df) >= 2:
        last_closed = closed_df.iloc[-1]
        save_prediction(
            interval,
            int(last_closed["open_time"]),
            int(last_closed["close_time"]),
            closed_result,
            float(last_closed["close"]),
            closed_result.get("indicator_votes", {})
        )

    return {
        "closed": closed_result,
        "live": live_result,
    }


def get_chart_data(interval, limit=50):
    raw = get_klines(SYMBOL, interval, limit=limit + 5)
    now_ms = int(time.time() * 1000)

    raw = raw.tail(limit).copy()
    raw["is_live"] = raw["close_time"] >= now_ms

    candles = []
    for _, row in raw.iterrows():
        candles.append({
            "time": int(row["open_time"] / 1000),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "is_live": bool(row["is_live"]),
        })

    return candles


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict")
def api_predict():
    try:
        evaluate_predictions()

        results = {}
        for tf in TIMEFRAMES:
            try:
                results[tf] = analyze_timeframe(tf)
            except Exception as e:
                results[tf] = {"error": str(e)}

        return jsonify({
            "symbol": SYMBOL,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timeframes": results,
            "accuracy": get_accuracy_stats(),
            "indicator_accuracy": get_indicator_accuracy(),  # NEW
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/<interval>")
def api_chart(interval):
    try:
        if interval not in TIMEFRAMES:
            return jsonify({"error": "Invalid interval"}), 400

        candles = get_chart_data(interval, limit=50)
        prediction = analyze_timeframe(interval)

        return jsonify({
            "interval": interval,
            "candles": candles,
            "prediction": prediction,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
