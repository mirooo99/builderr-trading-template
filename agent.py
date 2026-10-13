import math

DEFENSIVE = ["XLP", "XLU", "XLV", "XLE", "GLD", "TLT", "IEF"]
OFFENSIVE = [
    "SPY", "QQQ", "SMH",
    "XLK", "XLF", "XLY", "XLI", "XLB",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA",
]
CANDIDATES = OFFENSIVE + DEFENSIVE
TECH = {"QQQ", "SMH", "XLK", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "QLD"}
OVERLAY = {"QLD": 0.10, "SSO": 0.05}
BETA = {"QLD": 2.0, "SSO": 2.0, "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0}

MAX_WEIGHT = 0.24
DRIFT_WEIGHT = 0.27
MAX_BETA_GROSS = 1.35
TECH_CLUSTER_CAP = 0.60
TOP_N = 7
MAX_TECH_PICKS = 3
HELD_BONUS = 0.02
MIN_TRADE_FRAC = 0.025
CASH_BUFFER = 0.98
OFF_WEIGHT = 0.15
OFF_VOL_TARGET = 0.10
OFF_GROSS_MAX = 0.60
NEUTRAL_DEF_WEIGHT = 0.08
NEUTRAL_GROSS = 0.45
ON_GROSS_MAX = 0.92
ON_GROSS_MIN = 0.50
VOL_TARGET = 0.14


def _closes(ms, t):
    out = []
    for b in ms.get(t) or []:
        try:
            c = float(b["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if c > 0 and math.isfinite(c):
            out.append(c)
    return out


def _sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None


def _vol(c, n=20):
    if len(c) < n + 1:
        return None
    r = [math.log(c[i] / c[i - 1]) for i in range(len(c) - n, len(c))]
    m = sum(r) / n
    return math.sqrt(sum((x - m) ** 2 for x in r) / (n - 1)) * math.sqrt(252)


def _mom(c, n):
    return c[-1] / c[-1 - n] - 1.0 if len(c) > n else None


def _regime(ms, prev_on):
    spy = _closes(ms, "SPY")
    qqq = _closes(ms, "QQQ")
    spy50, spy200, qqq50 = _sma(spy, 50), _sma(spy, 200), _sma(qqq, 50)
    qvol = _vol(qqq, 20)
    if spy50 is None or qqq50 is None or qvol is None:
        return "off", qvol
    if spy[-1] < max(spy[-20:]) * 0.94 or spy[-1] / spy[-6] - 1.0 < -0.045 or qqq[-1] / qqq[-2] - 1.0 < -0.035:
        return "off", qvol
    band = 0.98 if prev_on else 1.005
    vlim = 0.40 if prev_on else 0.35
    if spy[-1] > spy50 * band and qqq[-1] > qqq50 * band and qvol < vlim:
        return "on", qvol
    if spy200 is not None and spy[-1] > spy200 and qvol < 0.45:
        return "neutral", qvol
    return "off", qvol


def _alloc(items, gross, cap):
    w = {}
    pool = dict(items)
    remaining = gross
    while pool and remaining > 1e-9:
        total = sum(pool.values())
        capped = False
        for t in list(pool):
            x = remaining * pool[t] / total
            if x >= cap:
                w[t] = cap
                remaining -= cap
                del pool[t]
                capped = True
        if not capped:
            for t, v in pool.items():
                w[t] = remaining * v / total
            break
    return w


def _picks(ms, held):
    scored = []
    for t in CANDIDATES:
        c = _closes(ms, t)
        if len(c) < 61:
            continue
        s50, v, m60, m20 = _sma(c, 50), _vol(c, 20), _mom(c, 60), _mom(c, 20)
        if None in (s50, v, m60, m20) or v <= 0:
            continue
        gap = c[-1] / s50 - 1.0
        if gap <= 0 or m60 <= 0:
            continue
        m120 = _mom(c, 120)
        if m120 is None:
            m120 = m60
        score = 0.35 * m60 + 0.20 * m20 + 0.25 * m120 + 0.20 * gap - 0.10 * v
        if t in held:
            score += HELD_BONUS
        scored.append((score, t, 1.0 / v))
    scored.sort(reverse=True)
    out = []
    tech_n = 0
    for _, t, iv in scored:
        if t in TECH:
            if tech_n >= MAX_TECH_PICKS:
                continue
            tech_n += 1
        out.append((t, iv))
        if len(out) >= TOP_N:
            break
    return out


def _merge(a, b):
    out = dict(a)
    for t, w in b.items():
        out[t] = out.get(t, 0.0) + w
    return out


def _defensive_book(ms, gross_max, per_cap):
    items = []
    for t in DEFENSIVE:
        c = _closes(ms, t)
        if len(c) < 51:
            continue
        s50, v, m20 = _sma(c, 50), _vol(c, 20), _mom(c, 20)
        if None in (s50, v, m20) or v <= 0:
            continue
        if c[-1] <= s50 or m20 < -0.02:
            continue
        items.append((t, 1.0 / v, v))
    if not items:
        return {}
    avg_v = sum(v for _, _, v in items) / len(items)
    gross = min(gross_max, OFF_VOL_TARGET / max(avg_v, 0.05))
    return _alloc([(t, iv) for t, iv, _ in items], gross, per_cap)


def _targets(ms, regime, qvol, held):
    if regime == "off":
        return _defensive_book(ms, OFF_GROSS_MAX, OFF_WEIGHT)
    picks = _picks(ms, held)
    if regime == "neutral":
        base = _alloc(picks, NEUTRAL_GROSS, MAX_WEIGHT) if picks else {}
        return _merge(base, _defensive_book(ms, 0.30, NEUTRAL_DEF_WEIGHT * 1.5))
    if not picks:
        return _defensive_book(ms, OFF_GROSS_MAX, OFF_WEIGHT)
    gross = min(ON_GROSS_MAX, max(ON_GROSS_MIN, VOL_TARGET / qvol)) if qvol else ON_GROSS_MIN
    targets = _alloc(picks, gross, MAX_WEIGHT)
    qqq = _closes(ms, "QQQ")
    q20, q50 = _sma(qqq, 20), _sma(qqq, 50)
    if qvol is not None and qvol < 0.22 and q20 and q50 and qqq[-1] > q20 > q50:
        for t, w in OVERLAY.items():
            if len(_closes(ms, t)) >= 2:
                targets[t] = w
    return targets


def _limit(targets):
    targets = {t: min(w, MAX_WEIGHT) for t, w in targets.items() if w > 0}
    tech = sum(w for t, w in targets.items() if t in TECH)
    if tech > TECH_CLUSTER_CAP:
        k = TECH_CLUSTER_CAP / tech
        targets = {t: (w * k if t in TECH else w) for t, w in targets.items()}
    gross = sum(w * BETA.get(t, 1.0) for t, w in targets.items())
    if gross > MAX_BETA_GROSS:
        k = MAX_BETA_GROSS / gross
        targets = {t: w * k for t, w in targets.items()}
    plain = sum(targets.values())
    if plain > CASH_BUFFER:
        k = CASH_BUFFER / plain
        targets = {t: w * k for t, w in targets.items()}
    return targets


def decide(market_state, portfolio_state, cash):
    positions = {}
    for p in portfolio_state.get("positions", []):
        q = float(p.get("quantity", 0) or 0)
        if q > 0:
            positions[p["ticker"]] = q

    prices = {}
    for t, px in (portfolio_state.get("last_prices") or {}).items():
        try:
            px = float(px)
        except (TypeError, ValueError):
            continue
        if px > 0 and math.isfinite(px):
            prices[t] = px
    for t in set(list(positions) + list(market_state)):
        if t not in prices:
            c = _closes(market_state, t)
            if c:
                prices[t] = c[-1]

    cash = float(portfolio_state.get("cash", cash) or 0)
    equity = cash + sum(q * prices.get(t, 0.0) for t, q in positions.items())
    if equity <= 0:
        return []

    offensive_set = set(OFFENSIVE) | set(OVERLAY)
    off_w = sum(q * prices.get(t, 0.0) for t, q in positions.items() if t in offensive_set) / equity
    prev_on = off_w > 0.25

    regime, qvol = _regime(market_state, prev_on)
    targets = _limit(_targets(market_state, regime, qvol, set(positions)))

    sells, buys = [], []
    for t in set(list(positions) + list(targets)):
        px = prices.get(t)
        if not px:
            continue
        cur_qty = positions.get(t, 0.0)
        cur_w = cur_qty * px / equity
        tgt_w = targets.get(t, 0.0)
        diff = (tgt_w - cur_w) * equity
        exiting = tgt_w == 0.0 and cur_qty > 0
        over = cur_w > DRIFT_WEIGHT
        if not (exiting or over or abs(diff) >= MIN_TRADE_FRAC * equity):
            continue
        if diff < 0:
            qty = cur_qty if exiting else min(cur_qty, math.floor(-diff / px))
            if qty > 0:
                sells.append((t, float(qty), qty * px))
        elif diff > 0:
            qty = math.floor(diff / px)
            if qty > 0:
                buys.append((t, float(qty), px))

    budget = (cash + sum(s[2] for s in sells)) * 0.99
    orders = [{"ticker": t, "side": "sell", "quantity": q} for t, q, _ in sells]
    buys.sort(key=lambda b: -b[1] * b[2])
    for t, q, px in buys:
        qty = min(q, math.floor(budget / px))
        if qty > 0:
            orders.append({"ticker": t, "side": "buy", "quantity": float(qty)})
            budget -= qty * px
    return orders
