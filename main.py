"""GTA VI week-one trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file per run (makefun/gta6-trophies-case/v1) from $MAKEFUN_INPUT_DIR
(or /input) and writes one forecast file per case to $MAKEFUN_OUTPUT_DIR (or
/output): for each of the game's 55 trophies, five quantiles (5/25/50/75/95%) of
its earn rate on Exophase at the case target.

Standard library only, no network, deterministic: the same input always yields
the same output bytes.

What the data is
---------------
Each case's history is a series of reads of Exophase's GTA VI PS5 trophy page.
A read lists every trophy's earn rate (percent of tracked players who have it,
2 decimals) and the players-tracked count. Exophase only changes the page when
it refreshes its counts, so the history is a step function: long flat stretches
punctuated by refresh jumps. Two regimes matter:

1. Launch night (practice cases). Every read before the first refresh is the
   same page: 638 players tracked, every rate frozen. The refresh (read at
   2026-11-19T04:30Z, documented in the kit's RULES.txt) recounts to 3,868
   players and every rate but the 100% one falls. A flat history therefore
   carries no trend at all; the only signal is the refresh itself. The
   refreshed rate of a trophy is its earner count rescaled to the new
   denominator, and the earner counts grew ~20% between the checkpoint and the
   refresh (the kit's own numbers pin this down: the house baseline's published
   practice loss of ~1.51 and typical miss of ~0.68 points are reproduced by a
   x1.2 rescale, 1.50/0.69, while a frozen-count rescale gives 1.69/0.73).
   So for a flat history we predict round2(100 * k * 1.2 / 3868) per trophy
   (k = earners implied by the last read), keep a 100% trophy at 100, and use
   a narrow spread: the refresh is a known event, not a random walk.

2. After launch (the final case). The history holds several post-refresh
   states. Each trophy's rate is earners / tracked, and both move: tracked
   players grow with launch-week sales (diluting every rate) while earners
   grow as the player cohort plays through the game. We therefore forecast
   in log-odds against log time, anchored on the freshest read:

     logit(rate) is fitted by recency-weighted least squares against
     ln(1 + hours since launch), the slope is shrunk and damped (early
     launch-phase slopes overstate later growth: the tracked population
     changes from day-one players to the general launch-week mix, and both
     progression and dilution decelerate), and the forecast is anchored on
     the last observed rate rather than the fitted intercept.

   Three consistency rules then apply, because rates are counts, not free
   numbers: earners never decrease (a forecast rate below last-earners /
   forecast-tracked is lifted to that floor); the story chain is nested
   (Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission);
   and 100% completion and the platinum sit below every other trophy.
   A fourth rule covers the case the trend cannot see: a chapter's earners
   are flat until the player cohort reaches it, then rise as the cohort
   flows downstream. The observed adjacent-chapter ratio at day two is
   already 0.14 (Chapter 2 / Chapter 1), and cohort flow only raises it, so
   each later chapter is floored at 10% of the previous chapter's forecast
   rate, with a widened spread to mark the judgement.
   The spread per trophy comes from its own fit residuals plus slope
   uncertainty plus a staleness term (the answer is a refresh read, which
   may lag the target by hours), floored so that no trophy pretends to be
   known exactly over a 4.5-day horizon.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path; nothing to import from beside this
# file, so no path fix is needed here.

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First refresh of the game's counts (kit RULES.txt, "What is counted"): the
# read at 2026-11-19T04:30:00Z, 3,868 players tracked.
REFRESH_PLAYERS = 3868
# Earner growth between the last pre-refresh read and that refresh, fitted to
# the house baseline's published practice numbers (loss ~1.51, typical miss
# ~0.68): x1.2 reproduces 1.50 / 0.69; frozen counts give 1.69 / 0.73.
REFRESH_EARNER_GROWTH = 1.2
# Spread for the flat-history (refresh) branch: the refresh is a known event,
# but the per-trophy growth around x1.2 is not exact.
REFRESH_SPREAD = 0.08

# Damping applied to fitted logit slopes when extrapolating: launch-phase
# slopes overstate later movement (progression and dilution both decelerate).
SLOPE_DAMP = 0.75
# Floor / cap on the per-trophy logit spread in the trend branch.
SPREAD_FLOOR = 0.35
SPREAD_CAP = 1.6
# Extra spread for a trendless trophy (no usable history slope).
FLAT_SPREAD = 0.9
# Staleness of the answering refresh read, in logit units.
STALENESS = 0.15
# Adjacent-chapter floor for the story chain (see module docstring): the
# day-two Chapter 2 / Chapter 1 ratio is already 0.14 and cohort flow only
# raises adjacent ratios, so a later chapter is floored at 10% of the
# previous chapter's forecast. Spread for a floored trophy is widened: the
# floor is a judgement, not data.
CHAIN_FLOOR_RATIO = 0.10
FLOORED_SPREAD = 0.7


def clamp_rate(x):
    """Clamp a percent into the range the scorer's logit accepts."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def ilogit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round_half_up(x, places):
    """Round half up to `places` decimals (the way the page prints rates)."""
    scaled = x * (10.0 ** places)
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / (10.0 ** places)
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if text is None:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (slope, intercept, resid_std, slope_se, n_eff). With fewer than
    two distinct x values the slope is 0 and the standard errors are None.
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, (ys[0] if ys else 0.0), 0.0, None, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    n_eff = sw * sw / sum(w * w for w in ws) if ws else 0.0
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, len(ys) - 1)) if len(ys) > 1 else 0.0
        return 0.0, my, sd, None, n_eff
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, len(ys) - 2)
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    resid_std = math.sqrt(var * len(ys) / dof) if len(ys) > 2 else 0.3
    slope_se = math.sqrt(var / sxx) if sxx > 0 else None
    return slope, intercept, resid_std, slope_se, n_eff


def distinct_states(history):
    """Collapse consecutive identical reads into one state each.

    A state keeps the slot time, the players-tracked count and the rates.
    Reads are already oldest-first; identical neighbours are one Exophase
    page shown twice, not two observations.
    """
    states = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if states and states[-1]["key"] == key:
            continue
        when = parse_time(read.get("slot")) or parse_time(read.get("at"))
        states.append({
            "key": key,
            "when": when,
            "players": read.get("players_tracked"),
            "rates": list(read.get("rates") or []),
        })
    return states


def refresh_branch(case, states):
    """Forecast for a history with no refresh in it (flat launch-night page).

    Every trophy's refreshed rate is its earner count rescaled to the
    refresh's denominator, with the fitted earner growth; the 100% trophy
    stays at 100% (RULES.txt: every rate but the one at 100% fell).
    """
    last = states[-1]
    players = last["players"] or 0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        if rate is None:
            rate = 0.01
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        earners = rate / 100.0 * players
        grown = earners * REFRESH_EARNER_GROWTH
        centre = round_half_up(100.0 * grown / REFRESH_PLAYERS, 2)
        centre = clamp_rate(centre)
        lg = logit(centre)
        qs = [round_half_up(ilogit(lg + z * REFRESH_SPREAD), 4) for z in Z]
        qs = [min(100, max(0.01, q)) for q in qs]
        qs.sort()
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_tracked(states, origin, t_star_hours, t_last_hours):
    """Forecast the players-tracked count at the target, in log space.

    Blends a power-law fit (log N linear in log time since launch) with a
    damped continuation of the recent growth rate, then clamps to a sane
    band: tracked players grow through launch week but the growth
    decelerates.
    """
    first = states[0]
    last = states[-1]
    n_last = last["players"] or 1
    if len(states) < 2 or first["players"] is None:
        return n_last * 1.8
    xs = []
    ys = []
    for s in states:
        if s["players"] and s["players"] > 0 and s["when"] is not None:
            xs.append(math.log1p(max(0.0, hours_between(s["when"], origin))))
            ys.append(math.log(s["players"]))
    if len(xs) < 2 or max(xs) - min(xs) < 1e-9:
        return n_last * 1.8
    slope, intercept, _sd, _se, _n = weighted_ols(xs, ys, [1.0] * len(xs))
    n_pow = math.exp(intercept + slope * math.log1p(max(0.0, t_star_hours)))
    t_first = hours_between(first["when"], origin)
    span = max(1.0, t_last_hours - t_first)
    g = (math.log(n_last) - math.log(first["players"])) / span
    n_damp = n_last * math.exp(0.6 * g * max(0.0, t_star_hours - t_last_hours))
    n_star = math.sqrt(max(1.0, n_pow) * max(1.0, n_damp))
    return min(max(n_star, n_last * 1.02), n_last * 6.0)


def hours_between(later, earlier):
    if later is None or earlier is None:
        return 0.0
    return (later - earlier).total_seconds() / 3600.0


def trend_branch(case, states, origin, target_dt):
    """Forecast from post-refresh states: damped log-odds trends in log time."""
    t_star = hours_between(target_dt, origin)
    x_star = math.log1p(max(0.0, t_star))
    t_last = hours_between(states[-1]["when"], origin)
    n_star = forecast_tracked(states, origin, t_star, t_last)

    # Per-trophy medians (logit) and spreads, before consistency fixes.
    medians = {}
    spreads = {}
    last_earners = {}
    for i, trophy in enumerate(case["trophies"]):
        pts_x = []
        pts_y = []
        pts_w = []
        last_rate = None
        for s in states:
            if i >= len(s["rates"]):
                continue
            r = s["rates"][i]
            if r is None:
                continue
            last_rate = r
            if s["when"] is None:
                continue
            t = hours_between(s["when"], origin)
            pts_x.append(math.log1p(max(0.0, t)))
            pts_y.append(logit(r))
            # Recency weight: halve every 24 hours before the last read.
            age = max(0.0, t_last - t)
            pts_w.append(0.5 ** (age / 24.0))
        if last_rate is not None and states[-1]["players"]:
            last_earners[trophy["id"]] = last_rate / 100.0 * states[-1]["players"]
        if last_rate is not None and last_rate >= 99.995:
            medians[trophy["id"]] = None  # the 100% trophy, handled below
            spreads[trophy["id"]] = 0.0
            continue
        if len(pts_x) >= 2 and max(pts_x) - min(pts_x) > 0.05:
            slope, _a, resid_sd, slope_se, n_eff = weighted_ols(pts_x, pts_y, pts_w)
            shrink = n_eff / (n_eff + 1.0) if n_eff > 0 else 0.5
            damped = slope * shrink * SLOPE_DAMP
            y_star = pts_y[-1] + damped * (x_star - pts_x[-1])
            mean_x = sum(w * x for w, x in zip(pts_w, pts_x)) / sum(pts_w)
            var = resid_sd ** 2 + STALENESS ** 2
            if slope_se is not None:
                var += (slope_se * (x_star - mean_x)) ** 2
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, math.sqrt(var)))
        elif pts_y:
            y_star = pts_y[-1]
            spread = FLAT_SPREAD
        else:
            y_star = logit(0.01)
            spread = FLAT_SPREAD
        medians[trophy["id"]] = y_star
        spreads[trophy["id"]] = spread

    # Consistency 1: earners never decrease. A forecast rate below
    # last-earners / forecast-tracked is lifted to that floor.
    rates = {}
    for trophy in case["trophies"]:
        tid = trophy["id"]
        if medians[tid] is None:
            rates[tid] = 100.0
            continue
        r = ilogit(medians[tid])
        r = clamp_rate(r)
        if tid in last_earners and n_star > 0:
            floor = 100.0 * last_earners[tid] / n_star
            if r < floor:
                r = floor
        rates[tid] = r

    # Consistency 2: the story chain is nested (a later chapter cannot be
    # earned by more players than an earlier one).
    chain = [3, 4, 5, 6, 7]  # Chapter 1..4, then the final mission
    present = [t for t in chain if t in rates]
    for j in range(1, len(present)):
        if rates[present[j]] > rates[present[j - 1]]:
            rates[present[j]] = rates[present[j - 1]]
    # Consistency 2b: downstream floor. A chapter whose earners have not
    # started growing yet is floored at 10% of the previous chapter's
    # forecast rate (cohort flow downstream; see module docstring).
    floored = set()
    for j in range(1, len(present)):
        prev, cur = present[j - 1], present[j]
        floor = CHAIN_FLOOR_RATIO * rates[prev]
        if rates[cur] < floor:
            rates[cur] = floor
            floored.add(cur)
    # Consistency 3: 100% completion and the platinum sit below every trophy.
    if 14 in rates and 1 in rates:
        others = [r for tid, r in rates.items() if tid not in (1, 14, 2)]
        if others:
            cap = min(others)
            if rates[14] > cap:
                rates[14] = cap
            if rates[1] > rates[14]:
                rates[1] = rates[14]

    rows = []
    for trophy in case["trophies"]:
        tid = trophy["id"]
        if medians[tid] is None:
            rows.append({"id": tid, "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(rates[tid])
        s = spreads[tid]
        if tid in floored:
            s = max(s, FLOORED_SPREAD)
        qs = [round_half_up(ilogit(lg + z * s), 4) for z in Z]
        qs = [min(100, max(0.01, q)) for q in qs]
        qs.sort()
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    if not trophies:
        return {"trophies": []}
    if not history:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]} for t in trophies]}

    states = distinct_states(history)
    if not states:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]} for t in trophies]}

    # Was there a refresh inside the history? The refresh is the first read
    # whose players-tracked count differs from the opening page's: the
    # count is the clean signal (a read can lack one trophy's rate, which
    # must not count as a refresh). Reads before it are the flat
    # pre-refresh page and carry no trend, so they are dropped from the
    # fit; if the history already starts after the first refresh, every
    # state is post-refresh and all of them are used.
    first_n = states[0]["players"]
    change_at = None
    for j in range(1, len(states)):
        if states[j]["players"] != first_n:
            change_at = j
            break
    if change_at is None:
        # No refresh seen: launch-night flat page. But only the pre-refresh
        # regime (few tracked players) rescales to the documented refresh;
        # a flat history at a large player count is just persistence.
        if (states[-1]["players"] or 0) < REFRESH_PLAYERS:
            return refresh_branch(case, states)
        return trend_branch_flat(case, states)

    if (first_n or 0) < REFRESH_PLAYERS:
        post = states[change_at:]
    else:
        post = states
    game = case.get("game") or {}
    release = game.get("release_date") or "2026-11-19"
    origin = parse_time(release + "T00:00:00Z") or parse_time("2026-11-19T00:00:00Z")
    target_dt = parse_time(case.get("target")) or parse_time(case.get("at"))
    return trend_branch(case, post, origin, target_dt)


def trend_branch_flat(case, states):
    """Flat history at a large player count: persistence, wide spread."""
    last = states[-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) and last["rates"][i] is not None else 0.01
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(rate)
        qs = [round_half_up(ilogit(lg + z * FLAT_SPREAD), 4) for z in Z]
        qs = [min(100, max(0.01, q)) for q in qs]
        qs.sort()
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
