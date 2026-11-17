"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Entry for make.fun competition gta6-twitch-ccu-forecast-01.

Program: python3 main.py (Python 3.13, standard library only, no network).
It reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
/output/<id>.out (or $MAKEFUN_OUTPUT_DIR) as
    {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}
for hours 2..24 of the release day (UTC).

Model (v2; chosen by leave-one-out cross-validation on the
comparables; see NOTES.md for the numbers, including why v1's
practice score was poor and what changed here)
---------------------------------------------------------------
The release-day curve of a big launch is mostly set by things you
already know at the end of hour 1:

1.  Hour 1's print itself.  Across past launches, log(viewers in hour h)
    is roughly linear in log(hour-1 viewers), with a slope that falls
    smoothly from ~1.06 at hour 2 (hour 2 is just a continuation of
    hour 1) to ~0.65 in the evening (the day's peak saturates: a launch
    that already drew 800k in hour 1 does not draw 8x that later, while
    a launch that drew 400 in hour 1 can draw 1000x later).  A flat
    "every hour stays at hour 1" forecast -- the house baseline -- misses
    that shape entirely.

2.  How busy the category was in the 24 hours before release day
    (pre24max, the largest hourly viewers in that window, relative to
    hour 1).  Launches whose category was already hot before release
    day tend to rise more during the day; launches whose category was
    quiet tend to stay closer to hour 1.  The feature is the clipped
    log-ratio log(pre24max / p1), clipped to the range seen in the
    comparables, so an extreme value can never dominate the forecast.

3.  Whether the launch is console-only (no PC at release).  The
    console-only comparables rise far more into the US evening than
    the PC-included ones (median peak 2.66x hour 1 vs 1.26x), and all
    four practice launches and GTA VI are console-only, while most of
    the training set is not.  A 0/1 indicator for this (from the
    comparables' platform lists; for the target, parsed from the case
    notes' release sentence) adds +0.3 to +0.8 log units from hour 5
    onward.  It is the single biggest v2 change: without it the fit
    under-predicts the evening rise of console-only launches.

4.  How many channels carried hour 1 (log channels at hour 1,
    clipped to the range seen in the comparables).  Holding viewers
    fixed, more (smaller) channels predict a larger rise through the
    day; the coefficient is +0.15 to +0.47 log units.  It is
    equivalent, given log(p1) is already in the fit, to using
    viewers-per-channel; channels were kept because they are what
    the case files actually report at hour 1.

Per hour h = 2..24 we fit, by ordinary least squares on the training
comparables,

    log viewers_h = a_h + b_h * log(p1) + c_h * log(pre24max / p1)
                    + d_h * console_only + e_h * log(channels_1),

then smooth the coefficients across neighbouring hours
(0.25 / 0.5 / 0.25) -- the true slope/intercept move smoothly through
the day, and smoothing removes per-hour fitting noise.  If a feature
is (near-)constant or the fit fails at any hour, that feature is
dropped for every hour and the fit is retried (full set, then without
log-channels, then without the console indicator, then the v1 set,
then log(p1) alone).  The forecast quantiles are normal quantiles
around the fitted median, with a per-hour spread equal to 1.15x the
residual standard deviation of the smoothed fit (floored at 0.18 log
units, capped at 0.90).

Which comparables train the fit matters more than the regression
itself.  Launches fall into two regimes, visible in their own release
day before any forecast is made:

* Gradual (rolling-midnight) launches: no single hour jumps by more
  than about 2.7x (log jump < 1.0).  The game unlocks region by region,
  hour 1 already contains New Zealand / Australia / Asia / Europe, and
  the US unlocks add viewers through the morning.  GTA VI, and all
  four practice launches, are of this kind (their case notes say so:
  "rolling midnight local").
* Sharp-step (simultaneous-global) launches: the audience multiplies
  several-fold in a single hour when the one global unlock hits
  (e.g. Hollow Knight: Silksong, Black Myth: Wukong).  Hour 1 is then
  only pre-unlock hype and says almost nothing about the peak.

For a gradual target we train on the gradual comparables only;
including the sharp-step ones pulls the slope down and roughly doubles
the residual spread.  For a simultaneous target (case notes say
"simultaneous" / "single global unlock", or the pre-day peak is under
a tenth of hour 1) we train on all comparables instead, which
cross-validates better on sharp-step launches than training on them
alone.  The GTA VI category's own pre-release history (a comparable
whose name says "before release") is not a launch and is never used
as a training curve; it is only the target's own history.

Fallbacks, in order: if fewer than 10 gradual comparables are usable,
train on all usable comparables; if no regression can be fitted at
all, fall back to the house baseline (persistence with a widening
spread).  Everything is deterministic: inputs are sorted, there is
no randomness, no clock, no network, and no state between runs.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file would import.  (This entry is a
# single file; the line is kept for the same reason the starter has it.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- model constants (fixed before seeing any case; see NOTES.md) ---
# A launch is "sharp-step" when some single hour of its release day
# multiplies viewers by more than exp(STEP_LOG) ~= 2.72x.
STEP_LOG = 1.0
# The pre-launch log-ratio feature is clipped to this range, the range
# observed across the comparables (ratios about 0.18x to 5.05x).
PRE_CLIP = (-1.7, 1.62)
# Spread: residual sd multiplier, floor and cap (natural-log units).
SD_MULT = 1.15
SD_FLOOR = 0.18
SD_CAP = 0.90
# Log channels at hour 1 is clipped to this range, the range observed
# across the comparables (6 to ~21,700 channels).
LOGCH_CLIP = (1.8, 10.0)
# Minimum training curves for the gradual-only fit; below this, use all.
MIN_GRADUAL = 10
# A curve needs at least this many of the 24 release-day hours present.
MIN_HOURS = 20


def parse_time(text):
    """Parse an ISO 8601 timestamp (with a trailing Z) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_index(day_start, text):
    """Whole hours from day_start to `text`, or None if not on the hour."""
    delta = (parse_time(text) - day_start).total_seconds() / 3600.0
    if delta == int(delta):
        return int(delta)
    return None


def round2(x):
    """Round to 2 decimals, half up (the starter's rounding)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None if the system is (near-)singular.
    Deterministic: partial pivoting picks the largest pivot, ties broken
    by row order.
    """
    n = len(rhs)
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        pv = rows[col][col]
        for j in range(col, n + 1):
            rows[col][j] /= pv
        for r in range(n):
            if r != col and rows[r][col] != 0.0:
                factor = rows[r][col]
                for j in range(col, n + 1):
                    rows[r][j] -= factor * rows[col][j]
    return [rows[i][n] for i in range(n)]


def fit_ols(xs, ys):
    """Least-squares fit of ys on rows xs; None if it cannot be fitted."""
    k = len(xs[0])
    matrix = [[sum(row[i] * row[j] for row in xs) for j in range(k)]
              for i in range(k)]
    rhs = [sum(row[i] * y for row, y in zip(xs, ys)) for i in range(k)]
    return solve_linear(matrix, rhs)


def pstdev(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def smooth3(values):
    """Three-point smoothing (0.25 / 0.5 / 0.25), endpoints 0.75 / 0.25."""
    if len(values) < 2:
        return list(values)
    out = []
    for i, v in enumerate(values):
        if i == 0:
            out.append(0.75 * v + 0.25 * values[1])
        elif i == len(values) - 1:
            out.append(0.25 * values[-2] + 0.75 * v)
        else:
            out.append(0.25 * values[i - 1] + 0.5 * v + 0.25 * values[i + 1])
    return out


class Curve:
    """One comparable launch's release day, reduced to what the fit needs."""

    def __init__(self, name, day_start, hourly, platforms="", unlock=""):
        self.name = name
        self.platforms = str(platforms or "")
        self.unlock = str(unlock or "")
        # viewers by release-day hour (1..24); only positive, on-the-hour.
        self.hours = {}
        # live channels by release-day hour (where reported).
        self.channels = {}
        # Largest hourly viewers in the 24 h before day_start, or None.
        pre_max = None
        for row in hourly:
            try:
                idx = hour_index(day_start, row["from"])
                viewers = row["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(viewers, (int, float)) or viewers <= 0:
                continue
            if idx is None:
                continue
            if 0 <= idx < 24:
                self.hours[idx + 1] = float(viewers)
                channels = row.get("channels")
                if isinstance(channels, (int, float)) and channels > 0:
                    self.channels[idx + 1] = float(channels)
            elif -24 <= idx < 0:
                if pre_max is None or viewers > pre_max:
                    pre_max = float(viewers)
        self.pre_max = pre_max

    @property
    def p1(self):
        return self.hours.get(1)

    @property
    def usable(self):
        return self.p1 is not None and len(self.hours) >= MIN_HOURS

    @property
    def max_step(self):
        """Largest one-hour log jump on the release day (inf if unknown)."""
        best = None
        for h in range(1, 24):
            if h in self.hours and h + 1 in self.hours:
                step = math.log(self.hours[h + 1] / self.hours[h])
                if best is None or step > best:
                    best = step
        return best if best is not None else float("inf")

    @property
    def sharp(self):
        return self.max_step >= STEP_LOG

    def pre_feature(self):
        """Clipped log(pre24max / p1); 0 when there is no pre-day peak."""
        if self.pre_max is None or self.p1 is None:
            return 0.0
        value = math.log(self.pre_max / self.p1)
        return max(PRE_CLIP[0], min(PRE_CLIP[1], value))

    @property
    def console_only(self):
        """True when the launch had no PC version at release.

        Read from the platform list: "PC" (or Stadia) anywhere in it
        means a PC launch.  The eight console-only comparables are
        exactly the ones whose platform list names no PC.
        """
        platforms = self.platforms.lower()
        return "pc" not in platforms and "stadia" not in platforms

    @property
    def log_channels(self):
        """Clipped log channels at hour 1; None when unreported."""
        channels = self.channels.get(1)
        if channels is None or channels <= 0:
            return None
        value = math.log(channels)
        return max(LOGCH_CLIP[0], min(LOGCH_CLIP[1], value))


def target_console_only(notes):
    """Whether the target launch is console-only, from the case notes.

    Reads only the release sentence (the text from "released" to the
    next ";"): the kit's case notes always state the release
    platforms there.  "console only" or "no PC" is decisive; a "PC"
    named as a launch platform means not console-only, unless the
    mention is a later PC date ("PC 2019-11"); a console name
    followed by "only", or a platform list with no PC in it at all,
    also means console-only.  Anything unclear defaults to False
    (no console boost), the conservative choice.
    """
    text = str(notes or "").lower()
    start = text.find("released")
    segment = text[start:start + 160] if start >= 0 else text[:160]
    segment = segment.split(";")[0]
    if "console only" in segment or "no pc" in segment or "without pc" in segment:
        return True
    if re.search(r"\bpc\b", segment):
        # A PC named with a later date ("pc 2019-11", "pc 2027") is a
        # later port, not a launch platform.
        if re.search(r"\bpc\s+\d{4}", segment):
            return True
        return False
    if "only" in segment and any(
            token in segment for token in ("ps5", "ps4", "xbox", "switch", "playstation")):
        return True
    if any(token in segment for token in ("ps5", "ps4", "xbox", "switch", "playstation")):
        return True
    return False


def target_pre_max(case, day_start):
    """Largest hourly viewers in the 24 h before the target's day_start.

    Prefers history_hourly (the same hourly measure the comparables
    use); falls back to the finer history points.
    """
    rows = case.get("history_hourly") or []
    best = None
    for row in rows:
        try:
            idx = hour_index(day_start, row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if (isinstance(viewers, (int, float)) and viewers > 0
                and idx is not None and -24 <= idx < 0):
            if best is None or viewers > best:
                best = float(viewers)
    if best is not None:
        return best
    for row in case.get("history") or []:
        try:
            at = parse_time(row["at"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(viewers, (int, float)) and viewers > 0:
            delta = (day_start - at).total_seconds()
            if 0 < delta <= 24 * 3600 and (best is None or viewers > best):
                best = float(viewers)
    return best


def baseline_quantiles(p1, hour):
    """The house baseline: persistence with a widening spread."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case):
    """Forecast hours 2..24 of one case as {"hours": [...]}."""
    day_start = parse_time(case["day_start"])
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    try:
        p1 = max(1.0, float(first["viewers"])) if first else 1.0
    except (KeyError, TypeError, ValueError):
        p1 = 1.0
    hours_to_forecast = [row["hour"] for row in case.get("forecast", [])]
    if not hours_to_forecast:
        hours_to_forecast = list(range(2, 25))

    # --- gather the training curves, in a fixed order -----------------
    curves = []
    for comp in sorted(case.get("comparables", []),
                       key=lambda c: str(c.get("name", ""))):
        name = str(comp.get("name", ""))
        unlock = str(comp.get("unlock", ""))
        # The GTA VI category's own pre-release history is not a launch.
        if "before release" in name.lower() or "not a launch" in unlock.lower():
            continue
        try:
            comp_day = parse_time(comp["day_start"])
            curve = Curve(name, comp_day, comp.get("hourly") or [],
                          comp.get("platforms", ""), comp.get("unlock", ""))
        except (KeyError, TypeError, ValueError):
            continue
        if curve.usable:
            curves.append(curve)

    # --- which regime is the target in? --------------------------------
    notes = str(case.get("notes", "")).lower()
    # A bare substring test for "simultaneous" is wrong: the notes
    # for God of War Ragnarok say "no simultaneous launch", which is
    # a rolling launch, and v1 took the simultaneous branch for it
    # (training on all comparables, sharp-step ones included) and
    # forecast a flat, declining day.  Strip negated phrases first,
    # then look for an affirmed simultaneous unlock.
    notes_affirmed = notes
    for negated in ("no simultaneous", "not simultaneous",
                    "not a simultaneous", "no single global",
                    "not a single global", "never simultaneous"):
        notes_affirmed = notes_affirmed.replace(negated, "")
    simultaneous_text = ("simultaneous" in notes_affirmed
                         or "single global" in notes_affirmed)
    rolling_text = ("rolling" in notes or "midnight local" in notes)
    pre_max = target_pre_max(case, day_start)
    if pre_max is not None and p1 > 0:
        pre_ratio = pre_max / p1
        pre_feature = max(PRE_CLIP[0], min(PRE_CLIP[1], math.log(pre_ratio)))
    else:
        pre_ratio = None
        pre_feature = 0.0
    simultaneous = simultaneous_text or (
        pre_ratio is not None and pre_ratio < 0.10 and not rolling_text)

    if simultaneous:
        train = curves
    else:
        gradual = [c for c in curves if not c.sharp]
        train = gradual if len(gradual) >= MIN_GRADUAL else curves

    # --- target features for the v2 fit --------------------------------
    # Console-only flag, parsed from the case notes' release sentence.
    target_cons = 1.0 if target_console_only(case.get("notes", "")) else 0.0
    # Log channels at hour 1, clipped; if the case does not report
    # channels, use the training set's median so the feature is neutral.
    try:
        ch1 = float(first["channels"]) if first else None
    except (KeyError, TypeError, ValueError):
        ch1 = None
    if ch1 is not None and ch1 > 0:
        logch_target = max(LOGCH_CLIP[0],
                           min(LOGCH_CLIP[1], math.log(ch1)))
    else:
        train_logch = sorted(c.log_channels for c in train
                             if c.log_channels is not None)
        logch_target = (train_logch[len(train_logch) // 2]
                        if train_logch else 0.0)

    def curve_features(curve, n_feat):
        """Feature row for one training curve (or the target)."""
        row = [1.0, math.log(curve.p1), curve.pre_feature()]
        if n_feat >= 4:
            row.append(1.0 if curve.console_only else 0.0)
        if n_feat >= 5:
            value = curve.log_channels
            if value is None:
                value = logch_target
            row.append(value)
        return row

    def target_features(n_feat):
        row = [1.0, log_p1, pre_feature]
        if n_feat >= 4:
            row.append(target_cons)
        if n_feat >= 5:
            row.append(logch_target)
        return row

    # --- fit log viewers_h per hour ------------------------------------
    # Try the full five-feature fit first; if any hour cannot be
    # fitted (a feature near-constant in this training set, too few
    # curves), drop the last feature for every hour and retry, down
    # to log(p1) alone.  This keeps the coefficients comparable
    # across hours, as in v1.
    log_p1 = math.log(p1)
    coefs = None
    n_coef = 0
    for n_feat in (5, 4, 3, 2):
        trial = []
        ok = bool(train)
        for hour in hours_to_forecast:
            xs, ys = [], []
            for curve in train:
                if hour in curve.hours:
                    xs.append(curve_features(curve, n_feat))
                    ys.append(math.log(curve.hours[hour]))
            beta = fit_ols(xs, ys) if len(xs) >= n_feat + 3 else None
            if beta is None:
                ok = False
                break
            trial.append(beta)
        if ok:
            coefs, n_coef = trial, n_feat
            break

    if coefs is None:
        # No usable fit: fall back to the house baseline for every hour.
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in hours_to_forecast]}

    # Smooth each coefficient across neighbouring hours, then measure
    # the residual spread of the smoothed fit, per hour.
    for j in range(n_coef):
        smoothed = smooth3([beta[j] for beta in coefs])
        for i in range(len(coefs)):
            coefs[i][j] = smoothed[i]

    out_hours = []
    for i, hour in enumerate(hours_to_forecast):
        beta = coefs[i]
        mu = sum(b * x for b, x in zip(beta, target_features(n_coef)))
        residuals = []
        for curve in train:
            if hour in curve.hours:
                residuals.append(
                    math.log(curve.hours[hour])
                    - sum(b * x for b, x in
                          zip(beta, curve_features(curve, n_coef))))
        sd = SD_FLOOR
        if len(residuals) >= 4:
            sd = max(SD_FLOOR, min(SD_CAP, SD_MULT * pstdev(residuals)))
        quantiles = []
        previous = 0.0
        for z in Z:
            q = math.exp(mu + z * sd)
            q = round2(q)
            q = min(max(q, 0.01), 1_000_000_000)
            if q < previous:  # keep the row non-decreasing after rounding
                q = previous
            quantiles.append(q)
            previous = q
        out_hours.append({"hour": hour, "q": quantiles})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
