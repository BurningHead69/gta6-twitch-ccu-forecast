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

Model (v6; v2 plus the "kitchen-sink" combination — see NOTES.md
section 10)
---------------------------------------------------------------
v6 keeps v2's model and adds three small, LOO-measured tweaks
stacked on top of it:

1.  Early-access down-weighting: comparables whose unlock notes
    affirm an early-access launch (EA_SOURCE = "text": any
    affirmed "early access" / "early-access" mention in the
    unlock text, negations stripped; 11 launches) are
    down-weighted to EA_WEIGHT (0.5) in the per-hour weighted least
    squares.  (EA_SOURCE = "tier" — only the literal
    "Tier: early-access-contaminated" label, 9 launches — was
    also tested; see NOTES.md section 10.)  Early-access launches
    have hour 1 already inflated by early-access play, so their
    release day is flatter than the model expects, and they are
    v2's worst cases (LOO 0.207 vs 0.142 for the rest);
    down-weighting them, rather than adding an EA indicator
    (which is estimated noisily and can point the wrong way), is
    the most consistent small gain v3/v4 found.

2.  Ridge shrinkage: RIDGE_LAMBDA (0.5) is added to the diagonal
    of the weighted normal equations for every coefficient except
    the intercept.  With n ~ 25 training curves and five
    coefficients per hour, a little shrinkage steadies the fit.

3.  Spread modulation by the pre-release slope: the per-hour
    spread is multiplied by exp(SPREAD_A * (target pre_slope48 -
    mean train pre_slope48)), with SPREAD_A = -2.0.  pre_slope48
    is the OLS slope of log viewers on hour index over the 48
    hours before day_start.  The slope marks *difficulty*, not
    level (it correlates -0.38 with v2's LOO loss): launches whose
    category was rising into release day are easier to predict,
    so their spread is narrowed; flat or falling pre-histories
    widen it.  The modulation is applied after the spread
    floor/cap, with no re-clipping, exactly as validated.  If the
    target has no usable pre-48h history the modulation is
    skipped (factor 1.0).

Honest warning (see NOTES.md section 10): the combination's LOO
gain over v2 is modest and uncertain (gradual mean -0.006,
paired t ~ -0.9, median worse), and the practice improvement is
not guaranteed.  v6 is a deliberate push for the practice
leaderboard, adopted by the user with that warning in view.

Model (v7; v6 plus v5's streams layer — see NOTES.md section 10)
---------------------------------------------------------------
v7 is v6's model unchanged (EA text down-weight 0.5, ridge 0.5,
spread modulation exp(-2 * pre_slope48 deviation)) with v5's
bounded streams adjustment applied last, on top of v6's
forecasts, exactly as v5 applied it on top of v2's: the
viewer-weighted mean stream age shifts the early-hour median by
at most 0.10 log units, and the Herfindahl concentration of the
top-20 viewer shares scales the spread by 0.90-1.20x.  The one
interaction is the spread: v6's spread multiplier is unbounded
and v5's is in [0.90, 1.20], so when the streams spread layer is
active at an hour the COMBINED multiplier (v6's times v5's) is
clipped to [0.90, 1.20] relative to v2's base spread (the spread
before v6's modulation).  When streams are absent, or the streams
spread multiplier is 1.0 at an hour, the spread is v6's exactly
— so with streams absent v7 is byte-identical to v6.  The
streams effect itself is unvalidated (null in every practice
case); see NOTES.md section 10.
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
# --- v6 "kitchen-sink" constants (see NOTES.md section 10) -----------
# Training weight for early-access-contaminated comparables.
EA_WEIGHT = 0.5
# How early access is detected in a comparable's unlock text:
# "text" = any affirmed "early access" / "early-access" mention
# (negations stripped first); "tier" = the literal
# "Tier: early-access-contaminated" label only.  Text is the
# default: in the fresh v6 LOO run it beat tier on the gradual
# mean (0.1585 vs 0.1598) with the median intact (0.1258 vs
# v2's 0.1258) and the stronger paired t (-1.24 vs -0.92); see
# NOTES.md section 10.  Text flags 11 launches, tier 9 (the
# extra two are Black Ops 6 and Modern Warfare III, whose unlock
# notes mention early access without the tier label).
EA_SOURCE = "text"
# Ridge penalty added to the diagonal of the weighted normal
# equations for every coefficient except the intercept.
RIDGE_LAMBDA = 0.5
# Spread modulation coefficient: sd *= exp(SPREAD_A *
# (target pre_slope48 - mean train pre_slope48)).
SPREAD_A = -2.0


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


def fit_wls(xs, ys, ws, ridge=0.0):
    """Weighted least squares with a ridge penalty on the diagonal.

    Solves (X'WX + ridge * I') beta = X'Wy, where I' is the identity
    with the intercept entry zeroed (the intercept is not shrunk).
    Returns None if the system is (near-)singular.
    """
    k = len(xs[0])
    matrix = [[sum(w * row[i] * row[j] for w, row in zip(ws, xs))
               for j in range(k)] for i in range(k)]
    rhs = [sum(w * row[i] * y for w, row, y in zip(ws, xs, ys))
           for i in range(k)]
    for i in range(1, k):
        matrix[i][i] += ridge
    return solve_linear(matrix, rhs)


def ea_from_text(text):
    """1.0 when `text` affirms an early-access launch, else 0.0.

    Negated mentions ("no early access", "not early access", ...)
    are stripped first, so "no early access" does not count.
    """
    t = str(text or "").lower()
    for negated in ("no early access", "not early access",
                    "never early access", "no early-access",
                    "without early access"):
        t = t.replace(negated, "")
    return 1.0 if ("early access" in t or "early-access" in t) else 0.0


def tier_ea(unlock):
    """1.0 when the unlock text carries the early-access tier label."""
    return 1.0 if "early-access-contaminated" in str(unlock or "").lower() else 0.0


def ols_slope(pairs):
    """OLS slope of log viewers on hour index over (idx, viewers) pairs.

    Returns 0.0 when fewer than 6 usable pairs are present.
    """
    rows = [(i, math.log(vw)) for i, vw in pairs
            if isinstance(vw, (int, float)) and vw > 0]
    if len(rows) < 6:
        return 0.0
    mx = sum(i for i, _ in rows) / len(rows)
    my = sum(y for _, y in rows) / len(rows)
    den = sum((i - mx) ** 2 for i, _ in rows)
    if den == 0:
        return 0.0
    return sum((i - mx) * (y - my) for i, y in rows) / den


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
        # (hour index, viewers) pairs in the 48 h before day_start,
        # for the pre-release slope used by the v6 spread modulation.
        pre48_pairs = []
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
            if -48 <= idx < 0:
                pre48_pairs.append((idx, float(viewers)))
        self.pre_max = pre_max
        self.pre_slope48 = ols_slope(pre48_pairs)
        # Early-access flags from the unlock text (v6 down-weighting).
        self.ea_tier = tier_ea(unlock)
        self.ea_text = ea_from_text(unlock)

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
    def train_weight(self):
        """v6 training weight: early-access curves are down-weighted."""
        ea = self.ea_tier if EA_SOURCE == "tier" else self.ea_text
        return EA_WEIGHT if ea else 1.0

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


def target_pre_slope48(case, day_start):
    """OLS slope of log viewers on hour index over the 48 h before
    day_start, from history_hourly (the same hourly measure the
    comparables use).  Returns None when fewer than 6 usable rows
    are present, in which case the v6 spread modulation is skipped.
    """
    pairs = []
    for row in case.get("history_hourly") or []:
        try:
            idx = hour_index(day_start, row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if (isinstance(viewers, (int, float)) and viewers > 0
                and idx is not None and -48 <= idx < 0):
            pairs.append((idx, float(viewers)))
    if len(pairs) < 6:
        return None
    return ols_slope(pairs)


# --- streams adjustment (v5 layer, merged into v7; see NOTES.md section 10) ---
# The final case ships `streams`: the category's top 20 live streams at
# the last valid read before the checkpoint, each {login, viewers,
# language, started_at}.  It is null in every practice case, so this
# adjustment is UNVALIDATED -- a bounded gamble the user explicitly
# chose.  Two features, each with a stated mechanism and a hard cap:
#
# * Viewer-weighted mean stream age at the checkpoint.  At 01:00Z the
#   live top streams are the early wave: New Zealand / Australia /
#   Asia streamers are 10-14 h into their launch day, Europe / UK
#   streamers 1-2 h in, and the US has not unlocked yet (05:00Z /
#   08:00Z).  A high weighted age means the top of the category is
#   still dominated by the Asia-Pacific wave, whose streamers are
#   near the end of their day and will wind down before the US
#   arrives, so hours 2-5 sag; a low age means the EU/UK wave
#   dominates -- those streamers just started their launch sessions
#   and will carry hours 2-5.  (The opposite reading -- old streams
#   are marathon survivors -- is noted in NOTES.md; the region-mix
#   reading is the basis, and the cap keeps the bet small either
#   way.)  Shift: -0.02 log units per hour of age above 5 h, hours
#   2-5 at full weight, fading to zero by hour 9, capped at 0.10.
# * Concentration (Herfindahl index of the top-20 viewer shares).
#   A concentrated audience is idiosyncratic risk: one big streamer
#   ending moves the category total, while a diffuse audience is
#   self-averaging.  That is variance, not level, so it touches only
#   the spread: multiplier 1 + 0.5 * (HHI - 0.10), clipped to
#   [0.90, 1.20], hours 2-8 at full weight, fading to 1 by hour 12.
# `language` is deliberately unused: it has no clean sign (a high
# English share at 01:00Z is UK/NZ/AU, not the US evening wave).
# If `streams` is None, empty, malformed, or has fewer than 5 usable
# entries, the adjustment is exactly zero and v7 behaves like v6.
STREAMS_MIN_ENTRIES = 5
STREAMS_AGE_CENTER = 5.0      # hours; the neutral weighted mean age
STREAMS_AGE_SLOPE = 0.02      # log units of median shift per hour of age
STREAMS_AGE_CLIP = (0.0, 16.0)
STREAMS_SHIFT_CAP = 0.10      # max |median shift|, log units
STREAMS_HHI_CENTER = 0.10
STREAMS_HHI_SLOPE = 0.5
STREAMS_SPREAD_CLIP = (0.90, 1.20)


def streams_adjustment(case):
    """Per-hour (median shift, spread multiplier) from `streams`.

    Returns a dict {hour: (shift, mult)} for hours 2..24, or {} when
    the streams field is absent or unusable (v6 behaviour exactly).
    Defensive: every field is checked; anything unexpected is ignored
    rather than crashing the run.
    """
    streams = case.get("streams")
    if not isinstance(streams, list):
        return {}
    entries = []
    for row in streams:
        if not isinstance(row, dict):
            continue
        viewers = row.get("viewers")
        if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0 or viewers != viewers:  # also rejects NaN
            continue
        entries.append((float(viewers), row.get("started_at")))
    if len(entries) < STREAMS_MIN_ENTRIES:
        return {}

    total = sum(v for v, _ in entries)
    if total <= 0:
        return {}
    # Concentration: Herfindahl index of the top-20 viewer shares.
    hhi = sum((v / total) ** 2 for v, _ in entries)
    spread_excess = max(STREAMS_SPREAD_CLIP[0],
                        min(STREAMS_SPREAD_CLIP[1],
                            1.0 + STREAMS_HHI_SLOPE * (hhi - STREAMS_HHI_CENTER))) - 1.0

    # Viewer-weighted mean stream age at the checkpoint, in hours.
    checkpoint = None
    raw_checkpoint = case.get("checkpoint")
    if isinstance(raw_checkpoint, str):
        try:
            checkpoint = parse_time(raw_checkpoint)
        except ValueError:
            checkpoint = None
    age_num, age_den = 0.0, 0.0
    if checkpoint is not None:
        for viewers, started in entries:
            if not isinstance(started, str):
                continue
            try:
                age = (checkpoint - parse_time(started)).total_seconds() / 3600.0
            except (TypeError, ValueError):
                continue
            age = max(STREAMS_AGE_CLIP[0], min(STREAMS_AGE_CLIP[1], age))
            age_num += viewers * age
            age_den += viewers
    if age_den > 0:
        shift_base = -STREAMS_AGE_SLOPE * (age_num / age_den - STREAMS_AGE_CENTER)
        shift_base = max(-STREAMS_SHIFT_CAP, min(STREAMS_SHIFT_CAP, shift_base))
    else:
        shift_base = 0.0

    out = {}
    for hour in range(2, 25):
        # Median shift: full weight hours 2-5, fading to 0 at hour 9.
        if hour <= 5:
            fade = 1.0
        elif hour <= 8:
            fade = (9 - hour) / 4.0
        else:
            fade = 0.0
        # Spread: full weight hours 2-8, fading to 1 at hour 12.
        if hour <= 8:
            sfade = 1.0
        elif hour <= 11:
            sfade = (12 - hour) / 4.0
        else:
            sfade = 0.0
        out[hour] = (shift_base * fade, 1.0 + spread_excess * sfade)
    return out


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
    # v6: weighted least squares (early-access curves down-weighted)
    # with a ridge penalty on every coefficient except the intercept.
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
            xs, ys, ws = [], [], []
            for curve in train:
                if hour in curve.hours:
                    xs.append(curve_features(curve, n_feat))
                    ys.append(math.log(curve.hours[hour]))
                    ws.append(curve.train_weight)
            beta = (fit_wls(xs, ys, ws, RIDGE_LAMBDA)
                    if len(xs) >= n_feat + 3 else None)
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
    # the residual spread of the smoothed fit, per hour.  Residuals
    # are unweighted (as validated): the EA down-weighting applies
    # to the fit, not to the spread measurement.
    for j in range(n_coef):
        smoothed = smooth3([beta[j] for beta in coefs])
        for i in range(len(coefs)):
            coefs[i][j] = smoothed[i]

    # v6 spread modulation: the target's pre-48h slope vs the mean
    # slope of the training curves.  Computed once, from the same
    # training set the fit used.
    target_slope = target_pre_slope48(case, day_start)
    if target_slope is not None and train:
        mean_slope = sum(c.pre_slope48 for c in train) / len(train)
        spread_mult = math.exp(SPREAD_A * (target_slope - mean_slope))
    else:
        spread_mult = 1.0

    out_hours = []
    streams_adj = streams_adjustment(case)
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
        sd_base = sd
        # v6: apply the spread modulation after the floor/cap, with
        # no re-clipping, exactly as validated in the LOO harness.
        sd = sd * spread_mult
        # v7 streams adjustment (last; zero unless the case ships
        # streams).  The median shift applies exactly as in v5.
        # For the spread, the two layers multiply: v6's spread_mult
        # is unbounded, v5's smult is in [0.90, 1.20].  When the
        # streams spread layer is active at this hour (smult != 1.0)
        # the COMBINED multiplier is clipped to [0.90, 1.20]
        # relative to the v2 base spread sd_base, so the interaction
        # is bounded.  When streams are absent, or smult == 1.0 at
        # this hour, the spread is v6's exactly.
        shift, smult = streams_adj.get(hour, (0.0, 1.0))
        mu += shift
        if smult != 1.0:
            combined = max(0.90, min(1.20, spread_mult * smult))
            sd = sd_base * combined
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
