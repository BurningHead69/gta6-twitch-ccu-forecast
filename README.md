# GTA VI Twitch viewers — hour-by-hour forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/BurningHead69/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/BurningHead69/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the make.fun bounty
**[Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)**
(`gta6-twitch-ccu-forecast-01`).

`main.py` reads one case — everything recorded up to the end of the
first hour of GTA VI's release day, 2026-11-19 (UTC), hour 1's Twitch
viewer count included — and forecasts, for each of hours 2–24, five
quantiles (5%, 25%, 50%, 75%, 95%) of the concurrent viewers in the
"Grand Theft Auto VI" Twitch category. It runs offline, standard
library only, deterministically, in well under a second per case.

## How it forecasts

In one sentence: **a launch's later hours are predicted from its
hour-1 print and from how busy its category was in the day before
release, using only past launches that unlocked the same way.**

- **Log-log regression per hour (v6: weighted + ridge).** For each hour h = 2..24,
  `log(viewers_h) = a_h + b_h * log(hour-1 viewers)
  + c_h * log(pre-day peak / hour-1 viewers)
  + d_h * console_only + e_h * log(channels at hour 1)`, fitted by
  weighted least squares on the case's comparables, with a ridge
  penalty of 0.5 on every coefficient except the intercept.
  Early-access-contaminated comparables (an affirmed "early access"
  mention in their unlock notes; 11 of 38) are down-weighted to 0.5.
  The slope `b_h` falls smoothly from ~1.06 at hour 2 to ~0.65 in
  the evening: big hour-1 launches stay roughly flat, small hour-1
  launches rise several-fold as the US unlocks and prime time
  arrives. The coefficients are smoothed across neighbouring hours,
  and the spread is 1.15× the per-hour residual SD (floored at
  0.18 log units), modulated by exp(−2 · (target pre-48h slope −
  mean train pre-48h slope)) — see NOTES.md section 10.
- **Console-only launches rise more.** Of the gradual comparables,
  the 8 console-only ones peak at a median 2.66× hour 1, vs 1.26×
  for the 17 multi-platform ones. GTA VI is console-only, so the
  console-only indicator (1 for console-only, 0 otherwise, read from
  the case notes' release sentence) and log channels at hour 1 are
  fitted as extra predictors. `d_h` is ~+0.5–0.8 from hour 5 on.
- **Only gradual (rolling-midnight) launches train the fit.** A launch
  whose audience multiplies several-fold in a single hour is a
  simultaneous-global unlock, a different regime: hour 1 is then only
  pre-unlock hype. GTA VI unlocks at midnight local, country by
  country, so those launches are excluded from training (they roughly
  double the residual spread if kept in). If a case's notes instead say
  the launch was simultaneous, the fit trains on all comparables.
  v2 fixes a v1 bug here: v1 took the simultaneous branch whenever
  the word "simultaneous" appeared in the notes — including "no
  simultaneous launch", which misclassified God of War Ragnarök.
  v2 strips negated phrases before testing for an affirmed
  simultaneous unlock.
- **Streams layer (v7, unvalidated gamble).** If the case ships
  `streams` — the top 20 live streams at the last read before the
  checkpoint — two small, capped adjustments apply last, on top of
  the v6 forecast: the viewer-weighted mean stream age shifts the
  early-hour median by at most ±0.10 log units (old top streams =
  the Asia-Pacific wave winding down before the US unlocks; young
  top streams = the EU/UK wave just started), and the Herfindahl
  concentration of the top-20 shares scales the spread by 0.90–1.20×.
  The two spread layers multiply, so when the streams spread layer
  is active the combined multiplier is clipped to [0.90, 1.20]
  relative to the base spread; with streams absent v7 is identical
  to v6. Streams are null in every practice case, so this effect
  cannot be validated locally — see NOTES.md section 10.
- **Fallbacks:** if too few gradual comparables are usable, train on
  all of them; if no fit is possible at all, fall back to the house
  baseline (persistence with a widening spread). The output is always
  a valid forecast.

See [NOTES.md](NOTES.md) for the cross-validation numbers behind
these choices.

## Run it

```sh
# Practice cases from the kit (input/ holds p00.in … p03.in):
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# → out/p00.out … out/p03.out, each {"hours": [{"hour": 2, "q": [...]}, …]}
```

The grader runs `python3 /bot/main.py` with `/input` and `/output`
(the environment variables are only a local stand-in). Python 3.13,
standard library only, no network, no randomness.

## Files

- `main.py` — the whole entry (single file).
- `NOTES.md` — model choice, cross-validation results per model tried,
  baseline comparison, runtime/determinism checks, and risks.
- `LICENSE` — MIT.

## Validation (summary)

Leave-one-out cross-validation on the comparables, treating each
launch as a pseudo-case (its hour-1 print known, hours 2–24 predicted
from the other launches):

| model | LOO pinball, gradual launches (n=25) | LOO pinball, console-only (n=8) | LOO pinball, all launches (n=37) |
|---|---|---|---|
| House baseline (persistence) | 0.278 | 0.323 | 0.930 |
| v1 (submitted) | 0.178 | 0.166 | 0.343 |
| v2 (submitted, practice 0.180) | 0.165 | 0.148 | 0.339 |
| **v6 (this entry's base)** | **0.159** | **0.144** | **0.330** |
| **v7 (this entry: v6 + streams layer)** | **0.159** | **0.144** | **0.330** |

v7's LOO is identical to v6's (streams are null in practice, so
the streams layer is inactive there — verified per-case, max
difference 0.000000). v7: 90% coverage and typical median miss on
the gradual launches are reported in NOTES.md section 10. The practice answers are not in the kit,
so no practice loss can be computed locally — the practice-proxy
evaluation was checked and found not to exist in the kit (no case
holds another case's answer); the house baseline's published practice
loss is ~0.387, v1's public practice loss was 0.289, and v2's public
practice loss is 0.180 (rank ~1237–1417 of ~4900–5500, moving with
the leaderboard). Full tables, every model tried, and the honest limits
of the LOO evidence in [NOTES.md](NOTES.md).

> **Status:** v7 is built and verified here but **not yet submitted**
> and has no repo. The current final entry on make.fun is v5
> (streams gamble, repo BurningHead69/gta6-twitch-ccu-forecast,
> commit f37bc5c7, submission 01a1e9de-8475-761c-b635-9078dfc3f1e1).
> v6 was never submitted; it is v7's base model, and v7 adds v5's
> streams layer on top of it. With streams absent v7 is
> byte-identical to v6, so its practice loss is expected to equal
> v6's (~0.173 if LOO transfers one-to-one).
