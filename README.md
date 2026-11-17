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

- **Log-log regression per hour.** For each hour h = 2..24,
  `log(viewers_h) = a_h + b_h * log(hour-1 viewers)
  + c_h * log(pre-day peak / hour-1 viewers)`, fitted by least squares
  on the case's comparables. The slope `b_h` falls smoothly from ~1.06
  at hour 2 to ~0.65 in the evening: big hour-1 launches stay roughly
  flat, small hour-1 launches rise several-fold as the US unlocks and
  prime time arrives. The coefficients are smoothed across neighbouring
  hours, and the spread is 1.15× the per-hour residual SD (floored at
  0.18 log units).
- **Only gradual (rolling-midnight) launches train the fit.** A launch
  whose audience multiplies several-fold in a single hour is a
  simultaneous-global unlock, a different regime: hour 1 is then only
  pre-unlock hype. GTA VI unlocks at midnight local, country by
  country, so those launches are excluded from training (they roughly
  double the residual spread if kept in). If a case's notes instead say
  the launch was simultaneous, the fit trains on all comparables.
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

| model | LOO pinball, gradual launches (n=25) | LOO pinball, all launches (n=37) |
|---|---|---|
| House baseline (persistence) | 0.278 | 0.930 |
| Chosen model | **0.178** | — (different regime) |

90% coverage 0.89, typical median miss 33.6% (baseline: 0.278 loss on
the same 25 launches). The practice answers are not in the kit, so no
practice loss can be computed locally; the house baseline's published
practice loss is ~0.387. Full tables in [NOTES.md](NOTES.md).
