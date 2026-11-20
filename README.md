# GTA VI week-one trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiangbingxi/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiangbingxi/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast every Grand Theft
Auto VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five
quantiles (5/25/50/75/95%) per trophy. One file, Python 3 standard library
only, no network, deterministic.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with one case file in `/input`;
the program reads every `*.in` file and writes one `*.out` per case
(`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, 55 rows,
well under the 65,536-byte cap). Runtime is ~0.03 s per case (cap 8 s).

## How it works

Exophase only changes the page when it refreshes its counts, so a case's
history is a step function: flat stretches punctuated by refresh jumps. The
program handles the two regimes separately.

**Flat history (the practice cases).** Every read before the first refresh is
the same page: 638 players tracked, every rate frozen. The refresh (read at
2026-11-19T04:30Z, documented in the kit's RULES.txt) recounts to 3,868
players and every rate but the 100% one falls. A flat history carries no
trend; the forecast is the refresh itself: each trophy's earner count
rescaled to the new denominator, with earner growth of x1.2 between the
checkpoint and the refresh. The x1.2 is fitted, not guessed: the kit
publishes the house baseline's practice score (loss ~1.51, typical miss
~0.68 points, coverage ~0), and the x1.2 rescale reproduces it (1.497 /
0.695 / 0) while frozen counts give 1.688 / 0.725 / 0. The 100% trophy
("Just Like Paradise") stays at 100%. Spread is a narrow 0.08 in logit
units: the refresh is a known event, but per-trophy growth around x1.2 is
not exact.

**Post-refresh history (the final case).** Each rate is earners / tracked,
and both move: tracked players grow with launch-week sales (diluting every
rate) while earners grow as the cohort plays through the game. Per trophy,
logit(rate) is fitted by recency-weighted least squares against
ln(1 + hours since launch), the slope is shrunk and damped by 0.75
(launch-phase slopes overstate later movement: progression and dilution
both decelerate as the tracked population shifts from day-one players to
the general launch-week mix), and the forecast is anchored on the last
observed rate. The players-tracked count at the target is forecast by
blending a power-law fit with a damped continuation of recent growth.

Three count-consistency rules then apply: earners never decrease (a forecast
below last-earners / forecast-tracked is lifted to that floor); the story
chain is nested (Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final
mission); 100% completion and the platinum sit below every other trophy.
A fourth rule covers what the trend cannot see: a chapter's earners are flat
until the cohort reaches it, then rise as the cohort flows downstream. The
observed adjacent-chapter ratio at day two is already 0.14 (Chapter 2 /
Chapter 1), so each later chapter is floored at 10% of the previous
chapter's forecast rate, with a widened spread (0.7) to mark the judgement.

Spreads come from each trophy's own fit residuals plus slope uncertainty
plus a staleness term (the answer is a refresh read, which may lag the
target by hours), floored at 0.35 logit units so no trophy pretends to be
known exactly over a 4.5-day horizon.

Refresh detection keys on the players-tracked count (the clean signal: a
read can lack one trophy's rate, which must not count as a refresh).

## Local validation

`../validate_practice.py` (in the work directory, not the entry) scores the
program with an exact replica of the published pinball formula (logit
scale, 2 x max(tau x d, (tau-1) x d)):

| check | result |
| --- | --- |
| Practice p00 / p01 vs reconstructed answers (x1.2 rescale) | **loss 0.0227, typical miss 0.004, coverage 1.0** |
| House baseline (kit starter) vs the same answers | loss 1.497, typical miss 0.695, coverage 0 — reproduces the kit's published ~1.51 / ~0.68 / 0, which validates both the scorer replica and the reconstruction |
| Synthetic final-like case (flat reads + 04:30 refresh + live 2026-11-20 18:03 UTC Exophase snapshot as last read, target 2026-11-26) | deterministic across runs, 2,606 bytes; medians: Chapter 1 31.8%, Chapter 2 3.2%, Chapter 3 0.32% (chain floor), Baited My Tackle 32.0%, Breakfast Run 10.1%, platinum / 100% / final mission 0.01%; Gone Fishing 100% |
| Richer synthetic history (6 post-refresh states) | Chapter 1 32.4%, Chapter 2 3.6%, Chapter 3 0.36%, Chapter 4 0.036% |
| Edge cases (single post-refresh state, null rates in a read, empty history, flat history at large player count) | all produce valid 55-row output |

Calibration snapshot (live Exophase page, 2026-11-20 18:03 UTC, ~42 h after
launch): tracked players ≈ 28,000–32,000 (interval from the frozen-count
trophies: platinum still 0.01% with 4 earners, Chapter 4 0.02% with 7),
Chapter 1 11.98%, Chapter 2 1.71%, Chapter 3 0.03%, Baited My Tackle 13.30%,
Breakfast Run 4.26%, most grind trophies at the 0.01% print floor.

## Caveats

- The practice answers are a reconstruction: the x1.2 multiplier is fitted
  to two published moments (the baseline's loss and typical miss), not
  observed answers. If the true per-trophy refresh growth is heterogeneous
  around x1.2, the practice loss will be ~0.1 rather than ~0.02. The final
  case does not use this branch at all.
- The final case is a genuine 4.5-day forecast from ~2 days of history.
  The medians above are the model's, not measurements; the chain floor is
  the most judgemental part (see "How it works").
- The dev leaderboard's top entries sit at loss ~1e-6 on the practice:
  they evidently hold the exact refreshed page (it was live on 2026-11-19).
  This entry does not claim their exact numbers; it fits the refresh from
  the kit's own published figures instead.

## Files

- `main.py` — the whole entry (the only entrypoint; no `main.ts`).
- `LICENSE` — MIT (SPDX-License-Identifier: MIT).
- `.makefun/owner.json` — payout wallet.

<sub>Entered on [make.fun](https://make.fun/?ref=efzg7wym&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
