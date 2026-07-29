"""Baseline + harness self-test strategies.

``donchian_production`` re-expresses the CURRENT production rule inside the new
harness at its exact frozen parameters. It exists to be compared against the
Phase 4 measured numbers -- if the harness disagrees materially with
``trading_bot.backtest.engine`` on the same rule, the harness is wrong and every
result built on it is void. It is deliberately gridless (one combo) so it costs
one trial.

``lookahead_canary`` is a strategy that peeks at the next bar on purpose. It must
FAIL ``core.assert_causal``. A green causality audit that includes the canary
proves the audit can actually fail; without it, "all strategies passed" might
just mean the check does nothing.
"""

from __future__ import annotations

import numpy as np

import indicators as ta
from core import LONG, SHORT, Plan
from registry import register


@register(
    family="trend",
    rationale=(
        "The current production rule, at its frozen parameters: 20-bar Donchian "
        "break on the 4H setup tier, ADX(14)>25 on the 1D regime tier, stop at "
        "1.5*ATR(14) of the setup tier. Included as the calibration reference "
        "the whole search is measured against, not as a candidate."
    ),
)
def donchian_production(ctx):
    f4 = ctx.frame("4h")
    hi = ctx.align(ta.rolling_high(f4, 20), "4h")
    lo = ctx.align(ta.rolling_low(f4, 20), "4h")
    atr = ctx.align(ta.atr(f4, 14), "4h")
    # 55-bar mid-line trend filter, as production uses it (only 55 does this).
    mid = ctx.align(
        (ta.rolling_high(f4, 55, exclude_current=False)
         + ta.rolling_low(f4, 55, exclude_current=False)) / 2.0,
        "4h",
    )
    adx = ctx.align(ta.adx(ctx.frame("1d"), 14), "1d")

    close = ctx.trigger["close"].to_numpy(dtype=float)
    trending = adx > 25.0
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[(close > hi) & (close > mid) & trending] = LONG
    entry[(close < lo) & (close < mid) & trending] = SHORT
    return Plan(entry=entry, stop_dist=1.5 * atr, note="production-equivalent")


@register(
    family="trend",
    rationale=(
        "HARNESS SELF-TEST, not a trading idea. Uses next bar's close, so it "
        "must be disqualified by the causality audit. Its presence is what "
        "proves a clean audit means something."
    ),
)
def lookahead_canary(ctx):
    close = ctx.trigger["close"].to_numpy(dtype=float)
    atr = ctx.align(ta.atr(ctx.frame("4h"), 14), "4h")
    # np.roll(-1) brings the FUTURE bar's close into the current bar. This is
    # the canonical lookahead bug, written on purpose.
    nxt = np.roll(close, -1)
    entry = np.zeros(ctx.n, dtype=np.int8)
    entry[nxt > close] = LONG
    entry[nxt < close] = SHORT
    return Plan(entry=entry, stop_dist=1.5 * atr, note="MUST FAIL assert_causal")
