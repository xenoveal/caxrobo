"""Research universe for the brute-force strategy search.

Deliberately separate from ``trading_bot.config.SYMBOLS``: production stays a
3-symbol product while research explores a wider universe. Every symbol here
was verified to have USDT-M perp 1d history at or before 2022-01-01, so all of
them cover the full 2023-01-01 -> present research span with warmup to spare.

Selection rule (recorded so it is auditable, not re-litigated later): the 20
highest-liquidity USDT-M perps that (a) have full history and (b) span
distinct sectors -- L1 majors, L2, DeFi, payments, memecoin, storage -- so
cross-sectional strategies see genuine dispersion rather than 20 proxies for
beta.
"""

# The three the PRD gate is defined on. Never reorder: reports key off this.
CORE = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

EXTENDED = (
    "BNBUSDT",   # exchange
    "XRPUSDT",   # payments
    "DOGEUSDT",  # memecoin
    "ADAUSDT",   # L1
    "AVAXUSDT",  # L1
    "LINKUSDT",  # oracle
    "DOTUSDT",   # L1
    "LTCUSDT",   # payments
    "TRXUSDT",   # L1
    "BCHUSDT",   # payments
    "NEARUSDT",  # L1
    "ATOMUSDT",  # L1
    "UNIUSDT",   # DeFi
    "AAVEUSDT",  # DeFi
    "FILUSDT",   # storage
    "OPUSDT",    # L2
    "APTUSDT",   # L1
)

UNIVERSE = CORE + EXTENDED

# Timeframes the research harness reads. 15m is intentionally excluded from the
# backfill for the extended set: the PRD retired the 15m tier on cost-frontier
# grounds, so paying ~125k bars/symbol for it would be pure storage cost.
RESEARCH_TIMEFRAMES = ("1d", "4h", "1h")
