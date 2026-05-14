# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


JOINQUANT_ASHARE_LIMIT_THRESHOLD = "joinquant_ashare"
JOINQUANT_ASHARE_ALIASES = frozenset(
    {
        JOINQUANT_ASHARE_LIMIT_THRESHOLD,
        "ashare_joinquant",
        "cn_ashare_joinquant",
    }
)


def is_joinquant_ashare_limit_threshold(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in JOINQUANT_ASHARE_ALIASES


@dataclass(frozen=True)
class JoinQuantAshareBacktestPolicy:
    """Local A-share policy preset for JoinQuant-comparable stock backtests.

    This is intentionally small and data-driven. If provider data includes
    Tushare-style up_limit/down_limit fields, those authoritative daily bounds
    drive tradability. If the fields are absent and mode is "auto", a board
    fallback is used only so legacy/simple Qlib datasets remain runnable.
    """

    trade_unit: int = 100
    deal_price: str = "close"
    open_cost: float = 0.0003
    close_cost: float = 0.0013
    min_cost: float = 5.0
    position_type: str = "AsharePosition"
    price_limit_mode: str = "auto"
    up_limit_field: str = "$up_limit"
    down_limit_field: str = "$down_limit"
    tolerance: float = 1e-8
    main_board_threshold: float = 0.095
    star_chinext_threshold: float = 0.195
    bse_threshold: float = 0.295

    def normalized_mode(self) -> str:
        mode = self.price_limit_mode.strip().lower()
        if mode not in {"auto", "strict", "board_fallback"}:
            raise ValueError(
                "price_limit_mode must be one of auto, strict, board_fallback; " f"got {self.price_limit_mode!r}"
            )
        return mode

    def limit_threshold_for_instrument(self, instrument: str) -> float:
        normalized = normalize_ashare_instrument(instrument)
        if normalized.startswith(("SH688", "SZ300")):
            return self.star_chinext_threshold
        if normalized.startswith("BJ") or normalized.startswith(("SH8", "SH4", "SH9", "SZ8", "SZ4", "SZ9")):
            return self.bse_threshold
        return self.main_board_threshold

    def apply_price_limits(self, quote_df: pd.DataFrame, *, buy_price: str, sell_price: str) -> pd.DataFrame:
        frame = quote_df.copy()
        suspended = frame["$close"].isna()
        if self._has_authoritative_limit_fields(frame):
            up_limit = pd.to_numeric(frame[self.up_limit_field], errors="coerce")
            down_limit = pd.to_numeric(frame[self.down_limit_field], errors="coerce")
            buy_values = pd.to_numeric(frame[buy_price], errors="coerce")
            sell_values = pd.to_numeric(frame[sell_price], errors="coerce")
            missing_bounds = up_limit.isna() | down_limit.isna()
            if self.normalized_mode() == "strict" and missing_bounds.any():
                missing_count = int(missing_bounds.sum())
                raise ValueError(
                    "joinquant_ashare strict price-limit mode requires non-null "
                    f"{self.up_limit_field}/{self.down_limit_field}; missing rows={missing_count}"
                )
            frame["limit_buy"] = buy_values.ge(up_limit - self.tolerance) | suspended | missing_bounds
            frame["limit_sell"] = sell_values.le(down_limit + self.tolerance) | suspended | missing_bounds
            return frame

        if self.normalized_mode() == "strict":
            raise ValueError(
                "joinquant_ashare strict price-limit mode requires provider fields "
                f"{self.up_limit_field} and {self.down_limit_field}"
            )
        threshold = self._board_threshold_series(frame)
        change = pd.to_numeric(frame["$change"], errors="coerce")
        frame["limit_buy"] = change.ge(threshold) | suspended
        frame["limit_sell"] = change.le(-threshold) | suspended
        return frame

    def _has_authoritative_limit_fields(self, quote_df: pd.DataFrame) -> bool:
        return self.up_limit_field in quote_df.columns and self.down_limit_field in quote_df.columns

    def _board_threshold_series(self, quote_df: pd.DataFrame) -> pd.Series:
        if isinstance(quote_df.index, pd.MultiIndex) and "instrument" in quote_df.index.names:
            instruments = quote_df.index.get_level_values("instrument")
        elif "instrument" in quote_df.columns:
            instruments = quote_df["instrument"]
        else:
            instruments = pd.Index([""] * len(quote_df))
        thresholds = [self.limit_threshold_for_instrument(str(instrument)) for instrument in instruments]
        return pd.Series(thresholds, index=quote_df.index, dtype="float64")


JOINQUANT_ASHARE_POLICY = JoinQuantAshareBacktestPolicy()


def normalize_ashare_instrument(instrument: str) -> str:
    raw = str(instrument).strip().upper()
    if "." in raw:
        code, exchange = raw.split(".", 1)
        if exchange in {"XSHG", "SH"}:
            return f"SH{code}"
        if exchange in {"XSHE", "SZ"}:
            return f"SZ{code}"
        if exchange in {"XBJ", "BJ"}:
            return f"BJ{code}"
    return raw


def joinquant_ashare_exchange_kwargs(*, strict_price_limit: bool = True) -> dict[str, Any]:
    """Return an Exchange kwargs preset aligned with JoinQuant stock costs."""

    return {
        "limit_threshold": JOINQUANT_ASHARE_LIMIT_THRESHOLD,
        "ashare_price_limit_mode": "strict" if strict_price_limit else "auto",
        "trade_unit": JOINQUANT_ASHARE_POLICY.trade_unit,
        "deal_price": JOINQUANT_ASHARE_POLICY.deal_price,
        "open_cost": JOINQUANT_ASHARE_POLICY.open_cost,
        "close_cost": JOINQUANT_ASHARE_POLICY.close_cost,
        "min_cost": JOINQUANT_ASHARE_POLICY.min_cost,
    }


def joinquant_ashare_backtest_kwargs(*, strict_price_limit: bool = True) -> dict[str, Any]:
    """Return top-level backtest kwargs for JoinQuant-style A-share stocks."""

    return {
        "pos_type": JOINQUANT_ASHARE_POLICY.position_type,
        "exchange_kwargs": joinquant_ashare_exchange_kwargs(strict_price_limit=strict_price_limit),
    }


def build_joinquant_ashare_policy(
    options: Mapping[str, Any] | None = None,
) -> JoinQuantAshareBacktestPolicy:
    if options is None:
        return JOINQUANT_ASHARE_POLICY
    allowed = set(JoinQuantAshareBacktestPolicy.__dataclass_fields__)
    unknown = sorted(str(key) for key in options if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown joinquant_ashare policy options: {unknown}")
    return JoinQuantAshareBacktestPolicy(**dict(options))
