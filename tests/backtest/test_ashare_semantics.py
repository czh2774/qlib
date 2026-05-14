from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "qlib/backtest/ashare_semantics.py"
EXCHANGE_PATH = REPO_ROOT / "qlib/backtest/exchange.py"

spec = importlib.util.spec_from_file_location(
    "ashare_semantics_under_test", MODULE_PATH
)
assert spec is not None and spec.loader is not None
ashare_semantics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ashare_semantics
spec.loader.exec_module(ashare_semantics)

JoinQuantAshareBacktestPolicy = ashare_semantics.JoinQuantAshareBacktestPolicy


def _quote_frame(rows: list[tuple[str, str, dict[str, float | None]]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(instrument, pd.Timestamp(trade_date)) for instrument, trade_date, _ in rows],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame([values for _, _, values in rows], index=index)


def test_joinquant_ashare_policy_uses_authoritative_up_down_limits() -> None:
    frame = _quote_frame(
        [
            (
                "SH600000",
                "2020-01-02",
                {
                    "$close": 11.0,
                    "$change": 0.10,
                    "$up_limit": 11.0,
                    "$down_limit": 9.0,
                },
            ),
            (
                "SH600001",
                "2020-01-02",
                {
                    "$close": 9.0,
                    "$change": -0.10,
                    "$up_limit": 11.0,
                    "$down_limit": 9.0,
                },
            ),
            (
                "SH600002",
                "2020-01-02",
                {
                    "$close": None,
                    "$change": None,
                    "$up_limit": 11.0,
                    "$down_limit": 9.0,
                },
            ),
        ]
    )

    limited = JoinQuantAshareBacktestPolicy(
        price_limit_mode="strict"
    ).apply_price_limits(
        frame,
        buy_price="$close",
        sell_price="$close",
    )

    assert bool(limited.loc[("SH600000", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert not bool(limited.loc[("SH600000", pd.Timestamp("2020-01-02")), "limit_sell"])
    assert not bool(limited.loc[("SH600001", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert bool(limited.loc[("SH600001", pd.Timestamp("2020-01-02")), "limit_sell"])
    assert bool(limited.loc[("SH600002", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert bool(limited.loc[("SH600002", pd.Timestamp("2020-01-02")), "limit_sell"])


def test_joinquant_ashare_strict_mode_requires_authoritative_limit_fields() -> None:
    frame = _quote_frame(
        [
            (
                "SH600000",
                "2020-01-02",
                {
                    "$close": 10.0,
                    "$change": 0.0,
                },
            )
        ]
    )

    with pytest.raises(
        ValueError, match="strict price-limit mode requires provider fields"
    ):
        JoinQuantAshareBacktestPolicy(price_limit_mode="strict").apply_price_limits(
            frame,
            buy_price="$close",
            sell_price="$close",
        )


def test_joinquant_ashare_board_fallback_uses_board_specific_thresholds() -> None:
    frame = _quote_frame(
        [
            ("SH600000", "2020-01-02", {"$close": 10.0, "$change": 0.096}),
            ("SZ300750", "2020-01-02", {"$close": 10.0, "$change": 0.096}),
            ("SZ300750", "2020-01-03", {"$close": 10.0, "$change": 0.196}),
            ("BJ430047", "2020-01-02", {"$close": 10.0, "$change": 0.296}),
        ]
    )

    limited = JoinQuantAshareBacktestPolicy(
        price_limit_mode="board_fallback"
    ).apply_price_limits(
        frame,
        buy_price="$close",
        sell_price="$close",
    )

    assert bool(limited.loc[("SH600000", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert not bool(limited.loc[("SZ300750", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert bool(limited.loc[("SZ300750", pd.Timestamp("2020-01-03")), "limit_buy"])
    assert bool(limited.loc[("BJ430047", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert not bool(limited["limit_sell"].any())


def test_exchange_source_delegates_joinquant_ashare_limits_to_policy() -> None:
    source = EXCHANGE_PATH.read_text(encoding="utf-8")

    assert "LT_JOINQUANT_ASHARE" in source
    assert "build_joinquant_ashare_policy(" in source
    assert "ashare_limit_options" in source
    assert "is_joinquant_ashare_limit_threshold(limit_threshold)" in source
    assert "self._joinquant_ashare_policy.apply_price_limits" in source
    assert (
        "necessary_fields.add(self._joinquant_ashare_policy.up_limit_field)" in source
    )
    assert (
        "necessary_fields.add(self._joinquant_ashare_policy.down_limit_field)" in source
    )
