from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "qlib/backtest/ashare_semantics.py"
EXCHANGE_PATH = REPO_ROOT / "qlib/backtest/exchange.py"

spec = importlib.util.spec_from_file_location("ashare_semantics_under_test", MODULE_PATH)
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

    limited = JoinQuantAshareBacktestPolicy(price_limit_mode="strict").apply_price_limits(
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

    with pytest.raises(ValueError, match="strict price-limit mode requires provider fields"):
        JoinQuantAshareBacktestPolicy(price_limit_mode="strict").apply_price_limits(
            frame,
            buy_price="$close",
            sell_price="$close",
        )


def test_joinquant_ashare_strict_mode_allows_suspended_rows_without_limit_bounds() -> None:
    frame = _quote_frame(
        [
            (
                "SH600000",
                "2020-01-02",
                {
                    "$close": 10.0,
                    "$change": 0.0,
                    "$up_limit": 11.0,
                    "$down_limit": 9.0,
                },
            ),
            (
                "SH600000",
                "2020-01-03",
                {
                    "$close": None,
                    "$change": None,
                    "$up_limit": None,
                    "$down_limit": None,
                },
            ),
        ]
    )

    limited = JoinQuantAshareBacktestPolicy(price_limit_mode="strict").apply_price_limits(
        frame,
        buy_price="$close",
        sell_price="$close",
    )

    suspended_row = ("SH600000", pd.Timestamp("2020-01-03"))
    assert bool(limited.loc[suspended_row, "limit_buy"])
    assert bool(limited.loc[suspended_row, "limit_sell"])


def test_joinquant_ashare_strict_mode_rejects_missing_limits_on_non_suspended_rows() -> None:
    frame = _quote_frame(
        [
            (
                "SH600000",
                "2020-01-02",
                {
                    "$close": 10.0,
                    "$change": 0.0,
                    "$up_limit": 11.0,
                    "$down_limit": None,
                },
            ),
        ]
    )

    with pytest.raises(ValueError, match="non-suspended rows; missing rows=1"):
        JoinQuantAshareBacktestPolicy(price_limit_mode="strict").apply_price_limits(
            frame,
            buy_price="$close",
            sell_price="$close",
        )


def test_joinquant_ashare_board_fallback_uses_board_specific_thresholds() -> None:
    frame = _quote_frame(
        [
            ("SH600000", "2020-01-02", {"$close": 10.0, "$change": 0.096}),
            ("SH688012", "2020-01-02", {"$close": 10.0, "$change": 0.096}),
            ("SH688012", "2020-01-03", {"$close": 10.0, "$change": 0.196}),
            ("SZ300750", "2020-08-21", {"$close": 10.0, "$change": 0.096}),
            ("SZ300750", "2020-08-24", {"$close": 10.0, "$change": 0.096}),
            ("SZ300750", "2020-08-25", {"$close": 10.0, "$change": 0.196}),
            ("BJ430047", "2020-01-02", {"$close": 10.0, "$change": 0.296}),
        ]
    )

    limited = JoinQuantAshareBacktestPolicy(price_limit_mode="board_fallback").apply_price_limits(
        frame,
        buy_price="$close",
        sell_price="$close",
    )

    assert bool(limited.loc[("SH600000", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert not bool(limited.loc[("SH688012", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert bool(limited.loc[("SH688012", pd.Timestamp("2020-01-03")), "limit_buy"])
    assert bool(limited.loc[("SZ300750", pd.Timestamp("2020-08-21")), "limit_buy"])
    assert not bool(limited.loc[("SZ300750", pd.Timestamp("2020-08-24")), "limit_buy"])
    assert bool(limited.loc[("SZ300750", pd.Timestamp("2020-08-25")), "limit_buy"])
    assert bool(limited.loc[("BJ430047", pd.Timestamp("2020-01-02")), "limit_buy"])
    assert not bool(limited["limit_sell"].any())


def test_joinquant_ashare_policy_charges_sell_tax_outside_min_commission() -> None:
    policy = JoinQuantAshareBacktestPolicy()

    assert policy.calculate_trade_cost("buy", 1_000.0) == pytest.approx(5.0)
    assert policy.calculate_trade_cost("sell", 1_000.0) == pytest.approx(6.0)
    assert policy.calculate_trade_cost("buy", 100_000.0) == pytest.approx(30.0)
    assert policy.calculate_trade_cost("sell", 100_000.0) == pytest.approx(130.0)
    assert policy.calculate_trade_cost("sell", 0.0) == pytest.approx(0.0)


def test_joinquant_ashare_exchange_kwargs_expose_split_cost_policy_options() -> None:
    kwargs = ashare_semantics.joinquant_ashare_exchange_kwargs()
    cost_options = kwargs["ashare_limit_options"]

    assert kwargs["open_cost"] == pytest.approx(0.0003)
    assert kwargs["close_cost"] == pytest.approx(0.0013)
    assert kwargs["min_cost"] == pytest.approx(5.0)
    assert cost_options == {
        "open_cost": pytest.approx(0.0003),
        "close_commission": pytest.approx(0.0003),
        "close_tax": pytest.approx(0.001),
        "min_cost": pytest.approx(5.0),
    }


def test_rdagent_ashare_contract_declares_qlib_authority_boundary() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()

    assert contract["contract_id"] == ("rdagent_qlib_joinquant_ashare_semantic_contract_v1")
    assert contract["source_component"] == "qlib.backtest.ashare_semantics"
    assert contract["consumer_component"] == "rdagent.scenarios.qlib.ashare_semantics"
    assert contract["relationship"] == {
        "qlib_role": "executable_backtest_semantic_authority",
        "rdagent_role": "research_candidate_generation_context_consumer",
        "relationship_rule": (
            "RD-Agent may consume Qlib's A-share contract for research generation and evaluation context, "
            "but it must not redefine trade unit, position, price-limit, or cost semantics."
        ),
        "fail_closed_on_missing_contract": True,
    }
    assert "cost_model" in contract["rdagent_must_not_redefine"]
    assert contract["market_semantics"]["region"] == "cn"
    assert contract["market_semantics"]["trade_unit"] == 100
    assert contract["market_semantics"]["position_type"] == "AsharePosition"
    assert contract["runtime_surfaces"]["exchange_kwargs"] == ashare_semantics.joinquant_ashare_exchange_kwargs()
    assert contract["runtime_surfaces"]["backtest_kwargs"] == ashare_semantics.joinquant_ashare_backtest_kwargs()


def test_rdagent_ashare_contract_is_machine_readable_json() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract(
        strict_price_limit=False,
    )

    round_tripped = json.loads(json.dumps(contract, sort_keys=True))

    assert round_tripped["runtime_surfaces"]["exchange_kwargs"]["ashare_price_limit_mode"] == "auto"
    assert round_tripped["market_semantics"]["cost_model"]["close_tax"] == pytest.approx(0.001)


def test_exchange_joinquant_ashare_cost_helper_preserves_split_sell_tax() -> None:
    from qlib.backtest.exchange import Exchange, Order

    exchange = object.__new__(Exchange)
    exchange._joinquant_ashare_policy = ashare_semantics.build_joinquant_ashare_policy()
    exchange.min_cost = 5.0

    assert exchange._calculate_trade_cost(Order.SELL, 1_000.0, 0.0013, 0.0) == pytest.approx(6.0)
    assert exchange._calculate_trade_cost(Order.SELL, 100_000.0, 0.0013, 0.0) == pytest.approx(130.0)

    exchange._joinquant_ashare_policy = None
    assert exchange._calculate_trade_cost(Order.SELL, 1_000.0, 0.0013, 0.0) == pytest.approx(5.0)
    assert exchange._calculate_trade_cost(Order.SELL, 0.0, 0.0013, 0.0) == pytest.approx(0.0)


def test_exchange_source_delegates_joinquant_ashare_limits_to_policy() -> None:
    source = EXCHANGE_PATH.read_text(encoding="utf-8")

    assert "LT_JOINQUANT_ASHARE" in source
    assert "build_joinquant_ashare_policy(" in source
    assert "ashare_limit_options" in source
    assert "is_joinquant_ashare_limit_threshold(limit_threshold)" in source
    assert "self._joinquant_ashare_policy.apply_price_limits" in source
    assert "self._joinquant_ashare_policy.calculate_trade_cost" in source
    assert "necessary_fields.add(self._joinquant_ashare_policy.up_limit_field)" in source
    assert "necessary_fields.add(self._joinquant_ashare_policy.down_limit_field)" in source
