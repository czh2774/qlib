from __future__ import annotations

import importlib.util
import json
import sys
import types
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


class StubOrder:
    SELL = 0
    BUY = 1


def _load_exchange_module_with_stubs():
    module_names = (
        "qlib",
        "qlib.backtest",
        "qlib.backtest.ashare_semantics",
        "qlib.backtest.decision",
        "qlib.backtest.exchange",
        "qlib.backtest.high_performance_ds",
        "qlib.backtest.position",
        "qlib.config",
        "qlib.constant",
        "qlib.data",
        "qlib.data.data",
        "qlib.log",
        "qlib.utils",
        "qlib.utils.index_data",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    qlib_pkg = types.ModuleType("qlib")
    qlib_pkg.__path__ = []
    backtest_pkg = types.ModuleType("qlib.backtest")
    backtest_pkg.__path__ = []
    data_pkg = types.ModuleType("qlib.data")
    data_pkg.__path__ = []
    utils_pkg = types.ModuleType("qlib.utils")
    utils_pkg.__path__ = []

    ashare_module = types.ModuleType("qlib.backtest.ashare_semantics")
    ashare_module.build_joinquant_ashare_policy = ashare_semantics.build_joinquant_ashare_policy
    ashare_module.is_joinquant_ashare_limit_threshold = ashare_semantics.is_joinquant_ashare_limit_threshold

    decision_module = types.ModuleType("qlib.backtest.decision")
    decision_module.Order = StubOrder
    decision_module.OrderDir = types.SimpleNamespace(BUY=StubOrder.BUY, SELL=StubOrder.SELL)
    decision_module.OrderHelper = object

    high_performance_module = types.ModuleType("qlib.backtest.high_performance_ds")
    high_performance_module.BaseQuote = object
    high_performance_module.NumpyQuote = object

    position_module = types.ModuleType("qlib.backtest.position")
    position_module.BasePosition = object

    config_module = types.ModuleType("qlib.config")
    config_module.C = types.SimpleNamespace(region="cn")
    constant_module = types.ModuleType("qlib.constant")
    constant_module.REG_CN = "cn"
    constant_module.REG_TW = "tw"
    data_module = types.ModuleType("qlib.data.data")
    data_module.D = types.SimpleNamespace()
    log_module = types.ModuleType("qlib.log")
    log_module.get_module_logger = lambda *_args, **_kwargs: types.SimpleNamespace(info=lambda *_a, **_k: None)
    index_data_module = types.ModuleType("qlib.utils.index_data")
    index_data_module.IndexData = object

    sys.modules.update(
        {
            "qlib": qlib_pkg,
            "qlib.backtest": backtest_pkg,
            "qlib.backtest.ashare_semantics": ashare_module,
            "qlib.backtest.decision": decision_module,
            "qlib.backtest.high_performance_ds": high_performance_module,
            "qlib.backtest.position": position_module,
            "qlib.config": config_module,
            "qlib.constant": constant_module,
            "qlib.data": data_pkg,
            "qlib.data.data": data_module,
            "qlib.log": log_module,
            "qlib.utils": utils_pkg,
            "qlib.utils.index_data": index_data_module,
        }
    )
    try:
        exchange_spec = importlib.util.spec_from_file_location("qlib.backtest.exchange", EXCHANGE_PATH)
        assert exchange_spec is not None and exchange_spec.loader is not None
        exchange_module = importlib.util.module_from_spec(exchange_spec)
        sys.modules[exchange_spec.name] = exchange_module
        exchange_spec.loader.exec_module(exchange_module)
        return exchange_module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


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
    assert contract["semantic_boundary"]["authority_component"] == "qlib.backtest.ashare_semantics"
    assert contract["semantic_boundary"]["consumer_component"] == "rdagent.scenarios.qlib.ashare_semantics"
    assert "render_contract_projection_in_research_context" in contract["semantic_boundary"]["rdagent_allowed_actions"]
    assert "redefine_cost_model_or_exchange_kwargs" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert set(contract["failure_semantics"].values()) == {"fail_closed"}
    assert "cost_model" in contract["rdagent_must_not_redefine"]
    assert contract["market_semantics"]["region"] == "cn"
    assert contract["market_semantics"]["trade_unit"] == 100
    assert contract["market_semantics"]["position_type"] == "AsharePosition"
    assert contract["market_semantics"]["settlement_rule"] == "t_plus_1_stock"
    assert contract["market_semantics"]["same_day_sell_policy"] == (
        "shares_bought_today_are_unsellable_until_day_commit"
    )
    assert contract["runtime_surfaces"]["exchange_kwargs"] == ashare_semantics.joinquant_ashare_exchange_kwargs()
    assert contract["runtime_surfaces"]["backtest_kwargs"] == ashare_semantics.joinquant_ashare_backtest_kwargs()


def test_rdagent_ashare_contract_declares_evidence_and_prompt_projection_boundary() -> None:
    strict_contract = ashare_semantics.rdagent_ashare_semantic_contract()
    relaxed_contract = ashare_semantics.rdagent_ashare_semantic_contract(strict_price_limit=False)
    evidence = strict_contract["evidence_contract"]
    prompt_payload = strict_contract["prompt_projection_payload"]

    assert evidence["fingerprint_algorithm"] == "sha256_json_canonical_v1"
    assert len(evidence["semantic_fingerprint"]) == 64
    assert all(char in "0123456789abcdef" for char in evidence["semantic_fingerprint"])
    assert (
        evidence["semantic_fingerprint"]
        == ashare_semantics.rdagent_ashare_semantic_contract()["evidence_contract"]["semantic_fingerprint"]
    )
    assert evidence["semantic_fingerprint"] != relaxed_contract["evidence_contract"]["semantic_fingerprint"]
    assert "qlib_contract_fingerprint" in evidence["rdagent_required_evidence_fields"]
    assert (
        "runtime_surfaces.backtest_kwargs" in strict_contract["projection_contract"]["rdagent_prompt_forbidden_fields"]
    )
    assert "market_semantics.cost_model" in strict_contract["projection_contract"]["rdagent_prompt_forbidden_fields"]
    assert prompt_payload["projection_id"] == "qlib_joinquant_ashare_prompt_projection_v1"
    assert prompt_payload["projection_schema_version"] == "qlib_ashare_prompt_projection.v1"
    assert prompt_payload["projection_kind"] == "research_prompt_context_only"
    assert prompt_payload["contract_schema_version"] == "qlib_ashare_semantic_contract.v1"
    assert prompt_payload["semantic_fingerprint"] == evidence["semantic_fingerprint"]
    assert prompt_payload["market_semantics"] == {
        "market": "china_a_share",
        "region": "cn",
        "trade_unit": 100,
        "position_type": "AsharePosition",
        "settlement_rule": "t_plus_1_stock",
        "limit_threshold": "joinquant_ashare",
        "authoritative_limit_fields": ["$up_limit", "$down_limit"],
    }
    assert prompt_payload["price_limit_semantics"] == {
        "limit_threshold": "joinquant_ashare",
        "price_limit_mode": "strict",
        "authoritative_limit_fields": ["$up_limit", "$down_limit"],
        "field_authority": "provider_up_down_limit_fields",
        "missing_authoritative_fields": "fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets",
        "board_fallback_policy": "runtime_compatibility_only_when_authoritative_fields_are_absent",
        "board_limit_thresholds": {
            "main_board": 0.095,
            "star_chinext": 0.195,
            "bse": 0.295,
            "chinext_registration_start_date": "2020-08-24",
        },
        "rdagent_rule": "describe_only_do_not_redefine_price_limit_thresholds_or_fields",
    }
    assert relaxed_contract["prompt_projection_payload"]["price_limit_semantics"]["price_limit_mode"] == "auto"
    assert prompt_payload["settlement_semantics"] == {
        "settlement_rule": "t_plus_1_stock",
        "same_day_sell_policy": "shares_bought_today_are_unsellable_until_day_commit",
        "position_type": "AsharePosition",
        "runtime_authority": "qlib.backtest.position.AsharePosition",
        "rdagent_rule": "describe_only_do_not_redefine_position_or_settlement",
    }
    assert prompt_payload["order_unit_semantics"] == {
        "semantic_name": "a_share_round_lot",
        "qlib_parameter": "trade_unit",
        "trade_unit": 100,
        "amount_unit": "share",
        "buy_rounding_rule": "round_buy_amount_down_to_trade_unit_after_cash_and_volume_limits",
        "sell_rounding_rule": "round_sell_amount_down_to_trade_unit_except_full_liquidation",
        "full_liquidation_rule": "sell_all_remaining_position_without_round_lot_residual",
        "factor_adjustment_rule": "apply_order_factor_when_trade_uses_unadjusted_prices",
        "runtime_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
        "rdagent_rule": "describe_only_do_not_redefine_trade_unit_or_round_lot_policy",
    }
    assert "price_limit_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "settlement_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "order_unit_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "settlement_rule" in strict_contract["rdagent_must_not_redefine"]
    assert "same_day_sell_policy" in strict_contract["rdagent_must_not_redefine"]
    assert not _contains_key(prompt_payload, {"runtime_surfaces", "cost_model", "exchange_kwargs", "backtest_kwargs"})
    assert "open_cost" not in json.dumps(prompt_payload, sort_keys=True)


def test_rdagent_ashare_contract_splits_prompt_projection_from_runtime_handoff() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    handoff = contract["runtime_handoff_contract"]

    assert handoff["handoff_id"] == "qlib_joinquant_ashare_runtime_handoff_v1"
    assert handoff["handoff_kind"] == "qlib_owned_execution_kwargs"
    assert handoff["authority_component"] == "qlib.backtest.ashare_semantics"
    assert handoff["consumer_component"] == "rdagent.scenarios.qlib.ashare_semantics"
    assert handoff["source_fingerprint"] == contract["evidence_contract"]["semantic_fingerprint"]
    assert handoff["payload_paths"] == [
        "runtime_surfaces.exchange_kwargs",
        "runtime_surfaces.backtest_kwargs",
    ]
    assert "runtime_surfaces.policy_defaults" in handoff["forbidden_prompt_paths"]
    assert handoff["mutation_policy"] == "pass_through_only"
    assert "do_not_mutate_runtime_payload_values" in handoff["consumer_obligations"]


def test_rdagent_ashare_contract_is_machine_readable_json() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract(
        strict_price_limit=False,
    )

    round_tripped = json.loads(json.dumps(contract, sort_keys=True))

    assert round_tripped["runtime_surfaces"]["exchange_kwargs"]["ashare_price_limit_mode"] == "auto"
    assert round_tripped["market_semantics"]["cost_model"]["close_tax"] == pytest.approx(0.001)
    assert round_tripped["failure_semantics"]["malformed_contract"] == "fail_closed"
    assert round_tripped["prompt_projection_payload"]["projection_id"] == "qlib_joinquant_ashare_prompt_projection_v1"
    assert round_tripped["prompt_projection_payload"]["price_limit_semantics"]["price_limit_mode"] == "auto"
    assert round_tripped["prompt_projection_payload"]["settlement_semantics"]["settlement_rule"] == "t_plus_1_stock"
    assert round_tripped["prompt_projection_payload"]["order_unit_semantics"]["trade_unit"] == 100
    assert (
        round_tripped["prompt_projection_payload"]["order_unit_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_trade_unit_or_round_lot_policy"
    )
    assert round_tripped["runtime_handoff_contract"]["mutation_policy"] == "pass_through_only"


def test_exchange_joinquant_ashare_cost_helper_preserves_split_sell_tax() -> None:
    exchange_module = _load_exchange_module_with_stubs()
    Exchange = exchange_module.Exchange
    Order = exchange_module.Order

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


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(str(key) in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False
