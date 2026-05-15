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
ACCOUNT_PATH = REPO_ROOT / "qlib/backtest/account.py"
EXCHANGE_PATH = REPO_ROOT / "qlib/backtest/exchange.py"
POSITION_PATH = REPO_ROOT / "qlib/backtest/position.py"
DATA_PATH = REPO_ROOT / "qlib/data/data.py"

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


def test_joinquant_ashare_normalizes_provider_instrument_codes_for_board_identity() -> None:
    policy = JoinQuantAshareBacktestPolicy()

    assert ashare_semantics.normalize_ashare_instrument("600000.XSHG") == "SH600000"
    assert ashare_semantics.normalize_ashare_instrument("000001.XSHE") == "SZ000001"
    assert ashare_semantics.normalize_ashare_instrument("430047.XBJ") == "BJ430047"
    assert policy.limit_threshold_for_instrument("688012.XSHG") == pytest.approx(0.195)
    assert policy.limit_threshold_for_instrument("300750.XSHE", "2020-08-21") == pytest.approx(0.095)
    assert policy.limit_threshold_for_instrument("300750.XSHE", "2020-08-24") == pytest.approx(0.195)
    assert policy.limit_threshold_for_instrument("430047.XBJ") == pytest.approx(0.295)


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
            "but it must not redefine universe-membership, trading-calendar/data-frequency, trade unit, position, execution-price, "
            "price-adjustment, "
            "suspension/tradability, price-limit, order-tradability, order-fill, account-position update, account valuation, settlement, cash-settlement, cash/shorting, liquidity/capacity, market-impact, or cost semantics."
        ),
        "fail_closed_on_missing_contract": True,
    }
    assert contract["semantic_boundary"]["authority_component"] == "qlib.backtest.ashare_semantics"
    assert contract["semantic_boundary"]["consumer_component"] == "rdagent.scenarios.qlib.ashare_semantics"
    assert "render_contract_projection_in_research_context" in contract["semantic_boundary"]["rdagent_allowed_actions"]
    assert "redefine_instrument_identity_or_board_mapping" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "redefine_universe_membership_or_instrument_filtering"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_trading_calendar_or_data_frequency" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_transaction_cost_model" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_suspension_or_tradability_rules" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_execution_price_or_frequency" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_price_adjustment_or_order_factor" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "treat_board_fallback_as_primary_price_limit_authority"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_order_tradability_or_limit_checks" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_order_fill_amount_or_clip_sequence" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_market_impact_or_cost_ratio" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "redefine_account_position_or_cash_mutation_order" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_account_valuation_or_bar_end_refresh" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "redefine_settlement_or_sellable_position_state" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_cash_settlement_or_sell_proceeds_availability"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_cash_buying_power_or_shorting_policy" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_liquidity_or_volume_capacity_policy" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_cost_model_or_exchange_kwargs" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert set(contract["failure_semantics"].values()) == {"fail_closed"}
    assert "instrument_identity_semantics" in contract["rdagent_must_not_redefine"]
    assert "universe_membership_semantics" in contract["rdagent_must_not_redefine"]
    assert "trading_calendar_semantics" in contract["rdagent_must_not_redefine"]
    assert "transaction_cost_semantics" in contract["rdagent_must_not_redefine"]
    assert "market_impact_semantics" in contract["rdagent_must_not_redefine"]
    assert "account_update_semantics" in contract["rdagent_must_not_redefine"]
    assert "account_valuation_semantics" in contract["rdagent_must_not_redefine"]
    assert "suspension_tradability_semantics" in contract["rdagent_must_not_redefine"]
    assert "execution_price_semantics" in contract["rdagent_must_not_redefine"]
    assert "price_adjustment_semantics" in contract["rdagent_must_not_redefine"]
    assert "price_limit_semantics" in contract["rdagent_must_not_redefine"]
    assert "order_tradability_semantics" in contract["rdagent_must_not_redefine"]
    assert "order_fill_amount_semantics" in contract["rdagent_must_not_redefine"]
    assert "settlement_semantics" in contract["rdagent_must_not_redefine"]
    assert "cash_settlement_semantics" in contract["rdagent_must_not_redefine"]
    assert "cash_constraint_semantics" in contract["rdagent_must_not_redefine"]
    assert "liquidity_capacity_semantics" in contract["rdagent_must_not_redefine"]
    assert "cost_model" in contract["rdagent_must_not_redefine"]
    assert contract["market_semantics"]["region"] == "cn"
    assert contract["market_semantics"]["data_frequency"] == "day"
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
    assert "universe_membership_semantics" in evidence["fingerprint_scope"]
    assert "cash_settlement_semantics" in evidence["fingerprint_scope"]
    assert "order_tradability_semantics" in evidence["fingerprint_scope"]
    assert "order_fill_amount_semantics" in evidence["fingerprint_scope"]
    assert "market_impact_semantics" in evidence["fingerprint_scope"]
    assert "account_update_semantics" in evidence["fingerprint_scope"]
    assert "account_valuation_semantics" in evidence["fingerprint_scope"]
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
        "data_frequency": "day",
        "trade_unit": 100,
        "position_type": "AsharePosition",
        "settlement_rule": "t_plus_1_stock",
        "limit_threshold": "joinquant_ashare",
        "authoritative_limit_fields": ["$up_limit", "$down_limit"],
    }
    assert prompt_payload["instrument_identity_semantics"] == {
        "semantic_name": "a_share_instrument_identity",
        "canonical_code_format": "exchange_prefix_plus_six_digit_code",
        "canonical_exchange_prefixes": ["SH", "SZ", "BJ"],
        "accepted_provider_suffixes": {
            "XSHG": "SH",
            "SH": "SH",
            "XSHE": "SZ",
            "SZ": "SZ",
            "XBJ": "BJ",
            "BJ": "BJ",
        },
        "normalization_examples": {
            "600000.XSHG": "SH600000",
            "000001.XSHE": "SZ000001",
            "430047.XBJ": "BJ430047",
        },
        "board_identity_rules": [
            {"match": "SH688*", "board": "star_market"},
            {
                "match": "SZ300*",
                "board": "chinext_registration_sensitive",
                "effective_start": "2020-08-24",
            },
            {"match": "BJ*|SH8*|SH4*|SH9*|SZ8*|SZ4*|SZ9*", "board": "beijing_stock_exchange"},
            {"match": "fallback", "board": "main_board"},
        ],
        "price_limit_dependency": "board_identity_is_runtime_fallback_only_when_authoritative_limit_fields_absent",
        "runtime_authority": "qlib.backtest.ashare_semantics.normalize_ashare_instrument",
        "board_classification_authority": (
            "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.limit_threshold_for_instrument"
        ),
        "rdagent_rule": "describe_only_do_not_redefine_instrument_or_board_identity",
    }
    assert prompt_payload["universe_membership_semantics"] == {
        "semantic_name": "a_share_universe_membership",
        "membership_input": "Exchange.codes_or_D.instruments_market",
        "instrument_provider_authority": "qlib.data.data.InstrumentProvider.list_instruments",
        "local_provider_authority": "qlib.data.data.LocalInstrumentProvider.list_instruments",
        "exchange_codes_authority": "qlib.backtest.exchange.Exchange.__init__",
        "market_universe_rule": "string_codes_are_resolved_by_qlib_D_instruments",
        "membership_window_rule": "instrument_start_end_spans_are_clipped_to_requested_calendar_window",
        "calendar_boundary_rule": "start_end_defaults_and_membership_filtering_use_qlib_calendar_boundaries",
        "filter_pipe_rule": "qlib_instrument_filter_pipe_is_applied_after_calendar_window_clipping",
        "as_list_rule": "as_list_returns_only_instruments_with_nonempty_effective_spans",
        "static_universe_rule": "rdagent_must_not_treat_all_a_or_index_universe_as_static_without_qlib_membership_spans",
        "survivorship_rule": "membership_must_remain_point_in_time_by_qlib_instrument_spans_and_filters",
        "rdagent_rule": "describe_only_do_not_redefine_universe_membership_or_filters",
    }
    assert prompt_payload["trading_calendar_semantics"] == {
        "semantic_name": "a_share_daily_trading_calendar",
        "calendar_frequency": "day",
        "calendar_provider_authority": "qlib.data.data.CalendarProvider.calendar",
        "calendar_locator_authority": "qlib.data.data.CalendarProvider.locate_index",
        "exchange_frequency_parameter": "freq",
        "exchange_default_frequency": "day",
        "index_level": "datetime",
        "instrument_window_rule": "instrument_membership_is_filtered_against_calendar_boundaries",
        "non_trading_day_rule": "calendar_locate_index_maps_start_forward_and_end_backward_to_real_trading_days",
        "future_calendar_rule": "future_trading_days_require_qlib_future_calendar_support_not_prompt_invention",
        "synthetic_session_rule": "rdagent_must_not_invent_non_qlib_calendar_sessions",
        "rdagent_rule": "describe_only_do_not_redefine_trading_calendar_or_data_frequency",
    }
    assert prompt_payload["transaction_cost_semantics"] == {
        "semantic_name": "a_share_transaction_cost_structure",
        "cost_model_scope": "qlib_runtime_execution_only",
        "buy_cost_components": ["commission", "minimum_commission_floor"],
        "sell_cost_components": ["commission", "stamp_tax", "minimum_commission_floor"],
        "minimum_fee_rule": "commission_floor_applies_to_nonzero_trade_value",
        "zero_trade_rule": "zero_trade_value_has_zero_cost",
        "market_impact_rule": "optional_impact_cost_is_added_by_runtime_execution",
        "numeric_values_exposure": "runtime_handoff_only_not_prompt_projection",
        "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.calculate_trade_cost",
        "rdagent_rule": "describe_only_do_not_redefine_transaction_cost_model",
    }
    assert prompt_payload["market_impact_semantics"] == {
        "semantic_name": "a_share_market_impact_cost_adjustment",
        "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "cost_authority": "qlib.backtest.exchange.Exchange._calculate_trade_cost",
        "volume_authority": "qlib.backtest.exchange.Exchange.get_volume",
        "capacity_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
        "configuration_parameter": "impact_cost",
        "volume_field": "$volume",
        "total_trade_value_rule": "total_trade_value_equals_quote_volume_times_trade_price",
        "impact_cost_ratio_rule": "impact_cost_times_post_volume_clip_trade_value_over_total_trade_value_squared",
        "missing_volume_rule": "missing_zero_or_nan_total_trade_value_uses_raw_impact_cost_ratio",
        "cost_ratio_rule": "adjusted_cost_ratio_is_added_to_buy_or_sell_cost_ratio_before_cash_guards",
        "final_cost_rule": "trade_cost_is_recomputed_after_final_deal_amount_with_adjusted_cost_ratio",
        "joinquant_cost_rule": "joinquant_ashare_policy_receives_adjusted_cost_ratio_as_impact_cost",
        "numeric_value_exposure": "runtime_handoff_only_not_prompt_projection",
        "rdagent_rule": "describe_only_do_not_redefine_market_impact_or_cost_ratio",
    }
    assert prompt_payload["account_update_semantics"] == {
        "semantic_name": "a_share_account_position_cash_mutation",
        "execution_authority": "qlib.backtest.exchange.Exchange.deal_order",
        "account_update_authority": "qlib.backtest.account.Account.update_order",
        "account_metrics_authority": "qlib.backtest.account.Account._update_state_from_order",
        "position_update_authority": "qlib.backtest.position.Position.update_order",
        "ashare_sellable_update_authority": "qlib.backtest.position.AsharePosition._sell_stock",
        "trade_update_trigger": "only_trade_value_greater_than_one_e_minus_five_mutates_account_or_position",
        "failed_or_zero_trade_rule": "failed_order_or_zero_trade_value_does_not_update_position_or_account",
        "handoff_rule": "exchange_passes_final_trade_value_cost_and_price_to_account_or_position_update",
        "trade_amount_rule": "mutated_amount_equals_trade_value_divided_by_trade_price",
        "buy_mutation_order": "position_updates_before_account_metrics",
        "sell_mutation_order": "account_metrics_update_before_position_update",
        "buy_cash_rule": "buy_subtracts_trade_value_plus_cost_from_cash",
        "sell_cash_rule": "sell_routes_trade_value_minus_cost_to_cash_or_cash_delay_by_settle_type",
        "sellable_amount_rule": "ashare_sells_reduce_sellable_amount_and_day_bar_count_refresh_releases_total_amount",
        "infinite_position_rule": "skip_update_position_does_not_mutate_account_or_position",
        "rdagent_rule": "describe_only_do_not_redefine_account_position_or_cash_mutation_order",
    }
    assert prompt_payload["account_valuation_semantics"] == {
        "semantic_name": "a_share_account_bar_end_valuation",
        "bar_end_authority": "qlib.backtest.account.Account.update_bar_end",
        "position_refresh_authority": "qlib.backtest.account.Account.update_current_position",
        "portfolio_metrics_authority": "qlib.backtest.account.Account.update_portfolio_metrics",
        "history_position_authority": "qlib.backtest.account.Account.update_hist_positions",
        "price_update_authority": "qlib.backtest.position.Position.update_stock_price",
        "value_authority": "qlib.backtest.position.Position.calculate_value",
        "stock_value_authority": "qlib.backtest.position.Position.calculate_stock_value",
        "holding_count_authority": "qlib.backtest.position.Position.add_count_all",
        "ashare_sellable_release_authority": "qlib.backtest.position.AsharePosition.add_count_all",
        "close_price_authority": "qlib.backtest.exchange.Exchange.get_close",
        "bar_end_sequence": [
            "refresh_current_position_prices_and_holding_counts",
            "update_portfolio_metrics_when_enabled",
            "snapshot_history_positions_when_enabled",
            "update_trade_indicators",
        ],
        "mark_price_rule": "non_suspended_positions_mark_to_bar_close_at_bar_end",
        "suspension_price_rule": "suspended_positions_keep_previous_price_during_bar_end_refresh",
        "account_value_rule": "account_value_equals_stock_value_plus_cash_plus_cash_delay",
        "stock_value_rule": "stock_value_equals_position_amount_times_current_position_price",
        "portfolio_return_rule": "return_rate_uses_account_earning_plus_current_cost_over_last_account_value",
        "history_snapshot_rule": "history_positions_store_deepcopy_after_now_account_value_and_weights_refresh",
        "holding_count_rule": "bar_end_refresh_increments_position_count_for_account_frequency",
        "daily_sellable_release_rule": "ashare_day_bar_count_refresh_releases_total_amount_to_sellable_amount",
        "infinite_position_rule": "skip_update_position_does_not_refresh_prices_counts_metrics_or_history",
        "rdagent_rule": "describe_only_do_not_redefine_account_valuation_or_bar_end_refresh",
    }
    assert prompt_payload["suspension_tradability_semantics"] == {
        "semantic_name": "a_share_suspension_tradability",
        "suspension_indicator_field": "$close",
        "suspension_indicator_rule": "missing_close_price_marks_suspended",
        "non_tradable_rule": "suspended_rows_are_not_buyable_or_sellable",
        "limit_flag_projection": "qlib_sets_limit_buy_and_limit_sell_true_for_suspended_rows",
        "authoritative_limit_interaction": "suspension_takes_precedence_over_up_down_limit_fields",
        "missing_limit_bounds_rule": "missing_limit_bounds_are_tolerated_only_when_close_is_missing",
        "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
        "rdagent_rule": "describe_only_do_not_redefine_suspension_or_tradability",
    }
    assert prompt_payload["execution_price_semantics"] == {
        "semantic_name": "a_share_daily_close_execution_price",
        "qlib_parameter": "deal_price",
        "execution_price_field": "$close",
        "execution_frequency": "daily_bar_backtest",
        "price_source_authority": "qlib_exchange_deal_price",
        "intraday_execution_rule": "not_intraday_or_auction_simulation",
        "candidate_research_rule": "generated_factors_must_not_assume_intraday_fill_prices",
        "runtime_authority": "qlib.backtest.ashare_semantics.joinquant_ashare_exchange_kwargs",
        "rdagent_rule": "describe_only_do_not_redefine_execution_price_or_frequency",
    }
    assert prompt_payload["price_adjustment_semantics"] == {
        "semantic_name": "a_share_price_adjustment_order_factor",
        "factor_field": "$factor",
        "factor_usage": "convert_adjusted_amounts_to_trade_unit_amounts_when_unadjusted_prices_are_used",
        "missing_factor_rule": (
            "non_suspended_rows_with_missing_factor_use_adjusted_price_mode_and_disable_trade_unit_rounding"
        ),
        "adjusted_price_mode_rule": "trade_unit_rounding_is_not_supported_when_adjusted_price_mode_is_active",
        "extra_quote_factor_rule": "missing_extra_quote_factor_defaults_to_one",
        "suspension_interaction": "missing_factor_is_tolerated_when_close_is_missing",
        "runtime_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
        "rdagent_rule": "describe_only_do_not_redefine_price_adjustment_or_order_factor",
    }
    assert prompt_payload["price_limit_semantics"] == {
        "semantic_name": "a_share_price_limit_authority",
        "limit_threshold": "joinquant_ashare",
        "price_limit_mode": "strict",
        "authoritative_limit_fields": ["$up_limit", "$down_limit"],
        "field_authority": "provider_up_down_limit_fields",
        "limit_flag_fields": ["limit_buy", "limit_sell"],
        "limit_flag_meaning": "true_flags_mark_direction_not_tradable",
        "buy_limit_rule": "buy_price_at_or_above_up_limit_or_suspended_sets_limit_buy",
        "sell_limit_rule": "sell_price_at_or_below_down_limit_or_suspended_sets_limit_sell",
        "missing_authoritative_fields": "fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets",
        "strict_mode_missing_fields_rule": "missing_authoritative_fields_or_non_suspended_bounds_fail_closed",
        "board_fallback_policy": "runtime_compatibility_only_when_authoritative_fields_are_absent",
        "fallback_authority_rule": "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority",
        "board_limit_thresholds": {
            "main_board": 0.095,
            "star_chinext": 0.195,
            "bse": 0.295,
            "chinext_registration_start_date": "2020-08-24",
        },
        "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
        "rdagent_rule": "describe_only_do_not_redefine_price_limit_thresholds_or_fields",
    }
    assert relaxed_contract["prompt_projection_payload"]["price_limit_semantics"]["price_limit_mode"] == "auto"
    assert prompt_payload["order_tradability_semantics"] == {
        "semantic_name": "a_share_order_tradability_gate",
        "runtime_authority": "qlib.backtest.exchange.Exchange.check_order",
        "tradability_authority": "qlib.backtest.exchange.Exchange.is_stock_tradable",
        "suspension_authority": "qlib.backtest.exchange.Exchange.check_stock_suspended",
        "price_limit_authority": "qlib.backtest.exchange.Exchange.check_stock_limit",
        "failure_result": "deal_amount_zero_trade_value_zero_cost_nan_price",
        "failed_order_state_field": "Order.deal_amount",
        "directional_limit_rule": "buy_orders_check_limit_buy_and_sell_orders_check_limit_sell",
        "all_direction_limit_rule": "missing_direction_checks_any_buy_or_sell_limit",
        "suspension_rule": "missing_close_or_unknown_stock_is_not_tradable",
        "limit_rule": "limit_flags_true_mark_direction_not_tradable",
        "decision_rule": "check_order_delegates_to_is_stock_tradable_before_deal_execution",
        "rdagent_rule": "describe_only_do_not_redefine_order_tradability_or_limit_checks",
    }
    assert prompt_payload["order_fill_amount_semantics"] == {
        "semantic_name": "a_share_order_fill_amount_gate",
        "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "fill_state_field": "Order.deal_amount",
        "initial_fill_rule": "deal_amount_starts_as_order_amount_before_runtime_clips",
        "clip_sequence": [
            "volume_capacity_clip",
            "sellable_position_clip",
            "sell_cash_cost_guard",
            "buy_cash_cost_guard",
            "round_lot_or_full_liquidation_clip",
        ],
        "volume_clip_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
        "sellable_position_authority": "qlib.backtest.position.Position.get_sellable_amount",
        "cash_authority": "qlib.backtest.position.Position.get_cash",
        "cash_limit_authority": "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit",
        "round_lot_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
        "factor_authority": "qlib.backtest.exchange.Exchange.get_factor",
        "unknown_position_rule": "unknown_position_uses_round_lot_without_cash_or_sellable_clips",
        "sell_full_liquidation_rule": (
            "sells_equal_to_current_sellable_amount_keep_full_liquidation_without_round_lot_residual"
        ),
        "trade_value_rule": "trade_value_is_final_deal_amount_times_trade_price",
        "cost_rule": "trade_cost_recomputed_after_final_deal_amount",
        "rdagent_rule": "describe_only_do_not_redefine_order_fill_amount_or_clip_sequence",
    }
    assert prompt_payload["settlement_semantics"] == {
        "semantic_name": "a_share_t_plus_1_stock_settlement",
        "settlement_rule": "t_plus_1_stock",
        "same_day_sell_policy": "shares_bought_today_are_unsellable_until_day_commit",
        "position_type": "AsharePosition",
        "sellable_state_field": "sellable_amount",
        "initial_sellable_rule": "existing_or_settled_holdings_are_sellable",
        "intraday_buy_rule": "same_day_buys_increase_total_amount_but_not_sellable_amount",
        "intraday_bar_rule": "non_day_bars_do_not_release_same_day_buys",
        "day_commit_rule": "day_bar_commit_sets_sellable_amount_to_total_amount",
        "sell_order_clip_rule": "sell_orders_are_clipped_by_position_get_sellable_amount",
        "sell_overdraft_rule": "AsharePosition_rejects_sells_above_sellable_amount",
        "runtime_authority": "qlib.backtest.position.AsharePosition",
        "exchange_clip_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "rdagent_rule": "describe_only_do_not_redefine_position_or_settlement",
    }
    assert prompt_payload["cash_constraint_semantics"] == {
        "semantic_name": "a_share_cash_buying_power_and_shorting_policy",
        "cash_state_field": "cash",
        "cash_query_rule": "buying_power_uses_position_get_cash_without_unsettled_cash",
        "buy_cash_rule": "buy_orders_are_clipped_by_available_cash_and_transaction_cost",
        "minimum_cost_rule": "orders_without_cash_for_minimum_cost_are_zeroed",
        "partial_buy_rule": "cash_insufficient_orders_are_reduced_by_exchange_cash_limit_then_round_lot",
        "shorting_policy": "equity_short_selling_is_not_enabled",
        "sell_position_rule": "sell_orders_are_clipped_by_position_get_sellable_amount",
        "sell_cash_rule": "sell_orders_zero_when_cash_plus_trade_value_cannot_cover_sell_cost",
        "runtime_authority": "qlib.backtest.exchange.Exchange._calc_trade_info_by_order",
        "cash_limit_authority": "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit",
        "position_cash_authority": "qlib.backtest.position.Position.get_cash",
        "rdagent_rule": "describe_only_do_not_redefine_cash_or_shorting_policy",
    }
    assert prompt_payload["cash_settlement_semantics"] == {
        "semantic_name": "a_share_sell_proceeds_cash_settlement",
        "settlement_authority": "qlib.backtest.position.Position",
        "settle_start_authority": "qlib.backtest.position.Position.settle_start",
        "settle_commit_authority": "qlib.backtest.position.Position.settle_commit",
        "available_cash_authority": "qlib.backtest.position.Position.get_cash",
        "delayed_cash_state_field": "cash_delay",
        "delayed_cash_mode": "Position.ST_CASH",
        "no_delay_cash_mode": "Position.ST_NO",
        "sell_proceeds_rule": "sell_proceeds_enter_cash_delay_when_settle_type_is_cash",
        "default_sell_proceeds_rule": "sell_proceeds_enter_cash_immediately_when_settle_type_is_none",
        "available_cash_rule": "get_cash_excludes_cash_delay_unless_include_settle_is_true",
        "account_value_rule": "calculate_value_includes_cash_delay",
        "commit_rule": "settle_commit_moves_cash_delay_into_cash_and_clears_delay_state",
        "rdagent_rule": "describe_only_do_not_redefine_cash_settlement_or_sell_proceeds_availability",
    }
    assert prompt_payload["liquidity_capacity_semantics"] == {
        "semantic_name": "a_share_volume_capacity_limit",
        "volume_field": "$volume",
        "capacity_parameter": "volume_threshold",
        "capacity_scope": "runtime_handoff_only_when_volume_threshold_is_configured",
        "default_capacity_rule": "no_prompt_defined_capacity_limit_in_default_joinquant_ashare_contract",
        "volume_limit_aggregation_rule": "multiple_volume_limits_are_aggregated_by_min",
        "cumulative_limit_rule": "cum_volume_limits_subtract_dealt_order_amount",
        "current_limit_rule": "current_volume_limits_use_current_quote_value",
        "dealt_order_state": "dealt_order_amount",
        "capacity_clip_rule": "order_deal_amount_is_clipped_to_nonnegative_configured_volume_capacity",
        "runtime_authority": "qlib.backtest.exchange.Exchange._clip_amount_by_volume",
        "threshold_parser_authority": "qlib.backtest.exchange.Exchange._get_vol_limit",
        "rdagent_rule": "describe_only_do_not_redefine_liquidity_or_volume_capacity",
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
    assert "instrument_identity_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "universe_membership_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "trading_calendar_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "transaction_cost_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "market_impact_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "account_update_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "account_valuation_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert (
        "suspension_tradability_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    )
    assert "execution_price_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "price_adjustment_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "price_limit_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "order_tradability_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "order_fill_amount_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "settlement_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "cash_settlement_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "cash_constraint_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "liquidity_capacity_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "order_unit_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "settlement_rule" in strict_contract["rdagent_must_not_redefine"]
    assert "same_day_sell_policy" in strict_contract["rdagent_must_not_redefine"]
    assert "data_frequency" in strict_contract["rdagent_must_not_redefine"]
    assert not _contains_key(prompt_payload, {"runtime_surfaces", "cost_model", "exchange_kwargs", "backtest_kwargs"})
    assert "open_cost" not in json.dumps(prompt_payload, sort_keys=True)
    assert "close_tax" not in json.dumps(prompt_payload, sort_keys=True)


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


def test_ashare_cash_constraint_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    cash_semantics = contract["prompt_projection_payload"]["cash_constraint_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert cash_semantics["runtime_authority"] == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    assert cash_semantics["cash_limit_authority"] == "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit"
    assert cash_semantics["position_cash_authority"] == "qlib.backtest.position.Position.get_cash"
    assert "cash = position.get_cash()" in exchange_source
    assert "cash < max(trade_val * cost_ratio, self.min_cost)" in exchange_source
    assert "max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio)" in exchange_source
    assert "TODO: make the trading shortable" in exchange_source
    assert "position.get_sellable_amount(order.stock_id)" in exchange_source


def test_ashare_order_tradability_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    order_tradability = contract["prompt_projection_payload"]["order_tradability_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert order_tradability["semantic_name"] == "a_share_order_tradability_gate"
    assert order_tradability["runtime_authority"] == "qlib.backtest.exchange.Exchange.check_order"
    assert order_tradability["tradability_authority"] == "qlib.backtest.exchange.Exchange.is_stock_tradable"
    assert order_tradability["suspension_authority"] == "qlib.backtest.exchange.Exchange.check_stock_suspended"
    assert order_tradability["price_limit_authority"] == "qlib.backtest.exchange.Exchange.check_stock_limit"
    assert order_tradability["directional_limit_rule"] == "buy_orders_check_limit_buy_and_sell_orders_check_limit_sell"
    assert order_tradability["all_direction_limit_rule"] == "missing_direction_checks_any_buy_or_sell_limit"
    assert order_tradability["suspension_rule"] == "missing_close_or_unknown_stock_is_not_tradable"
    assert order_tradability["limit_rule"] == "limit_flags_true_mark_direction_not_tradable"
    assert order_tradability["failure_result"] == "deal_amount_zero_trade_value_zero_cost_nan_price"
    assert order_tradability["decision_rule"] == "check_order_delegates_to_is_stock_tradable_before_deal_execution"
    assert "def check_stock_limit(" in exchange_source
    assert 'field="limit_buy", method="all"' in exchange_source
    assert 'field="limit_sell", method="all"' in exchange_source
    assert "return bool(buy_limit or sell_limit)" in exchange_source
    assert "def check_stock_suspended(" in exchange_source
    assert 'close = self.quote.get_data(stock_id, start_time, end_time, "$close")' in exchange_source
    assert "def is_stock_tradable(" in exchange_source
    assert "self.check_stock_suspended(stock_id, start_time, end_time)" in exchange_source
    assert "self.check_stock_limit(stock_id, start_time, end_time, direction)" in exchange_source
    assert "def check_order(self, order: Order) -> bool:" in exchange_source
    assert (
        "return self.is_stock_tradable(order.stock_id, order.start_time, order.end_time, order.direction)"
        in exchange_source
    )
    assert "order.deal_amount = 0.0" in exchange_source
    assert "return 0.0, 0.0, np.nan" in exchange_source


def test_ashare_order_fill_amount_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    fill_amount = contract["prompt_projection_payload"]["order_fill_amount_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert fill_amount["semantic_name"] == "a_share_order_fill_amount_gate"
    assert fill_amount["runtime_authority"] == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    assert fill_amount["fill_state_field"] == "Order.deal_amount"
    assert fill_amount["initial_fill_rule"] == "deal_amount_starts_as_order_amount_before_runtime_clips"
    assert fill_amount["clip_sequence"] == [
        "volume_capacity_clip",
        "sellable_position_clip",
        "sell_cash_cost_guard",
        "buy_cash_cost_guard",
        "round_lot_or_full_liquidation_clip",
    ]
    assert fill_amount["volume_clip_authority"] == "qlib.backtest.exchange.Exchange._clip_amount_by_volume"
    assert fill_amount["sellable_position_authority"] == "qlib.backtest.position.Position.get_sellable_amount"
    assert fill_amount["cash_authority"] == "qlib.backtest.position.Position.get_cash"
    assert fill_amount["cash_limit_authority"] == "qlib.backtest.exchange.Exchange._get_buy_amount_by_cash_limit"
    assert fill_amount["round_lot_authority"] == "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit"
    assert fill_amount["factor_authority"] == "qlib.backtest.exchange.Exchange.get_factor"
    assert "def _calc_trade_info_by_order(" in exchange_source
    assert "order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)" in exchange_source
    assert "order.deal_amount = order.amount  # set to full amount and clip it step by step" in exchange_source
    assert "self._clip_amount_by_volume(order, dealt_order_amount)" in exchange_source
    assert (
        "position.get_sellable_amount(order.stock_id) if position.check_stock(order.stock_id) else 0" in exchange_source
    )
    assert "order.deal_amount = self.round_amount_by_trade_unit(" in exchange_source
    assert "position.get_cash() + expected_trade_val < expected_trade_cost" in exchange_source
    assert "cash = position.get_cash()" in exchange_source
    assert "cash < max(trade_val * cost_ratio, self.min_cost)" in exchange_source
    assert "max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio)" in exchange_source
    assert "order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)" in exchange_source
    assert "trade_val = order.deal_amount * trade_price" in exchange_source
    assert "trade_cost = self._calculate_trade_cost(order.direction, trade_val, cost_ratio, adj_cost_ratio)" in (
        exchange_source
    )
    assert "return trade_price, trade_val, trade_cost" in exchange_source


def test_ashare_market_impact_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    market_impact = contract["prompt_projection_payload"]["market_impact_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert market_impact["semantic_name"] == "a_share_market_impact_cost_adjustment"
    assert market_impact["runtime_authority"] == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    assert market_impact["cost_authority"] == "qlib.backtest.exchange.Exchange._calculate_trade_cost"
    assert market_impact["volume_authority"] == "qlib.backtest.exchange.Exchange.get_volume"
    assert market_impact["capacity_authority"] == "qlib.backtest.exchange.Exchange._clip_amount_by_volume"
    assert market_impact["configuration_parameter"] == "impact_cost"
    assert market_impact["volume_field"] == "$volume"
    assert (
        market_impact["impact_cost_ratio_rule"]
        == "impact_cost_times_post_volume_clip_trade_value_over_total_trade_value_squared"
    )
    assert market_impact["missing_volume_rule"] == "missing_zero_or_nan_total_trade_value_uses_raw_impact_cost_ratio"
    assert (
        market_impact["final_cost_rule"] == "trade_cost_is_recomputed_after_final_deal_amount_with_adjusted_cost_ratio"
    )
    assert market_impact["numeric_value_exposure"] == "runtime_handoff_only_not_prompt_projection"
    assert (
        "total_trade_val = cast(float, self.get_volume(order.stock_id, order.start_time, order.end_time)) * trade_price"
        in exchange_source
    )
    assert "self._clip_amount_by_volume(order, dealt_order_amount)" in exchange_source
    assert "adj_cost_ratio = self.impact_cost" in exchange_source
    assert "adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2" in exchange_source
    assert "cost_ratio = self.close_cost + adj_cost_ratio" in exchange_source
    assert "cost_ratio = self.open_cost + adj_cost_ratio" in exchange_source
    assert "impact_cost=adj_cost_ratio" in exchange_source
    assert "trade_cost = self._calculate_trade_cost(order.direction, trade_val, cost_ratio, adj_cost_ratio)" in (
        exchange_source
    )


def test_ashare_account_update_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    account_update = contract["prompt_projection_payload"]["account_update_semantics"]
    account_source = ACCOUNT_PATH.read_text()
    exchange_source = EXCHANGE_PATH.read_text()
    position_source = POSITION_PATH.read_text()

    assert account_update["semantic_name"] == "a_share_account_position_cash_mutation"
    assert account_update["execution_authority"] == "qlib.backtest.exchange.Exchange.deal_order"
    assert account_update["account_update_authority"] == "qlib.backtest.account.Account.update_order"
    assert account_update["account_metrics_authority"] == "qlib.backtest.account.Account._update_state_from_order"
    assert account_update["position_update_authority"] == "qlib.backtest.position.Position.update_order"
    assert account_update["ashare_sellable_update_authority"] == "qlib.backtest.position.AsharePosition._sell_stock"
    assert (
        account_update["trade_update_trigger"]
        == "only_trade_value_greater_than_one_e_minus_five_mutates_account_or_position"
    )
    assert account_update["failed_or_zero_trade_rule"] == (
        "failed_order_or_zero_trade_value_does_not_update_position_or_account"
    )
    assert account_update["buy_mutation_order"] == "position_updates_before_account_metrics"
    assert account_update["sell_mutation_order"] == "account_metrics_update_before_position_update"
    assert account_update["buy_cash_rule"] == "buy_subtracts_trade_value_plus_cost_from_cash"
    assert account_update["sell_cash_rule"] == "sell_routes_trade_value_minus_cost_to_cash_or_cash_delay_by_settle_type"
    assert (
        account_update["sellable_amount_rule"]
        == "ashare_sells_reduce_sellable_amount_and_day_bar_count_refresh_releases_total_amount"
    )
    assert "if trade_val > 1e-5:" in exchange_source
    assert "trade_account.update_order(" in exchange_source
    assert "position.update_order(" in exchange_source
    assert "if self.current_position.skip_update():" in account_source
    assert "if order.direction == Order.SELL:" in account_source
    assert "self._update_state_from_order(order, trade_val, cost, trade_price)" in account_source
    assert "self.current_position.update_order(order, trade_val, cost, trade_price)" in account_source
    assert "trade_amount = trade_val / trade_price" in position_source
    assert 'self.position[stock_id]["amount"] += trade_amount' in position_source
    assert 'self.position["cash"] -= trade_val + cost' in position_source
    assert "new_cash = trade_val - cost" in position_source
    assert 'self.position["cash_delay"] += new_cash' in position_source
    assert 'self.position["cash"] += new_cash' in position_source
    assert "self.position[stock_id][self.SELLABLE_AMOUNT_FIELD] = max(" in position_source
    assert 'self.position[stock_id][self.SELLABLE_AMOUNT_FIELD] = self.position[stock_id]["amount"]' in (
        position_source
    )


def test_ashare_account_valuation_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    valuation = contract["prompt_projection_payload"]["account_valuation_semantics"]
    account_source = ACCOUNT_PATH.read_text()
    position_source = POSITION_PATH.read_text()
    exchange_source = EXCHANGE_PATH.read_text()

    assert valuation["semantic_name"] == "a_share_account_bar_end_valuation"
    assert valuation["bar_end_authority"] == "qlib.backtest.account.Account.update_bar_end"
    assert valuation["position_refresh_authority"] == "qlib.backtest.account.Account.update_current_position"
    assert valuation["portfolio_metrics_authority"] == "qlib.backtest.account.Account.update_portfolio_metrics"
    assert valuation["history_position_authority"] == "qlib.backtest.account.Account.update_hist_positions"
    assert valuation["price_update_authority"] == "qlib.backtest.position.Position.update_stock_price"
    assert valuation["value_authority"] == "qlib.backtest.position.Position.calculate_value"
    assert valuation["stock_value_authority"] == "qlib.backtest.position.Position.calculate_stock_value"
    assert valuation["holding_count_authority"] == "qlib.backtest.position.Position.add_count_all"
    assert valuation["ashare_sellable_release_authority"] == "qlib.backtest.position.AsharePosition.add_count_all"
    assert valuation["close_price_authority"] == "qlib.backtest.exchange.Exchange.get_close"
    assert valuation["bar_end_sequence"] == [
        "refresh_current_position_prices_and_holding_counts",
        "update_portfolio_metrics_when_enabled",
        "snapshot_history_positions_when_enabled",
        "update_trade_indicators",
    ]
    assert valuation["mark_price_rule"] == "non_suspended_positions_mark_to_bar_close_at_bar_end"
    assert valuation["suspension_price_rule"] == "suspended_positions_keep_previous_price_during_bar_end_refresh"
    assert valuation["account_value_rule"] == "account_value_equals_stock_value_plus_cash_plus_cash_delay"
    assert valuation["stock_value_rule"] == "stock_value_equals_position_amount_times_current_position_price"
    assert (
        valuation["portfolio_return_rule"]
        == "return_rate_uses_account_earning_plus_current_cost_over_last_account_value"
    )
    assert (
        valuation["history_snapshot_rule"]
        == "history_positions_store_deepcopy_after_now_account_value_and_weights_refresh"
    )
    assert valuation["holding_count_rule"] == "bar_end_refresh_increments_position_count_for_account_frequency"
    assert (
        valuation["daily_sellable_release_rule"]
        == "ashare_day_bar_count_refresh_releases_total_amount_to_sellable_amount"
    )
    assert (
        valuation["infinite_position_rule"] == "skip_update_position_does_not_refresh_prices_counts_metrics_or_history"
    )
    assert valuation["rdagent_rule"] == "describe_only_do_not_redefine_account_valuation_or_bar_end_refresh"
    assert "def update_bar_end(" in account_source
    assert "self.update_current_position(trade_start_time, trade_end_time, trade_exchange)" in account_source
    assert "self.update_portfolio_metrics(trade_start_time, trade_end_time)" in account_source
    assert "self.update_hist_positions(trade_start_time)" in account_source
    assert "if self.current_position.skip_update():" in account_source
    assert "if trade_exchange.check_stock_suspended(code, trade_start_time, trade_end_time):" in account_source
    assert "bar_close = cast(float, trade_exchange.get_close(code, trade_start_time, trade_end_time))" in account_source
    assert "self.current_position.update_stock_price(stock_id=code, price=bar_close)" in account_source
    assert "self.current_position.add_count_all(bar=self.freq)" in account_source
    assert "now_account_value = self.current_position.calculate_value()" in account_source
    assert "now_stock_value = self.current_position.calculate_stock_value()" in account_source
    assert "return_rate=(now_earning + now_cost) / last_account_value" in account_source
    assert 'self.current_position.position["now_account_value"] = now_account_value' in account_source
    assert "self.current_position.update_weight_all()" in account_source
    assert "copy.deepcopy(self.current_position)" in account_source
    assert 'self.position[stock_id]["price"] = price' in position_source
    assert 'value += self.position[stock_id]["amount"] * self.position[stock_id]["price"]' in position_source
    assert 'value += self.position["cash"] + self.position.get("cash_delay", 0.0)' in position_source
    assert 'self.position[code][f"count_{bar}"] += 1' in position_source
    assert 'self.position[stock_id][self.SELLABLE_AMOUNT_FIELD] = self.position[stock_id]["amount"]' in (
        position_source
    )
    assert "def get_close(" in exchange_source


def test_ashare_cash_settlement_contract_matches_position_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    cash_settlement = contract["prompt_projection_payload"]["cash_settlement_semantics"]
    position_source = POSITION_PATH.read_text()

    assert cash_settlement["semantic_name"] == "a_share_sell_proceeds_cash_settlement"
    assert cash_settlement["settlement_authority"] == "qlib.backtest.position.Position"
    assert cash_settlement["settle_start_authority"] == "qlib.backtest.position.Position.settle_start"
    assert cash_settlement["settle_commit_authority"] == "qlib.backtest.position.Position.settle_commit"
    assert cash_settlement["available_cash_authority"] == "qlib.backtest.position.Position.get_cash"
    assert cash_settlement["delayed_cash_state_field"] == "cash_delay"
    assert cash_settlement["delayed_cash_mode"] == "Position.ST_CASH"
    assert cash_settlement["no_delay_cash_mode"] == "Position.ST_NO"
    assert cash_settlement["sell_proceeds_rule"] == "sell_proceeds_enter_cash_delay_when_settle_type_is_cash"
    assert cash_settlement["default_sell_proceeds_rule"] == (
        "sell_proceeds_enter_cash_immediately_when_settle_type_is_none"
    )
    assert cash_settlement["available_cash_rule"] == "get_cash_excludes_cash_delay_unless_include_settle_is_true"
    assert cash_settlement["account_value_rule"] == "calculate_value_includes_cash_delay"
    assert cash_settlement["commit_rule"] == "settle_commit_moves_cash_delay_into_cash_and_clears_delay_state"
    assert 'ST_CASH = "cash"' in position_source
    assert 'ST_NO = "None"' in position_source
    assert 'self.position["cash_delay"] += new_cash' in position_source
    assert 'self.position["cash"] += new_cash' in position_source
    assert "def get_cash(self, include_settle: bool = False)" in position_source
    assert 'cash += self.position.get("cash_delay", 0.0)' in position_source
    assert 'value += self.position["cash"] + self.position.get("cash_delay", 0.0)' in position_source
    assert 'self.position["cash"] += self.position["cash_delay"]' in position_source
    assert 'del self.position["cash_delay"]' in position_source


def test_ashare_liquidity_capacity_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    capacity_semantics = contract["prompt_projection_payload"]["liquidity_capacity_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert capacity_semantics["volume_field"] == "$volume"
    assert capacity_semantics["capacity_parameter"] == "volume_threshold"
    assert capacity_semantics["runtime_authority"] == "qlib.backtest.exchange.Exchange._clip_amount_by_volume"
    assert capacity_semantics["threshold_parser_authority"] == "qlib.backtest.exchange.Exchange._get_vol_limit"
    assert "self._clip_amount_by_volume(order, dealt_order_amount)" in exchange_source
    assert "vol_limit_min = min(vol_limit_num)" in exchange_source
    assert "order.deal_amount = max(min(vol_limit_min, orig_deal_amount), 0)" in exchange_source
    assert "limit_value - dealt_order_amount[order.stock_id]" in exchange_source
    assert "self.volume_threshold = volume_threshold" in exchange_source


def test_ashare_trading_calendar_contract_matches_qlib_calendar_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    calendar_semantics = contract["prompt_projection_payload"]["trading_calendar_semantics"]
    data_source = DATA_PATH.read_text()
    exchange_source = EXCHANGE_PATH.read_text()

    assert calendar_semantics["calendar_frequency"] == "day"
    assert calendar_semantics["calendar_provider_authority"] == "qlib.data.data.CalendarProvider.calendar"
    assert calendar_semantics["calendar_locator_authority"] == "qlib.data.data.CalendarProvider.locate_index"
    assert calendar_semantics["exchange_frequency_parameter"] == "freq"
    assert calendar_semantics["exchange_default_frequency"] == "day"
    assert 'def calendar(self, start_time=None, end_time=None, freq="day", future=False)' in data_source
    assert "def locate_index(" in data_source
    assert "calendar[bisect.bisect_left(calendar, start_time)]" in data_source
    assert "calendar[bisect.bisect_right(calendar, end_time) - 1]" in data_source
    assert "cal = Cal.calendar(freq=freq)" in data_source
    assert 'freq: str = "day"' in exchange_source
    assert "self.freq = freq" in exchange_source


def test_ashare_universe_membership_contract_matches_qlib_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    universe = contract["prompt_projection_payload"]["universe_membership_semantics"]
    data_source = DATA_PATH.read_text()
    exchange_source = EXCHANGE_PATH.read_text()

    assert universe["semantic_name"] == "a_share_universe_membership"
    assert universe["instrument_provider_authority"] == "qlib.data.data.InstrumentProvider.list_instruments"
    assert universe["local_provider_authority"] == "qlib.data.data.LocalInstrumentProvider.list_instruments"
    assert universe["exchange_codes_authority"] == "qlib.backtest.exchange.Exchange.__init__"
    assert universe["market_universe_rule"] == "string_codes_are_resolved_by_qlib_D_instruments"
    assert universe["membership_window_rule"] == "instrument_start_end_spans_are_clipped_to_requested_calendar_window"
    assert universe["calendar_boundary_rule"] == (
        "start_end_defaults_and_membership_filtering_use_qlib_calendar_boundaries"
    )
    assert universe["filter_pipe_rule"] == "qlib_instrument_filter_pipe_is_applied_after_calendar_window_clipping"
    assert universe["survivorship_rule"] == "membership_must_remain_point_in_time_by_qlib_instrument_spans_and_filters"
    assert (
        'def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False)'
        in data_source
    )
    assert 'market = instruments["market"]' in data_source
    assert "cal = Cal.calendar(freq=freq)" in data_source
    assert "start_time = pd.Timestamp(start_time or cal[0])" in data_source
    assert "end_time = pd.Timestamp(end_time or cal[-1])" in data_source
    assert "lambda x: x[0] <= x[1]" in data_source
    assert 'filter_pipe = instruments["filter_pipe"]' in data_source
    assert "if as_list:" in data_source
    assert "if isinstance(codes, str):" in exchange_source
    assert "codes = D.instruments(codes)" in exchange_source


def test_rdagent_ashare_contract_is_machine_readable_json() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract(
        strict_price_limit=False,
    )

    round_tripped = json.loads(json.dumps(contract, sort_keys=True))

    assert round_tripped["runtime_surfaces"]["exchange_kwargs"]["ashare_price_limit_mode"] == "auto"
    assert round_tripped["market_semantics"]["cost_model"]["close_tax"] == pytest.approx(0.001)
    assert round_tripped["failure_semantics"]["malformed_contract"] == "fail_closed"
    assert round_tripped["prompt_projection_payload"]["projection_id"] == "qlib_joinquant_ashare_prompt_projection_v1"
    assert (
        round_tripped["prompt_projection_payload"]["instrument_identity_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_instrument_or_board_identity"
    )
    assert (
        round_tripped["prompt_projection_payload"]["universe_membership_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_universe_membership_or_filters"
    )
    assert (
        round_tripped["prompt_projection_payload"]["trading_calendar_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_trading_calendar_or_data_frequency"
    )
    assert (
        round_tripped["prompt_projection_payload"]["transaction_cost_semantics"]["numeric_values_exposure"]
        == "runtime_handoff_only_not_prompt_projection"
    )
    assert (
        round_tripped["prompt_projection_payload"]["market_impact_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_market_impact_or_cost_ratio"
    )
    assert (
        round_tripped["prompt_projection_payload"]["market_impact_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    )
    assert (
        round_tripped["prompt_projection_payload"]["account_update_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_account_position_or_cash_mutation_order"
    )
    assert (
        round_tripped["prompt_projection_payload"]["account_update_semantics"]["account_update_authority"]
        == "qlib.backtest.account.Account.update_order"
    )
    assert (
        round_tripped["prompt_projection_payload"]["account_valuation_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_account_valuation_or_bar_end_refresh"
    )
    assert (
        round_tripped["prompt_projection_payload"]["account_valuation_semantics"]["value_authority"]
        == "qlib.backtest.position.Position.calculate_value"
    )
    assert (
        round_tripped["prompt_projection_payload"]["suspension_tradability_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_suspension_or_tradability"
    )
    assert (
        round_tripped["prompt_projection_payload"]["execution_price_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_execution_price_or_frequency"
    )
    assert (
        round_tripped["prompt_projection_payload"]["price_adjustment_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_price_adjustment_or_order_factor"
    )
    assert round_tripped["prompt_projection_payload"]["price_limit_semantics"]["price_limit_mode"] == "auto"
    assert (
        round_tripped["prompt_projection_payload"]["price_limit_semantics"]["fallback_authority_rule"]
        == "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority"
    )
    assert (
        round_tripped["prompt_projection_payload"]["order_tradability_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_order_tradability_or_limit_checks"
    )
    assert (
        round_tripped["prompt_projection_payload"]["order_tradability_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange.check_order"
    )
    assert (
        round_tripped["prompt_projection_payload"]["order_fill_amount_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_order_fill_amount_or_clip_sequence"
    )
    assert (
        round_tripped["prompt_projection_payload"]["order_fill_amount_semantics"]["runtime_authority"]
        == "qlib.backtest.exchange.Exchange._calc_trade_info_by_order"
    )
    assert round_tripped["prompt_projection_payload"]["settlement_semantics"]["settlement_rule"] == "t_plus_1_stock"
    assert (
        round_tripped["prompt_projection_payload"]["settlement_semantics"]["sellable_state_field"] == "sellable_amount"
    )
    assert (
        round_tripped["prompt_projection_payload"]["cash_constraint_semantics"]["shorting_policy"]
        == "equity_short_selling_is_not_enabled"
    )
    assert (
        round_tripped["prompt_projection_payload"]["cash_settlement_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_cash_settlement_or_sell_proceeds_availability"
    )
    assert (
        round_tripped["prompt_projection_payload"]["liquidity_capacity_semantics"]["capacity_parameter"]
        == "volume_threshold"
    )
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
