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
BACKTEST_INIT_PATH = REPO_ROOT / "qlib/backtest/__init__.py"
ACCOUNT_PATH = REPO_ROOT / "qlib/backtest/account.py"
DECISION_PATH = REPO_ROOT / "qlib/backtest/decision.py"
EXCHANGE_PATH = REPO_ROOT / "qlib/backtest/exchange.py"
EXECUTOR_PATH = REPO_ROOT / "qlib/backtest/executor.py"
POSITION_PATH = REPO_ROOT / "qlib/backtest/position.py"
REPORT_PATH = REPO_ROOT / "qlib/backtest/report.py"
SIGNAL_PATH = REPO_ROOT / "qlib/backtest/signal.py"
UTILS_PATH = REPO_ROOT / "qlib/backtest/utils.py"
ALPHA_PATH = REPO_ROOT / "qlib/contrib/eva/alpha.py"
EVALUATE_PATH = REPO_ROOT / "qlib/contrib/evaluate.py"
ANALYSIS_POSITION_REPORT_PATH = REPO_ROOT / "qlib/contrib/report/analysis_position/report.py"
ANALYSIS_POSITION_RISK_PATH = REPO_ROOT / "qlib/contrib/report/analysis_position/risk_analysis.py"
HANDLER_PATH = REPO_ROOT / "qlib/contrib/data/handler.py"
ONLINE_OPERATOR_PATH = REPO_ROOT / "qlib/contrib/online/operator.py"
ONLINE_USER_PATH = REPO_ROOT / "qlib/contrib/online/user.py"
ORDER_GENERATOR_PATH = REPO_ROOT / "qlib/contrib/strategy/order_generator.py"
SIGNAL_STRATEGY_PATH = REPO_ROOT / "qlib/contrib/strategy/signal_strategy.py"
DATA_PATH = REPO_ROOT / "qlib/data/data.py"
STRATEGY_BASE_PATH = REPO_ROOT / "qlib/strategy/base.py"
RECORD_TEMP_PATH = REPO_ROOT / "qlib/workflow/record_temp.py"
CONFIG_PATH = REPO_ROOT / "qlib/tests/config.py"

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
            "suspension/tradability, price-limit, order-tradability, trade-window tradability, order-fill, account-position update, account valuation, trade indicator/execution-quality, executor/trade-decision lifecycle, strategy signal-to-order generation, supervised label, prediction signal, signal IC, portfolio risk analysis, benchmark-relative excess return, feedback metric consumption, benchmark return, universe/benchmark binding, runtime handoff template binding, research data-source, settlement, cash-settlement, cash/shorting, liquidity/capacity, market-impact, or cost semantics."
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
        "redefine_trade_execution_indicators_or_quality_metrics"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_executor_decision_lifecycle_or_nested_execution_order"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_strategy_signal_to_order_generation" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "redefine_supervised_label_expression_or_horizon" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_prediction_signal_score_or_return_realization"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert "redefine_signal_ic_or_rank_ic_metrics" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert "redefine_portfolio_risk_analysis_metrics" in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    assert (
        "redefine_benchmark_relative_excess_return_or_cost_treatment"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_feedback_metric_paths_or_label_derived_utility_as_qlib_metric"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_benchmark_return_series_or_default_benchmark"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_universe_benchmark_template_binding_or_cross_alias_market_and_benchmark"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_runtime_handoff_or_template_execution_kwargs"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_research_data_source_availability_or_imply_unregistered_sources"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
    assert (
        "redefine_trade_window_tradability_or_quote_window_aggregation"
        in contract["semantic_boundary"]["rdagent_forbidden_actions"]
    )
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
    assert "trade_indicator_semantics" in contract["rdagent_must_not_redefine"]
    assert "executor_decision_semantics" in contract["rdagent_must_not_redefine"]
    assert "strategy_order_semantics" in contract["rdagent_must_not_redefine"]
    assert "supervised_label_semantics" in contract["rdagent_must_not_redefine"]
    assert "prediction_signal_semantics" in contract["rdagent_must_not_redefine"]
    assert "signal_ic_semantics" in contract["rdagent_must_not_redefine"]
    assert "portfolio_risk_semantics" in contract["rdagent_must_not_redefine"]
    assert "excess_return_semantics" in contract["rdagent_must_not_redefine"]
    assert "feedback_metric_semantics" in contract["rdagent_must_not_redefine"]
    assert "benchmark_return_semantics" in contract["rdagent_must_not_redefine"]
    assert "universe_benchmark_binding_semantics" in contract["rdagent_must_not_redefine"]
    assert "runtime_handoff_template_binding_semantics" in contract["rdagent_must_not_redefine"]
    assert "research_data_source_semantics" in contract["rdagent_must_not_redefine"]
    assert "trade_window_tradability_semantics" in contract["rdagent_must_not_redefine"]
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
    assert "trade_indicator_semantics" in evidence["fingerprint_scope"]
    assert "executor_decision_semantics" in evidence["fingerprint_scope"]
    assert "strategy_order_semantics" in evidence["fingerprint_scope"]
    assert "supervised_label_semantics" in evidence["fingerprint_scope"]
    assert "prediction_signal_semantics" in evidence["fingerprint_scope"]
    assert "signal_ic_semantics" in evidence["fingerprint_scope"]
    assert "portfolio_risk_semantics" in evidence["fingerprint_scope"]
    assert "excess_return_semantics" in evidence["fingerprint_scope"]
    assert "feedback_metric_semantics" in evidence["fingerprint_scope"]
    assert "benchmark_return_semantics" in evidence["fingerprint_scope"]
    assert "universe_benchmark_binding_semantics" in evidence["fingerprint_scope"]
    assert "runtime_handoff_template_binding_semantics" in evidence["fingerprint_scope"]
    assert "research_data_source_semantics" in evidence["fingerprint_scope"]
    assert "trade_window_tradability_semantics" in evidence["fingerprint_scope"]
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
            {
                "match": "BJ*|SH8*|SH4*|SH9*|SZ8*|SZ4*|SZ9*",
                "board": "beijing_stock_exchange",
            },
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
    assert prompt_payload["trade_indicator_semantics"] == {
        "semantic_name": "a_share_trade_execution_indicator",
        "account_indicator_authority": "qlib.backtest.account.Account.update_indicator",
        "indicator_authority": "qlib.backtest.report.Indicator",
        "atomic_order_update_authority": "qlib.backtest.report.Indicator.update_order_indicators",
        "nested_order_aggregation_authority": "qlib.backtest.report.Indicator.agg_order_indicators",
        "trade_indicator_authority": "qlib.backtest.report.Indicator.cal_trade_indicators",
        "record_authority": "qlib.backtest.report.Indicator.record",
        "order_indicator_state": "Indicator.order_indicator",
        "trade_indicator_state": "Indicator.trade_indicator",
        "history_state": [
            "Indicator.order_indicator_his",
            "Indicator.trade_indicator_his",
        ],
        "order_metric_fields": [
            "amount",
            "inner_amount",
            "deal_amount",
            "trade_price",
            "trade_value",
            "trade_cost",
            "trade_dir",
            "pa",
            "ffr",
            "base_price",
            "base_volume",
        ],
        "trade_metric_fields": ["ffr", "pa", "pos", "deal_amount", "value", "count"],
        "bar_end_rule": "account_update_indicator_runs_after_current_position_valuation_and_portfolio_metrics",
        "atomic_rule": "atomic_executor_uses_trade_info_to_update_order_indicators",
        "nested_rule": "non_atomic_executor_aggregates_inner_order_indicators_and_outer_decision",
        "fulfill_rate_rule": "ffr_equals_deal_amount_reindexed_zero_for_missing_over_order_amount",
        "price_advantage_rule": "pa_equals_directional_trade_price_over_base_price_minus_one",
        "positive_rate_rule": "pos_equals_fraction_of_positive_pa",
        "deal_amount_metric_rule": "deal_amount_metric_sums_absolute_deal_amount",
        "trade_value_metric_rule": "value_metric_sums_absolute_trade_value",
        "order_count_rule": "count_metric_counts_order_amount_entries",
        "weighting_rule": "ffr_and_pa_support_mean_amount_weighted_value_weighted",
        "base_price_rule": "base_price_uses_exchange_deal_price_with_twap_or_vwap_aggregation",
        "unsupported_base_price_rule": "non_deal_price_base_price_is_not_supported",
        "record_rule": "bar_end_records_order_indicator_and_trade_indicator_by_trade_start_time",
        "portfolio_boundary_rule": "trade_indicators_are_execution_quality_metrics_not_portfolio_return_metrics",
        "rdagent_rule": "describe_only_do_not_redefine_trade_execution_indicators_or_quality_metrics",
    }
    assert prompt_payload["executor_decision_semantics"] == {
        "semantic_name": "a_share_executor_trade_decision_lifecycle",
        "base_executor_authority": "qlib.backtest.executor.BaseExecutor.collect_data",
        "simulator_executor_authority": "qlib.backtest.executor.SimulatorExecutor._collect_data",
        "nested_executor_authority": "qlib.backtest.executor.NestedExecutor._collect_data",
        "decision_authority": "qlib.backtest.decision.BaseTradeDecision",
        "decision_update_authority": "qlib.backtest.decision.BaseTradeDecision.update",
        "range_limit_authority": "qlib.backtest.decision.BaseTradeDecision.get_range_limit",
        "data_calendar_range_authority": "qlib.backtest.decision.BaseTradeDecision.get_data_cal_range_limit",
        "inner_decision_modification_authority": "qlib.backtest.decision.BaseTradeDecision.mod_inner_decision",
        "calendar_authority": "qlib.backtest.utils.TradeCalendarManager",
        "level_infra_authority": "qlib.backtest.utils.LevelInfrastructure",
        "atomicity_rule": "atomic_executor_rejects_trade_decision_range_limit",
        "settle_sequence_rule": "settle_start_runs_before_collection_and_settle_commit_after_bar_end_when_enabled",
        "bar_end_sequence_rule": "executor_updates_account_bar_end_before_trade_calendar_step",
        "track_data_rule": "track_data_yields_outer_trade_decision_for_training_data_only",
        "simulator_order_rule": "simulator_executor_retrieves_order_decisions_and_deals_each_order_through_exchange",
        "simulator_trade_type_rule": (
            "serial_preserves_order_sequence_parallel_sorts_buys_before_sells_to_surface_cash_conflicts"
        ),
        "daily_dealt_amount_rule": "simulator_resets_dealt_order_amount_when_trade_day_advances",
        "nested_init_rule": (
            "nested_executor_resets_inner_executor_to_outer_step_window_and_passes_outer_decision_to_inner_strategy"
        ),
        "nested_update_rule": "nested_executor_updates_outer_decision_with_inner_calendar_before_range_limit_alignment",
        "nested_range_rule": "nested_executor_skips_inner_steps_outside_range_limit_when_align_range_limit_is_enabled",
        "inner_decision_rule": (
            "outer_trade_decision_may_propagate_trade_range_into_inner_trade_decision_only_when_inner_range_absent"
        ),
        "empty_decision_rule": "empty_decision_can_skip_inner_loop_when_skip_empty_decision_is_enabled",
        "inner_result_rule": "nested_executor_collects_inner_execute_results_order_indicators_and_decision_time_windows",
        "rdagent_rule": "describe_only_do_not_redefine_executor_decision_lifecycle_or_nested_execution_order",
    }
    assert prompt_payload["strategy_order_semantics"] == {
        "semantic_name": "a_share_strategy_signal_to_order_generation",
        "base_strategy_authority": "qlib.strategy.base.BaseStrategy.generate_trade_decision",
        "topk_strategy_authority": "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision",
        "weight_strategy_authority": "qlib.contrib.strategy.signal_strategy.WeightStrategyBase.generate_trade_decision",
        "order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenerator.generate_order_list_from_target_weight_position"
        ),
        "interacting_order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenWInteract.generate_order_list_from_target_weight_position"
        ),
        "non_interacting_order_generator_authority": (
            "qlib.contrib.strategy.order_generator.OrderGenWOInteract.generate_order_list_from_target_weight_position"
        ),
        "target_amount_order_authority": "qlib.backtest.exchange.Exchange.generate_order_for_target_amount_position",
        "trade_decision_type": "qlib.backtest.decision.TradeDecisionWO",
        "signal_authority": "qlib.backtest.signal.Signal.get_signal",
        "template_strategy_binding": "qlib.contrib.strategy.TopkDropoutStrategy",
        "prediction_window_rule": "strategy_reads_signal_from_previous_calendar_step_shift_one",
        "dataframe_signal_rule": "topk_dropout_uses_first_signal_column_when_prediction_is_dataframe",
        "missing_signal_rule": "missing_signal_returns_empty_TradeDecisionWO",
        "topk_selection_rule": "topk_dropout_ranks_current_holdings_and_new_candidates_by_pred_score_descending",
        "dropout_rule": "combined_last_and_today_scores_prevent_dropping_higher_score_stock_for_lower_score_buy",
        "sell_order_rule": "sell_orders_are_generated_before_buy_orders_and_simulated_on_temp_position_for_cash",
        "buy_budget_rule": "buy_budget_equals_temp_cash_times_risk_degree_divided_by_buy_count",
        "hold_threshold_rule": "sell_requires_current_holding_count_at_least_hold_thresh",
        "only_tradable_rule": "only_tradable_filters_selection_candidates_by_exchange_tradability",
        "forbid_all_trade_at_limit_rule": (
            "forbid_all_trade_at_limit_checks_any_limit_direction_else_directional_limit"
        ),
        "buy_round_lot_rule": "buy_amount_uses_deal_price_factor_and_exchange_round_amount_by_trade_unit",
        "weight_strategy_rule": "weight_strategy_delegates_target_weight_to_order_generator_after_signal_lookup",
        "interacting_generator_rule": "interacting_order_generator_uses_trade_date_tradability_and_prices",
        "non_interacting_generator_rule": (
            "non_interacting_order_generator_uses_pred_date_close_or_current_holding_price"
        ),
        "target_order_rule": "exchange_generates_target_amount_orders_with_deterministic_shuffled_stock_order",
        "target_order_return_rule": "exchange_returns_sell_orders_before_buy_orders",
        "rdagent_rule": "describe_only_do_not_redefine_strategy_signal_to_order_generation",
    }
    assert prompt_payload["signal_ic_semantics"] == {
        "semantic_name": "a_share_signal_information_coefficient",
        "signal_record_authority": "qlib.workflow.record_temp.SignalRecord",
        "signal_analysis_authority": "qlib.workflow.record_temp.SigAnaRecord",
        "high_frequency_signal_analysis_authority": "qlib.workflow.record_temp.HFSignalRecord",
        "ic_calculation_authority": "qlib.contrib.eva.alpha.calc_ic",
        "prediction_artifact": "pred.pkl",
        "label_artifact": "label.pkl",
        "ic_artifact": "ic.pkl",
        "rank_ic_artifact": "ric.pkl",
        "prediction_column_rule": "series_prediction_is_converted_to_score_dataframe_else_first_prediction_column_is_used",
        "label_source_rule": "dataset_prepare_test_label_uses_DataHandlerLP_DK_R_when_supported_else_handler_default",
        "missing_label_rule": "missing_or_empty_label_skips_signal_analysis_generation",
        "label_column_rule": "SigAnaRecord_uses_configured_label_col_default_zero",
        "groupby_level": "datetime",
        "ic_rule": "IC_is_per_datetime_pearson_correlation_between_pred_and_label",
        "rank_ic_rule": "Rank_IC_is_per_datetime_spearman_correlation_between_pred_and_label",
        "dropna_rule": "calc_ic_preserves_nan_by_default_and_drops_nan_only_when_dropna_true",
        "metric_fields": ["IC", "ICIR", "Rank IC", "Rank ICIR"],
        "metric_aggregation_rule": "IC_and_Rank_IC_metrics_are_series_means",
        "icir_rule": "ICIR_is_IC_mean_divided_by_IC_sample_std",
        "rank_icir_rule": "Rank_ICIR_is_Rank_IC_mean_divided_by_Rank_IC_sample_std",
        "recorder_metric_rule": "SigAnaRecord_and_HFSignalRecord_log_metrics_with_exact_metric_names",
        "rdagent_consumed_metric_paths": ["IC", "ICIR", "Rank IC", "Rank ICIR"],
        "portfolio_boundary_rule": "signal_ic_metrics_are_prediction_label_quality_metrics_not_portfolio_return_metrics",
        "rdagent_rule": "describe_only_do_not_redefine_signal_ic_or_rank_ic_metrics",
    }
    assert prompt_payload["supervised_label_semantics"] == {
        "semantic_name": "a_share_supervised_forward_return_label",
        "handler_authority": "qlib.contrib.data.handler.Alpha158",
        "handler360_authority": "qlib.contrib.data.handler.Alpha360",
        "loader_authority": "qlib.contrib.data.loader.Alpha158DL",
        "processor_authority": "qlib.data.dataset.processor.DropnaLabel",
        "label_group": "label",
        "label_column": "LABEL0",
        "label_expression": "Ref($close, -2)/Ref($close, -1) - 1",
        "label_expression_source": "Alpha158.get_label_config_and_Alpha360.get_label_config",
        "label_horizon_rule": "label_at_datetime_t_is_close_t_plus_2_over_close_t_plus_1_minus_one",
        "prediction_execution_alignment_rule": (
            "label_horizon_matches_strategy_previous_step_signal_execution_without_same_day_fill_assumption"
        ),
        "dropna_processor_rule": "DropnaLabel_removes_missing_LABEL0_rows_before_training_or_evaluation",
        "template_binding_rule": "rdagent_templates_must_use_LABEL0_and_the_qlib_owned_label_expression",
        "prompt_wording_rule": (
            "describe_as_qlib_contract_defined_forward_return_label_not_undefined_next_several_days_return"
        ),
        "rdagent_template_paths": [
            "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
        ],
        "rdagent_prompt_paths": ["rdagent/scenarios/qlib/experiment/prompts.yaml"],
        "rdagent_rule": "describe_only_do_not_redefine_supervised_label_expression_or_horizon",
    }
    assert prompt_payload["prediction_signal_semantics"] == {
        "semantic_name": "a_share_prediction_signal_score",
        "model_signal_authority": "qlib.backtest.signal.ModelSignal",
        "signal_cache_authority": "qlib.backtest.signal.SignalWCache",
        "signal_interface_authority": "qlib.backtest.signal.Signal.get_signal",
        "signal_record_authority": "qlib.workflow.record_temp.SignalRecord",
        "strategy_consumption_authority": (
            "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision"
        ),
        "prediction_artifact": "pred.pkl",
        "prediction_column": "score",
        "model_predict_rule": "model_predict_output_is_prediction_score_not_realized_or_executable_return",
        "series_prediction_rule": "series_prediction_is_saved_as_score_column",
        "dataframe_prediction_rule": "first_prediction_column_is_used_when_prediction_is_dataframe",
        "resample_rule": "SignalWCache_uses_last_signal_between_decision_start_and_end",
        "strategy_ranking_rule": "TopkDropoutStrategy_sorts_prediction_scores_descending_for_candidate_selection",
        "missing_signal_rule": "missing_signal_returns_empty_TradeDecisionWO",
        "label_alignment_rule": "prediction_score_is_trained_against_qlib_owned_LABEL0_without_redefining_return_horizon",
        "prompt_wording_rule": (
            "describe_as_prediction_signal_score_for_LABEL0_not_realized_future_return_or_guaranteed_portfolio_return"
        ),
        "rdagent_model_output_format_rule": (
            "rdagent_model_experiment_output_format_must_describe_prediction_as_score_column_indexed_by_datetime_and_instrument_not_graph_node_output"
        ),
        "rdagent_model_task_boundary_rule": (
            "rdagent_qlib_model_tasks_must_carry_prediction_signal_score_boundary_to_model_implementation_coder"
        ),
        "rdagent_model_type_boundary_rule": (
            "rdagent_qlib_model_experiment_outputs_must_use_tabular_or_timeseries_model_type_only"
        ),
        "rdagent_model_implementation_prompt_boundary_rule": (
            "rdagent_qlib_model_implementation_prompts_must_treat_model_output_boundary_as_authority_over_generic_model_type_examples"
        ),
        "rdagent_model_evaluator_prompt_boundary_rule": (
            "rdagent_qlib_model_evaluator_prompts_must_reject_model_output_boundary_violations_even_when_execution_or_similar_examples_pass"
        ),
        "rdagent_supported_model_types": ["Tabular", "TimeSeries"],
        "rdagent_forbidden_model_types": ["Graph", "XGBoost"],
        "rdagent_implementation_prompt_paths": [
            "rdagent/components/coder/model_coder/prompts.yaml",
        ],
        "rdagent_prompt_paths": [
            "rdagent/scenarios/qlib/experiment/prompts.yaml",
            "rdagent/scenarios/qlib/prompts.yaml",
        ],
        "rdagent_rule": "describe_only_do_not_redefine_prediction_signal_score_or_return_realization",
    }
    assert prompt_payload["portfolio_risk_semantics"] == {
        "semantic_name": "a_share_portfolio_risk_analysis",
        "record_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "risk_analysis_authority": "qlib.contrib.evaluate.risk_analysis",
        "freq_authority": "qlib.utils.resam.Freq.parse",
        "backtest_source_rule": "PortAnaRecord_runs_normal_backtest_and_reads_portfolio_metric_dict_by_freq",
        "report_artifact_rule": "report_normal_dataframe_saved_as_portfolio_analysis_report_normal_{freq}_pkl",
        "risk_artifact_rule": "risk_analysis_dataframe_saved_as_portfolio_analysis_port_analysis_{freq}_pkl",
        "recorder_metric_rule": "risk_metrics_are_logged_as_{freq}.{report_type}.{risk_metric}",
        "default_frequency_rule": "missing_risk_analysis_freq_uses_first_executor_portfolio_metric_frequency",
        "required_report_columns": ["return", "bench", "cost", "turnover"],
        "turnover_report_metric_rule": (
            "report_turnover_is_post_backtest_portfolio_metric_not_default_factor_input_field"
        ),
        "report_type_fields": ["excess_return_without_cost", "excess_return_with_cost"],
        "excess_without_cost_rule": "report_return_minus_benchmark",
        "excess_with_cost_rule": "report_return_minus_benchmark_minus_cost",
        "risk_metric_fields": [
            "mean",
            "std",
            "annualized_return",
            "information_ratio",
            "max_drawdown",
        ],
        "default_accumulation_mode": "sum",
        "supported_accumulation_modes": ["sum", "product"],
        "sum_mode_rule": "qlib_sum_mode_uses_arithmetic_cumulative_return_not_geometric_compounding",
        "day_annualization_scaler": 238,
        "annualization_scaler_rule": "risk_analysis_parses_freq_when_N_is_absent_and_N_overrides_freq_when_present",
        "mean_rule": "sum_mode_mean_equals_return_series_mean",
        "std_rule": "sum_mode_std_uses_sample_standard_deviation_ddof_one",
        "annualized_return_rule": "sum_mode_annualized_return_equals_mean_times_annualization_scaler",
        "information_ratio_rule": "information_ratio_equals_mean_over_std_times_square_root_annualization_scaler",
        "max_drawdown_rule": "sum_mode_max_drawdown_equals_min_of_cumulative_return_minus_running_cumulative_max",
        "metric_path_format": "{freq}.{report_type}.{risk_metric}",
        "metric_path_frequency": "1day",
        "metric_path_whitespace_rule": "metric_paths_are_exact_without_leading_or_trailing_whitespace",
        "metric_path_report_type_rule": "prompt_context_uses_without_cost_and_feedback_bandit_ui_use_with_cost",
        "rdagent_prompt_metric_paths": [
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.max_drawdown",
        ],
        "rdagent_feedback_metric_paths": [
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "rdagent_bandit_metric_paths": [
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "rdagent_ui_metric_paths": [
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "rdagent_consumed_metric_paths": [
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.max_drawdown",
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "rdagent_rule": "describe_only_do_not_redefine_portfolio_risk_analysis_metrics",
    }
    assert prompt_payload["excess_return_semantics"] == {
        "semantic_name": "a_share_benchmark_relative_excess_return",
        "benchmark_dependency": "benchmark_return_semantics",
        "portfolio_risk_dependency": "portfolio_risk_semantics",
        "report_column_authority": "qlib.backtest.report.PortfolioMetrics",
        "risk_record_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "report_graph_authority": "qlib.contrib.report.analysis_position.report._calculate_report_data",
        "risk_graph_authority": "qlib.contrib.report.analysis_position.risk_analysis._get_risk_analysis_data_with_report",
        "online_analysis_authority": "qlib.contrib.online.operator",
        "user_analysis_authority": "qlib.contrib.online.user",
        "required_report_columns": ["return", "bench", "cost"],
        "without_cost_field": "excess_return_without_cost",
        "with_cost_field": "excess_return_with_cost",
        "without_cost_formula": "return - bench",
        "with_cost_formula": "return - bench - cost",
        "cumulative_without_cost_field": "cum_ex_return_wo_cost",
        "cumulative_with_cost_field": "cum_ex_return_w_cost",
        "cost_source": "reported_cost_column_from_trade_indicator_semantics",
        "benchmark_source": "reported_bench_column_from_benchmark_return_semantics",
        "metric_path_without_cost": "1day.excess_return_without_cost.annualized_return",
        "metric_path_with_cost": "1day.excess_return_with_cost.annualized_return",
        "rdagent_prompt_rule": "generated_research_must_report_benchmark_relative_excess_return_not_raw_return",
        "forbidden_substitutions": [
            "raw_return_as_excess_return",
            "market_universe_as_benchmark_return",
            "with_cost_metric_without_report_cost_column",
            "prompt_defined_cost_or_benchmark_formula",
        ],
        "rdagent_rule": "describe_only_do_not_redefine_benchmark_relative_excess_return",
    }
    assert prompt_payload["feedback_metric_semantics"] == {
        "semantic_name": "a_share_rd_agent_feedback_metric_consumption",
        "signal_metric_authority": "qlib.workflow.record_temp.SigAnaRecord",
        "portfolio_metric_authority": "qlib.workflow.record_temp.PortAnaRecord",
        "risk_metric_authority": "qlib.contrib.evaluate.risk_analysis",
        "prompt_metric_paths": [
            "IC",
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.max_drawdown",
        ],
        "feedback_metric_paths": [
            "IC",
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "bandit_metric_paths": [
            "IC",
            "ICIR",
            "Rank IC",
            "Rank ICIR",
            "1day.excess_return_with_cost.annualized_return",
            "1day.excess_return_with_cost.information_ratio",
            "1day.excess_return_with_cost.max_drawdown",
        ],
        "feedback_primary_metric": "1day.excess_return_with_cost.annualized_return",
        "sota_fallback_rule": "missing_explicit_feedback_decision_uses_feedback_primary_metric_improvement",
        "derived_bandit_utility_name": "drawdown_adjusted_return",
        "derived_bandit_utility_rule": "rdagent_may_compute_arr_over_abs_max_drawdown_as_derived_utility_not_qlib_metric",
        "forbidden_metric_aliases": ["sharpe", "Sharpe"],
        "prompt_metric_wording_rule": "describe_exact_qlib_metric_paths_not_generic_return_sharpe_or_and_so_on",
        "rdagent_source_paths": [
            "rdagent/scenarios/qlib/developer/feedback.py",
            "rdagent/scenarios/qlib/proposal/bandit.py",
            "rdagent/scenarios/qlib/experiment/prompts.yaml",
            "rdagent/scenarios/qlib/prompts.yaml",
            "rdagent/log/ui/app.py",
        ],
        "rdagent_rule": "consume_exact_qlib_metric_paths_and_label_derived_bandit_utility_as_non_qlib_metric",
    }
    assert prompt_payload["benchmark_return_semantics"] == {
        "semantic_name": "a_share_benchmark_return_series",
        "default_benchmark": "SH000300",
        "benchmark_constant_authority": "qlib.tests.config.CSI300_BENCH",
        "backtest_entry_authority": "qlib.backtest.backtest",
        "account_config_authority": "qlib.backtest.create_account_instance",
        "portfolio_metric_authority": "qlib.backtest.report.PortfolioMetrics",
        "benchmark_calculation_authority": "qlib.backtest.report.PortfolioMetrics._cal_benchmark",
        "benchmark_sampling_authority": "qlib.backtest.report.PortfolioMetrics._sample_benchmark",
        "feature_query_authority": "qlib.utils.resam.get_higher_eq_freq_feature",
        "resample_authority": "qlib.utils.resam.resam_ts_data",
        "accepted_benchmark_inputs": ["str", "list", "dict", "pd.Series", "None"],
        "default_rule": "missing_benchmark_key_uses_CSI300_BENCH_SH000300",
        "none_rule": "benchmark_config_none_or_benchmark_none_disables_benchmark_series",
        "series_rule": "pd_series_benchmark_is_used_directly_as_per_period_return_series",
        "code_rule": "str_benchmark_is_queried_as_single_code_close_over_ref_close_minus_one",
        "basket_rule": "list_or_dict_benchmark_is_queried_as_codes_and_averaged_by_datetime",
        "benchmark_field_expression": "$close/Ref($close,1)-1",
        "missing_frequency_rule": "non_series_benchmark_requires_freq_else_ValueError",
        "missing_benchmark_rule": "empty_feature_result_raises_ValueError",
        "fillna_rule": "queried_benchmark_returns_fillna_zero_after_datetime_average",
        "sample_rule": "bar_benchmark_return_equals_product_of_one_plus_period_returns_minus_one",
        "direct_bench_value_rule": "provided_bench_value_overrides_sampling",
        "unusable_benchmark_rule": "trade_end_time_and_bench_value_both_none_raise_ValueError",
        "report_column": "bench",
        "portfolio_risk_dependency": "portfolio_risk_excess_returns_use_report_normal_bench_column",
        "rdagent_rule": "describe_only_do_not_redefine_benchmark_return_series_or_default_benchmark",
    }
    assert prompt_payload["universe_benchmark_binding_semantics"] == {
        "semantic_name": "a_share_rd_agent_universe_benchmark_binding",
        "market_universe_authority": "qlib.tests.config.CSI300_MARKET",
        "benchmark_authority": "qlib.tests.config.CSI300_BENCH",
        "template_market_value": "csi300",
        "template_benchmark_value": "SH000300",
        "template_market_anchor": "market: &market csi300",
        "template_instruments_binding": "instruments: *market",
        "template_benchmark_anchor": "benchmark: &benchmark SH000300",
        "template_backtest_benchmark_binding": "benchmark: *benchmark",
        "market_universe_rule": "csi300_template_market_selects_instruments_only",
        "benchmark_rule": "SH000300_template_benchmark_is_portfolio_excess_return_baseline_only",
        "separation_rule": "market_universe_membership_and_benchmark_return_series_are_not_substitutable",
        "forbidden_template_values": ["all_a", "all", "SH000300_as_market", "csi300_as_benchmark"],
        "rdagent_template_paths": [
            "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
        ],
        "rdagent_rule": "bind_market_to_instruments_and_benchmark_to_backtest_without_cross_aliasing",
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
    assert prompt_payload["trade_window_tradability_semantics"] == {
        "semantic_name": "a_share_trade_window_tradability",
        "runtime_authority": "qlib.backtest.exchange.Exchange.is_stock_tradable",
        "suspension_window_authority": "qlib.backtest.exchange.Exchange.check_stock_suspended",
        "price_limit_window_authority": "qlib.backtest.exchange.Exchange.check_stock_limit",
        "order_gate_authority": "qlib.backtest.exchange.Exchange.check_order",
        "quote_membership_rule": "unknown_stock_id_is_regarded_as_suspended_and_not_tradable",
        "suspension_window_rule": "single_missing_close_or_all_missing_close_window_blocks_trading",
        "non_suspension_window_rule": "any_non_missing_close_in_window_keeps_suspension_gate_open",
        "price_limit_window_rule": "limit_buy_or_limit_sell_blocks_only_when_all_rows_in_window_are_limited",
        "buy_direction_rule": "buy_orders_consume_limit_buy_all_window_result",
        "sell_direction_rule": "sell_orders_consume_limit_sell_all_window_result",
        "no_direction_rule": "direction_none_blocks_when_all_rows_are_buy_limited_or_all_rows_are_sell_limited",
        "order_check_rule": "order_direction_is_preserved_when_delegating_to_is_stock_tradable",
        "daily_bar_rule": "daily_backtests_reduce_the_window_rules_to_the_single_queried_trading_day",
        "joinquant_boundary_rule": "generated_research_must_not_invent_partial_window_or_intraday_tradeability_rules",
        "rdagent_rule": "describe_only_do_not_redefine_trade_window_tradability",
    }
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
    assert "trade_indicator_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "executor_decision_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "strategy_order_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "supervised_label_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "prediction_signal_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "signal_ic_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "portfolio_risk_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "excess_return_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "feedback_metric_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert "benchmark_return_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    assert (
        "universe_benchmark_binding_semantics"
        in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    )
    assert (
        "runtime_handoff_template_binding_semantics"
        in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    )
    assert (
        "research_data_source_semantics" in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    )
    assert (
        "trade_window_tradability_semantics"
        in strict_contract["projection_contract"]["rdagent_prompt_projection_fields"]
    )
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
    assert not _contains_key(
        prompt_payload,
        {"runtime_surfaces", "cost_model", "exchange_kwargs", "backtest_kwargs"},
    )
    assert "open_cost" not in json.dumps(prompt_payload, sort_keys=True)
    assert "close_tax" not in json.dumps(prompt_payload, sort_keys=True)


def test_rdagent_ashare_contract_splits_prompt_projection_from_runtime_handoff() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    handoff = contract["runtime_handoff_contract"]
    template_binding = handoff["template_runtime_binding"]

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
    assert template_binding["semantic_name"] == "a_share_rd_agent_runtime_handoff_template_binding"
    assert template_binding["handoff_id"] == handoff["handoff_id"]
    assert template_binding["binding_kind"] == "rdagent_qlib_template_backtest_runtime_kwargs"
    assert template_binding["rdagent_template_paths"] == [
        "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
    ]
    assert template_binding["required_backtest_kwargs"] == ashare_semantics.joinquant_ashare_backtest_kwargs()
    assert template_binding["forbidden_legacy_exchange_kwargs"] == {
        "limit_threshold": 0.095,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
    }
    assert (
        template_binding["runtime_rule"]
        == "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values"
    )
    assert template_binding["prompt_boundary_rule"] == "execution_kwargs_remain_runtime_handoff_not_prompt_authority"
    assert (
        template_binding["rdagent_rule"]
        == "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs"
    )
    prompt_binding = contract["prompt_projection_payload"]["runtime_handoff_template_binding_semantics"]
    assert prompt_binding == {
        "semantic_name": "a_share_rd_agent_runtime_handoff_template_binding",
        "handoff_id": handoff["handoff_id"],
        "binding_kind": "rdagent_qlib_template_backtest_runtime_kwargs",
        "rdagent_template_paths": template_binding["rdagent_template_paths"],
        "runtime_rule": "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values",
        "prompt_boundary_rule": "execution_kwargs_remain_runtime_handoff_not_prompt_authority",
        "rdagent_rule": "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs",
    }
    assert "required_backtest_kwargs" not in prompt_binding
    assert "forbidden_legacy_exchange_kwargs" not in prompt_binding


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


def test_ashare_trade_window_tradability_contract_matches_exchange_source() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    tradability = contract["prompt_projection_payload"]["trade_window_tradability_semantics"]
    exchange_source = EXCHANGE_PATH.read_text()

    assert tradability["semantic_name"] == "a_share_trade_window_tradability"
    assert tradability["runtime_authority"] == "qlib.backtest.exchange.Exchange.is_stock_tradable"
    assert tradability["suspension_window_authority"] == "qlib.backtest.exchange.Exchange.check_stock_suspended"
    assert tradability["price_limit_window_authority"] == "qlib.backtest.exchange.Exchange.check_stock_limit"
    assert tradability["order_gate_authority"] == "qlib.backtest.exchange.Exchange.check_order"
    assert tradability["quote_membership_rule"] == "unknown_stock_id_is_regarded_as_suspended_and_not_tradable"
    assert tradability["suspension_window_rule"] == "single_missing_close_or_all_missing_close_window_blocks_trading"
    assert tradability["non_suspension_window_rule"] == "any_non_missing_close_in_window_keeps_suspension_gate_open"
    assert (
        tradability["price_limit_window_rule"]
        == "limit_buy_or_limit_sell_blocks_only_when_all_rows_in_window_are_limited"
    )
    assert tradability["buy_direction_rule"] == "buy_orders_consume_limit_buy_all_window_result"
    assert tradability["sell_direction_rule"] == "sell_orders_consume_limit_sell_all_window_result"
    assert (
        tradability["no_direction_rule"]
        == "direction_none_blocks_when_all_rows_are_buy_limited_or_all_rows_are_sell_limited"
    )
    assert tradability["order_check_rule"] == "order_direction_is_preserved_when_delegating_to_is_stock_tradable"
    assert tradability["daily_bar_rule"] == "daily_backtests_reduce_the_window_rules_to_the_single_queried_trading_day"
    assert tradability["rdagent_rule"] == "describe_only_do_not_redefine_trade_window_tradability"

    assert 'field="limit_buy", method="all"' in exchange_source
    assert 'field="limit_sell", method="all"' in exchange_source
    assert "return bool(buy_limit or sell_limit)" in exchange_source
    assert "cast(IndexData, close).isna().all()" in exchange_source
    assert "return np.isnan(close)" in exchange_source
    assert "return True" in exchange_source
    assert "self.check_stock_suspended(stock_id, start_time, end_time)" in exchange_source
    assert "self.check_stock_limit(stock_id, start_time, end_time, direction)" in exchange_source
    assert (
        "return self.is_stock_tradable(order.stock_id, order.start_time, order.end_time, order.direction)"
        in exchange_source
    )


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


def test_ashare_trade_indicator_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    indicator = contract["prompt_projection_payload"]["trade_indicator_semantics"]
    account_source = ACCOUNT_PATH.read_text()
    report_source = REPORT_PATH.read_text()

    assert indicator["semantic_name"] == "a_share_trade_execution_indicator"
    assert indicator["account_indicator_authority"] == "qlib.backtest.account.Account.update_indicator"
    assert indicator["indicator_authority"] == "qlib.backtest.report.Indicator"
    assert indicator["atomic_order_update_authority"] == "qlib.backtest.report.Indicator.update_order_indicators"
    assert indicator["nested_order_aggregation_authority"] == "qlib.backtest.report.Indicator.agg_order_indicators"
    assert indicator["trade_indicator_authority"] == "qlib.backtest.report.Indicator.cal_trade_indicators"
    assert indicator["record_authority"] == "qlib.backtest.report.Indicator.record"
    assert indicator["order_indicator_state"] == "Indicator.order_indicator"
    assert indicator["trade_indicator_state"] == "Indicator.trade_indicator"
    assert indicator["history_state"] == [
        "Indicator.order_indicator_his",
        "Indicator.trade_indicator_his",
    ]
    assert indicator["order_metric_fields"] == [
        "amount",
        "inner_amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
        "pa",
        "ffr",
        "base_price",
        "base_volume",
    ]
    assert indicator["trade_metric_fields"] == [
        "ffr",
        "pa",
        "pos",
        "deal_amount",
        "value",
        "count",
    ]
    assert (
        indicator["bar_end_rule"]
        == "account_update_indicator_runs_after_current_position_valuation_and_portfolio_metrics"
    )
    assert indicator["atomic_rule"] == "atomic_executor_uses_trade_info_to_update_order_indicators"
    assert indicator["nested_rule"] == "non_atomic_executor_aggregates_inner_order_indicators_and_outer_decision"
    assert indicator["fulfill_rate_rule"] == "ffr_equals_deal_amount_reindexed_zero_for_missing_over_order_amount"
    assert indicator["price_advantage_rule"] == "pa_equals_directional_trade_price_over_base_price_minus_one"
    assert indicator["positive_rate_rule"] == "pos_equals_fraction_of_positive_pa"
    assert indicator["deal_amount_metric_rule"] == "deal_amount_metric_sums_absolute_deal_amount"
    assert indicator["trade_value_metric_rule"] == "value_metric_sums_absolute_trade_value"
    assert indicator["order_count_rule"] == "count_metric_counts_order_amount_entries"
    assert indicator["weighting_rule"] == "ffr_and_pa_support_mean_amount_weighted_value_weighted"
    assert indicator["base_price_rule"] == "base_price_uses_exchange_deal_price_with_twap_or_vwap_aggregation"
    assert indicator["unsupported_base_price_rule"] == "non_deal_price_base_price_is_not_supported"
    assert indicator["record_rule"] == "bar_end_records_order_indicator_and_trade_indicator_by_trade_start_time"
    assert indicator["portfolio_boundary_rule"] == (
        "trade_indicators_are_execution_quality_metrics_not_portfolio_return_metrics"
    )
    assert indicator["rdagent_rule"] == "describe_only_do_not_redefine_trade_execution_indicators_or_quality_metrics"
    assert "def update_indicator(" in account_source
    assert "self.indicator.reset()" in account_source
    assert "self.indicator.update_order_indicators(trade_info)" in account_source
    assert "self.indicator.agg_order_indicators(" in account_source
    assert "self.indicator.cal_trade_indicators(trade_start_time, self.freq, indicator_config)" in account_source
    assert "self.indicator.record(trade_start_time)" in account_source
    assert "self.update_indicator(" in account_source
    assert "class Indicator:" in report_source
    assert "self.order_indicator_his: dict = OrderedDict()" in report_source
    assert "self.trade_indicator_his: dict = OrderedDict()" in report_source
    assert "self.trade_indicator: Dict[str, Optional[BaseSingleMetric]] = OrderedDict()" in report_source
    assert 'self.order_indicator.assign("amount", amount)' in report_source
    assert 'self.order_indicator.assign("inner_amount", amount)' in report_source
    assert 'self.order_indicator.assign("deal_amount", deal_amount)' in report_source
    assert 'self.order_indicator.assign("trade_price", trade_price)' in report_source
    assert 'self.order_indicator.assign("trade_value", trade_value)' in report_source
    assert 'self.order_indicator.assign("trade_cost", trade_cost)' in report_source
    assert 'self.order_indicator.assign("trade_dir", trade_dir)' in report_source
    assert 'self.order_indicator.assign("pa", pa)' in report_source
    assert "return tmp_deal_amount / amount" in report_source
    assert 'self.order_indicator.transfer(func, "ffr")' in report_source
    assert "self._agg_order_trade_info(inner_order_indicators)" in report_source
    assert "self._agg_base_price(inner_order_indicators, decision_list, trade_exchange, pa_config=pa_config)" in (
        report_source
    )
    assert "self._agg_order_price_advantage()" in report_source
    assert "price_s = trade_exchange.get_deal_price(" in report_source
    assert 'if agg == "vwap":' in report_source
    assert 'elif agg == "twap":' in report_source
    assert 'raise NotImplementedError(f"This type of input is not supported")' in report_source
    assert "return sign * (trade_price / base_price - 1)" in report_source
    assert "lambda ffr: ffr.mean()" in report_source
    assert "lambda ffr, deal_amount: (ffr * deal_amount.abs()).sum() / (deal_amount.abs().sum())" in report_source
    assert "lambda ffr, trade_value: (ffr * trade_value.abs()).sum() / (trade_value.abs().sum())" in report_source
    assert "lambda pa: pa.mean()" in report_source
    assert "lambda pa, deal_amount: (pa * deal_amount.abs()).sum() / (deal_amount.abs().sum())" in report_source
    assert "lambda pa, trade_value: (pa * trade_value.abs()).sum() / (trade_value.abs().sum())" in report_source
    assert "return (pa > 0).sum() / pa.count()" in report_source
    assert "return deal_amount.abs().sum()" in report_source
    assert "return trade_value.abs().sum()" in report_source
    assert "return amount.count()" in report_source
    assert 'self.trade_indicator["ffr"] = fulfill_rate' in report_source
    assert 'self.trade_indicator["pa"] = price_advantage' in report_source
    assert 'self.trade_indicator["pos"] = positive_rate' in report_source
    assert 'self.trade_indicator["deal_amount"] = deal_amount' in report_source
    assert 'self.trade_indicator["value"] = trade_value' in report_source
    assert 'self.trade_indicator["count"] = order_count' in report_source
    assert "self.order_indicator_his[trade_start_time] = self.get_order_indicator()" in report_source
    assert "self.trade_indicator_his[trade_start_time] = self.get_trade_indicator()" in report_source


def test_ashare_executor_decision_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    executor_decision = contract["prompt_projection_payload"]["executor_decision_semantics"]
    executor_source = EXECUTOR_PATH.read_text()
    decision_source = DECISION_PATH.read_text()
    utils_source = UTILS_PATH.read_text()

    assert executor_decision["semantic_name"] == "a_share_executor_trade_decision_lifecycle"
    assert executor_decision["base_executor_authority"] == "qlib.backtest.executor.BaseExecutor.collect_data"
    assert executor_decision["simulator_executor_authority"] == "qlib.backtest.executor.SimulatorExecutor._collect_data"
    assert executor_decision["nested_executor_authority"] == "qlib.backtest.executor.NestedExecutor._collect_data"
    assert executor_decision["decision_authority"] == "qlib.backtest.decision.BaseTradeDecision"
    assert executor_decision["decision_update_authority"] == "qlib.backtest.decision.BaseTradeDecision.update"
    assert executor_decision["range_limit_authority"] == "qlib.backtest.decision.BaseTradeDecision.get_range_limit"
    assert (
        executor_decision["data_calendar_range_authority"]
        == "qlib.backtest.decision.BaseTradeDecision.get_data_cal_range_limit"
    )
    assert (
        executor_decision["inner_decision_modification_authority"]
        == "qlib.backtest.decision.BaseTradeDecision.mod_inner_decision"
    )
    assert executor_decision["calendar_authority"] == "qlib.backtest.utils.TradeCalendarManager"
    assert executor_decision["level_infra_authority"] == "qlib.backtest.utils.LevelInfrastructure"
    assert executor_decision["atomicity_rule"] == "atomic_executor_rejects_trade_decision_range_limit"
    assert (
        executor_decision["settle_sequence_rule"]
        == "settle_start_runs_before_collection_and_settle_commit_after_bar_end_when_enabled"
    )
    assert executor_decision["bar_end_sequence_rule"] == "executor_updates_account_bar_end_before_trade_calendar_step"
    assert executor_decision["track_data_rule"] == "track_data_yields_outer_trade_decision_for_training_data_only"
    assert (
        executor_decision["simulator_order_rule"]
        == "simulator_executor_retrieves_order_decisions_and_deals_each_order_through_exchange"
    )
    assert executor_decision["simulator_trade_type_rule"] == (
        "serial_preserves_order_sequence_parallel_sorts_buys_before_sells_to_surface_cash_conflicts"
    )
    assert executor_decision["daily_dealt_amount_rule"] == "simulator_resets_dealt_order_amount_when_trade_day_advances"
    assert executor_decision["nested_init_rule"] == (
        "nested_executor_resets_inner_executor_to_outer_step_window_and_passes_outer_decision_to_inner_strategy"
    )
    assert (
        executor_decision["nested_update_rule"]
        == "nested_executor_updates_outer_decision_with_inner_calendar_before_range_limit_alignment"
    )
    assert (
        executor_decision["nested_range_rule"]
        == "nested_executor_skips_inner_steps_outside_range_limit_when_align_range_limit_is_enabled"
    )
    assert executor_decision["inner_decision_rule"] == (
        "outer_trade_decision_may_propagate_trade_range_into_inner_trade_decision_only_when_inner_range_absent"
    )
    assert (
        executor_decision["empty_decision_rule"]
        == "empty_decision_can_skip_inner_loop_when_skip_empty_decision_is_enabled"
    )
    assert executor_decision["inner_result_rule"] == (
        "nested_executor_collects_inner_execute_results_order_indicators_and_decision_time_windows"
    )
    assert (
        executor_decision["rdagent_rule"]
        == "describe_only_do_not_redefine_executor_decision_lifecycle_or_nested_execution_order"
    )
    assert "def collect_data(" in executor_source
    assert "if self.track_data:\n            yield trade_decision" in executor_source
    assert "atomic = not issubclass(self.__class__, NestedExecutor)" in executor_source
    assert 'raise ValueError("atomic executor doesn\'t support specify `range_limit`")' in executor_source
    assert "self.trade_account.current_position.settle_start(self._settle_type)" in executor_source
    assert "self.trade_account.update_bar_end(" in executor_source
    assert "self.trade_calendar.step()" in executor_source
    assert "self.trade_account.current_position.settle_commit()" in executor_source
    assert "def _get_order_iterator(" in executor_source
    assert "order_it = sorted(orders, key=lambda order: -order.direction)" in executor_source
    assert "self.dealt_order_amount = defaultdict(float)" in executor_source
    assert "self.trade_exchange.deal_order(" in executor_source
    assert "def _init_sub_trading(self, trade_decision: BaseTradeDecision) -> None:" in executor_source
    assert "self.inner_executor.reset(start_time=trade_start_time, end_time=trade_end_time)" in executor_source
    assert "self.inner_strategy.reset(level_infra=sub_level_infra, outer_trade_decision=trade_decision)" in (
        executor_source
    )
    assert "trade_decision = self._update_trade_decision(trade_decision)" in executor_source
    assert "if trade_decision.empty() and self._skip_empty_decision:" in executor_source
    assert "start_idx, end_idx = get_start_end_idx(sub_cal, trade_decision)" in executor_source
    assert "if not self._align_range_limit or start_idx <= sub_cal.get_trade_step() <= end_idx:" in executor_source
    assert "trade_decision.mod_inner_decision(_inner_trade_decision)" in executor_source
    assert "decision_list.append((_inner_trade_decision, *sub_cal.get_step_time()))" in executor_source
    assert "inner_order_indicators.append(" in executor_source
    assert "def update(self, trade_calendar: TradeCalendarManager)" in decision_source
    assert "self.total_step = trade_calendar.get_trade_len()" in decision_source
    assert "return self.strategy.update_trade_decision(self, trade_calendar)" in decision_source
    assert "def get_range_limit(self, **kwargs: Any)" in decision_source
    assert 'return kwargs["default_value"]' in decision_source
    assert "def get_data_cal_range_limit(" in decision_source
    assert "if inner_trade_decision.trade_range is None:" in decision_source
    assert "inner_trade_decision.trade_range = self.trade_range" in decision_source
    assert "class TradeCalendarManager:" in utils_source
    assert 'def get_data_cal_range(self, rtype: str = "full")' in utils_source


def test_ashare_strategy_order_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    strategy_order = contract["prompt_projection_payload"]["strategy_order_semantics"]
    base_strategy_source = STRATEGY_BASE_PATH.read_text()
    signal_strategy_source = SIGNAL_STRATEGY_PATH.read_text()
    order_generator_source = ORDER_GENERATOR_PATH.read_text()
    exchange_source = EXCHANGE_PATH.read_text()

    assert strategy_order["semantic_name"] == "a_share_strategy_signal_to_order_generation"
    assert strategy_order["base_strategy_authority"] == "qlib.strategy.base.BaseStrategy.generate_trade_decision"
    assert (
        strategy_order["topk_strategy_authority"]
        == "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision"
    )
    assert (
        strategy_order["weight_strategy_authority"]
        == "qlib.contrib.strategy.signal_strategy.WeightStrategyBase.generate_trade_decision"
    )
    assert strategy_order["trade_decision_type"] == "qlib.backtest.decision.TradeDecisionWO"
    assert strategy_order["signal_authority"] == "qlib.backtest.signal.Signal.get_signal"
    assert strategy_order["template_strategy_binding"] == "qlib.contrib.strategy.TopkDropoutStrategy"
    assert strategy_order["prediction_window_rule"] == "strategy_reads_signal_from_previous_calendar_step_shift_one"
    assert strategy_order["dataframe_signal_rule"] == (
        "topk_dropout_uses_first_signal_column_when_prediction_is_dataframe"
    )
    assert strategy_order["missing_signal_rule"] == "missing_signal_returns_empty_TradeDecisionWO"
    assert strategy_order["topk_selection_rule"] == (
        "topk_dropout_ranks_current_holdings_and_new_candidates_by_pred_score_descending"
    )
    assert strategy_order["dropout_rule"] == (
        "combined_last_and_today_scores_prevent_dropping_higher_score_stock_for_lower_score_buy"
    )
    assert strategy_order["sell_order_rule"] == (
        "sell_orders_are_generated_before_buy_orders_and_simulated_on_temp_position_for_cash"
    )
    assert strategy_order["buy_budget_rule"] == ("buy_budget_equals_temp_cash_times_risk_degree_divided_by_buy_count")
    assert strategy_order["hold_threshold_rule"] == "sell_requires_current_holding_count_at_least_hold_thresh"
    assert strategy_order["only_tradable_rule"] == "only_tradable_filters_selection_candidates_by_exchange_tradability"
    assert strategy_order["forbid_all_trade_at_limit_rule"] == (
        "forbid_all_trade_at_limit_checks_any_limit_direction_else_directional_limit"
    )
    assert strategy_order["buy_round_lot_rule"] == (
        "buy_amount_uses_deal_price_factor_and_exchange_round_amount_by_trade_unit"
    )
    assert strategy_order["weight_strategy_rule"] == (
        "weight_strategy_delegates_target_weight_to_order_generator_after_signal_lookup"
    )
    assert strategy_order["interacting_generator_rule"] == (
        "interacting_order_generator_uses_trade_date_tradability_and_prices"
    )
    assert strategy_order["non_interacting_generator_rule"] == (
        "non_interacting_order_generator_uses_pred_date_close_or_current_holding_price"
    )
    assert strategy_order["target_order_rule"] == (
        "exchange_generates_target_amount_orders_with_deterministic_shuffled_stock_order"
    )
    assert strategy_order["target_order_return_rule"] == "exchange_returns_sell_orders_before_buy_orders"
    assert strategy_order["rdagent_rule"] == "describe_only_do_not_redefine_strategy_signal_to_order_generation"

    assert "class BaseStrategy:" in base_strategy_source
    assert "def generate_trade_decision(" in base_strategy_source
    assert "class TopkDropoutStrategy(BaseSignalStrategy):" in signal_strategy_source
    assert "def generate_trade_decision(self, execute_result=None):" in signal_strategy_source
    assert "pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)" in (
        signal_strategy_source
    )
    assert "pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)" in (
        signal_strategy_source
    )
    assert "if isinstance(pred_score, pd.DataFrame):" in signal_strategy_source
    assert "pred_score = pred_score.iloc[:, 0]" in signal_strategy_source
    assert "return TradeDecisionWO([], self)" in signal_strategy_source
    assert "current_temp: Position = copy.deepcopy(self.trade_position)" in signal_strategy_source
    assert "last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index" in (
        signal_strategy_source
    )
    assert "comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index" in (
        signal_strategy_source
    )
    assert "direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL" in signal_strategy_source
    assert "if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:" in signal_strategy_source
    assert "if self.trade_exchange.check_order(sell_order):" in signal_strategy_source
    assert "self.trade_exchange.deal_order(" in signal_strategy_source
    assert "value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0" in signal_strategy_source
    assert "buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)" in (signal_strategy_source)
    assert "return TradeDecisionWO(sell_order_list + buy_order_list, self)" in signal_strategy_source
    assert "class WeightStrategyBase(BaseSignalStrategy):" in signal_strategy_source
    assert "target_weight_position = self.generate_target_weight_position(" in signal_strategy_source
    assert "self.order_generator.generate_order_list_from_target_weight_position(" in signal_strategy_source
    assert "return TradeDecisionWO(order_list, self)" in signal_strategy_source

    assert "class OrderGenWInteract(OrderGenerator):" in order_generator_source
    assert "if target_weight_position is None:\n            return []" in order_generator_source
    assert "current_tradable_value = trade_exchange.calculate_amount_position_value(" in order_generator_source
    assert "reserved_cash = (1.0 - risk_degree) * (current_total_value + current.get_cash())" in (
        order_generator_source
    )
    assert "current_tradable_value /= 1 + max(trade_exchange.close_cost, trade_exchange.open_cost)" in (
        order_generator_source
    )
    assert "trade_exchange.generate_amount_position_from_weight_position(" in order_generator_source
    assert "class OrderGenWOInteract(OrderGenerator):" in order_generator_source
    assert "risk_total_value = risk_degree * current.calculate_value()" in order_generator_source
    assert "trade_exchange.get_close(stock_id, start_time=pred_start_time, end_time=pred_end_time)" in (
        order_generator_source
    )
    assert "current.get_stock_price(stock_id)" in order_generator_source
    assert "trade_exchange.generate_order_for_target_amount_position(" in order_generator_source

    assert "def generate_order_for_target_amount_position(" in exchange_source
    assert "random.seed(0)" in exchange_source
    assert "random.shuffle(sorted_ids)" in exchange_source
    assert "if not self.is_stock_tradable(stock_id=stock_id, start_time=start_time, end_time=end_time):" in (
        exchange_source
    )
    assert "deal_amount = self.get_real_deal_amount(current_amount, target_amount, factor)" in exchange_source
    assert "return sell_order_list + buy_order_list" in exchange_source


def test_ashare_signal_ic_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    signal_ic = contract["prompt_projection_payload"]["signal_ic_semantics"]
    portfolio_risk = contract["prompt_projection_payload"]["portfolio_risk_semantics"]
    record_temp_source = RECORD_TEMP_PATH.read_text()
    alpha_source = ALPHA_PATH.read_text()

    assert signal_ic["semantic_name"] == "a_share_signal_information_coefficient"
    assert signal_ic["signal_record_authority"] == "qlib.workflow.record_temp.SignalRecord"
    assert signal_ic["signal_analysis_authority"] == "qlib.workflow.record_temp.SigAnaRecord"
    assert signal_ic["high_frequency_signal_analysis_authority"] == "qlib.workflow.record_temp.HFSignalRecord"
    assert signal_ic["ic_calculation_authority"] == "qlib.contrib.eva.alpha.calc_ic"
    assert signal_ic["prediction_artifact"] == "pred.pkl"
    assert signal_ic["label_artifact"] == "label.pkl"
    assert signal_ic["ic_artifact"] == "ic.pkl"
    assert signal_ic["rank_ic_artifact"] == "ric.pkl"
    assert (
        signal_ic["prediction_column_rule"]
        == "series_prediction_is_converted_to_score_dataframe_else_first_prediction_column_is_used"
    )
    assert (
        signal_ic["label_source_rule"]
        == "dataset_prepare_test_label_uses_DataHandlerLP_DK_R_when_supported_else_handler_default"
    )
    assert signal_ic["missing_label_rule"] == "missing_or_empty_label_skips_signal_analysis_generation"
    assert signal_ic["label_column_rule"] == "SigAnaRecord_uses_configured_label_col_default_zero"
    assert signal_ic["groupby_level"] == "datetime"
    assert signal_ic["ic_rule"] == "IC_is_per_datetime_pearson_correlation_between_pred_and_label"
    assert signal_ic["rank_ic_rule"] == "Rank_IC_is_per_datetime_spearman_correlation_between_pred_and_label"
    assert signal_ic["dropna_rule"] == "calc_ic_preserves_nan_by_default_and_drops_nan_only_when_dropna_true"
    assert signal_ic["metric_fields"] == ["IC", "ICIR", "Rank IC", "Rank ICIR"]
    assert signal_ic["metric_aggregation_rule"] == "IC_and_Rank_IC_metrics_are_series_means"
    assert signal_ic["icir_rule"] == "ICIR_is_IC_mean_divided_by_IC_sample_std"
    assert signal_ic["rank_icir_rule"] == "Rank_ICIR_is_Rank_IC_mean_divided_by_Rank_IC_sample_std"
    assert signal_ic["recorder_metric_rule"] == "SigAnaRecord_and_HFSignalRecord_log_metrics_with_exact_metric_names"
    assert signal_ic["rdagent_consumed_metric_paths"] == [
        "IC",
        "ICIR",
        "Rank IC",
        "Rank ICIR",
    ]
    assert (
        signal_ic["portfolio_boundary_rule"]
        == "signal_ic_metrics_are_prediction_label_quality_metrics_not_portfolio_return_metrics"
    )
    assert signal_ic["rdagent_rule"] == "describe_only_do_not_redefine_signal_ic_or_rank_ic_metrics"
    assert "IC" not in portfolio_risk["rdagent_consumed_metric_paths"]

    assert "class SignalRecord(RecordTemp):" in record_temp_source
    assert "raw_label = dataset.prepare(**params)" in record_temp_source
    assert 'del params["data_key"]' in record_temp_source
    assert "pred = self.model.predict(self.dataset)" in record_temp_source
    assert 'pred = pred.to_frame("score")' in record_temp_source
    assert 'self.save(**{"pred.pkl": pred})' in record_temp_source
    assert 'self.save(**{"label.pkl": raw_label})' in record_temp_source
    assert "class SigAnaRecord(ACRecordTemp):" in record_temp_source
    assert 'artifact_path = "sig_analysis"' in record_temp_source
    assert "depend_cls = SignalRecord" in record_temp_source
    assert "self.label_col = label_col" in record_temp_source
    assert 'pred = self.load("pred.pkl")' in record_temp_source
    assert 'label = self.load("label.pkl")' in record_temp_source
    assert "if label is None or not isinstance(label, pd.DataFrame) or label.empty:" in record_temp_source
    assert "ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, self.label_col])" in record_temp_source
    assert '"IC": ic.mean()' in record_temp_source
    assert '"ICIR": ic.mean() / ic.std()' in record_temp_source
    assert '"Rank IC": ric.mean()' in record_temp_source
    assert '"Rank ICIR": ric.mean() / ric.std()' in record_temp_source
    assert 'objects = {"ic.pkl": ic, "ric.pkl": ric}' in record_temp_source
    assert "self.recorder.log_metrics(**metrics)" in record_temp_source
    assert "class HFSignalRecord(SignalRecord):" in record_temp_source
    assert 'artifact_path = "hg_sig_analysis"' in record_temp_source
    assert 'raw_label = self.load("label.pkl")' in record_temp_source
    assert "ic, ric = calc_ic(pred.iloc[:, 0], raw_label.iloc[:, 0])" in record_temp_source

    assert 'def calc_ic(pred: pd.Series, label: pd.Series, date_col="datetime", dropna=False)' in alpha_source
    assert 'df = pd.DataFrame({"pred": pred, "label": label})' in alpha_source
    assert 'ic = df.groupby(date_col, group_keys=False).apply(lambda df: df["pred"].corr(df["label"]))' in (
        alpha_source
    )
    assert 'method="spearman"' in alpha_source
    assert "return ic.dropna(), ric.dropna()" in alpha_source
    assert "return ic, ric" in alpha_source


def test_ashare_supervised_label_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    label_semantics = contract["prompt_projection_payload"]["supervised_label_semantics"]
    handler_source = HANDLER_PATH.read_text()

    assert label_semantics["semantic_name"] == "a_share_supervised_forward_return_label"
    assert label_semantics["handler_authority"] == "qlib.contrib.data.handler.Alpha158"
    assert label_semantics["handler360_authority"] == "qlib.contrib.data.handler.Alpha360"
    assert label_semantics["loader_authority"] == "qlib.contrib.data.loader.Alpha158DL"
    assert label_semantics["processor_authority"] == "qlib.data.dataset.processor.DropnaLabel"
    assert label_semantics["label_group"] == "label"
    assert label_semantics["label_column"] == "LABEL0"
    assert label_semantics["label_expression"] == "Ref($close, -2)/Ref($close, -1) - 1"
    assert label_semantics["label_expression_source"] == "Alpha158.get_label_config_and_Alpha360.get_label_config"
    assert (
        label_semantics["label_horizon_rule"] == "label_at_datetime_t_is_close_t_plus_2_over_close_t_plus_1_minus_one"
    )
    assert label_semantics["prediction_execution_alignment_rule"] == (
        "label_horizon_matches_strategy_previous_step_signal_execution_without_same_day_fill_assumption"
    )
    assert (
        label_semantics["dropna_processor_rule"]
        == "DropnaLabel_removes_missing_LABEL0_rows_before_training_or_evaluation"
    )
    assert label_semantics["template_binding_rule"] == (
        "rdagent_templates_must_use_LABEL0_and_the_qlib_owned_label_expression"
    )
    assert label_semantics["prompt_wording_rule"] == (
        "describe_as_qlib_contract_defined_forward_return_label_not_undefined_next_several_days_return"
    )
    assert label_semantics["rdagent_template_paths"] == [
        "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
    ]
    assert label_semantics["rdagent_prompt_paths"] == ["rdagent/scenarios/qlib/experiment/prompts.yaml"]
    assert label_semantics["rdagent_rule"] == "describe_only_do_not_redefine_supervised_label_expression_or_horizon"

    assert 'def get_label_config(self):\n        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]' in (
        handler_source
    )
    assert '{"class": "DropnaLabel"}' in handler_source
    assert '"label": kwargs.pop("label", self.get_label_config())' in handler_source


def test_ashare_prediction_signal_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    signal_semantics = contract["prompt_projection_payload"]["prediction_signal_semantics"]
    signal_source = SIGNAL_PATH.read_text()
    signal_strategy_source = SIGNAL_STRATEGY_PATH.read_text()
    record_temp_source = RECORD_TEMP_PATH.read_text()

    assert signal_semantics["semantic_name"] == "a_share_prediction_signal_score"
    assert signal_semantics["model_signal_authority"] == "qlib.backtest.signal.ModelSignal"
    assert signal_semantics["signal_cache_authority"] == "qlib.backtest.signal.SignalWCache"
    assert signal_semantics["signal_interface_authority"] == "qlib.backtest.signal.Signal.get_signal"
    assert signal_semantics["signal_record_authority"] == "qlib.workflow.record_temp.SignalRecord"
    assert (
        signal_semantics["strategy_consumption_authority"]
        == "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision"
    )
    assert signal_semantics["prediction_artifact"] == "pred.pkl"
    assert signal_semantics["prediction_column"] == "score"
    assert (
        signal_semantics["model_predict_rule"]
        == "model_predict_output_is_prediction_score_not_realized_or_executable_return"
    )
    assert signal_semantics["series_prediction_rule"] == "series_prediction_is_saved_as_score_column"
    assert (
        signal_semantics["dataframe_prediction_rule"] == "first_prediction_column_is_used_when_prediction_is_dataframe"
    )
    assert signal_semantics["resample_rule"] == "SignalWCache_uses_last_signal_between_decision_start_and_end"
    assert (
        signal_semantics["strategy_ranking_rule"]
        == "TopkDropoutStrategy_sorts_prediction_scores_descending_for_candidate_selection"
    )
    assert signal_semantics["missing_signal_rule"] == "missing_signal_returns_empty_TradeDecisionWO"
    assert (
        signal_semantics["label_alignment_rule"]
        == "prediction_score_is_trained_against_qlib_owned_LABEL0_without_redefining_return_horizon"
    )
    assert signal_semantics["prompt_wording_rule"] == (
        "describe_as_prediction_signal_score_for_LABEL0_not_realized_future_return_or_guaranteed_portfolio_return"
    )
    assert (
        signal_semantics["rdagent_model_output_format_rule"]
        == "rdagent_model_experiment_output_format_must_describe_prediction_as_score_column_indexed_by_datetime_and_instrument_not_graph_node_output"
    )
    assert (
        signal_semantics["rdagent_model_task_boundary_rule"]
        == "rdagent_qlib_model_tasks_must_carry_prediction_signal_score_boundary_to_model_implementation_coder"
    )
    assert (
        signal_semantics["rdagent_model_type_boundary_rule"]
        == "rdagent_qlib_model_experiment_outputs_must_use_tabular_or_timeseries_model_type_only"
    )
    assert (
        signal_semantics["rdagent_model_implementation_prompt_boundary_rule"]
        == "rdagent_qlib_model_implementation_prompts_must_treat_model_output_boundary_as_authority_over_generic_model_type_examples"
    )
    assert (
        signal_semantics["rdagent_model_evaluator_prompt_boundary_rule"]
        == "rdagent_qlib_model_evaluator_prompts_must_reject_model_output_boundary_violations_even_when_execution_or_similar_examples_pass"
    )
    assert signal_semantics["rdagent_supported_model_types"] == ["Tabular", "TimeSeries"]
    assert signal_semantics["rdagent_forbidden_model_types"] == ["Graph", "XGBoost"]
    assert signal_semantics["rdagent_implementation_prompt_paths"] == [
        "rdagent/components/coder/model_coder/prompts.yaml",
    ]
    assert signal_semantics["rdagent_prompt_paths"] == [
        "rdagent/scenarios/qlib/experiment/prompts.yaml",
        "rdagent/scenarios/qlib/prompts.yaml",
    ]
    assert (
        signal_semantics["rdagent_rule"]
        == "describe_only_do_not_redefine_prediction_signal_score_or_return_realization"
    )

    assert "class ModelSignal(SignalWCache):" in signal_source
    assert "pred_scores = self.model.predict(dataset)" in signal_source
    assert "if isinstance(pred_scores, pd.DataFrame):" in signal_source
    assert "pred_scores = pred_scores.iloc[:, 0]" in signal_source
    assert 'signal = resam_ts_data(self.signal_cache, start_time=start_time, end_time=end_time, method="last")' in (
        signal_source
    )
    assert "pred = self.model.predict(self.dataset)" in record_temp_source
    assert 'pred = pred.to_frame("score")' in record_temp_source
    assert 'self.save(**{"pred.pkl": pred})' in record_temp_source
    assert "pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)" in (
        signal_strategy_source
    )
    assert "if isinstance(pred_score, pd.DataFrame):" in signal_strategy_source
    assert "pred_score = pred_score.iloc[:, 0]" in signal_strategy_source
    assert "return TradeDecisionWO([], self)" in signal_strategy_source
    assert "topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)" in (
        signal_strategy_source
    )


def test_ashare_portfolio_risk_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    portfolio_risk = contract["prompt_projection_payload"]["portfolio_risk_semantics"]
    evaluate_source = EVALUATE_PATH.read_text()
    record_temp_source = RECORD_TEMP_PATH.read_text()

    assert portfolio_risk["semantic_name"] == "a_share_portfolio_risk_analysis"
    assert portfolio_risk["record_authority"] == "qlib.workflow.record_temp.PortAnaRecord"
    assert portfolio_risk["risk_analysis_authority"] == "qlib.contrib.evaluate.risk_analysis"
    assert portfolio_risk["freq_authority"] == "qlib.utils.resam.Freq.parse"
    assert (
        portfolio_risk["backtest_source_rule"]
        == "PortAnaRecord_runs_normal_backtest_and_reads_portfolio_metric_dict_by_freq"
    )
    assert (
        portfolio_risk["report_artifact_rule"]
        == "report_normal_dataframe_saved_as_portfolio_analysis_report_normal_{freq}_pkl"
    )
    assert (
        portfolio_risk["risk_artifact_rule"]
        == "risk_analysis_dataframe_saved_as_portfolio_analysis_port_analysis_{freq}_pkl"
    )
    assert portfolio_risk["recorder_metric_rule"] == "risk_metrics_are_logged_as_{freq}.{report_type}.{risk_metric}"
    assert (
        portfolio_risk["default_frequency_rule"]
        == "missing_risk_analysis_freq_uses_first_executor_portfolio_metric_frequency"
    )
    assert portfolio_risk["required_report_columns"] == [
        "return",
        "bench",
        "cost",
        "turnover",
    ]
    assert (
        portfolio_risk["turnover_report_metric_rule"]
        == "report_turnover_is_post_backtest_portfolio_metric_not_default_factor_input_field"
    )
    assert portfolio_risk["report_type_fields"] == [
        "excess_return_without_cost",
        "excess_return_with_cost",
    ]
    assert portfolio_risk["excess_without_cost_rule"] == "report_return_minus_benchmark"
    assert portfolio_risk["excess_with_cost_rule"] == "report_return_minus_benchmark_minus_cost"
    assert portfolio_risk["risk_metric_fields"] == [
        "mean",
        "std",
        "annualized_return",
        "information_ratio",
        "max_drawdown",
    ]
    assert portfolio_risk["default_accumulation_mode"] == "sum"
    assert portfolio_risk["supported_accumulation_modes"] == ["sum", "product"]
    assert portfolio_risk["sum_mode_rule"] == (
        "qlib_sum_mode_uses_arithmetic_cumulative_return_not_geometric_compounding"
    )
    assert portfolio_risk["day_annualization_scaler"] == 238
    assert (
        portfolio_risk["annualization_scaler_rule"]
        == "risk_analysis_parses_freq_when_N_is_absent_and_N_overrides_freq_when_present"
    )
    assert portfolio_risk["mean_rule"] == "sum_mode_mean_equals_return_series_mean"
    assert portfolio_risk["std_rule"] == "sum_mode_std_uses_sample_standard_deviation_ddof_one"
    assert (
        portfolio_risk["annualized_return_rule"] == "sum_mode_annualized_return_equals_mean_times_annualization_scaler"
    )
    assert portfolio_risk["information_ratio_rule"] == (
        "information_ratio_equals_mean_over_std_times_square_root_annualization_scaler"
    )
    assert portfolio_risk["max_drawdown_rule"] == (
        "sum_mode_max_drawdown_equals_min_of_cumulative_return_minus_running_cumulative_max"
    )
    assert portfolio_risk["metric_path_format"] == "{freq}.{report_type}.{risk_metric}"
    assert portfolio_risk["metric_path_frequency"] == "1day"
    assert portfolio_risk["metric_path_whitespace_rule"] == (
        "metric_paths_are_exact_without_leading_or_trailing_whitespace"
    )
    assert portfolio_risk["metric_path_report_type_rule"] == (
        "prompt_context_uses_without_cost_and_feedback_bandit_ui_use_with_cost"
    )
    assert portfolio_risk["rdagent_prompt_metric_paths"] == [
        "1day.excess_return_without_cost.annualized_return",
        "1day.excess_return_without_cost.max_drawdown",
    ]
    assert portfolio_risk["rdagent_feedback_metric_paths"] == [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert portfolio_risk["rdagent_bandit_metric_paths"] == [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert portfolio_risk["rdagent_ui_metric_paths"] == [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert portfolio_risk["rdagent_consumed_metric_paths"] == [
        "1day.excess_return_without_cost.annualized_return",
        "1day.excess_return_without_cost.max_drawdown",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert all(path == path.strip() for path in portfolio_risk["rdagent_consumed_metric_paths"])
    assert portfolio_risk["rdagent_rule"] == "describe_only_do_not_redefine_portfolio_risk_analysis_metrics"

    assert "class PortAnaRecord(ACRecordTemp):" in record_temp_source
    assert 'artifact_path = "portfolio_analysis"' in record_temp_source
    assert "depend_cls = SignalRecord" in record_temp_source
    assert '"generate_portfolio_metrics": True' in record_temp_source
    assert "risk_analysis_freq = [self.all_freq[0]]" in record_temp_source
    assert '"{0}{1}".format(*Freq.parse(_analysis_freq))' in record_temp_source
    assert "portfolio_metric_dict, indicator_dict = normal_backtest(" in record_temp_source
    assert 'artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})' in record_temp_source
    assert 'analysis["excess_return_without_cost"] = risk_analysis(' in record_temp_source
    assert 'report_normal["return"] - report_normal["bench"], freq=_analysis_freq' in record_temp_source
    assert 'analysis["excess_return_with_cost"] = risk_analysis(' in record_temp_source
    assert 'report_normal["return"] - report_normal["bench"] - report_normal["cost"]' in record_temp_source
    assert "analysis_df = pd.concat(analysis)" in record_temp_source
    assert 'analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())' in record_temp_source
    assert 'self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})' in (
        record_temp_source
    )
    assert 'artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})' in record_temp_source

    assert 'def risk_analysis(r, N: int = None, freq: str = "day", mode: Literal["sum", "product"] = "sum")' in (
        evaluate_source
    )
    assert "Freq.NORM_FREQ_DAY: 238" in evaluate_source
    assert "if N is None:\n        N = cal_risk_analysis_scaler(freq)" in evaluate_source
    assert "mean = r.mean()" in evaluate_source
    assert "std = r.std(ddof=1)" in evaluate_source
    assert "annualized_return = mean * N" in evaluate_source
    assert "max_drawdown = (r.cumsum() - r.cumsum().cummax()).min()" in evaluate_source
    assert "information_ratio = mean / std * np.sqrt(N)" in evaluate_source
    assert '"annualized_return": annualized_return' in evaluate_source
    assert '"max_drawdown": max_drawdown' in evaluate_source
    assert 'res = pd.Series(data).to_frame("risk")' in evaluate_source


def test_ashare_excess_return_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    excess_return = contract["prompt_projection_payload"]["excess_return_semantics"]
    record_temp_source = RECORD_TEMP_PATH.read_text()
    analysis_report_source = ANALYSIS_POSITION_REPORT_PATH.read_text()
    analysis_risk_source = ANALYSIS_POSITION_RISK_PATH.read_text()
    online_operator_source = ONLINE_OPERATOR_PATH.read_text()
    online_user_source = ONLINE_USER_PATH.read_text()

    assert excess_return["semantic_name"] == "a_share_benchmark_relative_excess_return"
    assert excess_return["benchmark_dependency"] == "benchmark_return_semantics"
    assert excess_return["portfolio_risk_dependency"] == "portfolio_risk_semantics"
    assert excess_return["report_column_authority"] == "qlib.backtest.report.PortfolioMetrics"
    assert excess_return["risk_record_authority"] == "qlib.workflow.record_temp.PortAnaRecord"
    assert (
        excess_return["report_graph_authority"] == "qlib.contrib.report.analysis_position.report._calculate_report_data"
    )
    assert (
        excess_return["risk_graph_authority"]
        == "qlib.contrib.report.analysis_position.risk_analysis._get_risk_analysis_data_with_report"
    )
    assert excess_return["online_analysis_authority"] == "qlib.contrib.online.operator"
    assert excess_return["user_analysis_authority"] == "qlib.contrib.online.user"
    assert excess_return["required_report_columns"] == ["return", "bench", "cost"]
    assert excess_return["without_cost_field"] == "excess_return_without_cost"
    assert excess_return["with_cost_field"] == "excess_return_with_cost"
    assert excess_return["without_cost_formula"] == "return - bench"
    assert excess_return["with_cost_formula"] == "return - bench - cost"
    assert excess_return["cumulative_without_cost_field"] == "cum_ex_return_wo_cost"
    assert excess_return["cumulative_with_cost_field"] == "cum_ex_return_w_cost"
    assert excess_return["cost_source"] == "reported_cost_column_from_trade_indicator_semantics"
    assert excess_return["benchmark_source"] == "reported_bench_column_from_benchmark_return_semantics"
    assert excess_return["metric_path_without_cost"] == "1day.excess_return_without_cost.annualized_return"
    assert excess_return["metric_path_with_cost"] == "1day.excess_return_with_cost.annualized_return"
    assert (
        excess_return["rdagent_prompt_rule"]
        == "generated_research_must_report_benchmark_relative_excess_return_not_raw_return"
    )
    assert excess_return["forbidden_substitutions"] == [
        "raw_return_as_excess_return",
        "market_universe_as_benchmark_return",
        "with_cost_metric_without_report_cost_column",
        "prompt_defined_cost_or_benchmark_formula",
    ]
    assert excess_return["rdagent_rule"] == "describe_only_do_not_redefine_benchmark_relative_excess_return"

    assert 'analysis["excess_return_without_cost"] = risk_analysis(' in record_temp_source
    assert 'report_normal["return"] - report_normal["bench"], freq=_analysis_freq' in record_temp_source
    assert 'analysis["excess_return_with_cost"] = risk_analysis(' in record_temp_source
    assert 'report_normal["return"] - report_normal["bench"] - report_normal["cost"]' in record_temp_source

    assert 'report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()' in (analysis_report_source)
    assert 'report_df["cum_ex_return_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()' in (
        analysis_report_source
    )
    assert 'report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd((df["return"] - df["bench"]).cumsum())' in (
        analysis_report_source
    )
    assert (
        'report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"] - df["bench"]).cumsum())'
        in (analysis_report_source)
    )

    assert (
        'analysis["excess_return_without_cost"] = risk_analysis(report_normal_df["return"] - report_normal_df["bench"])'
        in (analysis_risk_source)
    )
    assert 'report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]' in (analysis_risk_source)
    assert 'r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()' in online_operator_source
    assert 'r = (portfolio_metrics["return"] - portfolio_metrics["bench"] - portfolio_metrics["cost"]).dropna()' in (
        online_operator_source
    )
    assert 'r = (portfolio_metrics["return"] - portfolio_metrics["bench"]).dropna()' in online_user_source
    assert 'r = (portfolio_metrics["return"] - portfolio_metrics["bench"] - portfolio_metrics["cost"]).dropna()' in (
        online_user_source
    )


def test_ashare_feedback_metric_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    feedback_metric = contract["prompt_projection_payload"]["feedback_metric_semantics"]
    record_temp_source = RECORD_TEMP_PATH.read_text()
    evaluate_source = EVALUATE_PATH.read_text()

    assert feedback_metric["semantic_name"] == "a_share_rd_agent_feedback_metric_consumption"
    assert feedback_metric["signal_metric_authority"] == "qlib.workflow.record_temp.SigAnaRecord"
    assert feedback_metric["portfolio_metric_authority"] == "qlib.workflow.record_temp.PortAnaRecord"
    assert feedback_metric["risk_metric_authority"] == "qlib.contrib.evaluate.risk_analysis"
    assert feedback_metric["prompt_metric_paths"] == [
        "IC",
        "1day.excess_return_without_cost.annualized_return",
        "1day.excess_return_without_cost.max_drawdown",
    ]
    assert feedback_metric["feedback_metric_paths"] == [
        "IC",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert feedback_metric["bandit_metric_paths"] == [
        "IC",
        "ICIR",
        "Rank IC",
        "Rank ICIR",
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
    ]
    assert feedback_metric["feedback_primary_metric"] == "1day.excess_return_with_cost.annualized_return"
    assert (
        feedback_metric["sota_fallback_rule"]
        == "missing_explicit_feedback_decision_uses_feedback_primary_metric_improvement"
    )
    assert feedback_metric["derived_bandit_utility_name"] == "drawdown_adjusted_return"
    assert (
        feedback_metric["derived_bandit_utility_rule"]
        == "rdagent_may_compute_arr_over_abs_max_drawdown_as_derived_utility_not_qlib_metric"
    )
    assert feedback_metric["forbidden_metric_aliases"] == ["sharpe", "Sharpe"]
    assert (
        feedback_metric["prompt_metric_wording_rule"]
        == "describe_exact_qlib_metric_paths_not_generic_return_sharpe_or_and_so_on"
    )
    assert feedback_metric["rdagent_source_paths"] == [
        "rdagent/scenarios/qlib/developer/feedback.py",
        "rdagent/scenarios/qlib/proposal/bandit.py",
        "rdagent/scenarios/qlib/experiment/prompts.yaml",
        "rdagent/scenarios/qlib/prompts.yaml",
        "rdagent/log/ui/app.py",
    ]
    assert (
        feedback_metric["rdagent_rule"]
        == "consume_exact_qlib_metric_paths_and_label_derived_bandit_utility_as_non_qlib_metric"
    )

    assert "class SigAnaRecord(ACRecordTemp):" in record_temp_source
    assert "class PortAnaRecord(ACRecordTemp):" in record_temp_source
    assert '"IC": ic.mean()' in record_temp_source
    assert '"ICIR": ic.mean() / ic.std()' in record_temp_source
    assert '"Rank IC": ric.mean()' in record_temp_source
    assert '"Rank ICIR": ric.mean() / ric.std()' in record_temp_source
    assert 'self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})' in (
        record_temp_source
    )
    assert '"annualized_return": annualized_return' in evaluate_source
    assert '"information_ratio": information_ratio' in evaluate_source
    assert '"max_drawdown": max_drawdown' in evaluate_source


def test_ashare_benchmark_return_contract_matches_runtime_sources() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    benchmark_return = contract["prompt_projection_payload"]["benchmark_return_semantics"]
    backtest_source = BACKTEST_INIT_PATH.read_text()
    report_source = REPORT_PATH.read_text()

    assert benchmark_return["semantic_name"] == "a_share_benchmark_return_series"
    assert benchmark_return["default_benchmark"] == "SH000300"
    assert benchmark_return["benchmark_constant_authority"] == "qlib.tests.config.CSI300_BENCH"
    assert benchmark_return["backtest_entry_authority"] == "qlib.backtest.backtest"
    assert benchmark_return["account_config_authority"] == "qlib.backtest.create_account_instance"
    assert benchmark_return["portfolio_metric_authority"] == "qlib.backtest.report.PortfolioMetrics"
    assert benchmark_return["benchmark_calculation_authority"] == "qlib.backtest.report.PortfolioMetrics._cal_benchmark"
    assert benchmark_return["benchmark_sampling_authority"] == "qlib.backtest.report.PortfolioMetrics._sample_benchmark"
    assert benchmark_return["feature_query_authority"] == "qlib.utils.resam.get_higher_eq_freq_feature"
    assert benchmark_return["resample_authority"] == "qlib.utils.resam.resam_ts_data"
    assert benchmark_return["accepted_benchmark_inputs"] == [
        "str",
        "list",
        "dict",
        "pd.Series",
        "None",
    ]
    assert benchmark_return["default_rule"] == "missing_benchmark_key_uses_CSI300_BENCH_SH000300"
    assert benchmark_return["none_rule"] == "benchmark_config_none_or_benchmark_none_disables_benchmark_series"
    assert benchmark_return["series_rule"] == "pd_series_benchmark_is_used_directly_as_per_period_return_series"
    assert benchmark_return["code_rule"] == "str_benchmark_is_queried_as_single_code_close_over_ref_close_minus_one"
    assert benchmark_return["basket_rule"] == "list_or_dict_benchmark_is_queried_as_codes_and_averaged_by_datetime"
    assert benchmark_return["benchmark_field_expression"] == "$close/Ref($close,1)-1"
    assert benchmark_return["missing_frequency_rule"] == "non_series_benchmark_requires_freq_else_ValueError"
    assert benchmark_return["missing_benchmark_rule"] == "empty_feature_result_raises_ValueError"
    assert benchmark_return["fillna_rule"] == "queried_benchmark_returns_fillna_zero_after_datetime_average"
    assert benchmark_return["sample_rule"] == "bar_benchmark_return_equals_product_of_one_plus_period_returns_minus_one"
    assert benchmark_return["direct_bench_value_rule"] == "provided_bench_value_overrides_sampling"
    assert benchmark_return["unusable_benchmark_rule"] == "trade_end_time_and_bench_value_both_none_raise_ValueError"
    assert benchmark_return["report_column"] == "bench"
    assert (
        benchmark_return["portfolio_risk_dependency"] == "portfolio_risk_excess_returns_use_report_normal_bench_column"
    )
    assert (
        benchmark_return["rdagent_rule"] == "describe_only_do_not_redefine_benchmark_return_series_or_default_benchmark"
    )

    assert 'benchmark: Optional[str] = "SH000300"' in backtest_source
    assert 'benchmark: str = "SH000300"' in backtest_source
    assert '"benchmark": benchmark' in backtest_source
    assert '"start_time": start_time' in backtest_source
    assert '"end_time": end_time' in backtest_source
    assert "benchmark_config=(" in backtest_source

    assert "from ..tests.config import CSI300_BENCH" in report_source
    assert "self.benches: dict = OrderedDict()" in report_source
    assert 'pm["bench"] = pd.Series(self.benches)' in report_source
    assert 'benchmark = benchmark_config.get("benchmark", CSI300_BENCH)' in report_source
    assert "if benchmark_config is None:\n            return None" in report_source
    assert "if benchmark is None:\n            return None" in report_source
    assert "if isinstance(benchmark, pd.Series):\n            return benchmark" in report_source
    assert "_codes = benchmark if isinstance(benchmark, (list, dict)) else [benchmark]" in report_source
    assert 'fields = ["$close/Ref($close,1)-1"]' in report_source
    assert "get_higher_eq_freq_feature(_codes, fields, start_time, end_time, freq=freq)" in report_source
    assert 'raise ValueError("benchmark freq can\'t be None!")' in report_source
    assert 'raise ValueError(f"The benchmark {_codes} does not exist. Please provide the right benchmark")' in (
        report_source
    )
    assert '.groupby(level="datetime", group_keys=False)' in report_source
    assert ".mean()\n                .fillna(0)" in report_source
    assert "def cal_change(x):\n            return (x + 1).prod()" in report_source
    assert "resam_ts_data(bench, trade_start_time, trade_end_time, method=cal_change)" in report_source
    assert "return 0.0 if _ret is None else _ret - 1" in report_source
    assert 'raise ValueError("Both trade_end_time and bench_value is None, benchmark is not usable.")' in report_source
    assert "bench_value = self._sample_benchmark(self.bench, trade_start_time, trade_end_time)" in report_source
    assert "self.benches[trade_start_time] = bench_value" in report_source


def test_ashare_universe_benchmark_binding_contract_matches_config_constants() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    binding = contract["prompt_projection_payload"]["universe_benchmark_binding_semantics"]
    config_source = CONFIG_PATH.read_text()

    assert binding["semantic_name"] == "a_share_rd_agent_universe_benchmark_binding"
    assert binding["market_universe_authority"] == "qlib.tests.config.CSI300_MARKET"
    assert binding["benchmark_authority"] == "qlib.tests.config.CSI300_BENCH"
    assert binding["template_market_value"] == "csi300"
    assert binding["template_benchmark_value"] == "SH000300"
    assert binding["template_market_anchor"] == "market: &market csi300"
    assert binding["template_instruments_binding"] == "instruments: *market"
    assert binding["template_benchmark_anchor"] == "benchmark: &benchmark SH000300"
    assert binding["template_backtest_benchmark_binding"] == "benchmark: *benchmark"
    assert binding["market_universe_rule"] == "csi300_template_market_selects_instruments_only"
    assert binding["benchmark_rule"] == "SH000300_template_benchmark_is_portfolio_excess_return_baseline_only"
    assert binding["separation_rule"] == "market_universe_membership_and_benchmark_return_series_are_not_substitutable"
    assert binding["forbidden_template_values"] == ["all_a", "all", "SH000300_as_market", "csi300_as_benchmark"]
    assert binding["rdagent_template_paths"] == [
        "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
        "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
        "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
    ]
    assert binding["rdagent_rule"] == "bind_market_to_instruments_and_benchmark_to_backtest_without_cross_aliasing"
    assert 'CSI300_MARKET = "csi300"' in config_source
    assert 'CSI300_BENCH = "SH000300"' in config_source


def test_ashare_research_data_source_contract_bounds_rd_agent_factor_prompts() -> None:
    contract = ashare_semantics.rdagent_ashare_semantic_contract()
    source_boundary = contract["prompt_projection_payload"]["research_data_source_semantics"]
    handler_source = HANDLER_PATH.read_text()

    assert source_boundary["semantic_name"] == "a_share_research_data_source_boundary"
    assert source_boundary["data_frequency"] == "day"
    assert source_boundary["source_scope"] == "qlib_daily_research_and_backtest_inputs"
    assert source_boundary["index_contract"] == ["datetime", "instrument"]
    assert source_boundary["primary_price_volume_fields"] == [
        "$open",
        "$close",
        "$high",
        "$low",
        "$vwap",
        "$volume",
    ]
    assert source_boundary["handler_authority"] == "qlib.contrib.data.handler.Alpha158"
    assert source_boundary["handler360_authority"] == "qlib.contrib.data.handler.Alpha360"
    assert source_boundary["loader_authority"] == "qlib.contrib.data.loader.Alpha158DL"
    assert source_boundary["allowed_prompt_source_tables"] == [
        "daily_stock_trade_data",
        "daily_price_volume_derived_features",
        "provider_supplied_point_in_time_fundamental_or_industry_fields",
    ]
    assert (
        source_boundary["derived_feature_source_rule"]
        == "alpha158_alpha360_derived_features_must_be_computed_only_from_registered_daily_price_volume_fields"
    )
    assert source_boundary["point_in_time_rule"] == (
        "non_price_volume_fields_are_allowed_only_when_user_or_provider_supplies_daily_point_in_time_data"
    )
    assert source_boundary["point_in_time_registration_rule"] == (
        "user_or_provider_supplied_non_price_volume_fields_must_name_source_owner_field_identity_and_daily_point_in_time_validity"
    )
    assert source_boundary["forbidden_default_prompt_sources"] == [
        "turnover",
        "minute_level_high_frequency_data",
        "analyst_consensus_expectation_factor",
        "unregistered_external_vendor_fields",
    ]
    assert (
        source_boundary["turnover_input_boundary_rule"]
        == "turnover_is_not_a_default_factor_input_field_even_when_qlib_reports_portfolio_turnover"
    )
    assert "turnover" not in source_boundary["primary_price_volume_fields"]
    assert (
        source_boundary["frequency_rule"]
        == "rdagent_factor_extraction_prompts_must_not_advertise_minute_or_intraday_data_as_default"
    )
    assert (
        source_boundary["rdagent_prompt_obligation_rule"]
        == "rdagent_factor_extraction_viability_relevance_duplicate_and_implementation_prompts_must_apply_source_boundary_forbidden_defaults_and_turnover_distinction"
    )
    assert source_boundary["rdagent_prompt_paths"] == [
        "rdagent/scenarios/qlib/factor_experiment_loader/prompts.yaml",
        "rdagent/scenarios/qlib/prompts.yaml",
        "rdagent/components/coder/factor_coder/prompts.yaml",
    ]
    assert (
        source_boundary["rdagent_rule"]
        == "describe_only_use_qlib_registered_daily_or_user_supplied_point_in_time_sources"
    )
    assert '"feature": ["OPEN", "HIGH", "LOW", "VWAP"]' in handler_source
    assert 'return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]' in handler_source


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
        round_tripped["prompt_projection_payload"]["trade_indicator_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_trade_execution_indicators_or_quality_metrics"
    )
    assert round_tripped["prompt_projection_payload"]["trade_indicator_semantics"]["trade_metric_fields"] == [
        "ffr",
        "pa",
        "pos",
        "deal_amount",
        "value",
        "count",
    ]
    assert (
        round_tripped["prompt_projection_payload"]["executor_decision_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_executor_decision_lifecycle_or_nested_execution_order"
    )
    assert (
        round_tripped["prompt_projection_payload"]["executor_decision_semantics"]["base_executor_authority"]
        == "qlib.backtest.executor.BaseExecutor.collect_data"
    )
    assert (
        round_tripped["prompt_projection_payload"]["strategy_order_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_strategy_signal_to_order_generation"
    )
    assert (
        round_tripped["prompt_projection_payload"]["strategy_order_semantics"]["topk_strategy_authority"]
        == "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision"
    )
    assert (
        round_tripped["prompt_projection_payload"]["prediction_signal_semantics"]["rdagent_model_task_boundary_rule"]
        == "rdagent_qlib_model_tasks_must_carry_prediction_signal_score_boundary_to_model_implementation_coder"
    )
    assert (
        round_tripped["prompt_projection_payload"]["prediction_signal_semantics"]["rdagent_model_type_boundary_rule"]
        == "rdagent_qlib_model_experiment_outputs_must_use_tabular_or_timeseries_model_type_only"
    )
    assert (
        round_tripped["prompt_projection_payload"]["prediction_signal_semantics"][
            "rdagent_model_implementation_prompt_boundary_rule"
        ]
        == "rdagent_qlib_model_implementation_prompts_must_treat_model_output_boundary_as_authority_over_generic_model_type_examples"
    )
    assert (
        round_tripped["prompt_projection_payload"]["prediction_signal_semantics"][
            "rdagent_model_evaluator_prompt_boundary_rule"
        ]
        == "rdagent_qlib_model_evaluator_prompts_must_reject_model_output_boundary_violations_even_when_execution_or_similar_examples_pass"
    )
    assert round_tripped["prompt_projection_payload"]["prediction_signal_semantics"][
        "rdagent_supported_model_types"
    ] == ["Tabular", "TimeSeries"]
    assert round_tripped["prompt_projection_payload"]["prediction_signal_semantics"][
        "rdagent_forbidden_model_types"
    ] == ["Graph", "XGBoost"]
    assert round_tripped["prompt_projection_payload"]["prediction_signal_semantics"][
        "rdagent_implementation_prompt_paths"
    ] == ["rdagent/components/coder/model_coder/prompts.yaml"]
    assert (
        round_tripped["prompt_projection_payload"]["signal_ic_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_signal_ic_or_rank_ic_metrics"
    )
    assert round_tripped["prompt_projection_payload"]["signal_ic_semantics"]["metric_fields"] == [
        "IC",
        "ICIR",
        "Rank IC",
        "Rank ICIR",
    ]
    assert (
        round_tripped["prompt_projection_payload"]["portfolio_risk_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_portfolio_risk_analysis_metrics"
    )
    assert (
        round_tripped["prompt_projection_payload"]["portfolio_risk_semantics"]["risk_analysis_authority"]
        == "qlib.contrib.evaluate.risk_analysis"
    )
    assert (
        round_tripped["prompt_projection_payload"]["excess_return_semantics"]["without_cost_formula"]
        == "return - bench"
    )
    assert (
        round_tripped["prompt_projection_payload"]["excess_return_semantics"]["with_cost_formula"]
        == "return - bench - cost"
    )
    assert (
        round_tripped["prompt_projection_payload"]["excess_return_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_benchmark_relative_excess_return"
    )
    assert (
        round_tripped["prompt_projection_payload"]["feedback_metric_semantics"]["derived_bandit_utility_name"]
        == "drawdown_adjusted_return"
    )
    assert round_tripped["prompt_projection_payload"]["feedback_metric_semantics"]["forbidden_metric_aliases"] == [
        "sharpe",
        "Sharpe",
    ]
    assert round_tripped["prompt_projection_payload"]["benchmark_return_semantics"]["default_benchmark"] == "SH000300"
    assert (
        round_tripped["prompt_projection_payload"]["benchmark_return_semantics"]["rdagent_rule"]
        == "describe_only_do_not_redefine_benchmark_return_series_or_default_benchmark"
    )
    assert (
        round_tripped["prompt_projection_payload"]["research_data_source_semantics"]["rdagent_rule"]
        == "describe_only_use_qlib_registered_daily_or_user_supplied_point_in_time_sources"
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
    assert round_tripped["runtime_handoff_contract"]["template_runtime_binding"][
        "required_backtest_kwargs"
    ] == ashare_semantics.joinquant_ashare_backtest_kwargs(strict_price_limit=False)


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
