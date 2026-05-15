# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
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
RDAGENT_ASHARE_CONTRACT_ID = "rdagent_qlib_joinquant_ashare_semantic_contract_v1"
RDAGENT_ASHARE_RUNTIME_HANDOFF_ID = "qlib_joinquant_ashare_runtime_handoff_v1"
RDAGENT_ASHARE_PROMPT_PROJECTION_ID = "qlib_joinquant_ashare_prompt_projection_v1"
RDAGENT_ASHARE_PROMPT_PROJECTION_SCHEMA_VERSION = "qlib_ashare_prompt_projection.v1"
QLIB_ASHARE_AUTHORITY_COMPONENT = "qlib.backtest.ashare_semantics"
RDAGENT_ASHARE_CONSUMER_COMPONENT = "rdagent.scenarios.qlib.ashare_semantics"


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
    close_commission: float = 0.0003
    close_tax: float = 0.001
    min_cost: float = 5.0
    position_type: str = "AsharePosition"
    price_limit_mode: str = "auto"
    up_limit_field: str = "$up_limit"
    down_limit_field: str = "$down_limit"
    tolerance: float = 1e-8
    main_board_threshold: float = 0.095
    star_chinext_threshold: float = 0.195
    bse_threshold: float = 0.295
    chinext_registration_start_date: str = "2020-08-24"

    def normalized_mode(self) -> str:
        mode = self.price_limit_mode.strip().lower()
        if mode not in {"auto", "strict", "board_fallback"}:
            raise ValueError(
                "price_limit_mode must be one of auto, strict, board_fallback; " f"got {self.price_limit_mode!r}"
            )
        return mode

    def limit_threshold_for_instrument(self, instrument: str, trade_date: object | None = None) -> float:
        normalized = normalize_ashare_instrument(instrument)
        if normalized.startswith("SH688"):
            return self.star_chinext_threshold
        if normalized.startswith("SZ300"):
            timestamp = pd.Timestamp(trade_date) if trade_date is not None else None
            if timestamp is not None and timestamp < pd.Timestamp(self.chinext_registration_start_date):
                return self.main_board_threshold
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
            missing_tradable_bounds = missing_bounds & ~suspended
            if self.normalized_mode() == "strict" and missing_tradable_bounds.any():
                missing_count = int(missing_tradable_bounds.sum())
                raise ValueError(
                    "joinquant_ashare strict price-limit mode requires non-null "
                    f"{self.up_limit_field}/{self.down_limit_field} on non-suspended rows; "
                    f"missing rows={missing_count}"
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
        instruments = self._quote_axis_values(quote_df, "instrument")
        datetimes = self._quote_axis_values(quote_df, "datetime", default=None)
        thresholds = [
            self.limit_threshold_for_instrument(str(instrument), trade_date=trade_date)
            for instrument, trade_date in zip(instruments, datetimes)
        ]
        return pd.Series(thresholds, index=quote_df.index, dtype="float64")

    def _quote_axis_values(self, quote_df: pd.DataFrame, name: str, default: object = "") -> pd.Index:
        if isinstance(quote_df.index, pd.MultiIndex) and name in quote_df.index.names:
            return quote_df.index.get_level_values(name)
        if quote_df.index.name == name:
            return pd.Index(quote_df.index)
        if name in quote_df.columns:
            return pd.Index(quote_df[name])
        return pd.Index([default] * len(quote_df))

    def calculate_trade_cost(self, side: str, trade_value: float, *, impact_cost: float = 0.0) -> float:
        if trade_value <= 1e-5:
            return 0.0
        normalized_side = side.strip().lower()
        if normalized_side == "buy":
            commission_rate = self.open_cost
            tax_rate = 0.0
        elif normalized_side == "sell":
            commission_rate = self.close_commission
            tax_rate = self.close_tax
        else:
            raise ValueError(f"Unsupported A-share trade side: {side!r}")
        commission = max(trade_value * commission_rate, self.min_cost)
        return commission + trade_value * (tax_rate + impact_cost)

    def cost_options(self) -> dict[str, float]:
        return {
            "open_cost": self.open_cost,
            "close_commission": self.close_commission,
            "close_tax": self.close_tax,
            "min_cost": self.min_cost,
        }


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

    cost_options = JOINQUANT_ASHARE_POLICY.cost_options()
    return {
        "limit_threshold": JOINQUANT_ASHARE_LIMIT_THRESHOLD,
        "ashare_price_limit_mode": "strict" if strict_price_limit else "auto",
        "ashare_limit_options": cost_options,
        "trade_unit": JOINQUANT_ASHARE_POLICY.trade_unit,
        "deal_price": JOINQUANT_ASHARE_POLICY.deal_price,
        "open_cost": cost_options["open_cost"],
        "close_cost": JOINQUANT_ASHARE_POLICY.close_cost,
        "min_cost": cost_options["min_cost"],
    }


def joinquant_ashare_backtest_kwargs(*, strict_price_limit: bool = True) -> dict[str, Any]:
    """Return top-level backtest kwargs for JoinQuant-style A-share stocks."""

    return {
        "pos_type": JOINQUANT_ASHARE_POLICY.position_type,
        "exchange_kwargs": joinquant_ashare_exchange_kwargs(strict_price_limit=strict_price_limit),
    }


def _stable_semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rdagent_ashare_semantic_contract(*, strict_price_limit: bool = True) -> dict[str, Any]:
    """Return the Qlib-owned A-share semantic contract consumed by RD-Agent.

    RD-Agent may use this payload to guide hypothesis, factor, and model
    generation, but the executable backtest semantics remain owned here.
    """

    policy = JOINQUANT_ASHARE_POLICY
    schema_version = "qlib_ashare_semantic_contract.v1"
    market_semantics = {
        "market": "china_a_share",
        "region": "cn",
        "data_frequency": "day",
        "trade_unit": policy.trade_unit,
        "position_type": policy.position_type,
        "settlement_rule": "t_plus_1_stock",
        "same_day_sell_policy": "shares_bought_today_are_unsellable_until_day_commit",
        "deal_price": policy.deal_price,
        "limit_threshold": JOINQUANT_ASHARE_LIMIT_THRESHOLD,
        "limit_threshold_aliases": sorted(JOINQUANT_ASHARE_ALIASES),
        "price_limit_modes": ["auto", "strict", "board_fallback"],
        "authoritative_limit_fields": [policy.up_limit_field, policy.down_limit_field],
        "board_threshold_fields": {
            "main_board_threshold": policy.main_board_threshold,
            "star_chinext_threshold": policy.star_chinext_threshold,
            "bse_threshold": policy.bse_threshold,
            "chinext_registration_start_date": policy.chinext_registration_start_date,
        },
        "cost_model": {
            "open_cost": policy.open_cost,
            "close_cost": policy.close_cost,
            "close_commission": policy.close_commission,
            "close_tax": policy.close_tax,
            "min_cost": policy.min_cost,
        },
    }
    runtime_surfaces = {
        "policy_class": f"{QLIB_ASHARE_AUTHORITY_COMPONENT}.JoinQuantAshareBacktestPolicy",
        "policy_defaults": asdict(policy),
        "exchange_kwargs": joinquant_ashare_exchange_kwargs(strict_price_limit=strict_price_limit),
        "backtest_kwargs": joinquant_ashare_backtest_kwargs(strict_price_limit=strict_price_limit),
    }
    rdagent_must_not_redefine = [
        "instrument_identity_semantics",
        "universe_membership_semantics",
        "trading_calendar_semantics",
        "transaction_cost_semantics",
        "market_impact_semantics",
        "account_update_semantics",
        "account_valuation_semantics",
        "trade_indicator_semantics",
        "executor_decision_semantics",
        "strategy_order_semantics",
        "supervised_label_semantics",
        "prediction_signal_semantics",
        "signal_ic_semantics",
        "portfolio_risk_semantics",
        "excess_return_semantics",
        "feedback_metric_semantics",
        "benchmark_return_semantics",
        "universe_benchmark_binding_semantics",
        "runtime_handoff_template_binding_semantics",
        "suspension_tradability_semantics",
        "execution_price_semantics",
        "price_adjustment_semantics",
        "price_limit_semantics",
        "order_tradability_semantics",
        "order_fill_amount_semantics",
        "settlement_semantics",
        "cash_settlement_semantics",
        "cash_constraint_semantics",
        "liquidity_capacity_semantics",
        "trade_unit",
        "position_type",
        "settlement_rule",
        "same_day_sell_policy",
        "data_frequency",
        "limit_threshold_aliases",
        "price_limit_modes",
        "authoritative_limit_fields",
        "board_threshold_fields",
        "cost_model",
    ]
    universe_membership_semantics = {
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
    cash_settlement_semantics = {
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
    order_tradability_semantics = {
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
    order_fill_amount_semantics = {
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
        "sell_full_liquidation_rule": "sells_equal_to_current_sellable_amount_keep_full_liquidation_without_round_lot_residual",
        "trade_value_rule": "trade_value_is_final_deal_amount_times_trade_price",
        "cost_rule": "trade_cost_recomputed_after_final_deal_amount",
        "rdagent_rule": "describe_only_do_not_redefine_order_fill_amount_or_clip_sequence",
    }
    market_impact_semantics = {
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
    account_update_semantics = {
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
    account_valuation_semantics = {
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
    trade_indicator_semantics = {
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
    executor_decision_semantics = {
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
    strategy_order_semantics = {
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
    signal_ic_semantics = {
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
    supervised_label_semantics = {
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
        "prompt_wording_rule": "describe_as_qlib_contract_defined_forward_return_label_not_undefined_next_several_days_return",
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
    prediction_signal_semantics = {
        "semantic_name": "a_share_prediction_signal_score",
        "model_signal_authority": "qlib.backtest.signal.ModelSignal",
        "signal_cache_authority": "qlib.backtest.signal.SignalWCache",
        "signal_interface_authority": "qlib.backtest.signal.Signal.get_signal",
        "signal_record_authority": "qlib.workflow.record_temp.SignalRecord",
        "strategy_consumption_authority": "qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy.generate_trade_decision",
        "prediction_artifact": "pred.pkl",
        "prediction_column": "score",
        "model_predict_rule": "model_predict_output_is_prediction_score_not_realized_or_executable_return",
        "series_prediction_rule": "series_prediction_is_saved_as_score_column",
        "dataframe_prediction_rule": "first_prediction_column_is_used_when_prediction_is_dataframe",
        "resample_rule": "SignalWCache_uses_last_signal_between_decision_start_and_end",
        "strategy_ranking_rule": "TopkDropoutStrategy_sorts_prediction_scores_descending_for_candidate_selection",
        "missing_signal_rule": "missing_signal_returns_empty_TradeDecisionWO",
        "label_alignment_rule": "prediction_score_is_trained_against_qlib_owned_LABEL0_without_redefining_return_horizon",
        "prompt_wording_rule": "describe_as_prediction_signal_score_for_LABEL0_not_realized_future_return_or_guaranteed_portfolio_return",
        "rdagent_prompt_paths": [
            "rdagent/scenarios/qlib/experiment/prompts.yaml",
            "rdagent/scenarios/qlib/prompts.yaml",
        ],
        "rdagent_rule": "describe_only_do_not_redefine_prediction_signal_score_or_return_realization",
    }
    portfolio_risk_semantics = {
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
    feedback_metric_semantics = {
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
    excess_return_semantics = {
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
    benchmark_return_semantics = {
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
    universe_benchmark_binding_semantics = {
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
    runtime_handoff_template_binding_semantics = {
        "semantic_name": "a_share_rd_agent_runtime_handoff_template_binding",
        "handoff_id": RDAGENT_ASHARE_RUNTIME_HANDOFF_ID,
        "binding_kind": "rdagent_qlib_template_backtest_runtime_kwargs",
        "rdagent_template_paths": [
            "rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml",
            "rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors_sota_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_baseline_factors_model.yaml",
            "rdagent/scenarios/qlib/experiment/model_template/conf_sota_factors_model.yaml",
        ],
        "required_backtest_kwargs": joinquant_ashare_backtest_kwargs(strict_price_limit=strict_price_limit),
        "forbidden_legacy_exchange_kwargs": {
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
        },
        "runtime_rule": "rdagent_templates_must_bind_port_analysis_backtest_to_qlib_runtime_handoff_values",
        "prompt_boundary_rule": "execution_kwargs_remain_runtime_handoff_not_prompt_authority",
        "rdagent_rule": "consume_qlib_runtime_handoff_values_without_redefining_a_share_execution_kwargs",
    }
    semantic_fingerprint = _stable_semantic_fingerprint(
        {
            "schema_version": schema_version,
            "market_semantics": market_semantics,
            "runtime_surfaces": runtime_surfaces,
            "universe_membership_semantics": universe_membership_semantics,
            "cash_settlement_semantics": cash_settlement_semantics,
            "order_tradability_semantics": order_tradability_semantics,
            "order_fill_amount_semantics": order_fill_amount_semantics,
            "market_impact_semantics": market_impact_semantics,
            "account_update_semantics": account_update_semantics,
            "account_valuation_semantics": account_valuation_semantics,
            "trade_indicator_semantics": trade_indicator_semantics,
            "executor_decision_semantics": executor_decision_semantics,
            "strategy_order_semantics": strategy_order_semantics,
            "supervised_label_semantics": supervised_label_semantics,
            "prediction_signal_semantics": prediction_signal_semantics,
            "signal_ic_semantics": signal_ic_semantics,
            "portfolio_risk_semantics": portfolio_risk_semantics,
            "excess_return_semantics": excess_return_semantics,
            "feedback_metric_semantics": feedback_metric_semantics,
            "benchmark_return_semantics": benchmark_return_semantics,
            "universe_benchmark_binding_semantics": universe_benchmark_binding_semantics,
            "runtime_handoff_template_binding_semantics": runtime_handoff_template_binding_semantics,
            "rdagent_must_not_redefine": rdagent_must_not_redefine,
        }
    )
    semantic_boundary = {
        "authority_component": QLIB_ASHARE_AUTHORITY_COMPONENT,
        "consumer_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "authority_rule": "Qlib owns executable JoinQuant-compatible A-share backtest semantics.",
        "consumer_rule": "RD-Agent may consume a bounded research-generation projection of this contract only.",
        "rdagent_allowed_actions": [
            "render_contract_projection_in_research_context",
            "carry_contract_id_schema_version_and_fingerprint_into_generated_evidence",
            "pass_qlib_owned_runtime_kwargs_to_execution_surfaces",
            "fail_closed_when_contract_is_missing_malformed_or_unsupported",
        ],
        "rdagent_forbidden_actions": [
            "redefine_instrument_identity_or_board_mapping",
            "redefine_universe_membership_or_instrument_filtering",
            "redefine_trading_calendar_or_data_frequency",
            "redefine_transaction_cost_model",
            "redefine_suspension_or_tradability_rules",
            "redefine_execution_price_or_frequency",
            "redefine_price_adjustment_or_order_factor",
            "redefine_trade_unit_or_position_type",
            "redefine_price_limit_thresholds_or_authoritative_fields",
            "treat_board_fallback_as_primary_price_limit_authority",
            "redefine_order_tradability_or_limit_checks",
            "redefine_order_fill_amount_or_clip_sequence",
            "redefine_market_impact_or_cost_ratio",
            "redefine_account_position_or_cash_mutation_order",
            "redefine_account_valuation_or_bar_end_refresh",
            "redefine_trade_execution_indicators_or_quality_metrics",
            "redefine_executor_decision_lifecycle_or_nested_execution_order",
            "redefine_strategy_signal_to_order_generation",
            "redefine_supervised_label_expression_or_horizon",
            "redefine_prediction_signal_score_or_return_realization",
            "redefine_signal_ic_or_rank_ic_metrics",
            "redefine_portfolio_risk_analysis_metrics",
            "redefine_benchmark_relative_excess_return_or_cost_treatment",
            "redefine_feedback_metric_paths_or_label_derived_utility_as_qlib_metric",
            "redefine_benchmark_return_series_or_default_benchmark",
            "redefine_universe_benchmark_template_binding_or_cross_alias_market_and_benchmark",
            "redefine_runtime_handoff_or_template_execution_kwargs",
            "redefine_settlement_or_sellable_position_state",
            "redefine_cash_settlement_or_sell_proceeds_availability",
            "redefine_cash_buying_power_or_shorting_policy",
            "redefine_liquidity_or_volume_capacity_policy",
            "redefine_cost_model_or_exchange_kwargs",
            "treat_research_prompt_projection_as_backtest_authority",
            "claim_a_share_alignment_without_qlib_contract_fingerprint",
        ],
    }
    failure_semantics = {
        "missing_contract": "fail_closed",
        "unsupported_schema_version": "fail_closed",
        "missing_required_field": "fail_closed",
        "malformed_contract": "fail_closed",
        "runtime_projection_drift": "fail_closed",
        "claim_without_evidence_fingerprint": "fail_closed",
    }
    prompt_projection_payload = {
        "projection_id": RDAGENT_ASHARE_PROMPT_PROJECTION_ID,
        "projection_schema_version": RDAGENT_ASHARE_PROMPT_PROJECTION_SCHEMA_VERSION,
        "projection_kind": "research_prompt_context_only",
        "contract_id": RDAGENT_ASHARE_CONTRACT_ID,
        "contract_schema_version": schema_version,
        "schema_version": schema_version,
        "source_component": QLIB_ASHARE_AUTHORITY_COMPONENT,
        "consumer_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "semantic_fingerprint": semantic_fingerprint,
        "semantic_boundary": semantic_boundary,
        "failure_semantics": failure_semantics,
        "market_semantics": {
            "market": market_semantics["market"],
            "region": market_semantics["region"],
            "data_frequency": market_semantics["data_frequency"],
            "trade_unit": market_semantics["trade_unit"],
            "position_type": market_semantics["position_type"],
            "settlement_rule": market_semantics["settlement_rule"],
            "limit_threshold": market_semantics["limit_threshold"],
            "authoritative_limit_fields": list(market_semantics["authoritative_limit_fields"]),
        },
        "instrument_identity_semantics": {
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
                    "effective_start": policy.chinext_registration_start_date,
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
        },
        "universe_membership_semantics": universe_membership_semantics,
        "trading_calendar_semantics": {
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
        },
        "transaction_cost_semantics": {
            "semantic_name": "a_share_transaction_cost_structure",
            "cost_model_scope": "qlib_runtime_execution_only",
            "buy_cost_components": ["commission", "minimum_commission_floor"],
            "sell_cost_components": [
                "commission",
                "stamp_tax",
                "minimum_commission_floor",
            ],
            "minimum_fee_rule": "commission_floor_applies_to_nonzero_trade_value",
            "zero_trade_rule": "zero_trade_value_has_zero_cost",
            "market_impact_rule": "optional_impact_cost_is_added_by_runtime_execution",
            "numeric_values_exposure": "runtime_handoff_only_not_prompt_projection",
            "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.calculate_trade_cost",
            "rdagent_rule": "describe_only_do_not_redefine_transaction_cost_model",
        },
        "market_impact_semantics": market_impact_semantics,
        "account_update_semantics": account_update_semantics,
        "account_valuation_semantics": account_valuation_semantics,
        "trade_indicator_semantics": trade_indicator_semantics,
        "executor_decision_semantics": executor_decision_semantics,
        "strategy_order_semantics": strategy_order_semantics,
        "supervised_label_semantics": supervised_label_semantics,
        "prediction_signal_semantics": prediction_signal_semantics,
        "signal_ic_semantics": signal_ic_semantics,
        "portfolio_risk_semantics": portfolio_risk_semantics,
        "excess_return_semantics": excess_return_semantics,
        "feedback_metric_semantics": feedback_metric_semantics,
        "benchmark_return_semantics": benchmark_return_semantics,
        "universe_benchmark_binding_semantics": universe_benchmark_binding_semantics,
        "runtime_handoff_template_binding_semantics": {
            "semantic_name": runtime_handoff_template_binding_semantics["semantic_name"],
            "handoff_id": runtime_handoff_template_binding_semantics["handoff_id"],
            "binding_kind": runtime_handoff_template_binding_semantics["binding_kind"],
            "rdagent_template_paths": list(runtime_handoff_template_binding_semantics["rdagent_template_paths"]),
            "runtime_rule": runtime_handoff_template_binding_semantics["runtime_rule"],
            "prompt_boundary_rule": runtime_handoff_template_binding_semantics["prompt_boundary_rule"],
            "rdagent_rule": runtime_handoff_template_binding_semantics["rdagent_rule"],
        },
        "suspension_tradability_semantics": {
            "semantic_name": "a_share_suspension_tradability",
            "suspension_indicator_field": "$close",
            "suspension_indicator_rule": "missing_close_price_marks_suspended",
            "non_tradable_rule": "suspended_rows_are_not_buyable_or_sellable",
            "limit_flag_projection": "qlib_sets_limit_buy_and_limit_sell_true_for_suspended_rows",
            "authoritative_limit_interaction": "suspension_takes_precedence_over_up_down_limit_fields",
            "missing_limit_bounds_rule": "missing_limit_bounds_are_tolerated_only_when_close_is_missing",
            "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
            "rdagent_rule": "describe_only_do_not_redefine_suspension_or_tradability",
        },
        "execution_price_semantics": {
            "semantic_name": "a_share_daily_close_execution_price",
            "qlib_parameter": "deal_price",
            "execution_price_field": "$close",
            "execution_frequency": "daily_bar_backtest",
            "price_source_authority": "qlib_exchange_deal_price",
            "intraday_execution_rule": "not_intraday_or_auction_simulation",
            "candidate_research_rule": "generated_factors_must_not_assume_intraday_fill_prices",
            "runtime_authority": "qlib.backtest.ashare_semantics.joinquant_ashare_exchange_kwargs",
            "rdagent_rule": "describe_only_do_not_redefine_execution_price_or_frequency",
        },
        "price_adjustment_semantics": {
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
        },
        "price_limit_semantics": {
            "semantic_name": "a_share_price_limit_authority",
            "limit_threshold": market_semantics["limit_threshold"],
            "price_limit_mode": runtime_surfaces["exchange_kwargs"]["ashare_price_limit_mode"],
            "authoritative_limit_fields": list(market_semantics["authoritative_limit_fields"]),
            "field_authority": "provider_up_down_limit_fields",
            "limit_flag_fields": ["limit_buy", "limit_sell"],
            "limit_flag_meaning": "true_flags_mark_direction_not_tradable",
            "buy_limit_rule": "buy_price_at_or_above_up_limit_or_suspended_sets_limit_buy",
            "sell_limit_rule": "sell_price_at_or_below_down_limit_or_suspended_sets_limit_sell",
            "missing_authoritative_fields": ("fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets"),
            "strict_mode_missing_fields_rule": "missing_authoritative_fields_or_non_suspended_bounds_fail_closed",
            "board_fallback_policy": "runtime_compatibility_only_when_authoritative_fields_are_absent",
            "fallback_authority_rule": "board_thresholds_are_runtime_compatibility_fallback_only_not_primary_authority",
            "board_limit_thresholds": {
                "main_board": policy.main_board_threshold,
                "star_chinext": policy.star_chinext_threshold,
                "bse": policy.bse_threshold,
                "chinext_registration_start_date": policy.chinext_registration_start_date,
            },
            "runtime_authority": "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.apply_price_limits",
            "rdagent_rule": "describe_only_do_not_redefine_price_limit_thresholds_or_fields",
        },
        "order_tradability_semantics": order_tradability_semantics,
        "order_fill_amount_semantics": order_fill_amount_semantics,
        "settlement_semantics": {
            "semantic_name": "a_share_t_plus_1_stock_settlement",
            "settlement_rule": market_semantics["settlement_rule"],
            "same_day_sell_policy": market_semantics["same_day_sell_policy"],
            "position_type": market_semantics["position_type"],
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
        },
        "cash_constraint_semantics": {
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
        },
        "cash_settlement_semantics": cash_settlement_semantics,
        "liquidity_capacity_semantics": {
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
        },
        "order_unit_semantics": {
            "semantic_name": "a_share_round_lot",
            "qlib_parameter": "trade_unit",
            "trade_unit": market_semantics["trade_unit"],
            "amount_unit": "share",
            "buy_rounding_rule": "round_buy_amount_down_to_trade_unit_after_cash_and_volume_limits",
            "sell_rounding_rule": "round_sell_amount_down_to_trade_unit_except_full_liquidation",
            "full_liquidation_rule": "sell_all_remaining_position_without_round_lot_residual",
            "factor_adjustment_rule": "apply_order_factor_when_trade_uses_unadjusted_prices",
            "runtime_authority": "qlib.backtest.exchange.Exchange.round_amount_by_trade_unit",
            "rdagent_rule": "describe_only_do_not_redefine_trade_unit_or_round_lot_policy",
        },
    }
    return {
        "schema_version": schema_version,
        "contract_id": RDAGENT_ASHARE_CONTRACT_ID,
        "status": "active",
        "source_component": QLIB_ASHARE_AUTHORITY_COMPONENT,
        "consumer_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
        "relationship": {
            "qlib_role": "executable_backtest_semantic_authority",
            "rdagent_role": "research_candidate_generation_context_consumer",
            "relationship_rule": (
                "RD-Agent may consume Qlib's A-share contract for research generation and evaluation context, "
                "but it must not redefine universe-membership, trading-calendar/data-frequency, trade unit, position, execution-price, price-adjustment, "
                "suspension/tradability, price-limit, order-tradability, order-fill, account-position update, account valuation, trade indicator/execution-quality, executor/trade-decision lifecycle, strategy signal-to-order generation, supervised label, prediction signal, signal IC, portfolio risk analysis, benchmark-relative excess return, feedback metric consumption, benchmark return, universe/benchmark binding, runtime handoff template binding, settlement, cash-settlement, cash/shorting, liquidity/capacity, market-impact, or cost semantics."
            ),
            "fail_closed_on_missing_contract": True,
        },
        "semantic_boundary": semantic_boundary,
        "failure_semantics": failure_semantics,
        "evidence_contract": {
            "semantic_fingerprint": semantic_fingerprint,
            "fingerprint_algorithm": "sha256_json_canonical_v1",
            "fingerprint_scope": [
                "schema_version",
                "market_semantics",
                "runtime_surfaces",
                "universe_membership_semantics",
                "cash_settlement_semantics",
                "order_tradability_semantics",
                "order_fill_amount_semantics",
                "market_impact_semantics",
                "account_update_semantics",
                "account_valuation_semantics",
                "trade_indicator_semantics",
                "executor_decision_semantics",
                "strategy_order_semantics",
                "supervised_label_semantics",
                "prediction_signal_semantics",
                "signal_ic_semantics",
                "portfolio_risk_semantics",
                "excess_return_semantics",
                "feedback_metric_semantics",
                "benchmark_return_semantics",
                "universe_benchmark_binding_semantics",
                "runtime_handoff_template_binding_semantics",
                "rdagent_must_not_redefine",
            ],
            "rdagent_required_evidence_fields": [
                "qlib_contract_id",
                "qlib_contract_schema_version",
                "qlib_contract_fingerprint",
                "qlib_source_component",
                "qlib_semantic_authority",
            ],
        },
        "projection_contract": {
            "rdagent_prompt_projection_fields": [
                "contract_id",
                "schema_version",
                "source_component",
                "consumer_component",
                "semantic_boundary",
                "failure_semantics",
                "evidence_contract.semantic_fingerprint",
                "market_semantics.market",
                "market_semantics.region",
                "market_semantics.data_frequency",
                "market_semantics.trade_unit",
                "market_semantics.position_type",
                "market_semantics.settlement_rule",
                "market_semantics.limit_threshold",
                "market_semantics.authoritative_limit_fields",
                "instrument_identity_semantics",
                "universe_membership_semantics",
                "trading_calendar_semantics",
                "transaction_cost_semantics",
                "market_impact_semantics",
                "account_update_semantics",
                "account_valuation_semantics",
                "trade_indicator_semantics",
                "executor_decision_semantics",
                "strategy_order_semantics",
                "supervised_label_semantics",
                "prediction_signal_semantics",
                "signal_ic_semantics",
                "portfolio_risk_semantics",
                "excess_return_semantics",
                "feedback_metric_semantics",
                "benchmark_return_semantics",
                "universe_benchmark_binding_semantics",
                "runtime_handoff_template_binding_semantics",
                "suspension_tradability_semantics",
                "execution_price_semantics",
                "price_adjustment_semantics",
                "price_limit_semantics",
                "order_tradability_semantics",
                "order_fill_amount_semantics",
                "settlement_semantics",
                "cash_settlement_semantics",
                "cash_constraint_semantics",
                "liquidity_capacity_semantics",
                "order_unit_semantics",
            ],
            "rdagent_prompt_forbidden_fields": [
                "runtime_surfaces.policy_defaults",
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
                "market_semantics.cost_model",
            ],
        },
        "prompt_projection_payload": prompt_projection_payload,
        "runtime_handoff_contract": {
            "handoff_id": RDAGENT_ASHARE_RUNTIME_HANDOFF_ID,
            "handoff_kind": "qlib_owned_execution_kwargs",
            "authority_component": QLIB_ASHARE_AUTHORITY_COMPONENT,
            "consumer_component": RDAGENT_ASHARE_CONSUMER_COMPONENT,
            "source_fingerprint": semantic_fingerprint,
            "payload_paths": [
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
            ],
            "forbidden_prompt_paths": [
                "runtime_surfaces.policy_defaults",
                "runtime_surfaces.exchange_kwargs",
                "runtime_surfaces.backtest_kwargs",
                "market_semantics.cost_model",
            ],
            "mutation_policy": "pass_through_only",
            "consumer_obligations": [
                "preserve_contract_id_schema_version_and_fingerprint",
                "preserve_qlib_source_component",
                "do_not_mutate_runtime_payload_values",
                "fail_closed_on_missing_payload_or_fingerprint",
            ],
            "template_runtime_binding": runtime_handoff_template_binding_semantics,
        },
        "market_semantics": market_semantics,
        "runtime_surfaces": runtime_surfaces,
        "rdagent_must_not_redefine": rdagent_must_not_redefine,
    }


def build_joinquant_ashare_policy(
    options: Mapping[str, Any] | None = None,
) -> JoinQuantAshareBacktestPolicy:
    if options is None:
        return JOINQUANT_ASHARE_POLICY
    allowed = {field.name for field in fields(JoinQuantAshareBacktestPolicy)}
    unknown = sorted(str(key) for key in options if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown joinquant_ashare policy options: {unknown}")
    return JoinQuantAshareBacktestPolicy(**dict(options))
