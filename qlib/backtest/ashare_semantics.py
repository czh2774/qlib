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
        "trade_unit",
        "position_type",
        "settlement_rule",
        "same_day_sell_policy",
        "limit_threshold_aliases",
        "price_limit_modes",
        "authoritative_limit_fields",
        "board_threshold_fields",
        "cost_model",
    ]
    semantic_fingerprint = _stable_semantic_fingerprint(
        {
            "schema_version": schema_version,
            "market_semantics": market_semantics,
            "runtime_surfaces": runtime_surfaces,
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
            "redefine_trade_unit_or_position_type",
            "redefine_price_limit_thresholds_or_authoritative_fields",
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
                {"match": "BJ*|SH8*|SH4*|SH9*|SZ8*|SZ4*|SZ9*", "board": "beijing_stock_exchange"},
                {"match": "fallback", "board": "main_board"},
            ],
            "price_limit_dependency": "board_identity_is_runtime_fallback_only_when_authoritative_limit_fields_absent",
            "runtime_authority": "qlib.backtest.ashare_semantics.normalize_ashare_instrument",
            "board_classification_authority": (
                "qlib.backtest.ashare_semantics.JoinQuantAshareBacktestPolicy.limit_threshold_for_instrument"
            ),
            "rdagent_rule": "describe_only_do_not_redefine_instrument_or_board_identity",
        },
        "price_limit_semantics": {
            "limit_threshold": market_semantics["limit_threshold"],
            "price_limit_mode": runtime_surfaces["exchange_kwargs"]["ashare_price_limit_mode"],
            "authoritative_limit_fields": list(market_semantics["authoritative_limit_fields"]),
            "field_authority": "provider_up_down_limit_fields",
            "missing_authoritative_fields": ("fail_closed_in_strict_mode_else_qlib_board_fallback_for_legacy_datasets"),
            "board_fallback_policy": "runtime_compatibility_only_when_authoritative_fields_are_absent",
            "board_limit_thresholds": {
                "main_board": policy.main_board_threshold,
                "star_chinext": policy.star_chinext_threshold,
                "bse": policy.bse_threshold,
                "chinext_registration_start_date": policy.chinext_registration_start_date,
            },
            "rdagent_rule": "describe_only_do_not_redefine_price_limit_thresholds_or_fields",
        },
        "settlement_semantics": {
            "settlement_rule": market_semantics["settlement_rule"],
            "same_day_sell_policy": market_semantics["same_day_sell_policy"],
            "position_type": market_semantics["position_type"],
            "runtime_authority": "qlib.backtest.position.AsharePosition",
            "rdagent_rule": "describe_only_do_not_redefine_position_or_settlement",
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
                "but it must not redefine trade unit, position, price-limit, or cost semantics."
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
                "market_semantics.trade_unit",
                "market_semantics.position_type",
                "market_semantics.settlement_rule",
                "market_semantics.limit_threshold",
                "market_semantics.authoritative_limit_fields",
                "instrument_identity_semantics",
                "price_limit_semantics",
                "settlement_semantics",
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
