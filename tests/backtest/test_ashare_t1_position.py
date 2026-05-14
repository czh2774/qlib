from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ASHARE_SEMANTICS_PATH = REPO_ROOT / "qlib/backtest/ashare_semantics.py"
POSITION_PATH = REPO_ROOT / "qlib/backtest/position.py"
EXCHANGE_PATH = REPO_ROOT / "qlib/backtest/exchange.py"


class StubOrder:
    SELL = 0
    BUY = 1

    def __init__(self, stock_id: str, direction: int) -> None:
        self.stock_id = stock_id
        self.direction = direction


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_position_stubs() -> None:
    qlib_pkg = types.ModuleType("qlib")
    qlib_pkg.__path__ = []
    backtest_pkg = types.ModuleType("qlib.backtest")
    backtest_pkg.__path__ = []
    data_pkg = types.ModuleType("qlib.data")
    data_pkg.__path__ = []
    data_module = types.ModuleType("qlib.data.data")
    data_module.D = types.SimpleNamespace()
    decision_module = types.ModuleType("qlib.backtest.decision")
    decision_module.Order = StubOrder
    sys.modules.setdefault("qlib", qlib_pkg)
    sys.modules.setdefault("qlib.backtest", backtest_pkg)
    sys.modules.setdefault("qlib.data", data_pkg)
    sys.modules["qlib.data.data"] = data_module
    sys.modules["qlib.backtest.decision"] = decision_module


def _load_position_module():
    _install_position_stubs()
    return _load_module("qlib.backtest.position", POSITION_PATH)


def test_ashare_position_keeps_intraday_buys_unsellable_until_day_commit() -> None:
    position_module = _load_position_module()
    position = position_module.AsharePosition(cash=10_000)
    order = StubOrder("SH600000", StubOrder.BUY)
    position.update_order(order, trade_val=1000.0, cost=0.0, trade_price=10.0)

    assert position.get_stock_amount("SH600000") == 100.0
    assert position.get_sellable_amount("SH600000") == 0.0

    position.add_count_all("1min")
    assert position.get_sellable_amount("SH600000") == 0.0

    position.add_count_all("day")
    assert position.get_sellable_amount("SH600000") == 100.0


def test_ashare_position_tracks_partial_sellable_reduction() -> None:
    position_module = _load_position_module()
    position = position_module.AsharePosition(
        cash=0.0,
        position_dict={
            "SH600000": {
                "amount": 1000.0,
                "price": 10.0,
                "sellable_amount": 300.0,
            }
        },
    )
    order = StubOrder("SH600000", StubOrder.SELL)

    position.update_order(order, trade_val=2000.0, cost=0.0, trade_price=10.0)

    assert position.get_stock_amount("SH600000") == 800.0
    assert position.get_sellable_amount("SH600000") == 100.0


def test_default_position_keeps_legacy_full_amount_sellability() -> None:
    position_module = _load_position_module()
    position = position_module.Position(
        cash=0.0,
        position_dict={"SH600000": {"amount": 1000.0, "price": 10.0}},
    )

    assert position.get_sellable_amount("SH600000") == 1000.0


def test_exchange_source_clips_sells_by_sellable_position_amount() -> None:
    source = EXCHANGE_PATH.read_text(encoding="utf-8")

    assert "position.get_sellable_amount(order.stock_id)" in source


def test_joinquant_ashare_backtest_kwargs_enable_ashare_position() -> None:
    ashare_semantics = _load_module(
        "ashare_semantics_t1_under_test",
        ASHARE_SEMANTICS_PATH,
    )

    kwargs = ashare_semantics.joinquant_ashare_backtest_kwargs()

    assert kwargs["pos_type"] == "AsharePosition"
    assert kwargs["exchange_kwargs"]["limit_threshold"] == "joinquant_ashare"
    assert np.isclose(kwargs["exchange_kwargs"]["open_cost"], 0.0003)
