from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_PATHS = (
    REPO_ROOT / ".github",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / "qlib",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
)
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
    }
)
RETIRED_TERMS = (
    "".join(("qlib", "_data", "_simple")),
    "".join(("cn", "_data", "_simple")),
    "".join(("simple", "_provider", "_uri")),
    "".join(("simple", "_provider")),
    "".join(("sync", "-", "simple", "-", "data")),
    "".join(("sync", "_simple")),
    "".join(("verify", "_sync", "_simple")),
    "".join(("qlib", "_simple", "_cache")),
)


def _iter_checked_files() -> list[Path]:
    files: list[Path] = []
    for checked_path in CHECKED_PATHS:
        if checked_path.is_file():
            files.append(checked_path)
            continue
        if not checked_path.exists():
            continue
        for path in checked_path.rglob("*"):
            if not path.is_file():
                continue
            if SKIP_DIR_NAMES.intersection(path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return sorted(files)


def _load_get_data_module(monkeypatch):
    qlib_stub = types.ModuleType("qlib")
    qlib_stub.__path__ = []
    qlib_stub.__version__ = "0.0.0"
    utils_stub = types.ModuleType("qlib.utils")
    utils_stub.exists_qlib_data = lambda _: False
    monkeypatch.setitem(sys.modules, "qlib", qlib_stub)
    monkeypatch.setitem(sys.modules, "qlib.utils", utils_stub)

    spec = importlib.util.spec_from_file_location(
        "_qlib_tests_data_boundary",
        REPO_ROOT / "qlib" / "tests" / "data.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pit_collector_module(monkeypatch):
    qlib_stub = types.ModuleType("qlib")
    qlib_stub.__path__ = []
    utils_stub = types.ModuleType("qlib.utils")
    utils_stub.code_to_fname = lambda code: code
    monkeypatch.setitem(sys.modules, "qlib", qlib_stub)
    monkeypatch.setitem(sys.modules, "qlib.utils", utils_stub)

    baostock_stub = types.ModuleType("baostock")
    monkeypatch.setitem(sys.modules, "baostock", baostock_stub)

    collector_utils_stub = types.ModuleType("data_collector.utils")
    collector_utils_stub.get_hs_stock_symbols = lambda: []

    def _external_calendar():
        raise AssertionError("external calendar source should not be used")

    collector_utils_stub.get_calendar_list = _external_calendar
    monkeypatch.setitem(sys.modules, "data_collector.utils", collector_utils_stub)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))

    spec = importlib.util.spec_from_file_location(
        "_pit_collector_boundary",
        REPO_ROOT / "scripts" / "data_collector" / "pit" / "collector.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_fixture_name_maps_to_public_archive_stem(monkeypatch) -> None:
    module = _load_get_data_module(monkeypatch)
    archive_stem = "".join(("qlib", "_data", "_simple"))

    file_name = module.GetData.qlib_data_file_name(
        name=module.SMOKE_FIXTURE_DATASET_NAME,
        version=None,
        interval="1d",
        region="cn",
        qlib_version="latest",
    )

    assert file_name == f"v2/{archive_stem}_cn_1d_latest.zip"


def test_pit_normalize_accepts_local_calendar_without_external_fetch(monkeypatch) -> None:
    module = _load_pit_collector_module(monkeypatch)
    local_calendar = ["2019-01-02", "2019-01-03"]

    normalize = module.PitNormalize(calendar_list=local_calendar)

    assert [str(date.date()) for date in normalize._calendar_list] == local_calendar


def test_retired_qlib_shortcut_terms_stay_out_of_public_paths() -> None:
    violations: list[str] = []
    for path in _iter_checked_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in RETIRED_TERMS:
            if term in text:
                violations.append(f"{path.relative_to(REPO_ROOT)} contains {term!r}")

    assert violations == []
