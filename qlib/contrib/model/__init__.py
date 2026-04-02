# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import importlib


OPTIONAL_MODEL_IMPORT_FAILURES = {}


def _record_optional_import_failure(group: str, exc: ImportError) -> None:
    OPTIONAL_MODEL_IMPORT_FAILURES[group] = {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _is_optional_pytorch_import_error(exc: ImportError) -> bool:
    missing_name = str(getattr(exc, "name", "") or "")
    if isinstance(exc, ModuleNotFoundError):
        return (
            not missing_name
            or missing_name == "torch"
            or missing_name.startswith("torch.")
            or missing_name.startswith("qlib.contrib.model.pytorch_")
        )

    message = str(exc)
    return (
        missing_name == "torch"
        or missing_name.startswith("torch.")
        or "libc10.so" in message
        or "static TLS block" in message
    )


def _load_pytorch_models():
    try:
        alstm = importlib.import_module(".pytorch_alstm", __name__)
        gats = importlib.import_module(".pytorch_gats", __name__)
        gru = importlib.import_module(".pytorch_gru", __name__)
        lstm = importlib.import_module(".pytorch_lstm", __name__)
        nn = importlib.import_module(".pytorch_nn", __name__)
        tabnet = importlib.import_module(".pytorch_tabnet", __name__)
        sfm = importlib.import_module(".pytorch_sfm", __name__)
        tcn = importlib.import_module(".pytorch_tcn", __name__)
        add = importlib.import_module(".pytorch_add", __name__)
    except ImportError as exc:
        if not _is_optional_pytorch_import_error(exc):
            raise
        _record_optional_import_failure("pytorch", exc)
        print(
            "ImportError. PyTorch models are skipped "
            "(optional: maybe installing or repairing pytorch can fix it)."
        )
        return (None,) * 9, ()

    classes = (
        alstm.ALSTM,
        gats.GATs,
        gru.GRU,
        lstm.LSTM,
        nn.DNNModelPytorch,
        tabnet.TabnetModel,
        sfm.SFM_Model,
        tcn.TCN,
        add.ADD,
    )
    return classes, classes


try:
    from .catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = None
    print("ModuleNotFoundError. CatBoostModel are skipped. (optional: maybe installing CatBoostModel can fix it.)")
try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print(
        "ModuleNotFoundError. DEnsembleModel and LGBModel are skipped. (optional: maybe installing lightgbm can fix it.)"
    )
try:
    from .xgboost import XGBModel
except ModuleNotFoundError:
    XGBModel = None
    print("ModuleNotFoundError. XGBModel is skipped(optional: maybe installing xgboost can fix it).")
try:
    from .linear import LinearModel
except ModuleNotFoundError:
    LinearModel = None
    print("ModuleNotFoundError. LinearModel is skipped(optional: maybe installing scipy and sklearn can fix it).")

(
    ALSTM,
    GATs,
    GRU,
    LSTM,
    DNNModelPytorch,
    TabnetModel,
    SFM_Model,
    TCN,
    ADD,
), pytorch_classes = _load_pytorch_models()

all_model_classes = (CatBoostModel, DEnsembleModel, LGBModel, XGBModel, LinearModel) + pytorch_classes
