from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from configs.training.config_training import Features, Spline


def row_product() -> FunctionTransformer:
    return FunctionTransformer(np.prod, kw_args={"axis": 1, "keepdims": True}, validate=True)


def build_preprocessor(features: Features, spline: Spline | None = None) -> ColumnTransformer:
    spline_columns = list(spline.columns) if spline else []
    linear_columns = [c for c in [*features.numeric, *features.boolean] if c not in spline_columns]
    transformers: list[tuple[str, Any, list[str]]] = [
        ("categorical", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="error"),
         list(features.categorical)),
        ("linear", StandardScaler(), linear_columns),
    ]
    for product in features.products:
        transformers.append((product.name, make_pipeline(row_product(), StandardScaler()), list(product.columns)))
    if spline:
        transformers.append(("spline", SplineTransformer(n_knots=spline.n_knots, degree=spline.degree,
                                                         include_bias=False), spline_columns))
    return ColumnTransformer(transformers, remainder="drop")
