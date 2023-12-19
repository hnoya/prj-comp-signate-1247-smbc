from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.model_selection._split import _BaseKFold


validate_functions: dict[str, _BaseKFold] = {
    "KFold": KFold,
    "GroupKFold": GroupKFold,
    "StratifiedKFold": StratifiedKFold,
    "StratifiedGroupKFold": StratifiedGroupKFold,
}


def get_fold_maker(validate_type: str, n_splits: int, **kwargs) -> _BaseKFold:
    fold_maker = validate_functions[validate_type]
    return fold_maker(n_splits=n_splits, **kwargs)


def get_fold_maker_split(
    validate_type: str,
    validate_func: _BaseKFold,
    X: pd.DataFrame,
    y: Optional[Union[list[int], list[float]]] = None,
    groups: Optional[Union[list[int], list[str]]] = None,
):
    if validate_type == "kfold":
        foldmaker = validate_func.split(X)
    elif validate_type == "GroupKFold":
        foldmaker = validate_func.split(X, y, groups)
    elif validate_type == "StratifiedKFold":
        foldmaker = validate_func.split(X, y)
    elif validate_type == "StratifiedGroupKFold":
        foldmaker = validate_func.split(X, y, groups)
    else:
        raise NotImplementedError()


def validate(
    validate_type: str,
    df: pd.DataFrame,
    n_splits: int,
    y: Optional[Union[list[int], list[float]]] = None,
    groups: Optional[Union[list[int], list[str]]] = None,
    **kwargs
) -> list[int]:
    validate_func = get_fold_maker(validate_type, n_splits, **kwargs)
    foldmaker = validate_func.split(df, y, groups)
    fold_idxs = np.zeros((len(df)))
    for fold_idx, (train_idxs, valid_idxs) in enumerate(foldmaker):
        fold_idxs[valid_idxs] = fold_idx
    return fold_idxs.astype(int).tolist()
