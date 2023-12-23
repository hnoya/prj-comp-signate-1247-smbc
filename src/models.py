import os
import pickle

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

import catboost
import lightgbm

from sklearn.metrics import f1_score

def macro_f1(y_pred, data):
    y_true = data.get_label()
    score = f1_score(y_true, np.argmax(y_pred, axis=1), average="macro")
    return 'macro_f1', score, True


def train_catboost(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    output_path: str = "../models",
) -> None:
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = catboost.Pool(
        X_train, label=y_train, cat_features=categorical_features
    )
    eval_data = catboost.Pool(X_valid, label=y_valid, cat_features=categorical_features)
    # see: https://catboost.ai/en/docs/concepts/loss-functions-ranking#usage-information
    ctb_params = {
        "objective": "MultiClass",  # "MultiClass",
        "loss_function": "CrossEntropy",  # "CrossEntropy",
        "eval_metric": "TotalF1:average=Macro;use_weights=false", # ;use_weights=false",
        "num_boost_round": 10_000,
        "early_stopping_rounds": 1_000,
        "learning_rate": 0.01,
        "verbose": 1_000,
        "random_seed": seed,
        "task_type": "GPU",
        # "used_ram_limit": "32gb",
        "class_weights": [1000/3535, 1000/15751, 1000/698],
    }

    model = catboost.CatBoost(ctb_params)
    model.fit(train_data, eval_set=[eval_data], use_best_model=True, plot=False)
    pickle.dump(
        model, open(os.path.join(output_path, "ctb_fold{}.ctbmodel".format(fold)), "wb")
    )


def eval_catboost(
    X_valid: pd.DataFrame,
    fold: int,
    categorical_features: list[str],
    model_path: str = "../models",
) -> np.ndarray:
    model = pickle.load(
        open(os.path.join(model_path, "ctb_fold{}.ctbmodel".format(fold)), "rb")
    )
    y_pred = model.predict(catboost.Pool(X_valid, cat_features=categorical_features))
    return y_pred


def predict_catboost(
    X_test: pd.DataFrame,
    folds: list[int],
    categorical_features: list[str],
    model_path: str = "../models",
):
    y_pred = np.zeros((X_test.shape[0], 3), dtype="float32")
    for fold in folds:
        model = pickle.load(
            open(os.path.join(model_path, "ctb_fold{}.ctbmodel".format(fold)), "rb")
        )
        y_pred += model.predict(
            catboost.Pool(X_test, cat_features=categorical_features)
        ) / len(folds)
    return y_pred


def train_lightgbm(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    output_path: str = "../models",
) -> None:
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = lightgbm.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features,
        weight=compute_sample_weight(class_weight='balanced', y=y_train.values)
    )
    valid_data = lightgbm.Dataset(
        X_valid, label=y_valid, categorical_feature=categorical_features
    )
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "custom",
        "learning_rate": 0.01,
        "seed": seed,
        "verbose": -1,
    }
    model = lightgbm.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, valid_data],
        categorical_feature=categorical_features,
        num_boost_round=10000,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=1000, verbose=True),
            lightgbm.log_evaluation(1000),
        ],
        feval=macro_f1,
    )
    pickle.dump(
        model, open(os.path.join(output_path, "lgb_fold{}.lgbmodel".format(fold)), "wb")
    )


def eval_lightgbm(
    X_valid: pd.DataFrame, fold: int, model_path: str = "../models"
) -> np.ndarray:
    model = pickle.load(
        open(os.path.join(model_path, "lgb_fold{}.lgbmodel".format(fold)), "rb")
    )
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    return y_pred


def predict_lightgbm(
    X_test: pd.DataFrame, folds: list[int], model_path: str = "../models"
) -> np.ndarray:
    y_pred = np.zeros((X_test.shape[0], 3), dtype="float32")
    for fold in folds:
        model = pickle.load(
            open(os.path.join(model_path, "lgb_fold{}.lgbmodel".format(fold)), "rb")
        )
        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
    return y_pred


def eval_folds(
    train: pd.DataFrame,
    n_fold: int,
    seed: int,
    model_type: str,
    label_col: str,
    not_use_cols: list[str],
    cat_cols: list[str],
    model_path: str = "../models",
) -> np.ndarray:
    train["pred"] = 0
    categorical_features = cat_cols
    for fold in range(n_fold):
        _, valid_df = train.loc[train["fold"] != fold], train.loc[train["fold"] == fold]
        use_columns = [
            col for col in valid_df.columns.tolist() if col not in not_use_cols
        ]
        X_valid = valid_df[use_columns]
        # y_valid = valid_df["score"]
        if model_type == "lgb":
            y_pred = eval_lightgbm(X_valid, fold, model_path)
        elif model_type == "ctb":
            y_pred = eval_catboost(X_valid, fold, categorical_features, model_path)
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            y_pred = eval_rec(valid_df, fold, _model_type, model_path)
        else:
            raise NotImplementedError()
        train.loc[train["fold"] == fold, "pred"] = y_pred
    return train["pred"].values


def eval_folds_v2(
    train: pd.DataFrame,
    folds: list[int],
    seed: int,
    model_type: str,
    label_col: str,
    not_use_cols: list[str],
    cat_cols: list[str],
    model_path: str = "../models",
) -> np.ndarray:
    # train["pred"] = 0
    oof_preds = np.zeros((len(train), 3))
    categorical_features = cat_cols
    for fold in folds:
        _, valid_df = train.loc[train["fold"] != fold], train.loc[train["fold"] == fold]
        use_columns = [
            col for col in valid_df.columns.tolist() if col not in not_use_cols
        ]
        X_valid = valid_df[use_columns]
        # y_valid = valid_df["score"]
        if model_type == "lgb":
            y_pred = eval_lightgbm(X_valid, fold, model_path)
        elif model_type == "ctb":
            y_pred = eval_catboost(X_valid, fold, categorical_features, model_path)
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            y_pred = eval_rec(valid_df, fold, _model_type, model_path)
        else:
            raise NotImplementedError()
        # train.loc[train["fold"] == fold] = y_pred
        oof_preds[train.loc[train["fold"] == fold].index.to_numpy(), :] = y_pred
    return oof_preds # train["pred"].values


def train_folds(
    train: pd.DataFrame,
    n_fold: int,
    seed: int,
    model_type: str,
    label_col: str,
    not_use_cols: list[str],
    cat_cols: list[str],
    output_path: str = "../models",
) -> None:
    for fold in range(n_fold):
        train_df, valid_df = (
            train.loc[train["fold"] != fold],
            train.loc[train["fold"] == fold],
        )
        use_columns = [
            col for col in train_df.columns.tolist() if col not in not_use_cols
        ]
        X_train = train_df[use_columns]
        y_train = train_df[label_col]
        X_valid = valid_df[use_columns]
        y_valid = valid_df[label_col]

        categorical_features = cat_cols
        if model_type == "lgb":
            train_lightgbm(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type == "ctb":
            train_catboost(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            train_rec(train_df, _model_type, fold, seed, output_path)
        else:
            raise NotImplementedError(model_type)


def train_folds_v2(
    train: pd.DataFrame,
    folds: list[int],
    seed: int,
    model_type: str,
    label_col: str,
    not_use_cols: list[str],
    cat_cols: list[str],
    output_path: str = "../models",
) -> None:
    for fold in folds:
        train_df, valid_df = (
            train.loc[train["fold"] != fold],
            train.loc[train["fold"] == fold],
        )
        use_columns = [
            col for col in train_df.columns.tolist() if col not in not_use_cols
        ]
        X_train = train_df[use_columns]
        y_train = train_df[label_col]
        X_valid = valid_df[use_columns]
        y_valid = valid_df[label_col]

        categorical_features = cat_cols
        if model_type == "lgb":
            train_lightgbm(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type == "ctb":
            train_catboost(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                output_path,
            )
        elif model_type[:4] == "rec_":
            _model_type = model_type.split("rec_")[1]
            train_rec(train_df, _model_type, fold, seed, output_path)
        else:
            raise NotImplementedError(model_type)
