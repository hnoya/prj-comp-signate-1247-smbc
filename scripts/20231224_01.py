# %%
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# %%
import sys
sys.path.append("../")

from src.utils import seed_everything, apply_sigmoid
from src.convert import apply_le
from src.validate import validate
from src.models import train_folds_v2, eval_folds_v2, predict_catboost, predict_lightgbm
from src.features import make_ratio, make_statvalue

# %%
@dataclass
class Config:
    experiment_name: str = "20231224_01"
    n_fold: int = 5
    seed: int = 0

@dataclass
class CSVPath:
    train: str = "/work/data/train.csv"
    test: str = "/work/data/test.csv"
    submission: str = "/work/data/sample_submission.csv"


seed_everything(Config.seed)
os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)

# %%
train = pd.read_csv(CSVPath.train).drop(["Unnamed: 0"], axis=1)
test = pd.read_csv(CSVPath.test).drop(["Unnamed: 0"], axis=1)

train = train.drop(["boroname", "nta_name", "spc_latin"], axis=1)
test = test.drop(["boroname", "nta_name", "spc_latin"], axis=1)

# %%
import time
import datetime

def convert_time_str_to_unix_float(time_str: str) -> float:
    return time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple())

def convert_steward_str_to_float(steward_str: str) -> float:
    if pd.isnull(steward_str):
        return steward_str
    else:
        if steward_str == "1or2":
            return 1.5
        elif steward_str == "3or4":
            return 3.5
        elif steward_str == "4orMore":
            return 4.5
        else:
            raise NotImplementedError(f"{steward_str} is unkown case to convert float.")


def apply_single_fe(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at_float"] = df["created_at"].apply(lambda x: convert_time_str_to_unix_float(x))
    df = df.drop(["created_at"], axis=1)

    df["steward_float"] = df["steward"].apply(lambda x: convert_steward_str_to_float(x))
    df = df.drop(["steward"], axis=1)
    return df


def convert_rawdata_to_traindata(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = apply_single_fe(train)
    test = apply_single_fe(test)
    
    train_and_test = pd.concat([train.drop(["health"], axis=1), test], ignore_index=True)
    for col in ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
                "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]:
        train_and_test = apply_le(train_and_test, col, keep_nan=False)
        train[col] = train_and_test[col].iloc[:len(train)].tolist()
        test[col] = train_and_test[col].iloc[-len(test):].tolist()

    return train, test

# %%
train, test = convert_rawdata_to_traindata(train, test)
train["fold"] = validate("StratifiedKFold", train, Config.n_fold, train["health"].tolist(), random_state=Config.seed, shuffle=True)

# %%
train.head(3)

# %%
test.head(3)

# %%
def add_te(train, test, col):
    for class_idx in range(3):
        train[f"health_is_{class_idx}"] = (train["health"].to_numpy() == class_idx).astype(int).tolist()
        ts = pd.Series(np.empty(train.shape[0]), index=train.index)

        agg_df = train.groupby(col).agg({f"health_is_{class_idx}": ['sum', 'count']})
        for fold_idx in range(Config.n_fold):
            holdout_df = train.loc[train["fold"] == fold_idx]
            holdout_agg_df = holdout_df.groupby(col).agg({f"health_is_{class_idx}": ['sum', 'count']})
            train_agg_df = agg_df - holdout_agg_df
            oof_ts = holdout_df.apply(lambda row: train_agg_df.loc[getattr(row, col)][(f"health_is_{class_idx}", 'sum')] \
                                                / (train_agg_df.loc[getattr(row, col)][(f"health_is_{class_idx}", 'count')] + 1), axis=1)
            ts[oof_ts.index] = oof_ts
        
        ts.name = f"health_is_{class_idx}_te_by_{col}"
        train = train.join(ts)

        all_train_df = train.groupby(
            col, as_index=False
        ).agg({f"health_is_{class_idx}": ["sum", "count"]})
        te = all_train_df.apply(lambda row: all_train_df.loc[getattr(row, col)][(f"health_is_{class_idx}", 'sum')] \
                                                / (all_train_df.loc[getattr(row, col)][(f"health_is_{class_idx}", 'count')] + 1), axis=1)
        te = te.rename(columns={f"health_is_{class_idx}": f"health_is_{class_idx}_te_by_{col}"})
        te_dict = te.to_dict()[f"health_is_{class_idx}_te_by_{col}"]
        test[f"health_is_{class_idx}_te_by_{col}"] = test[col].apply(lambda x: te_dict[x])

        train = train.drop([f"health_is_{class_idx}"], axis=1)
    return train, test

# %%
#train, test = add_te(train, test, "spc_common")
#train, test = add_te(train, test, "boro_ct")

# %%
train.head(3)

# %%
test.head(3)

# %% [markdown]
# - ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
# - ["tree_dbh", "steward", "create_at_float"]
# - ['mean', 'std', 'max', 'min']
# 

# %%
import warnings


cat_cols = ["spc_common"]
qua_cols = ["created_at_float"]
type_cols = ["mean"]
cols = cat_cols + qua_cols
# df = pd.concat([train[cols], test[cols]], ignore_index=True)
# df = make_statvalue(df, cat_cols, qua_cols, type_cols)
# add_cols = [col for col in df.columns.tolist() if col not in cols]
# for col in add_cols:
#     train[col] = df[col].iloc[:len(train)].reset_index(drop=True).tolist()
#     test[col] = df[col].iloc[-len(test):].reset_index(drop=True).tolist()
holdout_trains = []
for fold_idx in range(Config.n_fold):
    holdout_train = train.loc[train["fold"] == fold_idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        holdout_train = make_statvalue(holdout_train, cat_cols, qua_cols, type_cols)
    holdout_trains.append(holdout_train)

train = pd.concat(holdout_trains).sort_index()
test = make_statvalue(test, cat_cols, qua_cols, type_cols)

# %%
train.head(3)

# %%
test.head(3)

# %% [markdown]
# importance (20231222_01)
# spc_common, boro_ct, nta, st_assem, cb_num, problems, st_senate, cncldist, zip_city, user_type, guards, sidewalk, borocode, curb_loc
# 
# create_at_float, tree_dbh, steward_float

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]


train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}"
)


oof_preds_lgb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}"
)

score = f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
print(score)

# %% [markdown]
# - 0: baseline (20231223_03): 0.35466775297446107

# %%
train["pred"] = np.argmax(oof_preds_lgb, axis=1)

for class_idx in range(3):
    train[f"label_{class_idx}"] = (train["health"].to_numpy() == class_idx).astype(int)
    train[f"pred_{class_idx}"] = oof_preds_lgb[:, class_idx]

# %%
train.head(3)

# %%
from sklearn.metrics import log_loss


class_weight_dict = {0: 1000/3535, 1: 1000/15751, 2: 1000/698}
for class_idx in range(3):
    train[f"loss_{class_idx}"] = train.apply(
        lambda x: log_loss([int(getattr(x, f"label_{class_idx}"))], [getattr(x, f"pred_{class_idx}")],
                           sample_weight=[class_weight_dict[getattr(x, "health")]],
                           labels=[0, 1],),
        axis=1
    )

# %%
train.sort_values(by="pred_0", ascending=False).head(10)

# %%
train.sort_values(by="pred_1", ascending=False).head(10)

# %%
train.sort_values(by="pred_2", ascending=False).head(10)

# %%
from sklearn.metrics import f1_score


scores = []
for class_idx in range(3):
    score = f1_score(
                train[f"label_{class_idx}"].astype(int),
                (train["pred"].to_numpy() == class_idx).astype(int),
            )
    scores.append(score)
    print(class_idx, score)
print("mean", np.mean(scores))

# %%


# %%
train = pd.read_csv(CSVPath.train).drop(["Unnamed: 0"], axis=1)
test = pd.read_csv(CSVPath.test).drop(["Unnamed: 0"], axis=1)

train = train.drop(["boroname", "nta_name", "spc_latin"], axis=1)
test = test.drop(["boroname", "nta_name", "spc_latin"], axis=1)

train, test = convert_rawdata_to_traindata(train, test)


train["label"] = (train["health"].to_numpy() == 2).astype(int)
train = train.drop(["health"], axis=1)

train["fold"] = validate("StratifiedKFold", train, Config.n_fold, train["label"].tolist(), random_state=Config.seed, shuffle=True)

# %%
from typing import Any
import pickle

import lightgbm
import catboost


def train_lightgbm_v2(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    train_params: dict[str, Any],
    output_path: str = "../models",
) -> None:
    """
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "custom",
        "learning_rate": 0.01,
        "seed": seed,
        "verbose": -1,
    }
    """
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = lightgbm.Dataset(
        X_train, label=y_train, categorical_feature=categorical_features,
        weight=compute_sample_weight(class_weight='balanced', y=y_train.values)
    )
    valid_data = lightgbm.Dataset(
        X_valid, label=y_valid, categorical_feature=categorical_features
    )
    model = lightgbm.train(
        train_params,
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


def train_catboost_v2(
    train_set: tuple[pd.DataFrame, pd.DataFrame],
    valid_set: tuple[pd.DataFrame, pd.DataFrame],
    categorical_features: list[str],
    fold: int,
    seed: int,
    train_params: dict[str, Any],
    output_path: str = "../models",
) -> None:
    """
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
    """
    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    train_data = catboost.Pool(
        X_train, label=y_train, cat_features=categorical_features
    )
    eval_data = catboost.Pool(X_valid, label=y_valid, cat_features=categorical_features)
    # see: https://catboost.ai/en/docs/concepts/loss-functions-ranking#usage-information

    model = catboost.CatBoost(train_params)
    model.fit(train_data, eval_set=[eval_data], use_best_model=True, plot=False)
    pickle.dump(
        model, open(os.path.join(output_path, "ctb_fold{}.ctbmodel".format(fold)), "wb")
    )

def train_folds_v3(
    train: pd.DataFrame,
    folds: list[int],
    seed: int,
    model_type: str,
    label_col: str,
    not_use_cols: list[str],
    cat_cols: list[str],
    train_params: dict[str, Any],
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
            train_lightgbm_v2(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                train_params,
                output_path,
            )
        elif model_type == "ctb":
            train_catboost_v2(
                (X_train, y_train),
                (X_valid, y_valid),
                categorical_features,
                fold,
                seed,
                train_params,
                output_path,
            )
        else:
            raise NotImplementedError(model_type)

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]


train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}"
)


oof_preds_lgb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}"
)

score = f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
print(score)


