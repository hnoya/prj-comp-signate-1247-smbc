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
from src.models import train_folds_v3, eval_folds_v3
from src.features import make_ratio, make_statvalue

# %%
@dataclass
class Config:
    experiment_name: str = "20231226_02"
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
valid = train.loc[train["fold"] == Config.n_fold - 1].reset_index(drop=True)
train = train.loc[train["fold"] != Config.n_fold - 1].reset_index(drop=True)

Config.n_fold -= 1

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
        te_dict = (all_train_df[(f"health_is_{class_idx}", 'sum')] / (all_train_df[(f"health_is_{class_idx}", 'count')] + 1)).to_dict()
        test[f"health_is_{class_idx}_te_by_{col}"] = pd.to_numeric(test[col].apply(lambda x: te_dict[x] if x in te_dict.keys() else pd.NA), errors="coerce")

        train = train.drop([f"health_is_{class_idx}"], axis=1)
    return train, test

# %%
#train, valid = add_te(train, valid, "spc_common")
#train, valid = add_te(train, valid, "boro_ct")

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
"""
holdout_trains = []
for fold_idx in range(Config.n_fold):
    holdout_train = train.loc[train["fold"] == fold_idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        holdout_train = make_statvalue(holdout_train, cat_cols, qua_cols, type_cols)
    holdout_trains.append(holdout_train)

train = pd.concat(holdout_trains).sort_index()
test = make_statvalue(test, cat_cols, qua_cols, type_cols)
"""

# %%
train.head(3)

# %%
test.head(3)

# %% [markdown]
# importance (20231222_01)
# spc_common, boro_ct, nta, st_assem, cb_num, problems, st_senate, cncldist, zip_city, user_type, guards, sidewalk, borocode, curb_loc
# 
# create_at_float, tree_dbh, steward_float

# %% [markdown]
# cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
#      "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
# 
# 
# #$ os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)
# train_folds_v3(
#     train,
#     list(range(Config.n_fold)),
#     Config.seed,
#     "lgb",
#     "health",
#     ["health", "fold"],
#     cat_cols,
#     {
#         "objective": "multiclass",
#         "num_class": 3,
#         "metric": "custom",
#         "learning_rate": 0.01,
#         "seed": Config.seed,
#         "verbose": -1,
#     },
#     f"../models/{Config.experiment_name}"
# )
# 
# 
# oof_preds_lgb = eval_folds_v3(
#     train,
#     list(range(Config.n_fold)),
#     Config.seed,
#     "lgb",
#     "health",
#     ["health", "fold"],
#     cat_cols,
#     f"../models/{Config.experiment_name}"
# )
# 
# score = f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
# print(score)

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]


#$ os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)

oof_preds_lgb = np.zeros((len(train), 3))

for fold_idx in range(Config.n_fold):
    _train = train.copy()
    col = "boro_ct"
    for class_idx in range(3):
        _train[f"health_is_{class_idx}"] = (_train["health"].to_numpy() == class_idx).astype(int).tolist()
        ts = pd.Series(np.empty(_train.shape[0]), index=_train.index)
        train_df = _train.loc[_train["fold"] != fold_idx].reset_index(drop=True)
        train_df_agg = train_df.groupby(
            col, as_index=False
        ).agg({f"health_is_{class_idx}": ["sum", "count"]})
        te_dict = (train_df_agg[(f"health_is_{class_idx}", 'sum')] / (train_df_agg[(f"health_is_{class_idx}", 'count')] + 1)).to_dict()
        _train[f"health_is_{class_idx}_te_by_{col}"] = pd.to_numeric(
            _train[col].apply(lambda x: te_dict[x] if x in te_dict.keys() else pd.NA), errors="coerce")
        _train = _train.drop([f"health_is_{class_idx}"], axis=1)

    train_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "lgb",
        "health",
        ["health", "fold"],
        cat_cols,
        {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "custom",
            "learning_rate": 0.01,
            "seed": Config.seed,
            "verbose": -1,
        },
        f"../models/{Config.experiment_name}"
    )


    oof_preds_lgb += eval_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "lgb",
        "health",
        ["health", "fold"],
        cat_cols,
        f"../models/{Config.experiment_name}"
    )

score = f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
print(score)

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
        te_dict = (all_train_df[(f"health_is_{class_idx}", 'sum')] / (all_train_df[(f"health_is_{class_idx}", 'count')] + 1)).to_dict()
        test[f"health_is_{class_idx}_te_by_{col}"] = pd.to_numeric(test[col].apply(lambda x: te_dict[x] if x in te_dict.keys() else pd.NA), errors="coerce")

        train = train.drop([f"health_is_{class_idx}"], axis=1)
    return train, test

# %%
_, valid = add_te(train, valid, "boro_ct")
valid_preds_lgb = predict_lightgbm(
    valid[[col for col in valid.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    f"../models/{Config.experiment_name}",
)

# %%
f1_score(
    valid["health"],
    np.argmax(valid_preds_lgb, axis=1),
    average="macro",
)



