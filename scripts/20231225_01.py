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
    experiment_name: str = "20231225_01"
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

# %%
train["health_reg"] = train["health"].tolist()
train.loc[train["health"] == 2, "health_reg"] = -1
train["health_reg"] += 1
train["health_reg"].describe()

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]


# os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)
train_folds_v3(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health_reg",
    ["health", "health_reg", "fold"],
    cat_cols,
    {
        "objective": "mae",
        "metric": "custom",
        "learning_rate": 0.1,
        "seed": Config.seed,
        "verbose": -1,
    },
    f"../models/{Config.experiment_name}"
)


oof_preds_lgb = eval_folds_v3(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health_reg",
    ["health", "health_reg", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}"
)

# %%
import seaborn as sns


sns.kdeplot(oof_preds_lgb)

# %%
from scipy import optimize

def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    _x = np.array(x)
    arr[np.where(_x < param[0])] = 2
    arr[np.where((param[0] <= _x) & (_x < param[1]))] = 0
    arr[np.where(param[1] <= _x)] = 1
    return -1 * f1_score(y, arr.astype(int).tolist(), average='macro')

para = [-0.9, 0.9]

_oof_preds_lgb = np.array(train["health"].tolist()).astype(float)
_oof_preds_lgb[np.where(np.array(train["health"].tolist()) == 2)] = -1
_oof_preds_lgb[np.where(np.array(train["health"].tolist()) == 0)] = 0
_oof_preds_lgb[np.where(np.array(train["health"].tolist()) == 1)] = 1
_oof_preds_lgb += np.random.random((len(_oof_preds_lgb))) * 0.5

m = optimize.minimize(
    calc_f1, para, args=(_oof_preds_lgb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

print(para, m.x, m.fun)
lgb_thlds = m.x

# %%
from scipy import optimize

def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    _x = np.array(x)
    arr[np.where(_x < param[0])] = 2
    arr[np.where((param[0] <= _x) & (_x < param[1]))] = 0
    arr[np.where(param[1] <= _x)] = 1
    return -1 * f1_score(y, arr.astype(int).tolist(), average='macro')

para = [0.1, 1.9]

m = optimize.minimize(
    calc_f1, para, args=(oof_preds_lgb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

print(para, m.x, m.fun)
lgb_thlds = m.x

# %%
oof_preds_lgb_label = np.zeros(len(oof_preds_lgb))
oof_preds_lgb_label[np.where(oof_preds_lgb < lgb_thlds[0])] = 2
oof_preds_lgb_label[np.where((lgb_thlds[0] <= oof_preds_lgb) & (oof_preds_lgb < lgb_thlds[1]))] = 0
oof_preds_lgb_label[np.where(lgb_thlds[1] <= oof_preds_lgb)] = 1

f1_score(
    train["health"],
    oof_preds_lgb_label.astype(int),
    average='macro',
)

# %% [markdown]
# - mse: 0.33953187288145587
# - mae: 0.34027357818842213
# - other metric: lower than 0.33X

# %%



