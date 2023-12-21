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
    experiment_name: str = "20231220_02"
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
train.head(3)

# %%
test = pd.read_csv(CSVPath.test).drop(["Unnamed: 0"], axis=1)
test.head(3)

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
    for col in ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"]:
        train_and_test = apply_le(train_and_test, col, keep_nan=False)
        train[col] = train_and_test[col].iloc[:len(train)].tolist()
        test[col] = train_and_test[col].iloc[-len(test):].tolist()

    return train, test

# %%
train, test = convert_rawdata_to_traindata(train, test)

# %%
train.head(3)

# %%
test.head(3)

# %%
train["fold"] = validate("KFold", train, Config.n_fold, train["health"].tolist(), random_state=Config.seed, shuffle=True)

# %%
train.head(3)

# %%
import numpy as np
"""
for class_idx in range(3):
    train[f"health_is_{class_idx}"] = (train["health"].to_numpy() == class_idx).astype(int).tolist()
    ts = pd.Series(np.empty(train.shape[0]), index=train.index)

    agg_df = train.groupby('spc_common').agg({f"health_is_{class_idx}": ['sum', 'count']})
    for fold_idx in range(Config.n_fold):
        holdout_df = train.loc[train["fold"] == fold_idx]
        holdout_agg_df = holdout_df.groupby('spc_common').agg({f"health_is_{class_idx}": ['sum', 'count']})
        train_agg_df = agg_df - holdout_agg_df
        oof_ts = holdout_df.apply(lambda row: train_agg_df.loc[row.spc_common][(f"health_is_{class_idx}", 'sum')] \
                                            / (train_agg_df.loc[row.spc_common][(f"health_is_{class_idx}", 'count')] + 1), axis=1)
        ts[oof_ts.index] = oof_ts
    
    ts.name = f"health_is_{class_idx}_te"
    train = train.join(ts)

    te = train.groupby(
        'spc_common', as_index=False
    ).agg({f"health_is_{class_idx}": 'mean'}).rename(columns={f"health_is_{class_idx}": f"health_is_{class_idx}_te"})
    test = pd.merge(test, te, on='spc_common', right_index=True)

    train = train.drop([f"health_is_{class_idx}"], axis=1)
"""

# %% [markdown]
# - ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"]
# - ["tree_dbh", "steward", "create_at_float"]
# - ['mean', 'std', 'max', 'min']

# %%
"""
cat_cols = []
qua_cols = []
agg_types = []
df = pd.concat([train[cat_cols + qua_cols], test[cat_cols + qua_cols]], ignore_index=True)
# df = make_ratio(df, cols)
df = make_statvalue(df, cat_cols, qua_cols, agg_types)
train = df.iloc[:len(train)].reset_index(drop=True)
test = df.iloc[len(test):].reset_index(drop=True)
"""

# %%
train.head(3)

# %%
all_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name",
     "boroname", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
scores = []
for col in all_cols:
    train_folds_v2(
        train,
        [0],
        Config.seed,
        "lgb",
        "health",
        ["health", "fold"] + [_col for _col in all_cols if _col != col],
        [col],
        f"../models/{Config.experiment_name}"
    )

    oof_preds_lgb = eval_folds_v2(
        train,
        [0],
        Config.seed,
        "lgb",
        "health",
        ["health", "fold"] + [_col for _col in all_cols if _col != col],
        [col],
        f"../models/{Config.experiment_name}"
    )

    # score = f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
    score = f1_score(
        train.loc[train["fold"] == 0, "health"],
        np.argmax(oof_preds_lgb[train.loc[train["fold"] == 0].index.to_numpy()], axis=1), average='macro',
    )
    scores.append(score)

# %%
for col, socre in zip(all_cols, scores):
    print(col, score)

# %% [markdown]
# - ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"]
# - ["tree_dbh", "steward", "create_at_float"]
# - ['mean', 'std', 'max', 'min']

# %% [markdown]
# importance
# 
# created_at_float
# tree_dbh
# steward_float
# 
# spc_common
# nta
# problems
# boro_ct
# user_type
# zip_city
# st_assem
# guards
# cncldist
# st_senate
# cb_num
# spc_latin
# sidewalk
# nta_name
# boroname
# curb_loc
# (borocode)

# %% [markdown]
# - baseline(20231220_01): 0.35184990907259234
# - health te: 0.35722605341270736

# %%
raise NotImplementedError()

# %%
train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name",
     "boroname", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
    f"../models/{Config.experiment_name}"
)

oof_preds_ctb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name",
     "boroname", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
    f"../models/{Config.experiment_name}"
)

print(
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average='macro')
)


# %%


# %%
y_preds = predict_lightgbm(
    test[[col for col in test.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    f"../models/{Config.experiment_name}",
)

# %%



