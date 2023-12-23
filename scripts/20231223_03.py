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
    experiment_name: str = "20231223_03"
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

        te = train.groupby(
            col, as_index=False
        ).agg({f"health_is_{class_idx}": 'mean'}).rename(columns={f"health_is_{class_idx}": f"health_is_{class_idx}_te_by_{col}"})
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
"""
cat_cols = ["zip_city"]
qua_cols = ["steward_float"]
type_cols = ["mean"]
cols = cat_cols + qua_cols
df = pd.concat([train[cols], test[cols]], ignore_index=True)
df = make_statvalue(df, cat_cols, qua_cols, type_cols)
add_cols = [col for col in df.columns.tolist() if col not in cols]
for col in add_cols:
    train[col] = df[col].iloc[:len(train)].reset_index(drop=True).tolist()
    test[col] = df[col].iloc[-len(test):].reset_index(drop=True).tolist()
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
# - 0: baseline(20231222_01): 0.35653867230742026
# - 1: 0.35466775297446107

# %%
train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
    f"../models/{Config.experiment_name}"
)


oof_preds_ctb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
    f"../models/{Config.experiment_name}"
)

print(
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average='macro')
)


# %% [markdown]
# - 0: baseline(20231222_01): 0.34598202669025074
# - 1: 0.35215081169320245

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

feature_importance = None
for i in range(Config.n_fold):
    model = pickle.load(
        open(f"/work/models/{Config.experiment_name}/ctb_fold{i}.ctbmodel", "rb")
    )
    if feature_importance is None:
        feature_importance = model.feature_importances_
    else:
        feature_importance += model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(
    range(len(sorted_idx)),
    np.array([col for col in train.columns if col not in ["health", "fold"]])[sorted_idx]
)
plt.title('Feature Importance')


# %%


# %%
y_preds_lgb = predict_lightgbm(
    test[[col for col in test.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    f"../models/{Config.experiment_name}",
)

y_preds_ctb = predict_catboost(
    test[[col for col in test.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
    f"../models/{Config.experiment_name}",
)

# %%
def sigmoid(arr, k):
    return 1/(1+np.exp(-k*arr))

# %%
from scipy import optimize

def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    for i in range(3):
        arr[:, i] = sigmoid(np.array(x)[:, i], param[i])
    return -1 * f1_score(y, np.argmax(arr, axis=1), average='macro')

para = [1.0, 1.0, 1.0]

m = optimize.minimize(
    calc_f1, para, args=(oof_preds_ctb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

# %%
print(para, m.x, m.fun)
cbt_weight = m.x

# %%
"""
def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    for i in range(3):
        arr[:, i] = sigmoid(np.array(x)[:, i], param[i])
    return -1 * f1_score(y, np.argmax(arr, axis=1), average='macro')

para = [1.0, 1.0, 1.0]

m = optimize.basinhopping(
    calc_f1, para, 
    minimizer_kwargs={
        "method": "L-BFGS-B",
        "args": (oof_preds_ctb.tolist(), train["health"].tolist(), ),
    },
)

print(para, m.x, m.fun)
"""

# %%
def inv_sigmoid(arr):
    return np.log(arr / (1 - arr))

def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    for i in range(3):
        arr[:, i] = sigmoid(inv_sigmoid(np.array(x)[:, i]), param[i])
    return -1 * f1_score(y, np.argmax(arr, axis=1), average='macro')

para = [1.0, 1.0, 1.0]

m = optimize.minimize(
    calc_f1, para, args=(oof_preds_lgb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

# %%
print(para, m.x, m.fun)
lgb_weight = m.x

# %%
"""
def calc_f1(param, x, y):
    arr = np.zeros_like(x)
    for i in range(3):
        arr[:, i] = sigmoid(inv_sigmoid(np.array(x)[:, i]), param[i])
    return -1 * f1_score(y, np.argmax(arr, axis=1), average='macro')

para = [1.0, 1.0, 1.0]

m = optimize.basinhopping(
    calc_f1, para, 
    minimizer_kwargs={
        "method": "L-BFGS-B",
        "args": (oof_preds_lgb.tolist(), train["health"].tolist(), ),
    },
)

print(para, m.x, m.fun)
"""

# %%
oof_preds_ctb_calib = np.zeros_like(oof_preds_ctb)
for i in range(3):
    oof_preds_ctb_calib[:, i] = sigmoid(np.array(oof_preds_ctb)[:, i], cbt_weight[i])

oof_preds_ctb_calib /= np.sum(oof_preds_ctb_calib, axis=1).reshape(-1, 1)

oof_preds_lgb_calib = np.zeros_like(oof_preds_lgb)
for i in range(3):
    oof_preds_lgb_calib[:, i] = sigmoid(inv_sigmoid(np.array(oof_preds_lgb)[:, i]), lgb_weight[i])

oof_preds_lgb_calib /= np.sum(oof_preds_lgb_calib, axis=1).reshape(-1, 1)

# %%
best_weight = 0
best_score = 0
for weight in range(11):
    weighted_pred = np.argmax(oof_preds_lgb_calib * weight * 0.1 + oof_preds_ctb_calib * (10 - weight) * 0.1, axis=1)
    score = f1_score(train["health"], weighted_pred, average='macro')
    print(
        weight, score
    )
    if best_score < score:
        best_score = score
        best_weight = weight


# %%
test_preds = y_preds_lgb * best_weight * 0.1 + y_preds_ctb * (10 - best_weight) * 0.1

submission = pd.read_csv(CSVPath.submission, header=None)
submission.iloc[:, 1] = np.argmax(test_preds, axis=1)
submission.to_csv(f"submission_{Config.experiment_name}.csv", index=False, header=False)

# %%
train["health"].value_counts() / len(train)

# %%
pd.DataFrame(np.argmax(test_preds, axis=1)).value_counts() / len(y_preds_lgb)

# %%



