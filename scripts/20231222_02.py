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
    experiment_name: str = "20231222_02"
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
train["fold"] = validate("KFold", train, Config.n_fold, train["health"].tolist(), random_state=Config.seed, shuffle=True)

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
train, test = add_te(train, test, "spc_common")
train, test = add_te(train, test, "boro_ct")
# train, test = add_te(train, test, "nta")
# train, test = add_te(train, test, "st_assem")
# train, test = add_te(train, test, "cb_num")
# train, test = add_te(train, test, "problems")

# %%
train.head(3)

# %%
test.head(3)

# %% [markdown]
# - ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
# - ["tree_dbh", "steward", "create_at_float"]
# - ['mean', 'std', 'max', 'min']
# 

# %% [markdown]
# importance (20231222_01)
# spc_common, boro_ct, nta, st_assem, cb_num, problems, st_senate, cncldist, zip_city, user_type, guards, sidewalk, borocode, curb_loc
# 
# create_at_float, tree_dbh, steward_float

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
test.head(3)

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
# - 1: te by spc_common : 0.35743034180008587
# - 3: te by nta : 0.37301926074568037
# - 8: te by boro_ct: 0.43922730085917006
# 
# - 6: te by nta + st_assem: 0.3739260565224825
# - 7: te by boro_ct + nta: 0.437721159715496
# - 9: te by boro_ct + st_assem: 0.4371433545641558
# - 5: te by spc_common + st_assem: 0.4392300046476163
# - 2: te by spc_common + boro_ct : 0.4400959629763048 (leaked? boro_ct nunique = 1193, len(train) = 19984)
# 
# - 4: te by spc_common + boro_ct + nta: 0.43803654498780625
# - 10: te by spc_common + boro_ct + st_num: 0.4363921222206378
# - 11: te by spc_common + boro_ct + problems: 0.43903556217500767

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
# - base: 0.34598202669025074
# - 0.3603804713076419

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
oof_preds_ctb_calib = apply_sigmoid(oof_preds_ctb)
oof_preds_ctb_calib /= np.sum(oof_preds_ctb_calib, axis=1).reshape(-1, 1)

for i in range(10):
    print(
        i, f1_score(train["health"], np.argmax(oof_preds_lgb * i * 0.1 + oof_preds_ctb_calib * (10 - i) * 0.1, axis=1), average='macro')
    )

# %%
test_preds = y_preds_lgb # * 0.5 + y_preds_ctb * 0.5

submission = pd.read_csv(CSVPath.submission, header=None)
submission.iloc[:, 1] = np.argmax(test_preds, axis=1)
submission.to_csv(f"submission_{Config.experiment_name}.csv", index=False, header=False)

# %%
train["health"].value_counts() / len(train)

# %%
pd.DataFrame(np.argmax(y_preds_lgb, axis=1)).value_counts() / len(y_preds_lgb)

# %%



