# %%
from dataclasses import dataclass
import os
import warnings

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
    experiment_name: str = "20231229_01"
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
                "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
                "spc_common_pre", "spc_common_post"]:
        train_and_test = apply_le(train_and_test, col, keep_nan=False)
        train[col] = train_and_test[col].iloc[:len(train)].tolist()
        test[col] = train_and_test[col].iloc[-len(test):].tolist()

    return train, test

# %%
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def add_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    geolocator = Nominatim(user_agent="geoapiExercises")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    df["combined_address"] = df["zip_city"] + ", " + df["boroname"]
    unique_addresses = df[["zip_city", "boroname"]].drop_duplicates(ignore_index=True)
    unique_addresses["combined_address"] = unique_addresses["zip_city"] + ", " + unique_addresses["boroname"]
    unique_addresses["location"] = unique_addresses["combined_address"].apply(lambda addr: geocode(addr))
    unique_addresses["latitude"] = unique_addresses["location"].apply(lambda loc: loc.latitude if loc else None)
    unique_addresses["longitude"] = unique_addresses["location"].apply(lambda loc: loc.longitude if loc else None)

    df = pd.merge(df, unique_addresses[["combined_address", "latitude", "longitude"]], on="combined_address", how="left")
    return df

# %%
train = pd.read_csv(CSVPath.train).drop(["Unnamed: 0"], axis=1)
test = pd.read_csv(CSVPath.test).drop(["Unnamed: 0"], axis=1)

# %%
import datetime

def add_ymd(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["created_at"].apply(lambda x: int(x.split("-")[0]))
    df["month"] = df["created_at"].apply(lambda x: int(x.split("-")[1]))
    df["day"] = df["created_at"].apply(lambda x: int(x.split("-")[2]))

    df["weekday"] = df["created_at"].apply(lambda x: datetime.datetime(*map(int, x.split("-"))).weekday())
    return df

train = add_ymd(train)
test = add_ymd(test)

# %%
def add_splited_spc_common(df: pd.DataFrame):
    df["spc_common_pre"] = df["spc_common"].apply(lambda x: x.split(" ")[0])
    df["spc_common_post"] = df["spc_common"].apply(lambda x: x.split(" ")[-1])
    return df

train = add_splited_spc_common(train)
test = add_splited_spc_common(test)

# %%
# """
if os.path.exists("../features/location_20231227_01.csv"):
    combined_address = pd.read_csv("../features/location_20231227_01.csv")
    train["combined_address"] = train["zip_city"] + ", " + train["boroname"]
    train = pd.merge(train, combined_address[["combined_address", "latitude", "longitude"]], on="combined_address", how="left")
    train = train.drop(["combined_address"], axis=1)
    test["combined_address"] = test["zip_city"] + ", " + test["boroname"]
    test = pd.merge(test, combined_address[["combined_address", "latitude", "longitude"]], on="combined_address", how="left")
    test = test.drop(["combined_address"], axis=1)
else:
    train = add_lat_lon(train)
    test = add_lat_lon(test)

    pd.concat([
        train[["combined_address", "latitude", "longitude"]],
        test[["combined_address", "latitude", "longitude"]]
    ], ignore_index=True).drop_duplicates(ignore_index=True).to_csv(f"../features/location_{Config.experiment_name}.csv", index=False)
# """

# %%
train, test = convert_rawdata_to_traindata(train, test)
train["fold"] = validate("StratifiedKFold", train, Config.n_fold, train["health"].tolist(), random_state=Config.seed, shuffle=True)

train = train.drop(["boroname", "nta_name", "spc_latin"], axis=1)
test = test.drop(["boroname", "nta_name", "spc_latin"], axis=1)

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


cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
            "spc_common_pre", "spc_common_post"]
qua_cols = ["tree_dbh", "steward_float", "created_at_float"]
type_cols = ['mean', 'std', 'max', 'min']
cols = cat_cols + qua_cols


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train = make_statvalue(train, cat_cols, qua_cols, type_cols)
    test = make_statvalue(test, cat_cols, qua_cols, type_cols)


# %%
import warnings

"""
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
qua_cols = ["latitude", "longitude"]
type_cols = ['mean', 'std', 'max', 'min']
cols = cat_cols + qua_cols


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train = make_statvalue(train, cat_cols, qua_cols, type_cols)
    test = make_statvalue(test, cat_cols, qua_cols, type_cols)
"""


# %%
train.head(3)

# %%
test.head(3)

# %%
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train = make_ratio(train, cat_cols)
    test = make_ratio(test, cat_cols)

# %%
train["health_reg"] = train["health"].tolist()
train.loc[train["health"] == 2, "health_reg"] = -1
train["health_reg"] += 1
train["health_reg"].describe()

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post"]


#os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)

oof_preds_lgb = np.zeros((len(train))) # , 3))

for fold_idx in range(Config.n_fold):
    _train = train.copy()
    for col in cat_cols:
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

    """
    train_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "lgb",
        "health_reg",
        ["health", "fold", "health_reg"],
        cat_cols,
        {
            "objective": "mae", # "multiclass"
            # "num_class": 3,
            "metric": "custom",
            "learning_rate": 0.01,
            "seed": Config.seed,
            "verbose": -1,
        },
        f"../models/{Config.experiment_name}"
    )
    """


    oof_preds_lgb += eval_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "lgb",
        "health_reg",
        ["health", "fold", "health_reg"],
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

para = [0.1, 1.9]

m = optimize.minimize(
    calc_f1, para, args=(oof_preds_lgb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

print(para, m.x, m.fun)
lgb_thlds = m.x

# %% [markdown]
# - base (20231228_02): 0.36099044096590055
# - 0.33857891969416065

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post"]


oof_preds_ctb = np.zeros((len(train))) # , 3))
for fold_idx in range(Config.n_fold):
    _train = train.copy()
    for col in cat_cols:
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
        "ctb",
        "health_reg",
        ["health", "fold", "health_reg"],
        cat_cols,
        {
            "objective": "RMSE",  # "MultiClass",
            "loss_function": "RMSE",  # "CrossEntropy",
            # "eval_metric": "TotalF1:average=Macro;use_weights=false", # ;use_weights=false",
            "num_boost_round": 100_000,
            "early_stopping_rounds": 1_000,
            "learning_rate": 0.01, # 0.01
            "verbose": 10_000,
            "random_seed": Config.seed,
            "task_type": "GPU",
            # "class_weights": [1000/3535, 1000/15751, 1000/698],
        },
        f"../models/{Config.experiment_name}",
    )


    oof_preds_ctb += eval_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "ctb",
        "health_reg",
        ["health", "fold", "health_reg"],
        cat_cols,
        f"../models/{Config.experiment_name}"
    )

# %% [markdown]
# - base (20231228_02): 0.36431500437373465
# - RMSE
#     - no weight: 0.35150517023776184
#     - weight: 0.33754706742636303 (~70min/5folds with lr=0.5)
# - MAE:
#     - no weight: 0.3386548987938008
#     - weight: 0.30898382570729116

# %%
sns.kdeplot(oof_preds_ctb)

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
    calc_f1, para, args=(oof_preds_ctb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

print(para, m.x, m.fun)
ctb_thlds = m.x

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
plt.barh(range(len(sorted_idx))[-30:], feature_importance[sorted_idx][-30:], align='center')
plt.yticks(
    range(len(sorted_idx))[-30:],
    np.array([col for col in _train.columns if col not in ["health", "fold"]])[sorted_idx][-30:]
)
plt.title('Feature Importance')


# %%



