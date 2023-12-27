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
    experiment_name: str = "20231227_02"
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
"""
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
"""

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


cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta", "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]
qua_cols = ["tree_dbh", "steward_float", "created_at_float"]
type_cols = ['mean', 'std', 'max', 'min']
cols = cat_cols + qua_cols
"""
holdout_trains = []
for fold_idx in range(Config.n_fold):
    holdout_train = train.loc[train["fold"] == fold_idx]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        holdout_train = make_statvalue(holdout_train, cat_cols, qua_cols, type_cols)
    holdout_trains.append(holdout_train)
"""
# train = pd.concat(holdout_trains).sort_index()
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train = make_statvalue(train, cat_cols, qua_cols, type_cols)
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


for label_idx in range(3):
    os.makedirs(f"../models/{Config.experiment_name}/label_{label_idx}", exist_ok=True)
    train["label"] = (train["health"].to_numpy() == label_idx).tolist()
    """
    train_folds_v3(
        train,
        list(range(Config.n_fold)),
        Config.seed,
        "lgb",
        "label",
        ["health", "fold", "label"],
        cat_cols,
        {
            "objective": "binary",
            "metric": "custom",
            # "num_class": 3,
            "learning_rate": 0.01,
            "seed": Config.seed,
            "verbose": -1,
        },
        f"../models/{Config.experiment_name}/label_{label_idx}"
    )
    """


    oof_preds_lgb = eval_folds_v3(
        train,
        list(range(Config.n_fold)),
        Config.seed,
        "lgb",
        "label",
        ["health", "fold", "label"],
        cat_cols,
        f"../models/{Config.experiment_name}/label_{label_idx}"
    )

    score = f1_score(train["label"], (oof_preds_lgb >= 0.5).astype(int), average="binary")
    print(score)

# %%
def sigmoid(arr, k):
    return 1/(1+np.exp(-k*arr))

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]

weight_paris_list = [
    [(3535 + 15751 + 698) / (15751 + 698), (3535 + 15751 + 698) / 3535],
    [(3535 + 15751 + 698) / (3535 + 698), (3535 + 15751 + 698) / 15751],
    [(3535 + 15751 + 698) / (3535 + 15751), (3535 + 15751 + 698) / 698]
]

for label_idx in range(3):
    os.makedirs(f"../models/{Config.experiment_name}/label_{label_idx}", exist_ok=True)
    train["label"] = (train["health"].to_numpy() == label_idx).astype(int).tolist()
    """
    train_folds_v3(
        train,
        list(range(Config.n_fold)),
        Config.seed,
        "ctb",
        "label",
        ["health", "fold", "label"],
        cat_cols,
        {
            "objective": "Logloss",  # "MultiClass",
            "loss_function": "Logloss",  # "CrossEntropy",
            "eval_metric": "F1",
            "num_boost_round": 10_000,
            "early_stopping_rounds": 1_000,
            "learning_rate": 0.01,
            "verbose": 1_000,
            "random_seed": Config.seed,
            "task_type": "GPU",
            # "class_weights": weight_paris_list[label_idx],
            "auto_class_weights": 'Balanced',
        },
        f"../models/{Config.experiment_name}/label_{label_idx}",
    )
    """

    oof_preds_ctb = eval_folds_v3(
        train,
        list(range(Config.n_fold)),
        Config.seed,
        "ctb",
        "label",
        ["health", "fold", "label"],
        cat_cols,
        f"../models/{Config.experiment_name}/label_{label_idx}",
    )

    score = f1_score(train["label"], (sigmoid(oof_preds_ctb, 1) >= 0.5).astype(int), average="binary")
    print(score)


# %% [markdown]
# - base: lgb 0.35466775297446107 ctb 0.35466775297446107
# - lgb 0.3588919931342463 ctb 0.3608654695114177

# %%
y_preds_lgb = np.zeros((len(test), 3))
y_preds_ctb = np.zeros((len(test), 3))

for label_idx in range(3):
    y_preds_lgb[:, label_idx] = predict_lightgbm(
        test[[col for col in test.columns if col not in ["health", "fold"]]],
        list(range(Config.n_fold)),
        f"../models/{Config.experiment_name}/label_{label_idx}",
    )

    y_preds_ctb[:, label_idx] = predict_catboost(
        test[[col for col in test.columns if col not in ["health", "fold"]]],
        list(range(Config.n_fold)),
        ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
        "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"],
        f"../models/{Config.experiment_name}/label_{label_idx}",
    )

# %%
def sigmoid(arr, k):
    return 1/(1+np.exp(-k*arr))

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

print(para, m.x, m.fun)
cbt_weight = m.x

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

print(para, m.x, m.fun)
lgb_weight = m.x

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
    
print(best_weight, best_score)

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


# %%
train["health_reg"] = train["health"].tolist()
train.loc[train["health"] == 2, "health_reg"] = -1
train["health_reg"] += 1
train["health_reg"].describe()

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist"]


os.makedirs(f"../models/{Config.experiment_name}/reg", exist_ok=True)
train_folds_v3(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health_reg",
    ["health_reg", "health", "fold"],
    cat_cols,
    {
        "objective": "mse",
        "metric": "custom",
        # "num_class": 3,
        "learning_rate": 0.01,
        "seed": Config.seed,
        "verbose": -1,
    },
    f"../models/{Config.experiment_name}/reg"
)


oof_preds_lgb_reg = eval_folds_v3(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health_reg",
    ["health_reg", "health", "fold"],
    cat_cols,
    f"../models/{Config.experiment_name}/reg"
)

# %%
import seaborn as sns


sns.kdeplot(oof_preds_lgb_reg)

# %%
lgb_thld0 = 0.5
lgb_thld1 = 1.5

oof_preds_lgb_reg_cls = np.zeros_like(oof_preds_lgb_reg).astype(int)
oof_preds_lgb_reg_cls[np.where(oof_preds_lgb_reg <= lgb_thld0)] = 2
oof_preds_lgb_reg_cls[np.where(lgb_thld1 < oof_preds_lgb_reg)] = 1

score = f1_score(train["health"], oof_preds_lgb_reg_cls, average='macro')
print(score)

# %%
lgb_thld0 = np.quantile(oof_preds_lgb_reg, 0.034928)
lgb_thld1 = np.quantile(oof_preds_lgb_reg, 1 - 0.176892)

oof_preds_lgb_reg_cls = np.zeros_like(oof_preds_lgb_reg).astype(int)
oof_preds_lgb_reg_cls[np.where(oof_preds_lgb_reg <= lgb_thld0)] = 2
oof_preds_lgb_reg_cls[np.where(lgb_thld1 < oof_preds_lgb_reg)] = 1

score = f1_score(train["health"], oof_preds_lgb_reg_cls, average='macro')
print(score)

# %%
lgb_thld0, lgb_thld1

# %%



