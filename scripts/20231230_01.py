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
    experiment_name: str = "20231230_01"
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
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post"]


oof_preds_lgb = np.zeros((len(train), 3))
for label_idx in range(3):
    os.makedirs(f"../models/{Config.experiment_name}/label_{label_idx}", exist_ok=True)
    train["label"] = (train["health"].to_numpy() == label_idx).tolist()
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

        # """
        train_folds_v3(
            _train,
            [fold_idx],
            Config.seed,
            "lgb",
            "label",
            ["health", "fold", "label"],
            cat_cols,
            {
                "objective": "binary", # "multiclass"
                # "num_class": 3,
                "metric": "custom",
                "learning_rate": 0.01,
                "seed": Config.seed,
                "verbose": -1,
            },
            f"../models/{Config.experiment_name}"
        )
        # """


        oof_preds_lgb[:, label_idx] += eval_folds_v3(
            _train,
            [fold_idx],
            Config.seed,
            "lgb",
            "label",
            ["health", "fold", "label"],
            cat_cols,
            f"../models/{Config.experiment_name}"
        )

    score = f1_score(train["label"], (oof_preds_lgb[:, label_idx] >= 0.5).astype(int), average="binary")
    print(score)

print(f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average="macro"))

# %% [markdown]
# - base (20231228_02): 0.36099044096590055
# - 0.33857891969416065

# %%
def sigmoid(arr, k):
    return 1/(1+np.exp(-k*arr))

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post"]

oof_preds_ctb = np.zeros((len(train), 3))
for label_idx in range(3):
    os.makedirs(f"../models/{Config.experiment_name}/label_{label_idx}", exist_ok=True)
    train["label"] = (train["health"].to_numpy() == label_idx).tolist()
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
            "label",
            ["health", "fold", "label"],
            cat_cols,
            {
                "objective": "Logloss",  # "MultiClass",
                "loss_function": "Logloss",  # "CrossEntropy",
                "eval_metric": "F1",
                # "eval_metric": "TotalF1:average=Macro;use_weights=false", # ;use_weights=false",
                "num_boost_round": 10_000,
                "early_stopping_rounds": 1_000,
                "learning_rate": 0.01, # 0.01
                "verbose": 1_000,
                "random_seed": Config.seed,
                "task_type": "GPU",
                # "class_weights": [1000/3535, 1000/15751, 1000/698],
                "auto_class_weights": 'Balanced',
            },
            f"../models/{Config.experiment_name}",
        )


        oof_preds_ctb[:, label_idx] += eval_folds_v3(
            _train,
            [fold_idx],
            Config.seed,
            "ctb",
            "label",
            ["health", "fold", "label"],
            cat_cols,
            f"../models/{Config.experiment_name}"
        )

    score = f1_score(
        train["label"],
        (sigmoid(oof_preds_ctb[:, label_idx], 1) >= 0.5).astype(int),
        average="binary",
    )
    print(score)

print(f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average="macro"))

# %% [markdown]
# - base (20231228_02): 0.36431500437373465
# 

# %%
print(
    f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average=None),
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average=None),
)
print(
    np.mean([
        f1_score((train["health"].to_numpy() == label).astype(int), (oof_preds_lgb[:, label] >= 0.5).astype(int), average="binary")
        for label
        in range(3)
    ]),
    np.mean([
        f1_score((train["health"].to_numpy() == label).astype(int), (sigmoid(oof_preds_ctb[:, label], 1) >= 0.5).astype(int), average="binary")
        for label
        in range(3)
    ]),)
print(
    f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average="macro"),
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average="macro"),
)

# %%
oof_preds_lgb_cls = np.zeros((len(oof_preds_lgb))) - 1
oof_preds_lgb_cls[np.where(
    (oof_preds_lgb[:, 0] >= 0.5)
    & (oof_preds_lgb[:, 1] < 0.5)
    & (oof_preds_lgb[:, 2] < 0.5)
)] = 0
oof_preds_lgb_cls[np.where(
    (oof_preds_lgb[:, 0] < 0.5)
    & (oof_preds_lgb[:, 1] >= 0.5)
    & (oof_preds_lgb[:, 2] < 0.5)
)] = 1
oof_preds_lgb_cls[np.where(
    (oof_preds_lgb[:, 0] < 0.5)
    & (oof_preds_lgb[:, 1] < 0.5)
    & (oof_preds_lgb[:, 2] >= 0.5)
)] = 2

# %%
len(oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)]), len(oof_preds_lgb_cls[np.where(oof_preds_lgb_cls == -1)])

# %%
f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_lgb_cls != -1)],
    oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)],
    average="macro",
), f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_lgb_cls != -1)],
    oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)],
    average=None,
)

# %%
f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_lgb_cls == -1)],
    np.argmax(oof_preds_lgb[np.where(oof_preds_lgb_cls == -1)], axis=1),
    average="macro",
), f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_lgb_cls == -1)],
    np.argmax(oof_preds_lgb[np.where(oof_preds_lgb_cls == -1)], axis=1),
    average=None,
)

# %%
oof_preds_lgb_cls[np.where(oof_preds_lgb_cls == -1)] = np.argmax(oof_preds_lgb[np.where(oof_preds_lgb_cls == -1), :][0], axis=1)
f1_score(train["health"], oof_preds_lgb_cls, average="macro"), f1_score(train["health"], oof_preds_lgb_cls, average=None)

# %%
for i in range(11):
    thld = i * 0.1
    print(thld)
    oof_preds_lgb_cls = np.zeros((len(oof_preds_lgb))) - 1
    oof_preds_lgb_cls[np.where(
        (oof_preds_lgb[:, 0] >= thld)
        & (oof_preds_lgb[:, 1] < thld)
        & (oof_preds_lgb[:, 2] < thld)
    )] = 0
    oof_preds_lgb_cls[np.where(
        (oof_preds_lgb[:, 0] < thld)
        & (oof_preds_lgb[:, 1] >= thld)
        & (oof_preds_lgb[:, 2] < thld)
    )] = 1
    oof_preds_lgb_cls[np.where(
        (oof_preds_lgb[:, 0] < thld)
        & (oof_preds_lgb[:, 1] < thld)
        & (oof_preds_lgb[:, 2] >= thld)
    )] = 2
    print(
        len(oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)]), len(oof_preds_lgb_cls[np.where(oof_preds_lgb_cls == -1)])
    )
    print(
        f1_score(
            (train["health"].to_numpy())[np.where(oof_preds_lgb_cls != -1)],
            oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)],
            average="macro",
        ), f1_score(
            (train["health"].to_numpy())[np.where(oof_preds_lgb_cls != -1)],
            oof_preds_lgb_cls[np.where(oof_preds_lgb_cls != -1)],
            average=None,
        )
    )
    oof_preds_lgb_cls[np.where(oof_preds_lgb_cls == -1)] = np.argmax(oof_preds_lgb[np.where(oof_preds_lgb_cls == -1), :][0], axis=1)
    print(
        f1_score(train["health"], oof_preds_lgb_cls, average="macro"), f1_score(train["health"], oof_preds_lgb_cls, average=None)
    )
    print("--" * 10)

# %%
oof_preds_ctb_cls = np.zeros((len(oof_preds_ctb))) - 1
_oof_preds_ctb = sigmoid(oof_preds_ctb, 1)
oof_preds_ctb_cls[np.where(
    (_oof_preds_ctb[:, 0] >= 0.5)
    & (_oof_preds_ctb[:, 1] < 0.5)
    & (_oof_preds_ctb[:, 2] < 0.5)
)] = 0
oof_preds_ctb_cls[np.where(
    (_oof_preds_ctb[:, 0] < 0.5)
    & (_oof_preds_ctb[:, 1] >= 0.5)
    & (_oof_preds_ctb[:, 2] < 0.5)
)] = 1
oof_preds_ctb_cls[np.where(
    (_oof_preds_ctb[:, 0] < 0.5)
    & (_oof_preds_ctb[:, 1] < 0.5)
    & (_oof_preds_ctb[:, 2] >= 0.5)
)] = 2

len(oof_preds_ctb_cls[np.where(oof_preds_ctb_cls != -1)]), len(oof_preds_ctb_cls[np.where(oof_preds_ctb_cls == -1)])

# %%
f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_ctb_cls != -1)],
    oof_preds_ctb_cls[np.where(oof_preds_ctb_cls != -1)],
    average="macro",
), f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_ctb_cls != -1)],
    oof_preds_ctb_cls[np.where(oof_preds_ctb_cls != -1)],
    average=None,
)

# %%
f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_ctb_cls == -1)],
    np.argmax(oof_preds_ctb[np.where(oof_preds_ctb_cls == -1)], axis=1),
    average="macro",
), f1_score(
    (train["health"].to_numpy())[np.where(oof_preds_ctb_cls == -1)],
    np.argmax(oof_preds_ctb[np.where(oof_preds_ctb_cls == -1)], axis=1),
    average=None,
)

# %%
oof_preds_ctb_cls[np.where(oof_preds_ctb_cls == -1)] = np.argmax(oof_preds_ctb[np.where(oof_preds_ctb_cls == -1), :][0], axis=1)
f1_score(train["health"], oof_preds_ctb_cls, average="macro"), f1_score(train["health"], oof_preds_ctb_cls, average=None)

# %%
_oof_preds_ctb = sigmoid(oof_preds_ctb, 1)
for i in range(11):
    thld = i * 0.1
    print(thld)
    _oof_preds_ctb_cls = np.zeros((len(_oof_preds_ctb))) - 1
    _oof_preds_ctb_cls[np.where(
        (_oof_preds_ctb[:, 0] >= thld)
        & (_oof_preds_ctb[:, 1] < thld)
        & (_oof_preds_ctb[:, 2] < thld)
    )] = 0
    _oof_preds_ctb_cls[np.where(
        (_oof_preds_ctb[:, 0] < thld)
        & (_oof_preds_ctb[:, 1] >= thld)
        & (_oof_preds_ctb[:, 2] < thld)
    )] = 1
    _oof_preds_ctb_cls[np.where(
        (_oof_preds_ctb[:, 0] < thld)
        & (_oof_preds_ctb[:, 1] < thld)
        & (_oof_preds_ctb[:, 2] >= thld)
    )] = 2
    print(
        len(_oof_preds_ctb_cls[np.where(_oof_preds_ctb_cls != -1)]), len(_oof_preds_ctb_cls[np.where(_oof_preds_ctb_cls == -1)])
    )
    print(
        f1_score(
            (train["health"].to_numpy())[np.where(_oof_preds_ctb_cls != -1)],
            _oof_preds_ctb_cls[np.where(_oof_preds_ctb_cls != -1)],
            average="macro",
        ), f1_score(
            (train["health"].to_numpy())[np.where(_oof_preds_ctb_cls != -1)],
            _oof_preds_ctb_cls[np.where(_oof_preds_ctb_cls != -1)],
            average=None,
        )
    )
    _oof_preds_ctb_cls[np.where(_oof_preds_ctb_cls == -1)] = np.argmax(_oof_preds_ctb[np.where(_oof_preds_ctb_cls == -1), :][0], axis=1)
    print(
        f1_score(train["health"], _oof_preds_ctb_cls, average="macro"), f1_score(train["health"], _oof_preds_ctb_cls, average=None)
    )
    print("--" * 10)


