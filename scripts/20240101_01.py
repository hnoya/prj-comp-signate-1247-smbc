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
    experiment_name: str = "20240101_01"
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


def convert_str_to_float(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at_float"] = df["created_at"].apply(lambda x: convert_time_str_to_unix_float(x))
    df = df.drop(["created_at"], axis=1)

    df["steward_float"] = df["steward"].apply(lambda x: convert_steward_str_to_float(x))
    df = df.drop(["steward"], axis=1)
    return df


def add_ymd(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["created_at"].apply(lambda x: int(x.split("-")[0]))
    df["month"] = df["created_at"].apply(lambda x: int(x.split("-")[1]))
    df["day"] = df["created_at"].apply(lambda x: int(x.split("-")[2]))

    df["weekday"] = df["created_at"].apply(lambda x: datetime.datetime(*map(int, x.split("-"))).weekday())
    return df


def add_splited_spc_common(df: pd.DataFrame):
    df["spc_common_pre"] = df["spc_common"].apply(lambda x: x.split(" ")[0])
    df["spc_common_post"] = df["spc_common"].apply(lambda x: x.split(" ")[-1])
    return df


def add_splited_spc_latin(df: pd.DataFrame):
    df["spc_latin_pre"] = df["spc_latin"].apply(lambda x: x.split(" ")[0])
    df["spc_latin_post"] = df["spc_latin"].apply(lambda x: x.split(" ")[-1])
    return df

def add_splited_nta(df: pd.DataFrame):
    df["nta_pre"] = df["nta"].apply(lambda x: x[:2])
    return df


def convert_rawdata_to_traindata(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = add_ymd(train)
    test = add_ymd(test)

    train = add_splited_spc_common(train)
    test = add_splited_spc_common(test)

    train = add_splited_spc_latin(train)
    test = add_splited_spc_latin(test)

    #train = add_splited_nta(train)
    #test = add_splited_nta(test)

    train = convert_str_to_float(train)
    test = convert_str_to_float(test)
    
    train_and_test = pd.concat([train.drop(["health"], axis=1), test], ignore_index=True)
    for col in ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
                "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
                "spc_common_pre", "spc_common_post",
                "spc_latin_pre", "spc_latin_post"]:
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
import re


parse_problem_prefix_list = ["Branch", "Metal", "Root", "Stones", "Trunk", "Wires"]


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def get_problems_prefix_index(x: str, problem_prefix: str) -> int:
    camel_case_splited_x = camel_case_split(x)
    if problem_prefix in camel_case_splited_x:
        return camel_case_splited_x.index(problem_prefix)
    else:
        return np.nan


def add_parse_problems(df: pd.DataFrame) -> pd.DataFrame:
    df[f"problems_count"] = df["problems"].fillna("").apply(
        lambda x: len(camel_case_split(x))
    )
    """
    for problem_prefix in parse_problem_prefix_list:
        df[f"problems_has_{problem_prefix}"] = df["problems"].fillna("").apply(
            lambda x: int(problem_prefix in x)
        )
        df[f"problems_has_{problem_prefix}_index"] = df["problems"].fillna("").apply(
            lambda x: get_problems_prefix_index(x, problem_prefix)
        )
        #df[f"problems_has_{problem_prefix}_index"] = df[f"problems_has_{problem_prefix}_index"].fillna(
        #    df[f"problems_has_{problem_prefix}_index"].max() + 1
        #)
        df[f"problems_has_{problem_prefix}_count"] = df["problems"].fillna("").apply(
            lambda x: camel_case_split(x).count(problem_prefix)
        )
    """
    return df

train = add_parse_problems(train)
test = add_parse_problems(test)

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
            "spc_common_pre", "spc_common_post",
            "spc_latin_pre", "spc_latin_post"]
qua_cols = ["tree_dbh", "steward_float", "created_at_float", "problems_count"]
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
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train = make_ratio(train, cat_cols)
    test = make_ratio(test, cat_cols)

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post",
     "spc_latin_pre", "spc_latin_post"]


#$ os.makedirs(f"../models/{Config.experiment_name}", exist_ok=True)

oof_preds_lgb = np.zeros((len(train), 3))

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
        "health",
        ["health", "fold"],
        cat_cols,
        {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "custom",
            "learning_rate": 0.1, # 0.01
            "seed": Config.seed,
            "verbose": -1,
            # "is_unbalance": True,
        },
        f"../models/{Config.experiment_name}"
    )
    # """

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
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight


print(
    f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro'),
    log_loss(train["health"], oof_preds_lgb, sample_weight=compute_sample_weight(class_weight="balanced", y=train["health"].values)),
)

# %% [markdown]
# - base (20231231_01): 0.3608654717126763
# 
# lr = 0.01 -> 0.1
# 
# - use custom metric and dataset weight: 0.35700974849873685 1.9460744366997482
# - use metric=None and dataset weight: 0.3486495152424373 2.262327578821491
# - use auc_mu metric and dataset weight: 0.3486495152424373 2.262327578821491
# - use metric=None, is_unbalance=True, and dataset weight: 0.3486495152424373 2.262327578821491
# - use metric=None, is_unbalance=True: 0.29384823469054616 1.759782256919533
# - use metric=custom and is_unbalance=True: 0.32317611015670983 4.830897121247583
# 
# - use metric=custom and dataset weight (include valid): 0.35700974849873685 1.9460744366997482
# - use metric=None and dataset weight (include valid): 0.29185562320176606 1.0965846089866713
# - use metric=custom, is_unbalance=True, and dataset weight (include valid): 0.35700974849873685 1.9460744366997482
# 

# %%
cat_cols = ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "nta",
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post",
     "spc_latin_pre", "spc_latin_post"]


oof_preds_ctb = np.zeros((len(train), 3))
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
        "ctb",
        "health",
        ["health", "fold"],
        cat_cols,
        {
            "objective": "MultiClass",  # "MultiClass",
            "loss_function": "CrossEntropy",  # "CrossEntropy",
            "eval_metric": "TotalF1:average=Macro;use_weights=false", # ;use_weights=false",
            "num_boost_round": 10_000,
            "early_stopping_rounds": 1_000,
            "learning_rate": 0.1, # 0.01,
            "verbose": 1_000,
            "random_seed": Config.seed,
            "task_type": "GPU",
            "class_weights": [1000/3535, 1000/15751, 1000/698],
            # "auto_class_weights": 'Balanced',
        },
        f"../models/{Config.experiment_name}",
    )
    # """


    oof_preds_ctb += eval_folds_v3(
        _train,
        [fold_idx],
        Config.seed,
        "ctb",
        "health",
        ["health", "fold"],
        cat_cols,
        f"../models/{Config.experiment_name}"
    )

score = f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average='macro')
print(score)

# %%
def sigmoid(arr, k):
    return 1/(1+np.exp(-k*arr))

# %%
print(
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average='macro'),
    log_loss(train["health"], sigmoid(oof_preds_ctb, 1), sample_weight=compute_sample_weight(class_weight="balanced", y=train["health"].values)),
)

# %% [markdown]
# - base (20231228_02): 0.36431500437373465
# - base (20231231_01): 0.3664672870707581
# 
# lr = 0.01 -> 0.1
# 
# - custom metirc and class weight: 0.3720265447276095 1.1125984083168343
#     - lr = 0.01: 0.3611635018468653 1.0997275991858289
#     - lr = 0.1 (2th): 0.3720265447276095 1.1125971751204538
# - custom metric and class auto balance: 0.3684188268622066 1.1174606999146537
# - metric=None and class weight: 0.27447453366490165 1.0911756735624205
# 
# - custom metric and dataset weight: 0.3677834953838662 1.1155977168004305
# - custom metric, dataset weight, and class weight: 0.32751162115264515 1.1340418809338948
# 
# 

# %%
(
    f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average=None),
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average=None),
)

# %%
train.head(3)

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
def add_te(train, test, col):
    for class_idx in range(3):
        train[f"health_is_{class_idx}"] = (train["health"].to_numpy() == class_idx).astype(int).tolist()
        all_train_df = train.groupby(
            col, as_index=False
        ).agg({f"health_is_{class_idx}": ["sum", "count"]})
        te_dict = (all_train_df[(f"health_is_{class_idx}", 'sum')] / (all_train_df[(f"health_is_{class_idx}", 'count')] + 1)).to_dict()
        test[f"health_is_{class_idx}_te_by_{col}"] = pd.to_numeric(test[col].apply(lambda x: te_dict[x] if x in te_dict.keys() else pd.NA), errors="coerce")
        train = train.drop([f"health_is_{class_idx}"], axis=1)
    return train, test

# %%
import warnings

for col in cat_cols:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _, test = add_te(train, test, col)

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
     "borocode", "boro_ct", "zip_city", "st_assem", "st_senate", "cb_num", "cncldist",
     "spc_common_pre", "spc_common_post",
     "spc_latin_pre", "spc_latin_post"],
    f"../models/{Config.experiment_name}",
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
def inv_sigmoid(arr):
    return np.log(arr / (1 - arr))

def calc_f1(param, lgb_preds, ctb_preds, label):
    arr = np.zeros_like(lgb_preds)
    for i in range(3):
        arr[:, i] = (
            sigmoid(inv_sigmoid(np.array(lgb_preds)[:, i]), param[i * 4]) * param[i * 4 + 1]
            + sigmoid(np.array(ctb_preds)[:, i], param[i * 4 + 2]) * param[i * 4 + 3]
        )
    return -1 * f1_score(label, np.argmax(arr, axis=1), average='macro')

para = [1.0] * 12

m = optimize.minimize(
    calc_f1,
    para,
    args=(oof_preds_lgb.tolist(), oof_preds_ctb.tolist(), train["health"].tolist()),
    method="Nelder-Mead"
)

print(para, m.x, m.fun)
print(m)

# %%
test_preds_ctb_calib = np.zeros_like(y_preds_ctb)
for i in range(3):
    test_preds_ctb_calib[:, i] = sigmoid(np.array(y_preds_ctb)[:, i], cbt_weight[i])

test_preds_ctb_calib /= np.sum(test_preds_ctb_calib, axis=1).reshape(-1, 1)

test_preds_lgb_calib = np.zeros_like(y_preds_lgb)
for i in range(3):
    test_preds_lgb_calib[:, i] = sigmoid(inv_sigmoid(np.array(y_preds_lgb)[:, i]), lgb_weight[i])

test_preds_lgb_calib /= np.sum(test_preds_lgb_calib, axis=1).reshape(-1, 1)

# %%
test_preds = test_preds_lgb_calib * best_weight * 0.1 + test_preds_ctb_calib * (10 - best_weight) * 0.1

submission = pd.read_csv(CSVPath.submission, header=None)
submission.iloc[:, 1] = np.argmax(test_preds, axis=1)
submission.to_csv(f"submission_{Config.experiment_name}.csv", index=False, header=False)

# %%
train["health"].value_counts() / len(train)

# %%
pd.DataFrame(np.argmax(test_preds, axis=1)).value_counts() / len(y_preds_lgb)

# %%



