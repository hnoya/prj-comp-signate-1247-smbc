# %%
from dataclasses import dataclass
import os

import pandas as pd

# %%
import sys
sys.path.append("../")

from src.utils import seed_everything
from src.convert import apply_le
from src.validate import validate
from src.models import train_folds_v2, eval_folds_v2, predict_catboost

# %%

@dataclass
class Config:
    experiment_name: str = "20231220_01"
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
import seaborn as sns

sns.kdeplot(train["created_at_float"], color="blue")
sns.kdeplot(test["created_at_float"], color="red")

# %%
for col in ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"]:
    train_set = set(train[col].tolist())
    test_set = set(test[col].tolist())
    if len(test_set - train_set):
        print(col, test_set - train_set)

# %%
train["fold"] = validate("KFold", train, Config.n_fold, train["health"].tolist(), random_state=Config.seed, shuffle=True)

# %%
train.head(3)

# %%
train["health"].value_counts()

# %%
"""
train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"],
    f"../models/{Config.experiment_name}"
)
"""

# %%
oof_preds_ctb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "ctb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"],
    f"../models/{Config.experiment_name}"
)

# %%
import numpy as np
from sklearn.metrics import f1_score

print(
    f1_score(train["health"], np.argmax(oof_preds_ctb, axis=1), average='macro')
)


# %% [markdown]
# - TotalF1: 0.29437209999353947 (ctb: ~0.70)
# - TotalF1 with classweight: 0.3009543918454867 (ctb: ~0.40)
# - TotalF1(macro): 0.29437209999353947 (ctb: ~0.29)
# - TotalF1:average=Macro;use_weights=false: error
# - TotalF1(macro) with class_weights: 0.3010023251674656 (ctb: ~0.40)
# - TotalF1:average=Macro;use_weights=false with class_weight: 0.3358520966838716 (ctb: ~0.33)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

feature_importance = None
for i in range(Config.n_fold):
    model = pickle.load(
        open(f"/work/models/20231220_01/ctb_fold{i}.ctbmodel", "rb")
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
y_preds = predict_catboost(
    test[[col for col in test.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"],
    f"../models/{Config.experiment_name}",
)

# %%


# %%
"""
train_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"],
    f"../models/{Config.experiment_name}"
)
"""

# %%
oof_preds_lgb = eval_folds_v2(
    train,
    list(range(Config.n_fold)),
    Config.seed,
    "lgb",
    "health",
    ["health", "fold"],
    ["curb_loc", "guards", "sidewalk", "user_type", "problems", "spc_common", "spc_latin", "nta", "nta_name", "boroname", "zip_city"],
    f"../models/{Config.experiment_name}"
)

# %%
import numpy as np
from sklearn.metrics import f1_score

print(
    f1_score(train["health"], np.argmax(oof_preds_lgb, axis=1), average='macro')
)

# %%
def predict_lightgbm(
    X_test: pd.DataFrame, folds: list[int], model_path: str = "../models"
) -> np.ndarray:
    y_pred = np.zeros((X_test.shape[0], 3), dtype="float32")
    for fold in folds:
        model = pickle.load(
            open(os.path.join(model_path, "lgb_fold{}.lgbmodel".format(fold)), "rb")
        )
        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
    return y_pred

# %%
y_preds = predict_lightgbm(
    test[[col for col in test.columns if col not in ["health", "fold"]]],
    list(range(Config.n_fold)),
    f"../models/{Config.experiment_name}",
)

# %%


# %%
def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

# %%
oof_preds_ctb_calib = sigmoid(oof_preds_ctb)
oof_preds_ctb_calib /= np.sum(oof_preds_ctb_calib, axis=1).reshape(-1, 1)

# %%
np.sum(oof_preds_lgb, axis=1), np.sum(oof_preds_ctb_calib, axis=1)

# %%
import numpy as np
from sklearn.metrics import f1_score

for i in range(10):
    print(
        i, f1_score(train["health"], np.argmax(oof_preds_lgb * i * 0.1 + oof_preds_ctb * (10 - i) * 0.1, axis=1), average='macro')
    )


