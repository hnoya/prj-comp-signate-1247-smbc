# %%
from dataclasses import dataclass

import pandas as pd

# %%
import sys
sys.path.append("../")

from src.utils import seed_everything
from src.convert import apply_le

# %%

@dataclass
class Config:
    experiment_name: str = "20231220_01"
    seed: int = 0

@dataclass
class CSVPath:
    train: str = "/work/data/train.csv"
    test: str = "/work/data/test.csv"
    submission: str = "/work/data/sample_submission.csv"


seed_everything(Config.seed)

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
        train_and_test = apply_le(train_and_test, col, keep_nan=True)
        train[col] = train_and_test[col].iloc[:len(train)]
        test[col] = train_and_test[col].iloc[-len(test):]

    return train, test

# %%
train, test = convert_rawdata_to_traindata(train, test)

# %%
train.head(3)


