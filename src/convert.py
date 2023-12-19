import pandas as pd
from sklearn.preprocessing import LabelEncoder

def apply_le(
    df: pd.DataFrame, colname: str, keep_nan: bool
) -> pd.DataFrame:
    le = LabelEncoder()
    is_null = df[colname].isnull()
    df[colname] = le.fit_transform(df[colname])
    if keep_nan:
        df.loc[is_null, colname] = None
    df[colname] = pd.Series(df[colname], dtype="Int64")
    return df
