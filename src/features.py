import pandas as pd


def make_ratio(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col + "_ratio"] = df[col].map(df[col].value_counts(dropna=False)) / len(df)
    return df


def make_statvalue(
    df: pd.DataFrame, cat_cols: list[str], qua_cols: list[str], types: list[str]
) -> pd.DataFrame:
    for cat_col in cat_cols:
        for qua_col in qua_cols:
            for typ in types:
                df[cat_col + "_" + typ + "_" + qua_col] = df.groupby([cat_col])[
                    qua_col
                ].transform(typ)
                df[cat_col + "_" + typ + "_" + qua_col + "_diff"] = (
                    df[cat_col + "_" + typ + "_" + qua_col] - df[qua_col]
                )
                df[cat_col + "_" + typ + "_" + qua_col + "_ratio"] = df[
                    cat_col + "_" + typ + "_" + qua_col
                ] / (df[qua_col] + 1e-6)
    return df
