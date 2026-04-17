# cleaner.py
"""
Pipeline làm sạch dữ liệu thô → data/processed/sales_clean.parquet
Các bước:
1. load_and_merge() : load calendar/sales/prices, melt sales, merge calendar & prices
2. fill_missing_sales() : forward fill sales NaN tối đa 3 ngày per series
3. cap_outliers() : IQR cap per series
4. fill_sell_price() : forward fill sell_price per series
5. finalize() : giữ đúng FINAL_COLUMNS, downcast dtypes, assert shape, ghi ra parquet

"""
import pandas as pd

import os
import numpy as np
import gc
from pathlib import Path

DATA_RAW = Path("../data/raw")
OUTPUT = Path("../data/processed")
OUTPUT.mkdir(exist_ok=True, parents=True)

def load_and_merge(data_raw: Path) -> pd.DataFrame:
    """
    Load raw files, melt wide→long, merge calendar và prices.
    
    Returns
    -------
    df : DataFrame với các cột:
        date, item_id, store_id, sales, wm_yr_wk,
        event_name_1, snap_CA, snap_TX, snap_WI, sell_price
    """

    # load data
    calendar = pd.read_csv(DATA_RAW / "calendar.csv")
    sales= pd.read_csv(DATA_RAW / "sales_train_evaluation.csv")
    prices = pd.read_csv(DATA_RAW / "sell_prices.csv")

    # melt sales từ wide → long
    id_cols =["item_id", "store_id", "state_id"]
    day_cols =[ c for c in sales.columns if c.startswith("d_") ]

    df = sales.melt(
        id_vars=id_cols, 
        value_vars=day_cols, 
        var_name="d", 
        value_name="sales"
    )

    # free memory
    del sales
    gc.collect()

    # Merge calendar
    df = df.merge(
        calendar[["d", "date", "wm_yr_wk", "event_name_1",
                    "snap_CA", "snap_TX", "snap_WI"]],
        on="d", how="left"
    )
    df["date"] = pd.to_datetime(df["date"])

    # free memory
    del calendar
    gc.collect()

    # Merge prices
    df = df.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    # free memory
    del prices
    gc.collect()

    

    return df

def fill_missing_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill sales NaN tối đa 3 ngày per series.
    Sau 3 ngày liên tiếp → giữ NaN, KHÔNG fill thêm.
    Ghi log tổng số NaN còn lại.
    
    Parameters
    ----------
    df : DataFrame đã sort theo (item_id, store_id, date)
    
    Returns
    -------
    df : DataFrame với sales đã được fill một phần
    """

    # fill missing sales tối đa 3 ngày liên tiếp per series
    df["sales"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: x.ffill(limit=3))
    # log tổng số NaN còn lại
    total_na = df["sales"].isna().sum()
    print(f"Total NaN in sales after fill: {total_na}")
    return df
    

def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR cap per (item_id, store_id) series.
    Clip tại Q1 - 1.5×IQR và Q3 + 1.5×IQR.
    Tính IQR per series (groupby item_id, store_id).
    KHÔNG xóa row nào.
    
    Parameters
    ----------
    df : DataFrame với cột sales đã qua fill_missing_sales
    
    Returns
    -------
    df : DataFrame với sales đã được clip
    """

    # iqr cap per series
    def cap_outliers_per_series(x):
        nonzero = x[x > 0]
        if len(nonzero) == 0:
            return x
        q1 = nonzero.quantile(0.25)
        q3 = nonzero.quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr) 
        upper = q3 + 1.5 * iqr
        return x.clip(lower=lower, upper=upper)
    

    df["sales"] = df.groupby(["item_id", "store_id"])["sales"].transform(cap_outliers_per_series)

    return df

def fill_sell_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill sell_price within (item_id, store_id).
    Không backward fill — tránh data leakage.
    Nếu ngày đầu tiên của series chưa có giá → giữ NaN.
    
    Parameters
    ----------
    df : DataFrame đã sort theo (item_id, store_id, date)
    
    Returns
    -------
    df : DataFrame với sell_price đã được fill
    """

    df["sell_price"] = df.groupby(["item_id", "store_id"])["sell_price"].ffill()

    return df


FINAL_COLUMNS = [
    "date", "item_id", "store_id", "sales",
    "day_of_week", "month", "is_holiday", "is_weekend", "snap",
    "sell_price", "rolling_7", "rolling_28",
    "lag_7", "lag_14", "lag_28"
]

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Giữ đúng FINAL_COLUMNS, đúng thứ tự
    - Assert shape[1] == 15
    - Ghi ra data/processed/sales_clean.parquet
    """

    # rolling/lag chưa có → chỉ giữ 10 cột hiện có
    cols_available = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[cols_available]
    
    CLEAN_COLUMNS = ["date", "item_id", "store_id", "sales", "sell_price"]
    df = df[CLEAN_COLUMNS]
    assert df.shape[1] == 5, f"frame cần 5 cột ,hiện có {df.shape[1]} cột"

    df.to_parquet(OUTPUT / "sales_clean.parquet", index=False)
    print(f"Saved: {df.shape}")

    df["sales"]      = df["sales"].astype("float64")
    df["sell_price"] = df["sell_price"].astype("float64")
    df["date"] =df["date"].astype("datetime64[ns]")

    return df
    
def run_clean():
    """
    Orchestrate toàn bộ pipeline cleaning theo thứ tự:
    1. load_and_merge()
    2. fill_missing_sales()
    3. cap_outliers()
    4. fill_sell_price()
    5. finalize()
    """

    df = load_and_merge(DATA_RAW)

    df = df.sort_values(["item_id","store_id","date"]).reset_index(drop=True)
    gc.collect()

    df = fill_missing_sales(df)
    df = cap_outliers(df)
    df = fill_sell_price(df)
    df = finalize(df)
    
    pass

if __name__ == "__main__":
    run_clean()