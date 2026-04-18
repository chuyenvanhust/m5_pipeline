# lag.py
"""
Thêm rolling và lag features vào sales_clean.parquet:
- rolling_7  : rolling mean 7 ngày, shift(1)
- rolling_28 : rolling mean 28 ngày, shift(1)
- lag_7      : sales cách 7 ngày
- lag_14     : sales cách 14 ngày
- lag_28     : sales cách 28 ngày

Tất cả features PHẢI dùng .shift(1) — chỉ dùng data đến ngày t-1
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính rolling_7, rolling_28, lag_7, lag_14, lag_28
    per (item_id, store_id) với groupby trước khi tính.

    Parameters
    ----------
    df : DataFrame đã có calendar features (17 cột)

    Returns
    -------
    df : DataFrame với 5 cột lag/rolling mới (22 cột)

    Notes
    -----
    - groupby(['item_id','store_id']) TRƯỚC khi tính
    - rolling dùng .shift(1) để tránh data leakage
    - lag dùng .shift(N)
    """
    # Group by item_id và store_id
    df = df.sort_values(['item_id', 'store_id', 'date'])
    df_grouped = df.groupby(['item_id', 'store_id'])

    # Tính rolling features
    df['rolling_7'] = (
    df.groupby(['item_id', 'store_id'])['sales']
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)
    df['rolling_28'] = (
        df.groupby(['item_id', 'store_id'])['sales']
        .transform(lambda x: x.shift(1).rolling(window=28).mean())
    )

    # Tính lag features
    df['lag_7']  = df.groupby(['item_id','store_id'])['sales'].transform(lambda x: x.shift(7))
    df['lag_14'] = df.groupby(['item_id','store_id'])['sales'].transform(lambda x: x.shift(14))
    df['lag_28'] = df.groupby(['item_id','store_id'])['sales'].transform(lambda x: x.shift(28))


    return df

def run_lag():
    """
    1. Đọc sales_clean.parquet
    2. Gọi add_lag_features()
    3. Assert đúng cột
    4. Ghi đè sales_clean.parquet
    """

    #1. Đọc sales_clean.parquet
    df = pd.read_parquet(DATA_PROCESSED / "sales_clean.parquet", engine='pyarrow')
    #2. Gọi add_lag_features()
    df_with_lag = add_lag_features(df)
    #3. Assert đúng cột
    FINAL_COLUMNS = [
        "date", "item_id", "store_id", "sales",
        "day_of_week", "month", "is_holiday", "is_weekend", "snap",
        "sell_price", "rolling_7", "rolling_28",
        "lag_7", "lag_14", "lag_28"
    ]
    
    df_with_lag = df_with_lag[FINAL_COLUMNS]
    
    assert list(df_with_lag.columns) == FINAL_COLUMNS, \
        f"Schema sai: {list(df_with_lag.columns)}"
    assert df_with_lag.shape[1] == 15, \
        f"Expected 15 cols, got {df_with_lag.shape[1]}"
    
    #ep kieu
    df_with_lag['sales'] = df_with_lag['sales'].astype('float64')
    df_with_lag['rolling_7'] = df_with_lag['rolling_7'].astype('float64')
    df_with_lag['rolling_28'] = df_with_lag['rolling_28'].astype('float64')
    df_with_lag['lag_7'] = df_with_lag['lag_7'].astype('float64')
    df_with_lag['lag_14'] = df_with_lag['lag_14'].astype('float64')
    df_with_lag['lag_28'] = df_with_lag['lag_28'].astype('float64')
    df_with_lag['sell_price'] = df_with_lag['sell_price'].astype('float64')
    #4. Ghi đè sales_clean.parquet
    df_with_lag.to_parquet(
        DATA_PROCESSED / "sales_clean.parquet",
        engine='pyarrow',
        index=False
    )
    print(f"Saved: {df_with_lag.shape}")
    print(df_with_lag.info())

    pass

if __name__ == "__main__":
    run_lag()