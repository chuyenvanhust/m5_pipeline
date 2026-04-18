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

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT = PROJECT_ROOT / "data" / "processed"
OUTPUT.mkdir(exist_ok=True, parents=True)

def load_and_merge(data_raw: Path) -> pd.DataFrame:
    """
    Load raw files, melt wide→long, merge calendar và prices.
    
    Returns
    -------
    df : DataFrame với các cột:
        date, item_id, store_id,state_id, sales, wm_yr_wk,weekday,month
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
        calendar[["d", "date", "wm_yr_wk","weekday","month", "event_name_1",
                    "snap_CA", "snap_TX", "snap_WI"]],
        on="d", how="left"
    )
    df["date"] = pd.to_datetime(df["date"])

    df = df.drop(columns="d")

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
        "date", "item_id", "store_id", "state_id", "sales", "sell_price",
        "wm_yr_wk", "weekday", "month", "event_name_1", 
        "snap_CA", "snap_TX", "snap_WI"
    ]

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Assert shape[1] == 13
    - Ghi ra data/processed/sales_clean.parquet
    """

    
    df = df[FINAL_COLUMNS]
    
    
    assert df.shape[1] == 13, f"frame cần 13 cột ,hiện có {df.shape[1]} cột"

    df["sales"]      = df["sales"].astype("float64")
    df["sell_price"] = df["sell_price"].astype("float64")
    df["date"] =df["date"].astype("datetime64[ns]")

    df.to_parquet(OUTPUT / "sales_clean.parquet", index=False)
    print(f"Saved: {df.shape}")

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
    print(f"buoc 1")
    df = load_and_merge(DATA_RAW)

    
    df = df.sort_values(["item_id","store_id","date"]).reset_index(drop=True)
    gc.collect()

    print(f"buoc 2")
    df = fill_missing_sales(df)

    print(f"buoc 3")
    df = cap_outliers(df)

    print(f"buoc 4")
    df = fill_sell_price(df)

    print(f"buoc 5")
    df = finalize(df)
    
    pass

def run_batch_clean():
    """
        chạy theo batch cho mỗi store
    """
    print("--- chạy theo batch cho mỗi store---")
    
    
    
    calendar = pd.read_csv(DATA_RAW / "calendar.csv")
    prices = pd.read_csv(DATA_RAW / "sell_prices.csv")
    sales_raw = pd.read_csv(DATA_RAW / "sales_train_evaluation.csv")
    
    
    stores = sales_raw['store_id'].unique()
    all_chunks = []

    
    for store in stores:
        print(f"\n>>> Đang xử lý Batch cho Store: {store}")
        
        
        df_store = sales_raw[sales_raw['store_id'] == store].copy()
        
        id_cols = ["item_id", "store_id", "state_id"]
        day_cols = [c for c in df_store.columns if c.startswith("d_")]
        df_chunk = df_store.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
        
        del df_store; gc.collect()

       
        df_chunk = df_chunk.merge(
            calendar[["d", "date", "wm_yr_wk", "weekday", "month", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]],
            on="d", how="left"
        )
        df_chunk["date"] = pd.to_datetime(df_chunk["date"])
        df_chunk.drop(columns="d", inplace=True)

        
        prices_store = prices[prices['store_id'] == store]
        df_chunk = df_chunk.merge(
            prices_store[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
            on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )
        del prices_store; gc.collect()

       
        df_chunk.sort_values(["item_id", "store_id","date"], inplace=True)
        
      
        df_chunk = fill_missing_sales(df_chunk)
        df_chunk = cap_outliers(df_chunk)
        df_chunk = fill_sell_price(df_chunk)
        
        

        all_chunks.append(df_chunk)
        print(f"Hoàn thành Batch {store}. RAM đang sử dụng sẽ được giải phóng...")
        gc.collect()

    # gộp frame
    print("\n--- Đang gộp các cửa hàng thành file tổng ---")
    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df = full_df.sort_values(["item_id","store_id","date"]).reset_index(drop=True)
    del all_chunks; gc.collect()

    #  finalize
    full_df = finalize(full_df)


if __name__ == "__main__":
    #run_clean()
    run_batch_clean()