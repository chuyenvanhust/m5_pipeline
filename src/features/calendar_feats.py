# calendar_feats.py
"""
Thêm calendar features vào sales_clean.parquet:
- day_of_week : int8, 0=Monday
- is_holiday  : int8, 0/1
- is_weekend  : int8, 0/1
- snap        : int8, 0/1
- drop các cột trung gian cột date, event_name_1, state_id, snap_CA/TX/WI sau khi tính xong
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính day_of_week, is_holiday, is_weekend, snap
    từ các cột date, event_name_1, state_id, snap_CA/TX/WI.

    Parameters
    ----------
    df : DataFrame từ sales_clean.parquet (13 cột)

    Returns
    -------
    df : DataFrame với 4 cột mới, drop các cột trung gian
    """
    # Tính day_of_week (Monday =0 ,0-6)
    df['day_of_week'] = df['date'].dt.weekday.astype('int8')
    
    # Tính is_holiday
    df['is_holiday'] = np.where(df['event_name_1'].notna(), 1, 0)
    # Tính is_weekend
    df['is_weekend'] = np.where(df['day_of_week'].isin([5, 6]), 1, 0)
    # Tính snap
    df['snap'] = np.where(
        (df['state_id'] == 'CA') & (df['snap_CA'] == 1) |
        (df['state_id'] == 'TX') & (df['snap_TX'] == 1) |
        (df['state_id'] == 'WI') & (df['snap_WI'] == 1),
        1, 0
    )
    # Drop các cột trung gian
    df.drop(columns=['weekday', 'wm_yr_wk', 'event_name_1',
                 'state_id', 'snap_CA', 'snap_TX', 'snap_WI'],
        inplace=True)

    return df

def run_calendar():
    """
    1. Đọc sales_clean.parquet
    2. Gọi add_calendar_features()
    3. Assert đúng cột
    4. Ghi đè sales_clean.parquet
    """

    # 1. Đọc sales_clean.parquet
    df_schema = pd.read_parquet(DATA_PROCESSED / "sales_clean.parquet", engine='pyarrow')

    # 2. Gọi add_calendar_features()
    df_with_calendar = add_calendar_features(df_schema)

    # 3. Assert đúng cột
    expected_columns = ['day_of_week', 'is_holiday', 'is_weekend', 'snap']
    assert all(col in df_with_calendar.columns for col in expected_columns), "Cột calendar không đúng"

    # ep kieu
    df_with_calendar['day_of_week'] = df_with_calendar['day_of_week'].astype('int8')
    df_with_calendar['is_holiday'] = df_with_calendar['is_holiday'].astype('int8')
    df_with_calendar['is_weekend'] = df_with_calendar['is_weekend'].astype('int8')
    df_with_calendar['snap'] = df_with_calendar['snap'].astype('int8')
    df_with_calendar['month'] = df_with_calendar['date'].dt.month.astype('int8')
    
    # 4. Ghi đè sales_clean.parquet
    df_with_calendar.to_parquet(DATA_PROCESSED / "sales_clean.parquet", engine='pyarrow', index=False)

    print(df_with_calendar.info())

    pass

if __name__ == "__main__":
    run_calendar()