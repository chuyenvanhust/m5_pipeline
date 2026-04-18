# profiler.py
"""
Tính series profile từ sales_clean.parquet:
- sampling_profile.csv  : tính trên data < 2016-03-01 (pre-test)
- analysis_profile.csv  : tính trên toàn bộ data

Columns:
    item_id, store_id, mean_sales, std_sales, cv,
    zero_ratio, missing_ratio, trend_slope,
    total_days, demand_type, profile_scope
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

PRE_TEST_DATE = "2016-03-01"

def compute_profile(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    Tính profile metrics cho từng (item_id, store_id) series.

    Parameters
    ----------
    df    : DataFrame sales đã filter theo scope
    scope : "pre_test" hoặc "full"

    Returns
    -------
    profile : DataFrame với columns:
        item_id, store_id, mean_sales, std_sales, cv,
        zero_ratio, missing_ratio, trend_slope,
        total_days, demand_type, profile_scope

    Notes
    -----
    - cv = std / mean. Nếu mean_sales < 0.01 → cv = NaN
    - demand_type = "regular" nếu zero_ratio < 0.3
                    "intermittent" nếu zero_ratio >= 0.3
    - trend_slope : scipy.stats.linregress(day_index, sales).slope
                    day_index = 0, 1, 2, ...
                    Bỏ NaN trước khi tính
    """
    # Group by item_id và store_id
    profile = df.groupby(['item_id', 'store_id']).agg(
        mean_sales=('sales', 'mean'),
        std_sales=('sales', 'std'),
        zero_ratio=('sales', lambda x: (x == 0).mean()),
        missing_ratio=('sales', lambda x: x.isna().mean()),
        total_days=('sales', 'size')
    ).reset_index()
    # Tính cv
    profile['cv'] = profile.apply(lambda row: row['std_sales'] / row['mean_sales'] if row['mean_sales'] >= 0.01 else np.nan, axis=1)
    # Tính demand_type
    profile['demand_type'] = np.where(profile['zero_ratio'] < 0.3, 'regular', 'intermittent')
    # Tính trend_slope
    def compute_trend(group):
        group = group.sort_values('date')
        sales = group['sales'].values
        day_index = np.arange(len(sales))
        # Bỏ NaN
        mask = ~np.isnan(sales)
        if mask.sum() < 2:  # Nếu còn ít hơn 2 điểm sau khi bỏ NaN → slope = 0
            return 0.0
        slope, _, _, _, _ = stats.linregress(day_index[mask], sales[mask])
        return slope
    trend_slopes = (
        df.groupby(['item_id', 'store_id'])
        .apply(compute_trend, include_groups=False)
        .reset_index(name='trend_slope')
    )
    profile = profile.merge(trend_slopes, on=['item_id', 'store_id'])
    # Thêm profile_scope
    profile['profile_scope'] = scope
    

    return profile

def run_profile():
    """
    1. Đọc sales_clean.parquet
    2. Tính sampling_profile  : df[df.date < PRE_TEST_DATE]
    3. Tính analysis_profile  : toàn bộ df
    4. Assert đúng cột
    5. Ghi ra data/processed/
    """

    # 1. Đọc sales_clean.parquet
    df = pd.read_parquet(DATA_PROCESSED / "sales_clean.parquet", engine='pyarrow')

    # 2. Tính sampling_profile  : df[df.date < PRE_TEST_DATE]
    df_sampling = compute_profile(df[df['date'] < PRE_TEST_DATE], scope="pre_test")

    # 3. Tính analysis_profile  : toàn bộ df
    df_analysis = compute_profile(df, scope="full")

    # 4. Assert đúng cột
    FINAL_COLUMNS = [
        "item_id", "store_id", "mean_sales",
        "std_sales", "cv", "zero_ratio",
        "missing_ratio", "trend_slope",
        "total_days", "demand_type", "profile_scope"
    ]
    
    df_sampling = df_sampling[FINAL_COLUMNS]
    df_analysis = df_analysis[FINAL_COLUMNS]
    
    assert list(df_sampling.columns) == FINAL_COLUMNS,f"Schema sai: {list(df_sampling.columns)}"
    assert df_sampling.shape[1] == 11,  f"Expected 11 cols, got {df_sampling.shape[1]}"
    assert list(df_analysis.columns) == FINAL_COLUMNS,f"Schema sai: {list(df_analysis.columns)}"
    assert df_analysis.shape[1] == 11,  f"Expected 11 cols, got {df_analysis.shape[1]}"

    #4. Ghi ra data/processed/
    df_sampling.to_csv(DATA_PROCESSED / "sampling_profile.csv", index=False)
    df_analysis.to_csv(DATA_PROCESSED / "analysis_profile.csv", index=False)

    print("xong profile")

    pass

if __name__ == "__main__":
    run_profile()