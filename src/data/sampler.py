# sampler.py
"""
Stratified sampling 100 series từ sampling_profile.csv
→ config/selected_series.csv

Stratification:
    Bước 1: chia theo demand_type (regular / intermittent)
    Bước 2: trong mỗi nhóm, chia cv thành 3 tertiles (low/mid/high)
            và mean_sales thành 3 tertiles
    Bước 3: sample proportionally từ mỗi stratum
    Bước 4: đảm bảo 20-30 series intermittent trong 100

random_state = 42
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR     = PROJECT_ROOT / "config"

N_SERIES       = 100
RANDOM_STATE   = 42
MIN_INTERMITTENT = 20
MAX_INTERMITTENT = 30

def stratified_sample(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified sampling từ sampling_profile.

    Parameters
    ----------
    profile : DataFrame từ sampling_profile.csv

    Returns
    -------
    selected : DataFrame 100 rows với columns:
        item_id, store_id

    Notes
    -----
    - Drop rows có cv = NaN trước khi sample
    - Đảm bảo MIN_INTERMITTENT <= n_intermittent <= MAX_INTERMITTENT
    - random_state = RANDOM_STATE
    """
    profile = profile.dropna(subset=['cv']).copy()

    profile['cv_tertile'] = pd.qcut(
        profile['cv'], q=3,
        labels=['low','mid','high'],
        duplicates='drop'
    )
    profile['mean_tertile'] = pd.qcut(
        profile['mean_sales'], q=3,
        labels=['low','mid','high'],
        duplicates='drop'
    )

    regular      = profile[profile['demand_type'] == 'regular']
    intermittent = profile[profile['demand_type'] == 'intermittent']

    
    N_INTERMITTENT = 25
    N_REGULAR      = N_SERIES - N_INTERMITTENT  # 75

    
    intermittent = intermittent.copy()
    intermittent['stratum'] = intermittent['cv_tertile'].astype(str)
    inter_counts = intermittent['stratum'].value_counts()
    inter_props  = (inter_counts / inter_counts.sum() * N_INTERMITTENT).round().astype(int)
    
    diff = N_INTERMITTENT - inter_props.sum()
    if diff != 0:
        inter_props[inter_props.idxmax()] += diff

    selected_inter = (
        intermittent
        .groupby('stratum', group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(inter_props.get(x.name, 0), len(x)),
                random_state=RANDOM_STATE
            ),
            include_groups=False
        )
    )

    
    regular = regular.copy()
    regular['stratum'] = (
        regular['cv_tertile'].astype(str) + '_' +
        regular['mean_tertile'].astype(str)
    )
    reg_counts = regular['stratum'].value_counts()
    reg_props  = (reg_counts / reg_counts.sum() * N_REGULAR).round().astype(int)
    
    diff = N_REGULAR - reg_props.sum()
    if diff != 0:
        reg_props[reg_props.idxmax()] += diff

    selected_reg = (
        regular
        .groupby('stratum', group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(reg_props.get(x.name, 0), len(x)),
                random_state=RANDOM_STATE
            ),
            include_groups=False
        )
    )

    selected = pd.concat(
        [selected_inter, selected_reg]
    )[['item_id','store_id']].reset_index(drop=True)

    
    n_inter = selected.merge(
        profile[['item_id','store_id','demand_type']],
        on=['item_id','store_id']
    )['demand_type'].eq('intermittent').sum()

    assert MIN_INTERMITTENT <= n_inter <= MAX_INTERMITTENT, f"Intermittent = {n_inter}, cần {MIN_INTERMITTENT}–{MAX_INTERMITTENT}"

    return selected

def run_sample():
    """
    1. Đọc sampling_profile.csv
    2. Gọi stratified_sample()
    3. Assert shape == (100, 2)
    4. Assert không có duplicate (item_id, store_id)
    5. Ghi ra config/selected_series.csv
    6. In phân phối demand_type để gửi approve
    """
    
    profile = pd.read_csv(DATA_PROCESSED / "sampling_profile.csv")
    
    selected = stratified_sample(profile)

    
    assert selected.shape == (N_SERIES, 2), f"Expected shape {(N_SERIES, 2)}, got {selected.shape}"

    
    assert not selected.duplicated(subset=['item_id', 'store_id']).any(), "Duplicate (item_id, store_id) found"

    
    selected.to_csv(CONFIG_DIR / "selected_series.csv", index=False)

    
    demand_dist = profile.merge(selected, on=['item_id', 'store_id'])['demand_type'].value_counts()
    print("Distribution of demand_type in selected sample:")
    print(demand_dist)

    pass

if __name__ == "__main__":
    run_sample()