# splitter.py
"""
Walk-forward splitter cho M5 pipeline.
Expanding window: train luôn bắt đầu từ ngày đầu tiên của data,
test dịch 14 ngày mỗi fold.

Config đọc từ config/split_config.json:
{
    "test_start": "2016-03-01",
    "test_end":   "2016-06-19",
    "horizon":    14,
    "step":       14
}

Output: list of dicts
[
    {
        "fold": 1,
        "train_start": "2011-01-29",
        "train_end":   "2016-02-29",
        "test_start":  "2016-03-01",
        "test_end":    "2016-03-14"
    },
    ...
]
"""
import json
import pandas as pd
from pathlib import Path



BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def get_folds(config_path: str) -> list[dict]:
    """
    Đọc config và tạo walk-forward folds.

    Parameters
    ----------
    config_path : str — đường dẫn tới split_config.json

    Returns
    -------
    folds : list of dict, mỗi dict gồm:
        fold, train_start, train_end, test_start, test_end

    Notes
    -----
    - train_start luôn cố định = ngày đầu tiên trong data
    - train_end   = test_start - 1 ngày
    - test window dịch chuyển theo step mỗi fold
    - Khoảng 7-8 folds với config hiện tại
    """

    config = json.load(open(config_path))
    test_start = pd.to_datetime(config['test_start'])
    test_end   = pd.to_datetime(config['test_end'])
    horizon    = pd.Timedelta(days=config['horizon'])
    step       = pd.Timedelta(days=config['step'])
    folds = []
    fold_num = 1

    df = pd.read_parquet(DATA_PROCESSED / "sales_clean.parquet",columns=["date"])
    train_start = df["date"].min().strftime("%Y-%m-%d")
    del df

    while test_start <= test_end:
        train_end = test_start - pd.Timedelta(days=1)
        

        folds.append({
            "fold": fold_num,
            "train_start": train_start,
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": min(test_start + horizon - pd.Timedelta(days=1), test_end).strftime("%Y-%m-%d")
        })
        test_start += step
        fold_num += 1

    
    return folds

def run_splitter():
    """
    
    1. Gọi get_folds()
    2. In ra tất cả folds để verify
    3. Lưu folds ra config/folds.json để tham khảo
    """

        
    # 1. Gọi get_folds()
    folds = get_folds(CONFIG_DIR / "split_config.json")
    # 2. In ra tất cả folds để verify
    for fold in folds:
        print(fold)
    
    # 3. Lưu folds ra config/folds.json để tham khảo
    with open(CONFIG_DIR / "folds.json", "w") as f:
        json.dump(folds, f, indent=4)
    assert 7 <= len(folds) <= 8, f"Expected 7-8 folds, got {len(folds)}"
    print(f"Tổng số folds: {len(folds)}")
    pass

if __name__ == "__main__":
    run_splitter()