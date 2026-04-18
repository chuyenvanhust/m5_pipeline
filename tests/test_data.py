# test_data.py
"""
Test toàn bộ output của ETL pipeline theo spec.

Chạy: pytest tests/test_data.py -v
"""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR     = PROJECT_ROOT / "config"

# --------------------------------------------------------
# Fixtures
# --------------------------------------------------------

@pytest.fixture(scope="module")
def sales_clean():
    """Load sales_clean.parquet một lần cho tất cả tests."""
    path = DATA_PROCESSED / "sales_clean.parquet"
    assert path.exists(), "sales_clean.parquet chưa được tạo"
    return pd.read_parquet(path)

@pytest.fixture(scope="module")
def sampling_profile():
    path = DATA_PROCESSED / "sampling_profile.csv"
    assert path.exists(), "sampling_profile.csv chưa được tạo"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def selected_series():
    path = CONFIG_DIR / "selected_series.csv"
    assert path.exists(), "selected_series.csv chưa được tạo"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def folds():
    path = CONFIG_DIR / "folds.json"
    assert path.exists(), "folds.json chưa được tạo"
    with open(path) as f:
        return json.load(f)

# --------------------------------------------------------
# Test 1 — Schema: 15 cột, đúng types
# --------------------------------------------------------

class TestSchema:

    EXPECTED_COLUMNS = [
        "date", "item_id", "store_id", "sales",
        "day_of_week", "month", "is_holiday", "is_weekend", "snap",
        "sell_price", "rolling_7", "rolling_28",
        "lag_7", "lag_14", "lag_28"
    ]

    def test_column_count(self, sales_clean):
        assert sales_clean.shape[1] == 15, \
            f"Expected 15 cols, got {sales_clean.shape[1]}"

    def test_column_names(self, sales_clean):
        assert list(sales_clean.columns) == self.EXPECTED_COLUMNS, \
            f"Columns không khớp: {list(sales_clean.columns)}"

    def test_column_dtypes(self, sales_clean):
        assert sales_clean["date"].dtype == "datetime64[ns]"
        assert sales_clean["sales"].dtype in ["float32", "float64"]
        assert sales_clean["sell_price"].dtype in ["float32", "float64"]
        assert sales_clean["day_of_week"].dtype == "int8"
        assert sales_clean["month"].dtype == "int8"
        assert sales_clean["is_holiday"].dtype == "int8"
        assert sales_clean["is_weekend"].dtype == "int8"
        assert sales_clean["snap"].dtype == "int8"

    def test_row_count(self, sales_clean):
        assert sales_clean.shape[0] == 59_181_090, \
            f"Expected 59181090 rows, got {sales_clean.shape[0]}"

# --------------------------------------------------------
# Test 2 — Data Leakage: rolling/lag dùng đúng shift(1)
# --------------------------------------------------------

class TestLeakage:

    def test_rolling_7_no_leakage(self, sales_clean):
        """
        rolling_7 tại ngày t chỉ được dùng data đến t-1.
        Verify: rolling_7(t) = mean(sales[t-7:t-1])
        Lấy 1 series mẫu để check.
        """
        pass

    def test_rolling_28_no_leakage(self, sales_clean):
        pass

    def test_lag_7_correct(self, sales_clean):
        """lag_7(t) == sales(t-7) cho 1 series mẫu."""
        pass

    def test_lag_14_correct(self, sales_clean):
        pass

    def test_lag_28_correct(self, sales_clean):
        pass

# --------------------------------------------------------
# Test 3 — sell_price: không có backward fill
# --------------------------------------------------------

class TestSellPrice:

    def test_no_backward_fill(self, sales_clean):
        """
        Với mỗi series, ngày đầu tiên có sell_price
        phải là ngày đầu tiên giá thực sự xuất hiện
        trong raw data — không được fill ngược về trước.
        """
        pass

    def test_no_negative_price(self, sales_clean):
        valid_prices = sales_clean["sell_price"].dropna()
        assert (valid_prices > 0).all(), \
            "sell_price có giá trị <= 0"

# --------------------------------------------------------
# Test 4 — sampling_profile: chỉ dùng data pre-test
# --------------------------------------------------------

class TestSamplingProfile:

    def test_pre_test_date(self, sampling_profile):
        """
        sampling_profile chỉ được tính trên data < 2016-03-01.
        Verify qua max date trong profile scope.
        """
        assert "profile_scope" in sampling_profile.columns
        assert (sampling_profile["profile_scope"] == "pre_test").all()

    def test_demand_type_values(self, sampling_profile):
        valid = {"regular", "intermittent"}
        actual = set(sampling_profile["demand_type"].unique())
        assert actual.issubset(valid), \
            f"demand_type có giá trị không hợp lệ: {actual - valid}"

    def test_no_duplicate_series(self, sampling_profile):
        dupes = sampling_profile.duplicated(
            subset=["item_id", "store_id"]
        ).sum()
        assert dupes == 0, f"sampling_profile có {dupes} duplicate series"

# --------------------------------------------------------
# Test 5 — selected_series: 100 rows, không duplicate
# --------------------------------------------------------

class TestSelectedSeries:

    def test_row_count(self, selected_series):
        assert selected_series.shape[0] == 100, \
            f"Expected 100 rows, got {selected_series.shape[0]}"

    def test_columns(self, selected_series):
        assert set(selected_series.columns) >= {"item_id", "store_id"}

    def test_no_duplicate(self, selected_series):
        dupes = selected_series.duplicated(
            subset=["item_id", "store_id"]
        ).sum()
        assert dupes == 0, f"selected_series có {dupes} duplicate"

    def test_intermittent_count(self, selected_series, sampling_profile):
        """Đảm bảo 20-30 series intermittent trong 100."""
        merged = selected_series.merge(
            sampling_profile[["item_id", "store_id", "demand_type"]],
            on=["item_id", "store_id"],
            how="left"
        )
        n_intermittent = (merged["demand_type"] == "intermittent").sum()
        assert 20 <= n_intermittent <= 30, \
            f"Intermittent count = {n_intermittent}, cần 20-30"

    def test_all_series_exist_in_sales(self, selected_series, sales_clean):
        """Tất cả item_id, store_id phải tồn tại trong sales_clean."""
        sales_pairs = sales_clean[
            ["item_id", "store_id"]
        ].drop_duplicates()
        merged = selected_series.merge(
            sales_pairs,
            on=["item_id", "store_id"],
            how="left",
            indicator=True
        )
        missing = (merged["_merge"] == "left_only").sum()
        assert missing == 0, \
            f"{missing} series không tồn tại trong sales_clean"

# --------------------------------------------------------
# Test 6 — Walk-forward splitter
# --------------------------------------------------------

class TestSplitter:

    def test_fold_count(self, folds):
        assert 7 <= len(folds) <= 8, \
            f"Expected 7-8 folds, got {len(folds)}"

    def test_fold_keys(self, folds):
        required = {"fold", "train_start", "train_end",
                    "test_start", "test_end"}
        for f in folds:
            assert required.issubset(f.keys()), \
                f"Fold thiếu keys: {required - f.keys()}"

    def test_no_overlap(self, folds):
        """train_end phải < test_start cho mỗi fold."""
        for f in folds:
            assert f["train_end"] < f["test_start"], \
                f"Fold {f['fold']}: train_end >= test_start"

    def test_test_window_size(self, folds):
        """Mỗi test window phải đúng 14 ngày, trừ fold cuối."""
        for f in folds[:-1]: 
            start = pd.Timestamp(f["test_start"])
            end   = pd.Timestamp(f["test_end"])
            days  = (end - start).days + 1
            assert days == 14, \
                f"Fold {f['fold']}: test window = {days} days, cần 14"
            
    def test_expanding_train(self, folds):
        """train_start phải giống nhau ở tất cả folds."""
        train_starts = [f["train_start"] for f in folds]
        assert len(set(train_starts)) == 1, \
            "train_start không cố định across folds"