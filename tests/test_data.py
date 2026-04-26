# test_data.py
"""
Test toàn bộ output của ETL pipeline .

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
# Test 2 — Data Leakage
# --------------------------------------------------------

class TestLeakage:

    @pytest.fixture(scope="class")
    def sample_series(self, sales_clean):
        """Lấy 5 series ngẫu nhiên để test leakage."""
        np.random.seed(42)
        pairs = (
            sales_clean[["item_id", "store_id"]]
            .drop_duplicates()
            .sample(n=5, random_state=42)
            .values.tolist()
        )
        return [
            sales_clean[
                (sales_clean["item_id"] == item_id) &
                (sales_clean["store_id"] == store_id)
            ]
            .sort_values("date")
            .reset_index(drop=True)
            for item_id, store_id in pairs
        ]

    def test_rolling_7_no_leakage(self, sample_series):
        for series in sample_series:
            for i in range(10, 30):
                actual = series["rolling_7"].iloc[i]
                if pd.notna(actual):
                    expected = series["sales"].iloc[i-7:i].mean()
                    assert abs(actual - expected) < 1e-3, \
                        f"rolling_7 leakage tại {series['item_id'].iloc[0]} row {i}: "\
                        f"expected {expected:.4f}, got {actual:.4f}"

    def test_rolling_28_no_leakage(self, sample_series):
        for series in sample_series:
            for i in range(30, 50):
                actual = series["rolling_28"].iloc[i]
                if pd.notna(actual):
                    expected = series["sales"].iloc[i-28:i].mean()
                    assert abs(actual - expected) < 1e-3, \
                        f"rolling_28 leakage tại {series['item_id'].iloc[0]} row {i}"

    def test_lag_7_correct(self, sample_series):
        for series in sample_series:
            for i in range(10, 30):
                actual   = series["lag_7"].iloc[i]
                expected = series["sales"].iloc[i - 7]
                if pd.notna(actual):
                    assert abs(actual - expected) < 1e-3, \
                        f"lag_7 sai tại {series['item_id'].iloc[0]} row {i}"

    def test_lag_14_correct(self, sample_series):
        for series in sample_series:
            for i in range(20, 40):
                actual   = series["lag_14"].iloc[i]
                expected = series["sales"].iloc[i - 14]
                if pd.notna(actual):
                    assert abs(actual - expected) < 1e-3, \
                        f"lag_14 sai tại {series['item_id'].iloc[0]} row {i}"

    def test_lag_28_correct(self, sample_series):
        for series in sample_series:
            for i in range(30, 50):
                actual   = series["lag_28"].iloc[i]
                expected = series["sales"].iloc[i - 28]
                if pd.notna(actual):
                    assert abs(actual - expected) < 1e-3, \
                        f"lag_28 sai tại {series['item_id'].iloc[0]} row {i}"

# --------------------------------------------------------
# Test 3 — sell_price: không có backward fill
# --------------------------------------------------------

class TestSellPrice:

    @pytest.fixture(scope="class")
    def sample_series(self, sales_clean):
        """Lấy 10 series ngẫu nhiên để test sell_price."""
        pairs = (
            sales_clean[["item_id", "store_id"]]
            .drop_duplicates()
            .sample(n=10, random_state=42)
            .values.tolist()
        )
        return [
            sales_clean[
                (sales_clean["item_id"] == item_id) &
                (sales_clean["store_id"] == store_id)
            ]
            .sort_values("date")
            .reset_index(drop=True)
            for item_id, store_id in pairs
        ]

    def test_no_backward_fill(self, sample_series):
        for series in sample_series:
            first_valid = series["sell_price"].first_valid_index()
            if first_valid is not None and first_valid > 0:
                before = series["sell_price"].iloc[:first_valid]
                assert before.isna().all(), \
                    f"sell_price backward fill tại "\
                    f"{series['item_id'].iloc[0]}/{series['store_id'].iloc[0]}: "\
                    f"có giá trị trước ngày đầu tiên trong raw data"

    def test_no_negative_price(self, sales_clean):
        valid_prices = sales_clean["sell_price"].dropna()
        assert (valid_prices > 0).all(), \
            "sell_price có giá trị <= 0"

# --------------------------------------------------------
# Test 4 — sampling_profile
# --------------------------------------------------------

class TestSamplingProfile:

    def test_pre_test_date(self, sampling_profile):
        """
        
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
# Test 5 — selected_series
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
        assert len(folds) == 8, \
            f"Du kien 8 ,hien co {len(folds)}"

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