# main.py
"""
Entry point cho toàn bộ M5 ETL pipeline.

Thứ tự chạy:
    Tuần 3: cleaner       → data/processed/sales_clean.parquet (13 cột)
    Tuần 4: calendar_feats → sales_clean.parquet (13 cột, swap trung gian)
    Tuần 4: lag            → sales_clean.parquet (15 cột)
    Tuần 5: profiler       → sampling_profile.csv, analysis_profile.csv
    Tuần 5: sampler        → config/selected_series.csv
    Tuần 6: splitter       → config/folds.json

Cách chạy:
    python main.py                  # chạy toàn bộ pipeline
    python main.py --step clean     # chỉ chạy bước clean
    python main.py --step features  # chỉ chạy calendar + lag
    python main.py --step profile   # chỉ chạy profiler + sampler
    python main.py --step split     # chỉ chạy splitter
"""
import argparse
from src.data.cleaner        import run_batch_clean
from src.features.calendar_feats import run_calendar
from src.features.lag        import run_lag
from src.data.profiler       import run_profile
from src.data.sampler        import run_sample
from src.data.splitter       import run_splitter

def run_all():
    print("=" * 50)
    print("M5 ETL PIPELINE")
    print("=" * 50)

    print("\n[W3] Data Cleaning...")
    run_batch_clean()

    print("\n[W4] Calendar Features...")
    run_calendar()

    print("\n[W4] Lag Features...")
    run_lag()

    print("\n[W5] Profiling...")
    run_profile()

    print("\n[W5] Sampling...")
    run_sample()

    print("\n[W6] Walk-forward Splitter...")
    run_splitter()

    print("\n✓ Pipeline hoàn thành")

def main():
    parser = argparse.ArgumentParser(description="M5 ETL Pipeline")
    parser.add_argument(
        "--step",
        choices=["clean", "features", "profile", "split", "all"],
        default="all"
    )
    args = parser.parse_args()

    if args.step == "clean":
        run_batch_clean()
    elif args.step == "features":
        run_calendar()
        run_lag()
    elif args.step == "profile":
        run_profile()
        run_sample()
    elif args.step == "split":
        run_splitter()
    else:
        run_all()

if __name__ == "__main__":
    main()