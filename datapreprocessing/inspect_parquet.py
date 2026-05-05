"""
inspect_parquet.py
------------------
Quick inspection of MEDS-format parquet files.

Usage:
    python inspect_parquet.py                          # default: labs.parquet
    python inspect_parquet.py --file /path/to/file.parquet
    python inspect_parquet.py --patient C800881        # filter single patient
"""

import argparse
import pandas as pd

DEFAULT_PATH = "/path/to/CHD_MEDS/data/labs.parquet"

def inspect(file_path: str, patient_id: str = None, n: int = 20):
    print(f"\nLoading: {file_path}")
    df = pd.read_parquet(file_path)

    print(f"\n{'='*55}")
    print(f"  Shape            : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Unique patients  : {df['patient_id'].nunique():,}")
    print(f"  Unique codes     : {df['code'].nunique():,}")
    print(f"  Time range       : {df['time'].min()}  ->  {df['time'].max()}")
    pct_num = df['numeric_value'].notna().mean() * 100
    print(f"  Numeric results  : {pct_num:.1f}%")
    print(f"{'='*55}")

    print("\n--- dtypes ---")
    print(df.dtypes)

    print(f"\n--- Top 20 most common codes ---")
    print(df['code'].value_counts().head(20).to_string())

    print(f"\n--- numeric_value stats ---")
    print(df['numeric_value'].describe())

    if patient_id:
        sub = df[df['patient_id'] == patient_id]
        print(f"\n--- Patient {patient_id} ({len(sub)} rows) ---")
        print(sub.to_string(index=False))
    else:
        print(f"\n--- First {n} rows ---")
        print(df.head(n).to_string(index=False))

        print(f"\n--- Sample text_value rows ---")
        text_rows = df[df['text_value'].notna()].head(10)
        print(text_rows[['patient_id','time','code','numeric_value','text_value']].to_string(index=False))

        print(f"\n--- Sample //LT rows ---")
        lt_rows = df[df['code'].str.endswith('//LT')].head(10)
        print(lt_rows[['patient_id','time','code','numeric_value','text_value']].to_string(index=False))

        print(f"\n--- Sample //GT rows ---")
        gt_rows = df[df['code'].str.endswith('//GT')].head(10)
        print(gt_rows[['patient_id','time','code','numeric_value','text_value']].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    default=DEFAULT_PATH, help="Path to parquet file")
    parser.add_argument("--patient", default=None,         help="Filter to single patient_id (e.g. C800881)")
    parser.add_argument("--n",       default=20, type=int, help="Number of rows to preview")
    args = parser.parse_args()

    inspect(args.file, args.patient, args.n)
