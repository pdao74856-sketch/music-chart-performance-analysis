"""
step2_assign_quarter.py
-------------------------
Member 4 - Bước 2: Weekly -> Quarterly Organization (chạy trên data THẬT)

Mục đích:
  1. Pull toàn bộ chart_weekly (161,734 dòng) từ Supabase.
  2. Gán year + quarter (2021-Q1, 2021-Q2...) cho từng dòng.
  3. In tóm tắt để bạn kiểm tra bằng mắt trước khi sang bước sampling.
  4. Lưu kết quả ra file CSV (data/processed/chart_weekly_with_quarter.csv)
     để bước 3 (sampling) dùng lại, không cần pull lại từ Supabase mỗi lần.

Yêu cầu: đã cài sẵn supabase, python-dotenv, pandas (bước trước đã cài rồi)
Cách chạy: python step2_assign_quarter.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
except ImportError:
    print("Cần cài: pip install supabase --break-system-packages")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Kết nối + pull data (giống hệt script validate, tái sử dụng lại)
# ---------------------------------------------------------------------------
def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
    return create_client(url, key)


def fetch_table_paginated(client: Client, table: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows = []
    start = 0
    while True:
        resp = (
            client.table(table)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = resp.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 2. Hàm gán quarter (giống file assign_quarter.py đã viết trước đó)
# ---------------------------------------------------------------------------
def assign_quarter_to_chart_weekly(df: pd.DataFrame, date_col: str = "chart_date") -> pd.DataFrame:
    out = df.copy()
    parsed_dates = pd.to_datetime(out[date_col], errors="coerce")

    invalid_mask = parsed_dates.isna()
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        print(f"[CẢNH BÁO] Có {n_invalid} dòng chart_date không hợp lệ.")

    out["year"] = parsed_dates.dt.year
    quarter_num = (parsed_dates.dt.month - 1) // 3 + 1
    out["quarter"] = (
        out["year"].astype("Int64").astype(str) + "-Q" + quarter_num.astype("Int64").astype(str)
    )
    out.loc[invalid_mask, ["year", "quarter"]] = None
    return out


# ---------------------------------------------------------------------------
# 3. In tóm tắt để kiểm tra bằng mắt
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TÓM TẮT SAU KHI GÁN QUARTER")
    print("=" * 70)

    print(f"\nTổng số dòng: {len(df)}")
    print(f"Số dòng có quarter = None (lỗi ngày): {df['quarter'].isna().sum()}")

    print("\n--- Danh sách quarter đã xuất hiện (theo thời gian) ---")
    quarters = sorted(df["quarter"].dropna().unique())
    print(f"Từ {quarters[0]} đến {quarters[-1]}, tổng {len(quarters)} quarter")

    print("\n--- Số dòng theo market × quarter (5 dòng đầu để kiểm tra mẫu) ---")
    summary = df.groupby(["market", "quarter"]).size().reset_index(name="n_records")
    print(summary.head(5).to_string(index=False))

    print("\n--- Số BÀI HÁT DUY NHẤT theo market × quarter (5 dòng đầu) ---")
    # Đây là con số quan trọng cho bước sampling: mỗi market-quarter cần
    # ĐỦ 40 bài khác nhau (chưa tính tầng rank) mới sample được.
    unique_songs = (
        df.groupby(["market", "quarter"])["spotify_track_id"]
        .nunique()
        .reset_index(name="n_unique_songs")
    )
    print(unique_songs.head(5).to_string(index=False))

    n_insufficient = (unique_songs["n_unique_songs"] < 40).sum()
    print(f"\n[KIỂM TRA QUAN TRỌNG] Số market-quarter có ÍT HƠN 40 bài khác nhau: {n_insufficient}")
    if n_insufficient > 0:
        print("  -> Các market-quarter này sẽ KHÔNG đủ 40 bài để sample, cần lưu ý ở bước 3:")
        print(unique_songs[unique_songs["n_unique_songs"] < 40].to_string(index=False))
    else:
        print("  -> Tất cả market-quarter đều đủ ít nhất 40 bài khác nhau. Tốt.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = get_client()

    print("Đang tải bảng 'chart_weekly' từ Supabase (~161k dòng, có thể mất chút thời gian)...")
    chart = fetch_table_paginated(client, "chart_weekly")
    print(f"  -> Đã tải {len(chart)} dòng\n")

    print("Đang gán quarter...")
    chart_with_quarter = assign_quarter_to_chart_weekly(chart)
    print("  -> Xong.\n")

    print_summary(chart_with_quarter)

    # Lưu ra CSV để bước 3 (sampling) dùng lại, đỡ phải pull lại Supabase
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chart_weekly_with_quarter.csv"
    chart_with_quarter.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Đã lưu kết quả vào: {output_path}")
    print("    (dùng file này cho bước 3 - stratified sampling)")


if __name__ == "__main__":
    main()